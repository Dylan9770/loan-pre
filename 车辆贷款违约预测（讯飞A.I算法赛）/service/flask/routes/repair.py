"""数据修复 HTTP 端点。

GET  /repair/evaluation        ←  读 artifacts/repair_evaluation.json
POST /repair/record            ←  接收一条记录 JSON，对 Credit_level 与 5 个数值字段做修复

修复算法来自 src/repair.py：
- Credit_level：FP-growth 关联规则（基于 employment_type / area_id / age / credit_history）
- 数值字段（credit_score / disbursed_amount / asset_cost /
        total_outstanding_loan / total_monthly_payment）：自适应均值/中位数填补
"""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request

from src.config import get_config
from collections import defaultdict

from service.flask.repositories.mysql_repo import fetch_customer_profile
from src.repair import (
    apply_rule_repair,
    build_association_rules,
)

repair_bp = Blueprint("repair", __name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Credit_level 修复用的上下文列与目标列
_CAT_TARGET = "Credit_level"
_CAT_CONTEXT = ["employment_type", "area_id", "age", "credit_history"]

# 数值修复字段
_NUM_COLS = [
    "credit_score", "disbursed_amount", "asset_cost",
    "total_outstanding_loan", "total_monthly_payment",
]

# ---- 关联规则缓存（首次请求时加载，避免每次重算 49 条规则） ----
_RULE_CACHE: dict = {"rules": None, "fill_stats": None, "bin_edges": None}
_CACHE_LOCK = Lock()


def _compute_bin_edges(series: pd.Series, bins: int = 4) -> list[float] | None:
    """与 src.repair._bin_numeric 一致的 qcut/cut 切点；用于单条记录复用训练集 binning。"""
    s = series.dropna()
    s = s[s != -1]   # 与 _bin_numeric 一样把哨兵值排除
    if s.empty or s.nunique() < 2:
        return None
    try:
        _, edges = pd.qcut(s, q=bins, retbins=True, duplicates="drop")
    except Exception:
        try:
            _, edges = pd.cut(s, bins=bins, retbins=True)
        except Exception:
            return None
    return [float(e) for e in edges]


def _bin_value(v, edges: list[float] | None) -> str:
    """用训练集切点把单个数值落进区间，返回与 pd.cut Interval 相同的字符串表示
    （这样才能跟 build_association_rules 里 _tokenize_df 生成的 token 对得上）。"""
    if edges is None or v is None or pd.isna(v):
        return str(v)
    if v == -1:
        return "MISSING"
    try:
        # include_lowest=True：让 v == edges[0] 也能落进最低 bin
        binned = pd.cut([float(v)], bins=edges, include_lowest=True)
        cell = binned[0]
        if pd.notna(cell):
            return str(cell)
    except Exception:
        pass
    # 越界值向最近 bin 兜底
    if v <= edges[0]:
        binned = pd.cut([edges[0]], bins=edges, include_lowest=True)
    else:
        binned = pd.cut([edges[-1]], bins=edges, include_lowest=True)
    cell = binned[0]
    return str(cell) if pd.notna(cell) else str(v)


def _ensure_rules_loaded() -> tuple[list, dict, dict]:
    """懒加载关联规则、数值列填补统计量、上下文列 binning 切点。线程安全。"""
    if _RULE_CACHE["rules"] is not None:
        return _RULE_CACHE["rules"], _RULE_CACHE["fill_stats"], _RULE_CACHE["bin_edges"]
    with _CACHE_LOCK:
        if _RULE_CACHE["rules"] is not None:
            return _RULE_CACHE["rules"], _RULE_CACHE["fill_stats"], _RULE_CACHE["bin_edges"]

        cfg = get_config(_PROJECT_ROOT)
        cleaned = cfg.cleaned_dir / "train_cleaned.csv"
        if not cleaned.exists():
            raise FileNotFoundError(
                f"train_cleaned.csv not found at {cleaned}. Run run_ingest_storage.py first."
            )
        df = pd.read_csv(cleaned)

        rules = build_association_rules(
            df, _CAT_TARGET, _CAT_CONTEXT,
            min_support=0.005, min_confidence=0.15,
        )
        # 数值列：预先算 mean / median 用于单条修复（避免推理时再扫全表）
        fill_stats: dict = {}
        for c in _NUM_COLS:
            if c in df.columns:
                col = df[c].dropna()
                if not col.empty:
                    fill_stats[c] = {"mean": float(col.mean()), "median": float(col.median())}
        # 上下文列：保存训练集 binning 切点，单条修复时复用
        bin_edges: dict = {}
        for c in _CAT_CONTEXT:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                bin_edges[c] = _compute_bin_edges(df[c], bins=4)
        _RULE_CACHE["rules"] = rules
        _RULE_CACHE["fill_stats"] = fill_stats
        _RULE_CACHE["bin_edges"] = bin_edges
        return rules, fill_stats, bin_edges


@repair_bp.get("/repair/evaluation")
def repair_evaluation():
    """Return repair pipeline evaluation report."""
    eval_path = _PROJECT_ROOT / "artifacts" / "repair_evaluation.json"
    if not eval_path.exists():
        return jsonify({"error": "repair_evaluation.json not found; run run_repair_pipeline.py first"}), 503
    try:
        return jsonify(json.loads(eval_path.read_text(encoding="utf-8")))
    except Exception as exc:
        return jsonify({"error": f"failed to parse: {exc}"}), 500


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or v == -1 or v == ""


def _match_credit_level(record: dict, rules: list, bin_edges: dict) -> tuple[object, float | None]:
    """单条记录走规则匹配。返回 (修复值, 置信度)；无匹配返回 (None, None)。"""
    # 把上下文字段离散化成与训练集相同的 token
    ctx_tokens = []
    for c in _CAT_CONTEXT:
        v = record.get(c)
        if _is_missing(v):
            continue
        if c in bin_edges and bin_edges[c] is not None:
            token = _bin_value(v, bin_edges[c])
        else:
            token = str(v)
        ctx_tokens.append(f"{c}={token}")
    ctx_tokens = sorted(ctx_tokens)
    if not ctx_tokens:
        return None, None

    # 索引规则：按 antecedent
    rule_map = defaultdict(list)
    for r in rules:
        rule_map[r.antecedent].append(r)

    best = None
    n = len(ctx_tokens)
    for i in range(n):
        for r in rule_map.get((ctx_tokens[i],), []):
            if best is None or r.confidence > best.confidence:
                best = r
        for j in range(i + 1, n):
            for r in rule_map.get((ctx_tokens[i], ctx_tokens[j]), []):
                if best is None or r.confidence > best.confidence:
                    best = r
    if best is None:
        return None, None
    val = best.consequent.split("=", 1)[1]
    return _try_numeric(val), float(best.confidence)


def _do_repair(record: dict, rules: list, fill_stats: dict, bin_edges: dict) -> list[dict]:
    """对一条 record 应用修复，返回 repaired_fields list（共享给 /record 与 /by_customer）。"""
    repaired_fields: list[dict] = []

    # ---- 分类字段：Credit_level ----
    if _is_missing(record.get(_CAT_TARGET)):
        new_val, conf = _match_credit_level(record, rules, bin_edges)
        repaired_fields.append({
            "field":      _CAT_TARGET,
            "type":       "categorical",
            "method":     "FP-Growth 关联规则",
            "before":     None,
            "after":      new_val,
            "confidence": conf,
            "note":       None if new_val is not None else "无匹配规则，建议补充上下文字段",
        })

    # ---- 数值字段 ----
    for col in _NUM_COLS:
        if not _is_missing(record.get(col)):
            continue
        stats = fill_stats.get(col)
        if not stats:
            continue
        repaired_fields.append({
            "field":      col,
            "type":       "numeric",
            "method":     "自适应中位数填补 (ALS-style)",
            "before":     None,
            "after":      round(float(stats["median"]), 2),
            "confidence": None,
        })
    return repaired_fields


@repair_bp.post("/repair/record")
def repair_record():
    """对单条记录做修复。请求体: {"record": {...}}；返回 before/after 对比。"""
    body = request.get_json(silent=True) or {}
    record = body.get("record") or body
    if not isinstance(record, dict) or not record:
        return jsonify({"error": "请求体需要包含一个非空 record dict"}), 400
    try:
        rules, fill_stats, bin_edges = _ensure_rules_loaded()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"failed to load rules: {exc}"}), 500

    return jsonify({
        "input_record":    record,
        "repaired_fields": _do_repair(record, rules, fill_stats, bin_edges),
        "rules_count":     len(rules),
    })


# 修复展示给前端的字段白名单（按业务相关性排序）。
# 与 _NUM_COLS / _CAT_CONTEXT / _CAT_TARGET 对齐，确保前端能显示原值 + 修复值的差异。
_DISPLAY_FIELDS = (
    _CAT_CONTEXT + [_CAT_TARGET] + _NUM_COLS
)


@repair_bp.get("/repair/by_customer/<int:customer_id>")
def repair_by_customer(customer_id: int):
    """按客户 ID 拉数据库记录、自动识别缺失字段、跑修复，返回原值 + 修复值。"""
    profile = fetch_customer_profile(customer_id)
    if not profile:
        return jsonify({"error": f"未找到客户 {customer_id}"}), 404

    try:
        rules, fill_stats, bin_edges = _ensure_rules_loaded()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"failed to load rules: {exc}"}), 500

    # 只取修复算法关心的字段做 record 输入（其它字段不影响）
    record = {k: profile.get(k) for k in _DISPLAY_FIELDS if k in profile}

    repaired = _do_repair(record, rules, fill_stats, bin_edges)
    repaired_field_names = {r["field"] for r in repaired}

    # 把展示字段拆成两组：原本就有值的字段 + 缺失被修复的字段（已在 repaired 里）
    known_fields = [
        {"field": k, "value": _normalize_for_json(record.get(k))}
        for k in _DISPLAY_FIELDS
        if k in record and not _is_missing(record.get(k))
    ]
    missing_fields = [
        k for k in _DISPLAY_FIELDS
        if k in record and _is_missing(record.get(k))
    ]

    return jsonify({
        "customer_id":     customer_id,
        "known_fields":    known_fields,
        "missing_fields":  missing_fields,
        "repaired_fields": repaired,
        "rules_count":     len(rules),
    })


def _normalize_for_json(v):
    """把数据库返回的 Decimal / numpy 类型转成 JSON 友好类型。"""
    if v is None:
        return None
    try:
        if hasattr(v, "item"):
            v = v.item()  # numpy scalar
    except Exception:
        pass
    if isinstance(v, (int, float, str, bool)):
        return v
    try:
        return float(v)
    except Exception:
        return str(v)


def _try_numeric(v):
    """把字符串型数字转回 int/float，便于前端展示。"""
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (ValueError, TypeError):
        return v
