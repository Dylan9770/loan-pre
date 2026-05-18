"""Customer profile routes."""

from __future__ import annotations

import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify

from service.flask.model_loader import predict_default, predict_fraud, predict_limit
from service.flask.repositories.mysql_repo import (
    compute_peer_percentile,
    fetch_customer_credit_signals,
    fetch_customer_loan_facts,
    fetch_customer_profile,
    fetch_customer_similar,
    fetch_random_customer_id,
    insert_realtime_decision,
)

customer_bp = Blueprint("customer_bp", __name__, url_prefix="/customer")


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0 or math.isnan(b) or math.isnan(a):
        return default
    return a / b


def _compute_radar_scores(profile: dict) -> dict:
    """Compute 5-dimension radar chart scores for a customer."""
    credit_score = float(profile.get("credit_score") or 0)

    # Dimension 1: 信用评分 (0-850 -> 0-100 scaled)
    credit_dim = min(credit_score / 8.5, 100)

    # Dimension 2: 还款能力 (基于债务收入比)
    disbursed = float(profile.get("disbursed_amount") or 0)
    outstanding = float(profile.get("total_outstanding_loan") or 0)
    asset_cost = float(profile.get("asset_cost") or 1)
    ratio = _safe_div(outstanding, disbursed if disbursed else asset_cost)
    repay_ability = max(0, min(100, 100 - ratio * 80))

    # Dimension 3: 资产状况 (基于贷款资产比)
    ltv = _safe_div(disbursed, asset_cost)
    asset_status = max(0, min(100, 100 - (ltv - 0.5) * 100))

    # Dimension 4: 历史记录 (基于逾期次数和逾期率)
    overdue_no = float(profile.get("total_overdue_no") or 0)
    account_no = float(profile.get("total_account_loan_no") or 1)
    overdue_rate = _safe_div(overdue_no, account_no)
    history_score = max(0, min(100, 100 - overdue_no * 15 - overdue_rate * 40))

    # Dimension 5: 稳定性 (基于年龄和工作类型)
    age = float(profile.get("age") or 35)
    emp_type = int(profile.get("employment_type") or 0)
    stability = max(0, min(100, 100 - abs(age - 40) * 1.5 - (emp_type == 2) * 10))

    return {
        "credit": round(credit_dim, 2),
        "repay_ability": round(repay_ability, 2),
        "asset_status": round(asset_status, 2),
        "history": round(history_score, 2),
        "stability": round(stability, 2),
    }


def _build_mock_profile(customer_id: int) -> dict:
    """Generate deterministic mock profile for demo purposes."""
    rng_seed = customer_id % 1000
    np.random.seed(rng_seed)

    base_score = 500 + rng_seed
    overdue_count = rng_seed % 5

    profile = {
        "customer_id": customer_id,
        "age": 25 + (rng_seed % 40),
        "employment_type": rng_seed % 3,
        "area_id": rng_seed % 10,
        "credit_score": base_score,
        "disbursed_amount": 10000 + (rng_seed % 80000),
        "total_outstanding_loan": 5000 + (rng_seed % 30000),
        "asset_cost": 15000 + (rng_seed % 100000),
        "total_overdue_no": overdue_count,
        "total_account_loan_no": 1 + (rng_seed % 6),
        "main_account_overdue_no": overdue_count,
        "main_account_loan_no": 1 + (rng_seed % 4),
        "total_monthly_payment": 500 + (rng_seed % 3000),
        "total_disbursed_loan": 10000 + (rng_seed % 80000),
        "last_six_month_new_loan_no": rng_seed % 3,
        "last_six_month_defaulted_no": overdue_count % 2,
        "credit_history": 1 + (rng_seed % 10),
        "enquirie_no": rng_seed % 8,
        "loan_default": 1 if rng_seed % 7 == 0 else 0,
    }
    return profile


def _build_credit_profile(signals: dict | None, facts: dict | None) -> dict:
    """Assemble the customer credit profile shown in place of the loan timeline.

    Returns three sections:
      - recent_activity: 4 raw counters from customer_features (近6月新增/违约, 征信查询, 信用历史)
      - finance_health:  3 ratios (额度使用率, 还款进度, 杠杆率) clamped to 0~1 for ring display
      - peer_percentile: 4 indicators with percentile ranks vs. the whole customer base

    Either input may be None (eg. customer missing from one table); fields fall back to 0/None.
    """
    s = signals or {}
    f = facts or {}

    # Numbers fall through several sources because the two tables overlap partially.
    def _num(*keys, default=0):
        for k in keys:
            for src in (s, f):
                v = src.get(k)
                if v is not None:
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        continue
        return default

    new_6m       = int(_num("last_six_month_new_loan_no"))
    default_6m   = int(_num("last_six_month_defaulted_no"))
    enquiries    = int(_num("enquirie_no"))
    # credit_history 源 CSV 单位为"月"，直接用月展示（按年会大量出现 0 年）
    credit_hist_months = int(_num("credit_history"))

    asset_cost   = _num("asset_cost")
    monthly_pay_gauge = _num("total_monthly_payment")
    ltv          = _num("loan_to_asset_ratio", "ltv_ratio")


    # Peer percentile: positive metrics use the raw percentile;
    # negative metrics ("lower is better") flip to 100 - pctl so the bar always
    # reads as "better than X% of customers".
    def _pctl(metric: str, value, lower_is_better: bool = False):
        raw = compute_peer_percentile(value, metric)
        if raw is None:
            return None
        return round(100.0 - raw, 1) if lower_is_better else raw

    credit_score = _num("credit_score")
    overdue_no   = _num("total_overdue_no")
    monthly_pay  = _num("total_monthly_payment")

    peer = [
        {"label": "信用评分",  "value": round(credit_score, 0),
         "percentile": _pctl("credit_score", credit_score)},
        {"label": "逾期次数",  "value": int(overdue_no),
         "percentile": _pctl("total_overdue_no", overdue_no, lower_is_better=True)},
        {"label": "月供负担",  "value": round(monthly_pay, 0),
         "percentile": _pctl("total_monthly_payment", monthly_pay, lower_is_better=True)},
        {"label": "杠杆率(LTV)", "value": round(ltv, 3),
         "percentile": _pctl("loan_to_asset_ratio", ltv, lower_is_better=True)},
    ]

    return {
        "recent_activity": [
            {"label": "近6月新增贷款", "value": new_6m,      "unit": "笔",
             "tone": "danger" if new_6m >= 5 else ("warn" if new_6m >= 2 else "ok")},
            {"label": "近6月违约",     "value": default_6m,  "unit": "次",
             "tone": "danger" if default_6m >= 1 else "ok"},
            {"label": "征信查询次数", "value": enquiries,   "unit": "次",
             "tone": "danger" if enquiries >= 8 else ("warn" if enquiries >= 4 else "ok")},
            {"label": "信用历史",     "value": credit_hist_months, "unit": "月",
             "tone": "ok" if credit_hist_months >= 36 else ("warn" if credit_hist_months >= 12 else "danger")},
        ],
        "finance_health": {
            "asset_cost": {
                "label": "资产成本（车价）", "value": round(asset_cost, 0),
                "unit": "元", "max": 200000,
                "bands": {"ok": [0, 80000], "warn": [80000, 150000], "danger": [150000, 200000]},
            },
            "monthly_payment": {
                "label": "月供负担", "value": round(monthly_pay_gauge, 0),
                "unit": "元", "max": 50000,
                "bands": {"ok": [0, 15000], "warn": [15000, 30000], "danger": [30000, 50000]},
            },
            "ltv": {
                "label": "杠杆率(LTV)", "value": round(ltv, 3),
                "unit": "", "max": 1.2,
                "bands": {"ok": [0.5, 0.75], "warn_low": [0.3, 0.5], "warn_high": [0.75, 1.0],
                          "danger_low": [0, 0.3], "danger_high": [1.0, 1.2]},
            },
        },
        "peer_percentile": peer,
    }



@customer_bp.get("/random_id")
def get_random_customer_id():
    """Return a real customer_id from the DB so the dashboard random button never hits a missing record."""
    cid = fetch_random_customer_id()
    if cid is None:
        return jsonify({"error": "no customer available"}), 503
    return jsonify({"customer_id": cid})


@customer_bp.get("/<int:customer_id>/profile")
def get_customer_profile(customer_id: int):
    """Return complete customer profile with radar scores. 404 if the customer is not in the DB."""
    db_profile = fetch_customer_profile(customer_id)

    if not db_profile:
        return jsonify({
            "error": "customer not found",
            "customer_id": customer_id,
            "message": f"客户 {customer_id} 不存在于数据库中",
        }), 404

    profile = db_profile

    # Compute radar chart scores
    radar = _compute_radar_scores(profile)

    # Get prediction results
    prediction_real = False
    try:
        pred_default = predict_default([profile])           # 现在直接返回 credit_score
        pred_fraud   = predict_fraud([profile])
        # predict_limit 依赖 credit_score 和 fraud_prob，复用前两步结果
        credit_score_val = pred_default[0]["credit_score"]  if pred_default else 600.0
        fraud_prob       = pred_fraud[0]["fraud_probability"] if pred_fraud   else 0.0
        pred_limit       = predict_limit([profile],
                                          credit_scores=[credit_score_val],
                                          fraud_probs=[fraud_prob])
        limit_val = pred_limit[0]["predicted_limit"] if pred_limit else 0.0
        # 由 credit_score 反推 default_probability（评分卡公式逆运算）
        default_prob = 1.0 / (1.0 + 2 ** ((credit_score_val - 600) / 50))
        prediction_real = True
    except Exception:
        # Fallback mock predictions
        base_prob = (1000 - (customer_id % 1000)) / 1000.0
        default_prob = max(0.01, min(0.99, base_prob))
        fraud_prob = max(0.01, min(0.5, (customer_id % 50) / 100))
        limit_val = 10000 + (customer_id % 80000)
        credit_score_val = 600 - 50 * math.log(default_prob / (1 - default_prob))

    # 把真实预测结果落库到 realtime_decisions，看板"最新决策"会读这张表
    if prediction_real:
        try:
            insert_realtime_decision({
                "customer_id": customer_id,
                "default_probability": float(default_prob),
                "default_pred": 1 if default_prob >= 0.5 else 0,
                "fraud_probability": float(fraud_prob),
                "fraud_pred": 1 if fraud_prob >= 0.5 else 0,
                "predicted_limit": float(limit_val),
                "credit_score": float(credit_score_val),
            })
        except Exception:
            pass  # 落库失败不影响接口返回

    result = {
        "customer_id": customer_id,
        "profile": {
            "age": profile.get("age"),
            "employment_type": profile.get("employment_type"),
            "area_id": profile.get("area_id"),
            "credit_score": profile.get("credit_score"),
            "disbursed_amount": profile.get("disbursed_amount"),
            "total_overdue_no": profile.get("total_overdue_no"),
            "total_account_loan_no": profile.get("total_account_loan_no"),
            "loan_default": profile.get("loan_default"),
        },
        "radar_scores": radar,
        "decision": {
            "default_probability": round(default_prob, 4),
            "default_pred": 1 if default_prob >= 0.5 else 0,
            "fraud_probability": round(fraud_prob, 4),
            "fraud_pred": 1 if fraud_prob >= 0.5 else 0,
            "predicted_limit": round(limit_val, 2),
            "credit_score": round(credit_score_val, 1),
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    return jsonify(result)


@customer_bp.get("/<int:customer_id>/similar")
def get_similar_customers(customer_id: int):
    """Return Top-K similar customers based on feature cosine similarity."""
    db_similar = fetch_customer_similar(customer_id, k=5)

    if db_similar:
        return jsonify(db_similar)

    # Generate mock similar customers for demo
    rng_seed = customer_id % 1000
    np.random.seed(rng_seed)

    # Create mock target profile
    target_score = 500 + rng_seed
    target_overdue = rng_seed % 5

    similar = []
    for i in range(5):
        offset = (i + 1) * 5
        sim_id = customer_id + offset * 100
        sim_score = max(300, min(850, target_score + np.random.randint(-30, 30)))
        sim_overdue = max(0, min(5, target_overdue + np.random.randint(-1, 1)))

        default_actual = 1 if sim_overdue >= 3 else 0
        performance = "正常还款" if default_actual == 0 else "部分违约"
        similarity = max(0.70, 0.99 - i * 0.04 - abs(sim_score - target_score) / 1000)

        similar.append({
            "customer_id": int(sim_id),
            "credit_score": int(sim_score),
            "disbursed_amount": float(10000 + (sim_id % 80000)),
            "total_overdue_no": int(sim_overdue),
            "actual_default": int(default_actual),
            "actual_performance": performance,
            "similarity": round(similarity, 4),
        })

    return jsonify(similar)


@customer_bp.get("/<int:customer_id>/credit_profile")
def get_customer_credit_profile(customer_id: int):
    """Customer credit profile shown in place of the legacy loan timeline.

    Returns three sections (see `_build_credit_profile`):
      - recent_activity (4 数字卡)
      - finance_health  (3 圆环)
      - peer_percentile (4 同业百分位)

    数据全部来自 customer_features + loan_fact，无 mock 构造。
    """
    signals = fetch_customer_credit_signals(customer_id)
    facts   = fetch_customer_loan_facts(customer_id)
    if not signals and not facts:
        return jsonify({
            "error": "customer not found",
            "customer_id": customer_id,
        }), 404
    profile = _build_credit_profile(signals, facts)
    return jsonify({
        "customer_id": customer_id,
        **profile,
    })
