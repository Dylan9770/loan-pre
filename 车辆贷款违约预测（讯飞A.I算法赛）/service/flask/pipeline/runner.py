"""数据导入 Pipeline 执行器。

8 步流程：
  1. MySQL 业务库落库         → loan_ods.import_staging
  2. Flume 文件采集            → /data/flume/spool/<job>.csv
  3. HDFS Raw 入湖             → data_lake/raw/dt=YYYYMMDD/<job>.csv
  4. 数据清洗 (DWD)            → data_lake/cleaned/<job>_cleaned.csv
  5. 数据修复                  → 同上覆盖写
  6. 特征工程                  → data_lake/featured/<job>_featured.csv
  7. 模型推理                  → 内存中
  8. 结果落库                  → loan_rt.realtime_decisions

每步开始/结束都通过 job_store 写回 MySQL，前端可轮询。
"""
from __future__ import annotations

import json
import math
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pymysql

from features_v3 import add_features
from service.flask.config import Settings
from service.flask.model_loader import predict_default, predict_fraud, predict_limit
from service.flask.pipeline import job_store
from service.flask.repositories.mysql_repo import insert_realtime_decision

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# 演示路径布局：直接落到 data_lake 各层，与 Hive External Table 指向一致
FLUME_SPOOL_DIR    = _PROJECT_ROOT / "data" / "flume" / "spool"
HDFS_RAW_DIR       = _PROJECT_ROOT / "data_lake" / "raw"
HDFS_CLEANED_DIR   = _PROJECT_ROOT / "data_lake" / "cleaned"
HDFS_FEATURED_DIR  = _PROJECT_ROOT / "data_lake" / "featured"

# 入湖最少要保留这些列，缺失字段会被填为 NaN，由后续修复步骤处理
EXPECTED_COLS = [
    "customer_id", "age", "employment_type", "area_id", "credit_score",
    "credit_history", "disbursed_amount", "asset_cost", "ltv_ratio",
    "total_overdue_no", "total_outstanding_loan", "total_monthly_payment",
    "total_account_loan_no", "total_disbursed_loan",
    "main_account_loan_no", "main_account_active_loan_no",
    "main_account_overdue_no", "main_account_monthly_payment",
    "sub_account_loan_no",
    "last_six_month_new_loan_no", "last_six_month_defaulted_no",
    "enquirie_no", "Credit_level",
]

# features_v3.add_features 会引用很多原始列，缺失时补 0 即可（业务上 0 表示"无该项行为/账户"）
FEATURE_INPUT_COLS = [
    "year_of_birth", "disbursed_date",
    "sub_account_active_loan_no", "sub_account_overdue_no",
    "sub_account_outstanding_loan", "sub_account_sanction_loan",
    "sub_account_disbursed_loan", "sub_account_monthly_payment",
    "sub_account_inactive_loan_no", "sub_account_tenure",
    "main_account_outstanding_loan", "main_account_sanction_loan",
    "main_account_disbursed_loan", "main_account_inactive_loan_no",
    "main_account_tenure", "main_account_monthly_payment",
    "total_inactive_loan_no", "total_sanction_loan",
    "average_age", "loan_to_asset_ratio", "outstanding_disburse_ratio",
    "disburse_to_sactioned_ratio", "active_to_inactive_act_ratio",
    "branch_id", "supplier_id", "manufacturer_id", "employee_code_id",
    "mobileno_flag", "idcard_flag", "Driving_flag", "passport_flag",
]

# 演示用：每步之间 sleep 一下，老师看进度更清楚
DEMO_SLEEP_SEC = 0.6


# -------- 用 src.repair 里训练好的规则做缺失修复 --------
def _load_repair_artifacts() -> tuple[list, dict, dict]:
    from service.flask.routes.repair import _ensure_rules_loaded
    return _ensure_rules_loaded()


def _is_missing(v) -> bool:
    if v is None or v == "" or v == -1:
        return True
    try:
        return bool(pd.isna(v))
    except Exception:
        return False


def _bin_value(v, edges):
    """与 repair.py 中 _bin_value 一致，把数值落进训练集 binning 切点。"""
    if edges is None or v is None:
        return str(v)
    try:
        if float(v) == -1:
            return "MISSING"
    except Exception:
        pass
    try:
        binned = pd.cut([float(v)], bins=edges, include_lowest=True)
        cell = binned[0]
        return str(cell) if pd.notna(cell) else str(v)
    except Exception:
        return str(v)


# =============================================================================
#  Step 1: MySQL 业务库落库
# =============================================================================
def _step1_mysql_staging(job_id: str, df: pd.DataFrame) -> int:
    job_store.start_step(job_id, 1, "解析 Excel/CSV 并写入 loan_ods.import_staging")

    # 标准化列名（去空格、保留我们关心的列）
    df.columns = [str(c).strip() for c in df.columns]
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = np.nan

    conn = pymysql.connect(
        host=Settings.MYSQL_HOST, port=Settings.MYSQL_PORT,
        user=Settings.MYSQL_USER, password=Settings.MYSQL_PASSWORD,
        database=Settings.MYSQL_DB_ODS, charset="utf8mb4", autocommit=True,
    )
    inserted = 0
    try:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                vals = []
                for c in EXPECTED_COLS:
                    v = row.get(c)
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        vals.append(None)
                    else:
                        vals.append(v)
                raw_json = json.dumps(
                    {c: (None if (v is None or (isinstance(v, float) and math.isnan(v))) else v)
                     for c, v in zip(EXPECTED_COLS, vals)},
                    ensure_ascii=False, default=str,
                )
                cur.execute(
                    """INSERT INTO import_staging
                       (job_id, customer_id, age, employment_type, area_id, credit_score,
                        credit_history, disbursed_amount, asset_cost, ltv_ratio,
                        total_overdue_no, total_outstanding_loan, total_monthly_payment,
                        total_account_loan_no, total_disbursed_loan,
                        main_account_loan_no, main_account_active_loan_no,
                        main_account_overdue_no, main_account_monthly_payment,
                        sub_account_loan_no,
                        last_six_month_new_loan_no, last_six_month_defaulted_no,
                        enquirie_no, Credit_level, raw_json)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                               %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                               %s, %s, %s, %s, %s)""",
                    (job_id, *vals, raw_json),
                )
                inserted += 1
    finally:
        conn.close()

    job_store.finish_step(
        job_id, 1,
        message=f"已写入 loan_ods.import_staging（{inserted} 行）",
        extra={"rows": inserted, "table": "loan_ods.import_staging"},
    )
    time.sleep(DEMO_SLEEP_SEC)
    return inserted


# =============================================================================
#  Step 2: Flume 文件采集
# =============================================================================
def _step2_flume_spool(job_id: str, df: pd.DataFrame) -> Path:
    job_store.start_step(job_id, 2, "写入 Flume SpoolDir，模拟 SpoolDir Source 采集")
    FLUME_SPOOL_DIR.mkdir(parents=True, exist_ok=True)
    spool_file = FLUME_SPOOL_DIR / f"{job_id}.csv"
    df[EXPECTED_COLS].to_csv(spool_file, index=False)

    # 模拟 Flume 在采集（实际生产中是 Flume Agent 监控目录并搬运）
    time.sleep(DEMO_SLEEP_SEC)
    completed_marker = spool_file.with_suffix(".csv.COMPLETED")
    shutil.copy2(spool_file, completed_marker)

    size_kb = round(spool_file.stat().st_size / 1024, 2)
    job_store.finish_step(
        job_id, 2,
        message=f"Flume 采集完成（{size_kb} KB）",
        extra={
            "spool_path": str(spool_file),
            "size_kb": size_kb,
            "source_type": "SpoolDir",
            "sink_type": "HDFS",
        },
    )
    return spool_file


# =============================================================================
#  Step 3: HDFS Raw 入湖
# =============================================================================
def _step3_hdfs_raw(job_id: str, spool_file: Path) -> Path:
    today = datetime.now().strftime("%Y%m%d")
    job_store.start_step(job_id, 3, f"Flume HDFS Sink → /data_lake/raw/dt={today}/")

    partition_dir = HDFS_RAW_DIR / f"dt={today}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    raw_path = partition_dir / f"loan_raw_{job_id}.csv"
    shutil.copy2(spool_file, raw_path)

    time.sleep(DEMO_SLEEP_SEC)
    job_store.finish_step(
        job_id, 3,
        message=f"原始数据已入 HDFS Raw 层（partition dt={today}）",
        extra={
            "hdfs_path":  str(raw_path),
            "partition":  f"dt={today}",
            "hive_table": "loan_ods.raw_loan_data",
        },
    )
    return raw_path


# =============================================================================
#  Step 4: 数据清洗 (DWD)
# =============================================================================
def _step4_clean(job_id: str, raw_path: Path) -> tuple[Path, pd.DataFrame, dict]:
    job_store.start_step(job_id, 4, "清洗：去重、去 inf、数据类型规范")
    df = pd.read_csv(raw_path)
    rows_before = len(df)

    # 去重 customer_id
    df = df.drop_duplicates(subset=["customer_id"], keep="last").reset_index(drop=True)
    rows_after_dedup = len(df)
    # 去 inf
    df = df.replace([np.inf, -np.inf], np.nan)
    # customer_id 必须有
    df = df.dropna(subset=["customer_id"]).reset_index(drop=True)
    rows_after_clean = len(df)

    HDFS_CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_path = HDFS_CLEANED_DIR / f"{job_id}_cleaned.csv"
    df.to_csv(cleaned_path, index=False)

    stats = {
        "rows_in":         rows_before,
        "rows_dedup":      rows_after_dedup,
        "rows_out":        rows_after_clean,
        "duplicates_removed": rows_before - rows_after_dedup,
        "invalid_removed":    rows_after_dedup - rows_after_clean,
    }
    time.sleep(DEMO_SLEEP_SEC)
    job_store.finish_step(
        job_id, 4,
        message=f"清洗后保留 {rows_after_clean} 行（去重 {stats['duplicates_removed']}，无效 {stats['invalid_removed']}）",
        extra={"cleaned_path": str(cleaned_path), **stats,
               "hive_table": "loan_dwd.loan_cleaned"},
    )
    return cleaned_path, df, stats


# =============================================================================
#  Step 5: 数据修复（FP-Growth + 中位数填补）
# =============================================================================
def _step5_repair(job_id: str, cleaned_path: Path, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    job_store.start_step(job_id, 5, "FP-Growth 修复 Credit_level + 中位数填补数值字段")

    rules, fill_stats, bin_edges = _load_repair_artifacts()
    repaired_categorical = 0
    repaired_numeric = 0

    # ----- Credit_level FP-Growth -----
    from collections import defaultdict
    rule_map = defaultdict(list)
    for r in rules:
        rule_map[r.antecedent].append(r)

    CTX = ["employment_type", "area_id", "age", "credit_history"]
    for idx, row in df.iterrows():
        if not _is_missing(row.get("Credit_level")):
            continue
        ctx_tokens = []
        for c in CTX:
            v = row.get(c)
            if _is_missing(v):
                continue
            edges = bin_edges.get(c)
            token = _bin_value(v, edges) if edges else str(v)
            ctx_tokens.append(f"{c}={token}")
        ctx_tokens = sorted(ctx_tokens)
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
        if best is not None:
            val = best.consequent.split("=", 1)[1]
            try:
                df.at[idx, "Credit_level"] = int(float(val))
            except Exception:
                df.at[idx, "Credit_level"] = val
            repaired_categorical += 1

    # ----- 数值字段中位数填补 -----
    NUM_COLS = ["credit_score", "disbursed_amount", "asset_cost",
                "total_outstanding_loan", "total_monthly_payment"]
    for c in NUM_COLS:
        if c not in df.columns:
            continue
        stats = fill_stats.get(c)
        if not stats:
            continue
        miss_mask = df[c].isna() | (df[c] == -1)
        n_miss = int(miss_mask.sum())
        if n_miss > 0:
            df.loc[miss_mask, c] = stats["median"]
            repaired_numeric += n_miss

    # 覆盖写回 cleaned 目录（DWD 层）
    df.to_csv(cleaned_path, index=False)

    extra = {
        "credit_level_repaired": repaired_categorical,
        "numeric_filled":        repaired_numeric,
        "rules_count":           len(rules),
        "method_categorical":    "FP-Growth 关联规则",
        "method_numeric":        "训练集中位数填补",
    }
    time.sleep(DEMO_SLEEP_SEC)
    job_store.finish_step(
        job_id, 5,
        message=f"修复完成：分类 {repaired_categorical} 处，数值 {repaired_numeric} 处",
        extra=extra,
    )
    return df, extra


# =============================================================================
#  Step 6: 特征工程
# =============================================================================
def _step6_features(job_id: str, df: pd.DataFrame) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    """生成特征快照（落盘 + 展示用），同时返回补齐原始列的 DataFrame 供 Step 7 推理。

    注意：返回 raw_df（仅补列、未做特征工程），因为 model_loader.predict_*
    内部会自行调用 add_features；如果传入已特征化的 df，会导致列名重复。
    """
    job_store.start_step(job_id, 6, "features_v3.add_features() 衍生比率/交叉特征")
    raw_df = df.copy()
    # 补齐 features_v3 需要但用户上传时可能没有的列（业务上 0 表示无该项行为）
    for col in FEATURE_INPUT_COLS:
        if col not in raw_df.columns:
            raw_df[col] = 0 if col != "disbursed_date" else "2020-01-01"

    # 仅用于"展示特征工程跑过"的快照（落盘到 featured 层）
    feat_df = add_features(raw_df.copy()).replace([np.inf, -np.inf], np.nan).fillna(0)

    HDFS_FEATURED_DIR.mkdir(parents=True, exist_ok=True)
    featured_path = HDFS_FEATURED_DIR / f"{job_id}_featured.csv"
    feat_df.to_csv(featured_path, index=False)

    time.sleep(DEMO_SLEEP_SEC)
    job_store.finish_step(
        job_id, 6,
        message=f"特征工程完成，特征维度 {feat_df.shape[1]} 列",
        extra={
            "featured_path": str(featured_path),
            "feature_count": int(feat_df.shape[1]),
            "row_count":     int(feat_df.shape[0]),
        },
    )
    return featured_path, feat_df, raw_df


# =============================================================================
#  Step 7: 模型推理
# =============================================================================
def _step7_predict(job_id: str, df: pd.DataFrame) -> list[dict]:
    job_store.start_step(job_id, 7, "三阶段串行预测：信用评分 → 欺诈检测 → 额度计算")

    records = df.to_dict(orient="records")
    defaults = predict_default(records)
    frauds   = predict_fraud(records)
    credit_scores = [d["credit_score"]      for d in defaults]
    fraud_probs   = [f["fraud_probability"] for f in frauds]
    limits = predict_limit(records, credit_scores=credit_scores, fraud_probs=fraud_probs)

    results = []
    for i, rec in enumerate(records):
        cs = defaults[i]["credit_score"]
        fp = frauds[i]["fraud_probability"]
        default_prob = 1.0 / (1.0 + 2 ** ((cs - 600) / 50))
        results.append({
            "customer_id":         rec.get("customer_id"),
            "credit_score":        round(float(cs), 1),
            "default_probability": round(float(default_prob), 4),
            "default_pred":        1 if default_prob >= 0.5 else 0,
            "fraud_probability":   round(float(fp), 4),
            "fraud_pred":          frauds[i]["fraud_pred"],
            "predicted_limit":     round(float(limits[i]["predicted_limit"]), 2),
        })

    avg_score = round(sum(r["credit_score"] for r in results) / len(results), 1) if results else 0
    n_default = sum(1 for r in results if r["default_pred"] == 1)
    n_fraud   = sum(1 for r in results if r["fraud_pred"] == 1)
    time.sleep(DEMO_SLEEP_SEC)
    job_store.finish_step(
        job_id, 7,
        message=f"预测完成，{len(results)} 条记录（违约 {n_default}，欺诈 {n_fraud}）",
        extra={
            "rows":         len(results),
            "avg_score":    avg_score,
            "default_hits": n_default,
            "fraud_hits":   n_fraud,
            "models_used":  ["default_model", "fraud_model", "rule_engine_limit"],
        },
    )
    return results


# =============================================================================
#  Step 8: 结果落库
# =============================================================================
def _step8_persist(job_id: str, results: list[dict]) -> int:
    job_store.start_step(job_id, 8, "决策结果写入 loan_rt.realtime_decisions")
    ok = 0
    for r in results:
        try:
            insert_realtime_decision({
                "customer_id":         r["customer_id"],
                "default_probability": r["default_probability"],
                "default_pred":        r["default_pred"],
                "fraud_probability":   r["fraud_probability"],
                "fraud_pred":          r["fraud_pred"],
                "predicted_limit":     r["predicted_limit"],
                "credit_score":        r["credit_score"],
            })
            ok += 1
        except Exception:
            pass
    time.sleep(DEMO_SLEEP_SEC)
    job_store.finish_step(
        job_id, 8,
        message=f"已写入 loan_rt.realtime_decisions（{ok} 条）",
        extra={"persisted": ok, "table": "loan_rt.realtime_decisions"},
    )
    return ok


# =============================================================================
#  Pipeline 主入口
# =============================================================================
def _read_upload(file_path: Path) -> pd.DataFrame:
    ext = file_path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    if ext == ".csv":
        return pd.read_csv(file_path)
    raise ValueError(f"不支持的文件类型: {ext}（请上传 .xlsx / .xls / .csv）")


def _run(job_id: str, upload_path: Path) -> None:
    try:
        df_in = _read_upload(upload_path)
        job_store.set_total_rows(job_id, len(df_in))

        if df_in.empty:
            raise ValueError("上传文件为空")
        if "customer_id" not in df_in.columns:
            raise ValueError("缺少必填列 customer_id")

        # 8 步流水
        _step1_mysql_staging(job_id, df_in.copy())
        spool_path           = _step2_flume_spool(job_id, df_in.copy())
        raw_path             = _step3_hdfs_raw(job_id, spool_path)
        cleaned_path, df_cl, _      = _step4_clean(job_id, raw_path)
        df_rp, repair_info          = _step5_repair(job_id, cleaned_path, df_cl)
        _, _, df_for_predict        = _step6_features(job_id, df_rp)
        results                     = _step7_predict(job_id, df_for_predict)
        persisted                   = _step8_persist(job_id, results)

        # 完成汇总
        avg_score = round(sum(r["credit_score"] for r in results) / len(results), 1) if results else 0
        avg_limit = round(sum(r["predicted_limit"] for r in results) / len(results), 2) if results else 0
        n_default = sum(1 for r in results if r["default_pred"] == 1)
        n_fraud   = sum(1 for r in results if r["fraud_pred"] == 1)

        job_store.mark_done(job_id, {
            "rows_in":           int(len(df_in)),
            "rows_predicted":    len(results),
            "rows_persisted":    persisted,
            "avg_credit_score":  avg_score,
            "avg_predicted_limit": avg_limit,
            "default_hits":      n_default,
            "fraud_hits":        n_fraud,
            "repair_info":       repair_info,
            "results":           results[:200],   # 前端展示最多 200 条
        })
    except Exception as exc:
        import traceback
        traceback.print_exc()
        # 找到当前 running 的 step 标记失败
        job = job_store.get_job(job_id)
        cur_step = 1
        if job and job.get("steps_json"):
            for s in job["steps_json"]:
                if s.get("status") == "running":
                    cur_step = s["id"]
                    break
                if s.get("status") == "pending":
                    cur_step = s["id"]
                    break
        job_store.fail_step(job_id, cur_step, f"{type(exc).__name__}: {exc}")


def run_pipeline(job_id: str, upload_path: Path) -> threading.Thread:
    """后台线程异步跑 pipeline，前端通过 job_id 轮询进度。"""
    t = threading.Thread(target=_run, args=(job_id, upload_path), daemon=True,
                         name=f"pipeline-{job_id}")
    t.start()
    return t


def get_job(job_id: str) -> dict | None:
    return job_store.get_job(job_id)


def list_jobs(limit: int = 20) -> list[dict]:
    return job_store.list_jobs(limit)
