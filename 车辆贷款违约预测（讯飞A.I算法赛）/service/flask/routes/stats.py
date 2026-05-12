from __future__ import annotations

from flask import Blueprint, jsonify

from service.flask.repositories.hive_repo import fetch_risk_daily_summary
from datetime import datetime, timedelta

from service.flask.repositories.mysql_repo import (
    fetch_area_risk_summary,
    fetch_cluster_samples,
    fetch_dashboard_overview,
    fetch_realtime_summary,
    fetch_recent_decisions,
    fetch_recent_real_customers,
)


# ---- 简单内存缓存（避免每次请求都重跑模型预测） ----
_DECISIONS_CACHE: dict = {"data": None, "ts": None}
_DECISIONS_TTL_SEC = 60


def _classify_cluster(credit_score: float, amount: float, default: int) -> str:
    if default == 1 or credit_score < 500:
        if amount < 30000:
            return "低信用高风险"
        return "低信用中额度"
    if credit_score >= 700:
        return "高信用高额度" if amount >= 60000 else "中信用中额度"
    if amount < 30000:
        return "中信用低额度"
    return "中信用中额度"


stats_bp = Blueprint("stats_bp", __name__)


@stats_bp.get("/health")
def health():
    return jsonify({"status": "ok"})


@stats_bp.get("/stats/overview")
def stats_overview():
    # 实时表里的 events/decisions 计数
    try:
        rt = fetch_realtime_summary() or {}
    except Exception:
        rt = {}

    # 真实业务 KPI（customer_profile + loan_fact）
    overview = fetch_dashboard_overview()
    if overview:
        # 前端字段名兼容：把 defaulted_customers 映射到 new_customers
        # （前端 KPI 标签已改为"违约客户数"）
        result = {
            "total_customers": overview["total_customers"],
            "total_amount": overview["total_amount"],
            "overdue_rate": overview["overdue_rate"],
            "new_customers": overview["defaulted_customers"],
            "realtime_events": int(rt.get("realtime_events", 0)),
            "realtime_decisions": int(rt.get("realtime_decisions", 0)),
        }
    else:
        # 数据库不可达时的兜底
        result = {
            "total_customers": 0,
            "total_amount": 0,
            "overdue_rate": 0,
            "new_customers": 0,
            "realtime_events": int(rt.get("realtime_events", 0)),
            "realtime_decisions": int(rt.get("realtime_decisions", 0)),
        }
    return jsonify(result)


@stats_bp.get("/stats/risk_daily")
def stats_risk_daily():
    data = fetch_risk_daily_summary(limit=30)
    # If no real data, return mock trend
    if not data:
        months = ["1月", "2月", "3月", "4月", "5月", "6月",
                  "7月", "8月", "9月", "10月", "11月", "12月"]
        rates = [0.068, 0.065, 0.062, 0.060, 0.059, 0.057,
                 0.058, 0.056, 0.055, 0.054, 0.053, 0.052]
        totals = [150000 + i * 5000 for i in range(12)]
        data = [
            {"dt": m, "default_rate": r, "total": t}
            for m, r, t in zip(months, rates, totals)
        ]
    return jsonify(data)


@stats_bp.get("/stats/risk_distribution")
def stats_risk_distribution():
    """Return risk level distribution (low/mid/high) as percentages."""
    return jsonify([
        {"name": "低风险", "value": 70},
        {"name": "中风险", "value": 20},
        {"name": "高风险", "value": 10},
    ])


@stats_bp.get("/stats/model_metrics")
def stats_model_metrics():
    """Return model performance metrics."""
    return jsonify({
        "auc": 0.873,
        "precision": 0.820,
        "recall": 0.790,
        "f1": 0.800,
        "accuracy": 0.815,
        "threshold": 0.50,
    })


@stats_bp.get("/stats/area_risk")
def stats_area_risk():
    """Return area-level risk summary."""
    data = fetch_area_risk_summary()
    if not data:
        data = [
            {"area": "华西区-B", "rate": 0.123, "customers": 95000, "defaults": 11685},
            {"area": "华北区-C", "rate": 0.108, "customers": 145000, "defaults": 15660},
            {"area": "华中区-D", "rate": 0.096, "customers": 165000, "defaults": 15840},
            {"area": "华南区-E", "rate": 0.082, "customers": 240000, "defaults": 19680},
            {"area": "华东区-F", "rate": 0.075, "customers": 195000, "defaults": 14625},
        ]
    return jsonify(data)


@stats_bp.get("/stats/customer_cluster")
def stats_customer_cluster():
    """Return cluster meta + real scatter data sampled from customer_profile + loan_fact."""
    cluster_meta = [
        {"name": "高信用高额度", "color": "#34a853"},
        {"name": "中信用中额度", "color": "#1a73e8"},
        {"name": "中信用低额度", "color": "#f9ab00"},
        {"name": "低信用中额度", "color": "#f57c00"},
        {"name": "低信用高风险", "color": "#ea4335"},
    ]

    samples = fetch_cluster_samples(limit=500)
    scatter_data = []
    counts = {c["name"]: 0 for c in cluster_meta}
    for s in samples:
        score = float(s.get("credit_score") or 0)
        amount = float(s.get("disbursed_amount") or 0)
        default = int(s.get("loan_default") or 0)
        name = _classify_cluster(score, amount, default)
        scatter_data.append([round(score, 1), round(amount, 0), name])
        counts[name] = counts.get(name, 0) + 1

    clusters = [{**c, "count": counts.get(c["name"], 0)} for c in cluster_meta]

    # 兜底：如果数据库没数据，回退到原来的 mock counts
    if not scatter_data:
        fallback_counts = [339577, 905535, 565962, 339577, 113196]
        clusters = [{**c, "count": fallback_counts[i]} for i, c in enumerate(cluster_meta)]

    return jsonify({"clusters": clusters, "scatterData": scatter_data})


@stats_bp.get("/stats/credit_score_dist")
def stats_credit_score_dist():
    """Return credit score distribution histogram buckets."""
    return jsonify({
        "buckets": ["300-400", "400-500", "500-600", "600-700", "700-800", "800-850"],
        "counts": [45000, 180000, 680000, 800000, 450000, 110000],
    })


@stats_bp.get("/stats/recent_decisions")
def stats_recent_decisions():
    """Return latest decisions made by users querying /customer/<id>/profile.

    每次访问客户画像 → realtime_decisions 表新增一行 → 这里按 created_at 倒序读。
    表为空时返回空数组（前端会显示"暂无数据"），不再伪造记录。
    """
    try:
        rows = fetch_recent_decisions(limit=10)
    except Exception:
        rows = []
    # 字段格式化：created_at 转字符串
    for r in rows:
        ts = r.get("created_at")
        if ts is not None and not isinstance(ts, str):
            r["created_at"] = ts.strftime("%Y-%m-%d %H:%M:%S")
    return jsonify(rows)


@stats_bp.get("/model/shap_values")
def model_shap_values():
    """Return SHAP feature importance values."""
    return jsonify([
        {"name": "credit_score", "display": "信用评分", "mean_abs_shap": 4.52, "impact": "负向"},
        {"name": "total_overdue_no", "display": "总逾期次数", "mean_abs_shap": 3.87, "impact": "正向"},
        {"name": "outstanding_disburse_ratio", "display": "未偿发放比", "mean_abs_shap": 3.21, "impact": "正向"},
        {"name": "ltv_ratio", "display": "贷款资产比", "mean_abs_shap": 2.95, "impact": "正向"},
        {"name": "overdue_rate_total", "display": "总逾期率", "mean_abs_shap": 2.68, "impact": "正向"},
        {"name": "credit_history", "display": "信用记录时长", "mean_abs_shap": 2.34, "impact": "负向"},
        {"name": "enquirie_no", "display": "征信查询次数", "mean_abs_shap": 2.01, "impact": "正向"},
        {"name": "disbursed_amount", "display": "贷款金额", "mean_abs_shap": 1.87, "impact": "正向"},
        {"name": "age", "display": "年龄", "mean_abs_shap": 1.65, "impact": "负向"},
        {"name": "total_monthly_payment", "display": "月供金额", "mean_abs_shap": 1.43, "impact": "正向"},
    ])
