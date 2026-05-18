"""On-demand SHAP explanation for a single customer.

Reuses the XGBoost regressor in `default_model.joblib` and shap.TreeExplainer.
TreeExplainer + 1 row is fast (<10ms), so we compute on every request.
The explainer itself is initialised once (lazy) and cached at module scope.
"""

from __future__ import annotations

import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify

from features_v3 import add_features
from service.flask.config import Settings
from service.flask.repositories.mysql_repo import fetch_customer_profile

model_explain_bp = Blueprint("model_explain_bp", __name__, url_prefix="/model")

_explainer_lock = threading.Lock()
_explainer_cache: dict = {}

# 与 scripts/precompute_shap.py 保持同步的展示名映射（节选最常见的）
_DISPLAY_MAP: dict[str, str] = {
    "composite_risk_score": "复合风险评分",
    "composite_risk_level": "风险等级",
    "credit_score_sq": "信用评分平方",
    "credit_score": "信用评分",
    "credit_gap": "信用缺口",
    "credit_history": "信用记录时长",
    "credit_utilization": "信用利用率",
    "area_id": "地区编码",
    "branch_id": "网点编码",
    "branch_id_freq": "网点频率",
    "area_id_freq": "地区频率",
    "supplier_id": "供应商编码",
    "supplier_id_freq": "供应商频率",
    "manufacturer_id": "厂商编码",
    "total_overdue_no": "总逾期次数",
    "overdue_rate_total": "总逾期率",
    "enquirie_no": "征信查询次数",
    "inquiry_intensity": "查询强度",
    "inquiry_frequency": "查询频率",
    "disbursed_amount": "贷款金额",
    "asset_cost": "资产价值",
    "loan_to_asset_ratio": "贷款/资产比",
    "ltv_ratio": "贷款资产比",
    "loan_per_age": "贷款/年龄",
    "loan_asset_gap": "贷款-资产缺口",
    "age": "年龄",
    "average_age": "平均账龄",
    "total_monthly_payment": "月供金额",
    "payment_burden": "月供负担",
    "monthly_payment_ratio": "月供占比",
    "employment_type": "工作类型",
    "Credit_level": "信用等级",
    "active_account_ratio": "活跃账户比",
    "active_to_inactive_act_ratio": "活/非活账户比",
    "main_account_active_loan_no": "主账户活跃贷款数",
    "main_account_overdue_no": "主账户逾期次数",
    "main_account_tenure": "主账户存续期",
    "main_overdue_density": "主账户逾期密度",
    "sub_account_loan_no": "副账户贷款数",
    "sub_account_overdue_no": "副账户逾期次数",
    "sub_account_tenure": "副账户存续期",
    "last_six_month_new_loan_no": "近6月新增贷款",
    "last_six_month_defaulted_no": "近6月违约次数",
    "new_loan_velocity": "新贷款增速",
    "recent_stress": "近期压力指数",
    "debt_pressure": "债务压力",
    "outstanding_disburse_ratio": "未偿/发放比",
    "disburse_to_sactioned_ratio": "发放/批准比",
    "total_outstanding_loan": "未偿贷款总额",
    "total_sanction_loan": "批准贷款总额",
    "total_disbursed_loan": "已发放贷款总额",
    "sanction_disburse_gap": "审批-放款差额",
    "total_account_loan_no": "总账户数",
    "employee_code_id_freq": "员工编码频率",
}


def _load_explainer() -> dict:
    """Lazy-load XGB bundle + shap.TreeExplainer; cached at module scope."""
    if _explainer_cache:
        return _explainer_cache
    with _explainer_lock:
        if _explainer_cache:
            return _explainer_cache
        bundle_path = Path(Settings.MODEL_DIR) / "default_model.joblib"
        bundle = joblib.load(bundle_path)
        if bundle.get("model_type") != "xgboost":
            raise RuntimeError(
                f"SHAP 解释当前只支持 xgboost 胜出模型，winner={bundle.get('model_type')}"
            )
        import shap  # 延迟导入，避免无 SHAP 环境时整个应用启动失败
        model = bundle["model"]
        explainer = shap.TreeExplainer(model)
        _explainer_cache.update({
            "model":        model,
            "explainer":    explainer,
            "feature_cols": bundle["feature_cols"],
            "base_value":   float(np.array(explainer.expected_value).flatten()[0]),
        })
        return _explainer_cache


def _build_explanation(profile: dict, top_k: int = 8) -> dict:
    """Compute SHAP values for one customer and shape it like shap_samples.json items."""
    ctx = _load_explainer()
    model       = ctx["model"]
    explainer   = ctx["explainer"]
    feature_cols = ctx["feature_cols"]
    base_value  = ctx["base_value"]

    df = add_features(pd.DataFrame([profile])).replace([np.inf, -np.inf], np.nan)
    X = df.reindex(columns=feature_cols).fillna(0)

    shap_row = np.array(explainer.shap_values(X)).reshape(-1)
    pred = float(np.clip(model.predict(X), 300, 850)[0])
    p_default = 1.0 / (1.0 + 2 ** ((pred - 600) / 50))

    order = np.argsort(-np.abs(shap_row))[:top_k]
    items = []
    for fi in order:
        col = feature_cols[fi]
        items.append({
            "name":    col,
            "display": _DISPLAY_MAP.get(col, col),
            "value":   float(shap_row[fi]),
        })
    other_sum = float(shap_row.sum() - sum(it["value"] for it in items))
    items.append({"name": "_other", "display": "其他因素", "value": other_sum})

    if p_default < 0.25:
        label = "低风险客户"
    elif p_default > 0.55:
        label = "高风险客户"
    else:
        label = "中风险客户"

    return {
        "label":        label,
        "credit_score": round(pred, 1),
        "p_default":    round(p_default, 4),
        "base_value":   round(base_value, 2),
        "items":        items,
    }


@model_explain_bp.get("/explain/<int:customer_id>")
def explain_customer(customer_id: int):
    """Return on-demand SHAP waterfall for a single customer."""
    profile = fetch_customer_profile(customer_id)
    if not profile:
        return jsonify({"error": "customer not found", "customer_id": customer_id}), 404
    try:
        explanation = _build_explanation(profile)
    except Exception as exc:
        return jsonify({"error": f"shap compute failed: {exc}"}), 500
    return jsonify({"customer_id": customer_id, **explanation})
