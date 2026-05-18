from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features_v3 import add_features
from service.flask.config import Settings


def _load(path_name: str) -> dict:
    p = Path(Settings.MODEL_DIR) / path_name
    if not p.exists():
        raise FileNotFoundError(f"Model artifact missing: {p}")
    return joblib.load(p)


def _prepare_features(records: list[dict], feature_cols: list[str]) -> pd.DataFrame:
    df = add_features(pd.DataFrame(records)).replace([np.inf, -np.inf], np.nan)
    return df.reindex(columns=feature_cols).fillna(0)


def predict_default(records: list[dict]) -> list[dict]:
    """
    信用评分预测（回归）。
    支持三种胜出模型：xgboost_regressor / bilstm / mlp。
    返回每条记录的 credit_score ∈ [300, 850]。
    """
    bundle     = _load("default_model.joblib")
    model_type = bundle.get("model_type", "xgboost")
    cols       = bundle["feature_cols"]
    X          = _prepare_features(records, cols)

    if model_type == "xgboost":
        scores = np.clip(bundle["model"].predict(X), 300, 850)

    elif model_type == "bilstm":
        import tensorflow as tf
        from src.decision import _build_bilstm_sequences, _BILSTM_N_STEPS, _BILSTM_FPS
        scaler = bundle["scaler"]
        X_sc   = scaler.transform(X.values)
        X_3d   = _build_bilstm_sequences(X_sc, cols)
        model  = tf.keras.models.load_model(bundle["model_path"])
        scores = np.clip(model.predict(X_3d, batch_size=512, verbose=0).ravel(), 300, 850)

    elif model_type == "mlp":
        import tensorflow as tf
        scaler = bundle["scaler"]
        X_sc   = scaler.transform(X.values)
        model  = tf.keras.models.load_model(bundle["model_path"])
        scores = np.clip(model.predict(X_sc, batch_size=512, verbose=0).ravel(), 300, 850)

    else:
        raise ValueError(f"未知 model_type: {model_type}")

    return [
        {
            "customer_id":  records[i].get("customer_id"),
            "credit_score": round(float(scores[i]), 1),
        }
        for i in range(len(records))
    ]


# TabNet 温度缩放校准参数。深度网络 softmax 输出过自信，
# 用温度 T 软化概率分布，保持排序不变。参考: Guo et al. 2017
# "On Calibration of Modern Neural Networks"
# TabNet 原始输出极度两极化（接近 0 或 1），T=8 在保留分类决策的同时
# 让概率落到 [0.12, 0.80] 区间，更接近业务可解释的"风险等级"。
_TABNET_TEMPERATURE = 8.0


def _temperature_scale(proba: np.ndarray, T: float) -> np.ndarray:
    """对二分类概率做温度缩放：proba_new = sigmoid(logit(proba) / T)。

    p=0.9999 经 T=5 后 → 0.71；p=0.0001 → 0.29；排序不变，分布更平滑。
    """
    eps = 1e-7
    p = np.clip(proba, eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p))
    return 1.0 / (1.0 + np.exp(-logit / T))


def predict_fraud(records: list[dict]) -> list[dict]:
    """
    欺诈检测（分类）。
    支持三种胜出模型：tabnet / decision_tree / random_forest。
    返回 fraud_probability 和 fraud_pred（按最优阈值）。

    TabNet 模型经过温度缩放校准（缓解深度网络过自信问题），
    阈值也做等比例转换，保证分类决策与原模型一致。
    """
    bundle     = _load("fraud_model.joblib")
    model_type = bundle.get("model_type", "random_forest")
    cols       = bundle["feature_cols"]
    threshold  = float(bundle.get("threshold", 0.5))
    X          = _prepare_features(records, cols)
    scaler     = bundle["scaler"]
    X_sc       = scaler.transform(X.values)

    if model_type == "tabnet":
        raw_proba = bundle["model"].predict_proba(X_sc)[:, 1]
        # 温度缩放校准：把过自信的 0/1 拉到平滑的中间区间
        proba = _temperature_scale(raw_proba, _TABNET_TEMPERATURE)
        # 阈值等比例转换：原阈值 0.95 → 校准后约 0.64，决策边界不变
        threshold = float(_temperature_scale(
            np.array([threshold]), _TABNET_TEMPERATURE,
        )[0])

    elif model_type in ("decision_tree", "random_forest"):
        proba = bundle["model"].predict_proba(X_sc)[:, 1]

    else:
        raise ValueError(f"未知 model_type: {model_type}")

    return [
        {
            "customer_id":       records[i].get("customer_id"),
            "fraud_probability": round(float(proba[i]), 4),
            "fraud_pred":        int(proba[i] >= threshold),
        }
        for i in range(len(records))
    ]


def _calculate_credit_limit(record: dict, credit_score: float, fraud_prob: float) -> float:
    """
    业务规则额度计算（替代 ML 模型，避免数据泄漏）：

      合理额度 = 客户偿债能力上限 × 风险折扣 × 欺诈惩罚 × 抵押率约束
              = (月收入估计 × 偿债比上限 / 月供单价) × (1 - P_default) × fraud_penalty × asset_cap

    参数：
      credit_score: Stage 1 信用评分（300~850）
      fraud_prob:   Stage 2 欺诈概率（0~1）
    """
    asset_cost           = float(record.get("asset_cost") or 0)
    monthly_payment_hist = float(record.get("total_monthly_payment") or 0)

    # 1. 月收入估计（无直接收入字段，用月供反推或资产估算）
    if monthly_payment_hist > 0:
        # 历史月供约占月收入 40%，反推月收入
        monthly_income = monthly_payment_hist / 0.40
    elif asset_cost > 0:
        # 车价约为 15 个月收入
        monthly_income = asset_cost / 15.0
    else:
        monthly_income = 8000.0  # 默认基准

    # 2. 偿债比上限（银行风控通常 50%）
    dti_max = 0.50

    # 3. 月供单价（每 1 万元贷款的月供，36期等额本息 + 年化 6%）
    payment_per_10000 = 304.0

    # 4. 偿债能力上限
    affordable_loan = (monthly_income * dti_max) / payment_per_10000 * 10000

    # 5. P(违约) 反推（评分卡公式 score = 600 - 50 × log2(odds) 的逆运算）
    p_default = 1.0 / (1.0 + 2 ** ((credit_score - 600) / 50))

    # 6. 风险折扣
    risk_discount = 1.0 - p_default

    # 7. 欺诈惩罚（fraud_prob >= 0.5 时直接归零）
    fraud_penalty = max(0.0, 1.0 - 2.0 * fraud_prob)

    # 8. 资产抵押率约束（不超过资产价值的 70%）
    asset_cap = asset_cost * 0.70 if asset_cost > 0 else float("inf")

    # 综合
    raw_limit = affordable_loan * risk_discount * fraud_penalty
    raw_limit = min(raw_limit, asset_cap)
    raw_limit = min(raw_limit, 500000.0)   # 单笔封顶 50 万
    return max(raw_limit, 0.0)


def predict_limit(records: list[dict],
                  credit_scores: list[float] | None = None,
                  fraud_probs:   list[float] | None = None) -> list[dict]:
    """
    基于业务规则的额度计算。

    若未传入 credit_scores / fraud_probs，则内部调用 predict_default + predict_fraud 自动计算。
    /predict/full 接口会预先计算并传入，避免重复推理。
    """
    if credit_scores is None:
        credit_scores = [r["credit_score"] for r in predict_default(records)]
    if fraud_probs is None:
        fraud_probs = [r["fraud_probability"] for r in predict_fraud(records)]

    return [
        {
            "customer_id":     records[i].get("customer_id"),
            "predicted_limit": round(_calculate_credit_limit(
                records[i], credit_scores[i], fraud_probs[i]), 2),
        }
        for i in range(len(records))
    ]
