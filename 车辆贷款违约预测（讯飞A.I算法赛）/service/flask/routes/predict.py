from __future__ import annotations

from flask import Blueprint, jsonify, request

from service.flask.model_loader import predict_default, predict_fraud, predict_limit
from service.flask.repositories.mysql_repo import insert_realtime_decision

predict_bp = Blueprint("predict_bp", __name__)


@predict_bp.post("/predict/default")
def api_predict_default():
    """信用评分预测（回归），返回 credit_score ∈ [300, 850]。"""
    payload = request.get_json(force=True)
    records = payload if isinstance(payload, list) else [payload]
    return jsonify(predict_default(records))


@predict_bp.post("/predict/fraud")
def api_predict_fraud():
    """欺诈检测（分类），返回 fraud_probability 和 fraud_pred。"""
    payload = request.get_json(force=True)
    records = payload if isinstance(payload, list) else [payload]
    return jsonify(predict_fraud(records))


@predict_bp.post("/predict/limit")
def api_predict_limit():
    """贷款额度预测，返回 predicted_limit。"""
    payload = request.get_json(force=True)
    records = payload if isinstance(payload, list) else [payload]
    return jsonify(predict_limit(records))


@predict_bp.post("/predict/full")
def api_predict_full():
    """
    全量决策接口：信用评分 + 欺诈检测 + 额度计算，合并后写入数据库。
    额度采用业务规则公式，依赖前两步的输出。
    """
    payload = request.get_json(force=True)
    records = payload if isinstance(payload, list) else [payload]

    defaults = predict_default(records)
    frauds   = predict_fraud(records)
    # 把已算好的评分和欺诈概率传给 predict_limit，避免重复推理
    credit_scores = [d["credit_score"]      for d in defaults]
    fraud_probs   = [f["fraud_probability"] for f in frauds]
    limits        = predict_limit(records, credit_scores=credit_scores,
                                   fraud_probs=fraud_probs)

    merged = []
    for i in range(len(records)):
        item = {
            "customer_id":       records[i].get("customer_id"),
            "credit_score":      defaults[i]["credit_score"],
            "fraud_probability": frauds[i]["fraud_probability"],
            "fraud_pred":        frauds[i]["fraud_pred"],
            "predicted_limit":   limits[i]["predicted_limit"],
        }
        insert_realtime_decision(item)
        merged.append(item)

    return jsonify(merged)
