from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features_v2 import add_features
from service.flask.config import Settings


def _load(path_name: str) -> dict:
    p = Path(Settings.MODEL_DIR) / path_name
    if not p.exists():
        raise FileNotFoundError(f"Model artifact missing: {p}")
    return joblib.load(p)


def predict_default(records: list[dict]) -> list[dict]:
    bundle = _load("default_model.joblib")
    model = bundle["model"]
    cols = bundle["feature_cols"]
    df = add_features(pd.DataFrame(records)).replace([np.inf, -np.inf], np.nan)
    x = df.reindex(columns=cols)
    proba = model.predict_proba(x)[:, 1]
    out = []
    for i, p in enumerate(proba):
        out.append(
            {
                "customer_id": records[i].get("customer_id"),
                "default_probability": float(p),
                "default_pred": int(p >= 0.5),
            }
        )
    return out


def predict_fraud(records: list[dict]) -> list[dict]:
    bundle = _load("fraud_model.joblib")

    if bundle.get("type") == "fraud_deep":
        import tensorflow as tf
        model    = tf.keras.models.load_model(bundle["model_path"])
        scaler   = bundle["scaler"]
        cols     = bundle["feature_cols"]
        n_steps  = bundle["n_steps"]
        fps      = bundle["features_per_step"]
        df = add_features(pd.DataFrame(records)).replace([np.inf, -np.inf], np.nan)
        x = df.reindex(columns=cols).fillna(0).values.astype(np.float32)
        x_sc = scaler.transform(x)
        x_3d = x_sc[:, : n_steps * fps].reshape(-1, n_steps, fps)
        proba = model.predict(x_3d, batch_size=512, verbose=0).ravel()
    else:
        model = bundle["model"]
        cols  = bundle["feature_cols"]
        df = add_features(pd.DataFrame(records)).replace([np.inf, -np.inf], np.nan)
        x = df.reindex(columns=cols)
        proba = model.predict_proba(x)[:, 1]

    return [
        {
            "customer_id":      records[i].get("customer_id"),
            "fraud_probability": float(proba[i]),
            "fraud_pred":        int(proba[i] >= 0.5),
        }
        for i in range(len(records))
    ]


def predict_limit(records: list[dict]) -> list[dict]:
    bundle = _load("limit_model.joblib")
    model = bundle["model"]
    cols = bundle["feature_cols"]
    fill_values = bundle.get("fill_values", {})
    df = add_features(pd.DataFrame(records)).replace([np.inf, -np.inf], np.nan)
    x = df.reindex(columns=cols).fillna(fill_values).fillna(0)
    pred = model.predict(x)
    return [
        {
            "customer_id": records[i].get("customer_id"),
            "predicted_limit": float(max(pred[i], 0.0)),
        }
        for i in range(len(records))
    ]


def score_credit(default_probability: list[float]) -> list[float]:
    p = np.clip(np.array(default_probability, dtype=float), 1e-6, 1 - 1e-6)
    odds = p / (1 - p)
    score = 600 - 50 * np.log(odds)
    return np.clip(score, 300, 850).tolist()
