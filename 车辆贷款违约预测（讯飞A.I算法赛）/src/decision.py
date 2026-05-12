import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from features_v3 import add_features
from src.config import ProjectConfig

_FRAUD_N_STEPS = 5
# 构成欺诈标签的原始字段 + 其衍生列，训练时全部剔除，避免直接和间接泄漏
_FRAUD_LABEL_COLS = [
    # 原始标签构成列
    "enquirie_no", "last_six_month_new_loan_no",
    "total_overdue_no", "idcard_flag", "mobileno_flag",
    # enquirie_no 的衍生列
    "enquiry_per_age", "inquiry_frequency", "high_inquiry",
    # last_six_month_new_loan_no 的衍生列
    "recent_default_rate", "new_loan_velocity",
    # total_overdue_no 的衍生列
    "overdue_rate_total",
    # 包含上述列的复合特征
    "cs_x_overdue", "cs_x_recent_def",
    "composite_risk_score", "composite_risk_level",
    # 标识符列（不应参与训练）
    "customer_id",
]


def _load_training_data(cfg: ProjectConfig) -> pd.DataFrame:
    repaired = cfg.featured_dir / "train_repaired.csv"
    if repaired.exists():
        return pd.read_csv(repaired)
    fallback = cfg.cleaned_dir / "train_cleaned.csv"
    if fallback.exists():
        return pd.read_csv(fallback)
    raise FileNotFoundError("No cleaned/repaired training data found. Run ingest and repair modules first.")


def _clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """去除常数列和修复流程产生的重复列（repaired_* 前缀），避免引入噪声。"""
    const_cols = [c for c in X.columns if X[c].nunique() <= 1]
    rep_cols   = [c for c in X.columns if c.startswith("repaired_")]
    return X.drop(columns=const_cols + rep_cols, errors="ignore")


def train_default_model(cfg: ProjectConfig, df: pd.DataFrame) -> dict:
    work = df.copy()
    y = work["loan_default"].astype(int)
    X = work.drop(columns=["loan_default"])
    X = add_features(X).replace([np.inf, -np.inf], np.nan)
    X_num = _clean_features(X.select_dtypes(include=[np.number])).fillna(0)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_num, y, test_size=0.2, random_state=42, stratify=y
    )

    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = float(neg / max(pos, 1))
    feature_cols = X_train.columns.tolist()

    # --- LightGBM ---
    lgb_model = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.5,
        scale_pos_weight=pos_weight,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )
    p_lgb = lgb_model.predict_proba(X_valid)[:, 1]
    auc_lgb = float(roc_auc_score(y_valid, p_lgb))

    # --- XGBoost ---
    xgb_model = XGBClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=pos_weight,
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    p_xgb = xgb_model.predict_proba(X_valid)[:, 1]
    auc_xgb = float(roc_auc_score(y_valid, p_xgb))

    # --- 集成：等权混合两个模型概率 ---
    p_blend = 0.5 * p_lgb + 0.5 * p_xgb
    auc_blend = float(roc_auc_score(y_valid, p_blend))
    print(f"[Default] LightGBM={auc_lgb:.4f}  XGBoost={auc_xgb:.4f}  Blend={auc_blend:.4f}")

    # 最优阈值（F1 最大化）—— 用 blend 概率
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thr = max(
        thresholds,
        key=lambda t: f1_score(y_valid, (p_blend >= t).astype(int), zero_division=0),
    )
    y_pred = (p_blend >= best_thr).astype(int)

    metrics = {
        "auc":       float(roc_auc_score(y_valid, p_blend)),
        "accuracy":  float(accuracy_score(y_valid, y_pred)),
        "precision": float(precision_score(y_valid, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_valid, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_valid, y_pred, zero_division=0)),
        "threshold": float(best_thr),
        "lgb_auc":   auc_lgb,
        "xgb_auc":   auc_xgb,
        "blend_auc": auc_blend,
    }
    artifact = {
        "lgb_model":    lgb_model,
        "xgb_model":    xgb_model,
        "feature_cols": feature_cols,
        "metrics":      metrics,
        "type":         "default_blend",
        "threshold":    float(best_thr),
    }
    out = cfg.artifacts_dir / "default_model.joblib"
    joblib.dump(artifact, out)
    return {"artifact": str(out), "metrics": metrics}


def _prep_fraud_sequences(X_num: pd.DataFrame, scaler=None):
    """将数值特征标准化后切成 (N_STEPS, features_per_step) 的序列张量。"""
    arr = X_num.fillna(0).values.astype(np.float32)
    if scaler is None:
        scaler = StandardScaler()
        arr = scaler.fit_transform(arr)
    else:
        arr = scaler.transform(arr)
    fps = arr.shape[1] // _FRAUD_N_STEPS          # features per step
    arr = arr[:, : _FRAUD_N_STEPS * fps]           # 裁到整除
    return arr.reshape(-1, _FRAUD_N_STEPS, fps), scaler, fps


def _build_lstm_model(n_steps: int, fps: int) -> tf.keras.Model:
    """双向 LSTM 欺诈检测模型。
    结构：BiLSTM(64) → Dropout → LSTM(32) → Dropout → Dense(16) → 二分类输出
    """
    inp = tf.keras.Input(shape=(n_steps, fps), name="fraud_input")
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(inp)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="fraud_output")(x)
    return tf.keras.Model(inp, out, name="fraud_bilstm")


def _build_transformer_model(n_steps: int, fps: int) -> tf.keras.Model:
    """Transformer Encoder 欺诈检测模型。
    结构：MultiHeadAttention → Add&Norm → FFN → Add&Norm → GAP → Dense → 二分类输出
    每个"时间步"作为一个 token，4 头注意力捕捉特征组之间的全局依赖。
    """
    inp = tf.keras.Input(shape=(n_steps, fps), name="fraud_input")
    # --- Transformer Encoder Block ---
    attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(inp, inp)
    x = tf.keras.layers.Add()([inp, attn])                        # 残差连接
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    ff = tf.keras.layers.Dense(128, activation="relu")(x)
    ff = tf.keras.layers.Dropout(0.2)(ff)
    ff = tf.keras.layers.Dense(fps)(ff)
    x = tf.keras.layers.Add()([x, ff])                            # 残差连接
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    # --- 分类头 ---
    x = tf.keras.layers.GlobalAveragePooling1D()(x)               # 聚合序列维度
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="fraud_output")(x)
    return tf.keras.Model(inp, out, name="fraud_transformer")


def _train_deep_model(X_3d_tr, y_tr, X_3d_val, y_val, model, name, cfg, pos_weight):
    """训练单个 Keras 模型，保存到 artifacts/，返回 (metrics_dict, model_path_str)。"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
    ]
    model.fit(
        X_3d_tr, y_tr,
        validation_data=(X_3d_val, y_val),
        epochs=30,
        batch_size=1024,
        class_weight={0: 1.0, 1: float(pos_weight)},
        callbacks=callbacks,
        verbose=0,
    )
    proba = model.predict(X_3d_val, batch_size=1024, verbose=0).ravel()

    # 最优阈值（F1 最大化）
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thr = max(thresholds, key=lambda t: f1_score(y_val, (proba >= t).astype(int), zero_division=0))
    y_pred = (proba >= best_thr).astype(int)

    metrics = {
        "auc":       float(roc_auc_score(y_val, proba)),
        "accuracy":  float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_val, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_val, y_pred, zero_division=0)),
        "threshold": float(best_thr),
    }
    model_path = cfg.artifacts_dir / f"fraud_{name}_model.keras"
    model.save(str(model_path))
    print(
        f"[Fraud {name.upper():11s}] "
        f"AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}  "
        f"Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}  "
        f"Threshold={best_thr:.2f}"
    )
    return metrics, str(model_path)


def _build_fraud_label(df: pd.DataFrame) -> pd.Series:
    score = (
        (df["enquirie_no"] > df["enquirie_no"].quantile(0.9)).astype(int)
        + (df["last_six_month_new_loan_no"] > df["last_six_month_new_loan_no"].quantile(0.9)).astype(int)
        + (df["total_overdue_no"] > 2).astype(int)
        + ((df["idcard_flag"] == 0) | (df["mobileno_flag"] == 0)).astype(int)
    )
    return (score >= 2).astype(int)


def train_fraud_model(cfg: ProjectConfig, df: pd.DataFrame) -> dict:
    """训练 LSTM 与 Transformer 两个欺诈检测模型并对比，将 F1 更优者存为生产模型。"""
    work = df.copy()
    y = _build_fraud_label(work)
    X = work.drop(columns=["loan_default"], errors="ignore")
    X = add_features(X).replace([np.inf, -np.inf], np.nan)
    # 派生特征计算完毕后再删原始标签列，防止 features_v3 内部依赖这些列
    X = X.drop(columns=_FRAUD_LABEL_COLS, errors="ignore")
    X_num = X.select_dtypes(include=[np.number])
    feature_cols = X_num.columns.tolist()

    X_tr_df, X_val_df, y_tr, y_val = train_test_split(
        X_num, y, test_size=0.2, random_state=42, stratify=y
    )
    X_3d_tr, scaler, fps = _prep_fraud_sequences(X_tr_df)
    X_3d_val, _, _      = _prep_fraud_sequences(X_val_df, scaler=scaler)

    pos = y_tr.sum()
    neg = len(y_tr) - pos
    pos_weight = float(neg / max(pos, 1))

    print(f"[Fraud] 样本: {len(y_tr)} train / {len(y_val)} val  "
          f"正例权重: {pos_weight:.1f}  序列形状: ({_FRAUD_N_STEPS}, {fps})")

    comparison = {}

    lstm_metrics, lstm_path = _train_deep_model(
        X_3d_tr, y_tr.values, X_3d_val, y_val.values,
        _build_lstm_model(_FRAUD_N_STEPS, fps), "lstm", cfg, pos_weight,
    )
    comparison["lstm"] = {**lstm_metrics, "model_path": lstm_path}

    tf_metrics, tf_path = _train_deep_model(
        X_3d_tr, y_tr.values, X_3d_val, y_val.values,
        _build_transformer_model(_FRAUD_N_STEPS, fps), "transformer", cfg, pos_weight,
    )
    comparison["transformer"] = {**tf_metrics, "model_path": tf_path}

    winner = "transformer" if tf_metrics["f1"] >= lstm_metrics["f1"] else "lstm"
    winner_metrics  = comparison[winner]
    winner_path     = winner_metrics["model_path"]
    print(f"[Fraud] 胜出: {winner.upper()}  F1={winner_metrics['f1']:.4f}")

    artifact = {
        "type":             "fraud_deep",
        "winner":           winner,
        "model_path":       winner_path,
        "scaler":           scaler,
        "feature_cols":     feature_cols,
        "n_steps":          _FRAUD_N_STEPS,
        "features_per_step": fps,
        "metrics":          {k: v for k, v in winner_metrics.items() if k != "model_path"},
        "comparison":       comparison,
        "model_type":       f"TensorFlow {winner.capitalize()}",
    }
    out = cfg.artifacts_dir / "fraud_model.joblib"
    joblib.dump(artifact, out)
    return {"artifact": str(out), "metrics": winner_metrics, "comparison": comparison}


def train_limit_model(cfg: ProjectConfig, df: pd.DataFrame) -> dict:
    work = df.copy()
    y = work["disbursed_amount"].astype(float)
    X_full = add_features(work.drop(columns=["loan_default"], errors="ignore")).replace([np.inf, -np.inf], np.nan)
    X = X_full.drop(columns=["disbursed_amount"], errors="ignore")
    fill_values = X.median(numeric_only=True).to_dict()
    X = X.fillna(fill_values).fillna(0)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "gbr": GradientBoostingRegressor(random_state=42),
        "xgb_reg": XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}
    best_name = None
    best_rmse = float("inf")
    best_model = None
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_valid)
        rmse = float(np.sqrt(mean_squared_error(y_valid, pred)))
        mae = float(mean_absolute_error(y_valid, pred))
        mape = float(np.mean(np.abs((y_valid - pred) / np.clip(np.abs(y_valid), 1e-6, None))))
        results[name] = {"rmse": rmse, "mae": mae, "mape": mape}
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_model = m

    artifact = {
        "model": best_model,
        "feature_cols": X.columns.tolist(),
        "best_model_name": best_name,
        "fill_values": fill_values,
        "comparison": results,
        "type": "limit_regressor",
    }
    out = cfg.artifacts_dir / "limit_model.joblib"
    joblib.dump(artifact, out)
    return {"artifact": str(out), "best_model": best_name, "comparison": results}


def score_from_probability(default_prob: np.ndarray) -> np.ndarray:
    p = np.clip(default_prob, 1e-6, 1 - 1e-6)
    odds = p / (1 - p)
    score = 600 - 50 * np.log(odds)
    return np.clip(score, 300, 850)


def run_decision_suite(cfg: ProjectConfig) -> Path:
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    df = _load_training_data(cfg)
    default_info = train_default_model(cfg, df)
    fraud_info = train_fraud_model(cfg, df)
    limit_info = train_limit_model(cfg, df)

    default_bundle = joblib.load(default_info["artifact"])
    model = default_bundle["model"]
    cols = default_bundle["feature_cols"]
    x = add_features(df.drop(columns=["loan_default"]).copy()).replace([np.inf, -np.inf], np.nan)[cols]
    p = model.predict_proba(x)[:, 1]
    score = score_from_probability(p)
    score_report = {
        "score_min": float(np.min(score)),
        "score_max": float(np.max(score)),
        "score_mean": float(np.mean(score)),
    }

    registry = {
        "default_model": default_info,
        "fraud_model": fraud_info,
        "limit_model": limit_info,
        "credit_score_summary": score_report,
    }
    out = cfg.artifacts_dir / "model_registry.json"
    out.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
