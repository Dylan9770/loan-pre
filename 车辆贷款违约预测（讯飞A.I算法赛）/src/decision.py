from __future__ import annotations

import json
import os
from pathlib import Path

# 必须在 TF / PyTorch 任何一个导入前设置，避免两个框架争抢 CUDA 上下文导致 segfault
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor

from features_v3 import add_features
from src.config import ProjectConfig

# ---------------------------------------------------------------------------
# 回归任务：排除列（共线性 / 无信息 / 标签泄漏）
# ---------------------------------------------------------------------------
_REGRESSION_EXCLUDE_COLS = [
    "customer_id", "disbursed_date", "loan_default",
    "credit_score",               # 50% 缺失 + 会主导模型
    "year_of_birth",              # 与 age 完全共线 r=-1.0
    "main_account_loan_no",       # r=0.992 with total_account_loan_no
    "main_account_outstanding_loan",  # r=0.987 with total_outstanding_loan
    "main_account_sanction_loan", # r=0.998 with total_sanction_loan
    "main_account_disbursed_loan",# r=0.998 with total_disbursed_loan
    "main_account_monthly_payment",   # r=0.996 with total_monthly_payment
    "manufacturer_id",            # 与违约率差异仅 0.3%
    "employee_code_id",           # 员工编号，非客户风险信号
]

# ---------------------------------------------------------------------------
# BiLSTM 语义分组（9步 × 6特征 = 54维）
# 每步按业务含义聚合特征，赋予 BiLSTM 序列方向上的语义意义
# ---------------------------------------------------------------------------
_BILSTM_GROUPS = [
    # Step 1: 信用档案
    ["credit_history", "Credit_level", "average_age",
     "has_credit_score", "has_credit_level", "credit_depth_normalized"],
    # Step 2: 主账户状态
    ["main_account_active_loan_no", "main_account_overdue_no",
     "main_account_inactive_loan_no", "main_account_tenure",
     "total_account_loan_no", "active_account_ratio"],
    # Step 3: 子账户状态
    ["sub_account_loan_no", "sub_account_active_loan_no",
     "sub_account_overdue_no", "sub_account_inactive_loan_no",
     "sub_account_tenure", "sub_account_monthly_payment"],
    # Step 4: 违约逾期史
    ["total_overdue_no", "last_six_month_defaulted_no",
     "overdue_rate_total", "recent_stress",
     "inquiry_intensity", "main_overdue_density"],
    # Step 5: 近期信用行为
    ["last_six_month_new_loan_no", "enquirie_no",
     "active_ratio", "active_to_inactive_act_ratio",
     "new_loan_velocity", "inquiry_frequency"],
    # Step 6: 总体财务状况
    ["total_outstanding_loan", "total_sanction_loan",
     "total_disbursed_loan", "total_monthly_payment",
     "credit_gap", "credit_utilization"],
    # Step 7: 财务比率
    ["loan_to_asset_ratio", "outstanding_disburse_ratio",
     "disburse_to_sactioned_ratio", "payment_burden",
     "debt_pressure", "monthly_payment_ratio"],
    # Step 8: 本次贷款与资产
    ["disbursed_amount", "asset_cost",
     "loan_per_age", "ltv_ratio",
     "loan_asset_gap", "sanction_disburse_gap"],
    # Step 9: 人口统计与机构
    ["age", "employment_type", "identity_score",
     "branch_id_freq", "area_id_freq", "supplier_id_freq"],
]
_BILSTM_N_STEPS = len(_BILSTM_GROUPS)          # 9
_BILSTM_FPS     = len(_BILSTM_GROUPS[0])        # 6

# ---------------------------------------------------------------------------
# 欺诈检测：剔除列（放宽版）
# 设计原则：保留原始 5 列标签构成特征（它们本身就是真实风险指标），
#           仅剔除"明显是标签公式产物"的合成列，避免 1:1 数据泄漏
# ---------------------------------------------------------------------------
_FRAUD_LABEL_COLS = [
    # 显式合成列（直接由标签公式聚合而成，保留会导致 100% 泄漏）
    "composite_risk_score", "composite_risk_level",
    # 身份核验综合（= 标签构成项 mobileno_flag + idcard_flag 的和）
    "identity_score",
    # ID 列
    "customer_id",
]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _load_training_data(cfg: ProjectConfig) -> pd.DataFrame:
    repaired = cfg.featured_dir / "train_repaired.csv"
    if repaired.exists():
        return pd.read_csv(repaired)
    fallback = cfg.cleaned_dir / "train_cleaned.csv"
    if fallback.exists():
        return pd.read_csv(fallback)
    raise FileNotFoundError("No cleaned/repaired training data found.")


def _clean_features(X: pd.DataFrame) -> pd.DataFrame:
    const_cols = [c for c in X.columns if X[c].nunique() <= 1]
    rep_cols   = [c for c in X.columns if c.startswith("repaired_")]
    return X.drop(columns=const_cols + rep_cols, errors="ignore")


def score_from_probability(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    odds = p / (1 - p)
    score = 600 - 50 * np.log2(odds)
    return np.clip(score, 300, 850)


def _clip_outliers(df: pd.DataFrame, cols: list[str], q_lo=0.01, q_hi=0.99) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        lo = out[col].replace([np.inf, -np.inf], np.nan).quantile(q_lo)
        hi = out[col].replace([np.inf, -np.inf], np.nan).quantile(q_hi)
        out[col] = out[col].replace([np.inf, -np.inf], np.nan).clip(lo, hi)
    return out


def _regression_composite_score(metrics_dict: dict[str, dict]) -> dict[str, float]:
    """在 R²/RMSE/MAE/MSE 上对多个模型归一化打分，返回每个模型的综合得分。"""
    names = list(metrics_dict.keys())
    r2s   = np.array([metrics_dict[n]["r2"]   for n in names])
    rmses = np.array([metrics_dict[n]["rmse"]  for n in names])
    maes  = np.array([metrics_dict[n]["mae"]   for n in names])
    mses  = np.array([metrics_dict[n]["mse"]   for n in names])

    def norm_higher(arr):
        rng = arr.max() - arr.min()
        return (arr - arr.min()) / (rng + 1e-9)

    def norm_lower(arr):
        rng = arr.max() - arr.min()
        return 1 - (arr - arr.min()) / (rng + 1e-9)

    scores = (
        0.40 * norm_higher(r2s)
        + 0.25 * norm_lower(rmses)
        + 0.15 * norm_lower(maes)
        + 0.10 * norm_lower(mses)
        + 0.10 * norm_higher(r2s)   # KS 暂用 R² 代替（无真实 KS）
    )
    return {n: float(scores[i]) for i, n in enumerate(names)}


def _fraud_composite_score(metrics_dict: dict[str, dict]) -> dict[str, float]:
    """在 Recall/PR-AUC/F1/ROC-AUC 上对多个模型归一化打分。"""
    names    = list(metrics_dict.keys())
    recalls  = np.array([metrics_dict[n]["recall"]  for n in names])
    pr_aucs  = np.array([metrics_dict[n]["pr_auc"]  for n in names])
    f1s      = np.array([metrics_dict[n]["f1"]      for n in names])
    roc_aucs = np.array([metrics_dict[n]["roc_auc"] for n in names])

    def norm(arr):
        rng = arr.max() - arr.min()
        return (arr - arr.min()) / (rng + 1e-9)

    scores = (
        0.40 * norm(recalls)
        + 0.30 * norm(pr_aucs)
        + 0.20 * norm(f1s)
        + 0.10 * norm(roc_aucs)
    )
    return {n: float(scores[i]) for i, n in enumerate(names)}


# ---------------------------------------------------------------------------
# 任务一：信用评分预测（回归）
# ---------------------------------------------------------------------------

def _generate_credit_score_labels(X: pd.DataFrame, y_binary: pd.Series,
                                   n_splits: int = 5) -> np.ndarray:
    """
    两阶段评分卡：
      1. StratifiedKFold OOF XGBClassifier → P(违约)
      2. score = 600 - 50 × log₂(P/(1-P))，clip 到 [300, 850]
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(y_binary))
    pos = y_binary.sum()
    neg = len(y_binary) - pos
    spw = float(neg / max(pos, 1))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_binary), 1):
        clf = XGBClassifier(
            objective="binary:logistic", n_estimators=800,
            learning_rate=0.03, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=spw,
            eval_metric="auc", tree_method="hist",
            random_state=42, n_jobs=-1,
            early_stopping_rounds=50,
        )
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_binary.iloc[tr_idx], y_binary.iloc[val_idx]
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_proba[val_idx] = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, oof_proba[val_idx])
        print(f"  [ScoreLabel] Fold {fold}/{n_splits}  AUC={auc:.4f}")

    return score_from_probability(oof_proba)


def _prepare_regression_features(df: pd.DataFrame) -> pd.DataFrame:
    """特征工程 + 去除排除列 + 极端值处理。"""
    X = add_features(df.drop(columns=["loan_default"], errors="ignore"))
    X = X.replace([np.inf, -np.inf], np.nan)
    X = _clean_features(X)
    X = X.drop(columns=_REGRESSION_EXCLUDE_COLS, errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X = _clip_outliers(X, ["outstanding_disburse_ratio",
                            "disburse_to_sactioned_ratio",
                            "active_to_inactive_act_ratio"])
    X = X.fillna(X.median())
    return X


def _build_bilstm_sequences(X_scaled: np.ndarray,
                             feature_cols: list[str]) -> np.ndarray:
    """
    将标准化后的 2D 特征矩阵按语义分组 reshape 为
    (n_samples, N_STEPS, FPS) 的 3D 张量。
    每步按 _BILSTM_GROUPS 定义取对应列，缺失列填 0。
    """
    col_index = {c: i for i, c in enumerate(feature_cols)}
    out = np.zeros((len(X_scaled), _BILSTM_N_STEPS, _BILSTM_FPS), dtype=np.float32)
    for step_i, group in enumerate(_BILSTM_GROUPS):
        for feat_j, feat in enumerate(group):
            if feat in col_index:
                out[:, step_i, feat_j] = X_scaled[:, col_index[feat]]
    return out


def _build_bilstm_model() -> tf.keras.Model:
    # 轻量版：2核CPU可训练，参数量约为原版 1/4
    inp = tf.keras.Input(shape=(_BILSTM_N_STEPS, _BILSTM_FPS), name="bilstm_input")
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.2)
    )(inp)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(16, return_sequences=False, dropout=0.2)
    )(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1, activation="linear", name="score_output")(x)
    return tf.keras.Model(inp, out, name="bilstm_regressor")


def _build_mlp_model(input_dim: int) -> tf.keras.Model:
    # 轻量版：3层替代4层，单元数减半
    inp = tf.keras.Input(shape=(input_dim,), name="mlp_input")
    x = tf.keras.layers.Dense(128, activation="relu")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1, activation="linear", name="score_output")(x)
    return tf.keras.Model(inp, out, name="mlp_regressor")


def _eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r2   = float(r2_score(y_true, y_pred))
    mse  = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae  = float(mean_absolute_error(y_true, y_pred))
    return {"r2": r2, "rmse": rmse, "mae": mae, "mse": mse}


def train_credit_score_model(cfg: ProjectConfig, df: pd.DataFrame) -> dict:
    """
    信用评分预测（回归）：
      1. 两阶段评分卡生成连续标签 y_score ∈ [300, 850]
      2. 训练 XGBoost Regressor / BiLSTM / MLP 三个回归模型
      3. 按 R²/RMSE/MAE/MSE 加权综合得分择优，保存最优模型
    """
    # 最大化 CPU 线程利用率（2核）
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)

    print("\n" + "=" * 60)
    print("  任务一：信用评分预测（回归）")
    print("=" * 60)

    X = _prepare_regression_features(df)
    y_binary = df["loan_default"].astype(int)
    feature_cols = X.columns.tolist()

    print(f"特征维度: {X.shape[1]} 列，样本数: {len(X)}")
    score_label_cache = cfg.artifacts_dir / "ckpt_score_labels.npy"
    if score_label_cache.exists():
        y_score = np.load(str(score_label_cache))
        print(f"评分标签已有缓存，跳过 OOF 生成  "
              f"min={y_score.min():.1f}  max={y_score.max():.1f}  mean={y_score.mean():.1f}")
    else:
        print("生成两阶段评分卡标签（5-Fold OOF）...")
        y_score = _generate_credit_score_labels(X, y_binary)
        np.save(str(score_label_cache), y_score)
        print(f"评分分布: min={y_score.min():.1f}  max={y_score.max():.1f}  "
              f"mean={y_score.mean():.1f}  std={y_score.std():.1f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_score, test_size=0.2, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)
    X_test_sc= scaler.transform(X_test)

    comparison: dict[str, dict] = {}
    trained_models: dict[str, object] = {}

    # 检查点路径（各模型训练完后立即写入，重启可跳过已完成的模型）
    ckpt_xgb    = cfg.artifacts_dir / "ckpt_reg_xgb.joblib"
    ckpt_bilstm = cfg.artifacts_dir / "ckpt_reg_bilstm.joblib"
    ckpt_mlp    = cfg.artifacts_dir / "ckpt_reg_mlp.joblib"

    # ---- XGBoost Regressor ----
    if ckpt_xgb.exists():
        ck = joblib.load(ckpt_xgb)
        m_xgb, pred_xgb, xgb_reg = ck["metrics"], np.array(ck["predictions"]), ck["model"]
        comparison["xgboost_regressor"] = m_xgb
        trained_models["xgboost_regressor"] = xgb_reg
        print(f"\n[Reg] XGBoost Regressor 已有检查点，跳过训练  "
              f"R²={m_xgb['r2']:.4f}  RMSE={m_xgb['rmse']:.2f}")
    else:
        print("\n[Reg] 训练 XGBoost Regressor ...")
        xgb_reg = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=2000,
            learning_rate=0.02,
            max_depth=6,
            min_child_weight=10,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=2.0,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=100,
        )
        xgb_reg.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        pred_xgb = np.clip(xgb_reg.predict(X_test), 300, 850)
        m_xgb = _eval_regression(y_test, pred_xgb)
        comparison["xgboost_regressor"] = m_xgb
        trained_models["xgboost_regressor"] = xgb_reg
        joblib.dump({"metrics": m_xgb, "predictions": pred_xgb.tolist(), "model": xgb_reg}, ckpt_xgb)
        print(f"  XGB  R²={m_xgb['r2']:.4f}  RMSE={m_xgb['rmse']:.2f}  "
              f"MAE={m_xgb['mae']:.2f}  MSE={m_xgb['mse']:.2f}")

    _keras_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]

    # ---- BiLSTM ----
    bilstm_path = cfg.artifacts_dir / "reg_bilstm.keras"
    if ckpt_bilstm.exists():
        ck = joblib.load(ckpt_bilstm)
        m_bilstm, pred_bilstm = ck["metrics"], np.array(ck["predictions"])
        comparison["bilstm"] = m_bilstm
        trained_models["bilstm"] = str(bilstm_path)
        print(f"\n[Reg] BiLSTM 已有检查点，跳过训练  "
              f"R²={m_bilstm['r2']:.4f}  RMSE={m_bilstm['rmse']:.2f}")
    else:
        print("\n[Reg] 训练 BiLSTM ...")
        X_tr_3d   = _build_bilstm_sequences(X_tr_sc,   feature_cols)
        X_val_3d  = _build_bilstm_sequences(X_val_sc,  feature_cols)
        X_test_3d = _build_bilstm_sequences(X_test_sc, feature_cols)

        bilstm = _build_bilstm_model()
        bilstm.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
        bilstm.fit(
            X_tr_3d, y_tr,
            validation_data=(X_val_3d, y_val),
            epochs=80, batch_size=2048,
            callbacks=_keras_callbacks, verbose=0,
        )
        pred_bilstm = np.clip(bilstm.predict(X_test_3d, batch_size=2048, verbose=0).ravel(), 300, 850)
        m_bilstm = _eval_regression(y_test, pred_bilstm)
        comparison["bilstm"] = m_bilstm
        bilstm.save(str(bilstm_path))
        trained_models["bilstm"] = str(bilstm_path)
        joblib.dump({"metrics": m_bilstm, "predictions": pred_bilstm.tolist()}, ckpt_bilstm)
        print(f"  BiLSTM  R²={m_bilstm['r2']:.4f}  RMSE={m_bilstm['rmse']:.2f}  "
              f"MAE={m_bilstm['mae']:.2f}  MSE={m_bilstm['mse']:.2f}")

    # ---- MLP ----
    mlp_path = cfg.artifacts_dir / "reg_mlp.keras"
    if ckpt_mlp.exists():
        ck = joblib.load(ckpt_mlp)
        m_mlp, pred_mlp = ck["metrics"], np.array(ck["predictions"])
        comparison["mlp"] = m_mlp
        trained_models["mlp"] = str(mlp_path)
        print(f"\n[Reg] MLP 已有检查点，跳过训练  "
              f"R²={m_mlp['r2']:.4f}  RMSE={m_mlp['rmse']:.2f}")
    else:
        # 对标签做 Z-score 归一化，避免 MSE 梯度过大导致训练不稳定
        y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
        y_tr_norm  = (y_tr  - y_mean) / y_std
        y_val_norm = (y_val - y_mean) / y_std

        print("\n[Reg] 训练 MLP ...")
        mlp = _build_mlp_model(X_tr_sc.shape[1])
        mlp.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse", metrics=["mae"],
        )
        mlp.fit(
            X_tr_sc, y_tr_norm,
            validation_data=(X_val_sc, y_val_norm),
            epochs=80, batch_size=2048,
            callbacks=_keras_callbacks, verbose=0,
        )
        pred_mlp_norm = mlp.predict(X_test_sc, batch_size=2048, verbose=0).ravel()
        pred_mlp = np.clip(pred_mlp_norm * y_std + y_mean, 300, 850)
        m_mlp = _eval_regression(y_test, pred_mlp)
        comparison["mlp"] = m_mlp
        mlp.save(str(mlp_path))
        trained_models["mlp"] = str(mlp_path)
        joblib.dump({"metrics": m_mlp, "predictions": pred_mlp.tolist()}, ckpt_mlp)
        print(f"  MLP  R²={m_mlp['r2']:.4f}  RMSE={m_mlp['rmse']:.2f}  "
              f"MAE={m_mlp['mae']:.2f}  MSE={m_mlp['mse']:.2f}")

    # ---- 择优 ----
    comp_scores = _regression_composite_score(comparison)
    winner = max(comp_scores, key=comp_scores.get)
    print(f"\n[Reg] 综合得分: {comp_scores}")
    print(f"[Reg] 胜出模型: {winner.upper()}")

    artifact: dict = {
        "type":          "credit_score_regression",
        "winner":        winner,
        "feature_cols":  feature_cols,
        "scaler":        scaler,
        "bilstm_groups": _BILSTM_GROUPS,
        "n_steps":       _BILSTM_N_STEPS,
        "fps":           _BILSTM_FPS,
        "metrics":       comparison[winner],
        "comparison":    comparison,
        "comp_scores":   comp_scores,
    }
    if winner == "xgboost_regressor":
        artifact["model"] = xgb_reg
        artifact["model_type"] = "xgboost"
    elif winner == "bilstm":
        artifact["model_path"] = str(bilstm_path)
        artifact["model_type"] = "bilstm"
    else:
        artifact["model_path"] = str(mlp_path)
        artifact["model_type"] = "mlp"

    out = cfg.artifacts_dir / "default_model.joblib"
    joblib.dump(artifact, out)
    return {"artifact": str(out), "winner": winner, "metrics": comparison[winner],
            "comparison": comparison}


# ---------------------------------------------------------------------------
# 任务二：欺诈检测（分类）
# ---------------------------------------------------------------------------

def _build_fraud_label(df: pd.DataFrame) -> pd.Series:
    """
    加权规则合成欺诈标签（composite ≥ 0.30 → 正例）。
    说明：本标签为规则合成，非真实人工审计标注。
    """
    enq_p90   = df["enquirie_no"].quantile(0.90)
    loan_p85  = df["last_six_month_new_loan_no"].quantile(0.85)
    ratio_p95 = df["loan_to_asset_ratio"].quantile(0.95)

    composite = (
        (df["enquirie_no"] > enq_p90).astype(float)              * 0.30
        + (df["last_six_month_new_loan_no"] > loan_p85).astype(float) * 0.25
        + ((df["idcard_flag"] == 0) | (df["mobileno_flag"] == 0)).astype(float) * 0.25
        + (df["total_overdue_no"] > 2).astype(float)              * 0.10
        + (df["loan_to_asset_ratio"] > ratio_p95).astype(float)   * 0.10
    )
    return (composite >= 0.30).astype(int)


def _train_tabnet(X_tr: np.ndarray, y_tr: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray):
    """TabNet — 专为表格数据设计的深度学习分类器，用稀疏注意力逐步选择特征。"""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # 避免 TF/PyTorch 双框架 CUDA 上下文冲突
    from pytorch_tabnet.tab_model import TabNetClassifier
    clf = TabNetClassifier(
        n_d=32, n_a=32,
        n_steps=5,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        momentum=0.02,
        seed=42,
        device_name="cpu",
        verbose=10,
    )
    clf.fit(
        X_tr.astype(np.float32), y_tr.astype(int),
        eval_set=[(X_val.astype(np.float32), y_val.astype(int))],
        eval_metric=["auc"],
        max_epochs=200,
        patience=20,
        batch_size=1024,
        virtual_batch_size=256,
        num_workers=0,
        weights=1,
        drop_last=False,
    )
    return clf


def _find_best_threshold_recall(y_true: np.ndarray, y_proba: np.ndarray,
                                 min_precision: float = 0.15) -> float:
    """在 Precision >= min_precision 约束下最大化 Recall。"""
    best_thr, best_recall = 0.5, 0.0
    for thr in np.linspace(0.05, 0.95, 181):
        y_pred = (y_proba >= thr).astype(int)
        if y_pred.sum() == 0:
            continue
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        if prec >= min_precision and rec > best_recall:
            best_recall, best_thr = rec, float(thr)
    return best_thr


def _find_best_threshold_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """最大化 F1 的阈值搜索——比 Recall-under-Precision 更稳定，适合小模型。"""
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.05, 0.95, 181):
        y_pred = (y_proba >= thr).astype(int)
        if y_pred.sum() == 0:
            continue
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def _eval_fraud(y_true: np.ndarray, y_proba: np.ndarray,
                threshold: float, name: str) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_proba)),
        "pr_auc":    float(average_precision_score(y_true, y_proba)),
        "threshold": threshold,
    }
    print(f"  [{name}] Recall={metrics['recall']:.4f}  Precision={metrics['precision']:.4f}  "
          f"F1={metrics['f1']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}  "
          f"PR-AUC={metrics['pr_auc']:.4f}  thr={threshold:.2f}")
    return metrics


def train_fraud_model(cfg: ProjectConfig, df: pd.DataFrame) -> dict:
    """
    欺诈检测（分类）：
      训练 TabNet / 决策树 / 随机森林 三个模型
      按 Recall/PR-AUC/F1/ROC-AUC 加权综合得分择优，保存最优模型
    """
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)

    print("\n" + "=" * 60)
    print("  任务二：欺诈检测（分类）")
    print("=" * 60)

    work = df.copy()
    y    = _build_fraud_label(work)

    pos = y.sum()
    print(f"欺诈标签: 正例={pos}({pos/len(y)*100:.1f}%)  负例={len(y)-pos}")

    X_raw = add_features(work.drop(columns=["loan_default"], errors="ignore"))
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
    X_raw = _clean_features(X_raw)
    X_raw = X_raw.drop(columns=_FRAUD_LABEL_COLS + _REGRESSION_EXCLUDE_COLS, errors="ignore")
    X_raw = X_raw.select_dtypes(include=[np.number])
    X_raw = _clip_outliers(X_raw, ["outstanding_disburse_ratio",
                                    "disburse_to_sactioned_ratio",
                                    "active_to_inactive_act_ratio"])
    X_raw = X_raw.fillna(X_raw.median())
    feature_cols = X_raw.columns.tolist()
    print(f"欺诈特征维度: {len(feature_cols)} 列")

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    # SMOTE（仅对训练集，1:60 → 1:5）
    pos_tr = y_tr.sum()
    neg_tr = len(y_tr) - pos_tr
    target_ratio = min(0.20, pos_tr / neg_tr * 5)
    smote = SMOTE(sampling_strategy=target_ratio, k_neighbors=5, random_state=42)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
    print(f"SMOTE 后: 正例={y_tr_sm.sum()}  负例={(y_tr_sm==0).sum()}")

    scaler = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr_sm)
    X_val_sc = scaler.transform(X_val)
    X_test_sc= scaler.transform(X_test)

    # FT 专用：QuantileTransformer（拟合原始 X_tr，不含 SMOTE）
    qt = QuantileTransformer(
        n_quantiles=min(len(X_tr), 1000), output_distribution="normal", random_state=42)
    X_tr_qt   = qt.fit_transform(X_tr)
    X_val_qt  = qt.transform(X_val)
    X_test_qt = qt.transform(X_test)
    # FT class_weight 基于原始不平衡比例，不依赖 SMOTE
    pos_orig = int(y_tr.sum())
    neg_orig = len(y_tr) - pos_orig
    ft_class_weight = {0: 1.0, 1: float(neg_orig / max(pos_orig, 1))}

    pos_w = float((y_tr_sm == 0).sum() / max(y_tr_sm.sum(), 1))
    comparison: dict[str, dict] = {}
    trained_models: dict       = {}

    ckpt_tabnet = cfg.artifacts_dir / "ckpt_fraud_tabnet.joblib"
    ckpt_dt     = cfg.artifacts_dir / "ckpt_fraud_dt.joblib"
    ckpt_rf     = cfg.artifacts_dir / "ckpt_fraud_rf.joblib"

    # ---- TabNet ----
    if ckpt_tabnet.exists():
        ck = joblib.load(ckpt_tabnet)
        comparison["tabnet"] = ck["metrics"]
        trained_models["tabnet"] = ck["model"]
        if "qt" in ck:
            qt = ck["qt"]
        print(f"\n[Fraud] TabNet 已有检查点，跳过训练  "
              f"Recall={ck['metrics']['recall']:.4f}  F1={ck['metrics']['f1']:.4f}")
    else:
        print("\n[Fraud] 训练 TabNet ...")
        tabnet = _train_tabnet(X_tr_qt, y_tr.values, X_val_qt, y_val.values)
        proba_tabnet_val  = tabnet.predict_proba(X_val_qt)[:, 1]
        proba_tabnet_test = tabnet.predict_proba(X_test_qt)[:, 1]
        thr_tabnet = _find_best_threshold_f1(y_val.values, proba_tabnet_val)
        m_tabnet = _eval_fraud(y_test.values, proba_tabnet_test, thr_tabnet, "TabNet")
        comparison["tabnet"] = m_tabnet
        trained_models["tabnet"] = tabnet
        joblib.dump({"metrics": m_tabnet, "qt": qt, "model": tabnet}, ckpt_tabnet)

    # ---- 决策树 ----
    if ckpt_dt.exists():
        ck = joblib.load(ckpt_dt)
        comparison["decision_tree"] = ck["metrics"]
        trained_models["decision_tree"] = ck["model"]
        print(f"\n[Fraud] 决策树 已有检查点，跳过训练  "
              f"Recall={ck['metrics']['recall']:.4f}  F1={ck['metrics']['f1']:.4f}")
    else:
        print("\n[Fraud] 训练 决策树 ...")
        dt_base = DecisionTreeClassifier(
            max_depth=20,
            min_samples_leaf=10,
            min_samples_split=20,
            class_weight="balanced",
            criterion="gini",
            random_state=42,
        )
        dt_base.fit(X_tr_sm, y_tr_sm)
        # Platt 校准：在真实 val 分布上校准概率，避免 SMOTE 引入的概率偏移
        # sklearn 1.6+ 移除了 cv="prefit"，改用 FrozenEstimator 包装已训练模型
        from sklearn.frozen import FrozenEstimator
        dt = CalibratedClassifierCV(FrozenEstimator(dt_base), method="sigmoid")
        dt.fit(X_val_sc, y_val.values)
        proba_dt_val  = dt.predict_proba(X_val_sc)[:, 1]
        proba_dt_test = dt.predict_proba(X_test_sc)[:, 1]
        thr_dt = _find_best_threshold_f1(y_val.values, proba_dt_val)
        m_dt = _eval_fraud(y_test.values, proba_dt_test, thr_dt, "决策树")
        comparison["decision_tree"] = m_dt
        trained_models["decision_tree"] = dt
        joblib.dump({"metrics": m_dt, "model": dt}, ckpt_dt)

    # ---- 随机森林 ----
    if ckpt_rf.exists():
        ck = joblib.load(ckpt_rf)
        comparison["random_forest"] = ck["metrics"]
        trained_models["random_forest"] = ck["model"]
        print(f"\n[Fraud] 随机森林 已有检查点，跳过训练  "
              f"Recall={ck['metrics']['recall']:.4f}  F1={ck['metrics']['f1']:.4f}")
    else:
        print("\n[Fraud] 训练 随机森林 ...")
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=20,
            min_samples_split=50,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            oob_score=True,
            random_state=42,
        )
        rf.fit(X_tr_sm, y_tr_sm)
        print(f"  RF OOB Score: {rf.oob_score_:.4f}")
        proba_rf_val  = rf.predict_proba(X_val_sc)[:, 1]
        proba_rf_test = rf.predict_proba(X_test_sc)[:, 1]
        thr_rf = _find_best_threshold_recall(y_val.values, proba_rf_val)
        m_rf = _eval_fraud(y_test.values, proba_rf_test, thr_rf, "随机森林")
        comparison["random_forest"] = m_rf
        trained_models["random_forest"] = rf
        joblib.dump({"metrics": m_rf, "model": rf}, ckpt_rf)

    # ---- 择优 ----
    comp_scores = _fraud_composite_score(comparison)
    winner = max(comp_scores, key=comp_scores.get)
    print(f"\n[Fraud] 综合得分: {comp_scores}")
    print(f"[Fraud] 胜出模型: {winner.upper()}")

    winner_metrics = comparison[winner]
    artifact: dict = {
        "type":          "fraud_classification",
        "winner":        winner,
        "feature_cols":  feature_cols,
        "scaler":        scaler,
        "threshold":     winner_metrics["threshold"],
        "metrics":       winner_metrics,
        "comparison":    comparison,
        "comp_scores":   comp_scores,
    }
    if winner == "tabnet":
        artifact["scaler"]     = qt   # TabNet 用 QuantileTransformer，覆盖默认 StandardScaler
        artifact["model"]      = trained_models["tabnet"]
        artifact["model_type"] = "tabnet"
    elif winner == "decision_tree":
        artifact["model"] = dt
        artifact["model_type"] = "decision_tree"
    else:
        artifact["model"] = rf
        artifact["model_type"] = "random_forest"

    out = cfg.artifacts_dir / "fraud_model.joblib"
    joblib.dump(artifact, out)
    return {"artifact": str(out), "winner": winner,
            "metrics": winner_metrics, "comparison": comparison}


# ---------------------------------------------------------------------------
# 额度预测（保持原有逻辑不变）
# ---------------------------------------------------------------------------

def train_limit_model(cfg: ProjectConfig, df: pd.DataFrame) -> dict:
    work = df.copy()
    y = work["disbursed_amount"].astype(float)
    X_full = add_features(work.drop(columns=["loan_default"], errors="ignore")).replace(
        [np.inf, -np.inf], np.nan)
    X = X_full.drop(columns=["disbursed_amount"], errors="ignore")
    fill_values = X.median(numeric_only=True).to_dict()
    X = X.fillna(fill_values).fillna(0)
    X = X.select_dtypes(include=[np.number])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "xgb_reg": XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.5,
            objective="reg:squarederror", tree_method="hist",
            random_state=42, n_jobs=-1,
        ),
        "gbr": GradientBoostingRegressor(random_state=42),
    }

    results, best_name, best_rmse, best_model = {}, None, float("inf"), None
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_valid)
        rmse = float(np.sqrt(mean_squared_error(y_valid, pred)))
        mae  = float(mean_absolute_error(y_valid, pred))
        results[name] = {"rmse": rmse, "mae": mae}
        if rmse < best_rmse:
            best_rmse, best_name, best_model = rmse, name, m
        print(f"  [Limit/{name}] RMSE={rmse:.2f}  MAE={mae:.2f}")

    print(f"[Limit] 胜出: {best_name}  RMSE={best_rmse:.2f}")
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


# ---------------------------------------------------------------------------
# 流水线入口
# ---------------------------------------------------------------------------

def run_decision_suite(cfg: ProjectConfig) -> Path:
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    df = _load_training_data(cfg)

    default_info = train_credit_score_model(cfg, df)
    fraud_info   = train_fraud_model(cfg, df)
    # 限额预测改为业务规则计算，无需训练 ML 模型
    # 公式: 合理额度 = (月收入估计 × 偿债比上限) / 月供单价 × (1 - P(违约)) × 欺诈惩罚
    # 详见 service/flask/model_loader.py:_calculate_credit_limit

    registry = {
        "credit_score_model": {
            "winner":     default_info["winner"],
            "metrics":    default_info["metrics"],
            "comparison": default_info["comparison"],
        },
        "fraud_model": {
            "winner":     fraud_info["winner"],
            "metrics":    fraud_info["metrics"],
            "comparison": fraud_info["comparison"],
        },
        "limit_model": {
            "type":   "rule_based",
            "formula": "(月收入×偿债比) / 月供单价 × (1 - P_default) × 欺诈惩罚",
        },
    }
    out = cfg.artifacts_dir / "model_registry.json"
    out.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n模型注册表已写入: {out}")
    return out
