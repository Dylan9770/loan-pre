"""离线计算 SHAP，给 dashboard 模型解释页用。

输出：
  artifacts/shap_global.json   - 全局 mean|SHAP| 排序，前端 SHAP 条形图 + 详情列表
  artifacts/shap_samples.json  - 3 个典型客户（低/中/高违约概率）的 waterfall
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from features_v3 import add_features  # noqa: E402

ART_DIR = ROOT / "artifacts"
SAMPLE_N = 3000          # 计算 SHAP 的样本数（覆盖全局已足够）
TOP_K = 12               # 全局重要性返回 top-K
WATERFALL_TOP_K = 8      # 每个 waterfall 显示的特征数

# 中文显示名映射（覆盖常见 60+ 个特征；其它走原列名）
DISPLAY_MAP: dict[str, str] = {
    "composite_risk_score": "复合风险评分",
    "composite_risk_level": "风险等级",
    "credit_score_sq": "信用评分平方",
    "area_id": "地区编码",
    "branch_id": "网点编码",
    "supplier_id": "供应商编码",
    "manufacturer_id": "厂商编码",
    "credit_score": "信用评分",
    "total_overdue_no": "总逾期次数",
    "outstanding_disburse_ratio": "未偿发放比",
    "ltv_ratio": "贷款资产比",
    "overdue_rate_total": "总逾期率",
    "credit_history": "信用记录时长",
    "enquirie_no": "征信查询次数",
    "disbursed_amount": "贷款金额",
    "age": "年龄",
    "total_monthly_payment": "月供金额",
    "employment_type": "工作类型",
    "asset_cost": "资产价值",
    "loan_to_asset_ratio": "贷款/资产比",
    "Credit_level": "信用等级",
    "active_account_ratio": "活跃账户比",
    "main_account_active_loan_no": "主账户活跃贷款数",
    "main_account_overdue_no": "主账户逾期次数",
    "sub_account_loan_no": "副账户贷款数",
    "sub_account_overdue_no": "副账户逾期次数",
    "last_six_month_new_loan_no": "近6月新增贷款",
    "last_six_month_defaulted_no": "近6月违约次数",
    "credit_utilization": "信用利用率",
    "payment_burden": "月供负担",
    "debt_pressure": "债务压力",
    "inquiry_intensity": "查询强度",
    "inquiry_frequency": "查询频率",
    "recent_stress": "近期压力指数",
    "new_loan_velocity": "新贷款增速",
    "main_overdue_density": "主账户逾期密度",
    "active_to_inactive_act_ratio": "活/非活账户比",
    "outstanding_disburse_ratio": "未偿/发放比",
    "disburse_to_sactioned_ratio": "发放/批准比",
    "total_outstanding_loan": "未偿贷款总额",
    "total_sanction_loan": "批准贷款总额",
    "total_disbursed_loan": "已发放贷款总额",
    "credit_gap": "信用缺口",
    "monthly_payment_ratio": "月供占比",
    "loan_per_age": "贷款/年龄",
    "branch_id_freq": "网点频率",
    "area_id_freq": "地区频率",
    "supplier_id_freq": "供应商频率",
    "average_age": "平均账龄",
    "main_account_tenure": "主账户存续期",
    "sub_account_tenure": "副账户存续期",
}


def _load_data() -> pd.DataFrame:
    for candidate in [
        ART_DIR.parent / "data_lake" / "featured" / "train_repaired.csv",
        ART_DIR.parent / "data_lake" / "featured" / "train_featured.csv",
        ART_DIR.parent / "data_lake" / "cleaned" / "train_cleaned.csv",
    ]:
        if candidate.exists():
            print(f"[Data] Loading {candidate.relative_to(ROOT)}")
            return pd.read_csv(candidate)
    raise FileNotFoundError("No training data found in data_lake/")


def _prepare_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """与 model_loader._prepare_features 完全一致的预处理。"""
    feat = add_features(df).replace([np.inf, -np.inf], np.nan)
    return feat.reindex(columns=feature_cols).fillna(0)


def main() -> None:
    bundle = joblib.load(ART_DIR / "default_model.joblib")
    if bundle.get("model_type") != "xgboost":
        raise RuntimeError(
            f"SHAP precompute 当前只支持 xgboost 胜出模型，当前 winner={bundle.get('winner')}"
        )
    model = bundle["model"]
    feature_cols: list[str] = bundle["feature_cols"]
    print(f"[Model] XGB regressor loaded, {len(feature_cols)} features")

    df = _load_data()
    if len(df) > SAMPLE_N:
        df = df.sample(SAMPLE_N, random_state=42).reset_index(drop=True)
    X = _prepare_X(df, feature_cols)
    print(f"[Data] Sampled {len(X)} rows for SHAP computation")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    base_value = float(np.array(explainer.expected_value).flatten()[0])
    print(f"[SHAP] Computed values shape={shap_values.shape}, base={base_value:.2f}")

    # 全局重要性：mean(|SHAP|) 排序
    mean_abs = np.abs(shap_values).mean(axis=0)
    sign     = np.sign(shap_values.mean(axis=0))    # 正/负向
    order = np.argsort(-mean_abs)[:TOP_K]
    global_importance = []
    for idx in order:
        col = feature_cols[idx]
        global_importance.append({
            "name":          col,
            "display":       DISPLAY_MAP.get(col, col),
            "mean_abs_shap": float(mean_abs[idx]),
            "impact":        "正向" if sign[idx] >= 0 else "负向",
            "description":   f"该特征对评分的平均影响幅度为 {mean_abs[idx]:.2f} 分，"
                             f"整体方向{'升高' if sign[idx] < 0 else '降低'}信用评分（即{'降低' if sign[idx] < 0 else '升高'}违约风险）。",
        })

    out_global = ART_DIR / "shap_global.json"
    out_global.write_text(json.dumps(global_importance, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"[Out] {out_global.relative_to(ROOT)} ({len(global_importance)} features)")

    # 单客户 waterfall：按 P(default) 选低/中/高三个样本
    # XGB regressor 输出的是评分 [300,850]；评分越低 → P(违约) 越高
    preds = np.clip(model.predict(X), 300, 850)
    p_default = 1.0 / (1.0 + 2 ** ((preds - 600) / 50))
    samples_idx = [
        int(np.argmin(np.abs(p_default - q))) for q in [0.10, 0.40, 0.75]
    ]

    samples = []
    for i, idx in enumerate(samples_idx):
        row_shap = shap_values[idx]
        # 取本行 |SHAP| 最大的 K 个特征作为 waterfall 项
        local_order = np.argsort(-np.abs(row_shap))[:WATERFALL_TOP_K]
        items = []
        for fi in local_order:
            items.append({
                "name":    feature_cols[fi],
                "display": DISPLAY_MAP.get(feature_cols[fi], feature_cols[fi]),
                "value":   float(row_shap[fi]),
            })
        # 其它特征汇总成"其他因素"
        other_sum = float(row_shap.sum() - sum(it["value"] for it in items))
        items.append({"name": "_other", "display": "其他因素", "value": other_sum})

        samples.append({
            "label":         ["低风险样本", "中风险样本", "高风险样本"][i],
            "credit_score":  round(float(preds[idx]), 1),
            "p_default":     round(float(p_default[idx]), 4),
            "base_value":    round(base_value, 2),
            "items":         items,
        })

    out_samples = ART_DIR / "shap_samples.json"
    out_samples.write_text(json.dumps(samples, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"[Out] {out_samples.relative_to(ROOT)} ({len(samples)} samples)")
    print("Done.")


if __name__ == "__main__":
    main()
