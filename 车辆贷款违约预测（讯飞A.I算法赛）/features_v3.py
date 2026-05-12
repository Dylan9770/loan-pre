"""
Enhanced Feature Engineering Module (features_v3.py)

Provides comprehensive feature engineering for loan data, including:
- Time-based feature extraction from disbursed_date
- Categorical feature encoding (frequency, ordinal, one-hot)
- Customer stability features
- Risk behavior ratio features
- Log-transformed heavy-tail variables
- All features from features_v2.py (backward compatible)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a: pd.Series, b: pd.Series, default: float = 0.0) -> pd.Series:
    """Safe division avoiding divide-by-zero and NaN propagation."""
    return a / (b.replace(0, np.nan) + 1e-6)


# ============================================================
# Part 1: Ratio & Risk Behavior Features (from features_v2)
# ============================================================

def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio and risk behavior features from original features_v2."""
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)

    out["ltv_ratio"] = _safe_div(out["disbursed_amount"], out["asset_cost"])
    out["loan_asset_gap"] = out["asset_cost"] - out["disbursed_amount"]

    # Overdue rate features
    out["overdue_rate_total"] = _safe_div(out["total_overdue_no"], out["total_account_loan_no"] + 1)
    out["overdue_rate_main"] = _safe_div(
        out["main_account_overdue_no"], out["main_account_loan_no"] + 1
    )
    out["overdue_rate_sub"] = _safe_div(
        out["sub_account_overdue_no"], out["sub_account_loan_no"] + 1
    )

    # Recent default rate
    out["recent_default_rate"] = _safe_div(
        out["last_six_month_defaulted_no"], out["last_six_month_new_loan_no"] + 1
    )

    # Activity ratios
    out["active_ratio"] = _safe_div(
        out["main_account_active_loan_no"] + out["sub_account_active_loan_no"],
        out["total_account_loan_no"] + 1,
    )
    out["inactive_ratio"] = _safe_div(
        out["total_inactive_loan_no"], out["total_account_loan_no"] + 1
    )
    out["monthly_payment_ratio"] = _safe_div(
        out["total_monthly_payment"], out["total_disbursed_loan"] + 1
    )

    # Per-age features
    out["credit_per_age"] = _safe_div(out["credit_history"], out["age"] + 1)
    out["enquiry_per_age"] = _safe_div(out["enquirie_no"], out["age"] + 1)

    # Gap features
    out["sanction_disburse_gap"] = out["total_sanction_loan"] - out["total_disbursed_loan"]

    return out


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform heavy-tail numeric variables to reduce skewness."""
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)

    log_cols = [
        "disbursed_amount",
        "asset_cost",
        "total_outstanding_loan",
        "total_disbursed_loan",
        "total_sanction_loan",
        "main_account_outstanding_loan",
        "sub_account_outstanding_loan",
    ]

    for col in log_cols:
        if col in out.columns:
            out[f"log1p_{col}"] = np.log1p(np.clip(out[col], a_min=0, a_max=None))

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# ============================================================
# Part 2: Time-based Feature Extraction
# ============================================================

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from disbursed_date.

    Assumes disbursed_date is an integer in format YYYYMMDD or YYYYMM,
    or a string/datetime. Handles common formats gracefully.
    """
    out = df.copy()

    date_col = "disbursed_date"

    if date_col not in out.columns:
        return out

    # Parse date column
    def parse_year(val):
        try:
            if pd.isna(val):
                return np.nan
            s = str(int(val))
            if len(s) >= 4:
                return int(s[:4])
            return np.nan
        except Exception:
            return np.nan

    def parse_month(val):
        try:
            if pd.isna(val):
                return np.nan
            s = str(int(val))
            if len(s) >= 6:
                return int(s[4:6])
            return np.nan
        except Exception:
            return np.nan

    def parse_day(val):
        try:
            if pd.isna(val):
                return np.nan
            s = str(int(val))
            if len(s) >= 8:
                return int(s[6:8])
            return np.nan
        except Exception:
            return np.nan

    out["loan_year"] = out[date_col].apply(parse_year)
    out["loan_month"] = out[date_col].apply(parse_month)
    out["loan_day"] = out[date_col].apply(parse_day)

    # Year as numeric (centered)
    ref_year = 2020
    out["loan_year_centered"] = out["loan_year"] - ref_year

    # Quarter
    out["loan_quarter"] = ((out["loan_month"] - 1) // 3 + 1).replace({0: np.nan})

    # Is beginning/end of month
    out["loan_day_of_month_norm"] = out["loan_day"] / 31.0

    # Season
    def get_season(month):
        if pd.isna(month):
            return np.nan
        if month in [3, 4, 5]:
            return 1  # Spring
        if month in [6, 7, 8]:
            return 2  # Summer
        if month in [9, 10, 11]:
            return 3  # Autumn
        return 4  # Winter

    out["loan_season"] = out["loan_month"].apply(get_season)

    # Customer tenure from birth year (if available)
    if "year_of_birth" in out.columns:
        out["customer_age_at_loan"] = out["loan_year"] - out["year_of_birth"]

    # Credit history depth
    if "credit_history" in out.columns:
        out["credit_history_depth_score"] = out["credit_history"] * 10

    return out


# ============================================================
# Part 3: Categorical Feature Encoding
# ============================================================

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features with three strategies:
    - Frequency encoding for high-cardinality (area_id, branch_id, etc.)
    - Ordinal encoding for ordered categories (Credit_level)
    - One-hot encoding for unordered categories (employment_type)
    """
    out = df.copy()

    # Frequency encoding for high-cardinality categoricals
    freq_encode_cols = ["area_id", "branch_id", "supplier_id", "manufacturer_id", "employee_code_id"]
    for col in freq_encode_cols:
        if col in out.columns:
            freq = out[col].value_counts(normalize=True)
            out[f"{col}_freq"] = out[col].map(freq).fillna(0)

    # Ordinal encoding for Credit_level
    if "Credit_level" in out.columns:
        credit_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        out["Credit_level_ordinal"] = out["Credit_level"].map(credit_map).fillna(2)

    # One-hot encoding for employment_type
    if "employment_type" in out.columns:
        dummies = pd.get_dummies(out["employment_type"], prefix="emp_type")
        out = pd.concat([out, dummies], axis=1)

    # Employment type with ordinal: self-employed < salaried < business owner
    if "employment_type" in out.columns:
        emp_map = {0: 0, 1: 1, 2: 2}
        out["employment_type_ordinal"] = out["employment_type"].map(emp_map).fillna(0)

    # Flag combination features
    if all(c in out.columns for c in ["mobileno_flag", "idcard_flag", "Driving_flag", "passport_flag"]):
        flag_sum = out["mobileno_flag"].fillna(0) + out["idcard_flag"].fillna(0) + \
                   out["Driving_flag"].fillna(0) + out["passport_flag"].fillna(0)
        out["id_verification_count"] = flag_sum
        out["id_verification_complete"] = (flag_sum >= 3).astype(int)

    return out


# ============================================================
# Part 4: Customer Stability Features
# ============================================================

def add_stability_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features that measure customer stability:
    - Employment stability (based on tenure and type)
    - Residence stability (based on account tenure)
    - Credit history depth
    - Inquiry frequency
    """
    out = df.copy()

    # Employment stability
    if "employment_type" in out.columns:
        out["emp_stability_score"] = out["employment_type"].apply(
            lambda x: 1.0 if x == 1 else (0.5 if x == 2 else 0.3)
        )

    # Account tenure stability
    if "main_account_tenure" in out.columns:
        out["tenure_stability"] = _safe_div(out["main_account_tenure"], out["age"])
        out["long_tenure"] = (out["main_account_tenure"] >= 24).astype(int)

    # Credit history depth
    if "credit_history" in out.columns:
        out["credit_depth_normalized"] = _safe_div(out["credit_history"], out["age"])

    # Inquiry frequency (recent queries per month of history)
    if "enquirie_no" in out.columns and "credit_history" in out.columns:
        out["inquiry_frequency"] = _safe_div(
            out["enquirie_no"], out["credit_history"] + 1
        )
        out["high_inquiry"] = (out["enquirie_no"] > 5).astype(int)

    # Active account ratio (main + sub)
    if all(c in out.columns for c in ["main_account_active_loan_no", "sub_account_active_loan_no", "total_account_loan_no"]):
        out["active_account_ratio"] = _safe_div(
            out["main_account_active_loan_no"] + out["sub_account_active_loan_no"],
            out["total_account_loan_no"]
        )

    return out


# ============================================================
# Part 5: Composite Risk Score
# ============================================================

def add_composite_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a composite risk score combining multiple risk factors.
    Score range: 0-100 (higher = riskier)
    """
    out = df.copy()

    risk_components = []

    # Overdue factor
    if "total_overdue_no" in out.columns:
        overdue_score = out["total_overdue_no"] * 10
        risk_components.append(("overdue", overdue_score.clip(0, 30)))

    # Recent default factor
    if "recent_default_rate" in out.columns:
        risk_components.append(("recent_default", (out["recent_default_rate"] * 100).clip(0, 25)))

    # LTV factor
    if "ltv_ratio" in out.columns:
        ltv_score = ((out["ltv_ratio"] - 0.5) * 50).clip(0, 25)
        risk_components.append(("ltv", ltv_score))

    # Inquiry factor
    if "enquiry_per_age" in out.columns:
        inquiry_score = (out["enquiry_per_age"] * 50).clip(0, 20)
        risk_components.append(("inquiry", inquiry_score))

    # Compute composite
    if risk_components:
        out["composite_risk_score"] = sum(v for _, v in risk_components)
        out["composite_risk_score"] = out["composite_risk_score"].clip(0, 100)
        out["composite_risk_level"] = pd.cut(
            out["composite_risk_score"],
            bins=[-1, 20, 40, 60, 100],
            labels=[0, 1, 2, 3],
        ).astype(float).fillna(1)

    return out


# ============================================================
# Part 6: Interaction & Polynomial Features
# ============================================================

def add_plan_features(df: pd.DataFrame) -> pd.DataFrame:
    """方案要求的补充特征：has_credit_score / has_credit_level / recent_stress 等。"""
    out = df.copy()

    if "credit_score" in out.columns:
        out["has_credit_score"] = (out["credit_score"] > 0).astype(int)

    if "Credit_level" in out.columns:
        out["has_credit_level"] = (out["Credit_level"] > 0).astype(int)

    for col in ["last_six_month_new_loan_no", "last_six_month_defaulted_no"]:
        if col not in out.columns:
            out[col] = 0
    out["recent_stress"] = out["last_six_month_new_loan_no"] + out["last_six_month_defaulted_no"]

    if "enquirie_no" in out.columns and "credit_history" in out.columns:
        out["inquiry_intensity"] = _safe_div(out["enquirie_no"], out["credit_history"] + 1)

    if "disbursed_amount" in out.columns and "age" in out.columns:
        out["loan_per_age"] = _safe_div(out["disbursed_amount"], out["age"] + 1)

    if "total_sanction_loan" in out.columns and "total_outstanding_loan" in out.columns:
        out["credit_gap"] = out["total_sanction_loan"] - out["total_outstanding_loan"]

    if all(c in out.columns for c in ["mobileno_flag", "idcard_flag", "Driving_flag", "passport_flag"]):
        out["identity_score"] = (
            out["mobileno_flag"].fillna(0) + out["idcard_flag"].fillna(0)
            + out["Driving_flag"].fillna(0) + out["passport_flag"].fillna(0)
        )

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """高阶交叉特征：credit_score × 风险指标、账户压力指标等。"""
    out = df.copy()

    # credit_score 多项式
    if "credit_score" in out.columns:
        cs = out["credit_score"]
        out["credit_score_sq"]   = cs ** 2
        out["credit_score_sqrt"] = np.sqrt(cs.clip(0))
        out["credit_score_log"]  = np.log1p(cs.clip(0))

    # credit_score × 逾期率交叉
    if "credit_score" in out.columns and "overdue_rate_total" in out.columns:
        out["cs_x_overdue"] = out["credit_score"] * out["overdue_rate_total"]

    if "credit_score" in out.columns and "recent_default_rate" in out.columns:
        out["cs_x_recent_def"] = out["credit_score"] * out["recent_default_rate"]

    if "credit_score" in out.columns and "ltv_ratio" in out.columns:
        out["cs_x_ltv"] = out["credit_score"] * out["ltv_ratio"]

    # 账户负债压力：未偿 / 发放 × 月供
    if all(c in out.columns for c in ["total_outstanding_loan", "total_disbursed_loan"]):
        out["debt_pressure"] = _safe_div(out["total_outstanding_loan"], out["total_disbursed_loan"])

    if all(c in out.columns for c in ["total_monthly_payment", "disbursed_amount"]):
        out["payment_burden"] = _safe_div(out["total_monthly_payment"], out["disbursed_amount"])

    # 主账户质量：逾期/账户数 × 活跃比
    if all(c in out.columns for c in ["main_account_overdue_no", "main_account_loan_no"]):
        out["main_overdue_density"] = _safe_div(
            out["main_account_overdue_no"], out["main_account_loan_no"]
        )

    # 近6个月新增贷款 / 历史信用账户
    if all(c in out.columns for c in ["last_six_month_new_loan_no", "total_account_loan_no"]):
        out["new_loan_velocity"] = _safe_div(
            out["last_six_month_new_loan_no"], out["total_account_loan_no"]
        )

    # 信用利用率：outstanding / sanction
    if all(c in out.columns for c in ["total_outstanding_loan", "total_sanction_loan"]):
        out["credit_utilization"] = _safe_div(
            out["total_outstanding_loan"], out["total_sanction_loan"]
        )

    # 主/子账户逾期差异
    if all(c in out.columns for c in ["main_account_overdue_no", "sub_account_overdue_no"]):
        out["overdue_imbalance"] = out["main_account_overdue_no"] - out["sub_account_overdue_no"]

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# ============================================================
# Master Feature Function
# ============================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in sequence.
    This is the main entry point for feature generation.

    Steps:
    1. Ratio & risk behavior features (from features_v2)
    2. Log-transform heavy-tail variables
    3. Time-based features
    4. Categorical encoding
    5. Stability features
    6. Composite risk score

    Returns a DataFrame with all original and engineered features.
    """
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    # Step 1: Ratio features
    df = add_ratio_features(df)

    # Step 2: Log features
    df = add_log_features(df)

    # Step 3: Time features
    df = extract_time_features(df)

    # Step 4: Categorical encoding
    df = encode_categorical_features(df)

    # Step 5: Stability features
    df = add_stability_features(df)

    # Step 6: Composite risk
    df = add_composite_risk_score(df)

    # Step 7: Plan-specific supplement features
    df = add_plan_features(df)

    # Step 8: Interaction & polynomial features
    df = add_interaction_features(df)

    # Final cleanup
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to categorical dtype for sklearn compatibility."""
    out = df.copy()
    cat_cols = [
        "branch_id",
        "supplier_id",
        "manufacturer_id",
        "area_id",
        "employee_code_id",
        "mobileno_flag",
        "idcard_flag",
        "Driving_flag",
        "passport_flag",
        "Credit_level",
        "employment_type",
    ]
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].astype("Int64").astype("category")
    return out
