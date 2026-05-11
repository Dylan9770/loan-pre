import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Predict with upgraded XGBoost model.")
    parser.add_argument(
        "--test-path",
        type=str,
        default="test.csv",
        help="Path to test CSV file.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="model_xgb.joblib",
        help="Trained model path.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="submission_xgb.csv",
        help="Prediction output CSV path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional manual threshold. If not set, uses best_threshold in model bundle.",
    )
    args = parser.parse_args()

    test_path = Path(args.test_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]
    best_threshold = float(model_bundle.get("best_threshold", 0.5))
    use_threshold = float(args.threshold) if args.threshold is not None else best_threshold

    test_df = pd.read_csv(test_path)
    missing_cols = [c for c in feature_cols if c not in test_df.columns]
    if missing_cols:
        raise ValueError(f"Test set missing feature columns: {missing_cols}")

    X_test = test_df[feature_cols].copy()
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    pred_proba = model.predict_proba(X_test)[:, 1]
    pred_label = (pred_proba >= use_threshold).astype(int)

    output = pd.DataFrame(
        {
            "customer_id": test_df["customer_id"] if "customer_id" in test_df.columns else range(len(test_df)),
            "loan_default_proba": pred_proba,
            "loan_default_pred": pred_label,
        }
    )
    output.to_csv(args.output_path, index=False)

    print("Prediction completed.")
    print(f"Threshold used: {use_threshold:.4f}")
    print(f"Output saved to: {args.output_path}")
    print(f"Rows: {len(output)}")
    print("111")


if __name__ == "__main__":
    main()
