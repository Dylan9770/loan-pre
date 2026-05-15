#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_decision_suite.py — 完整训练流水线入口
==========================================
整合数据修复（repair）、特征工程（features_v3）、模型训练（XGBoost/RF/GBR）
并生成完整模型制品（default / fraud / limit）供实时服务使用。

Usage:
  # 完整流水线（使用 features_v3 增强特征）
  python run_decision_suite.py

  # 仅训练默认模型（轻量）
  python run_decision_suite.py --stages default

  # 跳过修复（使用已修复数据）
  python run_decision_suite.py --skip-repair

  # 指定特征工程版本
  python run_decision_suite.py --features v3   # 增强版（默认）
  python run_decision_suite.py --features v2   # 基础版

  # Spark 生产模式（从 Hive 读取已修复数据）
  python run_decision_suite.py --mode spark \
      --source-table loan_dwd.loan_repaired_als \
      --spark-master yarn
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.decision import (
    train_credit_score_model as train_default_model,
    train_fraud_model, train_limit_model,
    score_from_probability, run_decision_suite as _run_decision_suite,
)
from features_v3 import add_features as add_features_v3
from features_v2 import add_features as add_features_v2

FEATURE_VERSIONS = {"v2": add_features_v2, "v3": add_features_v3}


def _print_header(msg: str) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n  {msg}\n{sep}")


def run_local_pipeline(
    cfg,
    features_version: str = "v3",
    stages: list[str] | None = None,
    skip_repair: bool = False,
) -> dict:
    """
    本地模式流水线（使用本地 CSV 数据）。
    流程：加载数据 → 特征工程 → 训练 3 个模型 → 注册到制品库
    """
    _print_header(f"Local Pipeline (features_v{features_version})")
    stages = stages or ["default", "fraud"]   # limit 改为业务规则计算，无需训练

    # ---- 1. 加载数据 ----
    repaired_path = cfg.featured_dir / "train_repaired.csv"
    if not repaired_path.exists():
        fallback = cfg.cleaned_dir / "train_cleaned.csv"
        if not fallback.exists():
            raise FileNotFoundError(
                f"未找到训练数据。请先运行：\n"
                f"  python run_ingest_storage.py   # 数据摄入\n"
                f"  python run_repair_pipeline.py  # 数据修复"
            )
        repaired_path = fallback
        print(f"[INFO] 未找到已修复数据，使用清洁数据: {repaired_path}")

    print(f"[1/5] Loading data from: {repaired_path}")
    df = pd.read_csv(repaired_path)
    print(f"  Rows: {len(df):,}  Columns: {df.shape[1]}")

    # ---- 2. 特征工程 ----
    print(f"[2/5] Feature engineering (features_v{features_version})...")
    add_feat = FEATURE_VERSIONS[features_version]
    df_feat = add_feat(df).replace([np.inf, -np.inf], np.nan)
    feature_cols = df_feat.drop(columns=["loan_default"], errors="ignore").columns.tolist()
    print(f"  Total features: {len(feature_cols)}")

    results = {"feature_version": features_version, "stages": {}}

    # ---- 3. 训练模型 ----
    if "default" in stages:
        _print_header("Stage 1: Training Credit Score Model (XGBReg / BiLSTM / MLP)")
        t0 = time.time()
        default_result = train_default_model(cfg, df)
        m = default_result["metrics"]
        results["stages"]["default"] = {
            "artifact":    default_result["artifact"],
            "winner":      default_result["winner"],
            "metrics":     m,
            "elapsed_sec": round(time.time() - t0, 2),
        }
        print(f"  Winner={default_result['winner']}  "
              f"R²={m.get('r2', 0):.4f}  RMSE={m.get('rmse', 0):.2f}  "
              f"MAE={m.get('mae', 0):.2f}")

    if "fraud" in stages:
        _print_header("Stage 2: Training Fraud Model (TabNet / DT / RF)")
        t0 = time.time()
        fraud_result = train_fraud_model(cfg, df)
        m = fraud_result["metrics"]
        results["stages"]["fraud"] = {
            "artifact":    fraud_result["artifact"],
            "winner":      fraud_result["winner"],
            "metrics":     m,
            "elapsed_sec": round(time.time() - t0, 2),
        }
        print(f"  Winner={fraud_result['winner']}  "
              f"Recall={m.get('recall', 0):.4f}  F1={m.get('f1', 0):.4f}  "
              f"PR-AUC={m.get('pr_auc', 0):.4f}")

    if "limit" in stages:
        _print_header("Stage 3: Training Limit Model (XGBoost / RF / GBR ensemble)")
        t0 = time.time()
        limit_result = train_limit_model(cfg, df)
        best = limit_result["comparison"][limit_result["best_model"]]
        results["stages"]["limit"] = {
            "artifact": limit_result["artifact"],
            "best_model": limit_result["best_model"],
            "comparison": limit_result["comparison"],
            "elapsed_sec": round(time.time() - t0, 2),
        }
        print(f"  Best: {limit_result['best_model']}, "
              f"RMSE={best['rmse']:.2f}, MAE={best['mae']:.2f}")

    # ---- 4. 生成模型注册表 ----
    _print_header("Stage 4: Generating Model Registry")
    registry = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "feature_version": features_version,
        "total_features": len(feature_cols),
        "models": {},
    }
    for stage_name in stages:
        stage = results["stages"].get(stage_name, {})
        artifact_path = Path(stage.get("artifact", ""))
        registry["models"][stage_name] = {
            "artifact": artifact_path.name,
            "path": str(artifact_path),
            "metrics": stage.get("metrics", {}),
            "elapsed_sec": stage.get("elapsed_sec"),
        }

    registry_path = cfg.artifacts_dir / "model_registry.json"
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Registry saved: {registry_path}")
    print(json.dumps(registry, ensure_ascii=False, indent=2))

    # ---- 5. 打印摘要 ----
    _print_header("Pipeline Complete")
    total_elapsed = sum(s.get("elapsed_sec", 0) for s in results["stages"].values())
    print(f"  Feature version: v{features_version}")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Total elapsed: {total_elapsed:.1f}s")
    print(f"  Artifacts:")
    for stage_name, stage in results["stages"].items():
        print(f"    - {stage_name}: {Path(stage['artifact']).name}")
    print(f"  Registry: {registry_path}")

    return results


def run_spark_pipeline(
    cfg,
    source_table: str,
    spark_master: str,
    features_version: str = "v3",
    stages: list[str] | None = None,
) -> dict:
    """
    Spark 模式流水线（从 Hive 读取已修复数据）。
    流程：Spark 读取 Hive 表 → Pandas → 本地特征工程 → 训练模型 → 写回制品库
    """
    _print_header(f"Spark Pipeline (table={source_table}, features_v{features_version})")
    stages = stages or ["default"]

    try:
        from pyspark.sql import SparkSession
    except ImportError:
        raise ImportError("PySpark not installed. Install with: pip install pyspark[hadoop3]")

    spark = (
        SparkSession.builder
        .appName("loan_decision_suite_spark")
        .master(spark_master)
        .enableHiveSupport()
        .getOrCreate()
    )

    print(f"[1/3] Reading from Hive: {source_table}")
    sdf = spark.table(source_table)
    df = sdf.toPandas()
    spark.stop()
    print(f"  Loaded {len(df):,} rows from Hive")

    print(f"[2/3] Feature engineering...")
    add_feat = FEATURE_VERSIONS[features_version]
    df_feat = add_feat(df).replace([np.inf, -np.inf], np.nan)
    print(f"  Features: {len(df_feat.columns)}")

    results = {}
    if "default" in stages:
        print(f"[3/3] Training credit score model...")
        t0 = time.time()
        result = train_default_model(cfg, df)
        m = result["metrics"]
        results["default"] = {"winner": result["winner"], "metrics": m,
                               "elapsed_sec": round(time.time() - t0, 2)}
        print(f"  Winner={result['winner']}  R²={m.get('r2', 0):.4f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Loan Decision Suite — Full Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["local", "spark"],
        default="local",
        help="local=使用本地 CSV；spark=从 Hive 读取已修复数据",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["default", "fraud", "limit"],
        default=["default", "fraud"],   # limit 改为业务规则计算，无需训练
        help="要训练哪些模型（limit 已改为业务规则，无需训练）",
    )
    parser.add_argument(
        "--skip-repair",
        action="store_true",
        help="跳过修复步骤（使用已修复数据）",
    )
    parser.add_argument(
        "--features",
        choices=["v2", "v3"],
        default="v3",
        help="特征工程版本：v2=基础版（ltv_ratio/log1p），v3=增强版（+时序/稳定性/复合风险）",
    )
    parser.add_argument(
        "--source-table",
        default="loan_dwd.loan_repaired_als",
        help="Hive 源表（spark 模式）",
    )
    parser.add_argument(
        "--spark-master",
        default="yarn",
        help="Spark Master URL（spark 模式）",
    )
    args = parser.parse_args()

    cfg = get_config()
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "spark":
        run_spark_pipeline(cfg, args.source_table, args.spark_master, args.features, args.stages)
    else:
        run_local_pipeline(cfg, args.features, args.stages, args.skip_repair)


if __name__ == "__main__":
    main()
