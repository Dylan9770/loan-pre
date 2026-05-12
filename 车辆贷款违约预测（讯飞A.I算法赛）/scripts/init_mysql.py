#!/usr/bin/env python3
"""
一键初始化 MySQL：建库、建表、创建用户、导入 CSV 数据。
运行方式：
    python3 scripts/init_mysql.py --root-password 你的root密码
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root-password", default="", help="MySQL root 密码（留空表示无密码）")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=3306)
    p.add_argument("--csv", default=str(PROJECT_ROOT / "car_loan_train.csv"))
    return p.parse_args()


def run(args):
    try:
        import pymysql
    except ImportError:
        print("请先安装 pymysql：pip install pymysql")
        sys.exit(1)

    import pandas as pd

    # ── 1. 以 root 连接，建库建用户 ──────────────────────────────
    print("=== Step 1: 建库、建用户 ===")
    conn = pymysql.connect(
        host=args.host, port=args.port,
        user="root", password=args.root_password,
        charset="utf8mb4", autocommit=True,
    )
    with conn.cursor() as cur:
        cur.execute("CREATE DATABASE IF NOT EXISTS loan_ods DEFAULT CHARACTER SET utf8mb4;")
        cur.execute("CREATE DATABASE IF NOT EXISTS loan_rt  DEFAULT CHARACTER SET utf8mb4;")
        cur.execute(
            "CREATE USER IF NOT EXISTS 'loan_user'@'%' IDENTIFIED BY 'loan_pass_123';"
        )
        cur.execute("GRANT ALL PRIVILEGES ON loan_ods.* TO 'loan_user'@'%';")
        cur.execute("GRANT ALL PRIVILEGES ON loan_rt.*  TO 'loan_user'@'%';")
        cur.execute("FLUSH PRIVILEGES;")
    conn.close()
    print("  库和用户创建完毕")

    # ── 2. 执行建表 SQL ──────────────────────────────────────────
    print("=== Step 2: 建表 ===")
    sql_file = PROJECT_ROOT / "sql" / "mysql" / "create_tables.sql"
    if not sql_file.exists():
        print(f"  找不到 {sql_file}，跳过建表")
    else:
        conn = pymysql.connect(
            host=args.host, port=args.port,
            user="root", password=args.root_password,
            charset="utf8mb4", autocommit=True,
        )
        sql_text = sql_file.read_text(encoding="utf-8")
        # 按分号拆分，逐条执行
        statements = [s.strip() for s in sql_text.split(";") if s.strip()]
        with conn.cursor() as cur:
            for stmt in statements:
                try:
                    cur.execute(stmt)
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        print(f"  警告: {e}")
        conn.close()
        print("  建表完毕")

    # ── 3. 导入 CSV 数据 ─────────────────────────────────────────
    print(f"=== Step 3: 导入数据 ({args.csv}) ===")
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"  找不到 CSV 文件：{csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"  读取 {len(df):,} 行数据")

    conn = pymysql.connect(
        host=args.host, port=args.port,
        user="loan_user", password="loan_pass_123",
        database="loan_ods", charset="utf8mb4", autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
    )

    # 写入 customer_profile
    profile_cols = [
        "customer_id", "year_of_birth", "age", "employment_type",
        "credit_score", "Credit_level", "mobileno_flag", "idcard_flag",
        "Driving_flag", "passport_flag", "area_id", "loan_default",
    ]
    profile_cols = [c for c in profile_cols if c in df.columns]
    profile_df = df[profile_cols].copy()
    col_map = {"Credit_level": "credit_level", "Driving_flag": "driving_flag"}
    profile_df = profile_df.rename(columns=col_map)

    print("  写入 customer_profile ...")
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE customer_profile;")
        cols = profile_df.columns.tolist()
        placeholders = ", ".join(["%s"] * len(cols))
        col_names = ", ".join(cols)
        sql = f"INSERT INTO customer_profile ({col_names}) VALUES ({placeholders})"
        batch = 1000
        for i in range(0, len(profile_df), batch):
            rows = [tuple(r) for r in profile_df.iloc[i:i+batch].values.tolist()]
            cur.executemany(sql, rows)
            if (i // batch) % 10 == 0:
                print(f"    {min(i+batch, len(profile_df)):,} / {len(profile_df):,}")
    print(f"  customer_profile 写入完毕：{len(profile_df):,} 行")

    # 写入 loan_fact
    loan_cols = [
        "customer_id", "disbursed_date", "disbursed_amount", "asset_cost",
        "branch_id", "supplier_id", "manufacturer_id", "area_id",
        "employment_type", "credit_score", "loan_to_asset_ratio",
        "total_disbursed_loan", "total_monthly_payment",
        "total_overdue_no", "total_outstanding_loan", "loan_default",
        "year_of_birth",
    ]
    loan_cols = [c for c in loan_cols if c in df.columns]
    loan_df = df[loan_cols].copy()
    loan_col_map = {"loan_to_asset_ratio": "ltv_ratio"}
    loan_df = loan_df.rename(columns=loan_col_map)

    print("  写入 loan_fact ...")
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE loan_fact;")
        cols = loan_df.columns.tolist()
        placeholders = ", ".join(["%s"] * len(cols))
        col_names = ", ".join(cols)
        sql = f"INSERT INTO loan_fact ({col_names}) VALUES ({placeholders})"
        batch = 1000
        for i in range(0, len(loan_df), batch):
            rows = [tuple(
                None if (isinstance(v, float) and v != v) else v
                for v in r
            ) for r in loan_df.iloc[i:i+batch].values.tolist()]
            cur.executemany(sql, rows)
            if (i // batch) % 10 == 0:
                print(f"    {min(i+batch, len(loan_df)):,} / {len(loan_df):,}")
    print(f"  loan_fact 写入完毕：{len(loan_df):,} 行")

    conn.close()
    print()
    print("=== 全部完成！===")
    print("现在重启 Flask 服务，客户画像将从 MySQL 真实数据加载。")
    print("可查询的客户 ID 范围：搜索 CSV 中的 customer_id 列。")


if __name__ == "__main__":
    run(get_args())
