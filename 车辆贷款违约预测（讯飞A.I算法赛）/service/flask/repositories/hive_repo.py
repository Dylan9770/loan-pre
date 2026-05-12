from __future__ import annotations

from service.flask.config import Settings


def _connect():
    # 懒加载 pyhive，本地环境无 Hive 时不影响 Flask 启动
    from pyhive import hive
    return hive.Connection(
        host=Settings.HIVE_HOST,
        port=Settings.HIVE_PORT,
        username=Settings.HIVE_USERNAME,
        database=Settings.HIVE_DATABASE,
    )


def fetch_risk_daily_summary(limit: int = 30) -> list[dict]:
    try:
        conn = _connect()
    except Exception:
        return []   # Hive 不可达时返回空，stats.py 会回退到 mock 数据
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT dt, total_customers, avg_credit_score, default_rate, avg_disbursed_amount
            FROM risk_daily_summary
            ORDER BY dt DESC
            LIMIT {int(limit)}
            """
        )
        cols = [x[0] for x in cur.description]
        rows = cur.fetchall()
        return [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()
