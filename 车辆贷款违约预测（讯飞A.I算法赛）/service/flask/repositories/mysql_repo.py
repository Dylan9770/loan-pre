from __future__ import annotations

import bisect
import threading

import pymysql

from service.flask.config import Settings


_PEER_METRICS = (
    "credit_score",
    "total_overdue_no",
    "total_monthly_payment",
    "loan_to_asset_ratio",
)
_peer_cache: dict[str, list[float]] = {}
_peer_cache_lock = threading.Lock()


def load_peer_distributions() -> dict[str, list[float]]:
    """Load sorted distributions of key metrics from customer_features, lazy + cached.

    Returns a dict {metric_name: sorted_list_of_values}. Used by compute_peer_percentile.
    """
    if _peer_cache:
        return _peer_cache
    with _peer_cache_lock:
        if _peer_cache:  # double-check after acquiring lock
            return _peer_cache
        try:
            conn = _connect(Settings.MYSQL_DB_ODS)
        except Exception:
            return _peer_cache
        try:
            with conn.cursor() as cur:
                cols = ", ".join(_PEER_METRICS)
                cur.execute(
                    f"SELECT {cols} FROM customer_features "
                    f"WHERE credit_score IS NOT NULL"
                )
                rows = cur.fetchall() or []
            buckets: dict[str, list[float]] = {m: [] for m in _PEER_METRICS}
            for r in rows:
                for m in _PEER_METRICS:
                    v = r.get(m)
                    if v is None:
                        continue
                    try:
                        buckets[m].append(float(v))
                    except (TypeError, ValueError):
                        continue
            for m in _PEER_METRICS:
                buckets[m].sort()
                _peer_cache[m] = buckets[m]
            return _peer_cache
        except Exception:
            return _peer_cache
        finally:
            conn.close()


def compute_peer_percentile(value: float, metric: str) -> float | None:
    """Return percentile (0-100) of `value` against the cached distribution.

    Higher percentile = `value` is greater than that fraction of peers.
    Returns None if the metric is unknown or the distribution is empty.
    """
    dist = load_peer_distributions().get(metric)
    if not dist:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    idx = bisect.bisect_right(dist, v)
    return round(100.0 * idx / len(dist), 1)


def _connect(db_name: str):
    return pymysql.connect(
        host=Settings.MYSQL_HOST,
        port=Settings.MYSQL_PORT,
        user=Settings.MYSQL_USER,
        password=Settings.MYSQL_PASSWORD,
        database=db_name,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def insert_realtime_decision(row: dict) -> None:
    conn = _connect(Settings.MYSQL_DB_RT)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO realtime_decisions
                (customer_id, default_probability, default_pred, fraud_probability, fraud_pred, predicted_limit, credit_score)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    row.get("customer_id"),
                    row.get("default_probability"),
                    row.get("default_pred"),
                    row.get("fraud_probability"),
                    row.get("fraud_pred"),
                    row.get("predicted_limit"),
                    row.get("credit_score"),
                ),
            )
    finally:
        conn.close()


def fetch_realtime_summary() -> dict:
    conn = _connect(Settings.MYSQL_DB_RT)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM realtime_events")
            events = cur.fetchone()["c"]
            cur.execute("SELECT COUNT(*) AS c FROM realtime_decisions")
            decisions = cur.fetchone()["c"]
            return {"realtime_events": int(events), "realtime_decisions": int(decisions)}
    finally:
        conn.close()


def fetch_dashboard_overview() -> dict | None:
    """Compute real KPI numbers from customer_profile + loan_fact.

    Returns: total_customers, total_amount(元), overdue_rate(0~1), defaulted_customers.
    """
    try:
        conn = _connect(Settings.MYSQL_DB_ODS)
    except Exception:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total_customers,
                    SUM(CASE WHEN loan_default = 1 THEN 1 ELSE 0 END) AS defaulted_customers,
                    AVG(CASE WHEN loan_default = 1 THEN 1 ELSE 0 END) AS overdue_rate
                FROM customer_profile
                """
            )
            cp = cur.fetchone() or {}
            cur.execute("SELECT COALESCE(SUM(disbursed_amount), 0) AS total_amount FROM loan_fact")
            lf = cur.fetchone() or {}
        return {
            "total_customers": int(cp.get("total_customers") or 0),
            "defaulted_customers": int(cp.get("defaulted_customers") or 0),
            "overdue_rate": round(float(cp.get("overdue_rate") or 0), 4),
            "total_amount": float(lf.get("total_amount") or 0),
        }
    except Exception:
        return None
    finally:
        conn.close()


def fetch_cluster_samples(limit: int = 500) -> list[dict]:
    """Sample real (credit_score, disbursed_amount, loan_default) tuples for the cluster chart."""
    try:
        conn = _connect(Settings.MYSQL_DB_ODS)
    except Exception:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT c.credit_score, l.disbursed_amount, l.loan_default
                FROM customer_profile c
                JOIN loan_fact l ON c.customer_id = l.customer_id
                WHERE c.credit_score > 0 AND l.disbursed_amount > 0
                ORDER BY c.customer_id
                LIMIT {int(limit)}
                """
            )
            return [dict(r) for r in cur.fetchall() or []]
    except Exception:
        return []
    finally:
        conn.close()


def fetch_recent_real_customers(limit: int = 10) -> list[dict]:
    """Return a batch of real customer records with FULL feature set (for online scoring)."""
    try:
        conn = _connect(Settings.MYSQL_DB_ODS)
    except Exception:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT * FROM customer_features
                WHERE credit_score > 0 AND disbursed_amount > 0
                ORDER BY ingested_at DESC, customer_id DESC
                LIMIT {int(limit)}
                """
            )
            return [dict(r) for r in cur.fetchall() or []]
    except Exception:
        return []
    finally:
        conn.close()


def fetch_customer_loan_facts(customer_id: int) -> dict | None:
    """Fetch loan_fact row for a customer (most recent disbursed_date)."""
    try:
        conn = _connect(Settings.MYSQL_DB_ODS)
    except Exception:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT customer_id, disbursed_date, disbursed_amount, asset_cost,
                       ltv_ratio, total_disbursed_loan, total_sanction_loan,
                       total_monthly_payment, total_overdue_no, total_outstanding_loan,
                       outstanding_disburse_ratio, loan_default, area_id
                FROM loan_fact
                WHERE customer_id = %s
                ORDER BY disbursed_date DESC
                LIMIT 1
                """,
                (customer_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    except Exception:
        return None
    finally:
        conn.close()


def fetch_customer_credit_signals(customer_id: int) -> dict | None:
    """Fetch raw credit-activity columns from customer_features (CSV-mirrored)."""
    try:
        conn = _connect(Settings.MYSQL_DB_ODS)
    except Exception:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT customer_id,
                       last_six_month_new_loan_no, last_six_month_defaulted_no,
                       enquirie_no, credit_history,
                       credit_score, total_overdue_no,
                       total_monthly_payment, total_disbursed_loan,
                       total_sanction_loan, total_outstanding_loan,
                       total_account_loan_no, total_inactive_loan_no,
                       outstanding_disburse_ratio, loan_to_asset_ratio,
                       asset_cost, disbursed_amount
                FROM customer_features
                WHERE customer_id = %s
                LIMIT 1
                """,
                (customer_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    except Exception:
        return None
    finally:
        conn.close()


def fetch_recent_decisions(limit: int = 10) -> list[dict]:
    """Fetch latest realtime decisions, newest first."""
    try:
        conn = _connect(Settings.MYSQL_DB_RT)
    except Exception:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT customer_id, default_probability, credit_score,
                       predicted_limit, fraud_probability, created_at
                FROM realtime_decisions
                ORDER BY created_at DESC
                LIMIT {int(limit)}
                """
            )
            rows = cur.fetchall() or []
            return [dict(r) for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


def fetch_random_customer_id() -> int | None:
    """Pick a random customer_id that actually exists in customer_features."""
    try:
        conn = _connect(Settings.MYSQL_DB_ODS)
    except Exception:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM customer_features")
            total = int((cur.fetchone() or {}).get("c") or 0)
            if total <= 0:
                return None
            import random
            offset = random.randint(0, total - 1)
            cur.execute(f"SELECT customer_id FROM customer_features LIMIT 1 OFFSET {offset}")
            row = cur.fetchone()
            return int(row["customer_id"]) if row else None
    except Exception:
        return None
    finally:
        conn.close()


def fetch_customer_profile(customer_id: int) -> dict | None:
    """Fetch full customer feature row from customer_features (for model scoring + UI)."""
    try:
        conn = _connect(Settings.MYSQL_DB_ODS)
    except Exception:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM customer_features WHERE customer_id = %s LIMIT 1",
                (customer_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    except Exception:
        return None
    finally:
        conn.close()


def fetch_customer_similar(customer_id: int, k: int = 5) -> list[dict] | None:
    """Return Top-K similar customers via in-memory KNN engine (cosine similarity over 8 features).

    Falls back to the old single-dimension SQL lookup only if the engine is unavailable.
    """
    from service.flask.repositories import similarity_engine
    try:
        res = similarity_engine.query_similar(customer_id, k=k)
        if res is not None:
            return res
    except Exception:
        pass

    # 引擎不可用时的兜底（保留旧的"按信用分差距排序"逻辑）
    conn = _connect(Settings.MYSQL_DB_ODS)
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT customer_id, credit_score, disbursed_amount,
                       total_overdue_no, loan_default AS actual_default
                FROM customer_profile c
                LEFT JOIN loan_fact l ON c.customer_id = l.customer_id
                WHERE c.customer_id != %s
                ORDER BY ABS(credit_score - (
                    SELECT credit_score FROM customer_profile WHERE customer_id = %s
                )) ASC
                LIMIT {int(k)}
            """, (customer_id, customer_id))
            rows = cur.fetchall()
            if rows:
                result = []
                for r in rows:
                    rd = dict(r)
                    rd["actual_performance"] = "正常还款" if not rd.get("actual_default") else "部分违约"
                    rd["similarity"] = 0.90
                    result.append(rd)
                return result
            return None
    except Exception:
        return None
    finally:
        conn.close()


def fetch_area_risk_summary() -> list[dict]:
    """Fetch area-level risk summary."""
    conn = _connect(Settings.MYSQL_DB_ODS)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    c.area_id AS area_id,
                    COUNT(*) AS customer_count,
                    SUM(IF(l.loan_default=1,1,0)) AS default_count,
                    AVG(c.credit_score) AS avg_credit_score,
                    SUM(l.disbursed_amount) AS total_amount
                FROM customer_profile c
                LEFT JOIN loan_fact l ON c.customer_id = l.customer_id
                GROUP BY c.area_id
                ORDER BY default_count / COUNT(*) DESC
                LIMIT 20
            """)
            rows = cur.fetchall()
            area_names = {
                1: '华东区', 2: '华东区', 3: '华北区', 4: '华北区',
                5: '华南区', 6: '华南区', 7: '华中区', 8: '华中区',
                9: '华西区', 10: '西北区',
            }
            result = []
            for r in rows:
                rd = dict(r)
                rate = rd["default_count"] / rd["customer_count"] if rd["customer_count"] else 0
                area_label = area_names.get(rd["area_id"], f"地区{rd['area_id']}")
                result.append({
                    "area": area_label,
                    "rate": round(rate, 4),
                    "customers": int(rd["customer_count"]),
                    "defaults": int(rd["default_count"]),
                    "avg_score": round(float(rd["avg_credit_score"] or 0), 1),
                    "total_amount": round(float(rd["total_amount"] or 0), 2),
                })
            return result
    except Exception:
        return []
    finally:
        conn.close()
