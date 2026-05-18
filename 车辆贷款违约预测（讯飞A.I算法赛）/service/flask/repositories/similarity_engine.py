"""相似客户检索引擎 - KNN + 余弦相似度。

启动时一次性加载 customer_features 全表到内存，做 z-score 标准化 + L2 单位化。
查询时一次矩阵点积即可得到全表相似度，毫秒级返回 top-K。

特征选择原则：
  - 反映客户经济能力 + 信用行为 + 个体属性
  - 全部用连续值，避免类别字段稀释相似度
  - 8 维已经能区分出"职业近 + 年龄近 + 还款行为近"的客户
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np
import pymysql

from service.flask.config import Settings

logger = logging.getLogger(__name__)

# 用于相似度计算的特征列（全部数值）
_FEATURE_COLS = [
    "credit_score",
    "age",
    "credit_history",
    "total_overdue_no",
    "disbursed_amount",
    "total_outstanding_loan",
    "total_monthly_payment",
    "loan_to_asset_ratio",
]

# 这些字段业务上 0 == 缺失（不可能真的是 0），用中位数填补再做相似度
_TREAT_ZERO_AS_MISSING = {
    "credit_score", "credit_history", "disbursed_amount",
    "total_outstanding_loan", "total_monthly_payment",
}

# 额外返回给前端展示的字段（不参与相似度计算）
_DISPLAY_COLS = [
    "loan_default", "employment_type", "area_id",
]

_CACHE: dict = {
    "loaded":      False,
    "loading":     False,
    "ids":         None,   # np.ndarray int64 (N,)
    "id_to_idx":   None,   # dict customer_id -> row idx
    "matrix":      None,   # np.ndarray float32 (N, F) L2-normalized
    "raw_display": None,   # list[dict] for display fields
    "feature_mean": None,
    "feature_std":  None,
    "load_time_ms": 0,
    "row_count":   0,
}
_LOCK = threading.Lock()


def _connect():
    return pymysql.connect(
        host=Settings.MYSQL_HOST, port=Settings.MYSQL_PORT,
        user=Settings.MYSQL_USER, password=Settings.MYSQL_PASSWORD,
        database=Settings.MYSQL_DB_ODS, charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor, autocommit=True,
    )


def is_loaded() -> bool:
    return _CACHE["loaded"]


def load() -> bool:
    """从 MySQL 加载特征矩阵到内存。已加载则跳过。线程安全。"""
    if _CACHE["loaded"]:
        return True
    with _LOCK:
        if _CACHE["loaded"]:
            return True
        if _CACHE["loading"]:
            return False
        _CACHE["loading"] = True

    t0 = time.time()
    try:
        all_cols = ["customer_id"] + _FEATURE_COLS + _DISPLAY_COLS
        col_sql = ", ".join(f"`{c}`" for c in all_cols)
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT {col_sql} FROM customer_features")
                rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            logger.warning("[similarity] customer_features empty, skip")
            _CACHE["loading"] = False
            return False

        ids = np.fromiter(
            (int(r["customer_id"]) for r in rows), dtype=np.int64, count=len(rows),
        )
        X = np.full((len(rows), len(_FEATURE_COLS)), np.nan, dtype=np.float32)
        for i, r in enumerate(rows):
            for j, c in enumerate(_FEATURE_COLS):
                v = r.get(c)
                if v is None:
                    continue
                fv = float(v)
                # 业务上 0 == 缺失的字段，转 NaN，下面用中位数填补
                if fv == 0.0 and c in _TREAT_ZERO_AS_MISSING:
                    continue
                X[i, j] = fv

        # 中位数填补缺失值（替代之前的"用 0 填"，避免缺失值扎堆产生伪相似）
        for j in range(len(_FEATURE_COLS)):
            col = X[:, j]
            mask = np.isnan(col)
            if mask.any():
                med = np.nanmedian(col)
                if np.isnan(med):
                    med = 0.0
                col[mask] = med
                X[:, j] = col

        # z-score 标准化
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-8] = 1.0
        X_norm = ((X - mean) / std).astype(np.float32)

        raw_display = [
            {
                "customer_id":             int(r["customer_id"]),
                "credit_score":            float(r.get("credit_score") or 0),
                "disbursed_amount":        float(r.get("disbursed_amount") or 0),
                "total_overdue_no":        int(r.get("total_overdue_no") or 0),
                "age":                     int(r.get("age") or 0),
                "employment_type":         int(r.get("employment_type") or 0),
                "area_id":                 int(r.get("area_id") or 0),
                "loan_default":            int(r.get("loan_default") or 0),
            }
            for r in rows
        ]

        # 估算"典型距离尺度"用于把欧氏距离映射到 [0,1] 相似度
        # 取所有随机对距离的 5% 分位数作为 exp 衰减常数，能让 Top-K 的相似度
        # 落在 [0.4, 0.99] 区间，既不全是 0.99 也不会过分稀释
        rng = np.random.default_rng(42)
        sample_n = min(5000, len(rows))
        a_idx = rng.integers(0, len(rows), size=sample_n)
        b_idx = rng.integers(0, len(rows), size=sample_n)
        sample_dist = np.linalg.norm(X_norm[a_idx] - X_norm[b_idx], axis=1)
        valid = sample_dist[sample_dist > 0]
        dist_scale = float(np.percentile(valid, 5)) if len(valid) else 1.0
        if dist_scale < 1e-6:
            dist_scale = 1.0

        # 只把"信用分非缺失"的客户作为可返回的相似候选（避免一排 credit_score=0）。
        # 但所有客户都保留在矩阵中，确保任何客户都能作为查询目标。
        valid_mask = np.array(
            [(r.get("credit_score") or 0) > 0 for r in rows], dtype=bool,
        )

        with _LOCK:
            _CACHE.update({
                "loaded":       True,
                "loading":      False,
                "ids":          ids,
                "id_to_idx":    {int(c): i for i, c in enumerate(ids)},
                "matrix":       X_norm,
                "raw_display":  raw_display,
                "valid_mask":   valid_mask,
                "feature_mean": mean,
                "feature_std":  std,
                "dist_scale":   dist_scale,
                "load_time_ms": int((time.time() - t0) * 1000),
                "row_count":    len(rows),
            })
        logger.info(
            "[similarity] loaded %d rows (%d valid candidates), %d features, %d ms, dist_scale=%.3f",
            len(rows), int(valid_mask.sum()), len(_FEATURE_COLS),
            _CACHE["load_time_ms"], dist_scale,
        )
        return True
    except Exception as exc:
        logger.exception("[similarity] load failed: %s", exc)
        _CACHE["loading"] = False
        return False


def query_similar(customer_id: int, k: int = 5) -> list[dict] | None:
    """返回与目标客户最相似的 Top-K（按余弦相似度降序）。

    None  → 引擎未加载或 customer_id 不在库中
    list  → 真实 KNN 结果，含 similarity ∈ [0, 1]
    """
    if not _CACHE["loaded"]:
        load()
    if not _CACHE["loaded"]:
        return None

    idx = _CACHE["id_to_idx"].get(int(customer_id))
    if idx is None:
        return None

    matrix = _CACHE["matrix"]
    target = matrix[idx]                  # (F,)
    # 欧氏距离向量化：每行减目标向量，按行求 L2 范数
    diffs = matrix - target               # (N, F)
    dists = np.linalg.norm(diffs, axis=1) # (N,)
    dists[idx] = np.inf                   # 排除自身
    # 只在"有信用分"的客户里找相似，避免返回一排 credit_score=0
    valid_mask = _CACHE.get("valid_mask")
    if valid_mask is not None:
        dists[~valid_mask] = np.inf

    k = max(1, min(int(k), len(dists) - 1))
    top_idx = np.argpartition(dists, k)[:k]
    top_idx = top_idx[np.argsort(dists[top_idx])]

    # 距离 → 相似度：exp(-d/scale)，scale 取全样本距离中位数，让结果落在 [0.1, 1]
    scale = _CACHE.get("dist_scale", 1.0)
    results = []
    for ti in top_idx:
        d = _CACHE["raw_display"][int(ti)]
        dist = float(dists[int(ti)])
        sim_display = round(float(np.exp(-dist / max(scale, 1e-6))), 4)
        results.append({
            "customer_id":        d["customer_id"],
            "credit_score":       d["credit_score"],
            "disbursed_amount":   d["disbursed_amount"],
            "total_overdue_no":   d["total_overdue_no"],
            "age":                d["age"],
            "employment_type":    d["employment_type"],
            "actual_default":     d["loan_default"],
            "actual_performance": "正常还款" if not d["loan_default"] else "部分违约",
            "similarity":         sim_display,
        })
    return results


def stats() -> dict:
    """诊断接口：返回引擎状态。"""
    return {
        "loaded":        _CACHE["loaded"],
        "row_count":     _CACHE["row_count"],
        "feature_count": len(_FEATURE_COLS),
        "features":      list(_FEATURE_COLS),
        "load_time_ms":  _CACHE["load_time_ms"],
    }
