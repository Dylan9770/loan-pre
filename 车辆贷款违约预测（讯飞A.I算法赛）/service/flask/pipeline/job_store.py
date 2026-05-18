"""Pipeline 任务状态持久化（MySQL loan_rt.import_pipeline_jobs）。

每一步的执行进度都通过 update_step / mark_done / mark_failed 写回数据库，
前端轮询 GET /import/status/<job_id> 就能看到实时进度。
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pymysql

from service.flask.config import Settings


PIPELINE_STEPS = [
    {"id": 1, "name": "MySQL 业务库落库",   "layer": "MySQL loan_ods",            "icon": "1F4E5"},
    {"id": 2, "name": "Flume 文件采集",      "layer": "Flume SpoolDir",            "icon": "1F310"},
    {"id": 3, "name": "HDFS Raw 入湖",       "layer": "HDFS /data_lake/raw",       "icon": "1F4C2"},
    {"id": 4, "name": "数据清洗 (DWD)",      "layer": "HDFS /data_lake/cleaned",   "icon": "1F9F9"},
    {"id": 5, "name": "数据修复",            "layer": "FP-Growth + 中位数填补",     "icon": "1F527"},
    {"id": 6, "name": "特征工程",            "layer": "HDFS /data_lake/featured",  "icon": "2699"},
    {"id": 7, "name": "模型推理",            "layer": "XGBoost + RF + 规则引擎",   "icon": "1F9E0"},
    {"id": 8, "name": "结果落库",            "layer": "MySQL loan_rt",             "icon": "1F4BE"},
]


def _connect():
    return pymysql.connect(
        host=Settings.MYSQL_HOST, port=Settings.MYSQL_PORT,
        user=Settings.MYSQL_USER, password=Settings.MYSQL_PASSWORD,
        database=Settings.MYSQL_DB_RT, charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor, autocommit=True,
    )


def _init_steps() -> list[dict]:
    """初始化所有步骤为 pending 状态。"""
    return [
        {**s, "status": "pending", "message": "", "started_at": None, "ended_at": None, "duration_ms": 0}
        for s in PIPELINE_STEPS
    ]


def create_job(job_id: str, filename: str) -> None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO import_pipeline_jobs
                   (job_id, filename, status, current_step, steps_json)
                   VALUES (%s, %s, 'pending', '', %s)""",
                (job_id, filename, json.dumps(_init_steps(), ensure_ascii=False)),
            )
    finally:
        conn.close()


def _get_steps(cur, job_id: str) -> list[dict]:
    cur.execute("SELECT steps_json FROM import_pipeline_jobs WHERE job_id=%s", (job_id,))
    row = cur.fetchone()
    if not row or not row["steps_json"]:
        return _init_steps()
    raw = row["steps_json"]
    return json.loads(raw) if isinstance(raw, str) else raw


def start_step(job_id: str, step_id: int, message: str = "") -> None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            steps = _get_steps(cur, job_id)
            for s in steps:
                if s["id"] == step_id:
                    s["status"] = "running"
                    s["message"] = message
                    s["started_at"] = datetime.now().isoformat(timespec="seconds")
                    name = s["name"]
                    break
            else:
                name = f"step{step_id}"
            cur.execute(
                """UPDATE import_pipeline_jobs
                   SET status='running', current_step=%s, steps_json=%s
                   WHERE job_id=%s""",
                (name, json.dumps(steps, ensure_ascii=False), job_id),
            )
    finally:
        conn.close()


def finish_step(job_id: str, step_id: int, message: str = "", extra: dict | None = None) -> None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            steps = _get_steps(cur, job_id)
            for s in steps:
                if s["id"] == step_id:
                    s["status"] = "completed"
                    if message:
                        s["message"] = message
                    s["ended_at"] = datetime.now().isoformat(timespec="seconds")
                    # duration
                    if s.get("started_at"):
                        try:
                            t0 = datetime.fromisoformat(s["started_at"])
                            t1 = datetime.fromisoformat(s["ended_at"])
                            s["duration_ms"] = int((t1 - t0).total_seconds() * 1000)
                        except Exception:
                            pass
                    if extra:
                        s["extra"] = extra
                    break
            cur.execute(
                "UPDATE import_pipeline_jobs SET steps_json=%s WHERE job_id=%s",
                (json.dumps(steps, ensure_ascii=False), job_id),
            )
    finally:
        conn.close()


def fail_step(job_id: str, step_id: int, error: str) -> None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            steps = _get_steps(cur, job_id)
            for s in steps:
                if s["id"] == step_id:
                    s["status"] = "failed"
                    s["message"] = error
                    s["ended_at"] = datetime.now().isoformat(timespec="seconds")
                    break
            cur.execute(
                """UPDATE import_pipeline_jobs
                   SET status='failed', error_msg=%s, steps_json=%s,
                       completed_at=NOW()
                   WHERE job_id=%s""",
                (error[:1000], json.dumps(steps, ensure_ascii=False), job_id),
            )
    finally:
        conn.close()


def mark_done(job_id: str, result: dict[str, Any]) -> None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE import_pipeline_jobs
                   SET status='completed', current_step='完成',
                       result_json=%s, completed_at=NOW()
                   WHERE job_id=%s""",
                (json.dumps(result, ensure_ascii=False, default=str), job_id),
            )
    finally:
        conn.close()


def set_total_rows(job_id: str, n: int) -> None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE import_pipeline_jobs SET total_rows=%s WHERE job_id=%s",
                (int(n), job_id),
            )
    finally:
        conn.close()


def get_job(job_id: str) -> dict | None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT job_id, filename, total_rows, status, current_step,
                          steps_json, result_json, error_msg, created_at, completed_at
                   FROM import_pipeline_jobs WHERE job_id=%s""",
                (job_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            d = dict(row)
            for k in ("steps_json", "result_json"):
                v = d.get(k)
                if isinstance(v, str):
                    try:
                        d[k] = json.loads(v)
                    except Exception:
                        pass
            return d
    finally:
        conn.close()


def list_jobs(limit: int = 20) -> list[dict]:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT job_id, filename, total_rows, status, current_step,
                          created_at, completed_at
                   FROM import_pipeline_jobs
                   ORDER BY created_at DESC LIMIT %s""",
                (int(limit),),
            )
            return [dict(r) for r in cur.fetchall() or []]
    finally:
        conn.close()
