"""数据导入 Pipeline HTTP 接口。

POST /import/upload       上传 Excel/CSV，启动后台 pipeline，返回 job_id
GET  /import/status/<id>  查询 pipeline 进度（前端轮询）
GET  /import/jobs         最近 N 条任务列表
GET  /import/template     下载示例 Excel 模板
"""
from __future__ import annotations

import io
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Blueprint, jsonify, request, send_file

from service.flask.pipeline import job_store, runner

import_bp = Blueprint("import_bp", __name__, url_prefix="/import")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_UPLOAD_DIR = _PROJECT_ROOT / "data" / "uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_ALLOWED_EXT = {".xlsx", ".xls", ".csv"}


@import_bp.post("/upload")
def upload():
    if "file" not in request.files:
        return jsonify({"error": "请通过 form-data 字段 file 上传"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "未选择文件"}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in _ALLOWED_EXT:
        return jsonify({"error": f"仅支持 {sorted(_ALLOWED_EXT)}"}), 400

    job_id = uuid.uuid4().hex[:12]
    save_path = _UPLOAD_DIR / f"{job_id}_{f.filename}"
    f.save(save_path)

    try:
        job_store.create_job(job_id, f.filename)
    except Exception as exc:
        return jsonify({"error": f"创建任务失败: {exc}"}), 500

    runner.run_pipeline(job_id, save_path)
    return jsonify({"job_id": job_id, "filename": f.filename, "status": "running"})


@import_bp.get("/status/<job_id>")
def status(job_id: str):
    job = job_store.get_job(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job)


@import_bp.get("/jobs")
def jobs():
    limit = int(request.args.get("limit", 20))
    return jsonify(job_store.list_jobs(limit))


@import_bp.get("/template")
def template():
    """返回示例 Excel 模板，让用户知道列名格式。"""
    sample = pd.DataFrame([
        {
            "customer_id": 999001, "age": 35, "employment_type": 1, "area_id": 3,
            "credit_score": 680, "credit_history": 8,
            "disbursed_amount": 50000, "asset_cost": 100000, "ltv_ratio": 0.5,
            "total_overdue_no": 0, "total_outstanding_loan": 25000,
            "total_monthly_payment": 1500, "total_account_loan_no": 2,
            "total_disbursed_loan": 50000,
            "main_account_loan_no": 1, "main_account_active_loan_no": 1,
            "main_account_overdue_no": 0, "main_account_monthly_payment": 1500,
            "sub_account_loan_no": 0,
            "last_six_month_new_loan_no": 1, "last_six_month_defaulted_no": 0,
            "enquirie_no": 2, "Credit_level": 3,
        },
        {
            "customer_id": 999002, "age": 42, "employment_type": 2, "area_id": 5,
            "credit_score": None, "credit_history": 12,
            "disbursed_amount": 80000, "asset_cost": 160000, "ltv_ratio": 0.5,
            "total_overdue_no": 1, "total_outstanding_loan": 45000,
            "total_monthly_payment": 2200, "total_account_loan_no": 3,
            "total_disbursed_loan": 80000,
            "main_account_loan_no": 2, "main_account_active_loan_no": 2,
            "main_account_overdue_no": 1, "main_account_monthly_payment": 2200,
            "sub_account_loan_no": 1,
            "last_six_month_new_loan_no": 0, "last_six_month_defaulted_no": 0,
            "enquirie_no": 1, "Credit_level": -1,    # 缺失，会被 FP-Growth 修复
        },
    ])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        sample.to_excel(w, sheet_name="loan_import", index=False)
    buf.seek(0)
    return send_file(
        buf, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        download_name=f"loan_import_template_{datetime.now().strftime('%Y%m%d')}.xlsx",
        as_attachment=True,
    )
