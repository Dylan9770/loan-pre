"""Customer profile routes."""

from __future__ import annotations

import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify

from service.flask.model_loader import predict_default, predict_fraud, predict_limit, score_credit
from service.flask.repositories.mysql_repo import (
    fetch_customer_loan_facts,
    fetch_customer_profile,
    fetch_customer_similar,
)

customer_bp = Blueprint("customer_bp", __name__, url_prefix="/customer")


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0 or math.isnan(b) or math.isnan(a):
        return default
    return a / b


def _compute_radar_scores(profile: dict) -> dict:
    """Compute 5-dimension radar chart scores for a customer."""
    credit_score = float(profile.get("credit_score") or 0)

    # Dimension 1: 信用评分 (0-850 -> 0-100 scaled)
    credit_dim = min(credit_score / 8.5, 100)

    # Dimension 2: 还款能力 (基于债务收入比)
    disbursed = float(profile.get("disbursed_amount") or 0)
    outstanding = float(profile.get("total_outstanding_loan") or 0)
    asset_cost = float(profile.get("asset_cost") or 1)
    ratio = _safe_div(outstanding, disbursed if disbursed else asset_cost)
    repay_ability = max(0, min(100, 100 - ratio * 80))

    # Dimension 3: 资产状况 (基于贷款资产比)
    ltv = _safe_div(disbursed, asset_cost)
    asset_status = max(0, min(100, 100 - (ltv - 0.5) * 100))

    # Dimension 4: 历史记录 (基于逾期次数和逾期率)
    overdue_no = float(profile.get("total_overdue_no") or 0)
    account_no = float(profile.get("total_account_loan_no") or 1)
    overdue_rate = _safe_div(overdue_no, account_no)
    history_score = max(0, min(100, 100 - overdue_no * 15 - overdue_rate * 40))

    # Dimension 5: 稳定性 (基于年龄和工作类型)
    age = float(profile.get("age") or 35)
    emp_type = int(profile.get("employment_type") or 0)
    stability = max(0, min(100, 100 - abs(age - 40) * 1.5 - (emp_type == 2) * 10))

    return {
        "credit": round(credit_dim, 2),
        "repay_ability": round(repay_ability, 2),
        "asset_status": round(asset_status, 2),
        "history": round(history_score, 2),
        "stability": round(stability, 2),
    }


def _build_mock_profile(customer_id: int) -> dict:
    """Generate deterministic mock profile for demo purposes."""
    rng_seed = customer_id % 1000
    np.random.seed(rng_seed)

    base_score = 500 + rng_seed
    overdue_count = rng_seed % 5

    profile = {
        "customer_id": customer_id,
        "age": 25 + (rng_seed % 40),
        "employment_type": rng_seed % 3,
        "area_id": rng_seed % 10,
        "credit_score": base_score,
        "disbursed_amount": 10000 + (rng_seed % 80000),
        "total_outstanding_loan": 5000 + (rng_seed % 30000),
        "asset_cost": 15000 + (rng_seed % 100000),
        "total_overdue_no": overdue_count,
        "total_account_loan_no": 1 + (rng_seed % 6),
        "main_account_overdue_no": overdue_count,
        "main_account_loan_no": 1 + (rng_seed % 4),
        "total_monthly_payment": 500 + (rng_seed % 3000),
        "total_disbursed_loan": 10000 + (rng_seed % 80000),
        "last_six_month_new_loan_no": rng_seed % 3,
        "last_six_month_defaulted_no": overdue_count % 2,
        "credit_history": 1 + (rng_seed % 10),
        "enquirie_no": rng_seed % 8,
        "loan_default": 1 if rng_seed % 7 == 0 else 0,
    }
    return profile


def _build_mock_timeline(customer_id: int) -> list[dict]:
    """Generate realistic loan repayment timeline for demo (events up to today only)."""
    from datetime import date
    TODAY = date.today()
    rng_seed = customer_id % 1000

    # 贷款基本参数
    principal   = 20000 + (rng_seed % 16) * 5000          # 20000~95000，步进5000
    annual_rate = [4.9, 5.5, 6.0, 6.5, 7.0][rng_seed % 5]
    term        = [12, 24, 36, 48][rng_seed % 4]           # 期数
    monthly_rate = annual_rate / 100 / 12

    # 等额还款月供公式
    if monthly_rate > 0:
        monthly_payment = principal * monthly_rate * (1 + monthly_rate) ** term \
                          / ((1 + monthly_rate) ** term - 1)
    else:
        monthly_payment = principal / term
    monthly_payment = round(monthly_payment)

    # 逾期期数（0~2次，高风险客户更多）
    overdue_months = set()
    if rng_seed % 7 == 0:          # 约14%客户有逾期
        overdue_months.add(rng_seed % (term // 2) + 2)
    if rng_seed % 15 == 0:         # 约7%有两次逾期
        overdue_months.add(rng_seed % term + 1)

    # 申请和放款日期：确保贷款开始时间在今天之前至少 3 个月
    apply_year  = 2023 if rng_seed % 3 == 0 else 2024
    apply_month = 1 + (rng_seed % 10)       # 1~10月申请
    apply_day   = 5 + (rng_seed % 10)

    def fmt_date(y, m, d):
        m = ((m - 1) % 12) + 1
        y_adj = y + (apply_month + (m - apply_month) - 1) // 12
        return f"{y_adj}-{m:02d}-{min(d, 28):02d}"

    events = [
        {
            "date": fmt_date(apply_year, apply_month, apply_day),
            "type": "loan-apply",
            "title": "提交贷款申请",
            "detail": f"申请金额: {principal:,}元, 用途: 购车",
        },
        {
            "date": fmt_date(apply_year, apply_month, apply_day + 10),
            "type": "loan-disbursed",
            "title": "贷款发放",
            "detail": f"实际发放: {principal:,}元, 年利率: {annual_rate}%, 期限: {term}期, 月供: {monthly_payment:,}元",
        },
    ]

    # 还款记录：从放款下一个月开始
    balance = principal
    repay_month = apply_month + 1
    repay_year  = apply_year
    completed   = False

    for period in range(1, term + 1):
        if repay_month > 12:
            repay_month -= 12
            repay_year  += 1

        repay_day = apply_day + 10  # 每月固定还款日

        # 计算本期利息和本金
        interest   = round(balance * monthly_rate)
        principal_part = monthly_payment - interest
        balance    = max(0, balance - principal_part)

        # 该期还款日期
        event_date_str = fmt_date(repay_year, repay_month, repay_day)
        try:
            event_date = date.fromisoformat(event_date_str)
        except ValueError:
            event_date = TODAY  # 容错

        # 只记录今天及之前已发生的事件
        if event_date > TODAY:
            break

        if period in overdue_months:
            overdue_days = 3 + rng_seed % 12
            penalty      = round(monthly_payment * 0.0005 * overdue_days)
            events.append({
                "date": fmt_date(repay_year, repay_month, repay_day + overdue_days),
                "type": "overdue",
                "title": f"逾期还款（第{period}期）",
                "detail": f"逾期{overdue_days}天, 本期还款: {monthly_payment:,}元, 罚息: {penalty}元, 已补缴",
            })
        else:
            events.append({
                "date": event_date_str,
                "type": "ontime-repay",
                "title": f"按时还款（第{period}期）",
                "detail": f"本期还款: {monthly_payment:,}元, 其中利息: {interest}元, 剩余本金: {balance:,}元",
            })

        repay_month += 1

        # 提前结清（约20%客户在最后6期内提前结清，且结清日在今天之前）
        early_close_period = term - (rng_seed % 6)
        if period == early_close_period and rng_seed % 5 == 0 and balance > 0:
            close_date_str = fmt_date(repay_year, repay_month, repay_day)
            try:
                close_date = date.fromisoformat(close_date_str)
            except ValueError:
                close_date = TODAY
            if close_date <= TODAY:
                events.append({
                    "date": close_date_str,
                    "type": "closed",
                    "title": "提前结清",
                    "detail": f"提前偿还剩余本金: {balance:,}元, 结清证明已生成",
                })
                completed = True
            break

    if not completed and period == term:
        final_date_str = fmt_date(repay_year, repay_month, apply_day + 10)
        try:
            final_date = date.fromisoformat(final_date_str)
        except ValueError:
            final_date = TODAY
        if final_date <= TODAY:
            events.append({
                "date": final_date_str,
                "type": "closed",
                "title": "贷款结清",
                "detail": "全部本息已还清, 结清证明已生成",
            })

    return events


@customer_bp.get("/<int:customer_id>/profile")
def get_customer_profile(customer_id: int):
    """Return complete customer profile with radar scores."""
    # Try to fetch from MySQL first
    db_profile = fetch_customer_profile(customer_id)

    if db_profile:
        profile = db_profile
    else:
        # Fall back to mock data for demo
        profile = _build_mock_profile(customer_id)

    # Compute radar chart scores
    radar = _compute_radar_scores(profile)

    # Get prediction results
    try:
        pred_default = predict_default([profile])
        pred_fraud = predict_fraud([profile])
        pred_limit = predict_limit([profile])

        default_prob = pred_default[0]["default_probability"] if pred_default else 0.0
        fraud_prob = pred_fraud[0]["fraud_probability"] if pred_fraud else 0.0
        limit_val = pred_limit[0]["predicted_limit"] if pred_limit else 0.0
        credit_score_val = score_credit([default_prob])[0]
    except Exception:
        # Fallback mock predictions
        base_prob = (1000 - (customer_id % 1000)) / 1000.0
        default_prob = max(0.01, min(0.99, base_prob))
        fraud_prob = max(0.01, min(0.5, (customer_id % 50) / 100))
        limit_val = 10000 + (customer_id % 80000)
        credit_score_val = 600 - 50 * math.log(default_prob / (1 - default_prob))

    result = {
        "customer_id": customer_id,
        "profile": {
            "age": profile.get("age"),
            "employment_type": profile.get("employment_type"),
            "area_id": profile.get("area_id"),
            "credit_score": profile.get("credit_score"),
            "disbursed_amount": profile.get("disbursed_amount"),
            "total_overdue_no": profile.get("total_overdue_no"),
            "total_account_loan_no": profile.get("total_account_loan_no"),
            "loan_default": profile.get("loan_default"),
        },
        "radar_scores": radar,
        "decision": {
            "default_probability": round(default_prob, 4),
            "default_pred": 1 if default_prob >= 0.5 else 0,
            "fraud_probability": round(fraud_prob, 4),
            "fraud_pred": 1 if fraud_prob >= 0.5 else 0,
            "predicted_limit": round(limit_val, 2),
            "credit_score": round(credit_score_val, 1),
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    return jsonify(result)


@customer_bp.get("/<int:customer_id>/similar")
def get_similar_customers(customer_id: int):
    """Return Top-K similar customers based on feature cosine similarity."""
    db_similar = fetch_customer_similar(customer_id, k=5)

    if db_similar:
        return jsonify(db_similar)

    # Generate mock similar customers for demo
    rng_seed = customer_id % 1000
    np.random.seed(rng_seed)

    # Create mock target profile
    target_score = 500 + rng_seed
    target_overdue = rng_seed % 5

    similar = []
    for i in range(5):
        offset = (i + 1) * 5
        sim_id = customer_id + offset * 100
        sim_score = max(300, min(850, target_score + np.random.randint(-30, 30)))
        sim_overdue = max(0, min(5, target_overdue + np.random.randint(-1, 1)))

        default_actual = 1 if sim_overdue >= 3 else 0
        performance = "正常还款" if default_actual == 0 else "部分违约"
        similarity = max(0.70, 0.99 - i * 0.04 - abs(sim_score - target_score) / 1000)

        similar.append({
            "customer_id": int(sim_id),
            "credit_score": int(sim_score),
            "disbursed_amount": float(10000 + (sim_id % 80000)),
            "total_overdue_no": int(sim_overdue),
            "actual_default": int(default_actual),
            "actual_performance": performance,
            "similarity": round(similarity, 4),
        })

    return jsonify(similar)


def _build_timeline_from_facts(customer_id: int, facts: dict) -> list[dict]:
    """Build a loan timeline using real loan_fact fields.

    Real fields: disbursed_amount, total_monthly_payment, total_overdue_no, loan_default.
    Constructed (CSV lacks month/day & per-installment data): apply day, repayment day,
    which installments are overdue, period length when monthly_payment is missing/invalid.
    """
    from datetime import date
    TODAY = date.today()
    rng_seed = customer_id % 1000

    principal = float(facts.get("disbursed_amount") or 0)
    monthly_payment_raw = float(facts.get("total_monthly_payment") or 0)
    overdue_no = int(facts.get("total_overdue_no") or 0)
    defaulted = int(facts.get("loan_default") or 0)
    disbursed_year = str(facts.get("disbursed_date") or "2019")[:4]
    try:
        apply_year = int(disbursed_year)
    except ValueError:
        apply_year = 2019

    if principal <= 0:
        principal = 20000 + (rng_seed % 16) * 5000

    # 月供异常时（缺失或超过本金的 1/3，明显是数据噪声）按等额本息回算
    annual_rate = [4.9, 5.5, 6.0, 6.5, 7.0][rng_seed % 5]
    term = [12, 24, 36, 48][rng_seed % 4]
    monthly_rate = annual_rate / 100 / 12

    if monthly_payment_raw <= 0 or monthly_payment_raw > principal / 3:
        if monthly_rate > 0:
            monthly_payment = principal * monthly_rate * (1 + monthly_rate) ** term \
                              / ((1 + monthly_rate) ** term - 1)
        else:
            monthly_payment = principal / term
        monthly_payment = round(monthly_payment)
    else:
        monthly_payment = round(monthly_payment_raw)
        # 用真实月供反推期数
        if monthly_payment > monthly_rate * principal:
            est_term = principal / (monthly_payment - monthly_rate * principal) if monthly_rate > 0 else principal / monthly_payment
            est_term = int(min(60, max(6, est_term)))
            term = est_term

    apply_month = 1 + (rng_seed % 10)
    apply_day = 5 + (rng_seed % 10)

    # 把真实的 total_overdue_no 分布到不同期数上
    overdue_months = set()
    if overdue_no > 0:
        for k in range(min(overdue_no, term - 1)):
            slot = ((rng_seed * (k + 1)) % (term - 1)) + 1
            overdue_months.add(slot)

    def fmt_date(y, m, d):
        m = ((m - 1) % 12) + 1
        return f"{y}-{m:02d}-{min(d, 28):02d}"

    events = [
        {
            "date": fmt_date(apply_year, apply_month, apply_day),
            "type": "loan-apply",
            "title": "提交贷款申请",
            "detail": f"申请金额: {int(principal):,}元, 用途: 购车",
        },
        {
            "date": fmt_date(apply_year, apply_month, apply_day + 10),
            "type": "loan-disbursed",
            "title": "贷款发放",
            "detail": (
                f"实际发放: {int(principal):,}元, 年利率: {annual_rate}%, "
                f"期限: {term}期, 月供: {monthly_payment:,}元"
            ),
        },
    ]

    balance = principal
    repay_month = apply_month + 1
    repay_year = apply_year
    completed = False
    period = 0

    for period in range(1, term + 1):
        if repay_month > 12:
            repay_month -= 12
            repay_year += 1

        repay_day = apply_day + 10
        interest = round(balance * monthly_rate)
        principal_part = monthly_payment - interest
        balance = max(0, balance - principal_part)

        event_date_str = fmt_date(repay_year, repay_month, repay_day)
        try:
            event_date = date.fromisoformat(event_date_str)
        except ValueError:
            event_date = TODAY

        if event_date > TODAY:
            break

        if period in overdue_months:
            overdue_days = 3 + (rng_seed + period) % 12
            penalty = round(monthly_payment * 0.0005 * overdue_days)
            events.append({
                "date": fmt_date(repay_year, repay_month, repay_day + overdue_days),
                "type": "overdue",
                "title": f"逾期还款（第{period}期）",
                "detail": (
                    f"逾期{overdue_days}天, 本期应还: {monthly_payment:,}元, "
                    f"罚息: {penalty}元, 已补缴"
                ),
            })
        else:
            events.append({
                "date": event_date_str,
                "type": "ontime-repay",
                "title": f"按时还款（第{period}期）",
                "detail": (
                    f"本期还款: {monthly_payment:,}元, 利息: {interest}元, "
                    f"剩余本金: {int(balance):,}元"
                ),
            })

        repay_month += 1

    # 如果违约标签 = 1，最后追加一条违约状态记录
    if defaulted and events:
        last_date = events[-1]["date"]
        events.append({
            "date": last_date,
            "type": "overdue",
            "title": "标记违约",
            "detail": f"累计逾期 {overdue_no} 次，账户被标记为违约状态",
        })
    elif period == term and balance <= 1 and not defaulted:
        final_date_str = fmt_date(repay_year, repay_month, apply_day + 10)
        try:
            final_date = date.fromisoformat(final_date_str)
        except ValueError:
            final_date = TODAY
        if final_date <= TODAY:
            events.append({
                "date": final_date_str,
                "type": "closed",
                "title": "贷款结清",
                "detail": "全部本息已还清, 结清证明已生成",
            })
            completed = True

    return events


@customer_bp.get("/<int:customer_id>/loan_history")
def get_customer_loan_history(customer_id: int):
    """Return loan behavior timeline for a customer.

    数据策略：优先从 loan_fact 读取真实金额/月供/逾期次数/违约标签，
    其余（月份分配、具体逾期期数）按 customer_id 派生稳定构造。
    没有真实数据时退化为完全 mock。
    """
    facts = fetch_customer_loan_facts(customer_id)
    if facts:
        timeline = _build_timeline_from_facts(customer_id, facts)
        source = "loan_fact"
    else:
        timeline = _build_mock_timeline(customer_id)
        source = "mock"
    return jsonify({
        "customer_id": customer_id,
        "events": timeline,
        "total_events": len(timeline),
        "source": source,
    })
