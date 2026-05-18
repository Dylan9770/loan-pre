/* ================================================
   数据修复页 - 评估报告 + 单客户 ID 修复
   ================================================ */

const FIELD_DISPLAY = {
  employment_type:        '工作类型',
  area_id:                '地区编码',
  age:                    '年龄',
  credit_history:         '信用历史长度',
  Credit_level:           '信用等级',
  credit_score:           '信用评分',
  disbursed_amount:       '贷款金额',
  asset_cost:             '资产价值',
  total_outstanding_loan: '未偿贷款总额',
  total_monthly_payment:  '月供总额',
};

/* ---------- 渲染评估报告 ---------- */
function renderEvaluation(data) {
  if (!data || data.error) {
    document.getElementById('topStats').innerHTML =
      '<div class="card-body" style="grid-column:1/-1;color:#ea4335;text-align:center;">' +
        (data && data.error ? data.error : '无法加载 /repair/evaluation') +
      '</div>';
    return;
  }
  const fp  = data.fp_growth_style || {};
  const als = data.als_style || {};

  // 顶部 4 卡
  document.getElementById('statCoverage').textContent = pct(fp.coverage);
  document.getElementById('statRules').textContent    = fp.rules_count != null ? fp.rules_count : '--';
  document.getElementById('statConf').textContent     = pct(fp.avg_confidence);
  document.getElementById('statNumCov').textContent   = pct(als.coverage);

  document.getElementById('fpCoverage').textContent = pct(fp.coverage);
  document.getElementById('fpAcc').textContent      = pct(fp.accuracy);
  document.getElementById('fpConf').textContent     = pct(fp.avg_confidence);
  document.getElementById('fpRules').textContent    = fp.rules_count != null ? fp.rules_count : '--';

  document.getElementById('alsRmse').textContent     = als.rmse != null ? als.rmse.toLocaleString(undefined, {maximumFractionDigits:2}) : '--';
  document.getElementById('alsMape').textContent     = pct(als.mape);
  document.getElementById('alsCoverage').textContent = pct(als.coverage);
}

function pct(v) {
  if (v == null || isNaN(v)) return '--';
  return (v * 100).toFixed(2) + '%';
}

function fmtVal(v) {
  if (v == null) return '<span style="color:#9aa0a6;">缺失</span>';
  if (typeof v === 'number') {
    return Number.isInteger(v) ? v.toString() : v.toFixed(2);
  }
  return String(v);
}

function fieldName(k) {
  return FIELD_DISPLAY[k] ? `${FIELD_DISPLAY[k]} <span style="color:#9aa0a6;font-size:10px;">(${k})</span>` : k;
}

/* ---------- 渲染按客户 ID 修复结果 ---------- */
function renderRepairResult(data) {
  const box = document.getElementById('repairResultBox');
  if (!data) {
    box.innerHTML = `<div style="color:#ea4335;font-size:12px;text-align:center;padding:14px;">查询失败：后端不可达</div>`;
    return;
  }
  if (data.__not_found) {
    box.innerHTML = `<div style="color:#ea4335;font-size:12px;text-align:center;padding:14px;">未找到客户 ${data.customer_id}</div>`;
    return;
  }
  if (data.error) {
    box.innerHTML = `<div style="color:#ea4335;font-size:12px;text-align:center;padding:14px;">${data.error}</div>`;
    return;
  }

  const known    = data.known_fields    || [];
  const missing  = data.missing_fields  || [];
  const repaired = data.repaired_fields || [];

  let html = `
    <div style="display:flex;gap:12px;margin-bottom:12px;font-size:12px;">
      <div style="flex:1;background:#e6f4ea;padding:10px;border-radius:4px;text-align:center;">
        <div style="color:#34a853;font-weight:700;font-size:1.4rem;">${known.length}</div>
        <div style="color:#5f6368;">已存在字段</div>
      </div>
      <div style="flex:1;background:#fef7e0;padding:10px;border-radius:4px;text-align:center;">
        <div style="color:#f9ab00;font-weight:700;font-size:1.4rem;">${missing.length}</div>
        <div style="color:#5f6368;">缺失字段</div>
      </div>
      <div style="flex:1;background:#e8f0fe;padding:10px;border-radius:4px;text-align:center;">
        <div style="color:#1a73e8;font-weight:700;font-size:1.4rem;">${repaired.length}</div>
        <div style="color:#5f6368;">已修复字段</div>
      </div>
    </div>
  `;

  if (known.length) {
    html += `
      <div class="section-label" style="margin-top:10px;">数据库现有字段（客户 ${data.customer_id}）</div>
      <table class="known-fields-table">
        <thead><tr><th style="width:60%;">字段</th><th>原始值</th></tr></thead>
        <tbody>
          ${known.map(f => `<tr><td>${fieldName(f.field)}</td><td class="kv">${fmtVal(f.value)}</td></tr>`).join('')}
        </tbody>
      </table>`;
  }

  if (repaired.length) {
    html += `<div class="section-label" style="margin-top:14px;">修复结果</div>`;
    html += repaired.map(f => {
      const isEmpty = f.after == null;
      return `<div class="repair-row ${isEmpty ? 'empty' : ''}">
        <span class="rr-field">${fieldName(f.field)}</span>
        <span class="rr-method">${f.method}</span>
        <span class="rr-value">${isEmpty ? '<span style="color:#9aa0a6;">—</span>' : fmtVal(f.after)}</span>
        <span class="rr-conf">${f.confidence != null ? '置信 ' + (f.confidence * 100).toFixed(1) + '%'
                                : (f.note ? f.note : '')}</span>
      </div>`;
    }).join('');
    html += `<div style="font-size:11px;color:var(--color-text-muted);margin-top:8px;">
      规则库共 <b>${data.rules_count || 0}</b> 条 FP-Growth 关联规则参与匹配
    </div>`;
  } else if (missing.length === 0) {
    html += `<div style="font-size:12px;color:#34a853;text-align:center;padding:14px;background:#e6f4ea;border-radius:4px;margin-top:10px;">
      该客户所有关键字段都已存在，无需修复
    </div>`;
  }

  box.innerHTML = html;
}

/* ---------- 单客户修复 ---------- */
async function onRepairSubmit() {
  const idEl = document.getElementById('f_customer_id');
  const cid = parseInt((idEl.value || '').trim(), 10);
  if (!cid) {
    document.getElementById('repairResultBox').innerHTML =
      `<div style="color:#ea4335;font-size:12px;text-align:center;padding:14px;">请输入合法的客户 ID</div>`;
    return;
  }
  document.getElementById('repairResultBox').innerHTML =
    `<div style="text-align:center;padding:14px;color:var(--color-text-muted);font-size:12px;">查询中...</div>`;
  const data = await API.repairByCustomer(cid);
  renderRepairResult(data);
}

async function onPickRandom() {
  const r = await API.customerRandomId();
  if (r && r.customer_id) {
    document.getElementById('f_customer_id').value = r.customer_id;
    onRepairSubmit();
  } else {
    document.getElementById('repairResultBox').innerHTML =
      `<div style="color:#ea4335;font-size:12px;text-align:center;padding:14px;">随机选取失败</div>`;
  }
}

/* ---------- 页面初始化 ---------- */
async function boot() {
  const evalData = await API.repairEvaluation();
  renderEvaluation(evalData);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
