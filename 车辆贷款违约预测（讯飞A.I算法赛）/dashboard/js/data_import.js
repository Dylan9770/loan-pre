/* ================================================
   数据导入 Pipeline 前端逻辑
   ================================================ */

// 8 步定义（与后端 job_store.PIPELINE_STEPS 对应）
const STEP_DEFS = [
  { id: 1, name: 'MySQL 业务库落库',  layer: 'MySQL loan_ods.import_staging',  icon: '\u{1F4E5}' },
  { id: 2, name: 'Flume 文件采集',    layer: 'Flume SpoolDir → /data/flume/',  icon: '\u{1F310}' },
  { id: 3, name: 'HDFS Raw 入湖',     layer: 'HDFS /data_lake/raw/dt=TODAY',   icon: '\u{1F4C2}' },
  { id: 4, name: '数据清洗 (DWD层)',  layer: 'HDFS /data_lake/cleaned',        icon: '\u{1F9F9}' },
  { id: 5, name: '数据修复',          layer: 'FP-Growth + 中位数填补',         icon: '\u{1F527}' },
  { id: 6, name: '特征工程',          layer: 'HDFS /data_lake/featured',       icon: '\u{2699}️' },
  { id: 7, name: '模型推理',          layer: 'XGBoost + RandomForest + 规则',  icon: '\u{1F9E0}' },
  { id: 8, name: '结果落库',          layer: 'MySQL loan_rt.realtime_decisions', icon: '\u{1F4BE}' },
];

let currentJobId = null;
let pollTimer    = null;
let pickedFile   = null;

/* ---------- 初始化 ---------- */
window.addEventListener('DOMContentLoaded', () => {
  renderTimeline(initialSteps());
  bindUpload();
  loadHistory();
  setInterval(loadHistory, 5000);
});

function initialSteps() {
  return STEP_DEFS.map(s => ({ ...s, status: 'pending', message: '', extra: null }));
}

/* ---------- 上传交互 ---------- */
function bindUpload() {
  const zone  = document.getElementById('dropZone');
  const input = document.getElementById('fileInput');

  input.addEventListener('change', e => {
    if (e.target.files.length > 0) setFile(e.target.files[0]);
  });

  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) setFile(e.dataTransfer.files[0]);
  });
}

function setFile(f) {
  pickedFile = f;
  const nameEl = document.getElementById('fileName');
  nameEl.textContent = `${f.name}  (${(f.size / 1024).toFixed(1)} KB)`;
  nameEl.classList.remove('empty');
  document.getElementById('btnUpload').disabled = false;
}

async function startUpload() {
  if (!pickedFile) return;
  document.getElementById('btnUpload').disabled = true;
  document.getElementById('btnUpload').textContent = '上传中...';

  const fd = new FormData();
  fd.append('file', pickedFile);
  try {
    const r = await fetch('/import/upload', { method: 'POST', body: fd });
    const j = await r.json();
    if (!r.ok) {
      alert('上传失败: ' + (j.error || r.status));
      document.getElementById('btnUpload').disabled = false;
      document.getElementById('btnUpload').textContent = '开始导入';
      return;
    }
    currentJobId = j.job_id;
    document.getElementById('bJobId').textContent = j.job_id;
    document.getElementById('bStatus').textContent = '运行中';
    document.getElementById('btnUpload').textContent = '运行中...';
    document.getElementById('resultCard').style.display = 'none';
    renderTimeline(initialSteps());
    startPolling();
  } catch (err) {
    alert('上传异常: ' + err.message);
    document.getElementById('btnUpload').disabled = false;
    document.getElementById('btnUpload').textContent = '开始导入';
  }
}

function downloadTemplate() {
  window.location.href = '/import/template';
}

/* ---------- 轮询任务状态 ---------- */
function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollOnce();
  pollTimer = setInterval(pollOnce, 800);
}

async function pollOnce() {
  if (!currentJobId) return;
  const job = await API.get(`/import/status/${currentJobId}`);
  if (!job) return;

  document.getElementById('bRows').textContent = job.total_rows || 0;
  document.getElementById('bStep').textContent = job.current_step || '-';

  const statusMap = { pending: '等待中', running: '运行中', completed: '已完成', failed: '失败' };
  document.getElementById('bStatus').textContent = statusMap[job.status] || job.status;

  const banner = document.querySelectorAll('.banner-card');
  banner.forEach(b => b.classList.remove('success', 'warn', 'danger'));
  if (job.status === 'completed') banner[3].classList.add('success');
  else if (job.status === 'failed') banner[3].classList.add('danger');
  else if (job.status === 'running') banner[3].classList.add('warn');

  if (job.steps_json) renderTimeline(job.steps_json);

  if (job.status === 'completed' || job.status === 'failed') {
    clearInterval(pollTimer); pollTimer = null;
    document.getElementById('btnUpload').disabled = false;
    document.getElementById('btnUpload').textContent = '开始导入';
    if (job.status === 'completed' && job.result_json) {
      renderResults(job.result_json);
    }
    loadHistory();
  }
}

/* ---------- 渲染 Pipeline 时间线 ---------- */
function renderTimeline(steps) {
  const box = document.getElementById('pipelineTimeline');
  box.innerHTML = steps.map(s => stepNodeHTML(s)).join('');
}

function stepNodeHTML(s) {
  const def = STEP_DEFS.find(d => d.id === s.id) || {};
  const icon = def.icon || '';
  const layer = s.layer || def.layer || '';
  const statusIcon =
    s.status === 'completed' ? '✓' :
    s.status === 'failed'    ? '!' :
    s.status === 'running'   ? '...' : s.id;

  const extraRows = s.extra ? Object.entries(s.extra).map(([k, v]) => `
    <div class="ek">${escapeHtml(k)}</div>
    <div class="ev">${escapeHtml(formatVal(v))}</div>
  `).join('') : '';

  const timeInfo = (s.started_at || s.ended_at) ?
    `<div class="step-time">${s.started_at ? '开始 ' + fmtTime(s.started_at) : ''}${s.duration_ms ? ' · 耗时 ' + s.duration_ms + 'ms' : ''}</div>` : '';

  return `
    <div class="step-node ${s.status}">
      <div class="step-dot">${statusIcon}</div>
      <div class="step-head">
        <div class="step-name">${icon} ${escapeHtml(s.name)}</div>
        <div class="step-layer">${escapeHtml(layer)}</div>
      </div>
      <div class="step-msg">${escapeHtml(s.message || '等待执行')}</div>
      ${extraRows ? `<div class="step-extra">${extraRows}</div>` : ''}
      ${timeInfo}
    </div>
  `;
}

/* ---------- 渲染结果 ---------- */
function renderResults(result) {
  const card = document.getElementById('resultCard');
  const box  = document.getElementById('resultBox');
  card.style.display = '';

  const summary = `
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px;">
      <div class="stat-card-mini" style="background:#fafbfc;padding:10px 14px;border-radius:6px;">
        <div style="font-size:11px;color:var(--color-text-muted);">导入行数</div>
        <div style="font-size:1.4rem;font-weight:700;">${result.rows_in || 0}</div>
      </div>
      <div class="stat-card-mini" style="background:#e8f0fe;padding:10px 14px;border-radius:6px;">
        <div style="font-size:11px;color:var(--color-text-muted);">平均信用分</div>
        <div style="font-size:1.4rem;font-weight:700;color:#1a73e8;">${result.avg_credit_score || '--'}</div>
      </div>
      <div class="stat-card-mini" style="background:#fef7e0;padding:10px 14px;border-radius:6px;">
        <div style="font-size:11px;color:var(--color-text-muted);">违约预测</div>
        <div style="font-size:1.4rem;font-weight:700;color:#f9ab00;">${result.default_hits || 0} / ${result.rows_predicted || 0}</div>
      </div>
      <div class="stat-card-mini" style="background:#fce8e6;padding:10px 14px;border-radius:6px;">
        <div style="font-size:11px;color:var(--color-text-muted);">欺诈命中</div>
        <div style="font-size:1.4rem;font-weight:700;color:#ea4335;">${result.fraud_hits || 0} / ${result.rows_predicted || 0}</div>
      </div>
    </div>
  `;

  const results = result.results || [];
  const tableRows = results.slice(0, 50).map(r => `
    <tr>
      <td><a href="/dashboard/customer_profile.html?customer_id=${r.customer_id}" target="_blank" style="color:var(--color-primary);">${r.customer_id}</a></td>
      <td class="score">${r.credit_score}</td>
      <td class="score">${(r.default_probability * 100).toFixed(2)}%</td>
      <td>${r.default_pred ? '<span class="tag-red">违约</span>' : '<span class="tag-grn">正常</span>'}</td>
      <td>${r.fraud_pred ? '<span class="tag-red">疑似</span>' : '<span class="tag-grn">正常</span>'}</td>
      <td class="score">${Number(r.predicted_limit).toLocaleString()}</td>
    </tr>
  `).join('');

  const tableHtml = `
    <table class="result-table">
      <thead><tr>
        <th>客户ID</th><th>信用分</th><th>违约概率</th><th>违约判定</th>
        <th>欺诈判定</th><th>建议额度(元)</th>
      </tr></thead>
      <tbody>${tableRows}</tbody>
    </table>
    ${results.length > 50 ? `<div style="text-align:center;color:var(--color-text-muted);font-size:11px;margin-top:8px;">仅显示前 50 条 / 共 ${results.length} 条</div>` : ''}
  `;

  box.innerHTML = summary + tableHtml;
}

/* ---------- 历史任务 ---------- */
async function loadHistory() {
  const jobs = await API.get('/import/jobs?limit=10');
  const box = document.getElementById('historyList');
  if (!jobs || jobs.length === 0) {
    box.innerHTML = '<div style="padding:16px;text-align:center;color:var(--color-text-muted);font-size:12px;">暂无历史任务</div>';
    return;
  }
  box.innerHTML = jobs.map(j => `
    <div class="history-row" onclick="loadJob('${j.job_id}')">
      <div class="h-id">${j.job_id}</div>
      <div class="h-file" title="${escapeHtml(j.filename || '')}">${escapeHtml(j.filename || '-')}</div>
      <div class="h-rows">${j.total_rows || 0} 行</div>
      <div class="h-status ${j.status}">${{pending:'等待',running:'运行',completed:'完成',failed:'失败'}[j.status] || j.status}</div>
    </div>
  `).join('');
}

async function loadJob(jobId) {
  currentJobId = jobId;
  document.getElementById('bJobId').textContent = jobId;
  if (pollTimer) clearInterval(pollTimer);
  pollOnce();
  // 如果还在跑就继续轮询，已完成的拉一次就够
  const job = await API.get(`/import/status/${jobId}`);
  if (job && (job.status === 'running' || job.status === 'pending')) {
    startPolling();
  }
}

/* ---------- utils ---------- */
function escapeHtml(s) {
  if (s == null) return '';
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}
function fmtTime(iso) {
  try { return iso.slice(11, 19); } catch { return iso; }
}
function formatVal(v) {
  if (v == null) return '-';
  if (Array.isArray(v)) return v.join(', ');
  if (typeof v === 'object') return JSON.stringify(v);
  if (typeof v === 'number') return Number.isInteger(v) ? v : v.toFixed(2);
  return String(v);
}
