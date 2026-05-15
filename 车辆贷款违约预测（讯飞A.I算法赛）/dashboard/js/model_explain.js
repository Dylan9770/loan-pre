/* ================================================
   模型解释页面 - 真实 SHAP 与多模型对比
   ================================================ */

const CHART_COLORS = {
  primary: '#1a73e8',
  success: '#34a853',
  warning: '#fbbc04',
  danger: '#ea4335',
  purple: '#9334e6',
};

/* 模型名 → 中文显示名 + 速度标签（用于多模型对比表） */
const FRAUD_MODEL_META = {
  tabnet:        { display: 'TabNet',     speed: '深度学习', color: '#1a73e8' },
  random_forest: { display: '随机森林',   speed: '集成树',   color: '#34a853' },
  decision_tree: { display: '决策树',     speed: '单棵树',   color: '#f9ab00' },
};

const DEFAULT_MODEL_META = {
  xgboost_regressor: { display: 'XGBoost', speed: '梯度提升', color: '#1a73e8' },
  bilstm:            { display: 'BiLSTM',  speed: '深度学习', color: '#9334e6' },
  mlp:               { display: 'MLP',     speed: '深度学习', color: '#34a853' },
};

/* ---------- SHAP 特征重要性条形图 ---------- */
function renderShapBarChart(data) {
  const chart = echarts.init(document.getElementById('chartShapBar'));
  if (!chart) return;
  const d = data.slice(0, 12).reverse();

  chart.setOption({
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      formatter: (p) => `${p[0].name}<br/>SHAP: <b>${p[0].value.toFixed(3)}</b>`,
    },
    grid: { top: 10, right: 80, bottom: 10, left: 120 },
    xAxis: { type: 'value', axisLabel: { fontSize: 10, color: '#6b7785' }, splitLine: { lineStyle: { color: '#f0f0f0' } } },
    yAxis: {
      type: 'category',
      data: d.map(x => x.display || x.name),
      axisLabel: { fontSize: 11, color: '#5f6368' },
      axisLine: { show: false },
      axisTick: { show: false },
    },
    series: [{
      type: 'bar',
      data: d.map((x) => ({
        value: x.mean_abs_shap,
        itemStyle: { color: x.impact === '负向' ? '#34a853' : '#ea4335' },
      })),
      barMaxWidth: 22,
      label: { show: true, position: 'right', formatter: (p) => p.value.toFixed(2), fontSize: 10, color: '#6b7785' },
    }],
  });
}

/* ---------- 多模型对比图（欺诈分类指标） ---------- */
function renderModelCompareChart(fraudModels) {
  const chart = echarts.init(document.getElementById('chartModelCompare'));
  if (!chart || !fraudModels || !fraudModels.length) return;

  chart.setOption({
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (params) => {
        const m = fraudModels[params[0].dataIndex];
        const meta = FRAUD_MODEL_META[m.name] || { display: m.name };
        return `<b>${meta.display}</b><br/>
          AUC: ${m.auc.toFixed(3)}<br/>
          精确率: ${m.precision.toFixed(3)}<br/>
          召回率: ${m.recall.toFixed(3)}<br/>
          F1: ${m.f1.toFixed(3)}`;
      },
    },
    legend: { bottom: 0, itemWidth: 10, itemHeight: 10, textStyle: { fontSize: 10 } },
    grid: { top: 10, right: 10, bottom: 40, left: 10 },
    xAxis: {
      type: 'category',
      data: fraudModels.map(m => (FRAUD_MODEL_META[m.name] || {}).display || m.name),
      axisLabel: { fontSize: 10, color: '#6b7785' },
      axisLine: { lineStyle: { color: '#e0e4e8' } },
    },
    yAxis: {
      type: 'value',
      max: 1,
      axisLabel: { fontSize: 10, formatter: (v) => v.toFixed(1), color: '#6b7785' },
      splitLine: { lineStyle: { color: '#f0f0f0' } },
    },
    series: [
      { name: 'AUC',     type: 'bar', data: fraudModels.map(m => m.auc),       itemStyle: { color: '#1a73e8' }, barMaxWidth: 25 },
      { name: '精确率', type: 'bar', data: fraudModels.map(m => m.precision), itemStyle: { color: '#34a853' }, barMaxWidth: 25 },
      { name: '召回率', type: 'bar', data: fraudModels.map(m => m.recall),    itemStyle: { color: '#f9ab00' }, barMaxWidth: 25 },
    ],
  });
}

/* ---------- 多模型对比表（欺诈 + 违约 两套指标各一行小表） ---------- */
function renderModelCompareTable(comparison) {
  const tbody = document.getElementById('modelCompareTable');
  if (!tbody) return;

  const fraudRows = (comparison.fraud || [])
    .slice()
    .sort((a, b) => (b.is_winner ? 1 : 0) - (a.is_winner ? 1 : 0))
    .map((m) => {
      const meta = FRAUD_MODEL_META[m.name] || { display: m.name, speed: '-' };
      return `<tr class="${m.is_winner ? 'best-cell' : ''}">
        <td>${m.is_winner ? '<b>' + meta.display + ' &#x2605;</b>' : meta.display}</td>
        <td>${m.auc.toFixed(3)}</td>
        <td>${m.precision.toFixed(3)}</td>
        <td>${m.recall.toFixed(3)}</td>
        <td>${m.f1.toFixed(3)}</td>
        <td><span class="badge badge-info">${meta.speed}</span></td>
      </tr>`;
    }).join('');

  const defaultRows = (comparison.default || [])
    .slice()
    .sort((a, b) => (b.is_winner ? 1 : 0) - (a.is_winner ? 1 : 0))
    .map((m) => {
      const meta = DEFAULT_MODEL_META[m.name] || { display: m.name, speed: '-' };
      return `<tr class="${m.is_winner ? 'best-cell' : ''}">
        <td>${m.is_winner ? '<b>' + meta.display + ' &#x2605;</b>' : meta.display}</td>
        <td colspan="2" style="text-align:center;color:#6b7785;font-size:11px;">R² ${m.r2.toFixed(3)}</td>
        <td colspan="2" style="text-align:center;color:#6b7785;font-size:11px;">RMSE ${m.rmse.toFixed(2)}</td>
        <td><span class="badge badge-info">${meta.speed}</span></td>
      </tr>`;
    }).join('');

  tbody.innerHTML = `
    <tr><td colspan="6" style="background:#f5f7fa;font-weight:600;font-size:12px;color:#5f6368;">
      欺诈检测（分类指标）</td></tr>
    ${fraudRows}
    <tr><td colspan="6" style="background:#f5f7fa;font-weight:600;font-size:12px;color:#5f6368;">
      违约预测（回归指标，R² / RMSE）</td></tr>
    ${defaultRows}
  `;
}

/* ---------- SHAP Waterfall Chart（单客户解释） ---------- */
function renderShapWaterfall(sample) {
  const chart = echarts.init(document.getElementById('chartShapWaterfall'));
  if (!chart || !sample) return;

  // sample: { label, credit_score, p_default, base_value, items: [{display, value}] }
  const items = sample.items;
  const base  = sample.base_value;

  let cumValue = base;
  const waterfallData = [];
  const labels = ['Base'];
  // base bar
  waterfallData.push({ value: [0, 0, base, base], itemStyle: { color: '#6b7785' } });

  items.forEach((item, i) => {
    const idx = i + 1;
    if (item.value >= 0) {
      waterfallData.push({ value: [idx, cumValue, cumValue + item.value, item.value],
                            itemStyle: { color: '#ea4335' } });
    } else {
      waterfallData.push({ value: [idx, cumValue + item.value, cumValue, item.value],
                            itemStyle: { color: '#34a853' } });
    }
    cumValue += item.value;
    labels.push(item.display || item.name);
  });

  // Final prediction bar
  waterfallData.push({ value: [items.length + 1, 0, cumValue, cumValue],
                        itemStyle: { color: cumValue < 600 ? '#ea4335' : '#34a853' } });
  labels.push(`预测=${cumValue.toFixed(0)}`);

  chart.setOption({
    backgroundColor: 'transparent',
    tooltip: {
      formatter: (p) => {
        const i = p.dataIndex;
        if (i === 0) return `<b>Base value</b><br/>训练集平均评分: <b>${base.toFixed(1)}</b>`;
        if (i === items.length + 1) return `<b>最终预测</b><br/>评分: <b>${cumValue.toFixed(1)}</b><br/>P(违约): <b>${sample.p_default}</b>`;
        const it = items[i - 1];
        const sign = it.value >= 0 ? '+' : '';
        return `<b>${it.display || it.name}</b><br/>SHAP贡献: <b style="color:${it.value >= 0 ? '#ea4335' : '#34a853'}">${sign}${it.value.toFixed(2)}</b>`;
      },
    },
    grid: { top: 10, right: 60, bottom: 50, left: 60 },
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: { fontSize: 9, color: '#6b7785', rotate: 30, interval: 0 },
      axisLine: { lineStyle: { color: '#e0e4e8' } },
    },
    yAxis: {
      type: 'value',
      axisLabel: { fontSize: 10, formatter: (v) => v.toFixed(0), color: '#6b7785' },
      splitLine: { lineStyle: { color: '#f0f0f0' } },
    },
    series: [{
      type: 'custom',
      renderItem: (params, api) => {
        const val = api.value();
        const i = val[0];
        const start = api.coord([i, val[1]]);
        const end = api.coord([i, val[2]]);
        const height = Math.abs(end[1] - start[1]);
        const y = Math.min(start[1], end[1]);
        return {
          type: 'rect',
          shape: { x: start[0] - 18, y, width: 36, height: Math.max(2, height) },
          style: api.style(),
        };
      },
      data: waterfallData,
      encode: { x: 0, y: [1, 2] },
    }],
  });
}

/* ---------- SHAP Waterfall 详情列表 ---------- */
function renderShapWaterfallDetail(sample) {
  const el = document.getElementById('shapWaterfallDetail');
  if (!el || !sample) return;
  // 顶部加一行样本元信息
  const meta = `<div style="margin-bottom:8px;padding:8px 10px;background:#f5f7fa;border-radius:4px;font-size:12px;">
      <b>${sample.label}</b> &nbsp; 评分=${sample.credit_score} &nbsp; P(违约)=${(sample.p_default * 100).toFixed(1)}%
    </div>`;
  el.innerHTML = meta + sample.items.map(item => {
    const isPos = item.value > 0;
    const absScaled = Math.min(50, Math.abs(item.value) * 4);   // 评分量纲，调比例
    return `<div class="shap-bar-row">
      <span class="shap-bar-label">${item.display || item.name}</span>
      <div class="shap-bar-track">
        <div class="shap-bar-fill ${isPos ? 'positive' : 'negative'}"
             style="width:${absScaled}%;${isPos ? 'right:auto;left:50%;' : 'left:auto;right:50%;'}"></div>
      </div>
      <span class="shap-bar-value" style="color:${isPos ? '#ea4335' : '#34a853'};">${isPos ? '+' : ''}${item.value.toFixed(1)}</span>
    </div>`;
  }).join('');
}

/* ---------- 特征影响详情列表 Top8 ---------- */
function renderShapDetailList(data) {
  const el = document.getElementById('shapDetailList');
  if (!el) return;
  const d = data.slice(0, 8);
  el.innerHTML = d.map(item => `
    <div class="shap-detail-item">
      <div class="shap-detail-header">
        <div class="shap-detail-icon" style="background:${item.impact === '负向' ? '#e6f4ea' : '#fce8e6'};color:${item.impact === '负向' ? '#34a853' : '#ea4335'};">
          ${item.impact === '负向' ? '&#x2193;' : '&#x2191;'}
        </div>
        <div class="shap-detail-name">${item.display || item.name}</div>
        <div class="shap-detail-effect ${item.impact === '负向' ? 'negative' : 'positive'}">
          ${item.impact === '负向' ? '降低风险' : '增加风险'}
        </div>
      </div>
      <div class="shap-detail-desc">${item.description || ''}</div>
    </div>
  `).join('');
}

/* ---------- 决策案例卡片（接 /stats/recent_decisions） ---------- */
function renderDecisionCases(decisions) {
  const el = document.getElementById('decisionCases');
  if (!el) return;
  if (!decisions || !decisions.length) {
    el.innerHTML = '<div class="card-body" style="text-align:center;color:#6b7785;">暂无最近决策记录</div>';
    return;
  }
  // 取低/中/高三档（按 default_probability 排序后挑代表）
  const sorted = decisions.slice().sort((a, b) =>
    (a.default_probability || 0) - (b.default_probability || 0));
  const picks = sorted.length >= 3
    ? [sorted[0], sorted[Math.floor(sorted.length / 2)], sorted[sorted.length - 1]]
    : sorted;

  el.innerHTML = picks.map((c, idx) => {
    const p = c.default_probability || 0;
    const result = p < 0.25 ? 'approve' : p > 0.55 ? 'reject' : 'caution';
    const label  = result === 'approve' ? '建议批准'
                 : result === 'reject'  ? '建议拒绝' : '审慎批准';
    const badgeCls = result === 'approve' ? 'badge-success'
                    : result === 'reject' ? 'badge-danger' : 'badge-warning';
    const reason = result === 'approve'
      ? `客户信用评分 ${c.credit_score} 较高，违约概率仅 ${(p*100).toFixed(1)}%；欺诈概率 ${((c.fraud_probability||0)*100).toFixed(1)}%。建议批准，可参考额度 ${(c.predicted_limit||0).toLocaleString()} 元。`
      : result === 'reject'
        ? `客户信用评分 ${c.credit_score} 偏低，违约概率高达 ${(p*100).toFixed(1)}%；建议拒绝或要求增强担保。`
        : `客户信用评分 ${c.credit_score} 居中，违约概率 ${(p*100).toFixed(1)}%；欺诈概率 ${((c.fraud_probability||0)*100).toFixed(1)}%。建议审慎批准，可适度降低额度。`;

    return `<div class="decision-case ${result}">
      <div class="case-header">
        <div class="case-title">案例 #${idx + 1} - 客户 ${c.customer_id}</div>
        <span class="badge ${badgeCls}">${label}</span>
      </div>
      <div class="case-features">
        <div class="case-feature">
          <div class="cf-label">违约概率</div>
          <div class="cf-value" style="color:${p > 0.5 ? '#ea4335' : p > 0.3 ? '#f9ab00' : '#34a853'};">${(p * 100).toFixed(1)}%</div>
        </div>
        <div class="case-feature">
          <div class="cf-label">信用评分</div>
          <div class="cf-value">${c.credit_score || '-'}</div>
        </div>
        <div class="case-feature">
          <div class="cf-label">建议额度</div>
          <div class="cf-value">${(c.predicted_limit || 0).toLocaleString()} 元</div>
        </div>
      </div>
      <div class="case-reason">${reason}</div>
    </div>`;
  }).join('');
}

/* ---------- 阈值调整（仅显示，未来可接 /predict 重算） ---------- */
function updateThreshold(value) {
  const el = document.getElementById('thresholdDisplay');
  if (el) el.textContent = parseFloat(value).toFixed(2);
}

/* ---------- 页面初始化 ---------- */
let resizeTimer;
let _waterfallSamples = [];

async function boot() {
  // 并行拉所有真实数据
  const [shapGlobal, comparison, samples, decisions] = await Promise.all([
    API.modelShapValues(),
    API.modelComparison(),
    API.modelShapWaterfallSamples(),
    API.statsRecentDecisions(),
  ]);

  if (Array.isArray(shapGlobal)) {
    renderShapBarChart(shapGlobal);
    renderShapDetailList(shapGlobal);
  } else {
    console.warn('[ModelExplain] shap_global unavailable:', shapGlobal);
  }

  if (comparison && (comparison.default || comparison.fraud)) {
    renderModelCompareChart(comparison.fraud || []);
    renderModelCompareTable(comparison);
  } else {
    console.warn('[ModelExplain] comparison unavailable:', comparison);
  }

  if (Array.isArray(samples) && samples.length) {
    _waterfallSamples = samples;
    renderShapWaterfall(samples[0]);
    renderShapWaterfallDetail(samples[0]);
    _attachSampleSwitcher(samples);
  } else {
    console.warn('[ModelExplain] shap_samples unavailable:', samples);
  }

  renderDecisionCases(Array.isArray(decisions) ? decisions : []);

  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      ['chartShapBar', 'chartModelCompare', 'chartShapWaterfall'].forEach(id => {
        const inst = echarts.getInstanceByDom(document.getElementById(id));
        if (inst) inst.resize();
      });
    }, 300);
  });
}

/* 在 Waterfall 卡片上加"低/中/高风险样本"切换按钮 */
function _attachSampleSwitcher(samples) {
  const detailEl = document.getElementById('shapWaterfallDetail');
  if (!detailEl) return;
  const switcher = document.createElement('div');
  switcher.style.cssText = 'display:flex;gap:6px;margin-bottom:8px;';
  samples.forEach((s, i) => {
    const btn = document.createElement('button');
    btn.textContent = s.label;
    btn.style.cssText = `flex:1;padding:6px;font-size:11px;border:1px solid #d0d4d8;border-radius:4px;cursor:pointer;background:${i === 0 ? '#1a73e8' : '#fff'};color:${i === 0 ? '#fff' : '#5f6368'};`;
    btn.onclick = () => {
      switcher.querySelectorAll('button').forEach((b, j) => {
        b.style.background = (i === j) ? '#1a73e8' : '#fff';
        b.style.color      = (i === j) ? '#fff' : '#5f6368';
      });
      renderShapWaterfall(samples[i]);
      renderShapWaterfallDetail(samples[i]);
    };
    switcher.appendChild(btn);
  });
  // 插入到详情列表上方
  detailEl.parentElement.insertBefore(switcher, detailEl);
}

/* ---------- 启动 ---------- */
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
