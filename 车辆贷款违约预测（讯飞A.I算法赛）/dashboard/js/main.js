/* ================================================
   综合看板 - 主图表渲染逻辑
   ================================================ */

// ECharts 主题色
const CHART_COLORS = {
  primary: '#1a73e8',
  success: '#34a853',
  warning: '#fbbc04',
  danger: '#ea4335',
  purple: '#9334e6',
  orange: '#f57c00',
  textMuted: '#6b7785',
};

const ECHOICEBAR_COLOR = ['#34a853', '#1a73e8', '#f9ab00', '#f57c00', '#ea4335'];

/* ---------- 工具函数 ---------- */
function fmtMoney(v) {
  const n = Number(v) || 0;
  if (n >= 1e8) return (n / 1e8).toFixed(2) + '亿元';
  if (n >= 1e4) return (n / 1e4).toFixed(1) + '万元';
  return n.toLocaleString('zh-CN') + '元';
}

function fmtPct(v, decimals = 2) {
  return (v * 100).toFixed(decimals) + '%';
}

function fmtNum(v) {
  return Number(v).toLocaleString('zh-CN');
}

function getMetricClass(val, thresholds = [0.85, 0.70]) {
  if (val >= thresholds[0]) return 'good';
  if (val >= thresholds[1]) return 'warn';
  return 'bad';
}

/* ---------- 初始化所有图表 ---------- */
let charts = {};

function initCharts() {
  charts.overdueTrend = echarts.init(document.getElementById('chartOverdueTrend'));
  charts.riskPie = echarts.init(document.getElementById('chartRiskPie'));
  charts.areaRisk = echarts.init(document.getElementById('chartAreaRisk'));
  charts.clusterScatter = echarts.init(document.getElementById('chartClusterScatter'));
  charts.creditDist = echarts.init(document.getElementById('chartCreditDist'));
  charts.shapImportance = echarts.init(document.getElementById('chartShapImportance'));
}

/* ---------- KPI 卡片渲染 ---------- */
function renderKPICards(data) {
  const d = data || MockData.overview;
  document.getElementById('kpiTotalCustomers').textContent = fmtNum(d.total_customers || 0);
  document.getElementById('kpiTotalAmount').textContent = fmtMoney(d.total_amount || 0);
  document.getElementById('kpiOverdueRate').textContent = fmtPct(d.overdue_rate || 0);
  document.getElementById('kpiNewCustomers').textContent = fmtNum(d.new_customers || 0);
}

/* ---------- 逾期率趋势图 ---------- */
function renderOverdueTrend(data) {
  const d = data || MockData.riskDaily;
  const xData = d.map(x => x.dt);
  const yData = d.map(x => parseFloat((x.default_rate * 100).toFixed(2)));

  charts.overdueTrend.setOption({
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      formatter: (params) => `${params[0].name}<br/>逾期率: <b>${params[0].value}%</b>`,
    },
    grid: { top: 10, right: 10, bottom: 25, left: 40 },
    xAxis: {
      type: 'category',
      data: xData,
      axisLine: { lineStyle: { color: '#e0e4e8' } },
      axisLabel: { color: '#6b7785', fontSize: 10 },
    },
    yAxis: {
      type: 'value',
      axisLabel: { formatter: (v) => v + '%', color: '#6b7785', fontSize: 10 },
      splitLine: { lineStyle: { color: '#f0f0f0' } },
    },
    series: [{
      type: 'line',
      data: yData,
      smooth: true,
      lineStyle: { color: CHART_COLORS.primary, width: 2 },
      areaStyle: {
        color: {
          type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(26,115,232,0.25)' },
            { offset: 1, color: 'rgba(26,115,232,0.02)' },
          ],
        },
      },
      symbol: 'circle',
      symbolSize: 4,
      itemStyle: { color: CHART_COLORS.primary },
    }],
  });
}

/* ---------- 风险等级饼图 ---------- */
function renderRiskPie(data) {
  const d = data || MockData.riskDistribution;
  const colorMap = {
    '低风险': CHART_COLORS.success,
    '中风险': CHART_COLORS.warning,
    '高风险': CHART_COLORS.danger,
  };

  charts.riskPie.setOption({
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      formatter: (p) => `${p.name}<br/>占比: <b>${p.value}%</b>`,
    },
    legend: {
      orient: 'vertical',
      right: 8,
      top: 'center',
      itemWidth: 10,
      itemHeight: 10,
      textStyle: { fontSize: 11 },
    },
    series: [{
      type: 'pie',
      radius: ['45%', '72%'],
      center: ['38%', '50%'],
      avoidLabelOverlap: false,
      label: { show: false },
      data: d.map(x => ({
        name: x.name,
        value: x.value,
        itemStyle: { color: colorMap[x.name] || CHART_COLORS.primary },
      })),
    }],
  });
}

/* ---------- 地区风险 Top5 柱状图 ---------- */
function renderAreaRisk(data) {
  const d = data || MockData.areaRisk;
  const sorted = [...d].sort((a, b) => b.rate - a.rate).slice(0, 5);

  charts.areaRisk.setOption({
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      formatter: (p) => `${p[0].name}<br/>违约率: <b>${(p[0].value * 100).toFixed(2)}%</b>`,
    },
    grid: { top: 10, right: 8, bottom: 25, left: 8 },
    xAxis: { type: 'category', data: sorted.map(x => x.area), axisLine: { lineStyle: { color: '#e0e4e8' } }, axisLabel: { fontSize: 9, color: '#6b7785' } },
    yAxis: {
      type: 'value',
      axisLabel: { formatter: (v) => (v * 100).toFixed(1) + '%', fontSize: 9, color: '#6b7785' },
      splitLine: { lineStyle: { color: '#f0f0f0' } },
    },
    series: [{
      type: 'bar',
      data: sorted.map(x => ({
        value: x.rate,
        itemStyle: {
          color: x.rate > 0.06 ? CHART_COLORS.danger : x.rate > 0.04 ? CHART_COLORS.warning : CHART_COLORS.success,
        },
      })),
      barMaxWidth: 28,
      label: { show: true, position: 'top', formatter: (p) => (p.value * 100).toFixed(1) + '%', fontSize: 9, color: '#6b7785' },
    }],
  });
}

/* ---------- 客户聚类散点图 ---------- */
function renderClusterScatter(data) {
  const d = data || MockData.customerCluster;
  const series = d.clusters.map((c, i) => ({
    name: c.name,
    type: 'scatter',
    symbolSize: 8,
    data: d.scatterData.filter(x => x[2] === c.name).map(x => [x[0], x[1]]),
    itemStyle: { color: c.color, opacity: 0.6 },
  }));

  charts.clusterScatter.setOption({
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      formatter: (p) => `信用评分: ${p.value[0]}<br/>贷款金额: ${fmtMoney(p.value[1])}`,
    },
    legend: {
      orient: 'horizontal',
      bottom: 0,
      itemWidth: 10,
      itemHeight: 10,
      textStyle: { fontSize: 10 },
      data: d.clusters.map(c => c.name),
    },
    grid: { top: 10, right: 10, bottom: 35, left: 55 },
    xAxis: {
      type: 'value',
      name: '信用评分',
      nameLocation: 'center',
      nameGap: 22,
      nameTextStyle: { fontSize: 10, color: '#6b7785' },
      axisLabel: { fontSize: 10, color: '#6b7785' },
      splitLine: { lineStyle: { color: '#f0f0f0' } },
    },
    yAxis: {
      type: 'value',
      name: '贷款金额(元)',
      nameLocation: 'center',
      nameGap: 40,
      nameTextStyle: { fontSize: 10, color: '#6b7785' },
      axisLabel: { formatter: (v) => fmtMoney(v), fontSize: 9, color: '#6b7785' },
      splitLine: { lineStyle: { color: '#f0f0f0' } },
    },
    series,
  });
}

/* ---------- 信用评分分布直方图 ---------- */
function renderCreditDist(data) {
  const d = data || MockData.creditScoreDist;

  charts.creditDist.setOption({
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      formatter: (p) => `${p[0].name}<br/>客户数: <b>${fmtNum(p[0].value)}</b>`,
    },
    grid: { top: 10, right: 10, bottom: 25, left: 50 },
    xAxis: {
      type: 'category',
      data: d.buckets,
      axisLine: { lineStyle: { color: '#e0e4e8' } },
      axisLabel: { fontSize: 10, color: '#6b7785' },
    },
    yAxis: {
      type: 'value',
      axisLabel: { formatter: (v) => fmtMoney(v), fontSize: 9, color: '#6b7785' },
      splitLine: { lineStyle: { color: '#f0f0f0' } },
    },
    series: [{
      type: 'bar',
      data: d.counts.map((v, i) => ({
        value: v,
        itemStyle: {
          color: ECHOICEBAR_COLOR[i % ECHOICEBAR_COLOR.length],
        },
      })),
      barMaxWidth: 40,
      label: { show: false },
    }],
  });
}

/* ---------- SHAP 特征重要性图 ---------- */
function renderShapImportance(data) {
  const d = (data || MockData.shapImportance).slice(0, 10).reverse();

  charts.shapImportance.setOption({
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      formatter: (p) => `${p[0].name}<br/>SHAP: <b>${p[0].value.toFixed(3)}</b>`,
    },
    grid: { top: 5, right: 80, bottom: 5, left: 110 },
    xAxis: { type: 'value', axisLabel: { fontSize: 9, color: '#6b7785' }, splitLine: { lineStyle: { color: '#f0f0f0' } } },
    yAxis: {
      type: 'category',
      data: d.map(x => x.display || x.name),
      axisLabel: { fontSize: 10, color: '#5f6368' },
      axisLine: { show: false },
      axisTick: { show: false },
    },
    series: [{
      type: 'bar',
      data: d.map(x => x.mean_abs_shap),
      barMaxWidth: 18,
      label: { show: true, position: 'right', formatter: (p) => p.value.toFixed(2), fontSize: 9, color: '#6b7785' },
      itemStyle: {
        color: (p) => {
          const colors = ['#ea4335', '#f57c00', '#f9ab00', '#34a853', '#1a73e8', '#9334e6',
                          '#5f6368', '#5f6368', '#5f6368', '#5f6368'];
          return colors[p.dataIndex] || CHART_COLORS.primary;
        },
      },
    }],
  });
}

/* ---------- 模型指标渲染 ---------- */
function renderModelMetrics(data) {
  const d = data || MockData.modelMetrics;
  const cls = (v) => getMetricClass(v);

  const aucEl = document.getElementById('metricAuc');
  const preEl = document.getElementById('metricPrecision');
  const recEl = document.getElementById('metricRecall');
  const f1El = document.getElementById('metricF1');

  if (aucEl) { aucEl.textContent = d.auc?.toFixed(3) || '--'; aucEl.className = 'm-value ' + cls(d.auc); }
  if (preEl) { preEl.textContent = d.precision?.toFixed(3) || '--'; preEl.className = 'm-value ' + cls(d.precision); }
  if (recEl) { recEl.textContent = d.recall?.toFixed(3) || '--'; recEl.className = 'm-value ' + cls(d.recall); }
  if (f1El) { f1El.textContent = d.f1?.toFixed(3) || '--'; f1El.className = 'm-value ' + cls(d.f1); }
}

/* ---------- 系统运行指标 ---------- */
function renderSystemMetrics(data) {
  const d = data || MockData.systemMetrics;
  const apiEl = document.getElementById('sysApiCount');
  const latEl = document.getElementById('sysAvgLatency');
  const repEl = document.getElementById('sysRepairRate');
  if (apiEl) apiEl.textContent = fmtNum(d.api_calls || 0);
  if (latEl) latEl.textContent = d.avg_latency_ms || '--';
  if (repEl) repEl.textContent = fmtPct(d.repair_success_rate || 0);
}

/* ---------- 最新决策记录表格 ---------- */
function renderRecentDecisions(data) {
  const tbody = document.getElementById('recentDecisionsBody');
  if (!tbody) return;
  const rows = (data || MockData.recentDecisions).slice(0, 10);

  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted" style="padding:24px;">暂无数据</td></tr>';
    return;
  }

  tbody.innerHTML = rows.map(r => {
    const riskBadge = r.default_probability > 0.5
      ? '<span class="badge badge-danger">高风险</span>'
      : r.default_probability > 0.3
        ? '<span class="badge badge-warning">中风险</span>'
        : '<span class="badge badge-success">低风险</span>';
    return `<tr>
      <td class="highlight">${r.customer_id}</td>
      <td><span class="${r.default_probability > 0.5 ? 'text-danger' : r.default_probability > 0.3 ? 'text-warning' : 'text-success'}">${(r.default_probability * 100).toFixed(1)}%</span></td>
      <td>${r.credit_score ?? '--'}</td>
      <td>${r.predicted_limit != null ? fmtMoney(r.predicted_limit) : '--'}</td>
      <td>${(r.fraud_probability * 100).toFixed(1)}%</td>
      <td class="text-muted" style="font-size:12px;">${r.created_at || '--'}</td>
    </tr>`;
  }).join('');
}

/* ---------- 系统状态指示 ---------- */
function updateSystemStatus(health) {
  const dot = document.getElementById('sysStatusDot');
  const text = document.getElementById('sysStatusText');
  if (!dot || !text) return;

  if (health && health.status === 'ok') {
    dot.className = 'status-dot';
    text.textContent = '系统正常';
  } else {
    dot.className = 'status-dot error';
    text.textContent = '系统异常';
  }
}

/* ---------- 刷新所有数据 ---------- */
async function refreshAll() {
  document.getElementById('lastUpdate').textContent = '正在刷新...';

  // 并行加载所有数据
  const [overview, riskDaily, riskDist, modelMetrics, areaRisk, clusterData, creditDist, shapData, recentDecisions, health] = await Promise.all([
    API.statsOverview(),
    API.statsRiskDaily(),
    API.statsRiskDistribution(),
    API.statsModelMetrics(),
    API.statsAreaRisk(),
    API.statsCustomerCluster(),
    API.statsCreditScoreDist(),
    API.modelShapValues(),
    API.statsRecentDecisions ? API.statsRecentDecisions() : Promise.resolve(null),
    API.health(),
  ]);

  // 更新KPI
  renderKPICards(overview);

  // 更新图表
  renderOverdueTrend(riskDaily);
  renderRiskPie(riskDist);
  renderAreaRisk(areaRisk);
  renderClusterScatter(clusterData);
  renderCreditDist(creditDist);
  renderShapImportance(shapData);

  // 模型指标
  renderModelMetrics(modelMetrics);

  // 系统指标
  renderSystemMetrics(MockData.systemMetrics);

  // 最新决策
  renderRecentDecisions(recentDecisions);

  // 系统状态
  updateSystemStatus(health);

  // 更新时间
  document.getElementById('lastUpdate').textContent =
    '最后更新: ' + new Date().toLocaleString('zh-CN');
}

/* ---------- 页面初始化 ---------- */
let resizeTimer;
async function boot() {
  // 初始化图表
  initCharts();

  // 加载数据
  await refreshAll();

  // 监听窗口resize
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      Object.values(charts).forEach(c => c && c.resize());
    }, 300);
  });

  // 自动刷新
  setInterval(refreshAll, 5 * 60 * 1000);
}

// 挂载到全局
window.refreshAll = refreshAll;
window.boot = boot;

// 启动
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
