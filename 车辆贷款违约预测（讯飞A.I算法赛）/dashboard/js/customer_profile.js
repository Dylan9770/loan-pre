/* ================================================
   客户画像页面 - JS 逻辑
   ================================================ */

const CHART_COLORS = {
  primary: '#1a73e8',
  success: '#34a853',
  warning: '#fbbc04',
  danger: '#ea4335',
  purple: '#9334e6',
};

/* ---------- Mock 数据 ---------- */
const MockCustomerProfiles = {
  100001: {
    customer_id: 100001,
    age: 38,
    employment_type: '受薪员工',
    area: '华东区',
    credit_score: 742,
    disbursed_amount: 45000,
    total_overdue_no: 0,
    total_account_loan_no: 2,
    loan_default: 0,
    loan_asset_ratio: 0.78,
    recent_default_rate: 0.0,
  },
  100002: {
    customer_id: 100002,
    age: 45,
    employment_type: '自雇人士',
    area: '华北区',
    credit_score: 485,
    disbursed_amount: 12000,
    total_overdue_no: 3,
    total_account_loan_no: 5,
    loan_default: 1,
    loan_asset_ratio: 0.95,
    recent_default_rate: 0.6,
  },
};

const MockSimilarCustomers = [
  { customer_id: 100003, credit_score: 738, disbursed_amount: 44000, total_overdue_no: 0, actual_performance: '正常还款', similarity: 0.96 },
  { customer_id: 100004, credit_score: 745, disbursed_amount: 46000, total_overdue_no: 0, actual_performance: '正常还款', similarity: 0.94 },
  { customer_id: 100005, credit_score: 735, disbursed_amount: 43000, total_overdue_no: 1, actual_performance: '正常还款', similarity: 0.91 },
  { customer_id: 100006, credit_score: 740, disbursed_amount: 45500, total_overdue_no: 0, actual_performance: '正常还款', similarity: 0.89 },
  { customer_id: 100007, credit_score: 748, disbursed_amount: 47000, total_overdue_no: 0, actual_performance: '正常还款', similarity: 0.87 },
];

const MockLoanTimeline = [
  { date: '2024-01-15', type: 'loan-apply', title: '提交贷款申请', detail: '申请金额: 45000元, 用途: 购车' },
  { date: '2024-01-18', type: 'loan-disbursed', title: '贷款发放', detail: '实际发放: 45000元, 利率: 6.5%, 期限: 36期' },
  { date: '2024-02-15', type: 'ontime-repay', title: '按时还款', detail: '本期还款: 1380元, 余额: 43470元' },
  { date: '2024-03-15', type: 'ontime-repay', title: '按时还款', detail: '本期还款: 1380元, 余额: 42000元' },
  { date: '2024-04-15', type: 'ontime-repay', title: '按时还款', detail: '本期还款: 1380元, 余额: 40530元' },
  { date: '2024-05-15', type: 'overdue', title: '逾期还款', detail: '逾期5天, 罚款: 68元, 已补缴' },
  { date: '2024-06-15', type: 'ontime-repay', title: '按时还款', detail: '本期还款: 1380元, 余额: 37890元' },
  { date: '2024-07-15', type: 'closed', title: '贷款结清', detail: '提前还清全部本金, 结清证明已生成' },
];

/* ---------- 雷达图渲染 ---------- */
function renderRadarChart(profile) {
  const radarChart = echarts.init(document.getElementById('chartRadar'));
  if (!radarChart) return;

  const radarData = {
    credit: profile.credit_score || 0,
    repayAbility: Math.max(0, 100 - ((profile.loan_asset_ratio || 0) * 100)),
    assetStatus: Math.max(0, 100 - ((profile.recent_default_rate || 0) * 50)),
    history: Math.max(0, 100 - ((profile.total_overdue_no || 0) * 20)),
    stability: Math.max(50, 100 - Math.abs((profile.age || 35) - 40)),
  };

  const avgData = { credit: 650, repayAbility: 75, assetStatus: 78, history: 85, stability: 75 };

  const indicator = [
    { name: '信用评分', max: 850 },
    { name: '还款能力', max: 100 },
    { name: '资产状况', max: 100 },
    { name: '历史记录', max: 100 },
    { name: '稳定性', max: 100 },
  ];

  radarChart.setOption({
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      formatter: (p) => `${p.name}: <b>${p.value}</b>`,
    },
    legend: { show: false },
    radar: {
      indicator,
      radius: '68%',
      splitNumber: 4,
      axisName: { color: '#5f6368', fontSize: 12 },
      splitLine: { lineStyle: { color: '#e0e4e8' } },
      splitArea: { areaStyle: { color: ['#fafafa', '#f5f5f5', '#f0f0f0', '#ebebeb', '#e6e6e6'] } },
      axisLine: { lineStyle: { color: '#e0e4e8' } },
    },
    series: [{
      type: 'radar',
      data: [
        {
          value: [radarData.credit, radarData.repayAbility, radarData.assetStatus, radarData.history, radarData.stability],
          name: '当前客户',
          lineStyle: { color: CHART_COLORS.primary, width: 2 },
          areaStyle: { color: 'rgba(26,115,232,0.2)' },
          itemStyle: { color: CHART_COLORS.primary },
          symbol: 'circle',
          symbolSize: 5,
        },
        {
          value: [avgData.credit, avgData.repayAbility, avgData.assetStatus, avgData.history, avgData.stability],
          name: '参考平均',
          lineStyle: { color: 'rgba(26,115,232,0.3)', width: 1, type: 'dashed' },
          areaStyle: { color: 'rgba(26,115,232,0.05)' },
          itemStyle: { color: 'rgba(26,115,232,0.3)' },
          symbol: 'none',
        },
      ],
    }],
  });
}

/* ---------- 决策结果渲染 ---------- */
function renderDecisionResult(profile) {
  const el = document.getElementById('decisionResult');
  if (!el) return;

  // 优先用后端真实预测结果（profile 已展平了 decision 字段）
  const hasReal = profile.default_probability != null || profile.fraud_probability != null;
  const defaultProb = profile.default_probability != null
    ? Number(profile.default_probability)
    : (profile.loan_default === 1 ? 0.67 : 0.12);
  const fraudProb = profile.fraud_probability != null
    ? Number(profile.fraud_probability)
    : (profile.loan_default === 1 ? 0.15 : 0.03);
  const creditScore = Math.round(Number(profile.credit_score) || 650);
  const predictedLimit = profile.predicted_limit != null
    ? Number(profile.predicted_limit)
    : Number(profile.disbursed_amount || 0);

  const fmtAmount = (v) => {
    if (v >= 1e8) return (v / 1e8).toFixed(2) + '亿元';
    if (v >= 1e4) return (v / 1e4).toFixed(1) + '万元';
    return Number(v).toLocaleString('zh-CN') + '元';
  };

  const decisionData = [
    { label: '违约概率', value: (defaultProb * 100).toFixed(1) + '%', cls: defaultProb > 0.5 ? 'text-danger' : defaultProb > 0.3 ? 'text-warning' : 'text-success', bg: defaultProb > 0.5 ? '#fce8e6' : '#e6f4ea' },
    { label: '信用评分', value: creditScore, cls: creditScore >= 700 ? 'text-success' : creditScore >= 550 ? 'text-warning' : 'text-danger', bg: '#e8f0fe' },
    { label: '预测额度', value: fmtAmount(predictedLimit), cls: '', bg: '#e6f4ea' },
    { label: '欺诈概率', value: (fraudProb * 100).toFixed(1) + '%', cls: fraudProb > 0.5 ? 'text-danger' : fraudProb > 0.1 ? 'text-warning' : 'text-success', bg: '#fef7e0' },
  ];

  el.dataset.source = hasReal ? 'model' : 'fallback';

  el.innerHTML = decisionData.map(d => `
    <div class="decision-item" style="background:${d.bg};">
      <div class="d-label">${d.label}</div>
      <div class="d-value ${d.cls}">${d.value}</div>
    </div>
  `).join('');
}

/* ---------- SHAP Force Plot 模拟渲染 ---------- */
function renderShapForce(profile) {
  const chart = echarts.init(document.getElementById('chartShapForce'));
  if (!chart) return;

  const shapValues = [
    { feature: '信用评分高', value: -0.15, positive: false },
    { feature: '逾期次数多', value: 0.12, positive: true },
    { feature: '负债率高', value: 0.08, positive: true },
    { feature: '贷款金额大', value: 0.05, positive: false },
    { feature: '年龄适中', value: -0.04, positive: false },
    { feature: '其他因素', value: 0.02, positive: false },
  ];

  chart.setOption({
    backgroundColor: 'transparent',
    tooltip: {
      formatter: (p) => `<b>${p.data.name}</b>: ${p.data.value[1] > 0 ? '+' : ''}${p.data.value[1].toFixed(3)} (${p.data.value[1] > 0 ? '增风险' : '降风险'})`,
    },
    grid: { top: 10, right: 60, bottom: 30, left: 60 },
    xAxis: {
      type: 'category',
      data: shapValues.map(s => s.feature),
      axisLabel: { fontSize: 10, color: '#6b7785', rotate: 30 },
      axisLine: { lineStyle: { color: '#e0e4e8' } },
    },
    yAxis: {
      type: 'value',
      axisLabel: { formatter: (v) => v > 0 ? '+' + v.toFixed(2) : v.toFixed(2), fontSize: 9, color: '#6b7785' },
      splitLine: { lineStyle: { color: '#f0f0f0' } },
    },
    series: [{
      type: 'bar',
      data: shapValues.map(s => ({
        name: s.feature,
        value: s.value,
        itemStyle: { color: s.positive ? '#ea4335' : '#34a853', opacity: 0.85 },
      })),
      barWidth: '60%',
      label: {
        show: true,
        position: 'top',
        formatter: (p) => (p.value > 0 ? '+' : '') + p.value.toFixed(2),
        fontSize: 9,
        color: '#5f6368',
      },
    }],
  });
}

/* ---------- 相似客户表格渲染 ---------- */
function renderSimilarCustomers(profile, similar) {
  const tbody = document.getElementById('similarTableBody');
  if (!tbody) return;

  const isDefault = profile.loan_default === 1;
  const targetRow = `<tr class="similar-target">
    <td>${profile.customer_id}<br><span class="badge badge-${isDefault ? 'danger' : 'success'}">${isDefault ? '高风险' : '低风险'}</span></td>
    <td>${profile.credit_score || '--'}</td>
    <td>${profile.disbursed_amount != null ? profile.disbursed_amount.toLocaleString('zh-CN') : '--'}</td>
    <td>${profile.total_overdue_no ?? '--'}</td>
    <td><span class="badge badge-${isDefault ? 'danger' : 'success'}">${isDefault ? '违约' : '正常还款'}</span></td>
    <td>--</td>
  </tr>`;

  const similarRows = (similar || MockSimilarCustomers).map(s => `
    <tr>
      <td>${s.customer_id}</td>
      <td>${s.credit_score ?? '--'}</td>
      <td>${s.disbursed_amount != null ? s.disbursed_amount.toLocaleString('zh-CN') : '--'}</td>
      <td>${s.total_overdue_no ?? '--'}</td>
      <td><span class="badge badge-${s.actual_performance === '正常还款' ? 'success' : 'danger'}">${s.actual_performance}</span></td>
      <td>${(s.similarity * 100).toFixed(0)}%</td>
    </tr>
  `).join('');

  tbody.innerHTML = targetRow + similarRows;
}

/* ---------- 贷款时间轴渲染 ---------- */
function renderTimeline(timeline) {
  const el = document.getElementById('timelineContainer');
  if (!el) return;

  const events = timeline || MockLoanTimeline;
  const typeClass = {
    'loan-apply': 'loan-apply',
    'loan-disbursed': 'loan-disbursed',
    'ontime-repay': 'ontime-repay',
    'overdue': 'overdue',
    'default': 'default',
    'closed': 'closed',
  };

  let html = '<div class="timeline-line"></div>';
  events.forEach(e => {
    const tc = typeClass[e.type] || 'loan-apply';
    html += `<div class="timeline-event ${tc}">
      <div class="te-date">${e.date}</div>
      <div class="te-title">${e.title}</div>
      <div class="te-detail">${e.detail}</div>
    </div>`;
  });
  el.innerHTML = html;
}

/* ---------- 客户基本信息渲染 ---------- */
function renderCustomerInfo(profile) {
  const el = document.getElementById('customerInfoCard');
  if (!el) return;

  const employmentMap = { 0: '未知', 1: '受薪员工', 2: '自雇人士', 3: '企业主' };
  const isDefault = profile.loan_default === 1;

  el.innerHTML = `
    <div class="customer-avatar">
      <div class="avatar-icon">&#x1F464;</div>
      <div class="avatar-id">ID: ${profile.customer_id}</div>
    </div>
    <div class="customer-details">
      <div class="customer-detail-row">
        <span class="customer-detail-label">年龄:</span>
        <span class="customer-detail-value">${profile.age || '--'}岁</span>
      </div>
      <div class="customer-detail-row">
        <span class="customer-detail-label">职业:</span>
        <span class="customer-detail-value">${employmentMap[profile.employment_type] || profile.employment_type || '--'}</span>
      </div>
      <div class="customer-detail-row">
        <span class="customer-detail-label">地区:</span>
        <span class="customer-detail-value">${profile.area || '--'}</span>
      </div>
      <div class="customer-detail-row">
        <span class="customer-detail-label">信用评分:</span>
        <span class="customer-detail-value" style="color:${(profile.credit_score || 0) >= 700 ? '#34a853' : (profile.credit_score || 0) >= 550 ? '#f9ab00' : '#ea4335'};">${profile.credit_score || '--'}</span>
      </div>
      <div class="customer-detail-row">
        <span class="customer-detail-label">风险等级:</span>
        <span class="customer-detail-value"><span class="badge badge-${isDefault ? 'danger' : 'success'}">${isDefault ? '高风险' : '低风险'}</span></span>
      </div>
    </div>
  `;
}

/* ---------- 搜索客户 ---------- */
async function searchCustomer() {
  const input = document.getElementById('customerIdInput');
  const id = parseInt(input?.value || '0', 10);
  if (!id) { alert('请输入有效的客户ID'); return; }

  // 尝试从API加载
  const [profile, similar, timeline] = await Promise.all([
    API.customerProfile(id).catch(() => null),
    API.customerSimilar(id).catch(() => null),
    API.customerLoanHistory(id).catch(() => null),
  ]);

  // API 返回嵌套结构 { customer_id, profile:{...}, decision:{...}, radar_scores:{...} }
  // 将其展平为渲染函数所需的平铺对象
  let flatProfile;
  if (profile && profile.profile) {
    flatProfile = {
      customer_id: profile.customer_id,
      ...profile.profile,
      ...profile.decision,
      radar_scores: profile.radar_scores,
    };
  } else {
    // API失败时使用 mock 数据
    flatProfile = MockCustomerProfiles[id] || {
      customer_id: id,
      age: 35 + (id % 20),
      employment_type: id % 3,
      area_id: id % 10,
      credit_score: 500 + (id % 350),
      disbursed_amount: 10000 + (id % 80000),
      total_overdue_no: id % 4,
      total_account_loan_no: 1 + (id % 6),
      loan_default: id % 5 === 0 ? 1 : 0,
      loan_asset_ratio: 0.6 + (id % 40) / 100,
      recent_default_rate: (id % 5) / 10,
    };
  }

  // 时间轴：API 返回 { events:[...] }，渲染函数需要数组
  const timelineEvents = (timeline && timeline.events) ? timeline.events : (timeline || MockLoanTimeline);

  // 显示内容
  document.getElementById('profileContent').style.display = 'block';
  document.getElementById('noProfileState').style.display = 'none';

  renderCustomerInfo(flatProfile);
  renderRadarChart(flatProfile);
  renderDecisionResult(flatProfile);
  renderShapForce(flatProfile);
  renderSimilarCustomers(flatProfile, similar);
  renderTimeline(timelineEvents);

  // 调整图表大小
  setTimeout(() => {
    const radar = window._profileCharts?.radar;
    if (radar) radar.resize();
    const shap = echarts.getInstanceByDom(document.getElementById('chartShapForce'));
    if (shap) shap.resize();
  }, 100);
}

/* ---------- 随机客户 ---------- */
function loadRandomCustomer() {
  const id = Math.floor(Math.random() * 999999) + 100001;
  document.getElementById('customerIdInput').value = id;
  searchCustomer();
}

/* ---------- Enter 键搜索 ---------- */
document.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('customerIdInput');
  if (input) {
    input.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') searchCustomer();
    });
  }
});

/* ---------- 页面加载时显示随机客户 ---------- */
window.addEventListener('load', () => {
  loadRandomCustomer();
});

/* ---------- 窗口调整大小 ---------- */
window.addEventListener('resize', () => {
  const radar = window._profileCharts?.radar;
  if (radar) radar.resize();
  const shap = echarts.getInstanceByDom(document.getElementById('chartShapForce'));
  if (shap) shap.resize();
});
