/* ================================================
   API 调用层 - 统一封装所有后端接口
   ================================================ */

const API = {
  /* ---------- 基础配置 ---------- */
  baseUrl: '',

  /* ---------- 通用请求 ---------- */
  async request(url, options = {}) {
    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error(`[API] ${options.method || 'GET'} ${url} failed:`, error);
      return null;
    }
  },

  get(url) {
    return this.request(url, { method: 'GET' });
  },

  post(url, body) {
    return this.request(url, {
      method: 'POST',
      body: JSON.stringify(body),
    });
  },

  /* ---------- 健康检查 ---------- */
  async health() {
    return this.get('/health');
  },

  /* ---------- 统计分析 ---------- */
  async statsOverview() {
    return this.get('/stats/overview');
  },

  async statsRiskDaily() {
    return this.get('/stats/risk_daily');
  },

  async statsRiskDistribution() {
    return this.get('/stats/risk_distribution');
  },

  async statsModelMetrics() {
    return this.get('/stats/model_metrics');
  },

  async statsAreaRisk() {
    return this.get('/stats/area_risk');
  },

  async statsCustomerCluster() {
    return this.get('/stats/customer_cluster');
  },

  async statsCreditScoreDist() {
    return this.get('/stats/credit_score_dist');
  },

  async statsRecentDecisions() {
    return this.get('/stats/recent_decisions');
  },

  /* ---------- 预测接口 ---------- */
  async predictDefault(records) {
    return this.post('/predict/default', records);
  },

  async predictFraud(records) {
    return this.post('/predict/fraud', records);
  },

  async predictLimit(records) {
    return this.post('/predict/limit', records);
  },

  async predictFull(records) {
    return this.post('/predict/full', records);
  },

  /* ---------- 客户画像 ---------- */
  async customerProfile(customerId) {
    // 区分 404（客户不存在）与其他错误，让前端能弹"客户不存在"
    try {
      const response = await fetch(`/customer/${customerId}/profile`);
      if (response.status === 404) {
        return { __not_found: true, customer_id: customerId };
      }
      if (!response.ok) return null;
      return await response.json();
    } catch (e) {
      console.error(`[API] GET /customer/${customerId}/profile failed:`, e);
      return null;
    }
  },

  async customerRandomId() {
    return this.get('/customer/random_id');
  },

  async customerSimilar(customerId) {
    return this.get(`/customer/${customerId}/similar`);
  },

  async customerLoanHistory(customerId) {
    return this.get(`/customer/${customerId}/loan_history`);
  },

  /* ---------- 模型解释 ---------- */
  async modelShapValues() {
    return this.get('/model/shap_values');
  },

  async modelRegistry() {
    return this.get('/model/registry');
  },

  /* ---------- 修复模块 ---------- */
  async repairRecord(record) {
    return this.post('/repair/record', record);
  },

  async repairEvaluation() {
    return this.get('/repair/evaluation');
  },
};

/* ---------- 数据模拟（无后端时提供示例数据） ---------- */
const MockData = {
  overview: {
    total_customers: 2263847,
    total_amount: 15824000000,
    overdue_rate: 0.0582,
    new_customers: 12458,
  },

  riskDistribution: [
    { name: '低风险', value: 70 },
    { name: '中风险', value: 20 },
    { name: '高风险', value: 10 },
  ],

  riskDaily: (() => {
    const months = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月'];
    const rates = [6.8, 6.5, 6.2, 6.0, 5.9, 5.7, 5.8, 5.6, 5.5, 5.4, 5.3, 5.2];
    return months.map((m, i) => ({ dt: m, default_rate: rates[i] / 100, total: 150000 + i * 5000 }));
  })(),

  areaRisk: [
    { area: '华东区', rate: 0.038 },
    { area: '华北区', rate: 0.052 },
    { area: '华南区', rate: 0.045 },
    { area: '华西区', rate: 0.068 },
    { area: '华中区', rate: 0.055 },
  ],

  customerCluster: (() => {
    const clusters = [
      { name: '高信用高额度', color: '#34a853', count: 339577 },
      { name: '中信用中额度', color: '#1a73e8', count: 905535 },
      { name: '中信用低额度', color: '#f9ab00', count: 565962 },
      { name: '低信用中额度', color: '#f57c00', count: 339577 },
      { name: '低信用高风险', color: '#ea4335', count: 113196 },
    ];
    const scatterData = [];
    clusters.forEach((c, ci) => {
      const baseX = [750, 700, 680, 600, 520][ci];
      const baseY = [50000, 35000, 15000, 25000, 20000][ci];
      for (let i = 0; i < Math.min(c.count / 1000, 80); i++) {
        scatterData.push([baseX + (Math.random() - 0.5) * 60, baseY + (Math.random() - 0.5) * 8000, c.name]);
      }
    });
    return { clusters, scatterData };
  })(),

  creditScoreDist: (() => {
    const buckets = ['300-400', '400-500', '500-600', '600-700', '700-800', '800-850'];
    const counts = [45000, 180000, 680000, 800000, 450000, 110000];
    return { buckets, counts };
  })(),

  modelMetrics: {
    auc: 0.87,
    precision: 0.82,
    recall: 0.79,
    f1: 0.80,
  },

  shapImportance: [
    { name: 'credit_score', mean_abs_shap: 4.52, display: '信用评分' },
    { name: 'total_overdue_no', mean_abs_shap: 3.87, display: '总逾期次数' },
    { name: 'outstanding_disburse_ratio', mean_abs_shap: 3.21, display: '未偿发放比' },
    { name: 'ltv_ratio', mean_abs_shap: 2.95, display: '贷款资产比' },
    { name: 'overdue_rate_total', mean_abs_shap: 2.68, display: '总逾期率' },
    { name: 'credit_history', mean_abs_shap: 2.34, display: '信用记录时长' },
    { name: 'enquirie_no', mean_abs_shap: 2.01, display: '征信查询次数' },
    { name: 'disbursed_amount', mean_abs_shap: 1.87, display: '贷款金额' },
    { name: 'age', mean_abs_shap: 1.65, display: '年龄' },
    { name: 'total_monthly_payment', mean_abs_shap: 1.43, display: '月供金额' },
  ],

  systemMetrics: {
    api_calls: 1245893,
    avg_latency_ms: 47,
    repair_success_rate: 0.82,
  },

  recentDecisions: [
    { customer_id: 100001, default_probability: 0.12, credit_score: 742, predicted_limit: 45000, fraud_probability: 0.03, created_at: '2026-04-04 10:23' },
    { customer_id: 100002, default_probability: 0.67, credit_score: 485, predicted_limit: 12000, fraud_probability: 0.15, created_at: '2026-04-04 10:22' },
    { customer_id: 100003, default_probability: 0.21, credit_score: 698, predicted_limit: 28000, fraud_probability: 0.05, created_at: '2026-04-04 10:21' },
    { customer_id: 100004, default_probability: 0.08, credit_score: 785, predicted_limit: 65000, fraud_probability: 0.01, created_at: '2026-04-04 10:20' },
    { customer_id: 100005, default_probability: 0.45, credit_score: 562, predicted_limit: 18000, fraud_probability: 0.22, created_at: '2026-04-04 10:19' },
  ],
};

window.API = API;
window.MockData = MockData;
