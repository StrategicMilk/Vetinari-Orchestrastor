/**
 * Typed fetch wrappers for all Vetinari REST endpoints.
 *
 * Every function returns the parsed JSON response or throws on HTTP error.
 * Organized by domain: projects, models, training, memory, agents,
 * analytics, plans, rules, credentials, ADRs, skills.
 */

const API = '/api';
const API_V1 = '/api/v1';

// -- Internal helpers --------------------------------------------------------

async function request(url, options = {}) {
  // Determine whether this is a mutating method so we can attach the CSRF
  // header required by vetinari/web/csrf.py (CSRFMiddleware, ADR-0071).
  // Browsers cannot add custom headers to cross-origin requests without a
  // CORS preflight, so the presence of X-Requested-With proves same-origin.
  const method = (options.method ?? 'GET').toUpperCase();
  const csrfHeaders =
    method === 'POST' || method === 'PUT' || method === 'DELETE' || method === 'PATCH'
      ? { 'X-Requested-With': 'XMLHttpRequest' }
      : {};

  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...csrfHeaders, ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json();
}

function get(url) {
  return request(url);
}

function post(url, body) {
  return request(url, { method: 'POST', body: body != null ? JSON.stringify(body) : undefined });
}

function put(url, body) {
  return request(url, { method: 'PUT', body: JSON.stringify(body) });
}

function del(url) {
  return request(url, { method: 'DELETE' });
}

// -- Health ------------------------------------------------------------------

export function getHealth() {
  return get('/health');
}

// -- Projects ----------------------------------------------------------------

export function listProjects() {
  // GET /api/projects — the backend only exposes this as a GET (litestar_projects_api.py).
  // Previously called post() which returned 405; corrected to get().
  return get(`${API}/projects`);
}

export function createProject(config) {
  return post(`${API}/new-project`, config);
}

export function getProject(projectId) {
  return get(`${API}/project/${projectId}`);
}

export function sendMessage(projectId, message, attachments = []) {
  return post(`${API}/project/${projectId}/message`, { message, attachments });
}

export async function uploadAttachment(projectId, file) {
  const form = new FormData();
  form.append('file', file);
  form.append('project_id', projectId);
  const res = await fetch(`${API_V1}/chat/attachments`, {
    method: 'POST',
    // X-Requested-With satisfies CSRFMiddleware (vetinari/web/csrf.py, ADR-0071).
    // Content-Type is intentionally omitted — the browser sets multipart/form-data
    // with the correct boundary automatically when body is FormData.
    headers: { 'X-Requested-With': 'XMLHttpRequest' },
    body: form,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json();
}

export function getAttachmentUrl(attachmentId) {
  return `${API_V1}/chat/attachments/${attachmentId}`;
}

export function createTask(projectId, task) {
  return post(`${API}/project/${projectId}/task`, task);
}

export function updateTask(projectId, taskId, updates) {
  return put(`${API}/project/${projectId}/task/${taskId}`, updates);
}

export function deleteTask(projectId, taskId) {
  return del(`${API}/project/${projectId}/task/${taskId}`);
}

export function rerunTask(projectId, taskId) {
  return post(`${API}/project/${projectId}/task/${taskId}/rerun`);
}

export function cancelProject(projectId) {
  return post(`${API}/project/${projectId}/cancel`);
}

export function pauseProject(projectId) {
  return post(`${API}/project/${projectId}/pause`);
}

export function resumeProject(projectId) {
  return post(`${API}/project/${projectId}/resume`);
}

export function getProjectReview(projectId) {
  return get(`${API}/project/${projectId}/review`);
}

export function approveProject(projectId) {
  return post(`${API}/project/${projectId}/approve`);
}

export function archiveProject(projectId) {
  return post(`${API}/project/${projectId}/archive`);
}

export function deleteProject(projectId) {
  return del(`${API}/project/${projectId}`);
}

export function renameProject(projectId, name) {
  return post(`${API}/project/${projectId}/rename`, { name });
}

export function assembleProject(projectId) {
  return post(`${API}/project/${projectId}/assemble`);
}

// -- Models ------------------------------------------------------------------

export function listModels() {
  return get(`${API_V1}/models`);
}

export function refreshModels() {
  return post(`${API_V1}/models/refresh`);
}

export function scoreModels() {
  return post(`${API_V1}/score-models`);
}

export function getModelConfig() {
  return get(`${API_V1}/model-config`);
}

export function saveModelConfig(config) {
  return post(`${API_V1}/model-config`, config);
}

export function swapModel(modelId) {
  return post(`${API_V1}/swap-model`, { model_id: modelId });
}

export function getModelCatalog() {
  return get(`${API_V1}/model-catalog`);
}

export function getModelDetails(modelId) {
  return get(`${API_V1}/models/${modelId}`);
}

export function selectModel(modelId) {
  return post(`${API_V1}/models/select`, { model_id: modelId });
}

export function getModelPolicy() {
  return get(`${API_V1}/models/policy`);
}

export function updateModelPolicy(policy) {
  return put(`${API_V1}/models/policy`, policy);
}

export function reloadModels() {
  return post(`${API_V1}/models/reload`);
}

export function getModelFiles(repoId, { vramGb = 32, useCase = 'general' } = {}) {
  return get(`${API_V1}/models/files?repo_id=${encodeURIComponent(repoId)}&vram_gb=${vramGb}&use_case=${encodeURIComponent(useCase)}`);
}

export function downloadModel(spec) {
  return post(`${API_V1}/models/download`, spec);
}

export function discoverModels() {
  return get(`${API_V1}/discover`);
}

export function getPopularModels() {
  return get(`${API_V1}/models/popular`);
}

export function searchModels(query) {
  return post(`${API_V1}/models/search`, { query });
}

// -- Memory ------------------------------------------------------------------

export function getMemoryEntries(params = {}) {
  const qs = new URLSearchParams(params).toString();
  return get(`${API_V1}/memory${qs ? '?' + qs : ''}`);
}

export function searchMemory(query) {
  return get(`${API_V1}/memory/search?q=${encodeURIComponent(query)}`);
}

export function addMemoryEntry(entry) {
  return post(`${API_V1}/memory`, entry);
}

export function updateMemoryEntry(entryId, updates) {
  return put(`${API_V1}/memory/${entryId}`, updates);
}

export function deleteMemoryEntry(entryId) {
  return del(`${API_V1}/memory/${entryId}`);
}

export function getMemorySessions() {
  return get(`${API_V1}/memory/sessions`);
}

export function getMemoryStats() {
  return get(`${API_V1}/memory/stats`);
}

// -- Agents ------------------------------------------------------------------

export function getAgentStatus() {
  return get(`${API_V1}/agents/status`);
}

export function initializeAgents() {
  return post(`${API_V1}/agents/initialize`);
}

export function getActiveAgents() {
  return get(`${API_V1}/agents/active`);
}

export function getAgentTasks() {
  return get(`${API_V1}/agents/tasks`);
}

export function getAgentMemory() {
  return get(`${API_V1}/agents/memory`);
}

export function getPendingDecisions() {
  return get(`${API_V1}/decisions/pending`);
}

export function submitDecision(decision) {
  return post(`${API_V1}/decisions`, decision);
}

// -- Training ----------------------------------------------------------------

export function startTraining(config) {
  return post(`${API_V1}/training/start`, config);
}

export function pauseTraining() {
  return post(`${API_V1}/training/pause`);
}

export function stopTraining() {
  return post(`${API_V1}/training/stop`);
}

export function dryRunTraining(config) {
  return post(`${API_V1}/training/dry-run`, config);
}

export function setTrainingRules(rules) {
  return post(`${API_V1}/training/rules`, rules);
}

export function syncTrainingData() {
  return post(`${API_V1}/training/sync-data`);
}

export function generateSyntheticData(config) {
  return post(`${API_V1}/training/generate-synthetic`, config);
}

export function getTrainingStatus() {
  return get(`${API_V1}/training/status`);
}

export function getTrainingHistory() {
  return get(`${API_V1}/training/history`);
}

export function getIdleTrainingStats() {
  return get(`${API_V1}/training/idle-stats`);
}

// -- Plans -------------------------------------------------------------------

export function createPlan(plan) {
  return post(`${API_V1}/plan`, plan);
}

export function listPlans() {
  return get(`${API_V1}/plans`);
}

export function getPlan(planId) {
  return get(`${API_V1}/plans/${planId}`);
}

export function updatePlan(planId, updates) {
  return put(`${API_V1}/plans/${planId}`, updates);
}

export function deletePlan(planId) {
  return del(`${API_V1}/plans/${planId}`);
}

export function startPlan(planId) {
  return post(`${API_V1}/plans/${planId}/start`);
}

export function pausePlan(planId) {
  return post(`${API_V1}/plans/${planId}/pause`);
}

export function resumePlan(planId) {
  return post(`${API_V1}/plans/${planId}/resume`);
}

export function cancelPlan(planId) {
  return post(`${API_V1}/plans/${planId}/cancel`);
}

export function getPlanStatus(planId) {
  return get(`${API_V1}/plans/${planId}/status`);
}

export function getDecompositionTemplates() {
  return get(`${API_V1}/decomposition/templates`);
}

export function getDodDor() {
  return get(`${API_V1}/decomposition/dod-dor`);
}

export function decompose(task) {
  return post(`${API_V1}/decomposition/decompose`, task);
}

export function decomposeWithAgent(task) {
  return post(`${API_V1}/decomposition/decompose-agent`, task);
}

export function getDecompositionKnobs() {
  return get(`${API_V1}/decomposition/knobs`);
}

export function getDecompositionHistory() {
  return get(`${API_V1}/decomposition/history`);
}

// -- Rules -------------------------------------------------------------------

export function getRules() {
  return get(`${API_V1}/rules`);
}

/**
 * Persist a global system prompt that applies to all agents at runtime.
 *
 * Wraps the private post() helper so the shared request() function
 * automatically adds the X-Requested-With: XMLHttpRequest header required by
 * CSRFMiddleware (vetinari/web/csrf.py, ADR-0071). A raw fetch without that
 * header returns 403 and silently discards the save.
 *
 * @param {string} prompt - The global system prompt text to persist.
 * @returns {Promise} Resolves with the server response.
 */
export function saveGlobalPrompt(prompt) {
  return post('/api/v1/rules/global-prompt', { prompt });
}

// -- Receipts (Control Center / Attention track) ----------------------------

/**
 * List a project's WorkReceipts with optional filtering and pagination.
 *
 * Wraps GET /api/projects/{project_id}/receipts. The Control Center calls
 * this on initial page load to populate per-project counts before
 * subscribing to the SSE stream.
 *
 * @param {string} projectId - Project identifier.
 * @param {object} [options] - Optional filters.
 * @param {string} [options.kind] - WorkReceiptKind filter (e.g. "worker_task").
 * @param {boolean} [options.awaiting] - True/false to filter by awaiting_user.
 * @param {string} [options.since] - ISO-8601 lower bound on finished_at_utc.
 * @param {number} [options.limit=100] - Page size.
 * @param {number} [options.offset=0] - Page offset.
 * @returns {Promise} Resolves with `{project_id, total, offset, limit, receipts}`.
 */
export function listProjectReceipts(projectId, options = {}) {
  const params = new URLSearchParams();
  if (options.kind != null) params.set('kind', options.kind);
  if (options.awaiting != null) params.set('awaiting', String(options.awaiting));
  if (options.since != null) params.set('since', options.since);
  if (options.limit != null) params.set('limit', String(options.limit));
  if (options.offset != null) params.set('offset', String(options.offset));
  const qs = params.toString();
  return get(`${API}/projects/${encodeURIComponent(projectId)}/receipts${qs ? `?${qs}` : ''}`);
}

/**
 * List awaiting receipts across all projects for the Attention track.
 *
 * Wraps GET /api/attention. Each item carries the structured
 * ``awaiting_reason`` set by the Foreman/Inspector at the time the user
 * block was raised — never synthesised on the client.
 *
 * @returns {Promise} Resolves with `{count, items}`.
 */
export function listAttention() {
  return get(`${API}/attention`);
}

// -- Credentials -------------------------------------------------------------

export function listCredentials() {
  return get(`${API}/admin/credentials`);
}

export function setCredentials(sourceType, creds) {
  return post(`${API}/admin/credentials/${sourceType}`, creds);
}

export function rotateCredentials(sourceType) {
  return post(`${API}/admin/credentials/${sourceType}/rotate`);
}

export function deleteCredentials(sourceType) {
  return del(`${API}/admin/credentials/${sourceType}`);
}

export function getCredentialHealth() {
  return get(`${API}/admin/credentials/health`);
}

// -- ADRs --------------------------------------------------------------------

export function listAdrs() {
  return get(`${API}/adr`);
}

export function getAdr(adrId) {
  return get(`${API}/adr/${adrId}`);
}

export function createAdr(adr) {
  return post(`${API}/adr`, adr);
}

export function updateAdr(adrId, updates) {
  return put(`${API}/adr/${adrId}`, updates);
}

export function deprecateAdr(adrId) {
  return post(`${API}/adr/${adrId}/deprecate`);
}

export function getAdrStatistics() {
  return get(`${API}/adr/statistics`);
}

// -- Analytics ---------------------------------------------------------------

export function getAnalyticsOverview() {
  return get(`${API_V1}/analytics/overview`);
}

export function getAnalyticsAdapters() {
  return get(`${API_V1}/analytics/adapters`);
}

export function getAnalyticsMemory() {
  return get(`${API_V1}/analytics/memory`);
}

export function getAnalyticsPlan() {
  return get(`${API_V1}/analytics/plan`);
}

export function getAnalyticsTraces() {
  return get(`${API_V1}/traces`);
}

export function getAnalyticsCost() {
  return get(`${API_V1}/analytics/cost`);
}

export function getAnalyticsSla() {
  return get(`${API_V1}/analytics/sla`);
}

export function getAnalyticsAnomalies() {
  return get(`${API_V1}/analytics/anomalies`);
}

export function getAnalyticsForecast() {
  return get(`${API_V1}/analytics/forecasts`);
}

export function getAnalyticsAlerts() {
  return get(`${API_V1}/analytics/alerts`);
}

// -- Ponder (Planning Service) -----------------------------------------------

export function getPonderModels() {
  return get(`${API}/ponder/models`);
}

export function getPonderTemplates() {
  return get(`${API}/ponder/templates`);
}

export function getPonderHealth() {
  return get(`${API}/ponder/health`);
}
