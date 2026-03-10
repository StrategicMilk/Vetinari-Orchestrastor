/**
 * Vetinari Monitoring Dashboard — dashboard.js
 *
 * Responsibilities:
 *   - Fetch metrics, timeseries, traces and alerts from the REST API
 *   - Render KPI cards, Chart.js charts, data tables, alert panels
 *   - Auto-refresh at a configurable interval
 *   - Sidebar navigation between sections
 *   - Trace search and detail modal
 */

'use strict';

/* ──────────────────────────────────────────────────────────────
   Configuration
   ────────────────────────────────────────────────────────────── */
const API_BASE = '/api/v1';

/* ──────────────────────────────────────────────────────────────
   State
   ────────────────────────────────────────────────────────────── */
let _refreshTimer   = null;
let _charts         = {};          // keyed by canvas id
let _activeAlerts   = [];
let _alertHistory   = [];          // client-side accumulation

/* ──────────────────────────────────────────────────────────────
   DOM helpers
   ────────────────────────────────────────────────────────────── */
const $  = (id) => document.getElementById(id);
const fmtMs  = (v) => v == null ? '—' : `${(+v).toFixed(1)} ms`;
const fmtPct = (v) => v == null ? '—' : `${(+v).toFixed(1)}%`;
const fmtNum = (v) => v == null ? '—' : (+v).toLocaleString();
const fmtTs  = (iso) => {
    if (!iso) return '—';
    try { return new Date(iso).toLocaleTimeString(); } catch { return iso; }
};
const fmtUnixTs = (epoch) => {
    if (epoch == null) return '—';
    try { return new Date(epoch * 1000).toLocaleTimeString(); } catch { return String(epoch); }
};

function pill(text, type) {
    return `<span class="dash-pill dash-pill-${type}">${text}</span>`;
}

function statusPill(status) {
    const map = { success: 'success', error: 'error', in_progress: 'info' };
    return pill(status, map[status] || 'info');
}

/* ──────────────────────────────────────────────────────────────
   API calls
   ────────────────────────────────────────────────────────────── */
async function apiFetch(path) {
    const res = await fetch(`${API_BASE}${path}`);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
}

/* ──────────────────────────────────────────────────────────────
   Status dot
   ────────────────────────────────────────────────────────────── */
function setStatus(online) {
    const dot   = $('statusDot');
    const label = $('statusLabel');
    dot.className   = `dash-status-dot ${online ? 'online' : 'offline'}`;
    label.textContent = online ? 'Connected' : 'Offline';
}

/* ──────────────────────────────────────────────────────────────
   Charts
   ────────────────────────────────────────────────────────────── */
const CHART_DEFAULTS = {
    type: 'line',
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        plugins: { legend: { display: false } },
        scales: {
            x: { ticks: { color: '#7a8491', maxTicksLimit: 6 }, grid: { color: 'rgba(255,255,255,0.04)' } },
            y: { ticks: { color: '#7a8491' }, grid: { color: 'rgba(255,255,255,0.06)' } }
        }
    }
};

function makeChart(canvasId, label, colour) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    if (_charts[canvasId]) { _charts[canvasId].destroy(); }
    const cfg = JSON.parse(JSON.stringify(CHART_DEFAULTS));
    cfg.data = {
        labels: [],
        datasets: [{
            label,
            data: [],
            borderColor: colour,
            backgroundColor: colour.replace(')', ', 0.1)').replace('rgb', 'rgba'),
            borderWidth: 2,
            pointRadius: 3,
            tension: 0.35,
            fill: true
        }]
    };
    _charts[canvasId] = new Chart(ctx, cfg);
    return _charts[canvasId];
}

function updateChart(canvasId, timeseries) {
    const chart = _charts[canvasId];
    if (!chart || !timeseries || !timeseries.points) return;
    chart.data.labels   = timeseries.points.map(p => fmtTs(p.timestamp));
    chart.data.datasets[0].data = timeseries.points.map(p => p.value);
    chart.update('none');
}

function initCharts() {
    makeChart('latencyChart',    'Latency (ms)',   'rgb(78, 154, 249)');
    makeChart('successChart',    'Success (%)',    'rgb(56, 211, 159)');
    makeChart('tokenChart',      'Tokens',         'rgb(167, 139, 250)');
    makeChart('memLatencyChart', 'Mem Latency (ms)','rgb(33, 212, 253)');
}

/* ──────────────────────────────────────────────────────────────
   Latest metrics → KPI cards + tables
   ────────────────────────────────────────────────────────────── */
function renderOverviewKPIs(snap) {
    const a = snap.adapters || {};
    const p = snap.plan     || {};

    $('kpi-total-requests').textContent = fmtNum(a.total_requests);
    $('kpi-avg-latency').textContent    = fmtMs(a.average_latency_ms);
    $('kpi-tokens').textContent         = fmtNum(a.total_tokens_used);

    const total = (a.total_successful || 0) + (a.total_failed || 0);
    const rate  = total > 0 ? ((a.total_successful / total) * 100).toFixed(1) : '—';
    $('kpi-success-rate').textContent   = total > 0 ? `${rate}%` : '—';
    $('kpi-approval-rate').textContent  = fmtPct(p.approval_rate);

    $('kpi-active-alerts').textContent  = _activeAlerts.length;
    const badge = $('alertBadge');
    if (_activeAlerts.length > 0) {
        badge.textContent = _activeAlerts.length;
        badge.style.display = '';
    } else {
        badge.style.display = 'none';
    }

    const uptime = snap.uptime_ms != null
        ? `Uptime: ${(snap.uptime_ms / 1000).toFixed(0)} s`
        : '';
    $('uptimeLabel').textContent = uptime;
}

function renderAdapterTable(snap) {
    const providers = (snap.adapters || {}).providers || {};
    const providerKeys = Object.keys(providers);

    // Populate provider filter dropdown
    const sel = $('latencyProviderFilter');
    const current = sel.value;
    // Clear existing options except first
    while (sel.options.length > 1) sel.remove(1);
    providerKeys.forEach(k => {
        const opt = document.createElement('option');
        opt.value = k; opt.textContent = k;
        sel.appendChild(opt);
    });
    if (current && providerKeys.includes(current)) sel.value = current;

    const body = $('adapterTableBody');
    if (!providerKeys.length) {
        body.innerHTML = '<tr><td colspan="8" class="dash-empty">No adapter data yet.</td></tr>';
        return;
    }
    body.innerHTML = providerKeys.map(k => {
        const p = providers[k];
        return `<tr>
            <td>${p.provider || k}</td>
            <td><code>${p.model || '—'}</code></td>
            <td>${fmtNum(p.requests)}</td>
            <td>${fmtPct(p.success_rate)}</td>
            <td>${fmtMs(p.avg_latency_ms)}</td>
            <td>${fmtMs(p.min_latency_ms)} / ${fmtMs(p.max_latency_ms)}</td>
            <td>${fmtNum(p.tokens_used)}</td>
            <td>${fmtTs(p.last_request)}</td>
        </tr>`;
    }).join('');
    $('adapterTimestamp').textContent = fmtTs(snap.timestamp);
}

function renderMemoryTable(snap) {
    const backends = (snap.memory || {}).backends || {};
    const keys = Object.keys(backends);
    const body = $('memoryTableBody');
    if (!keys.length) {
        body.innerHTML = '<tr><td colspan="9" class="dash-empty">No memory data yet.</td></tr>';
        return;
    }
    body.innerHTML = keys.map(k => {
        const b = backends[k];
        return `<tr>
            <td>${b.backend || k}</td>
            <td>${fmtNum(b.writes)}</td>
            <td>${fmtNum(b.reads)}</td>
            <td>${fmtNum(b.searches)}</td>
            <td>${fmtMs(b.avg_write_latency_ms)}</td>
            <td>${fmtMs(b.avg_read_latency_ms)}</td>
            <td>${fmtMs(b.avg_search_latency_ms)}</td>
            <td>${fmtPct(b.dedup_hit_rate)}</td>
            <td>${fmtNum(b.sync_failures)}</td>
        </tr>`;
    }).join('');
}

function renderPlanKPIs(snap) {
    const p = snap.plan || {};
    $('plan-approved').textContent  = fmtNum(p.approved);
    $('plan-rejected').textContent  = fmtNum(p.rejected);
    $('plan-auto').textContent      = fmtNum(p.auto_approved);
    $('plan-rate').textContent      = fmtPct(p.approval_rate);
    $('plan-risk').textContent      = p.average_risk_score != null
        ? (+p.average_risk_score).toFixed(3) : '—';
    $('plan-time').textContent      = fmtMs(p.average_approval_time_ms);
}

/* ──────────────────────────────────────────────────────────────
   Traces
   ────────────────────────────────────────────────────────────── */
function renderTraceTable(data) {
    const traces = data.traces || [];
    $('traceCount').textContent = `${data.count || 0} trace(s)`;
    const body = $('traceTableBody');
    if (!traces.length) {
        body.innerHTML = '<tr><td colspan="7" class="dash-empty">No traces found.</td></tr>';
        return;
    }
    body.innerHTML = traces.map(t => `<tr>
        <td><code>${t.trace_id}</code></td>
        <td>${fmtTs(t.start_time)}</td>
        <td>${fmtMs(t.duration_ms)}</td>
        <td>${t.span_count}</td>
        <td>${statusPill(t.status)}</td>
        <td>${t.root_operation || '—'}</td>
        <td><button class="dash-btn dash-btn-secondary dash-btn-sm" onclick="loadTraceDetail('${t.trace_id}')">
            <i class="fas fa-eye"></i>
        </button></td>
    </tr>`).join('');
}

async function loadTraceDetail(traceId) {
    try {
        const detail = await apiFetch(`/traces/${encodeURIComponent(traceId)}`);
        showTraceModal(detail);
    } catch (e) {
        showTraceDetail(null, traceId, e.message);
    }
}

function showTraceModal(detail) {
    $('modalTraceId').textContent = detail.trace_id;
    const spans = detail.spans || [];

    // Find max duration for bar scaling
    const maxDur = Math.max(...spans.map(s => s.duration_ms || 0), 1);

    $('modalSpanList').innerHTML = spans.length
        ? `<div class="dash-trace-timeline">${spans.map(s => {
            const dur = s.duration_ms || 0;
            const pct = Math.max((dur / maxDur) * 100, 1).toFixed(1);
            return `<div class="dash-span-row">
                <div class="dash-span-label" title="${s.operation || ''}">${s.operation || s.span_id || '—'}</div>
                <div class="dash-span-bar-wrap">
                    <div class="dash-span-bar" style="width:${pct}%"></div>
                </div>
                <div class="dash-span-dur">${fmtMs(dur)}</div>
            </div>`;
          }).join('')}</div>`
        : '<div class="dash-empty">No spans in this trace.</div>';

    $('traceModal').style.display = 'flex';
}

/* ──────────────────────────────────────────────────────────────
   Alerts panel (client-side, polled from /api/v1/stats)
   Note: The backend AlertEngine is Python-only; we poll /health
   and /stats and display client-side stored alert state.
   ────────────────────────────────────────────────────────────── */
function renderAlerts() {
    const activeEl  = $('activeAlertsList');
    const historyEl = $('alertHistoryList');
    $('activeAlertCount').textContent = _activeAlerts.length;

    activeEl.innerHTML = _activeAlerts.length
        ? _activeAlerts.map(a => alertItemHtml(a, true)).join('')
        : '<div class="dash-empty">No active alerts.</div>';

    historyEl.innerHTML = _alertHistory.length
        ? [..._alertHistory].reverse().map(a => alertItemHtml(a, false)).join('')
        : '<div class="dash-empty">No alerts fired yet.</div>';
}

function alertItemHtml(a, isActive) {
    const sevMap = {
        high:   { cls: 'dash-icon-red',    icon: 'fas fa-exclamation-circle' },
        medium: { cls: 'dash-icon-yellow', icon: 'fas fa-exclamation-triangle' },
        low:    { cls: 'dash-icon-blue',   icon: 'fas fa-info-circle' }
    };
    const sev = sevMap[a.severity] || sevMap.medium;
    return `<div class="dash-alert-item">
        <div class="dash-alert-severity ${sev.cls}"><i class="${sev.icon}"></i></div>
        <div class="dash-alert-body">
            <div class="dash-alert-name">${a.name}</div>
            <div class="dash-alert-detail">${a.metric_key} = ${(+a.current_value).toFixed(3)} (threshold ${a.condition} ${a.threshold_value})</div>
        </div>
        <div class="dash-alert-time">${new Date(a.trigger_time * 1000).toLocaleTimeString()}</div>
    </div>`;
}

/* ──────────────────────────────────────────────────────────────
   Analytics: Cost, SLA, Anomalies, Forecast
   ────────────────────────────────────────────────────────────── */
function renderCostSection(report, topData) {
    $('cost-total').textContent = report.total_cost_usd != null
        ? `$${(+report.total_cost_usd).toFixed(4)}` : '—';
    $('cost-tokens').textContent = fmtNum(report.total_tokens);
    $('cost-requests').textContent = fmtNum(report.total_requests);

    const agentBody = $('costAgentTableBody');
    const agents = (topData && topData.top_agents) || [];
    agentBody.innerHTML = agents.length
        ? agents.map(a => `<tr><td>${a.agent || '(unknown)'}</td><td>$${(+a.cost_usd).toFixed(4)}</td></tr>`).join('')
        : '<tr><td colspan="2" class="dash-empty">No agent cost data.</td></tr>';

    const modelBody = $('costModelTableBody');
    const models = (topData && topData.top_models) || [];
    modelBody.innerHTML = models.length
        ? models.map(m => `<tr><td><code>${m.model}</code></td><td>$${(+m.cost_usd).toFixed(4)}</td></tr>`).join('')
        : '<tr><td colspan="2" class="dash-empty">No model cost data.</td></tr>';
}

function renderSLASection(reports) {
    const body = $('slaTableBody');
    if (!reports || !reports.length) {
        body.innerHTML = '<tr><td colspan="7" class="dash-empty">No SLA data yet.</td></tr>';
        return;
    }
    body.innerHTML = reports.map(r => {
        const slo = r.slo || {};
        const compliant = r.is_compliant != null ? r.is_compliant : (r.compliance_pct >= (slo.budget || 0));
        const statusCls = compliant ? 'success' : 'error';
        return `<tr>
            <td>${slo.name || '—'}</td>
            <td>${slo.slo_type || '—'}</td>
            <td>${slo.budget != null ? (+slo.budget).toFixed(1) : '—'}</td>
            <td>${r.current_value != null ? (+r.current_value).toFixed(2) : '—'}</td>
            <td>${fmtPct(r.compliance_pct)}</td>
            <td>${fmtNum(r.total_samples)}</td>
            <td>${pill(compliant ? 'OK' : 'BREACH', statusCls)}</td>
        </tr>`;
    }).join('');
}

function renderAnomalySection(data) {
    const anomalies = data.anomalies || [];
    $('anomalyCount').textContent = `${anomalies.length} anomaly(s)`;
    const body = $('anomalyTableBody');
    if (!anomalies.length) {
        body.innerHTML = '<tr><td colspan="6" class="dash-empty">No anomalies detected.</td></tr>';
        return;
    }
    body.innerHTML = anomalies.map(a => `<tr>
        <td>${fmtUnixTs(a.timestamp)}</td>
        <td>${a.metric}</td>
        <td>${(+a.value).toFixed(2)}</td>
        <td>${a.method || '—'}</td>
        <td>${(+a.score).toFixed(2)}</td>
        <td>${a.reason || '—'}</td>
    </tr>`).join('');
}

function renderForecastSection(result) {
    const preds = result.predictions || [];
    const lo = result.confidence_lo || [];
    const hi = result.confidence_hi || [];
    const body = $('forecastTableBody');
    if (!preds.length) {
        body.innerHTML = '<tr><td colspan="4" class="dash-empty">No forecast data.</td></tr>';
        $('forecastMeta').textContent = '';
        return;
    }
    body.innerHTML = preds.map((p, i) => `<tr>
        <td>${i + 1}</td>
        <td>${(+p).toFixed(2)}</td>
        <td>${lo[i] != null ? (+lo[i]).toFixed(2) : '—'}</td>
        <td>${hi[i] != null ? (+hi[i]).toFixed(2) : '—'}</td>
    </tr>`).join('');
    $('forecastMeta').textContent = `Method: ${result.method || '—'} | RMSE: ${
        result.rmse != null ? (+result.rmse).toFixed(2) : '—'} | Samples: ${
        fmtNum(result.samples_used)} | Slope: ${
        result.trend_slope != null ? (+result.trend_slope).toFixed(4) : '—'}`;
}

async function runForecast() {
    const metric = $('forecastMetric').value.trim();
    const method = $('forecastMethod').value;
    if (!metric) return;
    try {
        const result = await apiFetch(`/analytics/forecast?metric=${encodeURIComponent(metric)}&method=${encodeURIComponent(method)}&horizon=10`);
        renderForecastSection(result);
    } catch (e) {
        $('forecastTableBody').innerHTML = `<tr><td colspan="4" class="dash-empty">${e.message}</td></tr>`;
        $('forecastMeta').textContent = '';
    }
}

async function populateForecastMetrics() {
    try {
        const data = await apiFetch('/analytics/forecast?metric=latency_p95&horizon=1');
        const sel = $('forecastMetric');
        if (!sel || !data) return;
        const available = data.available_metrics || [];
        if (!available.length) return;
        const current = sel.value;
        sel.innerHTML = '';
        available.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m; opt.textContent = m;
            sel.appendChild(opt);
        });
        if (current && available.includes(current)) sel.value = current;
    } catch { /* ignore — forecast endpoint may not expose available_metrics */ }
}

/* ──────────────────────────────────────────────────────────────
   Main refresh cycle
   ────────────────────────────────────────────────────────────── */
async function refresh() {
    const icon = $('refreshIcon');
    icon.classList.add('dash-spinning');

    try {
        // 1. Health check
        await apiFetch('/health');
        setStatus(true);

        // 2. Latest metrics
        const snap = await apiFetch('/metrics/latest');
        renderOverviewKPIs(snap);
        renderAdapterTable(snap);
        renderMemoryTable(snap);
        renderPlanKPIs(snap);

        // 3. Time-series charts
        const provider = $('latencyProviderFilter').value || undefined;
        const providerQ = provider ? `&provider=${encodeURIComponent(provider)}` : '';

        const [latTS, sucTS, tokTS, memTS] = await Promise.allSettled([
            apiFetch(`/metrics/timeseries?metric=latency${providerQ}`),
            apiFetch('/metrics/timeseries?metric=success_rate'),
            apiFetch('/metrics/timeseries?metric=token_usage'),
            apiFetch('/metrics/timeseries?metric=memory_latency'),
        ]);

        if (latTS.status === 'fulfilled') updateChart('latencyChart',    latTS.value);
        if (sucTS.status === 'fulfilled') updateChart('successChart',    sucTS.value);
        if (tokTS.status === 'fulfilled') updateChart('tokenChart',      tokTS.value);
        if (memTS.status === 'fulfilled') updateChart('memLatencyChart', memTS.value);

        // 4. Traces (last 50)
        const traceData = await apiFetch('/traces?limit=50');
        renderTraceTable(traceData);

        renderAlerts();

        // 5. Analytics sections (non-blocking — failures are silently ignored)
        Promise.allSettled([
            apiFetch('/analytics/cost'),
            apiFetch('/analytics/cost/top'),
        ]).then(([costRes, topRes]) => {
            if (costRes.status === 'fulfilled' && topRes.status === 'fulfilled') {
                renderCostSection(costRes.value, topRes.value);
            }
        });

        apiFetch('/analytics/sla').then(data => {
            renderSLASection(data.reports || []);
        }).catch(() => {});

        apiFetch('/analytics/anomalies').then(data => {
            renderAnomalySection(data);
        }).catch(() => {});

        populateForecastMetrics();

    } catch (e) {
        setStatus(false);
        console.warn('Dashboard refresh error:', e);
    } finally {
        icon.classList.remove('dash-spinning');
    }
}

/* ──────────────────────────────────────────────────────────────
   Auto-refresh scheduling
   ────────────────────────────────────────────────────────────── */
function scheduleRefresh() {
    clearInterval(_refreshTimer);
    const toggle = $('autoRefreshToggle');
    if (!toggle.checked) return;
    const ms = parseInt($('refreshInterval').value, 10) || 15000;
    _refreshTimer = setInterval(refresh, ms);
}

/* ──────────────────────────────────────────────────────────────
   Navigation
   ────────────────────────────────────────────────────────────── */
function switchSection(name) {
    document.querySelectorAll('.dash-section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.dash-nav-item').forEach(a => a.classList.remove('active'));

    const sec = document.getElementById(`section-${name}`);
    if (sec) sec.classList.add('active');

    const item = document.querySelector(`.dash-nav-item[data-section="${name}"]`);
    if (item) item.classList.add('active');

    const titles = {
        overview: 'Overview', adapters: 'Adapters', memory: 'Memory',
        plan: 'Plan Gate', traces: 'Traces', alerts: 'Alerts',
        cost: 'Cost', sla: 'SLA', anomalies: 'Anomalies', forecast: 'Forecast'
    };
    $('pageTitle').textContent = titles[name] || name;
}

/* ──────────────────────────────────────────────────────────────
   Event bindings
   ────────────────────────────────────────────────────────────── */
function bindEvents() {
    // Sidebar nav
    document.querySelectorAll('.dash-nav-item').forEach(el => {
        el.addEventListener('click', (e) => {
            e.preventDefault();
            switchSection(el.dataset.section);
        });
    });

    // Refresh controls
    $('refreshNowBtn').addEventListener('click', refresh);
    $('autoRefreshToggle').addEventListener('change', scheduleRefresh);
    $('refreshInterval').addEventListener('change', scheduleRefresh);

    // Provider filter changes → re-fetch charts
    $('latencyProviderFilter').addEventListener('change', refresh);

    // Trace search
    $('traceSearchBtn').addEventListener('click', async () => {
        const id = $('traceSearchInput').value.trim();
        if (!id) return;
        try {
            const data = await apiFetch(`/traces?trace_id=${encodeURIComponent(id)}`);
            renderTraceTable(data);
        } catch (e) {
            $('traceTableBody').innerHTML = `<tr><td colspan="7" class="dash-empty">${e.message}</td></tr>`;
        }
    });

    $('traceClearBtn').addEventListener('click', async () => {
        $('traceSearchInput').value = '';
        try {
            const data = await apiFetch('/traces?limit=50');
            renderTraceTable(data);
        } catch { /* ignore */ }
    });

    // Trace detail close
    $('traceDetailClose') && $('traceDetailClose').addEventListener('click', () => {
        $('traceDetailPanel').style.display = 'none';
    });

    // Trace modal close
    $('traceModalClose').addEventListener('click', () => {
        $('traceModal').style.display = 'none';
    });
    $('traceModal').addEventListener('click', (e) => {
        if (e.target === $('traceModal')) $('traceModal').style.display = 'none';
    });

    // Forecast run button
    const forecastBtn = $('forecastRunBtn');
    if (forecastBtn) forecastBtn.addEventListener('click', runForecast);

    // Clear alert history (client-side)
    $('clearAlertHistory').addEventListener('click', () => {
        _alertHistory = [];
        renderAlerts();
    });

    // Keyboard: Escape closes modal
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') $('traceModal').style.display = 'none';
    });
}

/* ──────────────────────────────────────────────────────────────
   Boot
   ────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    bindEvents();
    refresh();
    scheduleRefresh();
});
