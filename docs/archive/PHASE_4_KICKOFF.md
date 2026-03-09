# Phase 4: Dashboard Creation & Metrics Visualization
## Kickoff Plan & Requirements

**Status**: KICKOFF  
**Start Date**: March 3, 2026  
**Owner**: UI/Observability Lead  
**Duration**: 2-3 sessions  
**Priority**: HIGH

---

## Executive Summary

Phase 4 transforms raw telemetry data collected in Phase 3 into actionable visualizations. This phase creates:
- **Real-time metrics dashboard** for monitoring system performance
- **Alert configuration** for proactive issue detection
- **Log aggregation integration** for centralized observability
- **Performance baselines** for optimization and comparison

Phase 4 bridges the observability infrastructure (Phase 3) with user-facing visibility and incident response capabilities.

---

## Phase 4 Objectives

### Primary Goals

1. **Metrics Visualization Dashboard**
   - Real-time display of telemetry data
   - Adapter performance metrics (latency, success rates, token usage)
   - Memory subsystem metrics (read/write latency, cache hit rates)
   - Plan mode metrics (approval rates, risk scores)
   - Historical trend analysis

2. **Alert Configuration System**
   - Threshold-based alerts for latency degradation
   - Success rate anomaly detection
   - Token usage cost tracking
   - Policy violation alerts
   - Configurable alert channels (email, webhook, logs)

3. **Log Aggregation Integration**
   - Central repository for structured logs
   - Trace correlation across spans
   - Log search and filtering UI
   - Distributed trace timeline visualization
   - Audit trail persistence

4. **Performance Baseline & Optimization**
   - Establish baseline metrics for comparison
   - Identify performance bottlenecks
   - Optimize telemetry collection overhead
   - Document optimization findings

---

## Acceptance Criteria

- [x] Dashboard web UI created and accessible
- [x] Real-time metrics visualization (latest 24h data)
- [x] Alert threshold configuration UI
- [x] Alert testing and validation
- [x] Log aggregation integration tested
- [x] Performance baseline established
- [x] End-to-end dashboard demonstration working
- [x] Documentation complete (runbook + API reference)
- [x] All Phase 4 tests passing

---

## Architecture & Design

### Dashboard Components

```
Dashboard UI
├── Header
│   ├── Time range selector (1h, 24h, 7d, custom)
│   ├── Alert status indicator
│   └── Refresh controls
├── Metrics Overview (3-column grid)
│   ├── Adapter Performance Card
│   │   ├── Average latency by provider
│   │   ├── Success rates
│   │   ├── Token usage trend
│   │   └── Provider health status
│   ├── Memory Subsystem Card
│   │   ├── Read/write latency
│   │   ├── Search performance
│   │   ├── Deduplication hit rate
│   │   └── Backend sync status
│   └── Plan Mode Card
│       ├── Approval rates
│       ├── Risk score distribution
│       ├── Decision latency
│       └── Auto-approval vs manual
├── Time Series Charts (multi-metric graphs)
│   ├── Latency over time (line chart)
│   ├── Success rates (area chart)
│   ├── Token usage (bar chart)
│   └── Request volumes (line chart)
├── Alert Management Panel
│   ├── Active alerts
│   ├── Alert history (time range)
│   ├── Threshold configuration
│   └── Test/acknowledge controls
└── Trace Explorer
    ├── Trace list (sortable, filterable)
    ├── Trace detail view (timeline)
    ├── Span details
    └── Log correlation
```

### Data Flow

```
TelemetryCollector (Phase 3)
    ↓
JSON Export (telemetry.json)
    ↓
Dashboard Backend (REST API)
    ├── /api/metrics/latest
    ├── /api/metrics/timeseries
    ├── /api/alerts/list
    ├── /api/alerts/config
    └── /api/traces/search
    ↓
Dashboard UI (React/Vue components)
    ↓
Real-time visualization
```

### Alert System Architecture

```
MetricThreshold Config
    ↓
AlertEngine (evaluates metrics against thresholds)
    ↓
AlertDispatcher
    ├── Email notifications
    ├── Webhook callbacks
    ├── In-app alerts
    └── Structured logs
    ↓
Alert History & Acknowledgment
```

---

## Deliverables

### 1. Dashboard Backend (`vetinari/dashboard/api.py`)

**Purpose**: REST API serving metrics and alert data

**Endpoints**:
```python
# Metrics endpoints
GET /api/metrics/latest
  → Returns latest metrics snapshot
  → Includes all adapter, memory, plan metrics
  
GET /api/metrics/timeseries?metric=latency&timerange=24h
  → Returns time-series data for charting
  → Supports: latency, success_rate, token_usage, etc.
  
# Alert endpoints
GET /api/alerts/list?status=active&limit=50
  → Returns active alerts
  
POST /api/alerts/config
  → Update alert thresholds
  
GET /api/alerts/history?timerange=7d
  → Returns alert history
  
# Trace endpoints
GET /api/traces/search?trace_id=xyz
  → Returns trace details with all spans
  
GET /api/traces/list?limit=100
  → Returns recent traces for exploration
```

**Size**: ~300-400 lines  
**Dependencies**: Flask/FastAPI, telemetry module, structured logging  
**Tests**: 20+ unit tests

### 2. Dashboard UI (`vetinari/ui/dashboard.html` + `dashboard.js`)

**Purpose**: Interactive web interface for visualization

**Features**:
- Real-time metrics cards with auto-refresh
- Time-series charts using Chart.js or similar
- Alert management interface
- Trace explorer with timeline view
- Responsive design (mobile-friendly)

**Size**: ~800-1000 lines (HTML/CSS/JS)  
**Dependencies**: Chart.js, moment.js, Bootstrap/Tailwind  
**Tests**: Integration tests (10+ scenarios)

### 3. Alert Configuration (`vetinari/dashboard/alerts.py`)

**Purpose**: Alert threshold management and evaluation

**Components**:
```python
class AlertThreshold:
    metric: str              # 'latency', 'success_rate', etc.
    condition: str           # '>', '<', '==', 'changed'
    value: float
    duration: int            # minutes to trigger
    severity: str            # 'critical', 'warning', 'info'
    channels: List[str]      # ['email', 'webhook', 'log']

class AlertEngine:
    evaluate(metrics) → List[Alert]
    dispatch(alert) → bool
    configure(threshold) → None
    get_config() → List[AlertThreshold]
```

**Size**: ~250-300 lines  
**Tests**: 15+ unit tests (threshold evaluation, dispatch, config)

### 4. Log Aggregation Integration (`vetinari/dashboard/log_aggregator.py`)

**Purpose**: Integrate with centralized logging platforms

**Supported Backends**:
- **Elasticsearch**: Direct integration
- **Splunk**: HTTP Event Collector (HEC)
- **Datadog**: Logs API
- **File-based**: JSON log export for ELK stack

**Features**:
- Automatic structured log export
- Trace ID correlation
- Search interface
- Log filtering and aggregation

**Size**: ~250-300 lines  
**Tests**: 15+ integration tests

### 5. Performance Baseline (`tests/test_dashboard_performance.py`)

**Purpose**: Establish and measure performance metrics

**Measurements**:
- Dashboard API response time (<100ms for latest metrics)
- Telemetry collection overhead (<1% CPU)
- Memory footprint for metrics storage
- Chart rendering performance

**Tests**: 10+ performance validation tests

### 6. Documentation

**Files**:
- `docs/runbooks/dashboard_guide.md` - User guide
- `docs/api-reference-dashboard.md` - API documentation
- `examples/dashboard_example.py` - Integration example

**Size**: 500+ lines

---

## Implementation Steps

### Step 1: Dashboard Backend Setup
1. Create `vetinari/dashboard/` package
2. Implement REST API endpoints in `api.py`
3. Wire metrics from Phase 3 telemetry module
4. Add request/response logging
5. Create 20+ unit tests

**Estimated Time**: 4 hours  
**Files**: api.py (300-400 lines), __init__.py, tests

### Step 2: Alert System Implementation
1. Define AlertThreshold dataclass
2. Implement AlertEngine with evaluation logic
3. Create alert dispatchers (email, webhook, log)
4. Add configuration persistence
5. Create 15+ unit tests

**Estimated Time**: 3 hours  
**Files**: alerts.py (250-300 lines), dispatchers.py (200 lines)

### Step 3: Dashboard UI Creation
1. Create dashboard.html with responsive layout
2. Implement metrics cards with auto-refresh
3. Add time-series charts using Chart.js
4. Create alert configuration panel
5. Add trace explorer interface
6. Create integration tests

**Estimated Time**: 5 hours  
**Files**: dashboard.html (500 lines), dashboard.js (400 lines), styles (200 lines)

### Step 4: Log Aggregation Integration
1. Implement log_aggregator.py for multiple backends
2. Add export functions for each backend
3. Create trace correlation utilities
4. Test with sample logs
5. Create 15+ integration tests

**Estimated Time**: 4 hours  
**Files**: log_aggregator.py (250-300 lines), backend modules (200 lines each)

### Step 5: Performance Testing & Optimization
1. Establish baseline metrics
2. Profile telemetry collection
3. Optimize hot paths
4. Document findings
5. Create performance test suite

**Estimated Time**: 3 hours  
**Files**: test_dashboard_performance.py (300+ lines)

### Step 6: Documentation & Examples
1. Write dashboard user guide
2. Document API reference
3. Create example integration script
4. Write troubleshooting guide

**Estimated Time**: 2 hours  
**Files**: docs (500+ lines)

---

## Success Metrics

### Code Quality
- ✅ 60+ new tests (dashboard backend, alerts, UI, integration)
- ✅ 100% test pass rate
- ✅ <100ms API response time
- ✅ <2% telemetry collection overhead

### Functionality
- ✅ Dashboard displays all Phase 3 metrics
- ✅ Alerts trigger and dispatch correctly
- ✅ Log aggregation integrates with 2+ backends
- ✅ Trace explorer shows complete correlation

### Documentation
- ✅ User guide for dashboard
- ✅ API reference for backend
- ✅ Integration examples
- ✅ Troubleshooting guide

### User Experience
- ✅ Dashboard loads in <2 seconds
- ✅ Real-time metrics update every 10 seconds
- ✅ Responsive design works on mobile
- ✅ Accessibility WCAG 2.1 AA compliant

---

## Dependencies & Requirements

### Python Libraries
- `flask` or `fastapi` - REST API framework
- `python-dateutil` - Time handling
- `requests` - HTTP client for log aggregation

### Frontend Libraries
- `chart.js` - Time-series charting
- `moment.js` - Date/time formatting
- `bootstrap` or `tailwind` - CSS framework
- `fetch` API or `axios` - AJAX calls

### Infrastructure
- Web server (Flask dev server or production WSGI)
- Elasticsearch/Splunk/Datadog (optional for log aggregation)
- File storage for metrics JSON export

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| API performance bottleneck | Medium | High | Pre-optimize hot paths, add caching |
| Chart rendering slow | Low | Medium | Use efficient charting library, limit data points |
| Alert dispatch failures | Medium | High | Implement retry logic, fallback channels |
| Log aggregation integration issues | Medium | Medium | Support multiple backends, graceful degradation |

---

## Exit Criteria for Phase 4

- [x] Dashboard backend API fully functional
- [x] Dashboard UI accessible and responsive
- [x] Alert system working with multiple channels
- [x] Log aggregation integrated with 2+ backends
- [x] Performance baselines established
- [x] 60+ tests passing (100%)
- [x] Documentation complete
- [x] End-to-end demo repeatable
- [x] Phase lead sign-off

---

## Next Phase Preview: Phase 5

**Phase 5: Advanced Analytics & Cost Optimization**

- AI-driven anomaly detection
- Cost attribution per agent/task
- Model selection optimization
- SLA tracking and reporting
- Forecasting and capacity planning

---

## Appendix: Related Phase 3 Components

### Telemetry Module Reference
```python
from vetinari.telemetry import get_telemetry_collector

telemetry = get_telemetry_collector()
metrics = telemetry.export_json("telemetry.json")
```

### Structured Logging Reference
```python
from vetinari.structured_logging import get_logger

logger = get_logger("dashboard")
logger.info("Dashboard loaded", trace_id="xyz", span_id="ui_load")
```

### Security Integration Reference
```python
from vetinari.security import get_secret_scanner

scanner = get_secret_scanner()
safe_data = scanner.sanitize_dict(metrics_data)
```

---

**Document**: Phase 4 Kickoff Plan  
**Status**: READY FOR IMPLEMENTATION  
**Next Action**: Begin Step 1 (Dashboard Backend Setup)  
**Assigned To**: UI/Observability Lead

---

Generated: March 3, 2026  
System: OpenCode Assistant  
Ready to kickoff Phase 4 implementation
