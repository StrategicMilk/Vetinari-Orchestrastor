# Vetinari Dashboard — API Reference

**Phase 4 | Last Updated: March 2026**

---

## Table of Contents

1. [Python API](#python-api)
   - [DashboardAPI](#dashboardapi)
   - [AlertEngine](#alertengine)
   - [LogAggregator](#logaggregator)
2. [REST API](#rest-api)
   - [GET /dashboard](#get-dashboard)
   - [GET /api/v1/health](#get-apiv1health)
   - [GET /api/v1/stats](#get-apiv1stats)
   - [GET /api/v1/metrics/latest](#get-apiv1metricslatest)
   - [GET /api/v1/metrics/timeseries](#get-apiv1metricstimeseries)
   - [GET /api/v1/traces](#get-apiv1traces)
   - [GET /api/v1/traces/\<trace_id\>](#get-apiv1tracestrace_id)
3. [Data Schemas](#data-schemas)

---

## Python API

### DashboardAPI

```python
from vetinari.dashboard.api import get_dashboard_api, reset_dashboard
```

**Singleton access**

| Function | Returns | Description |
|---|---|---|
| `get_dashboard_api()` | `DashboardAPI` | Return (or create) the global singleton |
| `reset_dashboard()` | `None` | Destroy the singleton (mainly for testing) |

---

#### `DashboardAPI.get_latest_metrics() → MetricsSnapshot`

Return the current snapshot of all telemetry metrics.

**Returns** — `MetricsSnapshot`

| Field | Type | Description |
|---|---|---|
| `timestamp` | `str` | ISO-8601 UTC timestamp |
| `uptime_ms` | `float` | Milliseconds since API initialisation |
| `adapter_summary` | `dict` | Aggregated adapter metrics (see below) |
| `memory_summary` | `dict` | Per-backend memory metrics |
| `plan_summary` | `dict` | Plan-gate decision metrics |

`adapter_summary` keys:

```
total_providers       int
total_requests        int
total_successful      int
total_failed          int
average_latency_ms    float
total_tokens_used     int
providers             dict[str, ProviderDetail]
```

`plan_summary` keys:

```
total_decisions             int
approved                    int
rejected                    int
auto_approved               int
approval_rate               float  (%)
average_risk_score          float  (0.0–1.0)
average_approval_time_ms    float
```

---

#### `DashboardAPI.get_timeseries_data(metric, timerange, provider) → TimeSeriesData | None`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `metric` | `str` | required | `latency`, `success_rate`, `token_usage`, `memory_latency` |
| `timerange` | `str` | `"24h"` | `"1h"`, `"24h"`, `"7d"` |
| `provider` | `str \| None` | `None` | Filter to a specific adapter provider key |

**Returns** — `TimeSeriesData`

| Field | Type | Description |
|---|---|---|
| `metric` | `str` | Metric name |
| `unit` | `str` | `ms`, `%`, `tokens` |
| `points` | `List[TimeSeriesPoint]` | Data points |
| `min_value` | `float` | Minimum value across points |
| `max_value` | `float` | Maximum value across points |
| `avg_value` | `float` | Average value across points |

`TimeSeriesPoint` fields: `timestamp` (ISO str), `value` (float), `metadata` (dict).

---

#### `DashboardAPI.search_traces(trace_id, limit) → List[TraceInfo]`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `trace_id` | `str \| None` | `None` | Exact trace ID to search; `None` = list all |
| `limit` | `int` | `100` | Maximum results (hard cap: 1 000) |

**Returns** — `List[TraceInfo]`, sorted by `start_time` descending.

`TraceInfo` fields: `trace_id`, `start_time`, `duration_ms`, `span_count`, `status`, `root_operation`.

---

#### `DashboardAPI.get_trace_detail(trace_id) → TraceDetail | None`

Returns the full `TraceDetail` (including all spans) for the given ID, or
`None` if not found.

`TraceDetail` fields: `trace_id`, `start_time`, `end_time`, `duration_ms`, `status`, `spans` (`List[dict]`).

---

#### `DashboardAPI.add_trace(trace_detail) → bool`

Store a `TraceDetail`. The circular buffer holds at most **1 000** traces;
the oldest is evicted when the limit is exceeded. Returns `True` on success.

---

#### `DashboardAPI.clear_traces() → None`

Discard all stored traces.

---

#### `DashboardAPI.get_stats() → dict`

```json
{
  "total_traces_stored": 42,
  "trace_list_size": 42,
  "timestamp": "2026-03-03T21:00:00+00:00"
}
```

---

### AlertEngine

```python
from vetinari.dashboard.alerts import (
    get_alert_engine, reset_alert_engine,
    AlertThreshold, AlertCondition, AlertSeverity, AlertRecord,
)
```

**Singleton access**

| Function | Returns | Description |
|---|---|---|
| `get_alert_engine()` | `AlertEngine` | Return (or create) the global singleton |
| `reset_alert_engine()` | `None` | Destroy the singleton |

---

#### `AlertThreshold` dataclass

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Unique rule name |
| `metric_key` | `str` | required | Dot-notation path into `MetricsSnapshot.to_dict()` |
| `condition` | `AlertCondition` | required | `GREATER_THAN`, `LESS_THAN`, `EQUALS` |
| `threshold_value` | `float` | required | Comparison value |
| `severity` | `AlertSeverity` | `MEDIUM` | `LOW`, `MEDIUM`, `HIGH` |
| `channels` | `List[str]` | `["log"]` | `log`, `email`, `webhook` (or custom) |
| `duration_seconds` | `int` | `0` | Seconds condition must hold before firing (0 = immediate) |

---

#### `AlertEngine` methods

| Method | Returns | Description |
|---|---|---|
| `register_threshold(t)` | `None` | Add or replace by `t.name` |
| `unregister_threshold(name)` | `bool` | Remove; returns `True` if existed |
| `clear_thresholds()` | `None` | Remove all thresholds and reset state |
| `list_thresholds()` | `List[AlertThreshold]` | All registered thresholds |
| `evaluate_all(api=None)` | `List[AlertRecord]` | Evaluate and return newly-fired alerts |
| `get_active_alerts()` | `List[AlertRecord]` | Currently firing alerts |
| `get_history()` | `List[AlertRecord]` | All alerts ever fired this session |
| `get_stats()` | `dict` | `registered_thresholds`, `active_alerts`, `total_fired` |

---

#### `AlertRecord` dataclass

| Field | Type | Description |
|---|---|---|
| `threshold` | `AlertThreshold` | The rule that fired |
| `current_value` | `float` | Metric value at time of firing |
| `trigger_time` | `float` | Unix timestamp of firing |

---

### LogAggregator

```python
from vetinari.dashboard.log_aggregator import (
    get_log_aggregator, reset_log_aggregator,
    LogRecord, AggregatorHandler,
    FileBackend, ElasticsearchBackend, SplunkBackend, DatadogBackend,
)
```

**Singleton access**

| Function | Returns | Description |
|---|---|---|
| `get_log_aggregator()` | `LogAggregator` | Return (or create) the global singleton |
| `reset_log_aggregator()` | `None` | Flush pending records and destroy singleton |

---

#### `LogRecord` dataclass

| Field | Type | Default | Description |
|---|---|---|---|
| `message` | `str` | required | Log message text |
| `level` | `str` | `"INFO"` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `timestamp` | `float` | `time.time()` | Unix timestamp |
| `trace_id` | `str \| None` | `None` | Distributed trace ID |
| `span_id` | `str \| None` | `None` | Span ID within trace |
| `request_id` | `str \| None` | `None` | HTTP / task request ID |
| `logger_name` | `str \| None` | `None` | Python logger name |
| `extra` | `dict` | `{}` | Arbitrary key-value metadata |

---

#### `LogAggregator` methods

**Backend management**

| Method | Returns | Description |
|---|---|---|
| `configure_backend(name, **kwargs)` | `None` | Add/replace backend; raises `ValueError` for unknown names |
| `remove_backend(name)` | `bool` | Remove backend; returns `True` if existed |
| `list_backends()` | `List[str]` | Active backend names |

**Ingestion**

| Method | Returns | Description |
|---|---|---|
| `ingest(record)` | `None` | Add one record; auto-flushes at `_batch_size` |
| `ingest_many(records)` | `None` | Add multiple records |
| `flush()` | `None` | Force-send all pending records to all backends |

**Search**

| Method | Returns | Description |
|---|---|---|
| `search(trace_id, level, logger_name, message_contains, since, limit)` | `List[LogRecord]` | AND-filtered search; results newest-first |
| `get_trace_records(trace_id)` | `List[LogRecord]` | All records for a trace, oldest-first |
| `correlate_span(trace_id, span_id)` | `List[LogRecord]` | All records for a specific span |

**Introspection**

| Method | Returns | Description |
|---|---|---|
| `get_stats()` | `dict` | `buffer_size`, `pending`, `backends`, `max_buffer`, `batch_size` |
| `clear_buffer()` | `None` | Discard all buffered records |

---

#### Backend configuration kwargs

**FileBackend**

| kwarg | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | `"logs/vetinari_audit.jsonl"` | Output file path (directories created automatically) |

**ElasticsearchBackend**

| kwarg | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | `"http://localhost:9200"` | Elasticsearch base URL |
| `index` | `str` | `"vetinari-logs"` | Target index name |
| `api_key` | `str \| None` | `None` | API key (`id:key` base64-encoded) |

**SplunkBackend**

| kwarg | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | `"http://localhost:8088"` | Splunk HEC base URL |
| `token` | `str` | `""` | HEC token |
| `source` | `str` | `"vetinari"` | Splunk `source` field |
| `sourcetype` | `str` | `"_json"` | Splunk `sourcetype` field |

**DatadogBackend**

| kwarg | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | `""` | Datadog API key |
| `service` | `str` | `"vetinari"` | `service` tag |
| `ddsource` | `str` | `"python"` | `ddsource` tag |
| `ddtags` | `str` | `""` | Comma-separated `ddtags` |

---

## REST API

All endpoints return `application/json`. CORS headers are set to `*` to allow
browser access during development.

---

### GET /dashboard

**Response**: `text/html` — the monitoring dashboard single-page application.

---

### GET /api/v1/health

**Response 200**

```json
{
  "status": "healthy",
  "timestamp": "2026-03-03T21:00:00+00:00",
  "service": "vetinari-dashboard"
}
```

---

### GET /api/v1/stats

**Response 200**

```json
{
  "total_traces_stored": 42,
  "trace_list_size": 42,
  "timestamp": "2026-03-03T21:00:00+00:00"
}
```

---

### GET /api/v1/metrics/latest

**Response 200**

```json
{
  "timestamp": "2026-03-03T21:00:00+00:00",
  "uptime_ms": 12345.6,
  "adapters": {
    "total_providers": 2,
    "total_requests": 100,
    "total_successful": 98,
    "total_failed": 2,
    "average_latency_ms": 123.4,
    "total_tokens_used": 50000,
    "providers": {
      "openai/gpt-4": {
        "provider": "openai",
        "model": "gpt-4",
        "requests": 60,
        "success_rate": 98.3,
        "avg_latency_ms": 140.0,
        "min_latency_ms": 80.0,
        "max_latency_ms": 350.0,
        "last_request": "2026-03-03T20:59:55+00:00"
      }
    }
  },
  "memory": {
    "backends": {
      "oc": {
        "backend": "oc",
        "writes": 200,
        "reads": 450,
        "searches": 30,
        "avg_write_latency_ms": 3.2,
        "avg_read_latency_ms": 1.8,
        "avg_search_latency_ms": 5.1,
        "dedup_hit_rate": 12.5,
        "sync_failures": 0
      }
    }
  },
  "plan": {
    "total_decisions": 10,
    "approved": 8,
    "rejected": 1,
    "auto_approved": 3,
    "approval_rate": 80.0,
    "average_risk_score": 0.23,
    "average_approval_time_ms": 95.0
  }
}
```

**Error 500**

```json
{ "error": "Internal server error description" }
```

---

### GET /api/v1/metrics/timeseries

**Query parameters**

| Name | Required | Values | Default |
|---|---|---|---|
| `metric` | No | `latency`, `success_rate`, `token_usage`, `memory_latency` | `latency` |
| `timerange` | No | `1h`, `24h`, `7d` | `24h` |
| `provider` | No | provider key string | (all) |

**Response 200**

```json
{
  "metric": "latency",
  "unit": "ms",
  "min": 80.0,
  "max": 350.0,
  "avg": 140.0,
  "points": [
    {
      "timestamp": "2026-03-03T20:55:00+00:00",
      "value": 120.5,
      "metadata": { "provider": "openai", "model": "gpt-4" }
    }
  ]
}
```

**Error 400** — unknown metric name

```json
{
  "error": "Invalid metric 'foo'. Valid metrics: ['latency', 'memory_latency', 'success_rate', 'token_usage']"
}
```

**Error 404** — no data available

```json
{ "error": "No data available for metric 'latency'" }
```

---

### GET /api/v1/traces

**Query parameters**

| Name | Required | Default | Description |
|---|---|---|---|
| `trace_id` | No | (all) | Return only traces with this ID |
| `limit` | No | `100` | 1–1000 |

**Response 200**

```json
{
  "count": 2,
  "traces": [
    {
      "trace_id": "abc-123",
      "start_time": "2026-03-03T20:59:00+00:00",
      "duration_ms": 345.6,
      "span_count": 3,
      "status": "success",
      "root_operation": "plan_approval"
    }
  ]
}
```

---

### GET /api/v1/traces/\<trace_id\>

**Response 200**

```json
{
  "trace_id": "abc-123",
  "start_time": "2026-03-03T20:59:00+00:00",
  "end_time": "2026-03-03T20:59:00.345+00:00",
  "duration_ms": 345.6,
  "status": "success",
  "spans": [
    { "span_id": "s1", "operation": "plan_approval", "duration_ms": 200.0 },
    { "span_id": "s2", "operation": "risk_eval", "duration_ms": 80.0 },
    { "span_id": "s3", "operation": "dispatch", "duration_ms": 65.6 }
  ]
}
```

**Error 404**

```json
{ "error": "Trace 'abc-123' not found" }
```

---

## Data Schemas

### MetricsSnapshot

```python
@dataclass
class MetricsSnapshot:
    timestamp:       str
    uptime_ms:       float
    adapter_summary: Dict[str, Any]
    memory_summary:  Dict[str, Any]
    plan_summary:    Dict[str, Any]
```

### TimeSeriesData

```python
@dataclass
class TimeSeriesData:
    metric:    str
    unit:      str
    points:    List[TimeSeriesPoint]
    min_value: float
    max_value: float
    avg_value: float
```

### TraceDetail

```python
@dataclass
class TraceDetail:
    trace_id:    str
    start_time:  str
    end_time:    str
    duration_ms: float
    status:      str          # "success" | "error" | "in_progress"
    spans:       List[Dict[str, Any]]
```

### AlertThreshold

```python
@dataclass
class AlertThreshold:
    name:            str
    metric_key:      str
    condition:       AlertCondition   # GREATER_THAN | LESS_THAN | EQUALS
    threshold_value: float
    severity:        AlertSeverity    # LOW | MEDIUM | HIGH
    channels:        List[str]
    duration_seconds: int
```

### LogRecord

```python
@dataclass
class LogRecord:
    message:     str
    level:       str
    timestamp:   float
    trace_id:    Optional[str]
    span_id:     Optional[str]
    request_id:  Optional[str]
    logger_name: Optional[str]
    extra:       Dict[str, Any]
```
