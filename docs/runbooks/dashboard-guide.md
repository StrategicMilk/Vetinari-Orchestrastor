# Vetinari Monitoring Dashboard — User Guide

**Phase 4 | Owner: Observability Lead | Last Updated: March 2026**

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Starting the Server](#starting-the-server)
4. [Dashboard UI Walkthrough](#dashboard-ui-walkthrough)
5. [Alert System](#alert-system)
6. [Log Aggregation](#log-aggregation)
7. [REST API Usage](#rest-api-usage)
8. [Configuration Reference](#configuration-reference)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The Vetinari Monitoring Dashboard gives you real-time visibility into your
multi-agent orchestration system. It surfaces the telemetry data collected by
Phase 3 (adapters, memory, plan-gate) through a browser-based UI, a REST API,
a threshold-based alert engine, and a pluggable log-aggregation layer.

**Components**

| Component | Module | Purpose |
|---|---|---|
| Dashboard API | `vetinari.dashboard.api` | In-process metrics & trace query engine |
| Web API | `vetinari.web.litestar_app` | Litestar ASGI app serving dashboard and API routes |
| Alert Engine | `vetinari.dashboard.alerts` | Threshold evaluation & dispatch |
| Log Aggregator | `vetinari.dashboard.log_aggregator` | Structured-log fan-out (file / ES / Splunk / Datadog) |
| Dashboard UI | `ui/templates/dashboard.html` | Single-page monitoring interface |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Launch the server
python -m vetinari serve --port 5000

# 3. Open in your browser
#    http://localhost:5000
```

---

## Starting the Server

### CLI (recommended)

```bash
python -m vetinari serve --port 5000              # default: 127.0.0.1:5000
python -m vetinari start                          # start with dashboard
python -m vetinari start --goal "..."             # start with goal
```

### Programmatic

```python
from vetinari.web.litestar_app import create_app

app = create_app()
# Run with uvicorn:
# uvicorn vetinari.web.litestar_app:get_app --factory --host 0.0.0.0 --port 5000
```

The `create_app()` factory creates a Litestar ASGI application with all API
routes registered. Static files are served from `ui/static/`.

---

## Dashboard UI Walkthrough

Navigate to `http://localhost:5000/dashboard` after starting the server.

### Sidebar Sections

| Section | What it shows |
|---|---|
| **Overview** | Six KPI cards + four Chart.js time-series charts |
| **Adapters** | Per-provider table: requests, success %, latency min/avg/max, tokens, last request |
| **Memory** | Per-backend table: reads, writes, searches, latency, dedup hit rate, sync failures |
| **Plan Gate** | Six KPI cards: approved, rejected, auto-approved, approval rate, avg risk, avg decision time |
| **Traces** | Searchable trace table + span timeline modal |
| **Alerts** | Active alerts + history (client-side accumulation) |

### Auto-refresh

Use the **Auto-refresh** toggle and the interval dropdown (5 / 15 / 30 / 60 s)
in the top-right corner. Click **Refresh** to force an immediate update.

### Trace Explorer

1. Enter a trace ID in the search box and click **Search**, or browse the list
   of recent traces.
2. Click the eye icon on any row to open the **Span Timeline** modal, which
   shows each span as a proportional horizontal bar.

---

## Alert System

### Concepts

| Term | Description |
|---|---|
| `AlertThreshold` | A named rule: metric key + condition + value + severity + channels |
| `AlertEngine` | Evaluates all thresholds against the live snapshot; fires alerts |
| `AlertRecord` | An alert instance that was fired (metric, value, time) |
| Dispatcher | Function that routes a fired alert to a channel (log / email / webhook) |

### Supported metric keys (dot-notation into `MetricsSnapshot.to_dict()`)

```
adapters.total_requests
adapters.total_failed
adapters.average_latency_ms
adapters.total_tokens_used
plan.total_decisions
plan.approval_rate
plan.average_risk_score
plan.average_approval_time_ms
```

### Basic usage

```python
from vetinari.dashboard.alerts import (
    get_alert_engine,
    AlertThreshold,
    AlertCondition,
    AlertSeverity,
)

engine = get_alert_engine()

# Fire if average adapter latency exceeds 500 ms
engine.register_threshold(AlertThreshold(
    name="high-latency",
    metric_key="adapters.average_latency_ms",
    condition=AlertCondition.GREATER_THAN,
    threshold_value=500.0,
    severity=AlertSeverity.HIGH,
    channels=["log"],
))

# Fire if plan approval rate drops below 70 %
engine.register_threshold(AlertThreshold(
    name="low-approval-rate",
    metric_key="plan.approval_rate",
    condition=AlertCondition.LESS_THAN,
    threshold_value=70.0,
    severity=AlertSeverity.MEDIUM,
    channels=["log", "webhook"],
))

# Evaluate (call this on a schedule, e.g. every 30 s)
fired = engine.evaluate_all()
for alert in fired:
    print(f"Alert fired: {alert.threshold.name} = {alert.current_value}")
```

### Duration-based alerts

Set `duration_seconds > 0` to require the condition to hold continuously before
firing. This prevents alert storms from transient spikes:

```python
AlertThreshold(
    name="sustained-latency",
    metric_key="adapters.average_latency_ms",
    condition=AlertCondition.GREATER_THAN,
    threshold_value=300.0,
    duration_seconds=60,   # only fires after 60 s above threshold
    severity=AlertSeverity.MEDIUM,
    channels=["log"],
)
```

### Suppression & re-fire

Once an alert fires it is suppressed until the condition clears. When the metric
returns within bounds the alert is removed from `get_active_alerts()`. If the
condition re-triggers, it fires again.

### Custom channels

Add your own dispatcher to `DISPATCHERS`:

```python
from vetinari.dashboard.alerts import DISPATCHERS, AlertRecord

def my_slack_dispatcher(alert: AlertRecord) -> None:
    import requests
    requests.post(
        "https://hooks.slack.com/services/...",
        json={"text": f":warning: {alert.threshold.name}: {alert.current_value}"},
    )

DISPATCHERS["slack"] = my_slack_dispatcher

# Then reference "slack" in any AlertThreshold.channels list
```

---

## Log Aggregation

### In-process buffer & search

Records ingested through `LogAggregator` are held in a circular buffer
(default 5 000 entries) and can be searched without any external service:

```python
from vetinari.dashboard.log_aggregator import get_log_aggregator, LogRecord

agg = get_log_aggregator()

agg.ingest(LogRecord(
    message="Plan approved",
    level="INFO",
    trace_id="abc-123",
    span_id="span-001",
    extra={"plan_id": "plan_007", "risk_score": 0.12},
))

# Search by trace
records = agg.get_trace_records("abc-123")

# Cross-filter search
records = agg.search(
    level="ERROR",
    message_contains="timeout",
    since=time.time() - 3600,   # last hour
    limit=50,
)
```

### Configuring a backend

```python
# File — no external dependencies
agg.configure_backend("file", path="logs/vetinari_audit.jsonl")

# Elasticsearch
agg.configure_backend(
    "elasticsearch",
    url="http://localhost:9200",
    index="vetinari-logs",
    api_key="my_api_key",          # optional
)

# Splunk HEC
agg.configure_backend(
    "splunk",
    url="http://splunk-hec:8088",
    token="your-hec-token",
    source="vetinari",
    sourcetype="_json",
)

# Datadog
agg.configure_backend(
    "datadog",
    api_key="your-dd-api-key",
    service="vetinari",
    ddsource="python",
    ddtags="env:prod,team:ml",
)
```

Multiple backends can be active simultaneously; records are fanned out to all
of them on each `flush()`.

### Attaching to Python logging

`AggregatorHandler` bridges stdlib `logging` into the aggregator automatically:

```python
import logging
from vetinari.dashboard.log_aggregator import AggregatorHandler

logging.getLogger().addHandler(AggregatorHandler())
# All log output from this point is captured in the aggregator buffer
```

### Batch flushing

Records are queued and flushed automatically when the batch reaches
`_batch_size` (default 100). Call `agg.flush()` to force an immediate send —
useful at application shutdown:

```python
import atexit
atexit.register(get_log_aggregator().flush)
```

---

## REST API Usage

Base URL: `http://localhost:5000`

| Method | Endpoint | Description |
|---|---|---|
| GET | `/dashboard` | Monitoring UI (HTML) |
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/stats` | Dashboard statistics |
| GET | `/api/v1/metrics/latest` | Latest MetricsSnapshot |
| GET | `/api/v1/metrics/timeseries` | Time-series data |
| GET | `/api/v1/traces` | List / search traces |
| GET | `/api/v1/traces/<trace_id>` | Trace detail with spans |

### Query parameters — `/api/v1/metrics/timeseries`

| Parameter | Values | Default |
|---|---|---|
| `metric` | `latency`, `success_rate`, `token_usage`, `memory_latency` | `latency` |
| `timerange` | `1h`, `24h`, `7d` | `24h` |
| `provider` | any provider key | (all) |

### Query parameters — `/api/v1/traces`

| Parameter | Values | Default |
|---|---|---|
| `trace_id` | any string | (all) |
| `limit` | 1–1000 | `100` |

---

## Configuration Reference

### AlertEngine

```python
engine = get_alert_engine()
engine.register_threshold(AlertThreshold(...))  # add / replace by name
engine.unregister_threshold("name")             # remove
engine.clear_thresholds()                       # remove all
engine.list_thresholds()                        # List[AlertThreshold]
engine.get_active_alerts()                      # List[AlertRecord] — currently firing
engine.get_history()                            # List[AlertRecord] — all that ever fired
engine.get_stats()                              # dict
```

### LogAggregator

```python
agg = get_log_aggregator()
agg.configure_backend(name, **kwargs)   # add backend
agg.remove_backend(name)                # remove backend
agg.list_backends()                     # List[str]
agg.ingest(record)                      # single record
agg.ingest_many(records)               # batch
agg.flush()                             # force dispatch
agg.search(...)                         # in-process filter
agg.get_trace_records(trace_id)         # ordered by timestamp
agg.correlate_span(trace_id, span_id)   # narrow to one span
agg.get_stats()                         # dict
agg.clear_buffer()                      # discard in-memory records
```

---

## Troubleshooting

### Dashboard page returns 404

Ensure you are navigating to `/dashboard` (not `/`). The root path is the
existing Vetinari app (`ui/templates/index.html`), not the monitoring dashboard.

### Charts show no data

The time-series endpoints derive data from the `TelemetryCollector` singleton.
If no adapter calls have been made yet, the charts will be empty. Generate some
traffic via `telemetry.record_adapter_latency(...)` or by running actual tasks.

### Alerts never fire

1. Verify `metric_key` matches the exact dot-notation path shown in the
   [Alert System](#alert-system) section.
2. Call `engine.evaluate_all()` explicitly — the engine does **not** poll
   automatically. Wrap it in a background thread or scheduled job.
3. Check `engine.get_stats()` to confirm the threshold is registered.

### File backend writes 0 records

Call `agg.flush()` after ingestion. Records are batched; without a flush call
they remain in `_pending` until the batch fills to `_batch_size` (default 100).

### Elasticsearch returns 4xx

- Confirm the index exists or that `auto_create_index` is enabled.
- Check the `api_key` is base64-encoded as `id:api_key` (standard ES format).
- Inspect the raw error with `logger.setLevel(logging.DEBUG)`.

### `requests` not installed

The Elasticsearch, Splunk, and Datadog backends require `requests`, which is installed through the project metadata:

```bash
python -m pip install -e ".[dev]"
```

The file backend and in-process search work without it.
