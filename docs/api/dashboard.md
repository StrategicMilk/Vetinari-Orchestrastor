# Vetinari Dashboard API Reference

The dashboard API is the telemetry and monitoring backend for the Vetinari
engine inside AM Workbench.

## UI Status

The canonical UI source is `ui/svelte`. The current public route table is
API-first and does not treat legacy `/` or `/dashboard` HTML template routes as
the supported UI contract. See `docs/runbooks/dashboard-guide.md` for the
Svelte workflow.

## Python API

```python
from vetinari.dashboard.api import get_dashboard_api, reset_dashboard
from vetinari.dashboard.alerts import get_alert_engine, reset_alert_engine
from vetinari.dashboard.log_aggregator import get_log_aggregator, reset_log_aggregator
```

| Function | Purpose |
|---|---|
| `get_dashboard_api()` | Return the metrics and trace dashboard singleton |
| `reset_dashboard()` | Reset dashboard state, mainly for tests |
| `get_alert_engine()` | Return the alert threshold evaluator |
| `reset_alert_engine()` | Reset alert state |
| `get_log_aggregator()` | Return the log aggregation singleton |
| `reset_log_aggregator()` | Flush and reset log aggregation state |

## REST API

All endpoints return JSON unless otherwise noted.

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/health` | System health summary |
| `GET /api/v1/metrics/latest` | Latest dashboard metrics snapshot |
| `GET /api/v1/metrics/timeseries` | Time-series telemetry |
| `GET /api/v1/traces` | Trace search/list endpoint |
| `GET /api/v1/traces/{trace_id}` | Trace detail endpoint |
| `GET /api/v1/dashboard` | Aggregated dashboard data |
| `GET /api/v1/dashboard/health` | Dashboard subsystem health |
| `GET /api/v1/dashboard/agents/{agent_type}` | Per-agent metrics |
| `GET /api/v1/dashboard/model-health` | Model health gauge readings |
| `GET /api/v1/dashboard/events/stream` | Dashboard event stream |

See `docs/security/route-auth-matrix.md` for route protection details.

## Alerting

`AlertEngine` evaluates named thresholds against dashboard snapshots. Supported
channels are implemented in `vetinari.dashboard.alerts` and can be extended by
adding dispatchers.

## Log Aggregation

`LogAggregator` keeps an in-process circular buffer and can fan out records to
file, Elasticsearch, Splunk, Datadog, or custom backends when configured.
