"""Vetinari Dashboard Package.

Provides REST API and UI for real-time metrics visualization, alert management,
and distributed trace exploration.

Key Modules:
    - api: Main REST API for dashboard endpoints
    - rest_api: Flask REST API wrapper and HTTP endpoints
    - alerts: Alert configuration and evaluation engine (Phase 4 Step 2)
    - log_aggregator: Integration with centralized logging platforms (Phase 4 Step 4)

Quick Start (Backend API):
    from vetinari.dashboard import get_dashboard_api

    api = get_dashboard_api()

    # Get latest metrics
    metrics = api.get_latest_metrics()
    logger.debug(metrics.to_dict())

    # Get time-series data
    latency_data = api.get_timeseries_data('latency', timerange='24h')
    logger.debug(latency_data.to_dict())

    # Search traces
    traces = api.search_traces(limit=50)
    for trace in traces:
        logger.debug(trace.to_dict())

Quick Start (Flask REST API):
    from vetinari.dashboard.rest_api import create_app

    app = create_app()
    app.run(port=5000)

    # Then access:
    # GET http://localhost:5000/api/v1/metrics/latest
    # GET http://localhost:5000/api/v1/metrics/timeseries?metric=latency
    # GET http://localhost:5000/api/v1/traces
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from .alerts import (  # noqa: E402
    AlertCondition,
    AlertEngine,
    AlertRecord,
    AlertSeverity,
    AlertThreshold,
    get_alert_engine,
    reset_alert_engine,
)
from .api import (  # noqa: E402
    DashboardAPI,
    MetricsSnapshot,
    TimeSeriesData,
    TimeSeriesPoint,
    TraceDetail,
    TraceInfo,
    get_dashboard_api,
    reset_dashboard,
)
from .log_aggregator import (  # noqa: E402
    AggregatorHandler,
    DatadogBackend,
    ElasticsearchBackend,
    FileBackend,
    LogAggregator,
    LogRecord,
    SplunkBackend,
    SSEBackend,
    get_log_aggregator,
    get_sse_backend,
    reset_log_aggregator,
    reset_sse_backend,
)

try:
    from .rest_api import create_app, run_server

    __all__ = [
        "AggregatorHandler",
        "AlertCondition",
        "AlertEngine",
        "AlertRecord",
        "AlertSeverity",
        "AlertThreshold",
        "DashboardAPI",
        "DatadogBackend",
        "ElasticsearchBackend",
        "FileBackend",
        "LogAggregator",
        "LogRecord",
        "MetricsSnapshot",
        "SSEBackend",
        "SplunkBackend",
        "TimeSeriesData",
        "TimeSeriesPoint",
        "TraceDetail",
        "TraceInfo",
        "create_app",
        "get_alert_engine",
        "get_dashboard_api",
        "get_log_aggregator",
        "get_sse_backend",
        "reset_alert_engine",
        "reset_dashboard",
        "reset_log_aggregator",
        "reset_sse_backend",
        "run_server",
    ]
except ImportError:
    # Flask not installed, REST API not available
    __all__ = [
        "AggregatorHandler",
        "AlertCondition",
        "AlertEngine",
        "AlertRecord",
        "AlertSeverity",
        "AlertThreshold",
        "DashboardAPI",
        "DatadogBackend",
        "ElasticsearchBackend",
        "FileBackend",
        "LogAggregator",
        "LogRecord",
        "MetricsSnapshot",
        "SSEBackend",
        "SplunkBackend",
        "TimeSeriesData",
        "TimeSeriesPoint",
        "TraceDetail",
        "TraceInfo",
        "get_alert_engine",
        "get_dashboard_api",
        "get_log_aggregator",
        "get_sse_backend",
        "reset_alert_engine",
        "reset_dashboard",
        "reset_log_aggregator",
        "reset_sse_backend",
    ]

__version__ = "1.0.0"
