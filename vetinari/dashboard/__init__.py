"""
Vetinari Dashboard Package

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
    print(metrics.to_dict())
    
    # Get time-series data
    latency_data = api.get_timeseries_data('latency', timerange='24h')
    print(latency_data.to_dict())
    
    # Search traces
    traces = api.search_traces(limit=50)
    for trace in traces:
        print(trace.to_dict())

Quick Start (Flask REST API):
    from vetinari.dashboard.rest_api import create_app
    
    app = create_app()
    app.run(port=5000)
    
    # Then access:
    # GET http://localhost:5000/api/v1/metrics/latest
    # GET http://localhost:5000/api/v1/metrics/timeseries?metric=latency
    # GET http://localhost:5000/api/v1/traces
"""

from .api import (
    DashboardAPI,
    get_dashboard_api,
    reset_dashboard,
    MetricsSnapshot,
    TimeSeriesData,
    TimeSeriesPoint,
    TraceInfo,
    TraceDetail,
)

from .alerts import (
    AlertEngine,
    AlertThreshold,
    AlertRecord,
    AlertCondition,
    AlertSeverity,
    get_alert_engine,
    reset_alert_engine,
)

from .log_aggregator import (
    LogAggregator,
    LogRecord,
    AggregatorHandler,
    FileBackend,
    ElasticsearchBackend,
    SplunkBackend,
    DatadogBackend,
    get_log_aggregator,
    reset_log_aggregator,
)

try:
    from .rest_api import create_app, run_server
    __all__ = [
        "DashboardAPI",
        "get_dashboard_api",
        "reset_dashboard",
        "MetricsSnapshot",
        "TimeSeriesData",
        "TimeSeriesPoint",
        "TraceInfo",
        "TraceDetail",
        "AlertEngine",
        "AlertThreshold",
        "AlertRecord",
        "AlertCondition",
        "AlertSeverity",
        "get_alert_engine",
        "reset_alert_engine",
        "LogAggregator",
        "LogRecord",
        "AggregatorHandler",
        "FileBackend",
        "ElasticsearchBackend",
        "SplunkBackend",
        "DatadogBackend",
        "get_log_aggregator",
        "reset_log_aggregator",
        "create_app",
        "run_server",
    ]
except ImportError:
    # Flask not installed, REST API not available
    __all__ = [
        "DashboardAPI",
        "get_dashboard_api",
        "reset_dashboard",
        "MetricsSnapshot",
        "TimeSeriesData",
        "TimeSeriesPoint",
        "TraceInfo",
        "TraceDetail",
        "AlertEngine",
        "AlertThreshold",
        "AlertRecord",
        "AlertCondition",
        "AlertSeverity",
        "get_alert_engine",
        "reset_alert_engine",
        "LogAggregator",
        "LogRecord",
        "AggregatorHandler",
        "FileBackend",
        "ElasticsearchBackend",
        "SplunkBackend",
        "DatadogBackend",
        "get_log_aggregator",
        "reset_log_aggregator",
    ]

__version__ = "1.0.0"
