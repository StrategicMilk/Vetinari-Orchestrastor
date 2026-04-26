"""Dashboard metrics API — native Litestar handlers for monitoring dashboard data.

Provides endpoints that the monitoring dashboard (dashboard.html) expects,
migrated from the Flask ``dashboard_metrics`` Blueprint.

Endpoints
---------
    GET    /api/v1/metrics/latest         — latest metrics snapshot
    GET    /api/v1/metrics/timeseries     — time-series data for a metric
    GET    /api/v1/traces                 — search or list traces
    GET    /api/v1/traces/{trace_id}      — trace detail
    DELETE /api/v1/traces                 — clear all stored traces
    GET    /api/v1/analytics/cost/top     — top agents/models by cost
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, delete, get

    from vetinari.web.responses import litestar_error_response

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_dashboard_metrics_handlers() -> list[Any]:
    """Create all Litestar route handlers for the dashboard metrics API.

    Called by ``vetinari.web.litestar_app.create_app()`` to register these
    handlers in the main Litestar application.

    Returns:
        List of Litestar route handler functions, or empty list when
        Litestar is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        logger.debug("Litestar not available — dashboard metrics API handlers not registered")
        return []

    # -- GET /api/v1/metrics/latest -------------------------------------------

    @get("/api/v1/metrics/latest", media_type=MediaType.JSON)
    async def get_latest_metrics() -> dict[str, Any]:
        """Return the latest metrics snapshot from the dashboard API.

        Returns:
            JSON response with the latest metrics snapshot data, or 503
            when the dashboard subsystem is unavailable.
        """
        try:
            from vetinari.dashboard.api import get_dashboard_api

            dashboard = get_dashboard_api()
            metrics = dashboard.get_latest_metrics()
            return metrics.to_dict()
        except Exception as exc:
            logger.warning("Latest metrics unavailable — subsystem error: %s", exc)
            return litestar_error_response("Metrics subsystem unavailable", 503)

    # -- GET /api/v1/metrics/timeseries ---------------------------------------

    @get("/api/v1/metrics/timeseries", media_type=MediaType.JSON)
    async def get_timeseries(
        metric: str = "latency",
        timerange: str = "24h",
        provider: str | None = None,
    ) -> dict[str, Any]:
        """Return time-series data for a specified metric.

        Args:
            metric: Metric name — one of latency, success_rate, token_usage, memory_latency.
            timerange: Time range string (1h, 24h, 7d).
            provider: Optional provider filter.

        Returns:
            JSON response with time-series data points, or a 400/404 response on
            invalid metric name or missing data.
        """
        try:
            from vetinari.dashboard.api import get_dashboard_api

            valid_metrics = ["latency", "success_rate", "token_usage", "memory_latency"]
            if metric not in valid_metrics:
                return litestar_error_response(  # type: ignore[return-value]
                    f"Invalid metric. Valid: {valid_metrics}", 400
                )

            dashboard = get_dashboard_api()
            ts_data = dashboard.get_timeseries_data(metric, timerange, provider)
            if ts_data is None:
                return litestar_error_response(  # type: ignore[return-value]
                    f"No data for metric '{metric}'", 404
                )

            return ts_data.to_dict()
        except Exception as exc:
            logger.warning("Timeseries metrics unavailable — subsystem error: %s", exc)
            return litestar_error_response("Metrics timeseries subsystem unavailable", 503)

    # -- GET /api/v1/traces ---------------------------------------------------

    @get("/api/v1/traces", media_type=MediaType.JSON)
    async def search_traces(
        trace_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Search or list traces with optional filtering.

        Args:
            trace_id: Optional specific trace ID to search for.
            limit: Maximum traces to return (default 100, clamped to [1, 1000]).

        Returns:
            JSON response with trace list and count.
        """
        try:
            from vetinari.dashboard.api import get_dashboard_api

            limit = max(1, min(limit, 1000))  # clamp to [1, 1000]
            dashboard = get_dashboard_api()
            traces = dashboard.search_traces(trace_id, limit)
            return {"count": len(traces), "traces": [t.to_dict() for t in traces]}
        except Exception as exc:
            logger.warning("Trace search unavailable — subsystem error: %s", exc)
            return litestar_error_response("Traces subsystem unavailable", 503)

    # -- GET /api/v1/traces/{trace_id} ----------------------------------------

    @get("/api/v1/traces/{trace_id:str}", media_type=MediaType.JSON)
    async def get_trace_detail(trace_id: str) -> dict[str, Any]:
        """Return detailed information about a specific trace.

        Args:
            trace_id: URL path parameter — the trace identifier to retrieve.

        Returns:
            JSON response with detailed trace data, 404 when trace is not found,
            or 503 when the traces subsystem is unavailable.
        """
        try:
            from vetinari.dashboard.api import get_dashboard_api

            dashboard = get_dashboard_api()
            trace = dashboard.get_trace_detail(trace_id)
            if trace is None:
                return litestar_error_response(f"Trace '{trace_id}' not found", 404)
            return trace.to_dict()
        except Exception as exc:
            logger.warning("Trace detail unavailable for %r — subsystem error: %s", trace_id, exc)
            return litestar_error_response("Traces subsystem unavailable", 503)

    # -- DELETE /api/v1/traces -------------------------------------------------

    @delete("/api/v1/traces", status_code=204)
    async def delete_all_traces() -> None:
        """Clear all stored traces from memory.

        Useful for testing or freeing memory in long-running instances.
        """
        from vetinari.dashboard.api import get_dashboard_api

        dashboard = get_dashboard_api()
        dashboard.clear_traces()

    # -- GET /api/v1/analytics/cost/top ---------------------------------------

    @get("/api/v1/analytics/cost/top", media_type=MediaType.JSON)
    async def get_cost_top(n: int = 5) -> dict[str, Any]:
        """Return top agents and models ranked by cost.

        Args:
            n: Number of top items to return (default 5).

        Returns:
            JSON response with top_agents and top_models lists.
        """
        try:
            from vetinari.analytics.cost import get_cost_tracker

            tracker = get_cost_tracker()
            return {
                "top_agents": tracker.get_top_agents(n),
                "top_models": tracker.get_top_models(n),
            }
        except Exception as exc:
            logger.warning("Cost analytics unavailable — subsystem error: %s", exc)
            return litestar_error_response("Cost analytics subsystem unavailable", 503)

    return [
        get_latest_metrics,
        get_timeseries,
        search_traces,
        get_trace_detail,
        delete_all_traces,
        get_cost_top,
    ]
