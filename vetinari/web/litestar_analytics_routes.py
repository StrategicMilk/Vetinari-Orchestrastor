"""Analytics routes — native Litestar handlers for the analytics dashboard.

Migrated from ``vetinari.web.analytics_routes`` Flask Blueprint. Native
Litestar equivalents (ADR-0066). URL paths identical to Flask.

Note on path overlap: ``vetinari.web.litestar_analytics`` covers the
``/api/v1/analytics/`` namespace (versioned API). This module covers the
original ``/api/analytics/`` namespace used by the dashboard. The two sets
of endpoints are complementary, not duplicates.

Endpoints
---------
    GET /api/analytics/cost         — cost attribution report (filterable)
    GET /api/analytics/sla          — SLA compliance for all registered SLOs
    GET /api/analytics/anomalies    — anomaly detection history (filterable)
    GET /api/analytics/forecast     — capacity forecasts per metric
    GET /api/analytics/models       — per-model performance combining cost + SLA
    GET /api/analytics/agents       — per-agent cost and quality metrics
    GET /api/analytics/drift/trend  — active goal-drift trend from execution graph
    GET /api/analytics/summary      — dashboard summary (all dimensions combined)
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, get

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_analytics_routes_handlers() -> list[Any]:
    """Create all Litestar route handlers for the analytics dashboard routes.

    Called by the Litestar app factory to register the ``/api/analytics/``
    family of endpoints. These mirror the original Flask Blueprint paths
    exactly so existing dashboard clients continue to work without path changes.

    Returns:
        List of Litestar route handler functions, or empty list when
        Litestar is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        logger.debug("Litestar not available — analytics route handlers not registered")
        return []

    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.responses import litestar_error_response

    # -- GET /api/analytics/cost ----------------------------------------------

    @get("/api/analytics/cost", media_type=MediaType.JSON, guards=[admin_guard])
    async def analytics_cost(
        agent: str = "",
        task_id: str = "",
        since_hours: float = 0.0,
    ) -> dict[str, Any]:
        """Return cost attribution report optionally filtered by agent or task.

        Args:
            agent: Filter costs to a specific agent type (empty = all agents).
            task_id: Filter costs to a specific task (empty = all tasks).
            since_hours: Restrict results to the last N hours (0.0 = all time).

        Returns:
            JSON object with ``status`` and ``cost`` keys on success; 500 on
            failure.
        """
        from vetinari.analytics.cost import get_cost_tracker

        try:
            tracker = get_cost_tracker()
            since = (time.time() - since_hours * 3600) if since_hours else None
            report = tracker.get_report(
                agent=agent or None,
                task_id=task_id or None,
                since=since,
            )
            return {"status": "ok", "cost": report.to_dict()}

        except Exception:
            logger.exception("Failed to retrieve cost analytics")
            return litestar_error_response("Failed to retrieve cost analytics", 500)  # type: ignore[return-value]

    # -- GET /api/analytics/sla -----------------------------------------------

    @get("/api/analytics/sla", media_type=MediaType.JSON, guards=[admin_guard])
    async def analytics_sla() -> dict[str, Any]:
        """Return SLA compliance reports for all registered service-level objectives.

        Returns:
            JSON object with ``status`` and ``sla`` (containing ``slos`` list
            and aggregate ``stats``) on success; 500 on failure.
        """
        from vetinari.analytics.sla import get_sla_tracker

        try:
            tracker = get_sla_tracker()
            reports = tracker.get_all_reports()
            return {
                "status": "ok",
                "sla": {
                    "slos": [r.to_dict() for r in reports],
                    "stats": tracker.get_stats(),
                },
            }

        except Exception:
            logger.exception("Failed to retrieve SLA analytics")
            return litestar_error_response("Failed to retrieve SLA analytics", 500)  # type: ignore[return-value]

    # -- GET /api/analytics/anomalies -----------------------------------------

    @get("/api/analytics/anomalies", media_type=MediaType.JSON, guards=[admin_guard])
    async def analytics_anomalies(metric: str = "") -> dict[str, Any]:
        """Return anomaly detection history, optionally filtered by metric name.

        Args:
            metric: Restrict results to anomalies on this metric name
                (empty = all metrics).

        Returns:
            JSON object with ``status`` and ``anomalies`` (``total``, ``items``,
            ``stats``) on success; 500 on failure. Returns at most 100 most
            recent anomaly items.
        """
        from vetinari.analytics.anomaly import get_anomaly_detector

        try:
            detector = get_anomaly_detector()
            history = detector.get_history(metric=metric or None)
            return {
                "status": "ok",
                "anomalies": {
                    "total": len(history),
                    "items": [a.to_dict() for a in history[-100:]],
                    "stats": detector.get_stats(),
                },
            }

        except Exception:
            logger.exception("Failed to retrieve anomaly analytics")
            return litestar_error_response("Failed to retrieve anomaly analytics", 500)  # type: ignore[return-value]

    # -- GET /api/analytics/forecast ------------------------------------------

    @get("/api/analytics/forecast", media_type=MediaType.JSON, guards=[admin_guard])
    async def analytics_forecast(
        metric: str = "",
        horizon: int = 10,
        method: str = "linear_trend",
    ) -> dict[str, Any]:
        """Return capacity forecasts for tracked metrics.

        Runs the configured forecasting method over up to 20 metrics per
        request. Individual metric failures are reported inline rather than
        aborting the entire response.

        Args:
            metric: Specific metric to forecast (empty = all tracked metrics).
            horizon: Number of future steps to forecast (1-365, default 10).
            method: Forecasting method — ``sma``, ``exp_smoothing``,
                ``linear_trend``, or ``seasonal`` (default ``linear_trend``).

        Returns:
            JSON object with ``status`` and ``forecast`` (``horizon``, ``method``,
            ``metrics``, ``forecaster_stats``) on success; 500 on total failure.
        """
        from vetinari.analytics.forecasting import ForecastRequest, get_forecaster

        try:
            fc = get_forecaster()
            # Clamp horizon to a safe range — mirrors Flask safe_int_param behaviour
            safe_horizon = max(1, min(horizon, 365))
            metrics = [metric] if metric else fc.list_metrics()
            results: dict[str, Any] = {}

            for m in metrics[:20]:  # Cap at 20 metrics per request
                try:
                    req = ForecastRequest(metric=m, horizon=safe_horizon, method=method)
                    result = fc.forecast(req)
                    results[m] = result.to_dict()
                except Exception:
                    logger.exception("Failed to generate forecast for metric %s", m)
                    results[m] = {"error": "Internal server error"}

            return {
                "status": "ok",
                "forecast": {
                    "horizon": safe_horizon,
                    "method": method,
                    "metrics": results,
                    "forecaster_stats": fc.get_stats(),
                },
            }

        except Exception:
            logger.exception("Failed to retrieve forecast analytics")
            return litestar_error_response("Failed to retrieve forecast analytics", 500)  # type: ignore[return-value]

    # -- GET /api/analytics/models --------------------------------------------

    @get("/api/analytics/models", media_type=MediaType.JSON, guards=[admin_guard])
    async def analytics_models() -> dict[str, Any]:
        """Return per-model performance statistics combining cost and SLA data.

        Builds a ``by_model`` map from the cost tracker and enriches each entry
        with request counts and SLA compliance percentages where available.

        Returns:
            JSON object with ``status`` and ``models`` (``by_model`` stats and
            ``top_cost`` rankings) on success; 500 on failure.
        """
        from vetinari.analytics.cost import get_cost_tracker
        from vetinari.analytics.sla import get_sla_tracker

        try:
            cost_tracker = get_cost_tracker()
            cost_report = cost_tracker.get_report()
            sla_reports = get_sla_tracker().get_all_reports()

            model_stats: dict[str, Any] = {}
            for model_key, cost_usd in cost_report.by_model.items():
                model_stats[model_key] = {"cost_usd": cost_usd, "requests": 0}

            for entry in cost_tracker._entries:
                key = f"{entry.provider}:{entry.model}"
                if key in model_stats:
                    model_stats[key]["requests"] += 1

            for report in sla_reports:
                slo_name = report.slo.name
                for key in model_stats:
                    if key in slo_name or slo_name in key:
                        model_stats[key]["sla_compliance_pct"] = report.compliance_pct

            return {
                "status": "ok",
                "models": {
                    "by_model": model_stats,
                    "top_cost": cost_tracker.get_top_models(n=5),
                },
            }

        except Exception:
            logger.exception("Failed to retrieve model analytics")
            return litestar_error_response("Failed to retrieve model analytics", 500)  # type: ignore[return-value]

    # -- GET /api/analytics/agents --------------------------------------------

    @get("/api/analytics/agents", media_type=MediaType.JSON, guards=[admin_guard])
    async def analytics_agents() -> dict[str, Any]:
        """Return per-agent cost and quality metrics.

        Combines cost tracker data with quality scores from the feedback loop
        where available. Feedback loop unavailability is non-fatal.

        Returns:
            JSON object with ``status`` and ``agents`` (``by_agent_cost``,
            ``top_agents``, ``agent_quality``, ``total_requests``) on success;
            500 on failure.
        """
        from vetinari.analytics.cost import get_cost_tracker

        try:
            cost_tracker = get_cost_tracker()
            cost_report = cost_tracker.get_report()
            top_agents = cost_tracker.get_top_agents(n=10)

            agent_quality: dict[str, float] = {}
            # Track whether the feedback loop failed so callers can distinguish
            # "no quality data recorded yet" from "quality subsystem is down".
            _feedback_degraded = False
            try:
                from vetinari.learning.feedback_loop import get_feedback_loop

                fl = get_feedback_loop()
                if hasattr(fl, "get_stats"):
                    for agent_key in cost_report.by_agent:
                        stats = fl.get_stats(agent_key) if hasattr(fl, "get_stats") else {}
                        agent_quality[agent_key] = stats.get("avg_quality", 0.0) if stats else 0.0
            except Exception:
                logger.warning(
                    "Could not load feedback loop quality scores — agent_quality will be empty",
                    exc_info=True,
                )
                _feedback_degraded = True

            result: dict[str, Any] = {
                "status": "ok",
                "agents": {
                    "by_agent_cost": cost_report.by_agent,
                    "top_agents": top_agents,
                    "agent_quality": agent_quality,
                    "total_requests": cost_report.total_requests,
                },
            }
            if _feedback_degraded:
                result["_degraded"] = True
            return result

        except Exception:
            logger.exception("Failed to retrieve agent analytics")
            return litestar_error_response("Failed to retrieve agent analytics", 500)  # type: ignore[return-value]

    # -- GET /api/analytics/drift/trend ---------------------------------------

    @get("/api/analytics/drift/trend", media_type=MediaType.JSON, guards=[admin_guard])
    async def analytics_drift_trend() -> dict[str, Any]:
        """Return the active goal-drift trend from the running execution graph.

        Reads the GoalTracker drift trend from the active AgentGraph. Returns a
        stable ``{"trend": "no_active_execution"}`` payload when no execution is
        currently in progress.

        Returns:
            JSON object with ``status`` and ``drift`` keys on success; 500 on
            failure.
        """
        from vetinari.drift.wiring import get_active_drift_trend

        try:
            trend = get_active_drift_trend()
            return {"status": "ok", "drift": trend}

        except Exception:
            logger.exception("Failed to retrieve drift trend")
            return litestar_error_response("Failed to retrieve drift trend", 500)  # type: ignore[return-value]

    # -- GET /api/analytics/summary -------------------------------------------

    @get("/api/analytics/summary", media_type=MediaType.JSON, guards=[admin_guard])
    async def analytics_summary() -> dict[str, Any]:
        """Return a dashboard summary combining all analytics dimensions.

        Aggregates cost totals, SLA compliance overview, anomaly counts,
        forecaster stats, and the current auto-tuner configuration. Auto-tuner
        unavailability is non-fatal and results in an empty ``tuner_config``.

        Returns:
            JSON object with ``status`` and ``summary`` keys on success; 500 on
            failure.
        """
        from vetinari.analytics.anomaly import get_anomaly_detector
        from vetinari.analytics.cost import get_cost_tracker
        from vetinari.analytics.forecasting import get_forecaster
        from vetinari.analytics.sla import get_sla_tracker

        try:
            cost_report = get_cost_tracker().get_report()
            sla_reports = get_sla_tracker().get_all_reports()
            anomaly_stats = get_anomaly_detector().get_stats()
            forecaster_stats = get_forecaster().get_stats()

            compliant_slos = sum(1 for r in sla_reports if r.is_compliant)
            total_slos = len(sla_reports)

            tuner_config: dict[str, Any] = {}
            # Track whether the auto-tuner failed so callers can distinguish
            # "no tuner config set yet" from "tuner subsystem is down".
            _tuner_degraded = False
            try:
                from vetinari.learning.auto_tuner import get_auto_tuner

                tuner_config = get_auto_tuner().get_config()
            except Exception:
                logger.warning(
                    "Could not load auto-tuner config for summary — tuner_config will be empty",
                    exc_info=True,
                )
                _tuner_degraded = True

            summary_result: dict[str, Any] = {
                "status": "ok",
                "summary": {
                    "cost": {
                        "total_usd": cost_report.total_cost_usd,
                        "total_tokens": cost_report.total_tokens,
                        "total_requests": cost_report.total_requests,
                    },
                    "sla": {
                        "compliant_slos": compliant_slos,
                        "total_slos": total_slos,
                        "compliance_pct": round(compliant_slos / max(total_slos, 1) * 100, 1),
                        "slo_names": [r.slo.name for r in sla_reports],
                    },
                    "anomalies": {
                        "total_detected": anomaly_stats.get("total_anomalies", 0),
                        "tracked_metrics": anomaly_stats.get("tracked_metrics", 0),
                    },
                    "forecasting": {
                        "tracked_metrics": forecaster_stats.get("tracked_metrics", 0),
                    },
                    "tuner_config": tuner_config,
                    "generated_at": time.time(),
                },
            }
            if _tuner_degraded:
                summary_result["_degraded"] = True
            return summary_result

        except Exception:
            logger.exception("Failed to generate analytics summary")
            return litestar_error_response("Failed to generate analytics summary", 500)  # type: ignore[return-value]

    return [
        analytics_cost,
        analytics_sla,
        analytics_anomalies,
        analytics_forecast,
        analytics_models,
        analytics_agents,
        analytics_drift_trend,
        analytics_summary,
    ]
