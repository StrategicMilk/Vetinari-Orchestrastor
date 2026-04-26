"""Analytics API — native Litestar handlers for the analytics subsystem.

Exposes cost tracker, SLA tracker, anomaly detector, forecaster, adapter
catalog, memory subsystem, plan execution, and alert aggregation data over
HTTP so the dashboard and operators can view live analytics.

Endpoints
---------
    GET  /api/v1/analytics/cost                          — token cost by model/agent/provider
    GET  /api/v1/analytics/sla                           — SLA compliance metrics
    GET  /api/v1/analytics/anomalies                     — recent anomaly detections
    GET  /api/v1/analytics/forecasts                     — capacity-planning forecasts
    GET  /api/v1/analytics/sla/model/{model_id}/compliance — per-model SLA compliance (path param)
    POST /api/v1/analytics/sla/breach                    — record a manual SLA breach event
    GET  /api/v1/analytics/sla/model-compliance          — per-model SLA compliance (query param)
    GET  /api/v1/analytics/overview                      — high-level summary across all subsystems
    GET  /api/v1/analytics/adapters                      — adapter performance statistics
    GET  /api/v1/analytics/memory                        — memory subsystem usage analytics
    GET  /api/v1/analytics/plan                          — plan execution summary statistics
    GET  /api/v1/analytics/alerts                        — active alerts and recent alert history
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Response, get, post

    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.responses import litestar_error_response

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_analytics_handlers() -> list[Any]:
    """Create all Litestar route handlers for the analytics API.

    Called by ``vetinari.web.litestar_app.create_app()`` to register these
    handlers in the main Litestar application.

    Returns:
        List of Litestar route handler functions, or empty list when
        Litestar is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        logger.debug("Litestar not available — analytics API handlers not registered")
        return []

    # -- GET /api/v1/analytics/cost -------------------------------------------

    @get("/api/v1/analytics/cost", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_cost_data() -> dict[str, Any]:
        """Return token cost attribution broken down by model, agent, and provider.

        Returns:
            JSON object with ``cost`` key containing the cost report dict,
            or a 503 response when the cost analytics subsystem is unavailable.
        """
        try:
            from vetinari.analytics.cost import get_cost_tracker

            tracker = get_cost_tracker()
            report = tracker.get_report()
            return {"cost": report.to_dict()}
        except Exception:
            logger.warning("Cost tracker unavailable — cannot serve cost analytics, returning 503")
            return litestar_error_response(  # type: ignore[return-value]
                "Cost analytics subsystem unavailable", 503
            )

    # -- GET /api/v1/analytics/sla --------------------------------------------

    @get("/api/v1/analytics/sla", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_sla_data() -> dict[str, Any]:
        """Return SLA compliance metrics for all registered service-level objectives.

        Returns:
            JSON object with ``sla`` key containing a list of SLO compliance reports,
            or a 503 response when the SLA tracker subsystem is unavailable.
        """
        try:
            from vetinari.analytics.sla import get_sla_tracker

            tracker = get_sla_tracker()
            reports = tracker.get_all_reports()
            serialised = [r.to_dict() for r in reports]
            return {"sla": serialised}
        except Exception:
            logger.warning("SLA tracker unavailable — cannot serve SLA analytics, returning 503")
            return litestar_error_response(  # type: ignore[return-value]
                "SLA analytics subsystem unavailable", 503
            )

    # -- GET /api/v1/analytics/anomalies --------------------------------------

    @get("/api/v1/analytics/anomalies", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_anomaly_data() -> dict[str, Any]:
        """Return statistics from the anomaly detector across all tracked metrics.

        Returns:
            JSON object with ``anomalies`` key containing detector stats,
            or a 503 response when the anomaly detector subsystem is unavailable.
        """
        try:
            from vetinari.analytics.anomaly import get_anomaly_detector

            detector = get_anomaly_detector()
            stats = detector.get_stats()
            return {"anomalies": stats}
        except Exception:
            logger.warning("Anomaly detector unavailable — cannot serve anomaly analytics, returning 503")
            return litestar_error_response(  # type: ignore[return-value]
                "Anomaly analytics subsystem unavailable", 503
            )

    # -- GET /api/v1/analytics/forecasts --------------------------------------

    @get("/api/v1/analytics/forecasts", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_forecast_data() -> dict[str, Any]:
        """Return capacity-planning forecast statistics for all tracked metrics.

        Returns:
            JSON object with ``forecasts`` key containing forecaster stats,
            or a 503 response when the forecasting subsystem is unavailable.
        """
        try:
            from vetinari.analytics.forecasting import get_forecaster

            forecaster = get_forecaster()
            stats = forecaster.get_stats()
            return {"forecasts": stats}
        except Exception:
            logger.warning("Forecaster unavailable — cannot serve forecast analytics, returning 503")
            return litestar_error_response(  # type: ignore[return-value]
                "Forecast analytics subsystem unavailable", 503
            )

    # -- GET /api/v1/analytics/sla/model/{model_id}/compliance ----------------

    @get(
        "/api/v1/analytics/sla/model/{model_id:str}/compliance",
        media_type=MediaType.JSON,
        guards=[admin_guard],
    )
    async def get_model_sla_compliance(
        model_id: str,
        budget_ms: float = 500.0,
    ) -> dict[str, Any]:
        """Return latency SLA compliance percentage for a specific model.

        Computes the percentage of recorded latency observations that fell
        within the budget window. Returns 404 when no observations exist,
        or 503 when the SLA tracker subsystem is unavailable.

        Args:
            model_id: URL path parameter — the model identifier to look up.
            budget_ms: Latency budget in milliseconds (default 500.0).

        Returns:
            JSON object with ``model_id``, ``budget_ms``, and ``compliance_pct``,
            or a 404 response when no data is available for the model,
            or a 503 response when the subsystem cannot be reached.
        """
        try:
            from vetinari.analytics.sla import get_sla_tracker

            tracker = get_sla_tracker()
            compliance = tracker.get_model_compliance(model_id, budget_ms)
            if compliance is None:
                return litestar_error_response(  # type: ignore[return-value]
                    "No observations found for model", 404, details={"model_id": model_id}
                )
            return {"model_id": model_id, "budget_ms": budget_ms, "compliance_pct": compliance}
        except Exception:
            logger.warning(
                "SLA tracker unavailable for model %s — cannot serve compliance data, returning 503",
                model_id,
            )
            return litestar_error_response(  # type: ignore[return-value]
                "SLA analytics subsystem unavailable", 503
            )

    # -- POST /api/v1/analytics/sla/breach ------------------------------------

    @post("/api/v1/analytics/sla/breach", media_type=MediaType.JSON, guards=[admin_guard])
    async def record_sla_breach(data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Record a manual SLA breach event against a named SLO.

        Accepts a JSON body with ``slo_name``, ``value`` (observed metric value),
        and ``budget`` (the violated budget threshold). An optional ``timestamp``
        (Unix epoch float) may be supplied; defaults to now.

        Args:
            data: JSON request body with breach details.

        Returns:
            JSON confirmation with the breach details (HTTP 201), or a 400
            response on invalid or missing input.
        """
        from vetinari.analytics.sla import SLABreach, get_sla_tracker

        body = data if data is not None else {}
        slo_name = body.get("slo_name")
        value = body.get("value")
        budget = body.get("budget")

        if not slo_name or value is None or budget is None:
            return litestar_error_response(  # type: ignore[return-value]
                "slo_name, value, and budget are required", 400
            )

        try:
            value = float(value)
            budget = float(budget)
        except (TypeError, ValueError) as exc:
            logger.warning("Invalid SLA breach payload — value/budget not numeric, rejecting request: %s", exc)
            return litestar_error_response(  # type: ignore[return-value]
                "value and budget must be numeric", 400
            )

        breach_kwargs: dict[str, Any] = {"slo_name": slo_name, "value": value, "budget": budget}
        if "timestamp" in body:
            with contextlib.suppress(TypeError, ValueError):
                breach_kwargs["timestamp"] = float(body["timestamp"])

        breach = SLABreach(**breach_kwargs)
        tracker = get_sla_tracker()
        tracker.record_breach(breach)
        return Response(  # type: ignore[return-value]
            content={"status": "ok", "breach": breach.to_dict()},
            status_code=201,
            media_type=MediaType.JSON,
        )

    # -- GET /api/v1/analytics/sla/model-compliance ---------------------------

    @get("/api/v1/analytics/sla/model-compliance", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_model_sla_compliance_query(
        model_id: str = "",
        budget_ms: float = 500.0,
    ) -> dict[str, Any]:
        """Return SLA compliance ratio for a specific model over a latency budget.

        Queries the fraction of requests for a given model that completed within
        the specified budget. Useful for per-model SLA dashboards and alerting.

        Args:
            model_id: The model identifier to check compliance for (required).
            budget_ms: Latency budget in milliseconds (default 500.0).

        Returns:
            JSON object with ``model_id``, ``budget_ms``, and ``compliance``
            (a float between 0.0 and 1.0, or null when no data is available).
            Returns a 400 response when ``model_id`` is missing,
            or a 503 response when the SLA tracker subsystem is unavailable.
        """
        if not model_id:
            return litestar_error_response(  # type: ignore[return-value]
                "model_id query parameter is required", 400
            )

        try:
            from vetinari.analytics.sla import get_sla_tracker

            tracker = get_sla_tracker()
            compliance = tracker.get_model_compliance(model_id, budget_ms)
            return {"model_id": model_id, "budget_ms": budget_ms, "compliance": compliance}
        except Exception:
            logger.warning(
                "SLA tracker unavailable for model %s — cannot serve compliance query, returning 503",
                model_id,
            )
            return litestar_error_response(  # type: ignore[return-value]
                "SLA analytics subsystem unavailable", 503
            )

    # -- GET /api/v1/analytics/overview ------------------------------------------

    @get("/api/v1/analytics/overview", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_analytics_overview() -> dict[str, Any]:
        """Return a high-level analytics summary across cost, SLA, and anomaly subsystems.

        Collects a brief snapshot from each subsystem independently so that a
        failure in one tracker does not suppress data from the others. Sections
        that fail are replaced with ``{"status": "unavailable"}`` sentinels.
        When ALL sections fail the response is 503. When only some sections fail
        a ``_degraded: True`` flag is added to signal partial data.

        Returns:
            JSON object with ``cost``, ``sla``, and ``anomalies`` top-level keys.
            503 when all subsystems are down. ``_degraded: True`` on partial failures.
        """
        overview: dict[str, Any] = {}
        failed_sections: int = 0

        try:
            from vetinari.analytics.cost import get_cost_tracker

            overview["cost"] = get_cost_tracker().get_report().to_dict()
        except Exception:
            logger.warning("Could not collect cost summary for analytics overview — section marked unavailable")
            overview["cost"] = {"status": "unavailable"}
            failed_sections += 1

        try:
            from vetinari.analytics.sla import get_sla_tracker

            tracker = get_sla_tracker()
            overview["sla"] = {"report_count": len(tracker.get_all_reports())}
        except Exception:
            logger.warning("Could not collect SLA summary for analytics overview — section marked unavailable")
            overview["sla"] = {"status": "unavailable"}
            failed_sections += 1

        try:
            from vetinari.analytics.anomaly import get_anomaly_detector

            stats = get_anomaly_detector().get_stats()
            overview["anomalies"] = {"count": stats.get("total_anomalies", 0)}
        except Exception:
            logger.warning("Could not collect anomaly count for analytics overview — section marked unavailable")
            overview["anomalies"] = {"status": "unavailable"}
            failed_sections += 1

        # All 3 subsystems down — nothing useful to return
        if failed_sections == 3:  # 3 = total sections in this overview
            logger.warning("All analytics subsystems unavailable — overview endpoint returning 503")
            return litestar_error_response(  # type: ignore[return-value]
                "Analytics overview subsystem unavailable", 503
            )

        # Some sections failed — mark response as degraded so callers can react
        if failed_sections > 0:
            overview["_degraded"] = True

        return overview

    # -- GET /api/v1/analytics/adapters -------------------------------------------

    @get("/api/v1/analytics/adapters", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_adapter_stats() -> dict[str, Any]:
        """Return performance statistics from the adapter catalog.

        Reads live stats from ``get_adapter_catalog()`` so operators can see
        which adapters are active and how they are performing. Returns 503
        when the adapter subsystem is not installed or has not been initialised.

        Returns:
            JSON object with ``adapters`` key containing the catalog stats dict,
            or a 503 response when the subsystem cannot be reached.
        """
        try:
            from vetinari.adapters import get_adapter_catalog  # type: ignore[import]

            catalog = get_adapter_catalog()
            stats = catalog.get_stats() if hasattr(catalog, "get_stats") else {}
            return {"adapters": stats}
        except ImportError:
            logger.warning("vetinari.adapters catalog not available — adapter stats endpoint returning 503")
            return litestar_error_response(  # type: ignore[return-value]
                "Adapter analytics subsystem not installed", 503
            )
        except Exception:
            logger.warning("Could not retrieve adapter catalog stats — returning 503")
            return litestar_error_response(  # type: ignore[return-value]
                "Adapter analytics subsystem unavailable", 503
            )

    # -- GET /api/v1/analytics/memory ---------------------------------------------

    @get("/api/v1/analytics/memory", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_memory_stats() -> dict[str, Any]:
        """Return usage analytics from the memory subsystem.

        Queries ``get_memory_manager()`` for current usage statistics (entry
        counts, size, hit rates, etc.). Returns 503 when the memory subsystem
        has not been installed or initialised.

        Returns:
            JSON object with ``memory`` key containing the usage stats dict,
            or a 503 response when the subsystem cannot be reached.
        """
        try:
            from vetinari.memory import get_memory_manager  # type: ignore[import]

            manager = get_memory_manager()
            stats = manager.get_stats() if hasattr(manager, "get_stats") else {}
            return {"memory": stats}
        except ImportError:
            logger.warning("vetinari.memory manager not available — memory stats endpoint returning 503")
            return litestar_error_response(  # type: ignore[return-value]
                "Memory analytics subsystem not installed", 503
            )
        except Exception:
            logger.warning("Could not retrieve memory manager stats — returning 503")
            return litestar_error_response(  # type: ignore[return-value]
                "Memory analytics subsystem unavailable", 503
            )

    # -- GET /api/v1/analytics/plan -----------------------------------------------

    @get("/api/v1/analytics/plan", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_plan_stats() -> dict[str, Any]:
        """Return plan execution summary statistics.

        Reads aggregate metrics from ``vetinari.analytics.plan_analytics`` such
        as total plans executed, success/failure counts, and average duration.
        Returns 503 when the plan analytics module has not been installed or
        initialised.

        Returns:
            JSON object with ``plan`` key containing the summary stats dict,
            or a 503 response when the subsystem cannot be reached.
        """
        try:
            from vetinari.analytics.plan_analytics import get_plan_analytics  # type: ignore[import]

            analytics = get_plan_analytics()
            stats = analytics.get_summary() if hasattr(analytics, "get_summary") else {}
            return {"plan": stats}
        except ImportError:
            logger.warning("vetinari.analytics.plan_analytics not available — plan stats endpoint returning 503")
            return litestar_error_response(  # type: ignore[return-value]
                "Plan analytics subsystem not installed", 503
            )
        except Exception:
            logger.warning("Could not retrieve plan analytics stats — returning 503")
            return litestar_error_response(  # type: ignore[return-value]
                "Plan analytics subsystem unavailable", 503
            )

    # -- GET /api/v1/analytics/alerts ---------------------------------------------

    @get("/api/v1/analytics/alerts", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_active_alerts() -> dict[str, Any]:
        """Return active alerts and recent alert history aggregated across subsystems.

        Collects anomaly detector alerts and recent SLA breaches into a unified
        alert list. Each alert carries a ``source`` field (``"anomaly"`` or
        ``"sla_breach"``) so the caller can route or filter by type.

        When both subsystems fail, returns 503. When only one fails, the
        response includes the data from the working subsystem and a
        ``_degraded: True`` flag so callers can detect partial data.

        Returns:
            JSON object with an ``alerts`` key containing a list of alert dicts,
            each with at minimum ``source`` and ``detail`` fields.
            503 when all subsystems are down. ``_degraded: True`` on partial failure.
        """
        alerts: list[dict[str, Any]] = []
        failed_sources: int = 0

        try:
            from vetinari.analytics.anomaly import get_anomaly_detector

            detector = get_anomaly_detector()
            recent = detector.get_recent_anomalies() if hasattr(detector, "get_recent_anomalies") else []
            for item in recent:
                entry = item if isinstance(item, dict) else {"detail": str(item)}
                alerts.append({"source": "anomaly", **entry})
        except Exception:
            logger.warning("Could not collect anomaly alerts — section omitted from alerts response")
            failed_sources += 1

        try:
            from vetinari.analytics.sla import get_sla_tracker

            tracker = get_sla_tracker()
            breaches = tracker.get_recent_breaches() if hasattr(tracker, "get_recent_breaches") else []
            for breach in breaches:
                entry = breach.to_dict() if hasattr(breach, "to_dict") else {"detail": str(breach)}
                alerts.append({"source": "sla_breach", **entry})
        except Exception:
            logger.warning("Could not collect SLA breach alerts — section omitted from alerts response")
            failed_sources += 1

        try:
            from vetinari.dashboard.alerts import get_alert_engine

            # get_active_alerts() is an instance method — use the singleton getter
            engine_alerts = get_alert_engine().get_active_alerts()
            for alert in engine_alerts:
                # AlertRecord is a dataclass; serialize via .to_dict() for JSON safety
                entry = alert.to_dict() if hasattr(alert, "to_dict") else {"detail": str(alert)}
                alerts.append({"source": "alert_engine", **entry})
        except Exception:
            logger.warning("Could not collect AlertEngine alerts — section omitted from alerts response")
            failed_sources += 1

        # All three sources down — nothing useful to return
        if failed_sources == 3:  # 3 = total alert sources (anomaly + SLA + alert_engine)
            logger.warning("All alert subsystems unavailable — alerts endpoint returning 503")
            return litestar_error_response(  # type: ignore[return-value]
                "Alerts subsystem unavailable", 503
            )

        # One source failed — surface what we have, but flag degraded state
        result: dict[str, Any] = {"alerts": alerts}
        if failed_sources > 0:
            result["_degraded"] = True
        return result

    return [
        get_cost_data,
        get_sla_data,
        get_anomaly_data,
        get_forecast_data,
        get_model_sla_compliance,
        record_sla_breach,
        get_model_sla_compliance_query,
        get_analytics_overview,
        get_adapter_stats,
        get_memory_stats,
        get_plan_stats,
        get_active_alerts,
    ]
