"""Manufacturing analytics API handlers for the Vetinari web UI.

Native Litestar equivalents of the routes previously registered by
``manufacturing_api``. Part of the Flask->Litestar migration (ADR-0066).
URL paths are identical to the Flask originals.

Covers: bottleneck identification, value stream mapping, model scout
recommendations, kaizen office, constraint violation stats, SPC charts,
and workflow quality gates.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Response, delete, get, post, put
    from litestar.params import Parameter

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Kaizen DB singleton — shared with manufacturing_api for process isolation
# ---------------------------------------------------------------------------

_KAIZEN_DB_PATH: str = ".vetinari/kaizen.db"
_kaizen_log_instance: object | None = None
_kaizen_lock = threading.Lock()


def _get_kaizen_log() -> Any:
    """Return the module-level ImprovementLog singleton.

    Creates the instance on first call using double-checked locking; subsequent
    calls return the same object to avoid opening a new SQLite connection per
    request.

    Returns:
        The shared ImprovementLog instance backed by the default DB path.
    """
    global _kaizen_log_instance
    if _kaizen_log_instance is None:
        with _kaizen_lock:
            if _kaizen_log_instance is None:
                from vetinari.kaizen.improvement_log import ImprovementLog

                _kaizen_log_instance = ImprovementLog(db_path=_KAIZEN_DB_PATH)
    return _kaizen_log_instance


# ---------------------------------------------------------------------------
# Gate runner singleton
# ---------------------------------------------------------------------------

_gate_runner_instance: object | None = None
_gate_runner_lock = threading.Lock()


def _get_gate_runner() -> Any:
    """Return the process-global WorkflowGateRunner, creating it on first call.

    Uses double-checked locking to ensure only one instance is created even
    under concurrent startup requests.

    Returns:
        The singleton WorkflowGateRunner.
    """
    global _gate_runner_instance
    if _gate_runner_instance is None:
        with _gate_runner_lock:
            if _gate_runner_instance is None:
                from vetinari.workflow.quality_gates import WorkflowGateRunner

                _gate_runner_instance = WorkflowGateRunner()
    return _gate_runner_instance


def create_manufacturing_handlers() -> list[Any]:
    """Create and return all manufacturing API route handlers.

    Returns an empty list when Litestar is not installed so the caller can
    safely extend its handler list without guarding the call.

    Returns:
        List of Litestar route handler objects for all manufacturing endpoints.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.responses import litestar_error_response

    # -- Bottleneck -----------------------------------------------------------

    @get("/api/v1/bottleneck")
    async def api_bottleneck() -> dict[str, Any]:
        """Return current pipeline bottleneck and per-agent metrics.

        Returns:
            JSON object with constraint agent, drum rate, and all agent metrics,
            or 503 when the bottleneck subsystem is unavailable.
        """
        try:
            from vetinari.orchestration.bottleneck import get_bottleneck_identifier

            identifier = get_bottleneck_identifier()
            return identifier.get_status()
        except Exception as exc:
            logger.warning("Bottleneck analysis unavailable — subsystem error: %s", exc)
            return litestar_error_response("Bottleneck subsystem unavailable", 503)

    # -- Value stream ---------------------------------------------------------

    @get("/api/v1/value-stream/aggregate")
    async def api_value_stream_aggregate(
        days: int = Parameter(query="days", default=7, ge=1, le=90),
    ) -> dict[str, Any]:
        """Return aggregate value stream metrics.

        Args:
            days: Number of days to include (1-90, default 7).

        Returns:
            JSON aggregate value stream report, or 503 when the analyzer is
            unavailable.
        """
        try:
            from vetinari.analytics.value_stream import get_value_stream_analyzer

            analyzer = get_value_stream_analyzer()
            report = analyzer.get_aggregate_report(days=days)
            return report.to_dict()
        except Exception as exc:
            logger.warning("Value stream aggregate unavailable — subsystem error: %s", exc)
            return litestar_error_response("Value stream subsystem unavailable", 503)

    # -- Model scout ----------------------------------------------------------

    @get("/api/v1/models/recommendations")
    async def api_model_recommendations(
        task_type: str = Parameter(query="task_type", default="general"),
    ) -> dict[str, Any]:
        """Return model scout recommendations for a task type.

        Args:
            task_type: Task type to get recommendations for (default "general").

        Returns:
            JSON object with task_type, is_underperforming flag, and
            recommendations list, or 503 when the model scout is unavailable.
        """
        try:
            from vetinari.models.model_scout import get_model_scout

            scout = get_model_scout()
            recs = scout.get_recommendations(task_type)
            return {
                "task_type": task_type,
                "is_underperforming": scout.is_underperforming(task_type),
                "recommendations": [r.to_dict() for r in recs],
            }
        except Exception as exc:
            logger.warning(
                "Model recommendations unavailable for task_type %r — subsystem error: %s",
                task_type,
                exc,
            )
            return litestar_error_response("Model scout subsystem unavailable", 503)

    # -- Kaizen report --------------------------------------------------------

    @get("/api/v1/kaizen/report")
    async def api_kaizen_report() -> dict[str, Any]:
        """Return the weekly kaizen report.

        Returns:
            JSON object with improvement counts, velocity, and recommendations.
            Includes a ``summary`` key built from the raw DB state via
            ``build_kaizen_report`` to expose per-status tallies.
            Returns 503 when the kaizen subsystem is unavailable.
        """
        try:
            from vetinari.kaizen.improvement_types import build_kaizen_report

            log = _get_kaizen_log()
            report = log.get_weekly_report()
            with log._connect() as conn:
                summary = build_kaizen_report(conn)
            return {
                "total_proposed": report.total_proposed,
                "total_active": report.total_active,
                "total_confirmed": report.total_confirmed,
                "total_failed": report.total_failed,
                "total_reverted": report.total_reverted,
                "avg_improvement_effect": report.avg_improvement_effect,
                "generated_at": report.generated_at.isoformat(),
                "summary": {
                    "total_proposed": summary.total_proposed,
                    "total_active": summary.total_active,
                    "total_confirmed": summary.total_confirmed,
                    "total_failed": summary.total_failed,
                    "total_reverted": summary.total_reverted,
                    "avg_improvement_effect": summary.avg_improvement_effect,
                },
            }
        except Exception as exc:
            logger.warning("Kaizen report unavailable — subsystem error: %s", exc)
            return litestar_error_response("Kaizen report subsystem unavailable", 503)

    # -- Kaizen improvements --------------------------------------------------

    @get("/api/v1/kaizen/improvements")
    async def api_kaizen_improvements(
        status: str | None = Parameter(query="status", default=None),
    ) -> list[dict[str, Any]]:
        """Return list of improvements, optionally filtered by status.

        Args:
            status: Filter by ImprovementStatus value (proposed, active,
                confirmed, etc.). Returns all when absent.

        Returns:
            JSON array of improvement records, or 503 when the kaizen subsystem
            is unavailable.
        """
        try:
            from vetinari.kaizen.improvement_log import ImprovementStatus

            log = _get_kaizen_log()
            if status:
                status_enum = ImprovementStatus(status)
                improvements = log.get_improvements_by_status(status_enum)
            else:
                improvements = []
                for st in ImprovementStatus:
                    improvements.extend(log.get_improvements_by_status(st))

            return [
                {
                    "id": imp.id,
                    "hypothesis": imp.hypothesis,
                    "metric": imp.metric,
                    "baseline_value": imp.baseline_value,
                    "target_value": imp.target_value,
                    "actual_value": imp.actual_value,
                    "applied_by": imp.applied_by,
                    "status": imp.status.value,
                    "regression_detected": imp.regression_detected,
                }
                for imp in improvements
            ]
        except Exception as exc:
            logger.warning("Kaizen improvements unavailable — subsystem error: %s", exc)
            return litestar_error_response("Kaizen improvements subsystem unavailable", 503)

    # -- Kaizen defect trends -------------------------------------------------

    @get("/api/v1/kaizen/defect-trends")
    async def api_kaizen_defect_trends() -> dict[str, Any]:
        """Return defect trend report.

        Returns:
            JSON object with per-category trends and recommendations, or 503
            when the defect trend subsystem is unavailable.
        """
        try:
            from vetinari.kaizen.defect_trends import DefectHotspot, DefectTrendAnalyzer
            from vetinari.validation import DefectCategory

            log = _get_kaizen_log()
            weekly_counts_raw = log.get_weekly_defect_counts(weeks=4)
            hotspot_rows = log.get_defect_hotspots(days=28)

            weekly_counts: list[dict[DefectCategory, int]] = []
            for week in weekly_counts_raw:
                converted: dict[DefectCategory, int] = {}
                for cat_str, count in week.items():
                    with contextlib.suppress(ValueError):
                        converted[DefectCategory(cat_str)] = count
                weekly_counts.append(converted)

            hotspots: list[DefectHotspot] = []
            for hs in hotspot_rows:
                with contextlib.suppress(ValueError):
                    hotspots.append(
                        DefectHotspot(
                            agent_type=hs["agent_type"],
                            mode=hs["mode"],
                            defect_category=DefectCategory(hs["category"]),
                            defect_count=hs["count"],
                            defect_rate=0.0,
                        )
                    )

            analyzer = DefectTrendAnalyzer()
            report = analyzer.analyze_trends(weekly_counts, hotspots=hotspots)
            return {
                "trends": {
                    cat.value: {
                        "current_count": trend.current_count,
                        "previous_count": trend.previous_count,
                        "change_pct": trend.change_pct,
                        "trend": trend.trend,
                        "is_concerning": trend.is_concerning,
                    }
                    for cat, trend in report.trends.items()
                },
                "hotspots": [
                    {
                        "agent_type": hs.agent_type,
                        "mode": hs.mode,
                        "defect_category": hs.defect_category.value,
                        "defect_count": hs.defect_count,
                    }
                    for hs in report.hotspots
                ],
                "top_defect_category": (report.top_defect_category.value if report.top_defect_category else None),
                "recommendations": report.recommendations,
            }
        except Exception as exc:
            logger.warning("Kaizen defect trends unavailable — subsystem error: %s", exc)
            return litestar_error_response("Kaizen defect trends subsystem unavailable", 503)

    # -- Constraint violation stats -------------------------------------------

    @get("/api/v1/constraints/violations/stats")
    async def api_constraint_violation_stats() -> dict[str, Any]:
        """Return aggregate constraint violation counts across all agent types.

        Reads the constraint registry's violation counter so the dashboard can
        show which constraints are being hit most frequently.

        Returns:
            JSON object with violation counts keyed by constraint identifier,
            or 503 when the constraint registry is unavailable.
        """
        try:
            from vetinari.constraints.registry import get_constraint_registry

            return get_constraint_registry().get_violation_stats()
        except Exception as exc:
            logger.warning("Constraint violation stats unavailable — subsystem error: %s", exc)
            return litestar_error_response("Constraint registry subsystem unavailable", 503)

    # -- Kaizen DB report -----------------------------------------------------

    @get("/api/v1/kaizen/report/db")
    async def api_kaizen_report_db() -> Any:
        """Return a KaizenReport built directly from the kaizen SQLite database.

        Unlike ``/api/v1/kaizen/report`` (which uses the ImprovementLog object),
        this endpoint builds the report by querying the database connection
        directly via ``build_kaizen_report``.

        Returns:
            JSON object with per-status improvement counts and average improvement
            effect, or 503 if the database is unavailable.
        """
        import sqlite3

        try:
            from vetinari.kaizen.improvement_types import build_kaizen_report
        except ImportError:
            logger.warning("kaizen.improvement_types not importable — kaizen DB report unavailable")
            return litestar_error_response("Kaizen module not available", 503)

        try:
            conn = sqlite3.connect(_KAIZEN_DB_PATH)
            conn.row_factory = sqlite3.Row
            try:
                report = build_kaizen_report(conn)
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            logger.warning(
                "Could not open kaizen database at %r — report unavailable: %s",
                _KAIZEN_DB_PATH,
                exc,
            )
            return litestar_error_response("Kaizen database not available", 503)

        return {
            "total_proposed": report.total_proposed,
            "total_active": report.total_active,
            "total_confirmed": report.total_confirmed,
            "total_failed": report.total_failed,
            "total_reverted": report.total_reverted,
            "avg_improvement_effect": report.avg_improvement_effect,
            "generated_at": report.generated_at.isoformat(),
        }

    # -- Benchmark run results ------------------------------------------------
    # NOTE: /api/v1/benchmarks/suites and /api/v1/benchmarks/suites/{suite_name}/comparison
    # are registered by litestar_dashboard_api.py — do not duplicate them here.

    @get("/api/v1/benchmarks/runs/{run_id:str}/results")
    async def api_benchmark_run_results(run_id: str) -> Any:
        """Return all individual case results for a benchmark run.

        Args:
            run_id: The unique benchmark run identifier.

        Returns:
            JSON array of case result dicts, or 404 if the run is not found,
            or 503 if the metric store is unavailable.
        """
        try:
            from vetinari.benchmarks.runner import MetricStore

            store = MetricStore()
            results = store.load_results(run_id)
        except Exception as exc:
            logger.warning("Benchmark run results unavailable for run %r — subsystem error: %s", run_id, exc)
            return litestar_error_response("Benchmark metric store unavailable", 503)
        if not results:
            return litestar_error_response(f"Run '{run_id}' not found or has no results", 404)
        return results

    # -- SPC Cpk --------------------------------------------------------------

    @get("/api/v1/spc/chart/{metric_name:str}/cpk")
    async def api_spc_cpk(
        metric_name: str,
        spec_upper: float | None = Parameter(query="spec_upper", default=None),
        spec_lower: float | None = Parameter(query="spec_lower", default=None),
    ) -> Any:
        """Return the process capability index (Cpk) for a tracked SPC metric.

        Cpk > 1.33 indicates a capable process; 1.0-1.33 is marginal; Cpk <=
        1.0 requires intervention.

        Args:
            metric_name: The SPC metric whose control chart to query.
            spec_upper: Upper specification limit (optional).
            spec_lower: Lower specification limit (optional).

        Returns:
            JSON object with ``metric_name`` and ``cpk`` (null when sigma is
            zero or no spec limits are provided), or 404 if metric has no chart,
            or 503 if the SPC monitor is unavailable.
        """
        try:
            from vetinari.workflow.spc import get_spc_monitor

            monitor = get_spc_monitor()
        except Exception as exc:
            logger.warning("SPC monitor unavailable for Cpk of metric %r — subsystem error: %s", metric_name, exc)
            return litestar_error_response("SPC monitor subsystem unavailable", 503)
        chart = monitor.get_chart(metric_name)
        if chart is None:
            return litestar_error_response("No chart found for metric", 404, details={"metric_name": metric_name})

        cpk = chart.get_cpk(spec_upper=spec_upper, spec_lower=spec_lower)
        return {"metric_name": metric_name, "cpk": cpk}

    # -- SPC chart ------------------------------------------------------------

    @get("/api/v1/spc/chart/{metric_name:str}")
    async def api_spc_chart(
        metric_name: str,
        spec_upper: float | None = Parameter(query="spec_upper", default=None),
        spec_lower: float | None = Parameter(query="spec_lower", default=None),
    ) -> Any:
        """Return control chart statistics for a single metric.

        Args:
            metric_name: The SPC metric to fetch chart data for.
            spec_upper: Upper spec limit for Cpk calculation (optional).
            spec_lower: Lower spec limit for Cpk calculation (optional).

        Returns:
            JSON object with count, mean, sigma, ucl, lcl, values, and
            optionally cpk; or 404 if no data for that metric, or 503 if
            the SPC monitor is unavailable.
        """
        try:
            from vetinari.workflow.spc import get_spc_monitor

            monitor = get_spc_monitor()
        except Exception as exc:
            logger.warning("SPC monitor unavailable for chart of metric %r — subsystem error: %s", metric_name, exc)
            return litestar_error_response("SPC monitor subsystem unavailable", 503)
        chart = monitor.get_chart(metric_name)
        if chart is None:
            return litestar_error_response(f"No chart data for metric '{metric_name}'", 404)

        payload: dict[str, Any] = {
            "metric_name": chart.metric_name,
            "count": len(chart.values),
            "mean": chart.mean,
            "sigma": chart.sigma,
            "ucl": chart.ucl,
            "lcl": chart.lcl,
            "value_count": len(chart.values),
            "in_control": chart.is_in_control(chart.values[-1]) if chart.values else True,
            "values": list(chart.values),
        }
        cpk = chart.get_cpk(spec_upper=spec_upper, spec_lower=spec_lower)
        if cpk is not None:
            payload["cpk"] = cpk
        return payload

    # -- SPC alerts -----------------------------------------------------------

    @get("/api/v1/spc/alerts")
    async def api_spc_alerts(
        metric_name: str | None = Parameter(query="metric_name", default=None),
    ) -> dict[str, Any]:
        """Return SPC alerts, optionally filtered by metric name.

        Args:
            metric_name: When provided, return only alerts for this metric.

        Returns:
            JSON object with an ``alerts`` list of alert records and ``count``,
            or 503 if the SPC monitor is unavailable.
        """
        try:
            from vetinari.workflow.spc import get_spc_monitor

            monitor = get_spc_monitor()
            alerts = monitor.get_alerts(metric_name)
        except Exception as exc:
            logger.warning("SPC alerts unavailable — subsystem error: %s", exc)
            return litestar_error_response("SPC monitor subsystem unavailable", 503)
        return {
            "alerts": [
                {
                    "metric_name": a.metric_name,
                    "value": a.value,
                    "ucl": a.ucl,
                    "lcl": a.lcl,
                    "mean": a.mean,
                    "sigma": a.sigma,
                    "alert_type": a.alert_type,
                    "timestamp": a.timestamp,
                }
                for a in alerts
            ],
            "count": len(alerts),
        }

    # -- Workflow gates list --------------------------------------------------

    @get("/api/v1/workflow/gates")
    async def api_workflow_gates_list() -> dict[str, Any]:
        """Return all currently configured workflow quality gates.

        Returns:
            JSON object mapping stage name to gate configuration, or 503
            if the gate runner subsystem is unavailable.
        """
        try:
            from vetinari.workflow.quality_gates import get_gate_runner

            runner = get_gate_runner()
            return {
                stage: {
                    "name": gate.name,
                    "stage": gate.stage,
                    "criteria": gate.criteria,
                    "failure_action": gate.failure_action.value,
                }
                for stage, gate in runner.gates.items()
            }
        except Exception as exc:
            logger.warning("Workflow gates unavailable — subsystem error: %s", exc)
            return litestar_error_response("Workflow gate runner subsystem unavailable", 503)

    # -- Add gate (POST) ------------------------------------------------------

    @post("/api/v1/workflow/gates/{stage:str}")
    async def api_add_gate(stage: str, data: dict[str, Any]) -> Any:
        """Register or replace the quality gate for a workflow stage (POST).

        The JSON body must include ``name``, ``criteria`` (a dict of threshold
        key/value pairs), and ``failure_action``
        (one of: re_plan, retry, escalate, block, continue).

        Args:
            stage: Workflow stage identifier (e.g. ``post_execution``).
            data: JSON request body.

        Returns:
            JSON confirmation with the registered gate details, or 400 on
            invalid input.
        """
        from vetinari.workflow.quality_gates import GateAction, WorkflowGate

        name = data.get("name")
        criteria = data.get("criteria")
        failure_action_str = data.get("failure_action")

        if not name or not isinstance(criteria, dict) or not failure_action_str:
            return litestar_error_response("name, criteria (dict), and failure_action are required", 400)

        # Quality gate thresholds must be numeric — reject strings before persisting bad config
        non_numeric = [k for k, v in criteria.items() if not isinstance(v, (int, float))]
        if non_numeric:
            return litestar_error_response("criteria values must be numeric thresholds", 400)

        try:
            failure_action = GateAction(failure_action_str)
        except ValueError as exc:
            logger.warning("Invalid failure_action %r for gate %r: %s", failure_action_str, name, exc)
            return litestar_error_response(
                "Invalid failure_action",
                400,
                details={"valid_values": [a.value for a in GateAction]},
            )

        gate = WorkflowGate(name=name, stage=stage, criteria=criteria, failure_action=failure_action)
        _get_gate_runner().add_gate(stage, gate)
        logger.info("Quality gate %r registered for stage %r", name, stage)
        return Response(
            content={
                "status": "ok",
                "stage": stage,
                "gate": {
                    "name": gate.name,
                    "stage": gate.stage,
                    "failure_action": gate.failure_action.value,
                },
            },
            status_code=201,
            media_type=MediaType.JSON,
        )

    # -- Update gate (PUT) ----------------------------------------------------

    @put("/api/v1/workflow/gates/{stage:str}")
    async def api_workflow_gate_add(stage: str, data: dict[str, Any]) -> Any:
        """Register or replace the quality gate for a workflow stage (PUT).

        Args:
            stage: The pipeline stage to configure
                (e.g. post_decomposition, post_execution).
            data: JSON request body with name, criteria, and failure_action.

        Returns:
            JSON confirmation with the configured gate, or 400 on invalid input.
        """
        from vetinari.workflow.quality_gates import GateAction, WorkflowGate, get_gate_runner

        _GATE_VALID_KEYS: frozenset[str] = frozenset({"name", "criteria", "failure_action"})
        if not (set(data.keys()) & _GATE_VALID_KEYS):
            return litestar_error_response(
                "Request body must contain at least one recognised key: name, criteria, or failure_action",
                400,
            )

        if "name" in data and not isinstance(data["name"], str):
            return litestar_error_response("'name' must be a string", 400)
        if "name" in data and len(data["name"]) > 200:
            return litestar_error_response("'name' must be 200 characters or fewer", 400)
        if "criteria" in data and not isinstance(data["criteria"], dict):
            return litestar_error_response("'criteria' must be an object", 400)
        if "criteria" in data:
            # Quality gate thresholds must be numeric — reject strings before persisting bad config
            non_numeric = [k for k, v in data["criteria"].items() if not isinstance(v, (int, float))]
            if non_numeric:
                return litestar_error_response("criteria values must be numeric thresholds", 400)
        if "failure_action" in data and not isinstance(data["failure_action"], str):
            return litestar_error_response("'failure_action' must be a string", 400)

        name = data.get("name", stage)
        criteria = data.get("criteria", {})
        action_str = data.get("failure_action", "block")
        try:
            failure_action = GateAction(action_str)
        except ValueError as exc:
            logger.warning("Invalid failure_action %r for workflow gate %r: %s", action_str, stage, exc)
            valid = [a.value for a in GateAction]
            return litestar_error_response(f"Invalid failure_action '{action_str}'. Valid: {valid}", 400)

        gate = WorkflowGate(name=name, stage=stage, criteria=criteria, failure_action=failure_action)
        get_gate_runner().add_gate(stage, gate)
        logger.info("Workflow gate configured for stage %s: %s", stage, name)
        return {
            "stage": stage,
            "name": name,
            "criteria": criteria,
            "failure_action": failure_action.value,
        }

    # -- Remove gate (DELETE) -------------------------------------------------

    @delete("/api/v1/workflow/gates/{stage:str}", status_code=200)
    async def api_remove_gate(stage: str) -> Any:
        """Remove the quality gate registered for a workflow stage.

        Args:
            stage: Workflow stage identifier to deregister
                (e.g. ``post_execution``).

        Returns:
            JSON confirmation with the removed gate name, or 404 if no gate was
            registered for that stage.
        """
        removed = _get_gate_runner().remove_gate(stage)
        if removed is None:
            return litestar_error_response("No gate found for stage", 404, details={"stage": stage})
        logger.info("Quality gate %r removed from stage %r", removed.name, stage)
        return {"status": "ok", "stage": stage, "removed_gate": removed.name}

    return [
        api_bottleneck,
        api_value_stream_aggregate,
        api_model_recommendations,
        api_kaizen_report,
        api_kaizen_improvements,
        api_kaizen_defect_trends,
        api_constraint_violation_stats,
        api_kaizen_report_db,
        api_benchmark_run_results,
        api_spc_cpk,
        api_spc_chart,
        api_spc_alerts,
        api_workflow_gates_list,
        api_add_gate,
        api_workflow_gate_add,
        api_remove_gate,
    ]
