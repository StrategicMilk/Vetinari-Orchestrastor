"""Dashboard API — native Litestar handlers for aggregated agent metrics and system health. Native Litestar equivalents (ADR-0066). URL paths identical to Flask.

Covers all routes from ``vetinari.web.dashboard_api`` that are NOT already
served by ``litestar_dashboard_metrics.py`` (which handles /api/v1/metrics/*,
/api/v1/traces/*, and /api/v1/analytics/cost/top).

Endpoints:
    GET  /api/v1/dashboard                             — aggregated dashboard data
    GET  /api/v1/dashboard/health                      — system health summary
    POST /api/v1/dashboard/quality/batch               — record batch quality scores
    GET  /api/v1/dashboard/quality/drift               — quality drift statistics
    GET  /api/v1/dashboard/agents/{agent_type}         — per-agent metrics
    GET  /api/v1/benchmarks/results/{run_id}           — benchmark run results
    GET  /api/v1/benchmarks/suites                     — registered benchmark suites
    GET  /api/v1/benchmarks/suites/{suite_name}/comparison — last two-run comparison
    POST /api/v1/dashboard/quality/scores-batch        — alternative batch quality endpoint
    GET  /api/v1/dashboard/quality/drift-stats         — drift stats (alias)
    GET  /api/v1/pipeline/status                       — pipeline stage run-time stats
    GET  /api/v1/dashboard/model-health                — model health gauge readings
    GET  /api/v1/dashboard/welcome-back                — activity summary since last visit
    GET  /api/v1/dashboard/events/stream               — SSE pipeline events
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, get, post
    from litestar.params import Parameter
    from litestar.response import ServerSentEvent

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False

# Maximum idle keepalive cycles before the stream times out (~10 min at 5 s poll).
_DASHBOARD_STREAM_MAX_IDLE = 120


def create_dashboard_api_handlers() -> list[Any]:
    """Create all Litestar route handlers for the dashboard API.

    Called by ``vetinari.web.litestar_app.create_app()`` to register these
    handlers in the main Litestar application.  Does not duplicate handlers
    already registered by ``create_dashboard_metrics_handlers()``.

    Returns:
        List of Litestar route handler functions, or empty list when
        Litestar is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        logger.debug("Litestar not available — dashboard API handlers not registered")
        return []

    from vetinari.web.request_validation import json_object_body
    from vetinari.web.responses import litestar_error_response

    # -- GET /api/v1/dashboard -------------------------------------------------

    @get("/api/v1/dashboard", media_type=MediaType.JSON)
    async def api_dashboard() -> dict[str, Any]:
        """Return aggregated dashboard data including agent metrics and system health.

        Returns:
            JSON response with ``status``, ``health``, ``agents``, and
            ``agent_modes`` data from the AgentDashboard singleton.
        """
        try:
            from vetinari.dashboard.agent_dashboard import get_agent_dashboard
        except ImportError:
            logger.warning("agent_dashboard module not available")
            return litestar_error_response("Dashboard module not available", 503)  # type: ignore[return-value]

        try:
            dashboard = get_agent_dashboard()
            data = dashboard.get_dashboard_data()
        except Exception as exc:
            logger.warning("Could not load dashboard data — returning 503: %s", exc)
            return litestar_error_response("Dashboard data unavailable", 503)  # type: ignore[return-value]

        return {"status": "ok", **data}

    # -- GET /api/v1/dashboard/health ------------------------------------------

    @get("/api/v1/dashboard/health", media_type=MediaType.JSON)
    async def api_dashboard_health() -> dict[str, Any]:
        """Return system health summary only.

        Returns:
            JSON response with system health metrics, or error with status code.
        """
        try:
            from vetinari.dashboard.agent_dashboard import get_agent_dashboard
        except ImportError:
            logger.warning("vetinari.dashboard.agent_dashboard not importable — dashboard endpoint unavailable")
            return litestar_error_response("Dashboard module not available", 503)  # type: ignore[return-value]

        try:
            dashboard = get_agent_dashboard()
            health = dashboard.get_system_health()
        except Exception as exc:
            logger.warning("Could not load dashboard health — returning 503: %s", exc)
            return litestar_error_response("Dashboard health unavailable", 503)  # type: ignore[return-value]

        return {"status": "ok", **health.to_dict()}

    # -- POST /api/v1/dashboard/quality/batch ----------------------------------

    @post("/api/v1/dashboard/quality/batch", media_type=MediaType.JSON)
    async def api_record_quality_batch(
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a batch of quality scores to the drift detector.

        Accepts a JSON body with a ``quality_scores`` list of floats in
        [0.0, 1.0].  More efficient than calling the single-score endpoint
        in a loop.

        Args:
            data: JSON request body containing ``quality_scores`` list.

        Returns:
            JSON object confirming the count of scores recorded.
        """
        if (body := json_object_body(data)) is None:
            return litestar_error_response("Request body must be a JSON object", 400)  # type: ignore[return-value]
        quality_scores = body.get("quality_scores")
        if not isinstance(quality_scores, list):
            return litestar_error_response(  # type: ignore[return-value]
                "quality_scores must be a list of floats", 400
            )

        try:
            quality_scores = [float(s) for s in quality_scores]
        except (TypeError, ValueError) as exc:
            logger.warning("Invalid quality_scores payload — could not convert to float: %s", exc)
            return litestar_error_response(  # type: ignore[return-value]
                "quality_scores must be a list of floats", 400
            )

        try:
            from vetinari.analytics.wiring import record_quality_scores_batch
        except ImportError:
            logger.warning("analytics.wiring not importable — batch quality record unavailable")
            return litestar_error_response("Analytics module not available", 503)  # type: ignore[return-value]

        record_quality_scores_batch(quality_scores)
        return {"status": "ok", "recorded": len(quality_scores)}

    # -- GET /api/v1/dashboard/quality/drift -----------------------------------

    @get("/api/v1/dashboard/quality/drift", media_type=MediaType.JSON)
    async def api_quality_drift_stats() -> dict[str, Any]:
        """Return summary statistics over the quality-score observation window.

        Exposes count, mean, median, stddev, p95, and p99 computed from the
        bounded window maintained by the drift detector.

        Returns:
            JSON object with drift statistics.
        """
        try:
            from vetinari.analytics.wiring import get_quality_drift_stats
        except ImportError:
            logger.warning("analytics.wiring not importable — quality drift stats unavailable")
            return litestar_error_response("Analytics module not available", 503)  # type: ignore[return-value]

        try:
            stats = get_quality_drift_stats()
        except Exception as exc:
            logger.warning("Could not load quality drift stats — returning 503: %s", exc)
            return litestar_error_response("Quality drift stats unavailable", 503)  # type: ignore[return-value]

        return {"status": "ok", **stats}

    # -- GET /api/v1/dashboard/agents/{agent_type} ----------------------------

    @get("/api/v1/dashboard/agents/{agent_type:str}", media_type=MediaType.JSON)
    async def api_dashboard_agent(agent_type: str) -> dict[str, Any]:
        """Return metrics for a specific agent type.

        Args:
            agent_type: The agent type identifier (foreman, worker, inspector).

        Returns:
            JSON response with agent-specific metrics.
        """
        try:
            from vetinari.dashboard.agent_dashboard import get_agent_dashboard
        except ImportError:
            logger.warning("vetinari.dashboard.agent_dashboard not importable — agent metrics endpoint unavailable")
            return litestar_error_response("Dashboard module not available", 503)  # type: ignore[return-value]

        try:
            dashboard = get_agent_dashboard()
            metrics = dashboard.get_agent_metrics(agent_type)
        except Exception as exc:
            logger.warning("Could not load agent metrics for %s — returning 503: %s", agent_type, exc)
            return litestar_error_response("Agent metrics unavailable", 503)  # type: ignore[return-value]

        return {"status": "ok", **metrics.to_dict()}

    # -- GET /api/v1/benchmarks/results/{run_id} -------------------------------

    @get("/api/v1/benchmarks/results/{run_id:str}", media_type=MediaType.JSON)
    async def api_benchmark_results(run_id: str) -> dict[str, Any]:
        """Return all individual case results for a benchmark run.

        Args:
            run_id: The unique benchmark run identifier to fetch results for.

        Returns:
            JSON object with a ``results`` list, or 404 when the run_id does
            not exist in the metric store.
        """
        try:
            from vetinari.benchmarks.runner import MetricStore
        except ImportError:
            logger.warning("benchmarks.runner not importable — benchmark results unavailable")
            return litestar_error_response("Benchmarks module not available", 503)  # type: ignore[return-value]

        try:
            store = MetricStore()
            results = store.load_results(run_id)
            if not results:
                run_summary = store.load_report(run_id)
                if run_summary is None:
                    return litestar_error_response(  # type: ignore[return-value]
                        f"Benchmark run not found: {run_id}", 404
                    )
        except Exception as exc:
            logger.warning("Could not load benchmark results for run %s — returning 503: %s", run_id, exc)
            return litestar_error_response("Benchmark results unavailable", 503)  # type: ignore[return-value]

        return {"run_id": run_id, "results": results}

    # -- GET /api/v1/benchmarks/suites ----------------------------------------

    @get("/api/v1/benchmarks/suites", media_type=MediaType.JSON)
    async def api_benchmark_suites() -> dict[str, Any]:
        """Return metadata for all registered benchmark suites.

        Returns:
            JSON object with a ``suites`` list containing name, layer, tier,
            and description for each registered adapter.
        """
        try:
            from vetinari.benchmarks.runner import get_default_runner
        except ImportError:
            logger.warning("benchmarks.runner not importable — suite listing unavailable")
            return litestar_error_response("Benchmarks module not available", 503)  # type: ignore[return-value]

        try:
            runner = get_default_runner()
            suites = runner.list_suites()
        except Exception as exc:
            logger.warning("Could not load benchmark suites — returning 503: %s", exc)
            return litestar_error_response("Benchmark suites unavailable", 503)  # type: ignore[return-value]

        return {"suites": suites}

    # -- GET /api/v1/benchmarks/suites/{suite_name}/comparison ----------------

    @get("/api/v1/benchmarks/suites/{suite_name:str}/comparison", media_type=MediaType.JSON)
    async def api_benchmark_last_comparison(suite_name: str) -> dict[str, Any]:
        """Compare the two most recent runs for a benchmark suite.

        Args:
            suite_name: Name of the benchmark suite whose runs to compare.

        Returns:
            JSON object with run IDs and per-metric deltas, or 404 when fewer
            than two runs exist for the suite.
        """
        try:
            from vetinari.benchmarks.runner import get_default_runner
        except ImportError:
            logger.warning("benchmarks.runner not importable — comparison unavailable")
            return litestar_error_response("Benchmarks module not available", 503)  # type: ignore[return-value]

        try:
            runner = get_default_runner()
            comparison = runner.get_last_comparison(suite_name)
        except Exception as exc:
            logger.warning("Could not load benchmark comparison for suite %s — returning 503: %s", suite_name, exc)
            return litestar_error_response("Benchmark comparison unavailable", 503)  # type: ignore[return-value]

        if comparison is None:
            return litestar_error_response(  # type: ignore[return-value]
                f"Not enough runs to compare for suite: {suite_name}", 404
            )
        return {
            "suite_name": suite_name,
            "run_a": comparison.run_a,
            "run_b": comparison.run_b,
            "delta_pass_at_1": comparison.delta_pass_at_1,
            "delta_avg_score": comparison.delta_avg_score,
            "delta_avg_latency_ms": comparison.delta_avg_latency_ms,
            "delta_total_tokens": comparison.delta_total_tokens,
            "delta_cost_usd": comparison.delta_cost_usd,
            "regressions": comparison.regressions,
            "improvements": comparison.improvements,
        }

    # -- POST /api/v1/dashboard/quality/scores-batch ---------------------------

    @post("/api/v1/dashboard/quality/scores-batch", media_type=MediaType.JSON)
    async def api_dashboard_quality_scores_batch(
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a batch of quality scores for drift-detection analysis.

        Accepts a JSON array of float scores so the caller can submit multiple
        inspection results in a single request instead of one call per task.

        Args:
            data: JSON request body with ``scores`` list of floats.

        Returns:
            JSON confirmation with the count of scores accepted.
        """
        from vetinari.analytics.wiring import record_quality_scores_batch

        if data is None:
            return litestar_error_response(  # type: ignore[return-value]
                "Request body must be a JSON object with a 'scores' key",
                400,
            )
        if "scores" not in data:
            return litestar_error_response(  # type: ignore[return-value]
                "Request body must contain a 'scores' key with a list of floats",
                400,
            )
        raw_scores = data["scores"]
        if not isinstance(raw_scores, list):
            return litestar_error_response("scores must be a list", 400)  # type: ignore[return-value]

        scores = [float(s) for s in raw_scores if isinstance(s, (int, float))]
        record_quality_scores_batch(scores)
        logger.info("Recorded quality scores batch of %d items via dashboard API", len(scores))
        return {"status": "ok", "accepted": len(scores)}

    # -- GET /api/v1/dashboard/quality/drift-stats ----------------------------

    @get("/api/v1/dashboard/quality/drift-stats", media_type=MediaType.JSON)
    async def api_dashboard_quality_drift_stats() -> dict[str, Any]:
        """Return current quality drift statistics for trend analysis.

        Exposes mean, standard deviation, and control-chart limits computed
        from the rolling quality score history so the dashboard can visualise
        whether quality is drifting outside acceptable bounds.

        Returns:
            JSON object with drift statistics from the analytics wiring layer.
        """
        try:
            from vetinari.analytics.wiring import get_quality_drift_stats
        except ImportError:
            logger.warning("analytics.wiring not importable — quality drift stats unavailable")
            return litestar_error_response("Analytics module not available", 503)  # type: ignore[return-value]

        try:
            stats = get_quality_drift_stats()
        except Exception as exc:
            logger.warning("Could not load quality drift stats — returning 503: %s", exc)
            return litestar_error_response("Quality drift stats unavailable", 503)  # type: ignore[return-value]

        return {"status": "ok", **stats}

    # -- GET /api/v1/pipeline/status ------------------------------------------

    @get("/api/v1/pipeline/status", media_type=MediaType.JSON)
    async def api_pipeline_status() -> dict[str, Any]:
        """Return pipeline stage run-time stats aggregated from the last 24 hours.

        Queries the ``sse_event_log`` table for stage_started and
        stage_completed events and computes entry/exit counts and drop rate
        per stage.  Determines which stage is currently active by looking for
        unmatched stage_started events in the last 5 minutes.

        Returns:
            JSON object with ``stages`` list and ``active_stage`` string (or None),
            or 503 when the database is unavailable.
        """
        from datetime import datetime, timedelta, timezone

        from vetinari.database import get_connection

        cutoff_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        cutoff_5min = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()

        pipeline_stages = ["plan", "execute", "verify"]
        stage_data: dict[str, dict[str, int]] = {s: {"entry_count": 0, "exit_count": 0} for s in pipeline_stages}
        active_stage: str | None = None

        try:
            conn = get_connection()
            rows = conn.execute(
                """
                SELECT
                    json_extract(data, '$.stage') AS stage,
                    COUNT(*) AS cnt
                FROM sse_event_log
                WHERE event_type = 'stage_started'
                  AND timestamp >= ?
                GROUP BY stage
                """,
                (cutoff_24h,),
            ).fetchall()
            for row in rows:
                stage_name = row[0]
                if stage_name in stage_data:
                    stage_data[stage_name]["entry_count"] = row[1]

            rows = conn.execute(
                """
                SELECT
                    json_extract(data, '$.stage') AS stage,
                    COUNT(*) AS cnt
                FROM sse_event_log
                WHERE event_type = 'stage_completed'
                  AND timestamp >= ?
                GROUP BY stage
                """,
                (cutoff_24h,),
            ).fetchall()
            for row in rows:
                stage_name = row[0]
                if stage_name in stage_data:
                    stage_data[stage_name]["exit_count"] = row[1]

            active_rows = conn.execute(
                """
                SELECT DISTINCT json_extract(data, '$.stage') AS stage
                FROM sse_event_log
                WHERE event_type = 'stage_started'
                  AND timestamp >= ?
                """,
                (cutoff_5min,),
            ).fetchall()
            recent_started = {r[0] for r in active_rows if r[0]}

            completed_rows = conn.execute(
                """
                SELECT DISTINCT json_extract(data, '$.stage') AS stage
                FROM sse_event_log
                WHERE event_type = 'stage_completed'
                  AND timestamp >= ?
                """,
                (cutoff_5min,),
            ).fetchall()
            recent_completed = {r[0] for r in completed_rows if r[0]}

            unmatched = recent_started - recent_completed
            if unmatched:
                for s in reversed(pipeline_stages):
                    if s in unmatched:
                        active_stage = s
                        break

        except Exception as exc:
            logger.warning(
                "Could not query sse_event_log for pipeline status — returning 503: %s",
                exc,
            )
            return litestar_error_response(  # type: ignore[return-value]
                "Pipeline status unavailable — database error", 503
            )

        stages = []
        for s in pipeline_stages:
            entry = stage_data[s]["entry_count"]
            exit_count = stage_data[s]["exit_count"]
            drop_rate = 1.0 - (exit_count / max(entry, 1))
            if s == active_stage:
                status = "active"
            elif exit_count > 0:
                status = "complete"
            else:
                status = "idle"
            stages.append({
                "name": s,
                "status": status,
                "entry_count": entry,
                "exit_count": exit_count,
                "drop_rate": round(drop_rate, 4),
            })

        return {"status": "ok", "stages": stages, "active_stage": active_stage}

    # -- GET /api/v1/dashboard/model-health ------------------------------------

    @get("/api/v1/dashboard/model-health", media_type=MediaType.JSON)
    async def api_model_health() -> dict[str, Any]:
        """Return model health gauge readings based on quality drift detection.

        Maps drift severity to traffic-light levels: green (no drift), amber
        (one detector triggered), or red (two or more detectors triggered).

        Returns:
            JSON object with ``input_drift``, ``behavior_drift``, and
            ``quality_drift`` gauge readings, each containing ``level``,
            ``value``, and ``detectors_triggered``.
        """
        try:
            from vetinari.analytics.quality_drift import get_drift_ensemble
            from vetinari.analytics.wiring import get_quality_drift_stats
        except ImportError as exc:
            logger.warning(
                "Drift detection modules not importable — model-health endpoint unavailable: %s",
                exc,
            )
            return litestar_error_response("Drift detection module not available", 503)  # type: ignore[return-value]

        try:
            stats = get_quality_drift_stats()
        except Exception as exc:
            logger.warning(
                "Could not read quality drift stats — returning 503: %s",
                exc,
            )
            return litestar_error_response(  # type: ignore[return-value]
                "Model health unavailable — quality drift stats error", 503
            )
        stddev = float(stats.get("stddev", 0.0))
        observation_count = int(stats.get("count", 0))

        def _level(is_drift: bool, triggered: list[str]) -> str:
            """Map drift state to a traffic-light level string.

            Args:
                is_drift: Whether drift has been detected.
                triggered: List of detector names that fired.

            Returns:
                One of ``"green"``, ``"amber"``, or ``"red"``.
            """
            if not is_drift:
                return "green"
            return "red" if len(triggered) >= 2 else "amber"

        triggered: list[str] = []
        is_drift = False
        quality_drift_value = 0.0
        try:
            ensemble = get_drift_ensemble()
            cusum = ensemble._cusum
            ph = ensemble._page_hinkley
            adwin = ensemble._adwin

            cusum_pressure = max(cusum._s_pos, cusum._s_neg) / max(cusum._threshold, 1e-9)
            ph_pressure = (ph._sum - ph._min_sum) / max(ph._threshold, 1e-9) if ph._min_sum != float("inf") else 0.0
            adwin_pressure = (1.0 - (adwin._window[-1] if adwin._window else 0.0)) if adwin._window else 0.0

            if cusum_pressure >= 1.0:
                triggered.append("cusum")
            if ph_pressure >= 1.0:
                triggered.append("page_hinkley")
            if adwin_pressure >= 0.5 and adwin._count >= adwin._min_samples:
                triggered.append("adwin")

            is_drift = len(triggered) >= 2
            quality_drift_value = min((cusum_pressure + ph_pressure + adwin_pressure) / 3.0, 1.0)
        except Exception as exc:
            logger.warning(
                "Could not read drift detector state — returning 503: %s",
                exc,
            )
            return litestar_error_response(  # type: ignore[return-value]
                "Model health unavailable — drift detector error", 503
            )

        input_drift_value = min(stddev, 1.0)
        behavior_drift_value = min(observation_count / 1000.0, 1.0)

        return {
            "status": "ok",
            "input_drift": {
                "level": _level(is_drift, triggered),
                "value": round(input_drift_value, 4),
                "detectors_triggered": triggered if is_drift else [],
            },
            "behavior_drift": {
                "level": _level(is_drift, triggered),
                "value": round(behavior_drift_value, 4),
                "detectors_triggered": triggered if is_drift else [],
            },
            "quality_drift": {
                "level": _level(is_drift, triggered),
                "value": round(quality_drift_value, 4),
                "detectors_triggered": triggered,
            },
        }

    # -- GET /api/v1/dashboard/welcome-back ------------------------------------

    @get("/api/v1/dashboard/welcome-back", media_type=MediaType.JSON)
    async def api_welcome_back(
        since_iso: str | None = Parameter(query="since_iso", default=None),
    ) -> dict[str, Any]:
        """Return a summary of activity since the user's last visit.

        Args:
            since_iso: ISO timestamp string marking the start of the absent
                period.  Defaults to 1 hour ago when omitted.

        Returns:
            JSON object with ``projects_completed``, ``quality_trend``,
            ``new_models``, ``learning_improvements``, ``needs_attention``,
            and ``show_summary`` fields.
        """
        from datetime import datetime, timedelta, timezone

        from vetinari.database import get_connection

        if since_iso:
            try:
                since_dt = datetime.fromisoformat(since_iso.replace("Z", "+00:00"))
                since_ts = since_dt.isoformat()
            except ValueError as exc:
                logger.warning("Invalid since_iso parameter %r: %s", since_iso, exc)
                return litestar_error_response(  # type: ignore[return-value]
                    "Invalid since_iso — must be ISO 8601 format", 400
                )
        else:
            since_ts = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

        now = datetime.now(timezone.utc)
        week_ago = (now - timedelta(days=7)).isoformat()
        two_weeks_ago = (now - timedelta(days=14)).isoformat()

        projects_completed = 0
        quality_trend = "stable"
        new_models: list[str] = []
        learning_improvements: list[str] = []
        needs_attention: list[str] = []

        try:
            conn = get_connection()

            row = conn.execute(
                "SELECT COUNT(*) FROM pipeline_traces WHERE completed_at > ? AND status = 'completed'",
                (since_ts,),
            ).fetchone()
            projects_completed = row[0] if row else 0

            row_last = conn.execute(
                "SELECT AVG(quality_score) FROM pipeline_traces"
                " WHERE started_at >= ? AND started_at < ? AND quality_score IS NOT NULL",
                (week_ago, now.isoformat()),
            ).fetchone()
            row_prev = conn.execute(
                "SELECT AVG(quality_score) FROM pipeline_traces"
                " WHERE started_at >= ? AND started_at < ? AND quality_score IS NOT NULL",
                (two_weeks_ago, week_ago),
            ).fetchone()
            avg_last = row_last[0] if row_last and row_last[0] is not None else None
            avg_prev = row_prev[0] if row_prev and row_prev[0] is not None else None
            if avg_last is not None and avg_prev is not None:
                diff = avg_last - avg_prev
                if diff > 0.02:
                    quality_trend = "up"
                elif diff < -0.02:
                    quality_trend = "down"

            before_models_rows = conn.execute(
                "SELECT DISTINCT model_id FROM pipeline_traces WHERE started_at < ? AND model_id IS NOT NULL",
                (since_ts,),
            ).fetchall()
            before_models = {r[0] for r in before_models_rows if r[0]}

            after_models_rows = conn.execute(
                "SELECT DISTINCT model_id FROM pipeline_traces WHERE started_at > ? AND model_id IS NOT NULL",
                (since_ts,),
            ).fetchall()
            new_models = [r[0] for r in after_models_rows if r[0] and r[0] not in before_models]

            try:
                count_row = conn.execute(
                    "SELECT COUNT(*) FROM training_records WHERE created_at > ?",
                    (since_ts,),
                ).fetchone()
                training_count = count_row[0] if count_row else 0
                if training_count > 0:
                    learning_improvements = [f"{training_count} new training examples recorded"]
            except Exception as exc:
                logger.warning(
                    "training_records table not queryable — skipping learning improvements: %s",
                    exc,
                )

            low_quality_row = conn.execute(
                "SELECT COUNT(*) FROM pipeline_traces"
                " WHERE started_at > ? AND quality_score < 0.5 AND quality_score IS NOT NULL",
                (since_ts,),
            ).fetchone()
            if low_quality_row and low_quality_row[0] > 0:
                needs_attention = ["Quality below threshold on recent tasks"]

        except Exception as exc:
            logger.warning(
                "Could not query pipeline_traces for welcome-back summary — returning 503: %s",
                exc,
            )
            return litestar_error_response(  # type: ignore[return-value]
                "Welcome-back summary unavailable — database error", 503
            )

        is_noteworthy = bool(
            projects_completed
            or (quality_trend and quality_trend != "stable")
            or new_models
            or learning_improvements
            or needs_attention
        )
        return {
            "status": "ok",
            "show_summary": is_noteworthy,
            "projects_completed": projects_completed,
            "quality_trend": quality_trend,
            "new_models": new_models,
            "new_models_discovered": new_models,
            "learning_improvements": learning_improvements,
            "learning_improvements_applied": learning_improvements,
            "needs_attention": needs_attention,
        }

    # -- GET /api/v1/dashboard/events/stream (SSE) ----------------------------

    @get("/api/v1/dashboard/events/stream", sync_to_thread=False)
    def api_dashboard_events_stream() -> ServerSentEvent:
        """Stream dashboard-level SSE events for live panel updates.

        Polls ``sse_event_log`` for ``pipeline_stage`` and ``quality_result``
        events and delivers them as typed SSE frames so the pipeline
        visualization panel can refresh without a full page reload.
        Bootstraps the most recent 10 events on connect so the client has
        data immediately.  Keepalive comments are emitted every 5 s when no
        new events are available.  The stream closes automatically after
        ``_DASHBOARD_STREAM_MAX_IDLE`` idle cycles (~10 min).

        Returns:
            A ``ServerSentEvent`` streaming response.
        """

        async def _generate():
            """Yield SSE frames polled from the sse_event_log table.

            Yields:
                Dicts with ``event`` + ``data`` keys for real events, or
                ``comment`` key for keepalives.
            """
            import asyncio
            import json

            from vetinari.database import get_connection

            _STREAMED_TYPES = ("pipeline_stage", "quality_result")

            last_id = 0
            idle_cycles = 0

            # Bootstrap: send the most recent events so the client has data immediately.
            try:
                conn = get_connection()
                placeholders = ",".join("?" * len(_STREAMED_TYPES))
                bootstrap_rows = conn.execute(
                    f"SELECT id, event_type, data FROM sse_event_log"  # noqa: S608 - SQL identifiers are constrained while values stay parameterized
                    f" WHERE event_type IN ({placeholders}) ORDER BY id DESC LIMIT 10",
                    _STREAMED_TYPES,
                ).fetchall()
                for row in reversed(bootstrap_rows):
                    last_id = max(last_id, row[0])
                    payload = row[2] if isinstance(row[2], str) else json.dumps(row[2])
                    yield {"event": row[1], "data": payload}
            except Exception as exc:
                logger.warning(
                    "Dashboard events stream: database unavailable — closing stream: %s",
                    exc,
                )
                yield {"event": "error", "data": json.dumps({"error": "Database unavailable", "code": 503})}
                return

            try:
                while idle_cycles < _DASHBOARD_STREAM_MAX_IDLE:
                    try:
                        conn = get_connection()
                        placeholders = ",".join("?" * len(_STREAMED_TYPES))
                        rows = conn.execute(
                            f"SELECT id, event_type, data FROM sse_event_log"  # noqa: S608 - SQL identifiers are constrained while values stay parameterized
                            f" WHERE event_type IN ({placeholders}) AND id > ?"
                            f" ORDER BY id LIMIT 20",
                            (*_STREAMED_TYPES, last_id),
                        ).fetchall()
                        if rows:
                            idle_cycles = 0
                            for row in rows:
                                last_id = row[0]
                                payload = row[2] if isinstance(row[2], str) else json.dumps(row[2])
                                yield {"event": row[1], "data": payload}
                        else:
                            idle_cycles += 1
                            yield {"comment": "keepalive"}
                    except Exception as exc:
                        logger.warning(
                            "Dashboard events stream: poll error (id=%d) — skipping cycle: %s",
                            last_id,
                            exc,
                        )
                        yield {"comment": "keepalive"}
                    await asyncio.sleep(5)
            except GeneratorExit:  # noqa: VET022 — expected on client disconnect
                pass

        return ServerSentEvent(
            _generate(),
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return [
        api_dashboard,
        api_dashboard_health,
        api_record_quality_batch,
        api_quality_drift_stats,
        api_dashboard_agent,
        api_benchmark_results,
        api_benchmark_suites,
        api_benchmark_last_comparison,
        api_dashboard_quality_scores_batch,
        api_dashboard_quality_drift_stats,
        api_pipeline_status,
        api_model_health,
        api_welcome_back,
        api_dashboard_events_stream,
    ]
