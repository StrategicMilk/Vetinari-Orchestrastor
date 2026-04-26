"""Litestar handlers for the training pipeline API, part 2. Native Litestar equivalents (ADR-0066).

Continuation of ``litestar_training_api.py``.  Covers lifecycle control,
data management, curriculum, history, job listing, summary, quality
comparison, model/adapter endpoints, and exposes the combined
``create_training_api_handlers()`` factory used by ``litestar_app.py``.

This module (part 2) covers:
    POST /api/v1/training/pause             — pause current training (admin)
    POST /api/v1/training/resume            — resume paused training (admin)
    POST /api/v1/training/stop              — stop training, alias for pause (admin)
    GET  /api/v1/training/data/stats        — training data statistics
    POST /api/v1/training/data/seed         — trigger seed dataset download (admin)
    GET  /api/v1/training/data/seed/stream  — SSE stream for seed download progress
    GET  /api/v1/training/curriculum        — curriculum status and next activity
    GET  /api/v1/training/curriculum/next   — next planned activity only
    GET  /api/v1/training/history           — training history from AgentTrainer
    GET  /api/v1/training/jobs              — training jobs from TrainingManager
    GET  /api/v1/training/summary           — unified training status snapshot
    GET  /api/v1/training/quality           — latest quality gate comparison
    GET  /api/v1/training/models            — registered LoRA adapter list
    GET  /api/v1/training/adapters          — list adapters by task type (?task_type=X)
    GET  /api/v1/training/adapters/deployed — list all deployed adapters
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Response, get, post
    from litestar.params import Parameter
    from litestar.response import Stream

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def _create_training_api_handlers_part2() -> list[Any]:
    """Create the second batch of training API route handlers.

    Covers lifecycle control (pause/resume/stop), data management, curriculum,
    history, jobs, summary, quality, models, and adapter endpoints.
    Called by ``create_training_api_handlers()`` in this module.

    Returns:
        List of 15 Litestar route handler functions.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.litestar_training_api import _get_scheduler

    @post("/api/v1/training/pause", media_type=MediaType.JSON, guards=[admin_guard])
    async def training_pause(data: dict[str, Any] | None = None) -> Response | dict[str, Any]:
        """Pause current training activity.

        Calls ``pause_for_user_request()`` on the TrainingScheduler singleton.
        Safe to call at any time — returns success even when no scheduler is
        running.  This endpoint accepts no body parameters; any non-empty body
        is rejected with 422.

        Args:
            data: Request body — must be empty (``{}``).

        Returns:
            JSON with status and paused flag.
        """
        from vetinari.web.responses import litestar_error_response

        if data is not None:
            return litestar_error_response("This endpoint takes no request body parameters", code=422)
        scheduler = _get_scheduler()
        if scheduler is None:
            return Response(
                content={"status": "error", "message": "Training scheduler not available"},
                status_code=503,
                media_type=MediaType.JSON,
            )

        try:
            scheduler.pause_for_user_request()
        except Exception as exc:
            logger.warning("training/pause: scheduler pause failed — returning 500: %s", exc)
            return litestar_error_response("Training scheduler pause failed — check server logs", code=500)

        return {"status": "ok", "paused": True}

    @post("/api/v1/training/resume", media_type=MediaType.JSON, guards=[admin_guard])
    async def training_resume(data: dict[str, Any] | None = None) -> Response | dict[str, Any]:
        """Resume a paused training activity.

        Calls ``resume_after_user_request()`` on the TrainingScheduler.  The
        scheduler only actually resumes if the system is still idle.  This
        endpoint accepts no body parameters; any non-empty body is rejected
        with 422.

        Args:
            data: Request body — must be empty (``{}``).

        Returns:
            JSON with status and resumed flag.
        """
        from vetinari.web.responses import litestar_error_response

        if data is not None:
            return litestar_error_response("This endpoint takes no request body parameters", code=422)
        scheduler = _get_scheduler()
        if scheduler is None:
            return Response(
                content={"status": "error", "message": "Training scheduler not available"},
                status_code=503,
                media_type=MediaType.JSON,
            )

        try:
            scheduler.resume_after_user_request()
        except Exception as exc:
            logger.warning("training/resume: scheduler resume failed — returning 500: %s", exc)
            return litestar_error_response("Training scheduler resume failed — check server logs", code=500)

        return {"status": "ok", "resumed": True}

    @post("/api/v1/training/stop", media_type=MediaType.JSON, guards=[admin_guard])
    async def training_stop(data: dict[str, Any] | None = None) -> Response | dict[str, Any]:
        """Stop current training activity (alias for pause).

        The frontend uses 'stop' semantics while the backend scheduler only
        distinguishes pause/resume.  Delegates to ``pause_for_user_request()``.
        This endpoint accepts no body parameters; any non-empty body is
        rejected with 422.

        Args:
            data: Request body — must be empty (``{}``).

        Returns:
            JSON with status and stopped flag.
        """
        from vetinari.web.responses import litestar_error_response

        if data is not None:
            return litestar_error_response("This endpoint takes no request body parameters", code=422)
        scheduler = _get_scheduler()
        if scheduler is None:
            return Response(
                content={"status": "error", "message": "Training scheduler not available"},
                status_code=503,
                media_type=MediaType.JSON,
            )

        try:
            scheduler.pause_for_user_request()
        except Exception as exc:
            logger.warning("training/stop: scheduler stop failed — returning 500: %s", exc)
            return litestar_error_response("Training scheduler stop failed — check server logs", code=500)

        return {"status": "ok", "stopped": True}

    @get("/api/v1/training/data/stats", media_type=MediaType.JSON)
    async def training_data_stats() -> Response | dict[str, Any]:
        """Return training data statistics from seeder and collector.

        Queries ``TrainingDataSeeder.get_seed_status()`` for seeded dataset
        information.  When the ``TrainingDataCollector`` is available its
        record count is also included.

        Returns:
            JSON with status and nested ``seed`` and ``collector`` stats.
        """
        from vetinari.web.responses import litestar_error_response

        seed_status: dict[str, Any] = {}
        collector_stats: dict[str, Any] = {}
        any_subsystem_available = False

        try:
            from vetinari.training.data_seeder import get_training_data_seeder

            seeder = get_training_data_seeder()
            seed_status = seeder.get_seed_status()
            any_subsystem_available = True
        except Exception as exc:
            logger.warning("training/data/stats: seeder unavailable: %s", exc)

        try:
            from vetinari.learning.training_data import get_training_collector

            collector = get_training_collector()
            if hasattr(collector, "get_stats"):
                collector_stats = collector.get_stats()
            elif hasattr(collector, "count_records"):
                collector_stats = {"record_count": collector.count_records()}
            any_subsystem_available = True
        except Exception as exc:
            logger.warning("training/data/stats: collector unavailable: %s", exc)

        if not any_subsystem_available:
            return litestar_error_response(  # type: ignore[return-value]
                "Training data subsystem unavailable — seeder and collector both unreachable", 503
            )

        return {
            "status": "ok",
            "data": {
                "seed": seed_status,
                "collector": collector_stats,
            },
        }

    @post("/api/v1/training/data/seed", media_type=MediaType.JSON, guards=[admin_guard])
    async def training_data_seed(data: dict[str, Any] | None = None) -> Response | dict[str, Any]:
        """Trigger seed dataset download.

        Calls ``TrainingDataSeeder.seed_if_empty()`` to download curated seed
        datasets.  Returns 0 when data already exists and no download was needed.
        This endpoint accepts no body parameters; any non-empty body is
        rejected with 422.

        Args:
            data: Request body — must be empty (``{}``).

        Returns:
            JSON with status and count of datasets seeded.
        """
        from vetinari.web.responses import litestar_error_response

        if data is not None:
            return litestar_error_response("This endpoint takes no request body parameters", code=422)
        try:
            from vetinari.training.data_seeder import get_training_data_seeder

            seeder = get_training_data_seeder()
            count_seeded = seeder.seed_if_empty()
        except Exception as exc:
            logger.warning("training/data/seed: seeder unavailable — returning 503: %s", exc)
            return litestar_error_response("Training data seeder unavailable", code=503)
        return {"status": "ok", "count_seeded": count_seeded}

    @get("/api/v1/training/data/seed/stream")
    async def training_data_seed_stream() -> Stream | Response:
        """SSE endpoint for seed dataset download with real-time progress.

        Streams progress events as the seeder downloads each dataset.  Each
        event is a JSON object with keys like ``event``, ``percent``,
        ``eta_seconds``, ``dataset``, and ``status``.

        Returns:
            SSE Stream response streaming progress events, or a 503 Response
            when the seeder is unavailable.
        """
        from vetinari.training.data_seeder import get_training_data_seeder
        from vetinari.web.responses import litestar_error_response

        try:
            seeder = get_training_data_seeder()
            progress_gen = seeder.seed_with_progress()
        except Exception as exc:
            logger.warning("training/data/seed/stream: seeder unavailable — returning 503: %s", exc)
            return litestar_error_response("Training data seeder unavailable", 503)

        async def event_generator() -> AsyncGenerator[bytes, None]:
            """Wrap the sync progress iterator as an async SSE byte stream."""
            try:
                for event in progress_gen:
                    yield f"data: {json.dumps(event)}\n\n".encode()
            except GeneratorExit:  # noqa: VET022 - best-effort optional path must not fail the primary flow
                pass
            except Exception:
                logger.exception("SSE seed stream error")
                yield (f"data: {json.dumps({'event': 'error', 'error': 'Internal server error'})}\n\n").encode()

        return Stream(event_generator(), media_type="text/event-stream")

    @get("/api/v1/training/curriculum", media_type=MediaType.JSON)
    async def training_curriculum() -> Response | dict[str, Any]:
        """Return curriculum status and next planned activity.

        Returns:
            JSON with status, curriculum phase, candidate count, and next
            activity description.  Returns 503 when the curriculum module is
            unavailable.
        """
        from vetinari.web.responses import litestar_error_response

        try:
            from vetinari.training.curriculum import get_training_curriculum

            curriculum = get_training_curriculum()
            curriculum_status = curriculum.get_status()
            return {"status": "ok", "curriculum": curriculum_status}
        except Exception as exc:
            logger.warning("training/curriculum: curriculum module unavailable, returning 503: %s", exc)
            return litestar_error_response(  # type: ignore[return-value]
                "Training curriculum subsystem unavailable", 503
            )

    @get("/api/v1/training/curriculum/next", media_type=MediaType.JSON)
    async def training_curriculum_next() -> Response | dict[str, Any]:
        """Return details of the next planned training activity only.

        Returns:
            JSON with status and next activity detail including type,
            description, hypothesis, metric, priority, and estimated duration.
            Returns 503 when the curriculum module is unavailable.
        """
        from vetinari.web.responses import litestar_error_response

        try:
            from vetinari.training.curriculum import get_training_curriculum

            curriculum = get_training_curriculum()
            activity = curriculum.next_activity()
            next_detail = {
                "type": activity.type.value,
                "description": activity.description,
                "hypothesis": activity.hypothesis,
                "metric": activity.metric,
                "priority": activity.priority,
                "estimated_duration_minutes": activity.estimated_duration_minutes,
                "estimated_vram_gb": activity.estimated_vram_gb,
            }
            return {"status": "ok", "next_activity": next_detail}
        except Exception as exc:
            logger.warning("training/curriculum/next: curriculum module unavailable, returning 503: %s", exc)
            return litestar_error_response(  # type: ignore[return-value]
                "Training curriculum subsystem unavailable", 503
            )

    @get("/api/v1/training/history", media_type=MediaType.JSON)
    async def training_history() -> Response | dict[str, Any]:
        """Return training history from AgentTrainer.

        Queries ``AgentTrainer.get_stats()`` for per-agent training run counts,
        timestamps, and latest model paths.  Also includes the current training
        priority ranking.

        Returns:
            JSON with status, per-agent stats, and ordered priority ranking.
            Returns 503 when the AgentTrainer is unavailable.
        """
        from vetinari.web.responses import litestar_error_response

        try:
            from vetinari.training.agent_trainer import get_agent_trainer

            trainer = get_agent_trainer()
            agent_stats = trainer.get_stats()
            priority_ranking = [
                {"agent": name, "priority_score": score} for name, score in trainer.get_training_priority()
            ]
            return {
                "status": "ok",
                "history": {
                    "agents": agent_stats,
                    "priority_ranking": priority_ranking,
                },
            }
        except Exception as exc:
            logger.warning("training/history: AgentTrainer unavailable, returning 503: %s", exc)
            return litestar_error_response(  # type: ignore[return-value]
                "Training history subsystem unavailable", 503
            )

    @get("/api/v1/training/jobs", media_type=MediaType.JSON)
    async def training_jobs() -> Response | dict[str, Any]:
        """Return the list of training jobs from the TrainingManager.

        Queries ``get_training_manager().list_jobs()`` for all known training jobs
        and serialises each job's identifier, status, and timestamps.

        Returns:
            JSON with status and a list of job records.  Returns 503 when
            TrainingManager is unavailable.
        """
        from vetinari.web.responses import litestar_error_response

        try:
            from vetinari.learning.training_manager import get_training_manager

            manager = get_training_manager()
            jobs = manager.list_jobs()
            serialised = []
            for job in jobs:
                entry: dict[str, Any] = {}
                for attr in ("job_id", "status", "agent_type", "started_at", "finished_at", "error"):
                    val = getattr(job, attr, None)
                    if val is not None:
                        entry[attr] = str(val) if not isinstance(val, (str, int, float, bool)) else val
                serialised.append(entry)
            return {"status": "ok", "jobs": serialised}
        except Exception as exc:
            logger.warning("training/jobs: TrainingManager unavailable, returning 503: %s", exc)
            return litestar_error_response(  # type: ignore[return-value]
                "Training jobs subsystem unavailable", 503
            )

    @get("/api/v1/training/summary", media_type=MediaType.JSON)
    async def training_summary() -> Response | dict[str, Any]:
        """Return a unified training status snapshot.

        Delegates to the module-level ``get_training_status()`` utility so the
        same data is available both via HTTP and to internal callers such as the
        dashboard module.  Returns 503 when all training subsystems are
        unavailable so callers can distinguish unavailable from idle.

        Returns:
            JSON object with keys: status, current_job, last_run,
            records_collected, curriculum_phase, and next_activity.
            Returns 503 when no training subsystem is reachable.
        """
        from vetinari.web.litestar_training_api import get_training_status
        from vetinari.web.responses import litestar_error_response

        try:
            result = get_training_status()
        except Exception as exc:
            logger.warning("training/summary: get_training_status failed, returning 503: %s", exc)
            return litestar_error_response(  # type: ignore[return-value]
                "Training summary subsystem unavailable", 503
            )

        # get_training_status() returns sentinel values when all subsystems fail.
        # Detect this by checking whether records are zero AND phase is "unknown"
        # AND status is "idle" — all three together indicate no subsystem responded.
        if (
            result.get("status") == "idle"
            and result.get("curriculum_phase") == "unknown"
            and result.get("records_collected") == 0
            and result.get("current_job") is None
        ):
            return litestar_error_response(  # type: ignore[return-value]
                "Training summary subsystem unavailable — no training modules could be loaded", 503
            )

        return result

    @get("/api/v1/training/quality", media_type=MediaType.JSON)
    async def training_quality() -> Response | dict[str, Any]:
        """Return the latest quality gate comparison between baseline and candidate.

        Delegates to the module-level ``get_quality_comparison()`` utility.
        Returns 503 when the quality gate is unavailable or has no history.

        Returns:
            JSON object with keys: baseline_quality, candidate_quality,
            quality_delta, decision, and latency_ratio.  Returns 503 when
            the quality gate is unreachable.
        """
        from vetinari.web.litestar_training_api import get_quality_comparison
        from vetinari.web.responses import litestar_error_response

        try:
            result = get_quality_comparison()
        except Exception as exc:
            logger.warning("training/quality: get_quality_comparison failed, returning 503: %s", exc)
            return litestar_error_response(  # type: ignore[return-value]
                "Training quality gate subsystem unavailable", 503
            )

        # get_quality_comparison() returns sentinel with decision="no_data" when
        # the quality gate is unavailable or its history is empty.
        if result.get("decision") == "no_data":
            return litestar_error_response(  # type: ignore[return-value]
                "Training quality gate unavailable — no comparison data exists", 503
            )

        return result

    @get("/api/v1/training/models", media_type=MediaType.JSON)
    async def training_models() -> Response | dict[str, Any]:
        """Return list of trained model adapters from the adapter registry.

        Queries ``get_adapter_registry().list_all()``.  Each entry includes the
        adapter ID, task type, path, eval score, and deployment status.

        Returns:
            JSON with status and list of adapter registrations.  Returns 503
            when the adapter registry is unavailable.
        """
        from vetinari.web.responses import litestar_error_response

        try:
            from vetinari.training.adapter_registry import get_adapter_registry

            registry = get_adapter_registry()
            records = registry.list_all()
            adapters = [
                {
                    "adapter_id": r.adapter_id,
                    "task_type": r.task_type,
                    "adapter_path": r.adapter_path,
                    "eval_score": r.eval_score,
                    "deployment_status": r.deployment_status,
                }
                for r in records
            ]
            return {
                "status": "ok",
                "models": {
                    "total": len(adapters),
                    "adapters": adapters,
                },
            }
        except Exception as exc:
            logger.warning("training/models: adapter registry unavailable, returning 503: %s", exc)
            return litestar_error_response(  # type: ignore[return-value]
                "Adapter registry subsystem unavailable", 503
            )

    @get("/api/v1/training/adapters", media_type=MediaType.JSON)
    async def training_adapters_by_task_type(
        task_type: str = Parameter(query="task_type", default=""),
    ) -> Response | dict[str, Any]:
        """List adapter records filtered by task type.

        Args:
            task_type: Task domain to filter by (required query parameter).

        Returns:
            JSON with ``status`` and ``adapters`` list.  Returns 400 when
            ``task_type`` query parameter is missing.
        """
        from vetinari.web.responses import litestar_error_response

        stripped = task_type.strip()
        if not stripped:
            return Response(
                content={"status": "error", "error": "task_type query parameter is required"},
                status_code=400,
                media_type=MediaType.JSON,
            )

        try:
            from vetinari.training.adapter_registry import get_adapter_registry

            registry = get_adapter_registry()
            records = registry.list_by_task_type(stripped)
            return {
                "status": "ok",
                "task_type": stripped,
                "adapters": [
                    {
                        "adapter_id": r.adapter_id,
                        "base_model": r.base_model,
                        "task_type": r.task_type,
                        "adapter_path": r.adapter_path,
                        "training_date": r.training_date,
                        "eval_score": r.eval_score,
                        "deployment_status": r.deployment_status,
                    }
                    for r in records
                ],
            }
        except Exception as exc:
            logger.warning("training/adapters: adapter registry unavailable, returning 503: %s", exc)
            return litestar_error_response(  # type: ignore[return-value]
                "Adapter registry subsystem unavailable", 503
            )

    @get("/api/v1/training/adapters/deployed", media_type=MediaType.JSON)
    async def training_adapters_deployed() -> Response | dict[str, Any]:
        """List all currently deployed adapters from the registry.

        Returns only adapters whose ``deployment_status`` is ``"deployed"``,
        sorted by training date descending.

        Returns:
            JSON with ``status`` and ``adapters`` list.  Returns 503 when
            the adapter registry is unavailable.
        """
        from vetinari.web.responses import litestar_error_response

        try:
            from vetinari.training.adapter_registry import get_adapter_registry

            registry = get_adapter_registry()
            records = registry.list_deployed()
            return {
                "status": "ok",
                "adapters": [
                    {
                        "adapter_id": r.adapter_id,
                        "base_model": r.base_model,
                        "task_type": r.task_type,
                        "adapter_path": r.adapter_path,
                        "training_date": r.training_date,
                        "eval_score": r.eval_score,
                        "deployment_status": r.deployment_status,
                    }
                    for r in records
                ],
            }
        except Exception as exc:
            logger.warning("training/adapters/deployed: adapter registry unavailable, returning 503: %s", exc)
            return litestar_error_response(  # type: ignore[return-value]
                "Adapter registry subsystem unavailable", 503
            )

    return [
        training_pause,
        training_resume,
        training_stop,
        training_data_stats,
        training_data_seed,
        training_data_seed_stream,
        training_curriculum,
        training_curriculum_next,
        training_history,
        training_jobs,
        training_summary,
        training_quality,
        training_models,
        training_adapters_by_task_type,
        training_adapters_deployed,
    ]


def create_training_api_handlers() -> list[Any]:
    """Create all 22 Litestar handlers for the training pipeline API.

    Combines part-1 handlers (status, lifecycle, data, curriculum) from
    ``litestar_training_api`` with part-2 handlers (history, jobs, summary,
    quality, models, adapters) defined in this module, and part-3 handlers
    (dry-run, rules, sync-data, generate-synthetic, idle-stats) from
    ``litestar_training_api_part3``.

    Returns:
        Combined list of 22 Litestar route handler functions, or an empty
        list when Litestar is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_training_api import _create_training_api_handlers_part1
    from vetinari.web.litestar_training_api_part3 import _create_training_api_handlers_part3

    return (
        _create_training_api_handlers_part1()
        + _create_training_api_handlers_part2()
        + _create_training_api_handlers_part3()
    )
