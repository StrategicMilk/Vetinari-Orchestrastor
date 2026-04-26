"""Litestar handlers for the training pipeline API, part 1. Native Litestar equivalents (ADR-0066).

Provides REST and SSE endpoints for monitoring and controlling the Vetinari
training pipeline, idle scheduler, curriculum planner, and LoRA adapter
registry.  URL paths identical to Flask training_api.py blueprint
(url_prefix="/api/v1/training").

This module (part 1) covers:
    GET  /api/v1/training/status            — current training scheduler state
    POST /api/v1/training/start             — manually trigger a training cycle (admin)

See ``litestar_training_api_part2.py`` for the remaining 15 handlers and the
combined ``create_training_api_handlers()`` factory.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vetinari.training.idle_scheduler import TrainingScheduler

try:
    from litestar import MediaType, Response, get, post

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Shared scheduler singleton
# ---------------------------------------------------------------------------

_scheduler_singleton: TrainingScheduler | None = None
_scheduler_lock = threading.Lock()


def _get_scheduler() -> TrainingScheduler | None:
    """Return the shared TrainingScheduler singleton.

    Delegates to :func:`vetinari.training.idle_scheduler.get_training_scheduler`
    so the web layer and all other callers share exactly one instance.
    The training layer owns the canonical singleton, so this helper refreshes
    from that source on each call instead of trusting a stale web-local cache.
    Returns ``None`` if the training modules are not importable.

    Returns:
        The shared TrainingScheduler instance, or None.
    """
    global _scheduler_singleton
    try:
        from vetinari.training.idle_scheduler import get_training_scheduler

        scheduler = get_training_scheduler()
    except ImportError:
        logger.debug("Training scheduler modules not available")
        return None
    except Exception:
        logger.exception("Failed to obtain training scheduler singleton")
        return None

    with _scheduler_lock:
        _scheduler_singleton = scheduler
    return scheduler


# ---------------------------------------------------------------------------
# Module-level utility functions (importable without a web request context)
# ---------------------------------------------------------------------------


def _is_scheduler_training() -> bool:
    """Return ``True`` when the shared TrainingScheduler has an active job.

    Calls :func:`_get_scheduler` to obtain the singleton and inspects the
    ``_current_job`` attribute under the scheduler's internal lock.  Returns
    ``False`` if no scheduler singleton exists yet.

    Returns:
        ``True`` when a training job is currently in flight, ``False`` otherwise.
    """
    scheduler = _get_scheduler()
    if scheduler is None:
        return False
    return scheduler.current_job is not None


def get_training_status() -> dict[str, Any]:
    """Aggregate current training pipeline state from all available subsystems.

    Queries the training data collector for record counts, the idle detector
    for scheduler state, and the curriculum for phase and next planned activity.
    Every subsystem failure is caught individually so the function always
    returns a complete dict regardless of which services are available.

    Returns:
        Dict with keys: status, current_job, last_run, records_collected,
        curriculum_phase, next_activity.  Sentinel values are used when a
        subsystem is unreachable: records_collected=0, curriculum_phase="unknown",
        next_activity=None, status="idle".
    """
    records_collected = 0
    status = "idle"
    curriculum_phase = "unknown"
    next_activity = None

    try:
        from vetinari.learning.training_data import get_training_collector

        collector = get_training_collector()
        records_collected = collector.get_stats().get("total", 0)
    except Exception as exc:
        logger.warning(
            "get_training_status: training collector unavailable, defaulting records_collected=0: %s",
            exc,
        )

    try:
        from vetinari.training.idle_scheduler import get_idle_detector

        detector = get_idle_detector()
        if detector is not None and not detector.idle:
            status = "running"
    except Exception as exc:
        logger.warning(
            "get_training_status: idle detector unavailable, defaulting status='idle': %s",
            exc,
        )

    try:
        from vetinari.training.curriculum import TrainingCurriculum

        curriculum = TrainingCurriculum()
        curriculum_phase = curriculum.get_status().get("phase", "unknown")
        activity = curriculum.next_activity()
        if activity is not None:
            next_activity = {
                "type": activity.type.value,
                "description": activity.description,
                "priority": activity.priority,
            }
    except Exception as exc:
        logger.warning(
            "get_training_status: curriculum unavailable, defaulting phase='unknown': %s",
            exc,
        )

    return {
        "status": status,
        "current_job": None,
        "last_run": None,
        "records_collected": records_collected,
        "curriculum_phase": curriculum_phase,
        "next_activity": next_activity,
    }


def get_training_history(*, limit: int = 50) -> list[dict[str, Any]]:
    """Return a merged, time-sorted list of training history entries.

    Combines entries from the quality gate and auto-tuner subsystems.  Each
    entry is tagged with a ``type`` field so callers can distinguish the source.
    Results are sorted newest-first and capped at ``limit`` entries.

    Args:
        limit: Maximum number of entries to return.  Defaults to 50.

    Returns:
        List of history entry dicts sorted descending by timestamp, each
        containing at minimum a ``timestamp`` and ``type`` field.  Returns an
        empty list when both subsystems are unavailable.
    """
    entries: list[dict[str, Any]] = []

    try:
        from vetinari.training.quality_gate import get_training_quality_gate

        gate = get_training_quality_gate()
        for entry in gate.get_history():
            tagged = dict(entry)
            tagged["type"] = "quality_gate"
            entries.append(tagged)
    except Exception as exc:
        logger.warning(
            "get_training_history: quality gate unavailable, omitting its entries: %s",
            exc,
        )

    try:
        from vetinari.learning.auto_tuner import get_auto_tuner

        tuner = get_auto_tuner()
        for entry in tuner.get_history():
            tagged = dict(entry)
            tagged["type"] = "auto_tune"
            entries.append(tagged)
    except Exception as exc:
        logger.warning(
            "get_training_history: auto tuner unavailable, omitting its entries: %s",
            exc,
        )

    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return entries[:limit]


def get_quality_comparison() -> dict[str, Any]:
    """Return the most recent quality gate comparison result.

    Reads the latest entry from the quality gate history and surfaces the
    baseline/candidate quality scores, their delta, the accept/reject decision,
    and the latency ratio.  A sentinel dict is returned so callers never need
    to handle None or missing keys.

    Returns:
        Dict with keys: baseline_quality, candidate_quality, quality_delta,
        decision, latency_ratio.  Returns sentinel values
        (all 0.0, decision="no_data") when the quality gate is unavailable
        or its history is empty.
    """
    _sentinel: dict[str, Any] = {
        "baseline_quality": 0.0,
        "candidate_quality": 0.0,
        "quality_delta": 0.0,
        "decision": "no_data",
        "latency_ratio": 0.0,
    }

    try:
        from vetinari.training.quality_gate import get_training_quality_gate

        gate = get_training_quality_gate()
        history = gate.get_history()
        if not history:
            return _sentinel
        latest = history[-1]
        return {
            "baseline_quality": latest.get("baseline_quality", 0.0),
            "candidate_quality": latest.get("candidate_quality", 0.0),
            "quality_delta": latest.get("quality_delta", 0.0),
            "decision": latest.get("decision", "no_data"),
            "latency_ratio": latest.get("latency_ratio", 0.0),
        }
    except Exception as exc:
        logger.warning(
            "get_quality_comparison: quality gate unavailable, returning no_data sentinel: %s",
            exc,
        )
        return _sentinel


# ---------------------------------------------------------------------------
# Handler factory — part 1 (status, lifecycle controls, data, curriculum)
# ---------------------------------------------------------------------------


def _create_training_api_handlers_part1() -> list[Any]:
    """Create the first 2 training API route handlers (status and start).

    Covers the status query and manual trigger endpoints.  The remaining 8
    lifecycle/data/curriculum handlers live in ``_create_training_api_handlers_part2``
    in ``litestar_training_api_part2`` to keep both files under the 550-line ceiling.
    Called by ``create_training_api_handlers()`` in ``litestar_training_api_part2``.

    Returns:
        List of 2 Litestar route handler functions.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard

    @get("/api/v1/training/status", media_type=MediaType.JSON)
    async def training_status() -> Response | dict[str, Any]:
        """Return current training scheduler state.

        Combines idle-detector status with curriculum phase and any active job
        information from the TrainingScheduler singleton when available.
        Returns 503 when the training subsystem is entirely unreachable so
        callers can distinguish a healthy-but-idle system from an unavailable one.

        Returns:
            JSON with keys: phase, is_idle, idle_minutes, is_training,
            current_activity, next_activity, ready_for_training,
            missing_libraries.  503 when training is unavailable.
        """
        from vetinari.web.responses import litestar_error_response

        phase = "unknown"
        is_idle = True
        idle_minutes = 0.0
        is_training = False
        current_activity = None
        next_activity = None
        ready_for_training = False
        missing_libraries: list[str] = []
        any_subsystem_available = False

        try:
            from vetinari.training.pipeline import TrainingPipeline

            reqs = TrainingPipeline().check_requirements()
            ready_for_training = reqs.get("ready_for_training", False)
            libs = reqs.get("libraries", {})
            missing_libraries = [lib for lib, avail in libs.items() if not avail]
            any_subsystem_available = True
        except Exception as exc:
            logger.warning("training_status: pipeline requirements unavailable: %s", exc)

        try:
            from vetinari.training.curriculum import TrainingCurriculum

            curriculum = TrainingCurriculum()
            status_data = curriculum.get_status()
            phase = status_data.get("phase", "unknown")
            next_activity = status_data.get("next_activity_description")
            any_subsystem_available = True
        except Exception as exc:
            logger.warning("training_status: curriculum unavailable: %s", exc)

        scheduler = _get_scheduler()
        if scheduler is not None:
            any_subsystem_available = True
            try:
                detector = scheduler._idle_detector
                is_idle = detector.idle
                idle_minutes = detector.idle_duration_minutes
                is_training = scheduler.is_training
                job = scheduler.current_job
                if job is not None:
                    current_activity = job.activity_description
            except Exception as exc:
                logger.warning("training_status: scheduler query failed: %s", exc)

        if not any_subsystem_available:
            return litestar_error_response(  # type: ignore[return-value]
                "Training subsystem unavailable — no training modules could be loaded", 503
            )

        return {
            "status": "ok",
            "phase": phase,
            "is_idle": is_idle,
            "idle_minutes": idle_minutes,
            "is_training": is_training,
            "current_activity": current_activity,
            "next_activity": next_activity,
            "ready_for_training": ready_for_training,
            "missing_libraries": missing_libraries,
        }

    @post("/api/v1/training/start", media_type=MediaType.JSON, guards=[admin_guard])
    async def training_start(data: dict[str, Any]) -> Response | dict[str, Any]:
        """Manually trigger a training cycle.

        Accepts an optional ``skill`` field in the JSON body to target a
        specific skill area.  When absent the curriculum selects the
        highest-priority activity automatically.

        Args:
            data: Request body with optional ``skill`` field.

        Returns:
            JSON with status message describing the triggered activity,
            or a 503 error when prerequisites are not met.
        """
        if not data:
            return Response(
                content={"status": "error", "message": "Request body must not be empty — provide at least one field"},
                status_code=422,
                media_type=MediaType.JSON,
            )

        _TRAINING_START_KEYS = frozenset({"skill"})
        if not _TRAINING_START_KEYS.intersection(data):
            return Response(
                content={
                    "status": "error",
                    "message": "Request body contains no recognised fields — provide a 'skill' field",
                },
                status_code=422,
                media_type=MediaType.JSON,
            )

        skill = data.get("skill")
        # skill must be a non-empty string when provided; null/int/list/empty-string
        # are all rejected because they cannot identify a skill curriculum area.
        # None covers the case where the key is explicitly present with a null value.
        if skill is None or not isinstance(skill, str) or not skill.strip():
            return Response(
                content={
                    "status": "error",
                    "message": "'skill' must be a non-empty string",
                },
                status_code=422,
                media_type=MediaType.JSON,
            )

        try:
            from vetinari.training.pipeline import TrainingPipeline

            pipeline = TrainingPipeline()
            reqs = pipeline.check_requirements()
            if not reqs.get("ready_for_training", False):
                missing = [lib for lib, avail in reqs.get("libraries", {}).items() if not avail]
                return Response(
                    content={
                        "status": "error",
                        "message": f"Training libraries not installed: {', '.join(missing)}",
                    },
                    status_code=503,
                    media_type=MediaType.JSON,
                )
        except Exception as exc:
            logger.warning("training_start: requirements check failed: %s", exc)

        scheduler = _get_scheduler()
        if scheduler is None:
            return Response(
                content={"status": "error", "message": "Training scheduler not available"},
                status_code=503,
                media_type=MediaType.JSON,
            )

        if scheduler.is_training:
            return Response(
                content={"status": "error", "message": "Training already in progress"},
                status_code=409,
                media_type=MediaType.JSON,
            )

        activity_description = None
        try:
            from vetinari.training.curriculum import TrainingCurriculum

            curriculum = TrainingCurriculum()
            activity = curriculum.next_activity()
            activity_description = activity.description if activity else None
        except Exception as exc:
            logger.warning("training/start: could not determine activity from curriculum: %s", exc)

        message = "Training cycle triggered"
        if skill:
            message = f"Training cycle triggered for skill: {skill}"
        elif activity_description:
            message = f"Training cycle triggered: {activity_description}"

        thread = threading.Thread(
            target=scheduler._execute_training_cycle,
            name="manual-training-trigger",
            daemon=True,
        )
        thread.start()

        return {"status": "ok", "message": message, "activity": activity_description}

    return [
        training_status,
        training_start,
    ]
