"""Litestar lifespan hooks — startup and shutdown for background services.

This is the centralized startup/shutdown entry point for the Litestar ASGI
server.  It replaces the daemon-thread approach previously used in
``cli.py:cmd_start()``.

Pipeline step: Server Startup → **Lifespan** → Request Handling → Shutdown.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)


@asynccontextmanager
async def vetinari_lifespan(_app: Any) -> AsyncGenerator[None, None]:
    """Manage Vetinari background services during the server's lifetime.

    On startup:
    1. Wire subsystems (learning, drift, analytics, security, events, etc.)
    2. Start health monitor background loop
    3. Register EventBus shutdown callback

    On shutdown:
    1. Signal all background workers to stop
    2. Flush pending telemetry and SSE queues
    3. Unload any loaded GGUF models
    4. Drain the EventBus

    Args:
        _app: Litestar application instance (unused but required by protocol).

    Yields:
        Control to the server while background services run.
    """
    # ── Startup ──────────────────────────────────────────────────────
    logger.info("Lifespan: starting background services")

    try:
        from vetinari.cli_startup import _wire_subsystems

        _wire_subsystems()
    except Exception as exc:
        logger.warning("Lifespan: subsystem wiring failed (non-fatal): %s", exc)

    # Start health monitor polling loop
    _health_task = None
    try:
        from vetinari.system.health_monitor import start_health_monitor

        _health_task = start_health_monitor()
    except Exception as exc:
        logger.warning("Lifespan: health monitor not started: %s", exc)

    # Start background scheduler for periodic tasks (PDCA, drift, regression)
    _scheduler = None
    try:
        from apscheduler.schedulers.background import BackgroundScheduler

        _scheduler = BackgroundScheduler(daemon=True)

        try:
            from vetinari.kaizen.wiring import scheduled_pdca_check, scheduled_regression_check

            _scheduler.add_job(
                scheduled_pdca_check,
                "interval",
                hours=24,
                id="pdca_check",
                misfire_grace_time=3600,
            )
            _scheduler.add_job(
                scheduled_regression_check,
                "interval",
                hours=24,
                id="regression_check",
                misfire_grace_time=3600,
            )
        except ImportError:
            logger.warning("Kaizen wiring not available — PDCA/regression scheduling skipped")

        try:
            from vetinari.drift.wiring import schedule_drift_audit

            _scheduler.add_job(
                schedule_drift_audit,
                "interval",
                hours=6,
                id="drift_audit",
                misfire_grace_time=3600,
            )
        except ImportError:
            logger.warning("Drift wiring not available — drift audit scheduling skipped")

        # Periodically update temperature matrix from Thompson Sampling learnings
        try:
            from vetinari.models.model_profiler_data import update_learned_temperatures

            _scheduler.add_job(
                update_learned_temperatures,
                "interval",
                hours=6,
                id="temperature_learning",
                misfire_grace_time=3600,
            )
        except ImportError:
            logger.warning("Temperature learning not available — scheduling skipped")

        _scheduler.start()
        logger.info(
            "Background scheduler started with %d periodic jobs",
            len(_scheduler.get_jobs()),
        )
    except Exception as exc:
        logger.warning(
            "APScheduler setup failed — periodic tasks (PDCA, drift, regression) will not run: %s",
            exc,
        )

    # Start TrainingScheduler so idle-time learning runs automatically
    _training_scheduler = None
    try:
        from vetinari.web.litestar_training_api import _get_scheduler

        _training_scheduler = _get_scheduler()
        if _training_scheduler is not None:
            _training_scheduler.start()
            logger.info("TrainingScheduler started — idle-time learning enabled")
    except Exception as exc:
        logger.warning(
            "Lifespan: TrainingScheduler not started — idle-time training will not run automatically: %s",
            exc,
        )

    # Start LearningOrchestrator — coordinates prompt evolution, training, and research.
    # Bug #15b: gate on VariantConfig.enable_self_improvement so LOW variant never
    # starts the self-improvement loop (too expensive for low-resource deployments).
    _learning_orchestrator = None
    try:
        from vetinari.web.variant_system import get_variant_manager

        _vm_config = get_variant_manager().get_config()
        if _vm_config.enable_self_improvement:
            from vetinari.learning.orchestrator import get_learning_orchestrator

            _learning_orchestrator = get_learning_orchestrator()
            _learning_orchestrator.start()
            logger.info("LearningOrchestrator started — self-improvement loop active")
        else:
            logger.info(
                "LearningOrchestrator skipped — enable_self_improvement=False for variant '%s'",
                getattr(_vm_config, "level", "unknown"),
            )
    except Exception as exc:
        logger.warning(
            "Lifespan: LearningOrchestrator not started — self-improvement will not run automatically: %s",
            exc,
        )

    # Check for newer, better models in the background (weekly, non-blocking)
    _freshness_future = None
    try:
        from concurrent.futures import ThreadPoolExecutor

        from vetinari.models.model_scout import ModelFreshnessChecker

        checker = ModelFreshnessChecker()
        if checker.should_check():
            _freshness_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-freshness")

            def _bg_freshness_check() -> None:
                try:
                    upgrades = checker.check_for_upgrades()
                    if upgrades:
                        logger.info(
                            "Model freshness check: %d upgrade(s) available",
                            len(upgrades),
                        )
                        for u in upgrades[:3]:
                            logger.info(
                                "  Upgrade available: %s (score=%.2f, replaces %s)",
                                u.candidate_name,
                                u.overall_score,
                                u.current_model_id,
                            )
                except Exception as e:
                    logger.warning("Background model freshness check failed: %s", e)

            _freshness_future = _freshness_pool.submit(_bg_freshness_check)
    except Exception as exc:
        logger.warning("Lifespan: model freshness check skipped: %s", exc)

    yield

    # ── Shutdown ─────────────────────────────────────────────────────
    logger.info("Lifespan: shutting down background services")

    # Wait for freshness check to finish (max 5 seconds)
    if _freshness_future is not None:
        try:
            _freshness_future.result(timeout=5)
        except Exception:
            logger.warning(
                "Lifespan: freshness check did not complete in time — shutdown will continue without waiting"
            )
        import contextlib

        with contextlib.suppress(Exception):
            _freshness_pool.shutdown(wait=False)

    # Stop health monitor
    if _health_task is not None:
        try:
            from vetinari.system.health_monitor import stop_health_monitor

            stop_health_monitor()
        except Exception as exc:
            logger.warning("Lifespan: health monitor stop failed: %s", exc)

    # Stop model warm-up workers before unloading model caches. A warm-up that
    # completes during shutdown also self-unloads when the stop flag is set.
    try:
        from vetinari.models.model_pool import stop_all_model_warmups

        stop_all_model_warmups()
    except Exception as exc:  # noqa: VET023 - handler intentionally converts optional failure to safe response
        logger.debug("Lifespan: model warm-up stop skipped: %s", exc)

    # Unload GGUF models to free VRAM
    try:
        from vetinari.adapters.llama_cpp_adapter import LlamaCppProviderAdapter

        LlamaCppProviderAdapter.unload_all()
    except Exception as exc:  # noqa: VET023  # optional: model unload at shutdown is best-effort, module may not be loaded
        logger.debug("Lifespan: model unload skipped: %s", exc)

    # Drain watch queue
    try:
        from vetinari.watch import WatchMode

        wm = WatchMode.get_instance()
        if wm is not None:
            wm.drain_queue()
    except Exception as exc:  # noqa: VET023  # optional: watch queue drain at shutdown is best-effort, module may not be loaded
        logger.debug("Lifespan: watch queue drain skipped: %s", exc)

    # Shut down EventBus
    try:
        from vetinari.events import get_event_bus

        bus = get_event_bus()
        if bus is not None:
            bus.shutdown()
    except Exception as exc:  # noqa: VET023  # optional: event bus shutdown at lifespan end is best-effort
        logger.debug("Lifespan: event bus shutdown skipped: %s", exc)

    # Drain pending batches before exit so in-flight requests are not lost
    try:
        from vetinari.adapters.batch_processor import reset_batch_processor

        reset_batch_processor()
    except Exception as exc:  # noqa: VET023  # optional: batch processor drain at shutdown is best-effort, module may not be loaded
        logger.debug("Lifespan: batch processor drain skipped: %s", exc)

    # Stop telemetry persistence after the final batch drain so shutdown does
    # not leave the periodic flush thread running behind the ASGI app.
    try:
        from vetinari.analytics.telemetry_persistence import reset_telemetry_persistence

        reset_telemetry_persistence()
    except Exception as exc:  # noqa: VET023 - handler intentionally converts optional failure to safe response
        logger.debug("Lifespan: telemetry persistence stop skipped: %s", exc)

    # Reset the Worker MCP bridge so external server subprocesses are stopped
    # through the mounted app shutdown path, not only process-exit hooks.
    try:
        from vetinari.mcp.worker_bridge import reset_worker_mcp_bridge

        reset_worker_mcp_bridge()
        logger.info("Lifespan: Worker MCP bridge shut down")
    except Exception as exc:
        logger.warning("Lifespan: Worker MCP bridge shutdown failed -- subprocesses may linger: %s", exc)

    # Release cascade router resources (thread pool, provider connections)

    try:
        from vetinari.cascade_router import reset_cascade_router

        reset_cascade_router()
    except Exception as exc:  # noqa: VET023  # optional: cascade router reset at shutdown is best-effort, module may not be loaded
        logger.debug("Lifespan: cascade router reset skipped: %s", exc)

    # Release LLM Guard scanner resources (model weights held in memory)

    try:
        from vetinari.safety.llm_guard_scanner import reset_llm_guard_scanner

        reset_llm_guard_scanner()
    except Exception as exc:  # noqa: VET023  # optional: scanner reset at shutdown is best-effort, module may not be loaded
        logger.debug("Lifespan: LLM Guard scanner reset skipped: %s", exc)

    # Clear in-memory violation history so stale entries do not survive a hot-reload

    try:
        from vetinari.constraints.registry import get_constraint_registry

        get_constraint_registry().clear_violations()
    except Exception as exc:  # noqa: VET023  # optional: violation clear at shutdown is best-effort, registry may not have been initialised
        logger.debug("Lifespan: constraint violation clear skipped: %s", exc)

    # Stop LearningOrchestrator (self-improvement loop)
    if _learning_orchestrator is not None:
        try:
            _learning_orchestrator.stop()
            logger.info("LearningOrchestrator stopped")
        except Exception as exc:
            logger.warning("Lifespan: LearningOrchestrator stop did not complete cleanly: %s", exc)

    # Stop TrainingScheduler (idle-time learning)
    if _training_scheduler is not None:
        try:
            _training_scheduler.stop()
            logger.info("TrainingScheduler stopped")
        except Exception as exc:
            logger.warning("Lifespan: TrainingScheduler stop did not complete cleanly: %s", exc)

    # Stop background scheduler
    if _scheduler is not None:
        try:
            _scheduler.shutdown(wait=False)
            logger.info("Background scheduler stopped")
        except Exception as exc:
            logger.warning("Background scheduler shutdown did not complete cleanly: %s", exc)

    try:
        from vetinari.shutdown import shutdown as run_registered_shutdown

        run_registered_shutdown()
    except Exception as exc:
        logger.warning("Lifespan: registered shutdown callbacks did not complete cleanly: %s", exc)

    logger.info("Lifespan: shutdown complete")
