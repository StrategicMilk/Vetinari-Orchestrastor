"""CLI startup helpers — system wiring and initialization for Vetinari.

Responsible for logging setup, config loading, orchestrator construction,
and wiring all optional subsystems together at startup.  Every wiring step
is non-fatal: a missing subsystem logs a warning and startup continues.

This is an internal support module consumed by ``vetinari.cli``.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import sys
from pathlib import Path

from vetinari.types import StatusEnum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skill modules that expose Tool subclasses for auto-registration.
# Written here: _wire_skills_to_registry reads it.
# Read by: _wire_skills_to_registry in this module.
# ---------------------------------------------------------------------------
_SKILL_MODULES: list[str] = [
    "vetinari.tools.file_tool",
    "vetinari.tools.git_tool",
    "vetinari.tools.web_search_tool",
    "vetinari.tools.brave_search_tool",
    "vetinari.tools.tool_registry_integration",
]


def _instantiate_tool_class(tool_class: type[object], module_name: str) -> object | None:
    """Create a Tool instance, using scoped factories for sandboxed tools.

    File and git tools require an explicit project/repository root so agents
    cannot inherit an unsafe process cwd.  Their existing factories provide
    that scope; other Tool subclasses keep the normal no-argument constructor
    path used by the auto-registration scanner.

    Args:
        tool_class: Candidate Tool subclass discovered by module scanning.
        module_name: Module where the class was discovered.

    Returns:
        A Tool-compatible object, or None when a scoped factory reports the
        tool is unavailable in the current runtime.
    """
    if module_name == "vetinari.tools.file_tool" and tool_class.__name__ == "FileOperationsTool":
        from vetinari.tools.tool_registry_integration import _make_file_tool

        return _make_file_tool()
    if module_name == "vetinari.tools.git_tool" and tool_class.__name__ == "GitOperationsTool":
        from vetinari.tools.tool_registry_integration import _make_git_tool

        return _make_git_tool()
    return tool_class()


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def _is_debug_mode() -> bool:
    """Check if VETINARI_DEBUG=1 environment variable is set.

    Returns:
        True if VETINARI_DEBUG is set to 1, true, or yes.
    """
    return os.environ.get("VETINARI_DEBUG", "").strip() in ("1", "true", "yes")


def _setup_logging(verbose: bool = False) -> None:
    """Configure root logging for the Vetinari process.

    When ``verbose`` or ``VETINARI_DEBUG=1`` is set, switches to DEBUG level
    and adds millisecond timestamps and line numbers to log output.

    Args:
        verbose: If True, enable DEBUG logging regardless of environment.
    """
    debug_mode = _is_debug_mode()
    level = logging.DEBUG if (verbose or debug_mode) else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    if verbose or debug_mode:
        # Enhanced format with timing and module location — active when verbose or VETINARI_DEBUG=1
        fmt = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%H:%M:%S",
    )
    if debug_mode:
        logging.getLogger("vetinari").setLevel(logging.DEBUG)
        logger.debug("VETINARI_DEBUG mode active — all debug logs promoted")


# ---------------------------------------------------------------------------
# Config and orchestrator construction
# ---------------------------------------------------------------------------


def _load_config(config_path: str) -> dict:
    """Load a YAML manifest config file, falling back to defaults if missing.

    Tries the path as given, then relative to the package root. Returns
    a minimal default config when the file cannot be found.

    Args:
        config_path: Relative or absolute path to the YAML manifest.

    Returns:
        Parsed YAML content as a dict, or a minimal default dict.
    """
    import yaml

    p = Path(config_path)
    if not p.exists():
        # Try relative to package directory
        pkg_root = Path(__file__).resolve().parents[1]
        p = pkg_root / config_path
    if not p.exists():
        logger.warning("Config file not found: %s, using defaults", config_path)
        return {"project_name": "vetinari", "tasks": []}
    with Path(p).open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        logger.warning(
            "Config file %s has unexpected format (expected dict, got %s) — using defaults",
            config_path,
            type(data).__name__,
        )
        return {"project_name": "vetinari", "tasks": []}
    return data


def _build_orchestrator(config_path: str, mode: str = "execution"):
    """Construct an Orchestrator for the given config path and mode.

    Args:
        config_path: Path to the manifest YAML file.
        mode: Execution mode (planning, execution, or sandbox).

    Returns:
        A configured Orchestrator instance.
    """
    from vetinari.orchestrator import Orchestrator

    return Orchestrator(config_path, execution_mode=mode)


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------


def _print_banner(mode: str) -> None:
    """Print the Vetinari startup banner.

    When VETINARI_DEBUG=1 is set, also prints feature status, model
    availability, and system information.

    Args:
        mode: The execution mode to display in the banner.
    """
    print("=" * 60)
    print(" VETINARI AI Orchestration System")
    print(f" Mode: {mode.upper()}")
    if _is_debug_mode():
        import time

        print(" Debug: ENABLED (VETINARI_DEBUG=1)")
        print(f" Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Python: {sys.version.split()[0]}")
        # Feature status
        try:
            from vetinari.constants import (
                OPERATOR_MODELS_CACHE_DIR,
                get_user_dir,
            )

            print(f" Models dir: {OPERATOR_MODELS_CACHE_DIR}")
            print(f" User dir: {get_user_dir()}")
        except ImportError:  # noqa: VET022 - best-effort optional path must not fail the primary flow
            pass  # Constants may not be available in minimal installs
        try:
            import llama_cpp  # noqa: F401 - import intentionally probes or re-exports API surface

            print(" llama-cpp-python: available")
        except ImportError:
            print(" llama-cpp-python: NOT installed")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Drift check at startup
# ---------------------------------------------------------------------------


def _check_drift_at_startup() -> None:
    """Run contract drift check at startup (non-fatal).

    Queries the drift monitor for any contract or schema changes since the
    last baseline.  Logs warnings but never prevents startup.
    """
    try:
        from vetinari.drift.monitor import get_drift_monitor

        monitor = get_drift_monitor()
        report = monitor.run_full_audit()
        if not report.is_clean:
            for issue in report.issues:
                logger.warning("[Drift] %s", issue)
            print("[Vetinari] WARNING: Contract drift detected. Run 'vetinari drift-check' for details.")
    except Exception as e:
        logger.warning("Drift check skipped: %s", e)


# ---------------------------------------------------------------------------
# Subsystem wiring
# ---------------------------------------------------------------------------


def _wire_subsystems() -> None:
    """Connect all Vetinari subsystems together at startup.

    Wires:
    1. Graceful shutdown handlers (SIGTERM/SIGINT + atexit)
    2. Learning pipeline -> web dashboard API blueprints
    3. Drift monitor -> orchestration pre-check hook
    4. Analytics -> web dashboard API blueprints
    5. Security scanner -> verification pipeline
    6. Skill Tool subclasses -> ToolRegistry (auto-registration)
    7. Durable execution recovery
    8. EventBus domain subscribers
    9. TelemetryPersistence background flush loop

    All wiring steps are non-fatal: failures are logged as warnings so that a
    missing optional subsystem never prevents Vetinari from starting.
    """
    try:
        from vetinari.shutdown import register_shutdown_handlers

        register_shutdown_handlers()
    except Exception as exc:
        logger.warning("Wiring: shutdown handlers failed: %s", exc)
    _wire_learning_to_dashboard()
    _wire_drift_to_orchestration()
    _wire_analytics_to_dashboard()
    _wire_security_to_verification()
    _wire_skills_to_registry()
    _wire_durable_recovery()
    _wire_event_subscribers()
    _wire_sse_event_cleanup()
    _wire_telemetry_persistence()
    _wire_autonomy_and_notifications()
    logger.info("Startup wiring complete")


def _wire_autonomy_and_notifications() -> None:
    """Initialize the autonomy governor, approval queue, and notification channels."""
    try:
        from vetinari.autonomy.wiring import wire_autonomy_and_notifications

        wire_autonomy_and_notifications()
    except Exception as exc:
        logger.warning("Wiring: autonomy/notifications failed: %s", exc)


def _wire_learning_to_dashboard() -> None:
    """Ensure the learning API handlers are importable and ready."""
    try:
        from vetinari.web.litestar_learning_api import (
            create_learning_api_handlers,  # noqa: F401 - import intentionally probes or re-exports API surface
        )

        logger.info("Wiring: learning -> dashboard OK")
    except Exception as exc:
        logger.warning("Wiring: learning -> dashboard failed: %s", exc)


def _wire_drift_to_orchestration() -> None:
    """Connect drift monitor into orchestration cycle.

    Verifies the drift monitor singleton is importable and initialised so that
    orchestration code can call ``get_drift_monitor()`` safely at runtime.
    """
    try:
        from vetinari.drift.monitor import get_drift_monitor

        monitor = get_drift_monitor()
        monitor.bootstrap()
        logger.info("Wiring: drift -> orchestration OK (bootstrap complete)")
    except Exception as exc:
        logger.warning("Wiring: drift -> orchestration failed: %s", exc)


def _wire_analytics_to_dashboard() -> None:
    """Ensure the analytics API handlers are importable and ready."""
    try:
        from vetinari.web.litestar_analytics import (
            create_analytics_handlers,  # noqa: F401 - import intentionally probes or re-exports API surface
        )

        logger.info("Wiring: analytics -> dashboard OK")
    except Exception as exc:
        logger.warning("Wiring: analytics -> dashboard failed: %s", exc)


def _wire_security_to_verification() -> None:
    """Ensure SecurityVerifier is present in the verification pipeline.

    The VerificationPipeline already creates a SecurityVerifier for BASIC+
    levels.  This step confirms the wiring is intact; if missing (e.g. NONE
    level) it re-adds the built-in SecurityVerifier.
    """
    try:
        from vetinari.security import get_secret_scanner
        from vetinari.validation import (
            SecurityVerifier as _SecurityVerifier,
        )
        from vetinari.validation import (
            get_verifier_pipeline,
        )

        pipeline = get_verifier_pipeline()
        scanner = get_secret_scanner()

        has_security = any(v.name == "security" for v in pipeline.verifiers)
        if not has_security:
            pipeline.add_verifier(_SecurityVerifier())
            logger.info("Wiring: added missing SecurityVerifier to pipeline")

        # Confirm the scanner is operational
        scanner.scan("test")

        logger.info("Wiring: security -> verification OK")
    except Exception as exc:
        logger.warning("Wiring: security -> verification failed: %s", exc)


def _wire_skills_to_registry() -> None:
    """Auto-register all skill Tool subclasses into the global ToolRegistry.

    Scans each module listed in ``_SKILL_MODULES`` for concrete Tool subclasses
    and registers them if not already present (idempotent).
    """
    try:
        from vetinari.tool_interface import Tool, get_tool_registry

        registry = get_tool_registry()
        registered: list[str] = []

        for mod_name in _SKILL_MODULES:
            try:
                mod = importlib.import_module(mod_name)
                for _attr_name, attr_value in inspect.getmembers(mod, inspect.isclass):
                    if issubclass(attr_value, Tool) and attr_value is not Tool and not inspect.isabstract(attr_value):
                        try:
                            instance = _instantiate_tool_class(attr_value, mod_name)
                            if instance is None:
                                continue
                            if not isinstance(instance, Tool):
                                logger.warning(
                                    "Could not register %s from %s: factory returned %s instead of Tool",
                                    _attr_name,
                                    mod_name,
                                    type(instance).__name__,
                                )
                                continue
                            if registry.get(instance.metadata.name) is None:
                                registry.register(instance)
                                registered.append(instance.metadata.name)
                        except Exception as inst_err:
                            logger.warning(
                                "Could not instantiate %s from %s: %s",
                                _attr_name,
                                mod_name,
                                inst_err,
                            )
            except Exception as mod_err:
                logger.warning("Could not import skill module %s: %s", mod_name, mod_err)

        logger.info("Wiring: skills -> registry OK (%d skills registered)", len(registered))
    except Exception as exc:
        logger.warning("Wiring: skills -> registry failed: %s", exc)


def _wire_durable_recovery() -> None:
    """Resume incomplete executions from durable checkpoints on startup.

    Queries the DurableExecutionEngine's SQLite store for any execution that
    was interrupted (neither completed nor failed) and resumes it.  Runs in
    the startup thread — each recovered execution logs its own outcome.
    """
    try:
        from vetinari.orchestration.two_layer import get_two_layer_orchestrator

        orch = get_two_layer_orchestrator()
        recovered = orch.recover_incomplete_on_startup()
        if recovered:
            logger.info(
                "Wiring: durable recovery OK — resumed %d execution(s)",
                len(recovered),
            )
        else:
            logger.info("Wiring: durable recovery OK — no incomplete executions")
    except Exception as exc:
        logger.warning("Wiring: durable recovery failed: %s", exc)


def _wire_sse_event_cleanup() -> None:
    """Register SSE event log cleanup as a shutdown callback.

    Ensures stale SSE audit log rows (older than 24 hours) are purged
    when the process shuts down, preventing unbounded table growth.
    """
    try:
        from vetinari.shutdown import register_callback
        from vetinari.web.sse_events import cleanup_stale_sse_events

        register_callback("SSE event log cleanup", cleanup_stale_sse_events)
        logger.info("Wiring: SSE event log cleanup -> shutdown OK")
    except Exception as exc:
        logger.warning("Wiring: SSE event log cleanup failed: %s", exc)


def _wire_event_subscribers() -> None:
    """Register domain-specific EventBus subscribers at startup.

    Connects event publishers to their intended consumers so that the EventBus
    is not write-only. Each subscriber is wired independently — one failure
    does not prevent others from registering.

    Wires:
    - RetrainingRecommended -> AgentTrainer.record_retraining_signal()
    - AnomalyDetected -> AlertEngine.evaluate_anomaly()
    - QualityGateResult -> SPCMonitor.update('quality_score', score)
    - TaskCompleted -> FeedbackLoop + training telemetry
    - KaizenImprovementProposed/Confirmed/Reverted -> improvement tracking
    """
    try:
        from vetinari.events import (
            AnomalyDetected,
            KaizenImprovementConfirmed,
            KaizenImprovementProposed,
            KaizenImprovementReverted,
            QualityGateResult,
            RetrainingRecommended,
            TaskCompleted,
            TaskTimingRecord,
            get_event_bus,
        )

        bus = get_event_bus()
        wired = 0

        # 1. RetrainingRecommended -> AgentTrainer
        try:
            from vetinari.training.agent_trainer import get_agent_trainer

            trainer = get_agent_trainer()
            bus.subscribe(RetrainingRecommended, trainer.record_retraining_signal)
            wired += 1
            logger.debug("EventBus: RetrainingRecommended -> AgentTrainer")
        except Exception as exc:
            logger.warning("EventBus: failed to wire RetrainingRecommended: %s", exc)

        # 1b. RetrainingRecommended -> TrainingManager (drift -> retrain check)
        try:
            from vetinari.learning.training_manager import get_training_manager

            tmgr = get_training_manager()

            def _on_retraining_recommended_tm(event: RetrainingRecommended) -> None:
                recommendation = tmgr.should_retrain(
                    model_id=event.metric or "default",
                    task_type="general",
                )
                if recommendation.recommended:
                    logger.info(
                        "[EventBus] TrainingManager recommends retraining: %s (degradation=%.2f)",
                        recommendation.reason,
                        recommendation.degradation,
                    )

            bus.subscribe(RetrainingRecommended, _on_retraining_recommended_tm)
            wired += 1
            logger.debug("EventBus: RetrainingRecommended -> TrainingManager.should_retrain()")
        except Exception as exc:
            logger.warning("EventBus: failed to wire drift -> TrainingManager: %s", exc)

        # 2. AnomalyDetected -> AlertEngine
        try:
            from vetinari.dashboard.alerts import get_alert_engine

            alert_engine = get_alert_engine()
            bus.subscribe(AnomalyDetected, alert_engine.evaluate_anomaly)
            wired += 1
            logger.debug("EventBus: AnomalyDetected -> AlertEngine")
        except Exception as exc:
            logger.warning("EventBus: failed to wire AnomalyDetected: %s", exc)

        # 3. QualityGateResult -> SPC monitor
        try:
            from vetinari.workflow import get_spc_monitor

            spc = get_spc_monitor()

            def _on_quality_gate(event: QualityGateResult) -> None:
                spc.update("quality_score", event.score)

            bus.subscribe(QualityGateResult, _on_quality_gate)
            wired += 1
            logger.debug("EventBus: QualityGateResult -> SPCMonitor")
        except Exception as exc:
            logger.warning("EventBus: failed to wire QualityGateResult: %s", exc)

        # 4. TaskCompleted -> FeedbackLoop telemetry + training data readiness check
        try:
            from vetinari.learning.feedback_loop import get_feedback_loop

            feedback_loop = get_feedback_loop()

            def _on_task_completed(event: TaskCompleted) -> None:
                # Feed task outcome into the learning feedback loop
                feedback_loop.update(
                    task_id=event.task_id,
                    agent_type=event.agent_type,
                    quality_score=1.0 if event.success else 0.0,
                    model_name="default",
                )
                # Check if training data watermark is reached
                try:
                    from vetinari.learning.training_data import check_training_data_ready

                    readiness = check_training_data_ready()
                    if readiness.get(StatusEnum.READY.value, False):
                        logger.info(
                            "[EventBus] Training data watermark reached — %d records available",
                            readiness.get("total_records", 0),
                        )
                except Exception:
                    logger.warning("Training data readiness check failed", exc_info=True)

            bus.subscribe(TaskCompleted, _on_task_completed)
            wired += 1
            logger.debug("EventBus: TaskCompleted -> FeedbackLoop + training telemetry")
        except Exception as exc:
            logger.warning("EventBus: failed to wire TaskCompleted: %s", exc)

        # 5. Kaizen events -> improvement tracking via structured logging
        try:
            from vetinari.structured_logging import log_event as sl_log_event

            def _on_kaizen_proposed(event: KaizenImprovementProposed) -> None:
                sl_log_event(
                    "info",
                    "vetinari.kaizen",
                    "improvement_proposed",
                    improvement_id=event.improvement_id,
                    metric=event.metric,
                    hypothesis=event.hypothesis,
                    applied_by=event.applied_by,
                )

            def _on_kaizen_confirmed(event: KaizenImprovementConfirmed) -> None:
                sl_log_event(
                    "info",
                    "vetinari.kaizen",
                    "improvement_confirmed",
                    improvement_id=event.improvement_id,
                    metric=event.metric,
                    baseline_value=event.baseline_value,
                    actual_value=event.actual_value,
                    applied_by=event.applied_by,
                )

            def _on_kaizen_reverted(event: KaizenImprovementReverted) -> None:
                sl_log_event(
                    "warning",
                    "vetinari.kaizen",
                    "improvement_reverted",
                    improvement_id=event.improvement_id,
                    metric=event.metric,
                    reason=event.reason,
                )

            bus.subscribe(KaizenImprovementProposed, _on_kaizen_proposed)
            bus.subscribe(KaizenImprovementConfirmed, _on_kaizen_confirmed)
            bus.subscribe(KaizenImprovementReverted, _on_kaizen_reverted)
            wired += 3
            logger.debug("EventBus: Kaizen events -> improvement tracking")
        except Exception as exc:
            logger.warning("EventBus: failed to wire Kaizen events: %s", exc)

        # 6. TaskTimingRecord -> ValueStreamAnalyzer (value stream mapping)
        try:
            from vetinari.analytics.value_stream import get_value_stream_analyzer

            vsm = get_value_stream_analyzer()

            def _on_timing_record(event: TaskTimingRecord) -> None:
                vsm.record_event(
                    execution_id=event.execution_id,
                    task_id=event.task_id,
                    agent_type=event.agent_type,
                    timing_event=event.timing_event,
                    metadata=event.metadata,
                )

            bus.subscribe(TaskTimingRecord, _on_timing_record)
            wired += 1
            logger.debug("EventBus: TaskTimingRecord -> ValueStreamAnalyzer")
        except Exception as exc:
            logger.warning("EventBus: failed to wire TaskTimingRecord: %s", exc)

        # 7. QUALITY_DRIFT -> calibration frequency increase + SSE notification
        try:
            from vetinari.events import QualityDriftDetected

            def _on_quality_drift(event: QualityDriftDetected) -> None:
                logger.warning(
                    "Quality drift detected: task_type=%s, detectors=%s, observations=%d",
                    event.task_type or "all",
                    ", ".join(event.triggered_detectors),
                    event.observation_count,
                )
                # Increase calibration frequency for affected task types
                try:
                    from vetinari.learning.quality_scorer import get_quality_scorer

                    scorer = get_quality_scorer()
                    # Halve the calibration interval to catch drift faster
                    old_interval = scorer._calibration_interval
                    scorer._calibration_interval = max(2, old_interval // 2)
                    logger.info(
                        "Quality drift response: calibration interval %d -> %d for faster LLM checks",
                        old_interval,
                        scorer._calibration_interval,
                    )
                except Exception:
                    logger.warning("Could not adjust calibration frequency after drift detection")

                # Emit SSE event for dashboard notification
                try:
                    from vetinari.web.shared import _push_sse_event

                    _push_sse_event(
                        "_system",
                        "quality_drift",
                        {
                            "task_type": event.task_type or "all",
                            "detectors": event.triggered_detectors,
                            "observation_count": event.observation_count,
                        },
                    )
                except Exception:
                    logger.warning("Could not push SSE event for quality drift notification")

            bus.subscribe(QualityDriftDetected, _on_quality_drift)
            wired += 1
            logger.debug("EventBus: QualityDriftDetected -> calibration + SSE")
        except Exception as exc:
            logger.warning("EventBus: failed to wire QualityDriftDetected: %s", exc)

        # 8. TaskCompleted -> AnomalyDetector (post-task latency anomaly detection)
        try:
            from vetinari.analytics.anomaly import get_anomaly_detector

            anomaly_detector = get_anomaly_detector()

            def _on_task_completed_anomaly(event: TaskCompleted) -> None:
                if event.duration_ms > 0:
                    result = anomaly_detector.detect("task.duration_ms", event.duration_ms)
                    if result.is_anomaly:
                        logger.warning(
                            "[AnomalyDetector] Task %s duration anomaly: %.1fms (%s)",
                            event.task_id,
                            event.duration_ms,
                            result.reason,
                        )

            bus.subscribe(TaskCompleted, _on_task_completed_anomaly)
            wired += 1
            logger.debug("EventBus: TaskCompleted -> AnomalyDetector (latency)")
        except Exception as exc:
            logger.warning("EventBus: failed to wire AnomalyDetector: %s", exc)

        logger.info("Wiring: EventBus subscribers OK — %d subscriber(s) registered", wired)
    except Exception as exc:
        logger.warning("Wiring: EventBus subscribers failed: %s", exc)


def _wire_telemetry_persistence() -> None:
    """Start the TelemetryPersistence background flush loop at startup.

    TelemetryPersistence batches telemetry records in memory and flushes
    them to SQLite periodically. Without calling start(), the flush loop
    never begins and records accumulate without being persisted.

    Also restores the most recent snapshot into the in-memory collector so
    that counters survive process restarts.
    """
    try:
        from vetinari.telemetry import get_telemetry_collector

        get_telemetry_collector().restore_from_snapshot()
        logger.info("Wiring: TelemetryCollector snapshot restore complete")
    except Exception as exc:
        logger.warning("Wiring: TelemetryCollector snapshot restore failed (non-fatal): %s", exc)

    try:
        from vetinari.analytics.telemetry_persistence import get_telemetry_persistence

        get_telemetry_persistence().start()
        logger.info("Wiring: TelemetryPersistence -> started OK")
    except Exception as exc:
        logger.warning("Wiring: TelemetryPersistence start failed: %s", exc)
