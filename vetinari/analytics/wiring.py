"""Analytics pipeline wiring — connects inference and execution paths to analytics consumers.

Every LLM inference, task completion, quality score, and pipeline stage transition flows
through this module to the appropriate analytics tracker. This is the single wiring point
that closes the feedback loops between execution and observability.

Call sites:
    - ``record_inference_cost`` / ``record_inference_failure``: called from inference adapters
      after each LLM call completes or fails.
    - ``record_task_metrics``: called from the executor after each task finishes.
    - ``record_periodic_metrics``: called from a background scheduler every ~5 minutes.
    - ``record_quality_score``: called from the quality scorer after every score.
    - ``record_pipeline_event``: called from the orchestrator at each stage transition.
    - ``predict_cost`` / ``record_actual_cost``: called from the planner before and after tasks.
    - ``record_hardware_metrics``: called from the hardware monitor or VRAM manager on each telemetry cycle.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vetinari.analytics.anomaly import AnomalyDetector
    from vetinari.analytics.cost import CostTracker
    from vetinari.analytics.cost_predictor import CostPredictor
    from vetinari.analytics.failure_registry import FailureRegistry
    from vetinari.analytics.forecasting import Forecaster
    from vetinari.analytics.isolation_forest import HardwareAnomalyDetector, PrimaryAnomalyDetector
    from vetinari.analytics.quality_drift import QualityDriftDetector
    from vetinari.analytics.sla import SLATracker
    from vetinari.analytics.value_stream import ValueStreamAnalyzer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level mutable state
# ---------------------------------------------------------------------------
# Lazy singleton cache for each analytics consumer.
# Written once on first use inside _lazy_*() under _lock.
# Read on every call thereafter without the lock (after assignment is visible).
# Protected by double-checked locking (PEP 703 / CPython GIL semantics).

_cost_tracker = None  # CostTracker singleton — who writes: record_inference_cost; who reads: cost API
_sla_tracker = None  # SLATracker singleton — who writes: record_inference_cost/failure; who reads: SLA API
_anomaly_detector = None  # AnomalyDetector singleton — who writes: record_task_metrics; who reads: anomaly API
_forecaster = None  # Forecaster singleton — who writes: record_periodic_metrics; who reads: forecast API
_drift_ensemble = None  # QualityDriftDetector — who writes: record_quality_score; who reads: drift API
_value_stream = None  # ValueStreamAnalyzer — who writes: record_pipeline_event; who reads: VSM API
_cost_predictor = None  # CostPredictor — who writes: record_actual_cost; who reads: predict_cost
_primary_detector = None  # PrimaryAnomalyDetector — who writes: record_task_metrics; who reads: anomaly API
_hardware_detector = None  # HardwareAnomalyDetector — who writes: record_hardware_metrics; who reads: anomaly API
_failure_registry = None  # FailureRegistry — who writes: record_failure; who reads: Inspector prevention gate

_lock = threading.Lock()  # guards all lazy initialization above

# ---------------------------------------------------------------------------
# Lazy singleton accessors (module-private)
# ---------------------------------------------------------------------------


def _lazy_cost_tracker() -> CostTracker | None:
    global _cost_tracker
    if _cost_tracker is None:
        with _lock:
            if _cost_tracker is None:
                from vetinari.analytics.cost import get_cost_tracker

                _cost_tracker = get_cost_tracker()
    return _cost_tracker


def _lazy_sla_tracker() -> SLATracker | None:
    global _sla_tracker
    if _sla_tracker is None:
        with _lock:
            if _sla_tracker is None:
                from vetinari.analytics.sla import get_sla_tracker, register_default_slos

                _sla_tracker = get_sla_tracker()
                register_default_slos()
    return _sla_tracker


def _lazy_anomaly_detector() -> AnomalyDetector | None:
    global _anomaly_detector
    if _anomaly_detector is None:
        with _lock:
            if _anomaly_detector is None:
                from vetinari.analytics.anomaly import get_anomaly_detector

                _anomaly_detector = get_anomaly_detector()
    return _anomaly_detector


def _lazy_forecaster() -> Forecaster | None:
    global _forecaster
    if _forecaster is None:
        with _lock:
            if _forecaster is None:
                from vetinari.analytics.forecasting import get_forecaster

                _forecaster = get_forecaster()
    return _forecaster


def _lazy_drift_ensemble() -> QualityDriftDetector | None:
    global _drift_ensemble
    if _drift_ensemble is None:
        with _lock:
            if _drift_ensemble is None:
                from vetinari.analytics.quality_drift import get_drift_ensemble

                _drift_ensemble = get_drift_ensemble()
    return _drift_ensemble


def _lazy_value_stream() -> ValueStreamAnalyzer | None:
    # Delegate to the canonical singleton so record_pipeline_event() writes
    # to the same instance that callers retrieve via get_value_stream_analyzer().
    from vetinari.analytics.value_stream import get_value_stream_analyzer

    return get_value_stream_analyzer()


def _lazy_cost_predictor() -> CostPredictor | None:
    global _cost_predictor
    if _cost_predictor is None:
        with _lock:
            if _cost_predictor is None:
                from vetinari.analytics.cost_predictor import CostPredictor

                _cost_predictor = CostPredictor()
    return _cost_predictor


def _lazy_primary_detector() -> PrimaryAnomalyDetector | None:
    global _primary_detector
    if _primary_detector is None:
        with _lock:
            if _primary_detector is None:
                from vetinari.analytics.isolation_forest import PrimaryAnomalyDetector

                _primary_detector = PrimaryAnomalyDetector()
    return _primary_detector


def _lazy_hardware_detector() -> HardwareAnomalyDetector | None:
    global _hardware_detector
    if _hardware_detector is None:
        with _lock:
            if _hardware_detector is None:
                from vetinari.analytics.isolation_forest import HardwareAnomalyDetector

                _hardware_detector = HardwareAnomalyDetector()
    return _hardware_detector


def _lazy_failure_registry() -> FailureRegistry | None:
    global _failure_registry
    if _failure_registry is None:
        with _lock:
            if _failure_registry is None:
                from vetinari.analytics.failure_registry import get_failure_registry

                _failure_registry = get_failure_registry()
    return _failure_registry


# ---------------------------------------------------------------------------
# Public wiring functions
# ---------------------------------------------------------------------------


def record_inference_cost(
    agent_type: str,
    task_id: str,
    provider: str,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
) -> None:
    """Record cost and SLA data after a successful LLM inference call.

    Feeds CostTracker, SLATracker. Call this from every inference adapter
    immediately after a successful completion response.

    Args:
        agent_type: The agent type that made the inference (e.g. "FOREMAN").
        task_id: The task ID associated with this inference.
        provider: Provider name (e.g. "local", "openai").
        model_id: Model identifier used for inference.
        input_tokens: Number of input tokens consumed.
        output_tokens: Number of output tokens generated.
        latency_ms: Wall-clock latency of the inference call in milliseconds.
    """
    try:
        from vetinari.analytics.cost import CostEntry

        tracker = _lazy_cost_tracker()
        if tracker is not None:
            tracker.record(
                CostEntry(
                    agent=agent_type,
                    task_id=task_id,
                    provider=provider,
                    model=model_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                )
            )
    except Exception:
        logger.warning(
            "Cost tracking failed for model %s (task %s) — cost report will be incomplete",
            model_id,
            task_id,
        )

    try:
        sla = _lazy_sla_tracker()
        if sla is not None:
            sla.record_latency(f"{provider}:{model_id}", latency_ms, success=True)
            sla.record_request(success=True)
    except Exception:
        logger.warning(
            "SLA tracking failed for %s:%s — SLA compliance reports may be stale",
            provider,
            model_id,
        )


def record_inference_failure(
    agent_type: str,
    provider: str,
    model_id: str,
    latency_ms: float,
) -> None:
    """Record SLA data after a failed LLM inference call.

    Feeds SLATracker failure counters. Call from inference adapters whenever
    a request raises an exception or returns an error response.

    Args:
        agent_type: The agent type that made the inference.
        provider: Provider name.
        model_id: Model identifier.
        latency_ms: Wall-clock latency before failure in milliseconds.
    """
    try:
        sla = _lazy_sla_tracker()
        if sla is not None:
            sla.record_latency(f"{provider}:{model_id}", latency_ms, success=False)
            sla.record_request(success=False)
    except Exception:
        logger.warning(
            "SLA failure tracking failed for %s:%s (agent %s) — error rate metrics may be understated",
            provider,
            model_id,
            agent_type,
        )

    # Log to persistent failure registry for kaizen analysis
    try:
        registry = _lazy_failure_registry()
        if registry is not None:
            registry.log_failure(
                category="model_timeout",
                severity="error",
                description=(
                    f"Inference failure for model {model_id} via {provider}"
                    f" (agent {agent_type}, latency {latency_ms:.0f}ms)"
                ),
                root_cause=f"Model {model_id} failed to respond via provider {provider}",
                affected_components=[agent_type, provider, model_id],
            )
    except Exception:
        logger.warning(
            "Failure registry logging failed for %s:%s — failure not persisted",
            provider,
            model_id,
        )


def record_task_metrics(
    task_id: str,
    agent_type: str,
    latency_ms: float,
    quality_score: float,
    token_count: int,
    success: bool,
) -> None:
    """Feed post-task metrics to the anomaly detector.

    Feeds AnomalyDetector with latency, quality, and token signals so it
    can alert when values deviate statistically from recent baselines.
    Call from the executor after each task completes or fails.

    Args:
        task_id: The completed task ID.
        agent_type: Agent type that executed the task.
        latency_ms: Total task execution latency in milliseconds.
        quality_score: Quality score from the inspector (0.0-1.0).
        token_count: Total tokens consumed by the task.
        success: Whether the task succeeded.
    """
    try:
        detector = _lazy_anomaly_detector()
        if detector is not None:
            detector.detect("task_latency", latency_ms)
            detector.detect("task_quality", quality_score)
            detector.detect("task_tokens", float(token_count))
    except Exception:
        logger.warning(
            "Anomaly detection failed for task %s (agent %s) — anomaly alerts may be delayed",
            task_id,
            agent_type,
        )

    try:
        primary = _lazy_primary_detector()
        if primary is not None:
            error_rate = 0.0 if success else 1.0
            primary.observe(
                latency=latency_ms,
                error_rate=error_rate,
                token_usage=float(token_count),
                quality_score=quality_score,
            )
    except Exception:
        logger.warning(
            "Isolation Forest primary detection failed for task %s (agent %s) "
            "— multivariate anomaly alerts may be delayed",
            task_id,
            agent_type,
        )


def record_periodic_metrics(
    request_rate: float,
    avg_latency_ms: float,
    queue_depth: int,
) -> None:
    """Feed periodic system metrics to the capacity forecaster.

    Feeds Forecaster with throughput and latency signals used for
    short-horizon capacity planning. Call every ~5 minutes from a
    background scheduler.

    Args:
        request_rate: Requests per second over the last interval.
        avg_latency_ms: Average inference latency over the last interval in milliseconds.
        queue_depth: Current request queue depth.
    """
    try:
        forecaster = _lazy_forecaster()
        if forecaster is not None:
            forecaster.ingest("request_rate", request_rate)
            forecaster.ingest("avg_latency_ms", avg_latency_ms)
            forecaster.ingest("queue_depth", float(queue_depth))
    except Exception:
        logger.warning(
            "Forecaster ingestion failed (rate=%.2f, latency=%.1fms, queue=%d) "
            "— capacity predictions will be stale until next successful ingest",
            request_rate,
            avg_latency_ms,
            queue_depth,
        )


def record_quality_score(quality_score: float) -> None:
    """Feed a quality score to the drift detector.

    Feeds QualityDriftDetector so it can identify when output quality
    shifts statistically. Call after every QualityScorer.score() call.

    Args:
        quality_score: The quality score in range 0.0-1.0.
    """
    validated_score = _validate_quality_score(quality_score)
    try:
        drift = _lazy_drift_ensemble()
        if drift is not None:
            drift.observe(validated_score)
    except Exception:
        logger.warning(
            "Quality drift observation failed (score=%.4f) — drift detection may miss degradation",
            validated_score,
        )


def record_quality_scores_batch(quality_scores: list[float]) -> None:
    """Feed multiple quality scores to the drift detector in a single call.

    More efficient than calling ``record_quality_score`` in a loop because
    ``observe_many`` acquires the internal lock once for all scores.
    Call this after batch quality evaluation (e.g., benchmark suite completion
    or inspector scoring of a work-unit list).

    Args:
        quality_scores: Sequence of quality scores in range [0.0, 1.0].
    """
    if not quality_scores:
        return
    validated_scores = [_validate_quality_score(score, index=index) for index, score in enumerate(quality_scores)]
    try:
        drift = _lazy_drift_ensemble()
        if drift is not None:
            drift.observe_many(validated_scores)
    except Exception:
        logger.warning(
            "Batch quality drift observation failed (%d scores) — drift detection may miss degradation",
            len(validated_scores),
        )


def _validate_quality_score(value: Any, *, index: int | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        location = f" at index {index}" if index is not None else ""
        raise ValueError(f"quality score{location} must be a numeric value")
    score = float(value)
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        location = f" at index {index}" if index is not None else ""
        raise ValueError(f"quality score{location} must be finite and in range [0.0, 1.0]")
    return score


def get_quality_drift_stats() -> dict[str, float]:
    """Return summary statistics over the retained quality-score observation window.

    Delegates to :meth:`~vetinari.analytics.quality_drift.QualityDriftDetector.get_raw_stats`
    to expose count, mean, median, stddev, p95, p99 over the bounded window.

    Returns:
        Dict with ``count``, ``mean``, ``median``, ``stddev``, ``p95``,
        ``p99`` keys.  Returns an empty dict when the drift ensemble is unavailable.
    """
    try:
        drift = _lazy_drift_ensemble()
        if drift is not None:
            return dict[str, float](drift.get_raw_stats())
    except Exception:
        logger.warning("Could not retrieve quality drift stats — drift ensemble may not be initialised")
    return {}


def record_pipeline_event(
    execution_id: str,
    task_id: str,
    agent_type: str,
    timing_event: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record a pipeline stage transition for value stream analysis.

    Feeds ValueStreamAnalyzer so it can compute lead time, queue time,
    and waste ratios across pipeline stages. Call from the orchestrator
    at every stage transition.

    Args:
        execution_id: The pipeline execution ID.
        task_id: The task being processed.
        agent_type: Agent type at this stage.
        timing_event: Stage event name — one of: task_queued, task_dispatched,
            task_completed, task_rejected, task_rework, task_skipped.
        metadata: Optional additional context for this event.
    """
    try:
        vs = _lazy_value_stream()
        if vs is not None:
            vs.record_event(
                execution_id=execution_id,
                task_id=task_id,
                agent_type=agent_type,
                timing_event=timing_event,
                metadata=metadata or {},  # noqa: VET112 — Optional per func param
            )
    except Exception:
        logger.warning(
            "Value stream recording failed for execution %s task %s event %s "
            "— lead time metrics for this execution will be incomplete",
            execution_id,
            task_id,
            timing_event,
        )


def predict_cost(
    task_type: str,
    complexity: float,
    scope_size: int,
    model_id: str,
) -> dict[str, Any]:
    """Predict cost before task execution using the calibrated cost model.

    Feeds CostPredictor.predict() and returns a structured estimate. Call
    from the planner before dispatching a task to decide whether to proceed
    or seek a cheaper model.

    Args:
        task_type: Type of task (e.g. "coding", "analysis", "planning").
        complexity: Estimated task complexity in range 0.0-1.0.
        scope_size: Number of files or items in scope.
        model_id: Model that will execute the task.

    Returns:
        Dict with keys ``tokens`` (int), ``latency_seconds`` (float),
        ``cost_usd`` (float), and ``confidence`` (float 0-1).
        Returns all-zero values when the predictor is unavailable.
    """
    try:
        predictor = _lazy_cost_predictor()
        if predictor is not None:
            estimate = predictor.predict(task_type, complexity, scope_size, model_id)
            return {
                "tokens": estimate.tokens,
                "latency_seconds": estimate.latency_seconds,
                "cost_usd": estimate.cost_usd,
                "confidence": estimate.confidence,
            }
    except Exception:
        logger.warning(
            "Cost prediction failed for task_type=%s model=%s — no pre-execution estimate available",
            task_type,
            model_id,
        )
    return {"tokens": 0, "latency_seconds": 0.0, "cost_usd": 0.0, "confidence": 0.0}


def record_actual_cost(
    task_type: str,
    complexity: float,
    scope_size: int,
    model_id: str,
    actual_tokens: int,
    actual_latency: float,
    actual_cost: float,
) -> None:
    """Record observed cost after task execution to calibrate the cost predictor.

    Feeds CostPredictor.record_actual() so the predictor's heuristic model
    can converge toward real outcomes over time. Call from the executor
    after every task that has a prior predict_cost() call.

    Args:
        task_type: Type of task executed.
        complexity: Complexity estimate used before execution.
        scope_size: Scope size used before execution.
        model_id: Model that executed the task.
        actual_tokens: Actual tokens consumed.
        actual_latency: Actual latency in seconds.
        actual_cost: Actual cost in USD.
    """
    try:
        predictor = _lazy_cost_predictor()
        if predictor is not None:
            predictor.record_actual(
                task_type=task_type,
                complexity=complexity,
                scope_size=scope_size,
                model=model_id,
                actual_tokens=actual_tokens,
                actual_latency=actual_latency,
                actual_cost=actual_cost,
            )
    except Exception:
        logger.warning(
            "Cost actual recording failed for task_type=%s model=%s — predictor calibration will be delayed",
            task_type,
            model_id,
        )


def record_hardware_metrics(
    gpu_util_pct: float,
    vram_util_pct: float,
    model_load_unload_freq: float,
    cache_hit_rate: float,
) -> None:
    """Feed hardware telemetry to the Isolation Forest hardware anomaly detector.

    Observes GPU and VRAM utilization, model swap frequency, and cache hit
    rate as a 4-dimensional vector.  The detector flags unusual metric
    combinations that single-metric thresholds miss (e.g. low GPU util with
    abnormally high model swap rate).

    Call from the hardware monitor or VRAM manager on each telemetry cycle
    (typically every 30-60 seconds).

    Args:
        gpu_util_pct: GPU utilization as a percentage (0-100).
        vram_util_pct: VRAM utilization as a percentage (0-100).
        model_load_unload_freq: Model load/unload operations per minute.
        cache_hit_rate: KV-cache or model cache hit rate as a fraction (0-1).
    """
    try:
        hw = _lazy_hardware_detector()
        if hw is not None:
            hw.observe(
                gpu_util_pct=gpu_util_pct,
                vram_util_pct=vram_util_pct,
                model_load_unload_freq=model_load_unload_freq,
                cache_hit_rate=cache_hit_rate,
            )
    except Exception:
        logger.warning(
            "Isolation Forest hardware detection failed "
            "(gpu=%.1f%%, vram=%.1f%%, swaps=%.2f/min, cache_hit=%.3f) "
            "— hardware anomaly alerts may be delayed",
            gpu_util_pct,
            vram_util_pct,
            model_load_unload_freq,
            cache_hit_rate,
        )


def record_failure(
    category: str,
    severity: str,
    description: str,
    root_cause: str = "",
    affected_components: list[str] | None = None,
) -> None:
    """Log a pipeline failure to the persistent failure registry.

    Delegates to FailureRegistry.log_failure() which appends a JSONL line
    and checks whether repeated failures should trigger prevention rule
    generation.  Non-fatal — if the registry is unavailable, the failure
    is logged at WARNING but the pipeline continues.

    Args:
        category: Failure category (e.g., ``"inspector_rejection"``).
        severity: One of ``"warning"``, ``"error"``, ``"critical"``.
        description: Human-readable description of the failure.
        root_cause: Root cause analysis text.
        affected_components: List of component names impacted.
    """
    try:
        registry = _lazy_failure_registry()
        if registry is not None:
            registry.log_failure(
                category=category,
                severity=severity,
                description=description,
                root_cause=root_cause,
                affected_components=affected_components,
            )
    except Exception:
        logger.warning(
            "Failure registry logging failed for category=%s — failure not persisted",
            category,
        )


def record_unknown_family_task_result(
    model_id: str,
    architecture: str,
    quality_score: float,
) -> None:
    """Record a task result for a model whose family was not in model_families.yaml.

    After enough tasks accumulate (threshold defined in model_profiler_data),
    a new family entry is created automatically. Non-fatal — failures are
    logged and swallowed.

    Args:
        model_id: The model identifier.
        architecture: The model's architecture string.
        quality_score: Quality score from this task execution (0.0-1.0).
    """
    try:
        from vetinari.models.model_profiler_data import record_unknown_family_task

        record_unknown_family_task(model_id, architecture, quality_score)
    except Exception:
        logger.warning(
            "Unknown-family task recording failed for model %s — family auto-creation may be delayed",
            model_id,
        )


def reset_all() -> None:
    """Reset all lazy singletons to None.

    Clears every cached analytics consumer so the next call to any wiring
    function will re-initialize from scratch. Intended for test isolation
    only — do not call in production code.
    """
    global \
        _cost_tracker, \
        _sla_tracker, \
        _anomaly_detector, \
        _forecaster, \
        _drift_ensemble, \
        _value_stream, \
        _cost_predictor, \
        _primary_detector, \
        _hardware_detector, \
        _failure_registry
    _cost_tracker = None
    _sla_tracker = None
    _anomaly_detector = None
    _forecaster = None
    _drift_ensemble = None
    _value_stream = None
    _cost_predictor = None
    _primary_detector = None
    _hardware_detector = None
    _failure_registry = None


# Alias for external consumers
reset_wiring = reset_all
