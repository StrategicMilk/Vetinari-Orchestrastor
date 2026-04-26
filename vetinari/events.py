"""Event bus for inter-agent communication.

Provides a publish/subscribe event system that enables decoupled communication
between agents in the Vetinari orchestration pipeline. Events are published
synchronously or asynchronously, and a bounded history is maintained for
debugging and audit purposes.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import queue
import threading
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vetinari.constants import (
    EVENTBUS_ASYNC_QUEUE_SIZE,
    EVENTBUS_HISTORY_MAX_LENGTH,
    QUEUE_TIMEOUT,
    THREAD_JOIN_TIMEOUT,
    THREAD_JOIN_TIMEOUT_SHORT,
)

logger = logging.getLogger(__name__)

_HISTORY_MAX_LENGTH = EVENTBUS_HISTORY_MAX_LENGTH  # Maximum number of events retained in the history ring buffer


# ---------------------------------------------------------------------------
# Event dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Event:
    """Base class for all events in the Vetinari event bus.

    Args:
        event_type: Discriminator string identifying the event kind.
        timestamp: Wall-clock time (``time.time()``) when the event was created.
    """

    event_type: str
    timestamp: float

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(event_type={self.event_type!r}, timestamp={self.timestamp!r})"


@dataclass(frozen=True, slots=True)
class TaskStarted(Event):
    """Published when a task begins execution.

    Args:
        task_id: Unique identifier of the task.
        agent_type: The ``AgentType.value`` string of the executing agent.
        timestamp: Wall-clock time when the event was created.
    """

    task_id: str = ""
    agent_type: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", "TaskStarted")

    def __repr__(self) -> str:
        return f"TaskStarted(task_id={self.task_id!r}, agent_type={self.agent_type!r})"


@dataclass(frozen=True, slots=True)
class TaskCompleted(Event):
    """Published when a task finishes execution.

    Args:
        task_id: Unique identifier of the task.
        agent_type: The ``AgentType.value`` string of the executing agent.
        success: Whether the task succeeded.
        duration_ms: Elapsed wall-clock time in milliseconds.
        timestamp: Wall-clock time when the event was created.
    """

    task_id: str = ""
    agent_type: str = ""
    success: bool = False
    duration_ms: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", "TaskCompleted")

    def __repr__(self) -> str:
        return (
            f"TaskCompleted(task_id={self.task_id!r}, agent_type={self.agent_type!r}, "
            f"success={self.success!r}, duration_ms={self.duration_ms!r})"
        )


@dataclass(frozen=True, slots=True)
class QualityGateResult(Event):
    """Published after a quality review completes.

    Args:
        task_id: Unique identifier of the reviewed task.
        passed: Whether the quality gate passed.
        score: Numeric quality score in the range ``[0.0, 1.0]``.
        issues: List of human-readable issue descriptions.
        timestamp: Wall-clock time when the event was created.
    """

    task_id: str = ""
    passed: bool = False
    score: float = 0.0
    issues: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", "QualityGateResult")

    def __repr__(self) -> str:
        return (
            f"QualityGateResult(task_id={self.task_id!r}, passed={self.passed!r}, "
            f"score={self.score!r}, issues={len(self.issues)})"
        )


@dataclass(frozen=True, slots=True)
class ResourceRequest(Event):
    """Published when an agent requests an external resource.

    Args:
        agent_type: The ``AgentType.value`` string of the requesting agent.
        resource_type: Category of resource being requested (e.g. ``"model"``, ``"tool"``).
        details: Arbitrary metadata describing the request.
        timestamp: Wall-clock time when the event was created.
    """

    agent_type: str = ""
    resource_type: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", "ResourceRequest")

    def __repr__(self) -> str:
        return f"ResourceRequest(agent_type={self.agent_type!r}, resource_type={self.resource_type!r})"


@dataclass(frozen=True, slots=True)
class HumanApprovalNeeded(Event):
    """Published when a task requires human approval to proceed.

    Args:
        task_id: Unique identifier of the task requiring approval.
        reason: Human-readable explanation of why approval is needed.
        context: Arbitrary metadata providing additional context for the approver.
        timestamp: Wall-clock time when the event was created.
    """

    task_id: str = ""
    reason: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", "HumanApprovalNeeded")

    def __repr__(self) -> str:
        return f"HumanApprovalNeeded(task_id={self.task_id!r}, reason={self.reason!r})"


@dataclass(frozen=True, slots=True)
class AnomalyDetected(Event):
    """Published when an ensemble anomaly detector confirms an anomaly.

    Args:
        agent_type: The agent type where the anomaly was detected.
        anomaly_type: The type of anomaly (e.g., "ensemble").
        triggered_detectors: List of detector names that triggered.
        score: Anomaly severity score.
        timestamp: Wall-clock time when the event was created.
    """

    agent_type: str = ""
    anomaly_type: str = ""
    triggered_detectors: list[str] = field(default_factory=list)
    score: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", "AnomalyDetected")

    def __repr__(self) -> str:
        return (
            f"AnomalyDetected(agent_type={self.agent_type!r}, anomaly_type={self.anomaly_type!r}, score={self.score!r})"
        )


@dataclass(frozen=True, slots=True)
class RetrainingRecommended(Event):
    """Published when forecasting predicts quality dropping below SLA threshold.

    Args:
        metric: The quality metric being forecast.
        predicted_quality: The predicted quality value at breach point.
        days_until_breach: Estimated days until SLA breach.
        confidence_interval: The confidence interval width at breach point.
        forecast_method_used: Which forecast method produced this prediction.
        timestamp: Wall-clock time when the event was created.
    """

    metric: str = ""
    predicted_quality: float = 0.0
    days_until_breach: int = 0
    confidence_interval: float = 0.0
    forecast_method_used: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", "RetrainingRecommended")

    def __repr__(self) -> str:
        return (
            f"RetrainingRecommended(metric={self.metric!r}, "
            f"predicted_quality={self.predicted_quality!r}, "
            f"days_until_breach={self.days_until_breach!r})"
        )


class TimingEvent(Enum):
    """Timing event types for value stream mapping."""

    TASK_QUEUED = "task_queued"
    TASK_DISPATCHED = "task_dispatched"
    TASK_COMPLETED = "task_completed"
    TASK_REJECTED = "task_rejected"
    TASK_REWORK = "task_rework"
    TASK_SKIPPED = "task_skipped"


@dataclass(frozen=True, slots=True)
class TaskTimingRecord(Event):
    """Timing record for value stream analysis.

    Captures when each stage transition happens for a task, enabling
    computation of queue time, processing time, and waste.

    Args:
        task_id: Unique identifier of the task.
        execution_id: ID of the overall execution this task belongs to.
        agent_type: The agent type processing this task.
        timing_event: Which stage transition occurred.
        metadata: Additional context (queue_depth_at_time, model_used, etc.).
        timestamp: Wall-clock time when the event was created.
    """

    task_id: str = ""
    execution_id: str = ""
    agent_type: str = ""
    timing_event: str = ""  # TimingEvent.value
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", "TaskTimingRecord")

    def __repr__(self) -> str:
        return (
            f"TaskTimingRecord(task_id={self.task_id!r}, execution_id={self.execution_id!r}, "
            f"timing_event={self.timing_event!r})"
        )


@dataclass(frozen=True, slots=True)
class KaizenImprovementProposed(Event):
    """Published when a new kaizen improvement is proposed.

    Args:
        improvement_id: Unique identifier of the proposed improvement.
        hypothesis: What the improvement is expected to achieve.
        metric: Which metric is being improved.
        applied_by: Which subsystem proposed this improvement.
        timestamp: Wall-clock time when the event was created.
    """

    improvement_id: str = ""
    hypothesis: str = ""
    metric: str = ""
    applied_by: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", "KaizenImprovementProposed")

    def __repr__(self) -> str:
        return (
            f"KaizenImprovementProposed(improvement_id={self.improvement_id!r}, "
            f"metric={self.metric!r}, applied_by={self.applied_by!r})"
        )


@dataclass(frozen=True, slots=True)
class KaizenImprovementConfirmed(Event):
    """Published when a kaizen improvement is confirmed (met its target).

    Args:
        improvement_id: Unique identifier of the confirmed improvement.
        metric: Which metric was improved.
        baseline_value: Metric value before improvement.
        actual_value: Measured metric value after observation.
        applied_by: Which subsystem applied this improvement.
        timestamp: Wall-clock time when the event was created.
    """

    improvement_id: str = ""
    metric: str = ""
    baseline_value: float = 0.0
    actual_value: float = 0.0
    applied_by: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", "KaizenImprovementConfirmed")

    def __repr__(self) -> str:
        return (
            f"KaizenImprovementConfirmed(improvement_id={self.improvement_id!r}, "
            f"metric={self.metric!r}, baseline_value={self.baseline_value!r}, "
            f"actual_value={self.actual_value!r})"
        )


@dataclass(frozen=True, slots=True)
class KaizenImprovementActive(Event):
    """Published when a kaizen improvement moves from PROPOSED to ACTIVE.

    Args:
        improvement_id: Unique identifier of the improvement now under trial.
        metric: Which metric the improvement targets.
        applied_by: Which subsystem activated this improvement.
        timestamp: Wall-clock time when the event was created.
    """

    improvement_id: str = ""
    metric: str = ""
    applied_by: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", "KaizenImprovementActive")

    def __repr__(self) -> str:
        return (
            f"KaizenImprovementActive(improvement_id={self.improvement_id!r}, "
            f"metric={self.metric!r}, applied_by={self.applied_by!r})"
        )


@dataclass(frozen=True, slots=True)
class KaizenImprovementReverted(Event):
    """Published when a kaizen improvement is reverted due to regression.

    Args:
        improvement_id: Unique identifier of the reverted improvement.
        metric: Which metric regressed.
        reason: Why the improvement was reverted.
        timestamp: Wall-clock time when the event was created.
    """

    improvement_id: str = ""
    metric: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", "KaizenImprovementReverted")

    def __repr__(self) -> str:
        return (
            f"KaizenImprovementReverted(improvement_id={self.improvement_id!r}, "
            f"metric={self.metric!r}, reason={self.reason!r})"
        )


@dataclass(frozen=True, slots=True)
class KaizenLintFinding(Event):
    """Published when knowledge lint detects an issue (stale, contradiction, orphan, drift).

    Args:
        finding_id: Unique identifier for the lint finding.
        category: Lint category (contradiction, stale, orphaned, vocabulary_drift).
        description: Human-readable description of the finding.
        severity: Finding severity (info, warning, error).
    """

    finding_id: str = ""
    category: str = ""
    description: str = ""
    severity: str = "warning"

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", "KaizenLintFinding")

    def __repr__(self) -> str:
        return f"KaizenLintFinding(finding_id={self.finding_id!r}, category={self.category!r})"


@dataclass(frozen=True, slots=True)
class QualityDriftDetected(Event):
    """Published when ensemble drift detectors confirm a quality shift.

    Fired by :class:`~vetinari.analytics.quality_drift.QualityDriftDetector`
    when 2+ of 3 detectors (CUSUM, Page-Hinkley, ADWIN) agree on drift.

    Args:
        task_type: The task type experiencing drift (empty string if unknown).
        triggered_detectors: Names of detectors that triggered (e.g. ``["cusum", "adwin"]``).
        observation_count: Total observations processed at time of detection.
        timestamp: Wall-clock time when the event was created.
    """

    task_type: str = ""
    triggered_detectors: list[str] = field(default_factory=list)
    observation_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", "QUALITY_DRIFT")

    def __repr__(self) -> str:
        return f"QualityDriftDetected(task_type={self.task_type!r}, triggered_detectors={self.triggered_detectors!r})"


@dataclass(frozen=True, slots=True)
class TelemetryAlertEvent(Event):
    """Published when a telemetry threshold breach is detected.

    Fired by :meth:`~vetinari.analytics.telemetry_persistence.TelemetryPersistence._emit_alert_event`
    when error rate or p95 latency exceeds configured thresholds.

    Args:
        alert_type: Short identifier for the breach (e.g. ``"high_error_rate"``,
            ``"high_p95_latency"``).
        message: Human-readable description of what breached and by how much.
        metadata: Numeric context for the alert (rates, thresholds, counts).
    """

    alert_type: str = ""
    message: str = ""
    metadata: dict = field(default_factory=dict)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", "TELEMETRY_ALERT")

    def __repr__(self) -> str:
        return f"TelemetryAlertEvent(alert_type={self.alert_type!r}, message={self.message!r})"


# ---------------------------------------------------------------------------
# Subscription record
# ---------------------------------------------------------------------------


@dataclass
class _Subscription:
    """Internal record for a single event subscription."""

    subscription_id: str
    event_type: type[Event]
    callback: Callable[[Event], None]


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


class EventBus:
    """Thread-safe publish/subscribe event bus.

    Subscribers register interest in a specific ``Event`` subclass and receive
    callbacks whenever a matching event is published. A bounded history deque
    retains recent events for diagnostic queries.

    This class should not be instantiated directly; use :func:`get_event_bus`
    to obtain the process-wide singleton.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscriptions: dict[str, _Subscription] = {}
        self._history: deque[Event] = deque(maxlen=_HISTORY_MAX_LENGTH)
        self._eviction_count: int = 0  # Total events evicted from history ring buffer
        self._async_queue: queue.Queue = queue.Queue(maxsize=EVENTBUS_ASYNC_QUEUE_SIZE)
        self._async_worker: threading.Thread | None = None
        self._shutdown = threading.Event()

    @property
    def eviction_count(self) -> int:
        """Total number of events evicted from the history ring buffer."""
        return self._eviction_count

    # -- public API --------------------------------------------------------

    def subscribe(
        self,
        event_type: type[Event],
        callback: Callable[[Event], None],
    ) -> str:
        """Register a callback for a specific event type.

        Args:
            event_type: The ``Event`` subclass to listen for.
            callback: Function invoked with the event instance when published.

        Returns:
            A unique subscription ID that can be passed to :meth:`unsubscribe`.
        """
        sub_id = uuid.uuid4().hex
        sub = _Subscription(
            subscription_id=sub_id,
            event_type=event_type,
            callback=callback,
        )
        with self._lock:
            self._subscriptions[sub_id] = sub
        logger.debug("Subscribed %s to %s", sub_id, event_type.__name__)
        return sub_id

    def unsubscribe(self, subscription_id: str) -> None:
        """Remove a subscription by its ID.

        Args:
            subscription_id: The ID returned by :meth:`subscribe`.

        Raises:
            KeyError: If the subscription ID is not found.
        """
        with self._lock:
            if subscription_id not in self._subscriptions:
                raise KeyError(f"Subscription not found: {subscription_id}")
            del self._subscriptions[subscription_id]
        logger.debug("Unsubscribed %s", subscription_id)

    def publish(self, event: Event) -> None:
        """Publish an event, invoking all matching subscribers synchronously.

        Subscriber exceptions are caught and logged so that a single bad
        subscriber cannot crash the publisher.

        Args:
            event: The event instance to publish.
        """
        # Deep-copy the event before storing so that subscribers mutating
        # mutable payload fields (e.g. metadata dicts) cannot corrupt the
        # history ring buffer.
        stored_event = copy.deepcopy(event)
        with self._lock:
            if len(self._history) == _HISTORY_MAX_LENGTH:
                self._eviction_count += 1
                if self._eviction_count % 100 == 1:
                    logger.warning(
                        "EventBus history full (%d max) — evicting oldest event (total evictions: %d)",
                        _HISTORY_MAX_LENGTH,
                        self._eviction_count,
                    )
            self._history.append(stored_event)
            matching = [sub for sub in self._subscriptions.values() if isinstance(event, sub.event_type)]

        for sub in matching:
            self._invoke_handler(sub, event)

    def _invoke_handler(self, sub: Any, event: Event) -> None:
        """Run a subscriber callback in a background thread with a timeout.

        Prevents I/O-bound handlers from blocking the publisher thread.
        """
        _HANDLER_TIMEOUT = 5.0  # seconds before logging a slow handler warning

        result_holder: list[Exception | None] = [None]

        def _run() -> None:
            try:
                sub.callback(event)
            except Exception as exc:
                result_holder[0] = exc

        handler_thread = threading.Thread(
            target=_run,
            name=f"eventbus-handler-{sub.subscription_id}",
            daemon=True,
        )
        handler_thread.start()
        handler_thread.join(timeout=_HANDLER_TIMEOUT)

        if handler_thread.is_alive():
            logger.warning(
                "EventBus handler %s exceeded %.1fs timeout for event %s — handler is still running in background",
                sub.subscription_id,
                _HANDLER_TIMEOUT,
                event.event_type,
            )
        elif result_holder[0] is not None:
            logger.error(
                "Subscriber %s raised an exception for event %s: %s",
                sub.subscription_id,
                event.event_type,
                result_holder[0],
                exc_info=result_holder[0],
            )

    def _ensure_async_worker(self) -> None:
        """Start the single async dispatch worker thread if not running."""
        if self._async_worker is not None and self._async_worker.is_alive():
            return
        self._shutdown.clear()
        self._async_worker = threading.Thread(
            target=self._async_dispatch_loop,
            daemon=True,
            name="eventbus-async",
        )
        self._async_worker.start()

    def _async_dispatch_loop(self) -> None:
        """Single worker thread that drains the async event queue."""
        while not self._shutdown.is_set():
            timed_out = False
            item = None
            try:
                item = self._async_queue.get(timeout=QUEUE_TIMEOUT)
            except queue.Empty:
                # Normal poll timeout — no event to dispatch; loop and check shutdown flag.
                timed_out = True
            if timed_out:
                continue
            if item is None:
                # Sentinel — shutdown() was called; exit immediately.
                return
            event, matching = item
            for sub in matching:
                try:
                    sub.callback(event)
                except Exception:
                    logger.exception(
                        "Async subscriber %s raised an exception for event %s",
                        sub.subscription_id,
                        event.event_type,
                    )

    def shutdown(self) -> None:
        """Signal the async worker thread to stop and wait for it to exit.

        Unlike :meth:`clear`, this does not remove subscriptions or history.
        Call this during application shutdown to release the background thread
        cleanly before the process exits.
        """
        self._shutdown.set()
        # Put a sentinel so the worker unblocks immediately instead of waiting
        # for the 1-second queue.get() timeout.
        with contextlib.suppress(queue.Full):
            self._async_queue.put_nowait(None)  # type: ignore[arg-type]
        if self._async_worker is not None and self._async_worker.is_alive():
            self._async_worker.join(timeout=THREAD_JOIN_TIMEOUT)
        self._async_worker = None

    def publish_async(self, event: Event) -> None:
        """Publish an event, invoking all matching subscribers in a background thread.

        The history is updated immediately (under lock) before dispatching
        callbacks asynchronously via a single reusable worker thread. The
        async queue uses a bounded put with a short timeout so the caller
        is never blocked indefinitely when the worker falls behind.

        Args:
            event: The event instance to publish.
        """
        # Deep-copy before storing in history — same reason as publish().
        stored_event = copy.deepcopy(event)
        with self._lock:
            if len(self._history) == _HISTORY_MAX_LENGTH:
                self._eviction_count += 1
                if self._eviction_count % 100 == 1:
                    logger.warning(
                        "EventBus history full (%d max) — evicting oldest event (total evictions: %d)",
                        _HISTORY_MAX_LENGTH,
                        self._eviction_count,
                    )
            self._history.append(stored_event)
            matching = [sub for sub in self._subscriptions.values() if isinstance(event, sub.event_type)]

        if matching:
            try:
                self._async_queue.put((event, matching), timeout=QUEUE_TIMEOUT)
            except queue.Full:
                logger.warning(
                    "EventBus async queue full (size=%d) — dropping dispatch for event type '%s'",
                    EVENTBUS_ASYNC_QUEUE_SIZE,
                    event.event_type,
                )
                return
            self._ensure_async_worker()

    def clear(self) -> None:
        """Remove all subscriptions and history, stop async worker.

        Intended for use in test teardown to avoid cross-test interference.
        Resets the shutdown event so the bus can be reused after clearing.
        """
        self._shutdown.set()
        if self._async_worker is not None and self._async_worker.is_alive():
            self._async_worker.join(timeout=THREAD_JOIN_TIMEOUT_SHORT)
        self._async_worker = None
        # Drain any remaining items from the queue
        while not self._async_queue.empty():
            try:
                self._async_queue.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            self._subscriptions.clear()
            self._history.clear()
            self._eviction_count = 0
        # Reset the shutdown flag so the bus is usable again after clearing.
        # Without this, the next publish_async call would find _shutdown already
        # set and the worker thread would exit immediately on start.
        self._shutdown.clear()
        logger.debug("EventBus cleared")

    def get_history(
        self,
        event_type: type[Event] | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """Return recent events, optionally filtered by type.

        Args:
            event_type: If provided, only events that are instances of this
                type are returned. ``None`` returns all event types.
            limit: Maximum number of events to return (must be positive).
                Newest events appear last (chronological order).

        Returns:
            A list of up to *limit* events in chronological order.

        Raises:
            ValueError: If *limit* is not a positive integer.
        """
        if limit <= 0:
            raise ValueError(f"limit must be a positive integer, got {limit!r}")
        with self._lock:
            if event_type is None:
                items = list(self._history)
            else:
                items = [e for e in self._history if isinstance(e, event_type)]
        return items[-limit:]


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------

_event_bus: EventBus | None = None
_singleton_lock = threading.Lock()


def _log_event_for_observability(event: Event) -> None:
    """Log every published event at INFO level for operational observability.

    This is the mandatory subscriber that ensures the EventBus is not
    write-only.  All published events are captured here so operators can
    observe system activity through standard log aggregation tools.

    Args:
        event: The event that was published.
    """
    logger.info("[EventBus] %r", event)


def get_event_bus() -> EventBus:
    """Return the process-wide ``EventBus`` singleton.

    Thread-safe; the instance is created on first call.  The first call also
    registers an observability subscriber that logs every published event at
    INFO level, ensuring no event is published without a reader.

    Returns:
        The shared ``EventBus`` instance.
    """
    global _event_bus
    if _event_bus is None:
        with _singleton_lock:
            if _event_bus is None:
                _event_bus = EventBus()
                # Register the observability subscriber so every published event
                # is logged.  This satisfies the requirement that the EventBus
                # has at least one subscriber.
                _event_bus.subscribe(Event, _log_event_for_observability)
    return _event_bus


def reset_event_bus() -> None:
    """Destroy the current singleton so the next call to :func:`get_event_bus` creates a fresh instance.

    Intended for test isolation. Also calls :meth:`EventBus.clear` on the
    existing instance before discarding it.
    """
    global _event_bus
    with _singleton_lock:
        if _event_bus is not None:
            _event_bus.shutdown()
            _event_bus.clear()
        _event_bus = None
