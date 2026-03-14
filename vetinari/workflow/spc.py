"""Statistical Process Control (SPC) & Manufacturing Controls.

===========================================================
Provides:

* **ControlChart** -- per-metric control chart with UCL/LCL, Cpk.
* **SPCMonitor**   -- multi-metric SPC monitor with alert generation.
* **AndonSignal / AndonSystem** -- stop-the-line alerts for critical failures.
* **WIPConfig / WIPTracker**   -- kanban-style WIP limits per agent type.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# =========================================================================
# SPC -- Statistical Process Control
# =========================================================================


@dataclass
class SPCAlert:
    """Alert raised when a metric goes out of statistical control."""

    metric_name: str
    value: float
    ucl: float
    lcl: float
    mean: float
    sigma: float
    alert_type: str  # "above_ucl", "below_lcl", "trend", "run"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ControlChart:
    """A control chart for a single metric.

    Maintains a rolling window of observations and derives the mean,
    standard deviation, and upper/lower control limits (mean +/- k * sigma).
    """

    metric_name: str
    values: list[float] = field(default_factory=list)
    timestamps: list[str] = field(default_factory=list)
    window_size: int = 30
    sigma_multiplier: float = 3.0

    # -- derived statistics -------------------------------------------------

    @property
    def _window(self) -> list[float]:
        """Return the most recent *window_size* values."""
        return self.values[-self.window_size :]

    @property
    def mean(self) -> float:
        """Arithmetic mean of the current window."""
        w = self._window
        if not w:
            return 0.0
        return sum(w) / len(w)

    @property
    def sigma(self) -> float:
        """Population standard deviation of the current window."""
        w = self._window
        n = len(w)
        if n < 2:
            return 0.0
        m = self.mean
        variance = sum((x - m) ** 2 for x in w) / n
        return math.sqrt(variance)

    @property
    def ucl(self) -> float:
        """Upper control limit = mean + k * sigma."""
        return self.mean + self.sigma_multiplier * self.sigma

    @property
    def lcl(self) -> float:
        """Lower control limit = mean - k * sigma."""
        return self.mean - self.sigma_multiplier * self.sigma

    # -- capability ---------------------------------------------------------

    def get_cpk(
        self,
        spec_upper: float | None = None,
        spec_lower: float | None = None,
    ) -> float | None:
        """Process capability index.

        * Cpk > 1.33  -- capable process
        * 1.0 < Cpk <= 1.33 -- marginally capable
        * Cpk <= 1.0  -- needs intervention

        Returns ``None`` when sigma is zero or no spec limits are given.

        Args:
            spec_upper: The spec upper.
            spec_lower: The spec lower.

        Returns:
            The computed value.
        """
        s = self.sigma
        if s == 0.0:
            return None
        m = self.mean

        cpk_values: list[float] = []
        if spec_upper is not None:
            cpk_values.append((spec_upper - m) / (3 * s))
        if spec_lower is not None:
            cpk_values.append((m - spec_lower) / (3 * s))

        if not cpk_values:
            return None
        return min(cpk_values)

    # -- control check ------------------------------------------------------

    def is_in_control(self, value: float) -> bool:
        """Return ``True`` if *value* falls within the control limits.

        Returns:
            True if successful, False otherwise.
        """
        if self.sigma == 0.0:
            return True
        return self.lcl <= value <= self.ucl

    def add_value(self, value: float, timestamp: str | None = None) -> None:
        """Append an observation to the chart.

        Args:
            value: The value.
            timestamp: The timestamp.
        """
        self.values.append(value)
        self.timestamps.append(timestamp or datetime.now().isoformat())

    # -- trend / run detection (Western Electric rules) ---------------------

    def _detect_trend(self, min_consecutive: int = 7) -> bool:
        """Detect a monotone trend of *min_consecutive* points."""
        w = self._window
        if len(w) < min_consecutive:
            return False
        tail = w[-min_consecutive:]
        increasing = all(tail[i] < tail[i + 1] for i in range(len(tail) - 1))
        decreasing = all(tail[i] > tail[i + 1] for i in range(len(tail) - 1))
        return increasing or decreasing

    def _detect_run(self, min_run: int = 8) -> bool:
        """Detect a run of *min_run* consecutive points on one side of the mean."""
        w = self._window
        if len(w) < min_run:
            return False
        m = self.mean
        tail = w[-min_run:]
        above = all(v > m for v in tail)
        below = all(v < m for v in tail)
        return above or below


# =========================================================================
# SPC Monitor
# =========================================================================


class SPCMonitor:
    """Multi-metric SPC monitor with alert generation.

    Usage::

        monitor = SPCMonitor()
        alert = monitor.update("quality_score", 0.92)
        if alert:
            logger.debug("Out of control: %s", alert.alert_type)
    """

    DEFAULT_METRICS = ["quality_score", "latency_ms", "token_count", "drift_score"]

    def __init__(self, window_size: int = 30, sigma_multiplier: float = 3.0):
        self._charts: dict[str, ControlChart] = {}
        self._alerts: list[SPCAlert] = []
        self._window_size = window_size
        self._sigma_multiplier = sigma_multiplier

    # -- public API ---------------------------------------------------------

    def update(self, metric_name: str, value: float) -> SPCAlert | None:
        """Record a new observation and return an alert if out of control.

        Args:
            metric_name: The metric name.
            value: The value.

        Returns:
            The SPCAlert | None result.
        """
        chart = self._get_or_create_chart(metric_name)

        # We need at least a few data points before alerting.
        chart.add_value(value)
        if len(chart.values) < 3:
            return None

        alert = self._check_control(chart, value)
        if alert is not None:
            self._alerts.append(alert)
            logger.warning(
                "SPC alert for %s: %s (value=%.4f, UCL=%.4f, LCL=%.4f)",
                metric_name,
                alert.alert_type,
                value,
                chart.ucl,
                chart.lcl,
            )
        return alert

    def get_chart(self, metric_name: str) -> ControlChart | None:
        """Return the control chart for *metric_name*, or ``None``."""
        return self._charts.get(metric_name)

    def get_alerts(self, metric_name: str | None = None) -> list[SPCAlert]:
        """Return alerts, optionally filtered by metric name.

        Returns:
            List of results.
        """
        if metric_name is None:
            return list(self._alerts)
        return [a for a in self._alerts if a.metric_name == metric_name]

    def get_summary(self) -> dict[str, Any]:
        """Return a summary dict of all monitored metrics.

        Returns:
            The result string.
        """
        summary: dict[str, Any] = {}
        for name, chart in self._charts.items():
            summary[name] = {
                "count": len(chart.values),
                "mean": chart.mean,
                "sigma": chart.sigma,
                "ucl": chart.ucl,
                "lcl": chart.lcl,
                "in_control": chart.is_in_control(chart.values[-1]) if chart.values else True,
                "alerts": len([a for a in self._alerts if a.metric_name == name]),
            }
        return summary

    # -- internals ----------------------------------------------------------

    def _get_or_create_chart(self, metric_name: str) -> ControlChart:
        if metric_name not in self._charts:
            self._charts[metric_name] = ControlChart(
                metric_name=metric_name,
                window_size=self._window_size,
                sigma_multiplier=self._sigma_multiplier,
            )
        return self._charts[metric_name]

    def _check_control(self, chart: ControlChart, value: float) -> SPCAlert | None:
        """Check whether *value* is out of control on *chart*."""
        # Basic limit check
        if value > chart.ucl:
            return SPCAlert(
                metric_name=chart.metric_name,
                value=value,
                ucl=chart.ucl,
                lcl=chart.lcl,
                mean=chart.mean,
                sigma=chart.sigma,
                alert_type="above_ucl",
            )
        if value < chart.lcl:
            return SPCAlert(
                metric_name=chart.metric_name,
                value=value,
                ucl=chart.ucl,
                lcl=chart.lcl,
                mean=chart.mean,
                sigma=chart.sigma,
                alert_type="below_lcl",
            )

        # Western Electric supplementary rules
        if chart._detect_trend():
            return SPCAlert(
                metric_name=chart.metric_name,
                value=value,
                ucl=chart.ucl,
                lcl=chart.lcl,
                mean=chart.mean,
                sigma=chart.sigma,
                alert_type="trend",
            )
        if chart._detect_run():
            return SPCAlert(
                metric_name=chart.metric_name,
                value=value,
                ucl=chart.ucl,
                lcl=chart.lcl,
                mean=chart.mean,
                sigma=chart.sigma,
                alert_type="run",
            )
        return None


# =========================================================================
# Andon System
# =========================================================================


@dataclass
class AndonSignal:
    """Stop-the-line alert for critical failures."""

    source: str  # Which gate / stage triggered it
    severity: str  # "warning", "critical", "emergency"
    message: str
    affected_tasks: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    acknowledged: bool = False


class AndonSystem:
    """Manages stop-the-line alerts.

    When a *critical* or *emergency* signal is raised the system enters
    a **paused** state.  Execution should not proceed until every
    critical/emergency signal has been explicitly acknowledged.
    """

    PAUSE_SEVERITIES = frozenset({"critical", "emergency"})

    def __init__(self) -> None:
        self._signals: list[AndonSignal] = []
        self._paused: bool = False

    # -- public API ---------------------------------------------------------

    def raise_signal(
        self,
        source: str,
        severity: str,
        message: str,
        affected_tasks: list[str] | None = None,
    ) -> AndonSignal:
        """Raise an Andon signal.

        Signals with severity ``critical`` or ``emergency`` automatically
        pause execution.

        Args:
            source: The source.
            severity: The severity.
            message: The message.
            affected_tasks: The affected tasks.

        Returns:
            The AndonSignal result.
        """
        signal = AndonSignal(
            source=source,
            severity=severity,
            message=message,
            affected_tasks=affected_tasks or [],
        )
        self._signals.append(signal)

        if severity in self.PAUSE_SEVERITIES:
            self._paused = True
            logger.critical(
                "ANDON %s from %s: %s (tasks: %s)",
                severity.upper(),
                source,
                message,
                affected_tasks or [],
            )
        else:
            logger.warning(
                "ANDON %s from %s: %s",
                severity.upper(),
                source,
                message,
            )

        return signal

    def acknowledge(self, signal_index: int) -> bool:
        """Acknowledge a signal by its index.

        After acknowledgment, if no unacknowledged critical/emergency
        signals remain the system resumes (unpauses).

        Returns ``True`` if the signal was found and acknowledged.

        Returns:
            True if successful, False otherwise.
        """
        if signal_index < 0 or signal_index >= len(self._signals):
            return False

        self._signals[signal_index].acknowledged = True

        # Check whether we can unpause
        still_critical = any(not s.acknowledged and s.severity in self.PAUSE_SEVERITIES for s in self._signals)
        if not still_critical:
            self._paused = False
            logger.info("Andon system resumed -- all critical signals acknowledged")

        return True

    def is_paused(self) -> bool:
        """Return ``True`` while unacknowledged critical/emergency signals exist."""
        return self._paused

    def get_active_signals(self) -> list[AndonSignal]:
        """Return all signals that have **not** been acknowledged."""
        return [s for s in self._signals if not s.acknowledged]

    def get_all_signals(self) -> list[AndonSignal]:
        """Return every signal (acknowledged or not)."""
        return list(self._signals)


# =========================================================================
# WIP Limits
# =========================================================================


@dataclass
class WIPConfig:
    """Work-in-progress limits per agent type.

    Default limits mirror a typical kanban board:

    * Bottleneck roles (PLANNER, ARCHITECT, META, RESILIENCE) get 1 slot.
    * Verification / documentation roles get 2--3 slots.
    * Generic fallback is 4 concurrent tasks.
    """

    limits: dict[str, int] = field(
        default_factory=lambda: {
            "BUILDER": 2,
            "TESTER": 3,
            "RESEARCHER": 2,
            "PLANNER": 1,
            "ARCHITECT": 1,
            "DOCUMENTER": 2,
            "RESILIENCE": 1,
            "META": 1,
            "default": 4,
        }
    )

    def get_limit(self, agent_type: str) -> int:
        """Return the WIP limit for *agent_type*."""
        return self.limits.get(agent_type, self.limits.get("default", 4))


class WIPTracker:
    """Track and enforce WIP limits (kanban-style pull system).

    Usage::

        tracker = WIPTracker()
        if tracker.can_start("BUILDER"):
            tracker.start_task("BUILDER", "task-42")
        else:
            tracker.enqueue("BUILDER", "task-42")
    """

    def __init__(self, config: WIPConfig | None = None) -> None:
        self._config = config or WIPConfig()
        self._active: dict[str, list[str]] = {}  # agent_type -> [task_ids]
        self._queued: list[dict[str, Any]] = []  # waiting for capacity

    # -- queries ------------------------------------------------------------

    def can_start(self, agent_type: str) -> bool:
        """Return ``True`` if the agent type has capacity for another task.

        Returns:
            True if successful, False otherwise.
        """
        limit = self._config.get_limit(agent_type)
        current = len(self._active.get(agent_type, []))
        return current < limit

    def get_active_count(self, agent_type: str) -> int:
        """Return the number of currently active tasks for *agent_type*."""
        return len(self._active.get(agent_type, []))

    def get_queue_depth(self) -> int:
        """Return the total number of queued (waiting) tasks."""
        return len(self._queued)

    def get_utilization(self) -> dict[str, dict[str, Any]]:
        """Return utilization info for every known agent type.

        Returns:
            The result string.
        """
        types = set(self._config.limits.keys()) | set(self._active.keys())
        types.discard("default")
        result: dict[str, dict[str, Any]] = {}
        for agent_type in sorted(types):
            limit = self._config.get_limit(agent_type)
            active = len(self._active.get(agent_type, []))
            result[agent_type] = {
                "active": active,
                "limit": limit,
                "utilization": active / limit if limit > 0 else 0.0,
                "available": max(0, limit - active),
            }
        return result

    # -- mutations ----------------------------------------------------------

    def start_task(self, agent_type: str, task_id: str) -> bool:
        """Start a task if capacity allows.  Returns ``True`` on success.

        Args:
            agent_type: The agent type.
            task_id: The task id.

        Returns:
            True if successful, False otherwise.
        """
        if not self.can_start(agent_type):
            logger.warning(
                "WIP limit reached for %s (limit=%d) -- cannot start %s",
                agent_type,
                self._config.get_limit(agent_type),
                task_id,
            )
            return False

        self._active.setdefault(agent_type, []).append(task_id)
        logger.debug("WIP started %s for %s", task_id, agent_type)
        return True

    def complete_task(self, agent_type: str, task_id: str) -> dict[str, Any] | None:
        """Mark a task as complete and pull the next queued task (if any).

        Returns the dequeued task dict, or ``None`` if the queue is empty.

        Args:
            agent_type: The agent type.
            task_id: The task id.

        Returns:
            The result string.
        """
        tasks = self._active.get(agent_type, [])
        if task_id in tasks:
            tasks.remove(task_id)
            logger.debug("WIP completed %s for %s", task_id, agent_type)

        # Pull from the queue (first task matching this agent type)
        for i, queued in enumerate(self._queued):
            if queued.get("agent_type") == agent_type:
                pulled = self._queued.pop(i)
                # Auto-start the pulled task
                self._active.setdefault(agent_type, []).append(pulled["task_id"])
                logger.info(
                    "WIP pulled %s from queue for %s",
                    pulled["task_id"],
                    agent_type,
                )
                return pulled

        return None

    def enqueue(self, agent_type: str, task_id: str, **extra: Any) -> None:
        """Place a task in the waiting queue.

        Args:
            agent_type: The agent type.
            task_id: The task id.
            **extra: Additional key-value pairs to store with the entry.
        """
        entry: dict[str, Any] = {
            "agent_type": agent_type,
            "task_id": task_id,
            "enqueued_at": datetime.now().isoformat(),
        }
        entry.update(extra)
        self._queued.append(entry)
        logger.debug("WIP enqueued %s for %s", task_id, agent_type)
