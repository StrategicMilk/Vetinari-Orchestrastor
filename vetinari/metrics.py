"""Metrics collection module for Vetinari.

Provides a thread-safe MetricsCollector for counters and histograms,
a module-level singleton, and convenience helper functions for
common Vetinari metrics (task duration, task count, model latency,
API request counts).

Usage:
    from vetinari.metrics import get_metrics, record_task_duration

    record_task_duration("task-123", 42.0)
    get_metrics().get_counter("vetinari.task.count", status=StatusEnum.COMPLETED.value)
"""

from __future__ import annotations

import logging
import threading
from collections import deque

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Thread-safe metrics collector for observability.

    Collects counters and histograms for monitoring. All operations
    acquire an internal lock, making the collector safe to use from
    multiple threads simultaneously.
    """

    def __init__(self) -> None:
        """Initialise empty counter and histogram stores."""
        self._counters: dict[str, int] = {}
        self._histograms: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def increment(self, metric_name: str, value: int = 1, **tags: object) -> None:
        """Increment a counter metric.

        Args:
            metric_name: The metric name.
            value: Amount to add to the counter (default 1).
            **tags: Dimension tags used to qualify the metric key.
        """
        key = self._make_key(metric_name, tags)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def record(self, metric_name: str, value: float, **tags: object) -> None:
        """Record a histogram value.

        Args:
            metric_name: The metric name.
            value: The numeric observation to record.
            **tags: Dimension tags used to qualify the metric key.
        """
        key = self._make_key(metric_name, tags)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = deque(maxlen=10000)
            self._histograms[key].append(value)

    def get_counter(self, metric_name: str, **tags: object) -> int:
        """Return the current counter value for a metric key.

        Args:
            metric_name: The metric name.
            **tags: Dimension tags used to qualify the metric key.

        Returns:
            Current counter value, or 0 if never incremented.
        """
        key = self._make_key(metric_name, tags)
        with self._lock:
            return self._counters.get(key, 0)

    def get_histogram_stats(
        self,
        metric_name: str,
        **tags: object,
    ) -> dict[str, float] | None:
        """Return summary statistics for a histogram metric.

        Args:
            metric_name: The metric name.
            **tags: Dimension tags used to qualify the metric key.

        Returns:
            Dictionary with keys ``count``, ``sum``, ``min``, ``max``,
            ``avg``, ``p50``, ``p95``, ``p99``, or ``None`` if no
            observations have been recorded.
        """
        key = self._make_key(metric_name, tags)
        with self._lock:
            values = self._histograms.get(key, [])
            if not values:
                return None

            sorted_values = sorted(values)
            return {
                "count": len(values),
                "sum": sum(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "p50": sorted_values[len(sorted_values) // 2],
                "p95": sorted_values[int(len(sorted_values) * 0.95)],
                "p99": sorted_values[int(len(sorted_values) * 0.99)],
            }

    def _make_key(self, metric_name: str, tags: dict[str, object]) -> str:
        """Build a unique metric key from name and tags.

        Args:
            metric_name: Base metric name.
            tags: Tag dictionary to encode into the key.

        Returns:
            A deterministic string key such as
            ``"vetinari.task.count{status=completed}"``.
        """
        if not tags:
            return metric_name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{metric_name}{{{tag_str}}}"


# Global metrics collector singleton
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Return the process-wide metrics collector singleton.

    Returns:
        The shared ``MetricsCollector`` instance.
    """
    return _metrics


def record_task_duration(task_id: str, duration_ms: float) -> None:
    """Record task execution duration in the global metrics collector.

    The task identifier is logged for debugging but is intentionally not used
    as a metric tag; task IDs are high-cardinality values that would fragment
    the histogram and make aggregate duration stats unavailable.

    Args:
        task_id: Identifier of the task whose duration is being recorded.
        duration_ms: Elapsed time in milliseconds.
    """
    logger.debug("Recording task duration for %s: %.2f ms", task_id, duration_ms)
    _metrics.record("vetinari.task.duration", duration_ms, task_type="generic")


def increment_task_count(status: str) -> None:
    """Increment the global task counter for a given status.

    Args:
        status: Completion status label (e.g. ``"completed"``, ``"failed"``).
    """
    _metrics.increment("vetinari.task.count", status=status)


def record_model_latency(duration_ms: float) -> None:
    """Record model inference latency in the global metrics collector.

    Args:
        duration_ms: Elapsed inference time in milliseconds.
    """
    _metrics.record("vetinari.model.latency", duration_ms)


def increment_api_request(status_code: int) -> None:
    """Increment the API request counter for a given HTTP status code.

    Args:
        status_code: HTTP response status code (e.g. 200, 404, 500).
    """
    _metrics.increment("vetinari.api.request", status=status_code)
