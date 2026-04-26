"""Bounded metrics collection using fixed-size deques.

Provides ``BoundedMetrics``, a base class for any component that needs
to track a sliding window of numeric observations without unbounded
memory growth.  Analytics, anomaly detection, and quality-drift modules
should subclass or compose this instead of appending to a plain ``list``.
"""

from __future__ import annotations

import statistics
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

__all__ = ["BoundedMetrics"]

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_WINDOW_SIZE: int = 1000  # Retain last 1 000 observations


@dataclass
class BoundedMetrics:
    """Thread-safe, fixed-size sliding-window metric collector.

    Stores up to *maxlen* observations in a :class:`collections.deque`.
    Oldest values are silently evicted when the window is full.

    Provides convenience statistics (mean, median, stddev, percentile)
    that operate only on the current window.

    Args:
        maxlen: Maximum number of observations to retain.
    """

    maxlen: int = DEFAULT_WINDOW_SIZE
    _values: deque[float] = field(init=False, repr=False)
    _lock: threading.Lock = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_values", deque(maxlen=self.maxlen))
        object.__setattr__(self, "_lock", threading.Lock())

    # ── Mutation ────────────────────────────────────────────────────────────

    def record(self, value: float) -> None:
        """Append a single observation to the sliding window.

        Args:
            value: The numeric value to record.
        """
        with self._lock:
            self._values.append(value)

    def record_many(self, values: list[float] | tuple[float, ...]) -> None:
        """Append multiple observations at once.

        Args:
            values: Sequence of numeric values to record.
        """
        with self._lock:
            self._values.extend(values)

    def clear(self) -> None:
        """Remove all recorded observations."""
        with self._lock:
            self._values.clear()

    # ── Queries ─────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        """Number of observations currently in the window."""
        return len(self._values)

    @property
    def is_empty(self) -> bool:
        """Whether the window contains zero observations."""
        return len(self._values) == 0

    def snapshot(self) -> list[float]:
        """Return a copy of all values currently in the window.

        Returns:
            A plain ``list[float]`` snapshot (safe to iterate without locks).
        """
        with self._lock:
            return list(self._values)

    def mean(self) -> float:
        """Arithmetic mean of the current window.

        Returns:
            The mean, or ``0.0`` if the window is empty.
        """
        vals = self.snapshot()
        return statistics.mean(vals) if vals else 0.0

    def median(self) -> float:
        """Median of the current window.

        Returns:
            The median, or ``0.0`` if the window is empty.
        """
        vals = self.snapshot()
        return statistics.median(vals) if vals else 0.0

    def stddev(self) -> float:
        """Population standard deviation of the current window.

        Returns:
            The stddev, or ``0.0`` if fewer than 2 observations.
        """
        vals = self.snapshot()
        return statistics.pstdev(vals) if len(vals) >= 2 else 0.0

    def percentile(self, p: float) -> float:
        """Compute the *p*-th percentile (0-100) of the current window.

        Args:
            p: Percentile rank between 0 and 100 inclusive.

        Returns:
            The percentile value, or ``0.0`` if the window is empty.

        Raises:
            ValueError: If *p* is outside [0, 100].
        """
        if not 0.0 <= p <= 100.0:
            msg = f"Percentile must be between 0 and 100, got {p}"
            raise ValueError(msg)
        vals = self.snapshot()
        if not vals:
            return 0.0
        sorted_vals = sorted(vals)
        idx = (p / 100.0) * (len(sorted_vals) - 1)
        lower = int(idx)
        upper = min(lower + 1, len(sorted_vals) - 1)
        weight = idx - lower
        return sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight

    def to_dict(self) -> dict[str, Any]:
        """Serialize current statistics to a dictionary.

        Returns:
            Dict with ``count``, ``mean``, ``median``, ``stddev``, ``p95``, ``p99``.
        """
        return {
            "count": self.count,
            "mean": round(self.mean(), 4),
            "median": round(self.median(), 4),
            "stddev": round(self.stddev(), 4),
            "p95": round(self.percentile(95), 4),
            "p99": round(self.percentile(99), 4),
        }
