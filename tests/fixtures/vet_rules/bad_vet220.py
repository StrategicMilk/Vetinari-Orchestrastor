"""Module with unbounded list[float] fields in an analytics class.

This fixture intentionally violates VET220: class-level list[float] and
list[dict] annotations in an analytics context should use bounded collections.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PerformanceTracker:
    """Tracks performance metrics over time without bounding stored values."""

    latencies: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    metadata: list[dict] = field(default_factory=list)
