"""Module with bounded metrics storage using collections.deque.

This fixture satisfies VET220: uses deque with maxlen instead of unbounded
list[float] or list[dict] fields so memory usage is capped automatically.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class PerformanceTracker:
    """Tracks performance metrics over time with bounded storage."""

    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    scores: deque = field(default_factory=lambda: deque(maxlen=1000))
    metadata: deque = field(default_factory=lambda: deque(maxlen=500))
