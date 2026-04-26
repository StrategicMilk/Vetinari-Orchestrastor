"""Shared fixtures for performance benchmarks.

Provides the benchmark_timer context manager and pytest markers used across
all benchmark test files.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Generator
from dataclasses import dataclass

import pytest


@dataclass
class TimingResult:
    """Wall-clock elapsed time captured by benchmark_timer.

    Args:
        elapsed_s: Elapsed time in seconds.
        operation: Human-readable name of the measured operation.
    """

    elapsed_s: float
    operation: str

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_s * 1000.0


@contextlib.contextmanager
def benchmark_timer(operation: str = "operation") -> Generator[TimingResult, None, None]:
    """Measure wall-clock elapsed time for a block of code.

    Yields a mutable TimingResult so the elapsed time is available after the
    ``with`` block exits. The result is populated when the context manager
    exits — do not read ``elapsed_s`` before the ``with`` block closes.

    Args:
        operation: Label for the operation being timed, used in assertions.

    Yields:
        TimingResult with ``elapsed_s`` populated after the block completes.

    Example::

        with benchmark_timer("plan_generation") as t:
            plan = engine.generate_plan(request)
        assert t.elapsed_s < 10.0
    """
    result = TimingResult(elapsed_s=0.0, operation=operation)
    start = time.monotonic()
    try:
        yield result
    finally:
        result.elapsed_s = time.monotonic() - start


def pytest_configure(config: pytest.Config) -> None:
    """Register the benchmark marker so pytest does not warn about unknown marks.

    Args:
        config: The pytest configuration object.
    """
    config.addinivalue_line(
        "markers",
        "benchmark: marks a test as a performance benchmark (may be slow)",
    )
