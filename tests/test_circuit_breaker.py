"""
Tests for vetinari.orchestration.stagnation and vetinari.resilience.circuit_breaker
===================================================================================

Covers StagnationDetector repeated-output / error / time-budget detection.
CircuitBreaker (resilience version) is tested in test_anomaly_circuit_breaker.py.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from vetinari.orchestration.stagnation import StagnationDetector
from vetinari.resilience import CircuitBreaker, ResilienceCircuitState
from vetinari.resilience.circuit_breaker import CircuitBreakerConfig

# ---------------------------------------------------------------------------
# CircuitBreaker (resilience) — basic smoke tests
# ---------------------------------------------------------------------------


class TestResilienceCircuitBreakerSmoke:
    """Smoke tests for the production-grade circuit breaker."""

    def test_defaults(self) -> None:
        cb = CircuitBreaker("test")
        assert cb.state is ResilienceCircuitState.CLOSED
        assert cb.config.failure_threshold == 5  # Local models need more chances
        assert cb.config.recovery_timeout == 30.0  # Local models recover faster

    def test_opens_at_threshold(self) -> None:
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=2))
        cb.record_failure()
        cb.record_failure()
        assert cb.state is ResilienceCircuitState.OPEN

    def test_open_blocks_requests(self) -> None:
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=1))
        cb.record_failure()
        assert cb.allow_request() is False

    def test_half_open_after_timeout(self) -> None:
        cb = CircuitBreaker(
            "test",
            config=CircuitBreakerConfig(failure_threshold=1, recovery_timeout=10.0),
        )
        cb.record_failure()
        assert cb.state is ResilienceCircuitState.OPEN
        # Simulate time passing beyond recovery_timeout
        cb._opened_at -= 11.0
        assert cb.state is ResilienceCircuitState.HALF_OPEN
        assert cb.allow_request() is True

    def test_half_open_success_closes(self) -> None:
        cb = CircuitBreaker(
            "test",
            config=CircuitBreakerConfig(failure_threshold=1, recovery_timeout=10.0),
        )
        cb.record_failure()
        cb._opened_at -= 11.0
        assert cb.state is ResilienceCircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state is ResilienceCircuitState.CLOSED

    def test_full_cycle(self) -> None:
        """CLOSED -> OPEN -> HALF_OPEN -> CLOSED round trip."""
        cb = CircuitBreaker(
            "test",
            config=CircuitBreakerConfig(failure_threshold=2, recovery_timeout=10.0),
        )
        cb.record_failure()
        cb.record_failure()
        assert cb.state is ResilienceCircuitState.OPEN
        cb._opened_at -= 11.0
        assert cb.state is ResilienceCircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state is ResilienceCircuitState.CLOSED


# ---------------------------------------------------------------------------
# StagnationDetector — construction
# ---------------------------------------------------------------------------


class TestStagnationDetectorInit:
    """Verify construction and parameter validation."""

    def test_defaults(self) -> None:
        d = StagnationDetector()
        assert d.max_repeated_outputs == 3
        assert d.error_threshold == 10
        assert d.time_budget is None
        assert d.error_count == 0
        assert d.repeat_count == 0
        assert not d.is_stagnant()

    def test_invalid_max_repeated_outputs(self) -> None:
        with pytest.raises(ValueError, match="max_repeated_outputs"):
            StagnationDetector(max_repeated_outputs=0)

    def test_invalid_error_threshold(self) -> None:
        with pytest.raises(ValueError, match="error_threshold"):
            StagnationDetector(error_threshold=0)

    def test_invalid_time_budget(self) -> None:
        with pytest.raises(ValueError, match="time_budget"):
            StagnationDetector(time_budget=-1.0)

        with pytest.raises(ValueError, match="time_budget"):
            StagnationDetector(time_budget=0.0)


# ---------------------------------------------------------------------------
# StagnationDetector — repeated output detection
# ---------------------------------------------------------------------------


class TestStagnationRepeatedOutput:
    """Verify repeated-output stagnation trigger."""

    def test_not_stagnant_with_varied_outputs(self) -> None:
        d = StagnationDetector(max_repeated_outputs=3)
        d.record_output("a")
        d.record_output("b")
        d.record_output("c")
        assert not d.is_stagnant()

    def test_stagnant_after_repeated_outputs(self) -> None:
        d = StagnationDetector(max_repeated_outputs=3)
        d.record_output("same")
        d.record_output("same")
        assert not d.is_stagnant()
        d.record_output("same")
        assert d.is_stagnant()

    def test_repeat_count_resets_on_different_output(self) -> None:
        d = StagnationDetector(max_repeated_outputs=3)
        d.record_output("x")
        d.record_output("x")
        d.record_output("y")  # breaks the streak
        assert d.repeat_count == 1
        assert not d.is_stagnant()

    def test_output_history_tracked(self) -> None:
        d = StagnationDetector()
        d.record_output("a")
        d.record_output("b")
        assert d.output_history == ["a", "b"]

    def test_single_repeat_threshold(self) -> None:
        """Stagnation requires at least one repetition (two identical outputs)."""
        d = StagnationDetector(max_repeated_outputs=2)
        d.record_output("anything")
        assert not d.is_stagnant()  # First output is not a repeat
        d.record_output("anything")
        assert d.is_stagnant()  # Second identical output triggers stagnation


# ---------------------------------------------------------------------------
# StagnationDetector — error threshold detection
# ---------------------------------------------------------------------------


class TestStagnationErrorThreshold:
    """Verify error-count stagnation trigger."""

    def test_not_stagnant_below_threshold(self) -> None:
        d = StagnationDetector(error_threshold=3)
        d.record_error()
        d.record_error()
        assert not d.is_stagnant()

    def test_stagnant_at_threshold(self) -> None:
        d = StagnationDetector(error_threshold=3)
        for _ in range(3):
            d.record_error()
        assert d.is_stagnant()

    def test_stagnant_above_threshold(self) -> None:
        d = StagnationDetector(error_threshold=2)
        for _ in range(5):
            d.record_error()
        assert d.is_stagnant()
        assert d.error_count == 5


# ---------------------------------------------------------------------------
# StagnationDetector — time budget detection
# ---------------------------------------------------------------------------


class TestStagnationTimeBudget:
    """Verify time-budget stagnation trigger."""

    def test_not_stagnant_within_budget(self) -> None:
        d = StagnationDetector(time_budget=100.0)
        assert not d.is_stagnant()

    def test_stagnant_when_budget_exceeded(self) -> None:
        d = StagnationDetector(time_budget=60.0)
        # Simulate time passing beyond the budget
        d._start_time -= 61.0
        assert d.is_stagnant()

    def test_no_time_budget_means_no_time_stagnation(self) -> None:
        d = StagnationDetector(time_budget=None)
        # Even with elapsed time, should not trigger.
        assert not d.is_stagnant()

    def test_time_budget_mocked(self) -> None:
        d = StagnationDetector(time_budget=60.0)
        fake_start = 1000.0

        with patch("vetinari.orchestration.stagnation.time") as mock_time:
            mock_time.monotonic.return_value = fake_start
            d._start_time = fake_start

            mock_time.monotonic.return_value = fake_start + 30.0
            assert not d.is_stagnant()

            mock_time.monotonic.return_value = fake_start + 61.0
            assert d.is_stagnant()


# ---------------------------------------------------------------------------
# StagnationDetector — stagnation reasons
# ---------------------------------------------------------------------------


class TestStagnationReasons:
    """Verify human-readable stagnation reason strings."""

    def test_no_reasons_when_healthy(self) -> None:
        d = StagnationDetector()
        assert d.stagnation_reasons() == []

    def test_repeated_output_reason(self) -> None:
        d = StagnationDetector(max_repeated_outputs=2)
        d.record_output("stuck")
        d.record_output("stuck")
        reasons = d.stagnation_reasons()
        assert len(reasons) == 1
        assert "Repeated output" in reasons[0]

    def test_error_threshold_reason(self) -> None:
        d = StagnationDetector(error_threshold=1)
        d.record_error()
        reasons = d.stagnation_reasons()
        assert len(reasons) == 1
        assert "Error threshold" in reasons[0]

    def test_multiple_reasons(self) -> None:
        d = StagnationDetector(max_repeated_outputs=2, error_threshold=1)
        d.record_output("x")
        d.record_output("x")  # Triggers repeated-output stagnation
        d.record_error()  # Triggers error stagnation
        reasons = d.stagnation_reasons()
        assert len(reasons) == 2


# ---------------------------------------------------------------------------
# StagnationDetector — reset
# ---------------------------------------------------------------------------


class TestStagnationReset:
    """Verify reset clears all state."""

    def test_reset_clears_everything(self) -> None:
        d = StagnationDetector(max_repeated_outputs=2, error_threshold=2)
        d.record_output("a")
        d.record_output("a")
        d.record_error()
        d.record_error()
        assert d.is_stagnant()

        d.reset()
        assert not d.is_stagnant()
        assert d.error_count == 0
        assert d.repeat_count == 0
        assert d.output_history == []


# ---------------------------------------------------------------------------
# StagnationDetector — elapsed property
# ---------------------------------------------------------------------------


class TestStagnationElapsed:
    """Verify the elapsed property."""

    def test_elapsed_increases(self) -> None:
        d = StagnationDetector()
        t1 = d.elapsed
        # Simulate time passing by shifting start time backward
        d._start_time -= 1.0
        t2 = d.elapsed
        assert t2 > t1
