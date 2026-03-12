"""
Tests for vetinari.orchestration.circuit_breaker
==================================================

Covers CircuitBreaker state transitions, failure counting, reset timeout,
and StagnationDetector repeated-output / error / time-budget detection.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from vetinari.orchestration.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    StagnationDetector,
)


# ---------------------------------------------------------------------------
# CircuitBreaker — construction
# ---------------------------------------------------------------------------


class TestCircuitBreakerInit:
    """Verify construction and parameter validation."""

    def test_defaults(self) -> None:
        cb = CircuitBreaker()
        assert cb.state is CircuitState.CLOSED
        assert cb.failure_threshold == 5
        assert cb.reset_timeout == 60.0
        assert cb.max_retries == 3
        assert cb.failure_count == 0
        assert cb.success_count == 0

    def test_custom_parameters(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=10.0, max_retries=1)
        assert cb.failure_threshold == 2
        assert cb.reset_timeout == 10.0
        assert cb.max_retries == 1

    def test_invalid_failure_threshold(self) -> None:
        with pytest.raises(ValueError, match="failure_threshold"):
            CircuitBreaker(failure_threshold=0)

    def test_invalid_reset_timeout(self) -> None:
        with pytest.raises(ValueError, match="reset_timeout"):
            CircuitBreaker(reset_timeout=-1)

    def test_invalid_max_retries(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            CircuitBreaker(max_retries=-1)


# ---------------------------------------------------------------------------
# CircuitBreaker — state transitions
# ---------------------------------------------------------------------------


class TestCircuitBreakerTransitions:
    """Test CLOSED -> OPEN -> HALF_OPEN -> CLOSED lifecycle."""

    def test_stays_closed_below_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state is CircuitState.CLOSED
        assert cb.failure_count == 2

    def test_opens_at_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state is CircuitState.OPEN

    def test_open_blocks_requests(self) -> None:
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state is CircuitState.OPEN
        assert cb.allow_request() is False

    def test_half_open_after_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.05)
        cb.record_failure()
        assert cb.state is CircuitState.OPEN
        time.sleep(0.06)
        assert cb.state is CircuitState.HALF_OPEN
        assert cb.allow_request() is True

    def test_half_open_success_closes(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.05)
        cb.record_failure()
        time.sleep(0.06)
        assert cb.state is CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state is CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_half_open_failure_reopens(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.05)
        cb.record_failure()
        time.sleep(0.06)
        assert cb.state is CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state is CircuitState.OPEN

    def test_success_resets_failure_count(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0
        # One more failure should not open (counter was reset).
        cb.record_failure()
        assert cb.state is CircuitState.CLOSED

    def test_full_cycle(self) -> None:
        """CLOSED -> OPEN -> HALF_OPEN -> CLOSED round trip."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.05)
        # CLOSED -> OPEN
        cb.record_failure()
        cb.record_failure()
        assert cb.state is CircuitState.OPEN
        # OPEN -> HALF_OPEN
        time.sleep(0.06)
        assert cb.state is CircuitState.HALF_OPEN
        # HALF_OPEN -> CLOSED
        cb.record_success()
        assert cb.state is CircuitState.CLOSED


# ---------------------------------------------------------------------------
# CircuitBreaker — counters & error rate
# ---------------------------------------------------------------------------


class TestCircuitBreakerCounters:
    """Verify lifetime counters and error rate computation."""

    def test_total_counters(self) -> None:
        cb = CircuitBreaker(failure_threshold=10)
        cb.record_success()
        cb.record_success()
        cb.record_failure()
        cb.record_success()
        assert cb.total_successes == 3
        assert cb.total_failures == 1

    def test_error_rate_zero_calls(self) -> None:
        cb = CircuitBreaker()
        assert cb.error_rate == 0.0

    def test_error_rate_all_failures(self) -> None:
        cb = CircuitBreaker(failure_threshold=10)
        for _ in range(4):
            cb.record_failure()
        assert cb.error_rate == 1.0

    def test_error_rate_mixed(self) -> None:
        cb = CircuitBreaker(failure_threshold=10)
        cb.record_success()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert cb.error_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# CircuitBreaker — reset
# ---------------------------------------------------------------------------


class TestCircuitBreakerReset:
    """Verify manual reset."""

    def test_reset_clears_state(self) -> None:
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state is CircuitState.OPEN
        cb.reset()
        assert cb.state is CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0

    def test_reset_allows_requests_again(self) -> None:
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.allow_request() is False
        cb.reset()
        assert cb.allow_request() is True


# ---------------------------------------------------------------------------
# CircuitBreaker — timeout with mocked time
# ---------------------------------------------------------------------------


class TestCircuitBreakerMockedTime:
    """Use monkeypatched time.monotonic to avoid real sleeps."""

    def test_timeout_transition_mocked(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=30.0)
        fake_time = 1000.0

        with patch("vetinari.orchestration.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = fake_time
            cb.record_failure()
            assert cb._state is CircuitState.OPEN

            # Advance past timeout.
            mock_time.monotonic.return_value = fake_time + 31.0
            assert cb.state is CircuitState.HALF_OPEN

    def test_timeout_not_yet_elapsed(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=30.0)
        fake_time = 1000.0

        with patch("vetinari.orchestration.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = fake_time
            cb.record_failure()

            mock_time.monotonic.return_value = fake_time + 10.0
            assert cb.state is CircuitState.OPEN  # Still open.


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
        d = StagnationDetector(max_repeated_outputs=1)
        d.record_output("anything")
        assert d.is_stagnant()  # First output already meets threshold of 1.


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
        d = StagnationDetector(time_budget=0.05)
        time.sleep(0.06)
        assert d.is_stagnant()

    def test_no_time_budget_means_no_time_stagnation(self) -> None:
        d = StagnationDetector(time_budget=None)
        # Even with elapsed time, should not trigger.
        assert not d.is_stagnant()

    def test_time_budget_mocked(self) -> None:
        d = StagnationDetector(time_budget=60.0)
        fake_start = 1000.0

        with patch("vetinari.orchestration.circuit_breaker.time") as mock_time:
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
        d = StagnationDetector(max_repeated_outputs=1, error_threshold=1)
        d.record_output("x")
        d.record_error()
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
        time.sleep(0.02)
        t2 = d.elapsed
        assert t2 > t1
