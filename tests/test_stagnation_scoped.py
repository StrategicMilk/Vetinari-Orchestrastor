"""Tests for scoped stagnation detection (item 7.12).

Covers the new detect_scoped(), get_stagnant_scopes(), reset_scope(),
and per_scope_state additions to StagnationDetector.  Backward
compatibility with the global API is verified in the same suite.
"""

from __future__ import annotations

import pytest

from vetinari.orchestration.stagnation import StagnationDetector, _ScopeState


class TestScopeStateDataclass:
    """Sanity-check the internal _ScopeState helper."""

    def test_defaults(self) -> None:
        """_ScopeState initialises with zero counts and empty history."""
        state = _ScopeState()
        assert state.last_output is None
        assert state.repeat_count == 0
        assert state.error_count == 0
        assert list(state.output_history) == []


class TestGlobalBackwardCompat:
    """Existing global API must still work unchanged."""

    def test_record_output_global(self) -> None:
        """record_output increments repeat_count on duplicate outputs."""
        d = StagnationDetector(max_repeated_outputs=2, error_threshold=5)
        d.record_output("x")
        d.record_output("x")
        assert d.is_stagnant() is True

    def test_record_error_global(self) -> None:
        """record_error increments the global error count."""
        d = StagnationDetector(max_repeated_outputs=3, error_threshold=2)
        d.record_error()
        d.record_error()
        assert d.is_stagnant() is True

    def test_reset_clears_global_state(self) -> None:
        """reset() clears global counters without touching scoped state."""
        d = StagnationDetector(max_repeated_outputs=2, error_threshold=5)
        d.record_output("a")
        d.record_output("a")
        assert d.is_stagnant() is True
        d.reset()
        assert d.is_stagnant() is False

    def test_stagnation_reasons_not_empty_when_stagnant(self) -> None:
        """stagnation_reasons() returns non-empty list when stagnant."""
        d = StagnationDetector(max_repeated_outputs=2, error_threshold=5)
        d.record_output("y")
        d.record_output("y")
        reasons = d.stagnation_reasons()
        assert len(reasons) >= 1
        assert any("Repeated output" in r for r in reasons)


class TestDetectScoped:
    """detect_scoped() tracks each scope independently."""

    def test_scope_not_stagnant_initially(self) -> None:
        """A fresh scope is not stagnant after a single output."""
        d = StagnationDetector(max_repeated_outputs=3, error_threshold=5)
        result = d.detect_scoped("dept-A", "output1")
        assert result is False

    def test_scope_stagnant_on_repeat_threshold(self) -> None:
        """detect_scoped returns True when repeat count reaches the threshold."""
        d = StagnationDetector(max_repeated_outputs=3, error_threshold=10)
        d.detect_scoped("dept-A", "same")
        d.detect_scoped("dept-A", "same")
        result = d.detect_scoped("dept-A", "same")
        assert result is True

    def test_different_output_resets_scope_repeat(self) -> None:
        """A new output string resets the consecutive repeat counter for the scope."""
        d = StagnationDetector(max_repeated_outputs=3, error_threshold=10)
        d.detect_scoped("dept-A", "same")
        d.detect_scoped("dept-A", "same")
        # New output resets the streak
        d.detect_scoped("dept-A", "different")
        result = d.detect_scoped("dept-A", "same")
        assert result is False

    def test_scopes_are_independent(self) -> None:
        """Stagnation in one scope does not affect another scope."""
        d = StagnationDetector(max_repeated_outputs=3, error_threshold=10)
        # Saturate dept-A
        d.detect_scoped("dept-A", "stuck")
        d.detect_scoped("dept-A", "stuck")
        d.detect_scoped("dept-A", "stuck")
        # dept-B should still be clean
        result_b = d.detect_scoped("dept-B", "fresh")
        assert result_b is False

    def test_scope_stagnant_on_error_threshold(self) -> None:
        """detect_scoped returns True when scope error count reaches threshold."""
        d = StagnationDetector(max_repeated_outputs=10, error_threshold=3)
        d.detect_scoped("dept-X", "out", error=True)
        d.detect_scoped("dept-X", "out2", error=True)
        result = d.detect_scoped("dept-X", "out3", error=True)
        assert result is True

    def test_error_flag_increments_scope_error_count(self) -> None:
        """Passing error=True increments the scope error_count."""
        d = StagnationDetector(max_repeated_outputs=10, error_threshold=5)
        d.detect_scoped("dept-Y", "out", error=True)
        d.detect_scoped("dept-Y", "out2", error=True)
        state = d.per_scope_state["dept-Y"]
        assert state.error_count == 2

    def test_scope_state_populated(self) -> None:
        """detect_scoped creates an entry in per_scope_state."""
        d = StagnationDetector(max_repeated_outputs=3, error_threshold=5)
        assert "dept-Z" not in d.per_scope_state
        d.detect_scoped("dept-Z", "hello")
        assert "dept-Z" in d.per_scope_state

    def test_output_history_recorded_per_scope(self) -> None:
        """Each output is appended to the scope's history."""
        d = StagnationDetector(max_repeated_outputs=5, error_threshold=10)
        d.detect_scoped("dept-Q", "a")
        d.detect_scoped("dept-Q", "b")
        d.detect_scoped("dept-Q", "c")
        history = list(d.per_scope_state["dept-Q"].output_history)
        assert history == ["a", "b", "c"]


class TestGetStagnantScopes:
    """get_stagnant_scopes() returns names of all stagnant scopes."""

    def test_empty_when_no_scopes(self) -> None:
        """Returns empty list when no scopes have been seen."""
        d = StagnationDetector(max_repeated_outputs=3, error_threshold=5)
        assert d.get_stagnant_scopes() == []

    def test_returns_stagnant_scope(self) -> None:
        """Stagnant scope appears in the returned list."""
        d = StagnationDetector(max_repeated_outputs=2, error_threshold=5)
        d.detect_scoped("dept-A", "repeat")
        d.detect_scoped("dept-A", "repeat")
        stagnant = d.get_stagnant_scopes()
        assert "dept-A" in stagnant

    def test_healthy_scope_not_in_list(self) -> None:
        """A scope below all thresholds is not returned."""
        d = StagnationDetector(max_repeated_outputs=5, error_threshold=10)
        d.detect_scoped("dept-B", "ok")
        assert "dept-B" not in d.get_stagnant_scopes()

    def test_multiple_stagnant_scopes(self) -> None:
        """Multiple stagnant scopes all appear in the result."""
        d = StagnationDetector(max_repeated_outputs=2, error_threshold=5)
        for scope in ("dept-A", "dept-B", "dept-C"):
            d.detect_scoped(scope, "stuck")
            d.detect_scoped(scope, "stuck")
        stagnant = d.get_stagnant_scopes()
        assert set(stagnant) == {"dept-A", "dept-B", "dept-C"}

    def test_sorted_order(self) -> None:
        """get_stagnant_scopes returns scopes in sorted order."""
        d = StagnationDetector(max_repeated_outputs=2, error_threshold=5)
        for scope in ("zzz", "aaa", "mmm"):
            d.detect_scoped(scope, "x")
            d.detect_scoped(scope, "x")
        stagnant = d.get_stagnant_scopes()
        assert stagnant == sorted(stagnant)


class TestResetScope:
    """reset_scope() clears stagnation state for a single scope."""

    def test_reset_clears_stagnant_scope(self) -> None:
        """After reset_scope a previously stagnant scope is no longer stagnant."""
        d = StagnationDetector(max_repeated_outputs=2, error_threshold=5)
        d.detect_scoped("dept-A", "stuck")
        d.detect_scoped("dept-A", "stuck")
        assert "dept-A" in d.get_stagnant_scopes()

        d.reset_scope("dept-A")
        assert "dept-A" not in d.get_stagnant_scopes()

    def test_reset_unknown_scope_is_noop(self) -> None:
        """reset_scope on a scope that was never seen does not raise."""
        d = StagnationDetector(max_repeated_outputs=3, error_threshold=5)
        d.reset_scope("nonexistent")  # must not raise
        assert d.get_stagnant_scopes() == []

    def test_reset_scope_does_not_affect_other_scopes(self) -> None:
        """Resetting one scope does not clear another scope's state."""
        d = StagnationDetector(max_repeated_outputs=2, error_threshold=5)
        d.detect_scoped("dept-A", "stuck")
        d.detect_scoped("dept-A", "stuck")
        d.detect_scoped("dept-B", "stuck")
        d.detect_scoped("dept-B", "stuck")

        d.reset_scope("dept-A")

        assert "dept-A" not in d.get_stagnant_scopes()
        assert "dept-B" in d.get_stagnant_scopes()


class TestTimeBudgetScoped:
    """Global time_budget also triggers scoped stagnation."""

    def test_time_budget_marks_scope_stagnant(self) -> None:
        """A scope is stagnant when the global time_budget is exceeded."""
        from unittest.mock import patch

        fake_start = 1000.0
        with patch("vetinari.orchestration.stagnation.time") as mock_time:
            mock_time.monotonic.return_value = fake_start
            d = StagnationDetector(
                max_repeated_outputs=100,
                error_threshold=100,
                time_budget=5.0,
            )
            # Advance clock past the 5-second budget
            mock_time.monotonic.return_value = fake_start + 10.0
            d.detect_scoped("dept-time", "any-output")
            stagnant = d.get_stagnant_scopes()
            assert "dept-time" in stagnant
