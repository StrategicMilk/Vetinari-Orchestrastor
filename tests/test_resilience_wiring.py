"""Tests for vetinari.resilience.wiring — circuit breaker integration hooks."""

from __future__ import annotations

import pytest

from vetinari.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpen,
    reset_circuit_breaker_registry,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_circuit_breaker_registry()
    yield
    reset_circuit_breaker_registry()


class TestGetInferenceBreaker:
    """Tests for get_inference_breaker()."""

    def test_get_inference_breaker_returns_breaker(self):
        """Returns a CircuitBreaker instance for the given agent type.

        The name stored in the breaker is the normalised (uppercase) form so
        that it matches the canonical AgentType.value strings used in the
        _AGENT_BREAKER_CONFIGS lookup table.
        """
        from vetinari.resilience.wiring import get_inference_breaker

        breaker = get_inference_breaker("foreman")

        assert isinstance(breaker, CircuitBreaker)
        # Normalisation converts to uppercase to align with AgentType.value keys.
        assert breaker.name == "FOREMAN"

    def test_get_inference_breaker_case_insensitive(self):
        """Lower-case and upper-case inputs return the exact same instance."""
        from vetinari.resilience.wiring import get_inference_breaker

        lower = get_inference_breaker("worker")
        upper = get_inference_breaker("WORKER")

        assert lower is upper

    def test_get_inference_breaker_same_instance_on_repeat(self):
        """Returns the same instance on repeated calls for the same agent type."""
        from vetinari.resilience.wiring import get_inference_breaker

        b1 = get_inference_breaker("worker")
        b2 = get_inference_breaker("worker")

        assert b1 is b2


class TestCallWithBreaker:
    """Tests for call_with_breaker()."""

    def test_call_with_breaker_success(self):
        """Returns the callable's return value when the circuit is closed."""
        from vetinari.resilience.wiring import call_with_breaker

        result = call_with_breaker("inspector", lambda: 42)

        assert result == 42

    def test_call_with_breaker_propagates_return_value(self):
        """Passes args and kwargs through to the wrapped callable."""
        from vetinari.resilience.wiring import call_with_breaker

        def add(a: int, b: int) -> int:
            return a + b

        result = call_with_breaker("foreman", add, 3, b=7)

        assert result == 10

    def test_call_with_breaker_failure_records(self):
        """Records a failure on the breaker when the callable raises."""
        from vetinari.resilience.wiring import call_with_breaker, get_inference_breaker

        def boom() -> None:
            raise ValueError("inference failed")

        with pytest.raises(ValueError, match="inference failed"):
            call_with_breaker("worker", boom)

        breaker = get_inference_breaker("worker")
        assert breaker.stats.total_failures == 1

    def test_call_with_breaker_open_circuit_raises(self):
        """Raises CircuitBreakerOpen when the breaker has been tripped."""
        from vetinari.resilience.wiring import call_with_breaker, get_inference_breaker

        # Manually trip the breaker
        breaker = get_inference_breaker("planner")
        breaker.trip()

        with pytest.raises(CircuitBreakerOpen):
            call_with_breaker("planner", lambda: "should not run")


class TestCheckBreakerHealth:
    """Tests for check_breaker_health()."""

    def test_check_breaker_health_returns_dict(self):
        """Returns a dict with the expected health keys for the given agent type."""
        from vetinari.resilience.wiring import check_breaker_health

        health = check_breaker_health("foreman")

        assert isinstance(health, dict)
        assert "name" in health
        assert "state" in health
        assert "stats" in health

    def test_check_breaker_health_reflects_name(self):
        """The returned dict's name field matches the requested agent type."""
        from vetinari.resilience.wiring import check_breaker_health

        health = check_breaker_health("inspector")

        assert health["name"] == "inspector"


class TestGetAllBreakerHealth:
    """Tests for get_all_breaker_health()."""

    def test_get_all_breaker_health_returns_dict(self):
        """Returns a dict (possibly empty when no breakers accessed yet)."""
        from vetinari.resilience.wiring import get_all_breaker_health

        health = get_all_breaker_health()

        assert isinstance(health, dict)

    def test_get_all_breaker_health_includes_accessed_breakers(self):
        """Includes all breakers that have been accessed during the test."""
        from vetinari.resilience.wiring import call_with_breaker, get_all_breaker_health

        # Touch two breakers so they appear in the summary
        call_with_breaker("foreman", lambda: None)
        call_with_breaker("worker", lambda: None)

        health = get_all_breaker_health()

        # Keys are normalised to uppercase to match AgentType.value canonical form.
        assert "FOREMAN" in health
        assert "WORKER" in health


class TestWireResilienceSubsystem:
    """Tests for wire_resilience_subsystem()."""

    def test_wire_resilience_subsystem_runs_without_error(self, caplog):
        """wire_resilience_subsystem() logs a ready message when it completes."""
        import logging

        from vetinari.resilience.wiring import wire_resilience_subsystem

        with caplog.at_level(logging.INFO, logger="vetinari.resilience.wiring"):
            wire_resilience_subsystem()
        assert any("ready" in r.message for r in caplog.records)

    def test_wire_resilience_subsystem_is_idempotent(self, caplog):
        """Calling wire_resilience_subsystem() twice emits two log records (idempotent)."""
        import logging

        from vetinari.resilience.wiring import wire_resilience_subsystem

        with caplog.at_level(logging.INFO, logger="vetinari.resilience.wiring"):
            wire_resilience_subsystem()
            wire_resilience_subsystem()
        assert len(caplog.records) == 2
