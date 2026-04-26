"""Tests for vetinari.system.remediation — tiered remediation engine."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from vetinari.system.remediation import (
    FailureMode,
    RemediationAction,
    RemediationEngine,
    RemediationPlan,
    RemediationResult,
    RemediationTier,
    get_remediation_engine,
)

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def engine() -> RemediationEngine:
    """Fresh RemediationEngine for each test."""
    return RemediationEngine()


# ── FailureMode enum ──────────────────────────────────────────────────


def test_failure_mode_values_are_stable() -> None:
    """Enum values must not drift — callers serialize them to disk."""
    assert FailureMode.OOM.value == "oom"
    assert FailureMode.HANG.value == "hang"
    assert FailureMode.QUALITY_DROP.value == "quality_drop"
    assert FailureMode.DISK_FULL.value == "disk_full"
    assert FailureMode.THERMAL.value == "thermal"


# ── RemediationTier enum ──────────────────────────────────────────────


def test_remediation_tier_values_are_stable() -> None:
    """Enum values must not drift — callers serialize them to disk."""
    assert RemediationTier.AUTO_FIX.value == "auto_fix"
    assert RemediationTier.ALERT.value == "alert"
    assert RemediationTier.PAUSE.value == "pause"
    assert RemediationTier.SHUTDOWN.value == "shutdown"


# ── diagnose ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "failure_mode",
    [
        FailureMode.OOM,
        FailureMode.HANG,
        FailureMode.QUALITY_DROP,
        FailureMode.DISK_FULL,
        FailureMode.THERMAL,
    ],
)
def test_diagnose_returns_plan_for_every_mode(engine: RemediationEngine, failure_mode: FailureMode) -> None:
    """Every FailureMode must produce a non-empty RemediationPlan."""
    plan = engine.diagnose(failure_mode)
    assert isinstance(plan, RemediationPlan)
    assert plan.failure_mode is failure_mode
    assert len(plan.actions) > 0
    assert plan.diagnosis


def test_diagnose_plan_has_max_tier(engine: RemediationEngine) -> None:
    """max_tier on the plan must equal the highest tier present in actions."""
    plan = engine.diagnose(FailureMode.OOM)
    tier_order = [RemediationTier.AUTO_FIX, RemediationTier.ALERT, RemediationTier.PAUSE, RemediationTier.SHUTDOWN]
    expected_max = max(plan.actions, key=lambda a: tier_order.index(a.tier)).tier
    assert plan.max_tier is expected_max


def test_diagnose_with_context_does_not_raise(engine: RemediationEngine) -> None:
    """Passing a context dict must not raise and must produce a valid plan."""
    plan = engine.diagnose(FailureMode.HANG, context={"project_id": "p-001"})
    assert plan.failure_mode is FailureMode.HANG


def test_diagnose_repr_is_readable(engine: RemediationEngine) -> None:
    """RemediationPlan repr must include key identifying fields."""
    plan = engine.diagnose(FailureMode.DISK_FULL)
    r = repr(plan)
    assert "DISK_FULL" in r or "disk_full" in r
    assert "actions=" in r


# ── execute_remediation ───────────────────────────────────────────────


def test_execute_succeeds_on_first_successful_action(engine: RemediationEngine) -> None:
    """Execution must stop at the first successful action and report success."""
    plan = RemediationPlan(
        failure_mode=FailureMode.HANG,
        diagnosis="test",
        actions=[
            RemediationAction("Action A", RemediationTier.AUTO_FIX, lambda: True),
            RemediationAction("Action B", RemediationTier.ALERT, lambda: True),
        ],
        max_tier=RemediationTier.ALERT,
    )
    result = engine.execute_remediation(plan)
    assert result.success is True
    assert "Action A" in result.actions_taken
    # Second action should not have been attempted.
    assert "Action B" not in result.actions_taken


def test_execute_escalates_past_failing_action(engine: RemediationEngine) -> None:
    """When the first action fails, execution must continue to the next tier."""
    plan = RemediationPlan(
        failure_mode=FailureMode.HANG,
        diagnosis="test",
        actions=[
            RemediationAction("Fail action", RemediationTier.AUTO_FIX, lambda: False),
            RemediationAction("Success action", RemediationTier.ALERT, lambda: True),
        ],
        max_tier=RemediationTier.ALERT,
    )
    result = engine.execute_remediation(plan)
    assert result.success is True
    assert "Fail action" in result.actions_taken
    assert "Success action" in result.actions_taken
    assert result.tier_reached is RemediationTier.ALERT


def test_execute_all_fail_returns_failure(engine: RemediationEngine) -> None:
    """When all actions fail, the result must report failure with error set."""
    plan = RemediationPlan(
        failure_mode=FailureMode.OOM,
        diagnosis="test",
        actions=[
            RemediationAction("Bad A", RemediationTier.AUTO_FIX, lambda: False),
            RemediationAction("Bad B", RemediationTier.PAUSE, lambda: False),
        ],
        max_tier=RemediationTier.PAUSE,
    )
    result = engine.execute_remediation(plan)
    assert result.success is False
    assert result.error is not None
    assert len(result.actions_taken) == 2


def test_execute_action_exception_treated_as_failure(engine: RemediationEngine) -> None:
    """An exception inside an action callable must be caught and counted as failure."""

    def _boom() -> bool:
        raise RuntimeError("kaboom")

    plan = RemediationPlan(
        failure_mode=FailureMode.THERMAL,
        diagnosis="test",
        actions=[
            RemediationAction("Exploding action", RemediationTier.AUTO_FIX, _boom),
            RemediationAction("Recovery action", RemediationTier.ALERT, lambda: True),
        ],
        max_tier=RemediationTier.ALERT,
    )
    result = engine.execute_remediation(plan)
    # The second action should rescue the situation.
    assert result.success is True


def test_execute_records_result_in_history(engine: RemediationEngine) -> None:
    """Every execution must append a RemediationResult to the history."""
    plan = engine.diagnose(FailureMode.QUALITY_DROP)
    assert len(engine.get_history()) == 0
    engine.execute_remediation(plan)
    assert len(engine.get_history()) == 1


def test_execute_action_with_none_fn_skips_without_error(engine: RemediationEngine) -> None:
    """An action with action_fn=None must be recorded but not crash execution."""
    plan = RemediationPlan(
        failure_mode=FailureMode.DISK_FULL,
        diagnosis="test",
        actions=[
            RemediationAction("Informational only", RemediationTier.AUTO_FIX, None),
            RemediationAction("Actual fix", RemediationTier.ALERT, lambda: True),
        ],
        max_tier=RemediationTier.ALERT,
    )
    result = engine.execute_remediation(plan)
    assert result.success is True


# ── circuit breaker interaction ───────────────────────────────────────


def test_circuit_breaker_open_blocks_execution(engine: RemediationEngine) -> None:
    """When the breaker is open, execute_remediation must return without running actions."""
    # Trip the breaker by exhausting its failure threshold.
    engine._breaker.trip()

    plan = engine.diagnose(FailureMode.OOM)
    result = engine.execute_remediation(plan)
    assert result.success is False
    assert result.error is not None
    assert "circuit breaker" in result.error.lower()
    # No actions should have been attempted.
    assert result.actions_taken == []


# ── get_history ───────────────────────────────────────────────────────


def test_get_history_returns_snapshot(engine: RemediationEngine) -> None:
    """get_history must return a list (not the live deque) so callers can't mutate it."""
    plan = engine.diagnose(FailureMode.HANG)
    engine.execute_remediation(plan)
    history = engine.get_history()
    assert isinstance(history, list)
    assert len(history) == 1
    assert isinstance(history[0], RemediationResult)


# ── get_stats ─────────────────────────────────────────────────────────


def test_get_stats_empty_engine(engine: RemediationEngine) -> None:
    """Stats on a fresh engine must return zero counts and no dominant mode."""
    stats = engine.get_stats()
    assert stats["total"] == 0
    assert stats["success_rate"] == 0.0
    assert stats["most_common_failure_mode"] is None
    assert stats["breaker_state"] == "closed"


def test_get_stats_after_successes(engine: RemediationEngine) -> None:
    """Stats must accurately reflect success rate and most common failure mode."""
    for _ in range(3):
        plan = engine.diagnose(FailureMode.OOM)
        engine.execute_remediation(plan)
    plan = engine.diagnose(FailureMode.HANG)
    engine.execute_remediation(plan)

    stats = engine.get_stats()
    assert stats["total"] == 4
    assert stats["most_common_failure_mode"] == "oom"
    assert 0.0 <= stats["success_rate"] <= 1.0


# ── singleton ─────────────────────────────────────────────────────────


def test_get_remediation_engine_returns_same_instance() -> None:
    """The singleton factory must always return the same RemediationEngine object."""
    a = get_remediation_engine()
    b = get_remediation_engine()
    assert a is b


def test_get_remediation_engine_is_remediation_engine() -> None:
    """The singleton must be an instance of RemediationEngine."""
    assert isinstance(get_remediation_engine(), RemediationEngine)


# ── system package re-export ──────────────────────────────────────────


def test_system_package_re_exports_public_api() -> None:
    """vetinari.system must re-export all public symbols from remediation."""
    from vetinari import system

    assert hasattr(system, "FailureMode")
    assert hasattr(system, "RemediationEngine")
    assert hasattr(system, "RemediationTier")
    assert hasattr(system, "get_remediation_engine")
