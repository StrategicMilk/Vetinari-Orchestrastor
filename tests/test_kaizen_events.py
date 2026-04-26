"""Tests for kaizen event emission — ImprovementLog events on EventBus."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from vetinari.events import (
    KaizenImprovementActive,
    KaizenImprovementConfirmed,
    KaizenImprovementProposed,
    KaizenImprovementReverted,
    get_event_bus,
    reset_event_bus,
)
from vetinari.kaizen.improvement_log import ImprovementLog


@pytest.fixture(autouse=True)
def _clean_event_bus():
    """Reset the EventBus singleton before and after each test."""
    reset_event_bus()
    yield
    reset_event_bus()


@pytest.fixture
def improvement_log(tmp_path):
    """Create an ImprovementLog backed by a temporary SQLite database."""
    return ImprovementLog(tmp_path / "events_test.db")


class TestKaizenEvents:
    """Test that ImprovementLog emits events on lifecycle transitions."""

    def test_propose_emits_event(self, improvement_log):
        """Proposing an improvement emits KaizenImprovementProposed."""
        bus = get_event_bus()
        received = []
        bus.subscribe(KaizenImprovementProposed, received.append)

        improvement_log.propose(
            hypothesis="Test hypothesis",
            metric="quality",
            baseline=0.5,
            target=0.7,
            applied_by="test",
            rollback_plan="revert",
        )

        assert len(received) == 1
        assert received[0].hypothesis == "Test hypothesis"
        assert received[0].metric == "quality"
        assert received[0].applied_by == "test"

    def test_evaluate_confirmed_emits_event(self, improvement_log):
        """Confirming an improvement emits KaizenImprovementConfirmed."""
        bus = get_event_bus()
        received = []
        bus.subscribe(KaizenImprovementConfirmed, received.append)

        imp_id = improvement_log.propose(
            hypothesis="H1",
            metric="quality",
            baseline=0.5,
            target=0.7,
            applied_by="prompt_evolver",
            rollback_plan="revert",
        )
        improvement_log.activate(imp_id)
        improvement_log.observe(imp_id, 0.75, 10)
        improvement_log.evaluate(imp_id)

        assert len(received) == 1
        assert received[0].improvement_id == imp_id
        assert received[0].baseline_value == 0.5
        assert received[0].actual_value >= 0.7

    def test_revert_emits_event(self, improvement_log):
        """Reverting an improvement emits KaizenImprovementReverted."""
        bus = get_event_bus()
        received = []
        bus.subscribe(KaizenImprovementReverted, received.append)

        imp_id = improvement_log.propose(
            hypothesis="H1",
            metric="quality",
            baseline=0.5,
            target=0.7,
            applied_by="test",
            rollback_plan="revert",
        )
        improvement_log.activate(imp_id)
        improvement_log.revert(imp_id)

        assert len(received) == 1
        assert received[0].improvement_id == imp_id
        assert received[0].reason == "regression_detected"

    def test_activate_emits_active_event(self, improvement_log):
        """Activating an improvement emits KaizenImprovementActive exactly once."""
        bus = get_event_bus()
        received = []
        bus.subscribe(KaizenImprovementActive, received.append)

        imp_id = improvement_log.propose(
            hypothesis="Test activate event",
            metric="quality",
            baseline=0.5,
            target=0.7,
            applied_by="test_activator",
            rollback_plan="revert",
        )
        improvement_log.activate(imp_id)

        assert len(received) == 1, (
            f"Expected exactly 1 KaizenImprovementActive event, got {len(received)}"
        )
        assert received[0].improvement_id == imp_id
        assert received[0].metric == "quality"
        assert received[0].applied_by == "test_activator"

    def test_activate_emits_active_event_exactly_once_on_repeated_check(self, improvement_log):
        """Calling activate once emits exactly one KaizenImprovementActive (no double-fire)."""
        bus = get_event_bus()
        received = []
        bus.subscribe(KaizenImprovementActive, received.append)

        imp_id = improvement_log.propose(
            hypothesis="Double-fire guard",
            metric="quality",
            baseline=0.4,
            target=0.8,
            applied_by="test",
            rollback_plan="revert",
        )
        improvement_log.activate(imp_id)

        # Simulate a caller checking the record immediately after activation —
        # no second event should appear.
        improvement_log.get_improvement(imp_id)

        assert len(received) == 1, (
            "activate() must emit KaizenImprovementActive exactly once; "
            f"got {len(received)} events"
        )

    def test_emit_failure_propagates_from_propose(self, improvement_log):
        """EventBus publish failures during propose() must not be silently swallowed.

        The emit functions log AND re-raise on failure (see improvement_events.py).
        This test proves the exception reaches the caller of propose() so that a
        broken EventBus is immediately visible rather than leaving the improvement
        in PROPOSED state with no subscribers aware of it.
        """
        with patch(
            "vetinari.kaizen.improvement_events.emit_proposed",
            side_effect=RuntimeError("EventBus unavailable"),
        ):
            with pytest.raises(RuntimeError, match="EventBus unavailable"):
                improvement_log.propose(
                    hypothesis="Failure propagation check",
                    metric="quality",
                    baseline=0.5,
                    target=0.7,
                    applied_by="test",
                    rollback_plan="revert",
                )

    def test_emit_failure_propagates_from_activate(self, improvement_log):
        """EventBus publish failures during activate() must not be silently swallowed."""
        imp_id = improvement_log.propose(
            hypothesis="Activate emit-failure check",
            metric="quality",
            baseline=0.5,
            target=0.7,
            applied_by="test",
            rollback_plan="revert",
        )
        with patch(
            "vetinari.kaizen.improvement_events.emit_active",
            side_effect=RuntimeError("EventBus down"),
        ):
            with pytest.raises(RuntimeError, match="EventBus down"):
                improvement_log.activate(imp_id)
