"""Tests for scoped Andon signals (item 7.13).

Covers the new scope field on AndonSignal, pause_scope(), resume_scope(),
get_paused_scopes(), acknowledge_scope(), and is_scope_paused() additions
to AndonSystem.  Backward compatibility with the global API is verified.
"""

from __future__ import annotations

import pytest

from vetinari.workflow.andon import AndonSignal, AndonSystem, reset_andon_system


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    """Reset the global AndonSystem singleton before and after each test."""
    reset_andon_system()
    yield
    reset_andon_system()


class TestAndonSignalScopeField:
    """AndonSignal dataclass must carry the new scope field."""

    def test_scope_defaults_to_none(self) -> None:
        """AndonSignal.scope defaults to None for backward compatibility."""
        sig = AndonSignal(source="test", severity="warning", message="msg")
        assert sig.scope is None

    def test_scope_can_be_set(self) -> None:
        """AndonSignal.scope can be set to a string."""
        sig = AndonSignal(source="test", severity="critical", message="msg", scope="dept-A")
        assert sig.scope == "dept-A"

    def test_repr_includes_scope(self) -> None:
        """__repr__ includes the scope field."""
        sig = AndonSignal(source="src", severity="warning", message="m", scope="dept-X")
        assert "dept-X" in repr(sig)

    def test_existing_fields_unchanged(self) -> None:
        """Adding scope field did not break existing AndonSignal fields."""
        sig = AndonSignal(
            source="gate-1",
            severity="emergency",
            message="fire",
            affected_tasks=["t1", "t2"],
        )
        assert sig.source == "gate-1"
        assert sig.severity == "emergency"
        assert sig.message == "fire"
        assert sig.affected_tasks == ["t1", "t2"]
        assert sig.acknowledged is False


class TestGlobalBackwardCompat:
    """Legacy global Andon API must work unchanged."""

    def test_raise_signal_critical_pauses_globally(self) -> None:
        """raise_signal with critical severity sets global is_paused."""
        system = AndonSystem()
        system.raise_signal(source="test", severity="critical", message="halt")
        assert system.is_paused() is True

    def test_raise_signal_warning_no_global_pause(self) -> None:
        """raise_signal with warning severity does not pause globally."""
        system = AndonSystem()
        system.raise_signal(source="test", severity="warning", message="degraded")
        assert system.is_paused() is False

    def test_acknowledge_resumes_global(self) -> None:
        """Acknowledging the critical signal resumes global execution."""
        system = AndonSystem()
        system.raise_signal(source="test", severity="critical", message="halt")
        assert system.is_paused() is True
        system.acknowledge(0)
        assert system.is_paused() is False

    def test_get_active_signals_returns_unacked(self) -> None:
        """get_active_signals returns only unacknowledged signals."""
        system = AndonSystem()
        system.raise_signal(source="s", severity="warning", message="w")
        system.raise_signal(source="s", severity="critical", message="c")
        system.acknowledge(0)
        active = system.get_active_signals()
        assert len(active) == 1
        assert active[0].severity == "critical"


class TestPauseScope:
    """pause_scope() pauses a named scope independently of global state."""

    def test_scope_paused_after_pause_scope(self) -> None:
        """is_scope_paused returns True after pause_scope is called."""
        system = AndonSystem()
        sig = AndonSignal(source="gate-A", severity="critical", message="overload")
        system.pause_scope("dept-planning", sig)
        assert system.is_scope_paused("dept-planning") is True

    def test_global_not_paused_by_scope_pause(self) -> None:
        """pause_scope does not affect the global is_paused state."""
        system = AndonSystem()
        sig = AndonSignal(source="gate-B", severity="critical", message="overload")
        system.pause_scope("dept-build", sig)
        assert system.is_paused() is False

    def test_signal_scope_field_set_by_pause_scope(self) -> None:
        """pause_scope sets the signal's scope field to the given scope name."""
        system = AndonSystem()
        sig = AndonSignal(source="gate-C", severity="warning", message="slow")
        system.pause_scope("dept-quality", sig)
        assert sig.scope == "dept-quality"

    def test_signal_added_to_all_signals(self) -> None:
        """The signal passed to pause_scope is recorded in get_all_signals."""
        system = AndonSystem()
        sig = AndonSignal(source="gate-D", severity="critical", message="err")
        system.pause_scope("dept-X", sig)
        assert sig in system.get_all_signals()

    def test_callback_invoked_on_scope_pause(self) -> None:
        """Registered callbacks are fired when a scope is paused."""
        system = AndonSystem()
        received: list[AndonSignal] = []
        system.register_callback(received.append)

        sig = AndonSignal(source="cb-test", severity="critical", message="cb")
        system.pause_scope("dept-cb", sig)
        assert len(received) == 1
        assert received[0] is sig

    def test_multiple_scopes_independent(self) -> None:
        """Multiple scopes can be paused independently."""
        system = AndonSystem()
        sig_a = AndonSignal(source="a", severity="critical", message="a")
        sig_b = AndonSignal(source="b", severity="critical", message="b")
        system.pause_scope("scope-A", sig_a)
        system.pause_scope("scope-B", sig_b)
        assert system.is_scope_paused("scope-A") is True
        assert system.is_scope_paused("scope-B") is True


class TestResumeScope:
    """resume_scope() clears the paused state for a single scope."""

    def test_resume_scope_unpauses_scope(self) -> None:
        """is_scope_paused returns False after resume_scope."""
        system = AndonSystem()
        sig = AndonSignal(source="x", severity="critical", message="x")
        system.pause_scope("dept-A", sig)
        result = system.resume_scope("dept-A")
        assert result is True
        assert system.is_scope_paused("dept-A") is False

    def test_resume_scope_acknowledges_signal(self) -> None:
        """resume_scope marks the causing signal as acknowledged."""
        system = AndonSystem()
        sig = AndonSignal(source="x", severity="critical", message="x")
        system.pause_scope("dept-A", sig)
        system.resume_scope("dept-A")
        assert sig.acknowledged is True

    def test_resume_scope_returns_false_if_not_paused(self) -> None:
        """resume_scope returns False when the scope was not paused."""
        system = AndonSystem()
        result = system.resume_scope("nonexistent-scope")
        assert result is False

    def test_resume_one_scope_leaves_other_paused(self) -> None:
        """Resuming one scope does not affect other paused scopes."""
        system = AndonSystem()
        sig_a = AndonSignal(source="a", severity="critical", message="a")
        sig_b = AndonSignal(source="b", severity="critical", message="b")
        system.pause_scope("scope-A", sig_a)
        system.pause_scope("scope-B", sig_b)
        system.resume_scope("scope-A")
        assert system.is_scope_paused("scope-A") is False
        assert system.is_scope_paused("scope-B") is True


class TestGetPausedScopes:
    """get_paused_scopes() returns names of all currently paused scopes."""

    def test_empty_when_no_scopes_paused(self) -> None:
        """Returns empty list when nothing is paused."""
        system = AndonSystem()
        assert system.get_paused_scopes() == []

    def test_returns_paused_scope_name(self) -> None:
        """Paused scope appears in get_paused_scopes."""
        system = AndonSystem()
        sig = AndonSignal(source="s", severity="critical", message="m")
        system.pause_scope("dept-paused", sig)
        assert "dept-paused" in system.get_paused_scopes()

    def test_resumed_scope_removed_from_list(self) -> None:
        """After resume_scope the scope no longer appears in get_paused_scopes."""
        system = AndonSystem()
        sig = AndonSignal(source="s", severity="critical", message="m")
        system.pause_scope("dept-resume", sig)
        system.resume_scope("dept-resume")
        assert "dept-resume" not in system.get_paused_scopes()

    def test_sorted_order(self) -> None:
        """get_paused_scopes returns scope names in sorted order."""
        system = AndonSystem()
        for name in ("zzz", "aaa", "mmm"):
            sig = AndonSignal(source="s", severity="critical", message="m")
            system.pause_scope(name, sig)
        scopes = system.get_paused_scopes()
        assert scopes == sorted(scopes)


class TestAcknowledgeScope:
    """acknowledge_scope() provides hierarchical acknowledgment."""

    def test_acknowledge_scope_resumes_child(self) -> None:
        """acknowledge_scope resumes the named scope."""
        system = AndonSystem()
        sig = AndonSignal(source="s", severity="critical", message="m")
        system.pause_scope("child", sig)
        result = system.acknowledge_scope("child")
        assert result is True
        assert system.is_scope_paused("child") is False

    def test_acknowledge_scope_returns_false_if_not_paused(self) -> None:
        """acknowledge_scope returns False when the scope is not paused."""
        system = AndonSystem()
        result = system.acknowledge_scope("never-paused")
        assert result is False

    def test_acknowledge_scope_with_parent_resumes_both(self) -> None:
        """When parent_scope is provided and paused, both child and parent are resumed."""
        system = AndonSystem()
        child_sig = AndonSignal(source="s", severity="critical", message="child")
        parent_sig = AndonSignal(source="s", severity="critical", message="parent")
        system.pause_scope("child", child_sig)
        system.pause_scope("parent", parent_sig)

        result = system.acknowledge_scope("child", parent_scope="parent")
        assert result is True
        assert system.is_scope_paused("child") is False
        assert system.is_scope_paused("parent") is False

    def test_acknowledge_scope_with_unpaused_parent_still_succeeds(self) -> None:
        """acknowledge_scope succeeds even when the parent scope is not paused."""
        system = AndonSystem()
        child_sig = AndonSignal(source="s", severity="critical", message="child")
        system.pause_scope("child", child_sig)

        # Parent is not paused — should not raise
        result = system.acknowledge_scope("child", parent_scope="nonexistent-parent")
        assert result is True
        assert system.is_scope_paused("child") is False
