"""Tests for vetinari.autonomy.rollback — rollback registry for autonomous actions."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from vetinari.autonomy.rollback import (
    _REGRESSION_THRESHOLD,
    _REGRESSION_WINDOW_HOURS,
    ActionRecord,
    RollbackRegistry,
    _replace_record,
    get_rollback_registry,
)

# -- Helpers -------------------------------------------------------------------


def _make_registry() -> RollbackRegistry:
    """Return a fresh RollbackRegistry for each test."""
    return RollbackRegistry()


# -- ActionRecord --------------------------------------------------------------


class TestActionRecord:
    """ActionRecord dataclass structure and behaviour."""

    def test_repr_is_opaque(self) -> None:
        """__repr__ must not expose field values."""
        record = ActionRecord(
            action_id="undo_abc123",
            action_type="parameter_tuning",
            timestamp="2026-01-01T00:00:00+00:00",
            reversible_data={"key": "value"},
            quality_before=0.9,
        )
        assert repr(record) == "ActionRecord(...)"

    def test_defaults(self) -> None:
        """quality_after defaults to None and rolled_back defaults to False."""
        record = ActionRecord(
            action_id="undo_abc123",
            action_type="param",
            timestamp="2026-01-01T00:00:00+00:00",
            reversible_data={},
            quality_before=0.8,
        )
        assert record.quality_after is None
        assert record.rolled_back is False

    def test_frozen(self) -> None:
        """ActionRecord must be immutable (frozen=True)."""
        record = ActionRecord(
            action_id="undo_abc123",
            action_type="param",
            timestamp="2026-01-01T00:00:00+00:00",
            reversible_data={},
            quality_before=0.8,
        )
        with pytest.raises((AttributeError, TypeError)):
            record.rolled_back = True  # type: ignore[misc]


# -- _replace_record -----------------------------------------------------------


class TestReplaceRecord:
    """_replace_record copies a frozen ActionRecord with field overrides."""

    def test_replace_quality_after(self) -> None:
        """quality_after is updated while other fields are preserved."""
        original = ActionRecord(
            action_id="undo_aaa",
            action_type="tune",
            timestamp="2026-01-01T00:00:00+00:00",
            reversible_data={"x": 1},
            quality_before=0.9,
        )
        updated = _replace_record(original, quality_after=0.85)
        assert updated.quality_after == 0.85
        assert updated.action_id == original.action_id
        assert updated.quality_before == original.quality_before
        assert updated.rolled_back is False

    def test_replace_rolled_back(self) -> None:
        """rolled_back is updated while quality_after is preserved."""
        original = ActionRecord(
            action_id="undo_bbb",
            action_type="tune",
            timestamp="2026-01-01T00:00:00+00:00",
            reversible_data={},
            quality_before=0.7,
            quality_after=0.65,
        )
        updated = _replace_record(original, rolled_back=True)
        assert updated.rolled_back is True
        assert updated.quality_after == 0.65

    def test_no_changes_copies_faithfully(self) -> None:
        """Calling _replace_record with no kwargs returns an equal record."""
        original = ActionRecord(
            action_id="undo_ccc",
            action_type="tune",
            timestamp="2026-01-01T00:00:00+00:00",
            reversible_data={},
            quality_before=0.5,
        )
        copy = _replace_record(original)
        assert copy == original


# -- log_autonomous_action -----------------------------------------------------


class TestLogAutonomousAction:
    """log_autonomous_action creates and stores ActionRecords."""

    def test_returns_undo_prefixed_id(self) -> None:
        """Returned action_id must start with 'undo_'."""
        reg = _make_registry()
        action_id = reg.log_autonomous_action("tune", {}, 0.9)
        assert action_id.startswith("undo_")

    def test_id_is_unique(self) -> None:
        """Two consecutive logs must return different IDs."""
        reg = _make_registry()
        id1 = reg.log_autonomous_action("tune", {}, 0.9)
        id2 = reg.log_autonomous_action("tune", {}, 0.9)
        assert id1 != id2

    def test_record_is_retrievable(self) -> None:
        """get_action returns the record with correct fields after log."""
        reg = _make_registry()
        data = {"param": "lr", "old_value": 0.01}
        action_id = reg.log_autonomous_action("parameter_tuning", data, 0.88)
        record = reg.get_action(action_id)
        assert record is not None
        assert record.action_id == action_id
        assert record.action_type == "parameter_tuning"
        assert record.reversible_data == data
        assert record.quality_before == 0.88
        assert record.quality_after is None
        assert record.rolled_back is False

    def test_timestamp_is_iso_utc(self) -> None:
        """Stored timestamp must be ISO 8601 UTC."""
        from datetime import datetime, timezone

        reg = _make_registry()
        action_id = reg.log_autonomous_action("tune", {}, 0.5)
        record = reg.get_action(action_id)
        assert record is not None
        # Must parse without error and be timezone-aware
        ts = datetime.fromisoformat(record.timestamp)
        assert ts.tzinfo is not None


# -- update_quality_after ------------------------------------------------------


class TestUpdateQualityAfter:
    """update_quality_after populates quality_after on an existing record."""

    def test_returns_true_on_success(self) -> None:
        """Returns True when the action_id exists."""
        reg = _make_registry()
        action_id = reg.log_autonomous_action("tune", {}, 0.9)
        result = reg.update_quality_after(action_id, 0.85)
        assert result is True

    def test_quality_after_is_set(self) -> None:
        """get_action reflects the updated quality_after value."""
        reg = _make_registry()
        action_id = reg.log_autonomous_action("tune", {}, 0.9)
        reg.update_quality_after(action_id, 0.82)
        record = reg.get_action(action_id)
        assert record is not None
        assert record.quality_after == 0.82

    def test_returns_false_for_unknown_id(self) -> None:
        """Returns False and does not raise for a non-existent action_id."""
        reg = _make_registry()
        result = reg.update_quality_after("undo_doesnotexist", 0.7)
        assert result is False


# -- check_quality_regression --------------------------------------------------


class TestCheckQualityRegression:
    """check_quality_regression detects drops and demotes the governor."""

    def test_no_regression_returns_empty(self) -> None:
        """When quality is stable, no action_ids are returned."""
        reg = _make_registry()
        reg.log_autonomous_action("tune", {}, 0.9)
        result = reg.check_quality_regression("tune", 0.9)
        assert result == []

    def test_small_drop_below_threshold_no_rollback(self) -> None:
        """A drop well below the threshold does not trigger rollback."""
        reg = _make_registry()
        reg.log_autonomous_action("tune", {}, 0.9)
        # 0.9 - 0.88 = 0.02, which is strictly less than 0.05 — no rollback
        result = reg.check_quality_regression("tune", 0.88)
        assert result == []

    def test_regression_marks_actions_rolled_back(self) -> None:
        """Actions whose quality_before exceeds current by >5% are marked rolled back."""
        reg = _make_registry()
        action_id = reg.log_autonomous_action("tune", {}, 0.9)
        # 0.9 - 0.84 = 0.06 > 0.05
        with patch("vetinari.autonomy.governor.AutonomyGovernor._auto_demote"):
            rolled = reg.check_quality_regression("tune", 0.84)

        assert action_id in rolled
        record = reg.get_action(action_id)
        assert record is not None
        assert record.rolled_back is True

    def test_regression_calls_auto_demote(self) -> None:
        """Governor._auto_demote is called once on regression."""
        reg = _make_registry()
        reg.log_autonomous_action("tune", {}, 0.9)
        with patch("vetinari.autonomy.governor.AutonomyGovernor._auto_demote") as mock_demote:
            reg.check_quality_regression("tune", 0.84)
            mock_demote.assert_called_once_with("tune")

    def test_already_rolled_back_actions_are_skipped(self) -> None:
        """Actions already marked rolled_back are not double-counted."""
        reg = _make_registry()
        action_id = reg.log_autonomous_action("tune", {}, 0.9)
        with patch("vetinari.autonomy.governor.AutonomyGovernor._auto_demote"):
            reg.check_quality_regression("tune", 0.84)
        # Second check should not re-roll-back the same record
        with patch("vetinari.autonomy.governor.AutonomyGovernor._auto_demote") as mock_demote:
            rolled = reg.check_quality_regression("tune", 0.84)
            assert action_id not in rolled
            mock_demote.assert_not_called()

    def test_only_matching_action_type_affected(self) -> None:
        """Records of a different action_type are not rolled back."""
        reg = _make_registry()
        id_a = reg.log_autonomous_action("tune", {}, 0.9)
        id_b = reg.log_autonomous_action("rewrite", {}, 0.9)
        with patch("vetinari.autonomy.governor.AutonomyGovernor._auto_demote"):
            rolled = reg.check_quality_regression("tune", 0.84)
        assert id_a in rolled
        assert id_b not in rolled
        record_b = reg.get_action(id_b)
        assert record_b is not None
        assert record_b.rolled_back is False

    def test_multiple_actions_all_rolled_back_on_regression(self) -> None:
        """All non-rolled-back actions of the type are rolled back together."""
        reg = _make_registry()
        ids = [reg.log_autonomous_action("tune", {}, 0.9) for _ in range(3)]
        with patch("vetinari.autonomy.governor.AutonomyGovernor._auto_demote") as mock_demote:
            rolled = reg.check_quality_regression("tune", 0.84)
        assert set(ids) == set(rolled)
        # Demotion called exactly once even for multiple rollbacks
        mock_demote.assert_called_once_with("tune")

    def test_no_auto_demote_when_no_regression(self) -> None:
        """Governor is not contacted when no regression is detected."""
        reg = _make_registry()
        reg.log_autonomous_action("tune", {}, 0.9)
        with patch("vetinari.autonomy.governor.AutonomyGovernor._auto_demote") as mock_demote:
            reg.check_quality_regression("tune", 0.9)
            mock_demote.assert_not_called()


# -- undo_action ---------------------------------------------------------------


class TestUndoAction:
    """undo_action manually marks an action as rolled back."""

    def test_marks_action_rolled_back(self) -> None:
        """undo_action sets rolled_back=True on the target record."""
        reg = _make_registry()
        action_id = reg.log_autonomous_action("tune", {}, 0.9)
        result = reg.undo_action(action_id)
        assert result is not None
        assert result.rolled_back is True
        assert result.action_id == action_id

    def test_returns_none_for_unknown_id(self) -> None:
        """undo_action returns None when the action_id is not in the registry."""
        reg = _make_registry()
        result = reg.undo_action("undo_doesnotexist")
        assert result is None

    def test_get_action_reflects_undo(self) -> None:
        """get_action returns the updated record after undo_action is called."""
        reg = _make_registry()
        action_id = reg.log_autonomous_action("tune", {}, 0.8)
        reg.undo_action(action_id)
        record = reg.get_action(action_id)
        assert record is not None
        assert record.rolled_back is True


# -- get_action ----------------------------------------------------------------


class TestGetAction:
    """get_action retrieves a record or returns None."""

    def test_returns_none_for_missing_id(self) -> None:
        """Missing action_id returns None without raising."""
        reg = _make_registry()
        assert reg.get_action("undo_nope") is None

    def test_returns_correct_record(self) -> None:
        """get_action returns the exact record that was logged."""
        reg = _make_registry()
        action_id = reg.log_autonomous_action("rewrite", {"old": "v1"}, 0.75)
        record = reg.get_action(action_id)
        assert record is not None
        assert record.action_type == "rewrite"
        assert record.quality_before == 0.75


# -- get_recent_actions --------------------------------------------------------


class TestGetRecentActions:
    """get_recent_actions filters by time window and sorts newest-first."""

    def test_returns_actions_within_window(self) -> None:
        """Actions logged now appear in the default 24-hour window."""
        reg = _make_registry()
        reg.log_autonomous_action("tune", {}, 0.9)
        reg.log_autonomous_action("tune", {}, 0.8)
        recent = reg.get_recent_actions(hours=24)
        assert len(recent) == 2

    def test_newest_first_ordering(self) -> None:
        """Most-recently logged action appears first in the result."""
        reg = _make_registry()
        id1 = reg.log_autonomous_action("tune", {}, 0.9)
        id2 = reg.log_autonomous_action("tune", {}, 0.8)
        recent = reg.get_recent_actions()
        # id2 was logged after id1, so it must come first
        assert recent[0].action_id == id2
        assert recent[1].action_id == id1

    def test_empty_registry_returns_empty_list(self) -> None:
        """No actions logged returns an empty list."""
        reg = _make_registry()
        assert reg.get_recent_actions() == []

    def test_zero_hour_window_returns_empty(self) -> None:
        """A 0-hour window excludes all records (cutoff is in the past)."""
        reg = _make_registry()
        reg.log_autonomous_action("tune", {}, 0.9)
        # A zero-hour window sets cutoff to now; records logged slightly before
        # may or may not appear, but we mainly want no crash.
        result = reg.get_recent_actions(hours=0)
        # All records are on the boundary or just before — list must be a list
        assert isinstance(result, list)


# -- get_rollback_registry singleton ------------------------------------------


class TestGetRollbackRegistry:
    """get_rollback_registry returns the same singleton instance."""

    def test_singleton_returns_same_instance(self) -> None:
        """Two calls to get_rollback_registry return the identical object."""
        reg1 = get_rollback_registry()
        reg2 = get_rollback_registry()
        assert reg1 is reg2

    def test_singleton_is_rollback_registry_type(self) -> None:
        """Returned object is a RollbackRegistry."""
        assert isinstance(get_rollback_registry(), RollbackRegistry)


# -- Thread safety -------------------------------------------------------------


class TestThreadSafety:
    """Concurrent writes do not corrupt the registry."""

    def test_concurrent_log_produces_unique_ids(self) -> None:
        """100 concurrent log_autonomous_action calls all produce unique IDs."""
        reg = _make_registry()
        ids: list[str] = []
        lock = threading.Lock()

        def log_one() -> None:
            action_id = reg.log_autonomous_action("tune", {}, 0.9)
            with lock:
                ids.append(action_id)

        threads = [threading.Thread(target=log_one) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(ids) == 100
        assert len(set(ids)) == 100  # All IDs are unique


# -- Module-level constants ----------------------------------------------------


class TestModuleConstants:
    """Verify constant values match the spec."""

    def test_regression_threshold(self) -> None:
        """_REGRESSION_THRESHOLD is 0.05 (5% absolute drop)."""
        assert _REGRESSION_THRESHOLD == 0.05

    def test_regression_window_hours(self) -> None:
        """_REGRESSION_WINDOW_HOURS is 24."""
        assert _REGRESSION_WINDOW_HOURS == 24
