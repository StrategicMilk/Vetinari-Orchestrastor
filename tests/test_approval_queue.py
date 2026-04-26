"""Tests for vetinari.autonomy.approval_queue — SQLite-backed approval lifecycle."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vetinari.autonomy.approval_queue import ApprovalQueue
from vetinari.types import AutonomyLevel, PermissionDecision

# -- enqueue ------------------------------------------------------------------


class TestEnqueue:
    """enqueue() persists an action and returns a unique action_id."""

    def test_returns_action_id(self, tmp_path: Path) -> None:
        """enqueue() returns a non-empty string ID."""
        queue = ApprovalQueue(db_path=tmp_path / "aq.db")
        action_id = queue.enqueue("prompt_optimization", confidence=0.75)
        assert isinstance(action_id, str)
        assert len(action_id) > 0

    def test_unique_ids_per_call(self, tmp_path: Path) -> None:
        """Each enqueue() call produces a distinct action_id."""
        queue = ApprovalQueue(db_path=tmp_path / "aq.db")
        id1 = queue.enqueue("action_a")
        id2 = queue.enqueue("action_a")
        assert id1 != id2


# -- approve / reject ---------------------------------------------------------


class TestDecide:
    """approve() and reject() update status correctly."""

    def test_approve_changes_status(self, tmp_path: Path) -> None:
        """approve() removes the action from pending and writes a decision_log row."""
        queue = ApprovalQueue(db_path=tmp_path / "aq.db")
        action_id = queue.enqueue("model_swap")
        result = queue.approve(action_id)
        assert result is True
        pending_ids = [p.action_id for p in queue.get_pending()]
        assert action_id not in pending_ids
        log = queue.get_decision_log()
        assert len(log) == 1
        assert log[0].action_id == action_id
        assert log[0].decision == "approve"

    def test_reject_changes_status(self, tmp_path: Path) -> None:
        """reject() removes the action from pending and writes a decision_log row."""
        queue = ApprovalQueue(db_path=tmp_path / "aq.db")
        action_id = queue.enqueue("dangerous_op")
        result = queue.reject(action_id, reason="too risky")
        assert result is True
        pending_ids = [p.action_id for p in queue.get_pending()]
        assert action_id not in pending_ids
        log = queue.get_decision_log()
        assert len(log) == 1
        assert log[0].action_id == action_id
        assert log[0].decision == "deny"

    def test_approve_nonexistent_returns_false(self, tmp_path: Path) -> None:
        """Approving an ID that does not exist returns False."""
        queue = ApprovalQueue(db_path=tmp_path / "aq.db")
        result = queue.approve("act_nonexistent")
        assert result is False

    def test_double_approve_second_returns_false(self, tmp_path: Path) -> None:
        """Approving an already-approved action returns False (not pending)."""
        queue = ApprovalQueue(db_path=tmp_path / "aq.db")
        action_id = queue.enqueue("idempotent_op")
        queue.approve(action_id)
        result = queue.approve(action_id)
        assert result is False

    def test_approve_invokes_callback(self, tmp_path: Path) -> None:
        """approve() calls the on_decided callback with correct arguments."""
        queue = ApprovalQueue(db_path=tmp_path / "aq.db")
        calls: list[tuple[str, str, dict]] = []

        def on_decided(action_id: str, status: str, details: dict) -> None:
            calls.append((action_id, status, details))

        action_id = queue.enqueue("model_swap", details={"model": "llama3"}, on_decided=on_decided)
        result = queue.approve(action_id)
        assert result is True
        assert len(calls) == 1
        called_id, called_status, called_details = calls[0]
        assert called_id == action_id
        assert called_status == "approved"
        assert called_details == {"model": "llama3"}

    def test_reject_invokes_callback(self, tmp_path: Path) -> None:
        """reject() calls the on_decided callback with status 'rejected'."""
        queue = ApprovalQueue(db_path=tmp_path / "aq.db")
        calls: list[tuple[str, str, dict]] = []

        def on_decided(action_id: str, status: str, details: dict) -> None:
            calls.append((action_id, status, details))

        action_id = queue.enqueue("dangerous_op", on_decided=on_decided)
        result = queue.reject(action_id, reason="too risky")
        assert result is True
        assert len(calls) == 1
        called_id, called_status, _ = calls[0]
        assert called_id == action_id
        assert called_status == "rejected"

    def test_callback_exception_does_not_break_approval(self, tmp_path: Path) -> None:
        """A callback that raises must not prevent approve() from returning True."""
        queue = ApprovalQueue(db_path=tmp_path / "aq.db")

        def bad_callback(action_id: str, status: str, details: dict) -> None:
            raise RuntimeError("simulated callback failure")

        action_id = queue.enqueue("risky_action", on_decided=bad_callback)
        result = queue.approve(action_id)
        # Decision must still be recorded and returned as success
        assert result is True
        pending_ids = [p.action_id for p in queue.get_pending()]
        assert action_id not in pending_ids

    def test_approve_writes_decision_log(self, tmp_path: Path) -> None:
        """approve() writes a decision_log row with matching action_id and action_type."""
        queue = ApprovalQueue(db_path=tmp_path / "aq.db")
        action_id = queue.enqueue("prompt_optimization", details={"key": "val"}, confidence=0.7)
        queue.approve(action_id)
        log = queue.get_decision_log()
        assert len(log) == 1
        entry = log[0]
        assert entry.action_id == action_id
        assert entry.action_type == "prompt_optimization"
        assert entry.decision == "approve"

    def test_reject_writes_decision_log(self, tmp_path: Path) -> None:
        """reject() writes a decision_log row with decision='deny'."""
        queue = ApprovalQueue(db_path=tmp_path / "aq.db")
        action_id = queue.enqueue("dangerous_mutation", confidence=0.2)
        queue.reject(action_id, reason="safety threshold exceeded")
        log = queue.get_decision_log()
        assert len(log) == 1
        entry = log[0]
        assert entry.action_id == action_id
        assert entry.action_type == "dangerous_mutation"
        assert entry.decision == "deny"

    def test_decision_log_atomic_with_status_update(self, tmp_path: Path) -> None:
        """After approve, no pending row exists AND a decision_log row exists — atomically."""
        queue = ApprovalQueue(db_path=tmp_path / "aq.db")
        action_id = queue.enqueue("atomic_op", confidence=0.9)
        queue.approve(action_id)
        # (a) No pending row
        pending_ids = [p.action_id for p in queue.get_pending()]
        assert action_id not in pending_ids
        # (b) Decision log row is present
        log = queue.get_decision_log()
        log_ids = [e.action_id for e in log]
        assert action_id in log_ids

    def test_expired_action_cannot_be_approved_without_prior_scan(self, tmp_path: Path) -> None:
        """Decision paths enforce expiry even if get_pending() was never called."""
        queue = ApprovalQueue(db_path=tmp_path / "aq.db", expiry_hours=1)
        action_id = queue.enqueue("stale_action")
        old_created_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        conn = queue._get_connection()
        try:
            conn.execute("UPDATE approval_queue SET created_at = ? WHERE action_id = ?", (old_created_at, action_id))
            conn.commit()
        finally:
            conn.close()

        assert queue.approve(action_id) is False
        log = queue.get_decision_log()
        assert len(log) == 1
        assert log[0].action_id == action_id
        assert log[0].decision == "expired"
        assert queue.get_pending() == []


# -- Restart / callback loss --------------------------------------------------


class TestRestartCallbackLoss:
    """Verify behaviour when approval happens after a simulated process restart."""

    def test_callback_lost_after_restart(self, tmp_path: Path) -> None:
        """Approving via a fresh queue instance (simulated restart) must: return True,
        write the decision_log row, and NOT invoke the callback from the dead instance.
        """
        db_path = tmp_path / "aq_restart.db"
        callback_invoked: list[bool] = []

        def on_decided(action_id: str, status: str, details: dict) -> None:
            callback_invoked.append(True)

        # Original process: enqueue with callback
        queue_original = ApprovalQueue(db_path=db_path)
        action_id = queue_original.enqueue("restart_test_op", confidence=0.5, on_decided=on_decided)

        # Simulate restart: new queue instance on the same DB, no callback registered
        queue_restarted = ApprovalQueue(db_path=db_path)
        result = queue_restarted.approve(action_id)

        assert result is True
        # Decision must be persisted even without a live callback
        log = queue_restarted.get_decision_log()
        log_ids = [e.action_id for e in log]
        assert action_id in log_ids
        # Callback belongs to the dead queue — it must NOT have been called
        assert callback_invoked == []


# -- get_pending --------------------------------------------------------------


class TestGetPending:
    """get_pending() returns only un-decided, non-expired items."""

    def test_returns_only_pending(self, tmp_path: Path) -> None:
        """Approved and rejected items are excluded from get_pending()."""
        queue = ApprovalQueue(db_path=tmp_path / "aq.db")
        keep = queue.enqueue("stay_pending")
        gone = queue.enqueue("will_be_approved")
        queue.approve(gone)
        pending = queue.get_pending()
        ids = [p.action_id for p in pending]
        assert keep in ids
        assert gone not in ids

    def test_expired_items_excluded(self, tmp_path: Path) -> None:
        """Items with created_at older than expiry_hours do not appear in get_pending()."""
        # Use 0-hour expiry so any item inserted even a millisecond ago is expired
        queue = ApprovalQueue(db_path=tmp_path / "aq.db", expiry_hours=0)
        queue.enqueue("old_action")
        # Immediately call get_pending — with 0h expiry the item should have expired
        pending = queue.get_pending()
        assert pending == []


# -- log_decision / get_decision_log ------------------------------------------


class TestDecisionLog:
    """log_decision() and get_decision_log() implement the audit trail."""

    def test_log_decision_records_entry(self, tmp_path: Path) -> None:
        """log_decision() inserts a row that get_decision_log() returns."""
        queue = ApprovalQueue(db_path=tmp_path / "aq.db")
        queue.log_decision(
            action_type="param_tuning",
            autonomy_level=AutonomyLevel.L3_ACT_LOG,
            decision=PermissionDecision.APPROVE,
            confidence=0.9,
        )
        entries = queue.get_decision_log()
        assert len(entries) >= 1
        entry = entries[0]
        assert entry.action_type == "param_tuning"
        assert entry.autonomy_level == AutonomyLevel.L3_ACT_LOG.value
        assert entry.decision == PermissionDecision.APPROVE.value

    def test_get_decision_log_filtered_by_action_type(self, tmp_path: Path) -> None:
        """get_decision_log(action_type=) returns only matching entries."""
        queue = ApprovalQueue(db_path=tmp_path / "aq.db")
        queue.log_decision("type_a", AutonomyLevel.L2_ACT_REPORT, PermissionDecision.APPROVE)
        queue.log_decision("type_b", AutonomyLevel.L1_SUGGEST, PermissionDecision.DEFER)
        entries = queue.get_decision_log(action_type="type_a")
        assert all(e.action_type == "type_a" for e in entries)
        assert len(entries) == 1


# -- Full lifecycle -----------------------------------------------------------


class TestFullLifecycle:
    """End-to-end enqueue -> approve/reject -> logged."""

    def test_enqueue_approve_lifecycle(self, tmp_path: Path) -> None:
        """A complete approval lifecycle leaves zero pending items."""
        queue = ApprovalQueue(db_path=tmp_path / "aq.db")
        action_id = queue.enqueue("safe_mutation", confidence=0.88)
        assert len(queue.get_pending()) == 1
        result = queue.approve(action_id)
        assert result is True
        assert queue.get_pending() == []

    def test_enqueue_reject_lifecycle(self, tmp_path: Path) -> None:
        """A complete rejection lifecycle leaves zero pending items."""
        queue = ApprovalQueue(db_path=tmp_path / "aq.db")
        action_id = queue.enqueue("dangerous_mutation", confidence=0.3)
        assert len(queue.get_pending()) == 1
        result = queue.reject(action_id, reason="change exceeds safety threshold")
        assert result is True
        assert queue.get_pending() == []


# -- record_outcome -----------------------------------------------------------


class TestOutcomeRecording:
    """record_outcome() persists execution results on decided actions."""

    def test_record_outcome_after_approval(self, tmp_path: Path) -> None:
        """record_outcome() returns True and stores the outcome for an approved action."""
        queue = ApprovalQueue(db_path=tmp_path / "aq.db")
        action_id = queue.enqueue("model_swap", confidence=0.8)
        queue.approve(action_id)
        recorded = queue.record_outcome(action_id, outcome="Model substitution applied successfully.")
        assert recorded is True

    def test_record_outcome_on_pending_is_rejected(self, tmp_path: Path) -> None:
        """record_outcome() returns False when the action is still pending."""
        queue = ApprovalQueue(db_path=tmp_path / "aq.db")
        action_id = queue.enqueue("pending_op", confidence=0.5)
        # Do NOT approve — action remains pending
        recorded = queue.record_outcome(action_id, outcome="Should not be stored.")
        assert recorded is False
