"""Approval Queue — SQLite-backed queue for actions requiring human sign-off.

When the governor returns DEFER for an L1 action, that action is enqueued
here. Humans approve or reject via the dashboard API. All decisions at
ALL autonomy levels are logged to the decision_log table for audit.

Tables:
  - approval_queue: pending actions awaiting human approval
  - decision_log: complete audit trail of every autonomous decision
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from vetinari.constants import get_user_dir
from vetinari.types import AutonomyLevel, PermissionDecision

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = get_user_dir() / "autonomy.db"
_DEFAULT_EXPIRY_HOURS = 24  # Pending approvals expire after this many hours

# Approval statuses
_STATUS_PENDING = "pending"
_STATUS_APPROVED = "approved"
_STATUS_REJECTED = "rejected"
_STATUS_EXPIRED = "expired"


@dataclass(frozen=True)
class PendingAction:
    """A queued action awaiting human approval.

    Args:
        action_id: Unique identifier for this approval request.
        action_type: The kind of action (e.g. ``"model_substitution"``).
        details: JSON-serializable metadata about the specific action.
        confidence: Agent's confidence score for this action (0.0-1.0).
        status: Current status (pending/approved/rejected/expired).
        created_at: When the action was enqueued (ISO 8601 UTC).
    """

    action_id: str
    action_type: str
    details: dict[str, Any]
    confidence: float
    status: str
    created_at: str

    def __repr__(self) -> str:
        return "PendingAction(...)"


@dataclass(frozen=True)
class DecisionLogEntry:
    """An entry in the decision audit log.

    Every autonomous decision at ALL levels is recorded here.

    Args:
        action_id: Unique identifier for the action.
        action_type: The kind of action.
        autonomy_level: The level that was applied.
        decision: The permission decision (approve/deny/defer).
        confidence: Agent confidence for this action.
        outcome: Result of the action (if executed).
        timestamp: When the decision was made (ISO 8601 UTC).
    """

    action_id: str
    action_type: str
    autonomy_level: str
    decision: str
    confidence: float
    outcome: str
    timestamp: str

    def __repr__(self) -> str:
        return "DecisionLogEntry(...)"


class ApprovalQueue:
    """SQLite-backed approval queue with decision audit logging.

    All database operations are guarded by a threading lock to prevent
    concurrent write corruption from Litestar's async workers.

    Side effects in __init__:
      - Creates SQLite database file at ``db_path`` if it doesn't exist
      - Creates approval_queue and decision_log tables
    """

    def __init__(self, db_path: Path | None = None, expiry_hours: int = _DEFAULT_EXPIRY_HOURS) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._expiry_hours = expiry_hours
        self._lock = threading.Lock()
        # Maps action_id -> callback(action_id, decision_status, details)
        # Populated by enqueue() when the caller provides on_decided.
        # Entries are consumed (popped) exactly once in _decide().
        self._callbacks: dict[str, Callable[[str, str, dict[str, Any]], None]] = {}
        self._ensure_tables()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with WAL mode and busy timeout."""
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        """Create tables if they don't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            conn = self._get_connection()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS approval_queue (
                        action_id TEXT PRIMARY KEY,
                        action_type TEXT NOT NULL,
                        details_json TEXT NOT NULL DEFAULT '{}',
                        confidence REAL NOT NULL DEFAULT 0.0,
                        status TEXT NOT NULL DEFAULT 'pending',
                        created_at TEXT NOT NULL,
                        decided_at TEXT,
                        decided_by TEXT,
                        outcome TEXT NOT NULL DEFAULT '',
                        outcome_at TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_aq_status ON approval_queue(status);
                    CREATE INDEX IF NOT EXISTS idx_aq_created ON approval_queue(created_at);

                    CREATE TABLE IF NOT EXISTS decision_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        action_id TEXT NOT NULL,
                        action_type TEXT NOT NULL,
                        autonomy_level TEXT NOT NULL,
                        decision TEXT NOT NULL,
                        confidence REAL NOT NULL DEFAULT 0.0,
                        details_json TEXT NOT NULL DEFAULT '{}',
                        outcome TEXT NOT NULL DEFAULT '',
                        timestamp TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_dl_action_type ON decision_log(action_type);
                    CREATE INDEX IF NOT EXISTS idx_dl_timestamp ON decision_log(timestamp);
                """)
                conn.commit()
            finally:
                conn.close()

    def enqueue(
        self,
        action_type: str,
        details: dict[str, Any] | None = None,
        confidence: float = 0.0,
        on_decided: Callable[[str, str, dict[str, Any]], None] | None = None,
    ) -> str:
        """Add an action to the approval queue.

        The ``on_decided`` callback is stored **in memory only**. On process
        restart, any pending approval's callback is lost. The decision itself
        is always persisted to SQLite; it is the caller's responsibility to
        reconcile unresumed actions after restart (e.g. by querying
        ``get_pending()`` and re-registering callbacks).

        Args:
            action_type: The kind of action needing approval.
            details: JSON-serializable metadata about the action.
            confidence: Agent's confidence score (0.0-1.0).
            on_decided: Optional callback invoked when the action is approved or
                rejected. Signature: ``callback(action_id, status, details)`` where
                ``status`` is ``"approved"`` or ``"rejected"``. Exceptions raised by
                the callback are caught and logged — the decision is always persisted
                regardless of callback success.

        Returns:
            The unique action_id for tracking this approval request.

        Raises:
            sqlite3.Error: If the approval request cannot be persisted.
            TypeError: If details cannot be serialized as JSON.
        """
        action_id = f"act_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        details_json = json.dumps(details or {})  # noqa: VET112 — Optional per func param

        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """INSERT INTO approval_queue
                       (action_id, action_type, details_json, confidence, status, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (action_id, action_type, details_json, confidence, _STATUS_PENDING, now),
                )
                if on_decided is not None:
                    self._callbacks[action_id] = on_decided
                conn.commit()
            except Exception:
                self._callbacks.pop(action_id, None)
                conn.rollback()
                raise
            finally:
                conn.close()

        logger.info(
            "Enqueued action %s (type=%s, confidence=%.2f) for human approval",
            action_id,
            action_type,
            confidence,
        )
        return action_id

    def approve(self, action_id: str, decided_by: str = "human") -> bool:
        """Approve a pending action.

        Args:
            action_id: The action to approve.
            decided_by: Who approved it (for audit trail).

        Returns:
            True if approved successfully, False if action not found or not pending.
        """
        return self._decide(action_id, _STATUS_APPROVED, decided_by)

    def reject(self, action_id: str, reason: str = "", decided_by: str = "human") -> bool:
        """Reject a pending action.

        Args:
            action_id: The action to reject.
            reason: Optional rejection reason (stored in decision log).
            decided_by: Who rejected it (for audit trail).

        Returns:
            True if rejected successfully, False if action not found or not pending.
        """
        return self._decide(action_id, _STATUS_REJECTED, decided_by, reason=reason)

    def _decide(self, action_id: str, status: str, decided_by: str, reason: str = "") -> bool:
        """Apply a decision (approve/reject) to a pending action.

        The queue-status UPDATE and decision_log INSERT are performed in a
        single SQLite transaction so the audit trail is never disconnected from
        the actual approve/reject event.  After the transaction commits, any
        registered in-memory callback is popped and invoked.  Callback
        exceptions are caught and logged — the decision is always considered
        final regardless of whether the callback succeeds.

        If no callback is registered (e.g. after a process restart), a WARNING
        is logged so operators know the resumer must re-register or treat the
        action as unresumable without explicit recovery.
        """
        self._expire_stale()
        now = datetime.now(timezone.utc).isoformat()
        details: dict[str, Any] = {}
        decision_logged = False

        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT action_type, details_json, confidence, created_at FROM approval_queue "
                    "WHERE action_id = ? AND status = ?",
                    (action_id, _STATUS_PENDING),
                )
                row = cursor.fetchone()
                if row is None:
                    return False

                details = json.loads(row["details_json"])
                action_type = row["action_type"]
                confidence = row["confidence"]
                if self._is_expired(row["created_at"]):
                    self._expire_row_locked(conn, action_id, action_type, details, confidence, now)
                    conn.commit()
                    self._callbacks.pop(action_id, None)
                    return False

                # Queue status update and audit-log insert are an atomic pair.
                # Using one connection + conn.commit() guarantees both writes
                # land in the same WAL transaction.
                conn.execute(
                    "UPDATE approval_queue SET status = ?, decided_at = ?, decided_by = ? WHERE action_id = ?",
                    (status, now, decided_by, action_id),
                )

                # Map queue status string to the canonical decision string used
                # in decision_log.  Queue actions originate from DEFER paths
                # which are L1_SUGGEST level — the governor only enqueues when
                # it cannot auto-approve.
                decision_str = "approve" if status == _STATUS_APPROVED else "deny"

                conn.execute(
                    "INSERT INTO decision_log "
                    "(action_id, action_type, autonomy_level, decision, confidence, details_json, outcome, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        action_id,
                        action_type,
                        AutonomyLevel.L1_SUGGEST.value,  # DEFER path is always L1_SUGGEST
                        decision_str,
                        confidence,
                        json.dumps(details),
                        reason,
                        now,
                    ),
                )
                conn.commit()
                decision_logged = True
            finally:
                conn.close()

        logger.info(
            "Action %s %s by %s (reason=%s)%s",
            action_id,
            status,
            decided_by,
            reason or "none",
            "" if decision_logged else " [WARNING: audit row NOT written]",
        )

        callback = self._callbacks.pop(action_id, None)
        if callback is None:
            # Expected after process restart — the decision is persisted but
            # any caller that passed on_decided and then restarted must re-register
            # or handle recovery explicitly via get_pending().
            logger.warning(
                "Action %s %s but no in-memory callback registered — "
                "expected across process restart; decision is persisted "
                "but any resumer must re-register or treat callback as unresumable",
                action_id,
                status,
            )
        else:
            try:
                callback(action_id, status, details)
            except Exception:
                logger.warning(
                    "Callback for action %s raised an exception — decision recorded but action may not have resumed",
                    action_id,
                )
        return True

    def record_outcome(self, action_id: str, outcome: str) -> bool:
        """Record the execution outcome for a decided action.

        Only decided (approved/rejected/expired) actions can have an outcome
        recorded. Attempting to record an outcome on a still-pending action
        returns ``False`` so callers know the action has not been acted on yet.

        Args:
            action_id: The action whose outcome to record.
            outcome: Description of what happened when the action was executed.

        Returns:
            True if the outcome was stored, False if the action was not found
            or is still pending.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "UPDATE approval_queue SET outcome = ?, outcome_at = ? WHERE action_id = ? AND status != ?",
                    (outcome, now, action_id, _STATUS_PENDING),
                )
                conn.commit()
                updated = cursor.rowcount > 0
            finally:
                conn.close()

        if updated:
            logger.info("Recorded outcome for action %s", action_id)
        else:
            logger.warning(
                "Could not record outcome for action %s — action not found or still pending",
                action_id,
            )
        return updated

    def log_decision(
        self,
        action_type: str,
        autonomy_level: AutonomyLevel,
        decision: PermissionDecision,
        confidence: float = 0.0,
        details: dict[str, Any] | None = None,
        outcome: str = "",
    ) -> None:
        """Record a decision in the audit log (for ALL autonomy levels, not just L1).

        Args:
            action_type: The kind of action.
            autonomy_level: The autonomy level that was applied.
            decision: The permission decision.
            confidence: Agent confidence for this action.
            details: Optional metadata.
            outcome: Result of the action execution.
        """
        action_id = f"dec_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """INSERT INTO decision_log
                       (action_id, action_type, autonomy_level, decision, confidence, details_json, outcome, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        action_id,
                        action_type,
                        autonomy_level.value,
                        decision.value,
                        confidence,
                        json.dumps(details or {}),  # noqa: VET112 — Optional per func param
                        outcome,
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def get_pending(self) -> list[PendingAction]:
        """Return all pending approval requests, excluding expired ones.

        Also expires any stale entries that have exceeded the expiry window.

        Returns:
            List of PendingAction objects awaiting human decision.
        """
        self._expire_stale()
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT action_id, action_type, details_json, confidence, status, created_at "
                    "FROM approval_queue WHERE status = ? ORDER BY created_at ASC",
                    (_STATUS_PENDING,),
                )
                rows = cursor.fetchall()
            finally:
                conn.close()

        return [
            PendingAction(
                action_id=row["action_id"],
                action_type=row["action_type"],
                details=json.loads(row["details_json"]),
                confidence=row["confidence"],
                status=row["status"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_decision_log(
        self,
        action_type: str | None = None,
        limit: int = 100,
    ) -> list[DecisionLogEntry]:
        """Query the decision audit log.

        Args:
            action_type: Optional filter by action type.
            limit: Maximum entries to return.

        Returns:
            List of DecisionLogEntry objects, most recent first.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                if action_type:
                    cursor = conn.execute(
                        "SELECT action_id, action_type, autonomy_level, decision, confidence, outcome, timestamp "
                        "FROM decision_log WHERE action_type = ? ORDER BY timestamp DESC LIMIT ?",
                        (action_type, limit),
                    )
                else:
                    cursor = conn.execute(
                        "SELECT action_id, action_type, autonomy_level, decision, confidence, outcome, timestamp "
                        "FROM decision_log ORDER BY timestamp DESC LIMIT ?",
                        (limit,),
                    )
                rows = cursor.fetchall()
            finally:
                conn.close()

        return [
            DecisionLogEntry(
                action_id=row["action_id"],
                action_type=row["action_type"],
                autonomy_level=row["autonomy_level"],
                decision=row["decision"],
                confidence=row["confidence"],
                outcome=row["outcome"],
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

    def _expire_stale(self) -> None:
        """Mark pending approvals as expired if they've exceeded the timeout."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=self._expiry_hours)).isoformat()
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute(
                    "SELECT action_id, action_type, details_json, confidence FROM approval_queue "
                    "WHERE status = ? AND created_at < ?",
                    (_STATUS_PENDING, cutoff),
                ).fetchall()
                for row in rows:
                    details = json.loads(row["details_json"])
                    self._expire_row_locked(
                        conn,
                        row["action_id"],
                        row["action_type"],
                        details,
                        row["confidence"],
                        now,
                    )
                    self._callbacks.pop(row["action_id"], None)
                conn.commit()
            finally:
                conn.close()

    def _is_expired(self, created_at: str) -> bool:
        """Return True when *created_at* is older than this queue's expiry window."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._expiry_hours)
        try:
            created = datetime.fromisoformat(created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            logger.warning("Treating malformed approval timestamp as expired: %r", created_at)
            expired = True
            return expired
        return created < cutoff

    def _expire_row_locked(
        self,
        conn: sqlite3.Connection,
        action_id: str,
        action_type: str,
        details: dict[str, Any],
        confidence: float,
        timestamp: str,
    ) -> None:
        """Expire one pending action and write the matching audit row."""
        conn.execute(
            "UPDATE approval_queue SET status = ?, decided_at = ?, decided_by = ? "
            "WHERE action_id = ? AND status = ?",
            (_STATUS_EXPIRED, timestamp, "system-expiry", action_id, _STATUS_PENDING),
        )
        conn.execute(
            "INSERT INTO decision_log "
            "(action_id, action_type, autonomy_level, decision, confidence, details_json, outcome, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                action_id,
                action_type,
                AutonomyLevel.L1_SUGGEST.value,
                _STATUS_EXPIRED,
                confidence,
                json.dumps(details),
                "approval expired before decision",
                timestamp,
            ),
        )


# -- Singleton ----------------------------------------------------------------

_queue: ApprovalQueue | None = None
_queue_lock = threading.Lock()


def get_approval_queue(db_path: Path | None = None) -> ApprovalQueue:
    """Get or create the singleton ApprovalQueue.

    Args:
        db_path: Optional override for database path (used in tests).

    Returns:
        The singleton ApprovalQueue instance.
    """
    global _queue
    if _queue is None:
        with _queue_lock:
            if _queue is None:
                _queue = ApprovalQueue(db_path=db_path)
    return _queue
