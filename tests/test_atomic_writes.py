"""Atomic write tests for Vetinari's persistence layer.

Verifies that SQLite writes across checkpoint, SSE, and conversation
subsystems are crash-safe: either all data lands or none does.  The
checkpoint store is the critical case because it writes to two tables
in a single logical save; the others use single-INSERT writes that are
inherently atomic in SQLite.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def _db_path(tmp_path: Path) -> str:
    """Return a temp DB path and ensure the database module uses it."""
    import os

    path = str(tmp_path / "atomic_test.db")
    os.environ["VETINARI_DB_PATH"] = path
    import vetinari.database as db_mod

    db_mod.reset_for_testing()
    # Force schema init
    db_mod.get_connection()
    yield path
    os.environ.pop("VETINARI_DB_PATH", None)
    db_mod.reset_for_testing()


# ---------------------------------------------------------------------------
# Test 1 — checkpoint save atomicity
# ---------------------------------------------------------------------------


class TestCheckpointAtomicity:
    """Verify CheckpointStore.save_checkpoint() writes both tables atomically."""

    def test_checkpoint_save_writes_both_tables(self, _db_path: str) -> None:
        """A successful save_checkpoint populates both execution_state and task_checkpoints."""
        from vetinari.orchestration.checkpoint_store import CheckpointStore

        store = CheckpointStore()
        plan_id = "plan-atomic-001"
        graph_dict = {"goal": "test", "tasks": [], "created_at": "2026-01-01T00:00:00"}
        task_rows = [
            ("task-1", plan_id, "worker", "default", "pending", "{}", "{}", "", "", "", 0),
        ]
        store.save_checkpoint(plan_id, graph_dict, "executing", task_rows, "2026-01-01T00:00:00")

        conn = sqlite3.connect(_db_path)
        exec_rows = conn.execute(
            "SELECT execution_id, pipeline_state FROM execution_state WHERE execution_id = ?",
            (plan_id,),
        ).fetchall()
        task_rows_db = conn.execute(
            "SELECT task_id, status FROM task_checkpoints WHERE execution_id = ?",
            (plan_id,),
        ).fetchall()
        conn.close()

        assert len(exec_rows) == 1
        assert exec_rows[0][1] == "executing"
        assert len(task_rows_db) == 1
        assert task_rows_db[0][0] == "task-1"

    def test_failed_save_does_not_corrupt_existing(self, _db_path: str) -> None:
        """A valid checkpoint survives a subsequent failed save attempt."""
        from vetinari.orchestration.checkpoint_store import CheckpointStore

        store = CheckpointStore()
        plan_id = "plan-atomic-002"
        graph = {"goal": "original", "tasks": [], "created_at": "2026-01-01T00:00:00"}
        store.save_checkpoint(plan_id, graph, "executing", [], "2026-01-01T00:00:00")

        # Attempt a save that triggers a DB error by patching executemany
        # to raise mid-transaction
        with patch.object(
            store._db,
            "execute_in_transaction",
            side_effect=sqlite3.OperationalError("simulated crash"),
        ):
            with pytest.raises(sqlite3.OperationalError):
                bad_graph = {"goal": "corrupted", "tasks": [], "created_at": "2026-01-01T00:00:00"}
                store.save_checkpoint(plan_id, bad_graph, "failed", [], "2026-01-01T00:00:00")

        # Original checkpoint must still be intact
        loaded = store.load_checkpoint_graph_json(plan_id)
        assert loaded is not None
        assert json.loads(loaded)["goal"] == "original"


# ---------------------------------------------------------------------------
# Test 2 — transaction rollback
# ---------------------------------------------------------------------------


class TestTransactionRollback:
    """Verify execute_in_transaction rolls back on failure."""

    def test_rollback_on_second_statement_failure(self, _db_path: str) -> None:
        """If the second statement fails, the first statement's changes are rolled back."""
        from vetinari.orchestration.checkpoint_store import CheckpointStore

        store = CheckpointStore()

        # Direct test of execute_in_transaction with bad SQL
        valid_stmt = (
            "INSERT INTO execution_state (execution_id, goal, pipeline_state, task_dag_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("rollback-test", "goal", "executing", "{}", "2026-01-01", "2026-01-01"),
        )
        bad_stmt = (
            "INSERT INTO nonexistent_table (col) VALUES (?)",
            ("oops",),
        )

        with pytest.raises(sqlite3.OperationalError):
            store._db.execute_in_transaction(
                statements=[valid_stmt, bad_stmt],
            )

        # The valid statement should have been rolled back
        conn = sqlite3.connect(_db_path)
        rows = conn.execute(
            "SELECT COUNT(*) FROM execution_state WHERE execution_id = ?",
            ("rollback-test",),
        ).fetchone()
        conn.close()
        assert rows[0] == 0, "First statement should have been rolled back"


# ---------------------------------------------------------------------------
# Test 3 — SSE event write atomicity
# ---------------------------------------------------------------------------


class TestSSEWriteAtomicity:
    """Single-INSERT SSE writes are inherently atomic; verify round-trip."""

    def test_sse_event_persists_completely(self, _db_path: str) -> None:
        """An SSE event written to sse_event_log is fully readable."""
        from vetinari.database import get_connection

        conn = get_connection()
        payload = json.dumps({"task": "test-task", "status": "completed"})
        conn.execute(
            "INSERT INTO sse_event_log (project_id, event_type, payload_json, sequence_num) VALUES (?, ?, ?, ?)",
            ("proj-001", "task.completed", payload, 1),
        )
        conn.commit()

        rows = conn.execute(
            "SELECT event_type, payload_json FROM sse_event_log WHERE project_id = ?",
            ("proj-001",),
        ).fetchall()

        assert len(rows) == 1
        assert rows[0][0] == "task.completed"
        parsed = json.loads(rows[0][1])
        assert parsed["task"] == "test-task"


# ---------------------------------------------------------------------------
# Test 4 — conversation persist atomicity
# ---------------------------------------------------------------------------


class TestConversationWriteAtomicity:
    """Single-INSERT conversation writes are inherently atomic."""

    def test_conversation_message_persists(self, _db_path: str) -> None:
        """A message inserted into conversation_messages is fully readable."""
        from vetinari.database import get_connection

        conn = get_connection()
        conn.execute(
            "INSERT INTO conversation_messages "
            "(session_id, role, content, timestamp, metadata_json) "
            "VALUES (?, ?, ?, ?, ?)",
            ("sess-atomic", "user", "test message", 1000.0, "{}"),
        )
        conn.commit()

        rows = conn.execute(
            "SELECT role, content FROM conversation_messages WHERE session_id = ?",
            ("sess-atomic",),
        ).fetchall()

        assert len(rows) == 1
        assert rows[0][0] == "user"
        assert rows[0][1] == "test message"


# ---------------------------------------------------------------------------
# Test 5 — WAL mode
# ---------------------------------------------------------------------------


class TestDatabaseWALMode:
    """Verify WAL mode is enabled for crash safety."""

    def test_wal_mode_enabled(self, _db_path: str) -> None:
        """The database connection uses WAL journal mode."""
        from vetinari.database import get_connection

        conn = get_connection()
        result = conn.execute("PRAGMA journal_mode").fetchone()
        assert result[0] == "wal", f"Expected WAL mode, got {result[0]}"
