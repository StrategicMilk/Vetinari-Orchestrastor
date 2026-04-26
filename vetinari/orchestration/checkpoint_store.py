"""Checkpoint persistence layer for durable execution.

Provides the SQLite-backed storage primitives used by ``DurableExecutionEngine``
to survive crashes and resume interrupted plans.

Pipeline role: Plan -> DurableExecution -> **CheckpointStore** (persist) -> Verify -> Learn.
This is the storage half of the durable execution system; the engine half lives in
``durable_execution.py``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from vetinari.constants import _PROJECT_ROOT

logger = logging.getLogger(__name__)


# SQL schema for durable execution state persistence
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS execution_state (
    execution_id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    tier TEXT NOT NULL DEFAULT 'standard',
    request_spec_json TEXT,
    pipeline_state TEXT NOT NULL DEFAULT 'executing',
    current_agent TEXT,
    task_dag_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    terminal_status TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS task_checkpoints (
    task_id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL,
    agent_type TEXT NOT NULL DEFAULT '',
    mode TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    input_json TEXT,
    output_json TEXT,
    manifest_hash TEXT,
    started_at TEXT,
    completed_at TEXT,
    retry_count INTEGER DEFAULT 0,
    FOREIGN KEY (execution_id) REFERENCES execution_state(execution_id)
);

CREATE TABLE IF NOT EXISTS paused_questions (
    question_id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL,
    task_id TEXT,
    questions_json TEXT NOT NULL,
    answers_json TEXT,
    asked_at TEXT NOT NULL,
    answered_at TEXT,
    FOREIGN KEY (execution_id) REFERENCES execution_state(execution_id)
);

CREATE TABLE IF NOT EXISTS execution_events (
    event_id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    task_id TEXT,
    timestamp TEXT NOT NULL,
    data_json TEXT,
    FOREIGN KEY (execution_id) REFERENCES execution_state(execution_id)
);
"""


@dataclass(frozen=True)
class ExecutionEvent:
    """An immutable record of a single state transition in execution history."""

    event_id: str
    event_type: str  # task_started, task_completed, task_failed, etc.
    task_id: str
    timestamp: str
    data: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ExecutionEvent(event_id={self.event_id!r}, event_type={self.event_type!r}, "
            f"task_id={self.task_id!r}, timestamp={self.timestamp!r})"
        )


@dataclass
class Checkpoint:
    """A point-in-time snapshot of an in-progress execution graph."""

    checkpoint_id: str
    plan_id: str
    created_at: str
    graph_state: dict[str, Any]
    completed_tasks: list[str]
    running_tasks: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Checkpoint(checkpoint_id={self.checkpoint_id!r}, plan_id={self.plan_id!r}, "
            f"completed_tasks={len(self.completed_tasks)}, "
            f"running_tasks={len(self.running_tasks)})"
        )


class _DatabaseManager:
    """Thread-safe SQLite wrapper for durable execution checkpoints.

    When an explicit ``db_path`` is provided (e.g. in tests), uses a
    direct per-thread connection to that path.  When ``db_path`` is
    ``None``, delegates to the unified ``vetinari.database`` module so
    production data lands in the consolidated database (ADR-0072).
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path
        self._shared_conn: sqlite3.Connection | None = None  # For standalone path mode
        # Write lock serializes INSERT/UPDATE across threads for this engine
        self._write_lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a single shared SQLite connection for this engine.

        Uses one connection with ``check_same_thread=False``, protected
        by the engine's ``_write_lock``. This avoids ``database is locked``
        errors from multiple thread-local connections competing for
        SQLite's single-writer lock.

        When ``_db_path is None`` (production), connects to the unified
        database at ``VETINARI_DB_PATH``. When ``_db_path`` is set (tests),
        connects to that specific file.

        Returns:
            A ``sqlite3.Connection`` with WAL mode enabled.
        """
        if self._shared_conn is not None:
            return self._shared_conn

        if self._db_path is None:
            from vetinari.database import _get_db_path

            db_path = _get_db_path()
        else:
            db_path = self._db_path

        db_path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None enables autocommit mode so Python's sqlite3 module
        # does NOT issue implicit BEGIN before DML statements.  Without this,
        # execute_in_transaction's explicit BEGIN raises "cannot start a transaction
        # within a transaction" because Python's deferred transaction is already open.
        self._shared_conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            timeout=30.0,
            isolation_level=None,
        )
        self._shared_conn.execute("PRAGMA journal_mode=WAL")
        self._shared_conn.execute("PRAGMA synchronous=NORMAL")
        self._shared_conn.execute("PRAGMA foreign_keys=ON")
        self._shared_conn.execute("PRAGMA busy_timeout=5000")
        return self._shared_conn

    def _init_db(self) -> None:
        """Initialise the schema in the target database.

        For the unified database path, ensures the shared connection is
        created and the full unified schema (including ``execution_events``)
        is applied. For standalone paths, applies only the durable execution
        schema from ``_SCHEMA_SQL``.
        """
        conn = self._get_conn()
        if self._db_path is None:
            # Production: apply the full unified schema which includes
            # execution_events, quality_scores, etc.
            from vetinari.database import _UNIFIED_SCHEMA

            conn.executescript(_UNIFIED_SCHEMA)
            conn.commit()
        else:
            # Tests: apply only the durable execution schema
            conn.executescript(_SCHEMA_SQL)
            conn.commit()

        # Migrate pre-existing databases that lack terminal_status.
        # ALTER TABLE ADD COLUMN is a no-op if the column already exists in
        # SQLite >=3.37. For older SQLite, we catch OperationalError and ignore
        # it — the column not existing means we're on a fresh db that already
        # has it from the CREATE TABLE above.
        try:
            conn.execute("ALTER TABLE execution_state ADD COLUMN terminal_status TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            # Column already exists — migration already applied, nothing to do.
            logger.info("terminal_status column already present in execution_state; skipping migration")

    def execute(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Execute a SQL statement with thread-safe write serialization.

        All operations go through the write lock to prevent concurrent
        thread-local connections from hitting ``database is locked``
        errors. This serializes writes across all threads that share
        this ``_DatabaseManager`` instance.

        Args:
            sql: SQL statement to execute.
            params: Parameters for the SQL statement.

        Returns:
            List of result rows.
        """
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            if sql.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "DROP")):
                conn.commit()
            return rows

    def executemany(self, sql: str, params_list: list[tuple]) -> None:
        """Execute a SQL statement for many parameter sets.

        Args:
            sql: SQL statement to execute.
            params_list: List of parameter tuples.
        """
        with self._write_lock:
            conn = self._get_conn()
            conn.executemany(sql, params_list)
            conn.commit()

    def execute_in_transaction(
        self,
        statements: list[tuple[str, tuple]],
        many_statements: list[tuple[str, list[tuple]]] | None = None,
    ) -> None:
        """Execute multiple SQL statements atomically in a single transaction.

        All statements are committed together or rolled back together if any
        fails.  This is the correct primitive for multi-table writes that must
        be crash-safe: either all rows land or none do.

        Args:
            statements: List of ``(sql, params)`` pairs executed with
                ``conn.execute()``.
            many_statements: Optional list of ``(sql, params_list)`` pairs
                executed with ``conn.executemany()``.

        Raises:
            sqlite3.Error: Re-raised after rolling back if any statement fails.
        """
        with self._write_lock:
            conn = self._get_conn()
            try:
                conn.execute("BEGIN")
                for sql, params in statements:
                    conn.execute(sql, params)
                for sql, params_list in many_statements or []:  # noqa: VET112 — param is list | None
                    conn.executemany(sql, params_list)
                conn.execute("COMMIT")
            except sqlite3.Error:
                conn.execute("ROLLBACK")
                raise

    def close(self) -> None:
        """Close the database connection.

        When using the unified database (``_db_path is None``), delegates
        to ``vetinari.database.close_connection()``. When using a standalone
        path (tests), closes the shared connection directly.
        """
        if self._shared_conn is not None:
            self._shared_conn.close()
            self._shared_conn = None
        if self._db_path is None:
            from vetinari.database import close_connection

            close_connection()


class CheckpointStore:
    """Persistence facade for durable execution state.

    Wraps ``_DatabaseManager`` and exposes higher-level methods for saving,
    loading, and querying execution checkpoints, events, and paused questions.
    ``DurableExecutionEngine`` owns one of these and calls it on every state
    transition to ensure crash-safe durability.

    Args:
        checkpoint_dir: Optional directory path for standalone SQLite databases
            (used in tests). When ``None``, production code stores data in the
            consolidated ``vetinari.database`` database (ADR-0072).
    """

    def __init__(self, checkpoint_dir: Path | None = None) -> None:
        if checkpoint_dir is None:
            db_path = None
        else:
            db_path = checkpoint_dir / "execution_state.db"
        self._db = _DatabaseManager(db_path)

    # ------------------------------------------------------------------
    # Event persistence
    # ------------------------------------------------------------------

    def save_event(self, event: ExecutionEvent, execution_id: str = "") -> None:
        """Persist an execution event to SQLite for audit trail and crash recovery.

        Failures are logged at WARNING and swallowed — event persistence must
        not interrupt task execution.

        Args:
            event: The event to persist.
            execution_id: Optional execution ID for foreign key linkage.
        """
        try:
            self._db.execute(
                """INSERT OR IGNORE INTO execution_events
                   (event_id, execution_id, event_type, task_id, timestamp, data_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    event.event_id,
                    execution_id,
                    event.event_type,
                    event.task_id,
                    event.timestamp,
                    json.dumps(event.data),
                ),
            )
        except Exception:
            logger.warning("Failed to persist event %s to SQLite", event.event_id, exc_info=True)

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        plan_id: str,
        graph_dict: dict[str, Any],
        pipeline_state: str,
        task_rows: list[tuple],
        now: str,
        completed_at: str | None = None,
        terminal_status: str | None = None,
    ) -> None:
        """Persist a plan's execution graph state to SQLite (atomic, crash-safe).

        When the plan has reached a terminal state, pass ``completed_at`` and
        ``terminal_status`` so retention and cleanup queries can identify and
        act on finished executions.  Both default to ``None`` for in-progress
        checkpoints so partially-complete rows are never mistaken for finished ones.

        Args:
            plan_id: The plan identifier.
            graph_dict: Serialized graph dict from ``ExecutionGraph.to_dict()``.
            pipeline_state: Current pipeline state string (e.g. ``"executing"``).
            task_rows: Pre-built parameter tuples for the ``task_checkpoints`` upsert.
            now: ISO-format UTC timestamp for ``updated_at``.
            completed_at: ISO-format UTC timestamp to write as ``completed_at`` when
                terminal; ``None`` for in-progress checkpoints.
            terminal_status: Enum value string (e.g. ``"completed"``) to write when
                terminal; ``None`` for in-progress checkpoints.
        """
        state_sql = (
            """INSERT INTO execution_state
                   (execution_id, goal, pipeline_state, task_dag_json, created_at, updated_at,
                    completed_at, terminal_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(execution_id) DO UPDATE SET
                   pipeline_state = excluded.pipeline_state,
                   task_dag_json = excluded.task_dag_json,
                   updated_at = excluded.updated_at,
                   completed_at = excluded.completed_at,
                   terminal_status = excluded.terminal_status""",
            (
                plan_id,
                graph_dict.get("goal", ""),
                pipeline_state,
                json.dumps(graph_dict),
                graph_dict.get("created_at", now),
                now,
                completed_at,
                terminal_status,
            ),
        )
        checkpoint_many = (
            (
                """INSERT INTO task_checkpoints
                   (task_id, execution_id, agent_type, mode, status, input_json, output_json, manifest_hash, started_at, completed_at, retry_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(task_id) DO UPDATE SET
                       status = excluded.status,
                       output_json = excluded.output_json,
                       completed_at = excluded.completed_at,
                       retry_count = excluded.retry_count""",
                task_rows,
            )
            if task_rows
            else None
        )
        self._db.execute_in_transaction(
            statements=[state_sql],
            many_statements=[checkpoint_many] if checkpoint_many else None,
        )

    def load_checkpoint_graph_json(self, plan_id: str) -> str | None:
        """Load the raw serialized graph JSON for a plan from SQLite.

        Args:
            plan_id: Plan identifier to query.

        Returns:
            JSON string if a checkpoint exists, or None.
        """
        rows = self._db.execute(
            "SELECT task_dag_json FROM execution_state WHERE execution_id = ?",
            (plan_id,),
        )
        if not rows or not rows[0][0]:
            return None
        return rows[0][0]

    def list_checkpoint_ids(self) -> list[str]:
        """Return all plan IDs that have persisted checkpoints.

        Returns:
            Sorted list of execution IDs.
        """
        rows = self._db.execute("SELECT execution_id FROM execution_state")
        return sorted({r[0] for r in rows})

    def find_incomplete_ids(self, completed_state: str, failed_state: str) -> list[str]:
        """Query for execution IDs whose pipeline state is neither completed nor failed.

        Args:
            completed_state: The string value of the completed status enum.
            failed_state: The string value of the failed status enum.

        Returns:
            List of plan IDs pending recovery.
        """
        rows = self._db.execute(
            "SELECT execution_id FROM execution_state WHERE pipeline_state NOT IN (?, ?)",
            (completed_state, failed_state),
        )
        return [r[0] for r in rows]

    def find_completed_before(self, cutoff_iso: str) -> list[str]:
        """Find execution IDs that completed at or before a cutoff timestamp.

        Args:
            cutoff_iso: ISO-format UTC timestamp; executions completed no later than
                this value are returned.

        Returns:
            List of execution IDs eligible for cleanup.
        """
        rows = self._db.execute(
            "SELECT execution_id FROM execution_state WHERE completed_at IS NOT NULL AND completed_at <= ?",
            (cutoff_iso,),
        )
        return [r[0] for r in rows]

    def list_retention_candidates(self, older_than_seconds: float) -> list[str]:
        """Return execution IDs that finished more than *older_than_seconds* ago.

        Queries for rows where ``completed_at`` is set and older than the
        derived cutoff.  Only rows that have a ``terminal_status`` value (i.e.
        rows written by the updated ``save_checkpoint``) are returned, so
        in-progress executions whose ``completed_at`` is NULL are never
        included.

        Args:
            older_than_seconds: Age threshold in seconds.  Executions whose
                ``completed_at`` timestamp is older than this are candidates
                for deletion.

        Returns:
            List of execution IDs eligible for retention cleanup.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=older_than_seconds)).isoformat()
        return self.find_completed_before(cutoff)

    def delete_execution(self, exec_id: str) -> None:
        """Remove all persisted state for a completed execution.

        Args:
            exec_id: The execution ID to purge from the database.
        """
        self._db.execute_in_transaction(
            statements=[
                ("DELETE FROM execution_events WHERE execution_id = ?", (exec_id,)),
                ("DELETE FROM task_checkpoints WHERE execution_id = ?", (exec_id,)),
                ("DELETE FROM paused_questions WHERE execution_id = ?", (exec_id,)),
                ("DELETE FROM execution_state WHERE execution_id = ?", (exec_id,)),
            ],
        )

    # ------------------------------------------------------------------
    # Pause / resume for clarification questions
    # ------------------------------------------------------------------

    def save_paused_questions(
        self,
        execution_id: str,
        questions: list[str],
        task_id: str | None = None,
    ) -> str:
        """Persist a set of questions that require user input before execution can resume.

        Inserts a stub ``execution_state`` row if one does not already exist so
        the foreign key constraint is satisfied.

        Args:
            execution_id: Execution being paused.
            questions: Questions to surface to the user.
            task_id: Optional task that triggered the pause.

        Returns:
            A unique ``question_id`` for later answer retrieval.
        """
        question_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self._db.execute(
            """INSERT OR IGNORE INTO execution_state
               (execution_id, goal, pipeline_state, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (execution_id, "", "paused", now, now),
        )
        self._db.execute(
            """INSERT INTO paused_questions (question_id, execution_id, task_id, questions_json, asked_at)
               VALUES (?, ?, ?, ?, ?)""",
            (question_id, execution_id, task_id, json.dumps(questions), now),
        )
        return question_id

    def answer_paused_questions(self, question_id: str, answers: list[str]) -> None:
        """Record user answers for a paused question set, enabling pipeline resume.

        Args:
            question_id: The question set identifier.
            answers: User-provided answers in order.
        """
        now = datetime.now(timezone.utc).isoformat()
        self._db.execute(
            "UPDATE paused_questions SET answers_json = ?, answered_at = ? WHERE question_id = ?",
            (json.dumps(answers), now, question_id),
        )

    def get_paused_questions(self, execution_id: str) -> list[dict[str, Any]]:
        """Retrieve all paused question sets for an execution.

        Args:
            execution_id: Execution to query.

        Returns:
            List of question dicts with ``question_id``, ``task_id``,
            ``questions``, ``answers``, ``asked_at``, and ``answered_at``.
        """
        rows = self._db.execute(
            "SELECT question_id, task_id, questions_json, answers_json, asked_at, answered_at "
            "FROM paused_questions WHERE execution_id = ?",
            (execution_id,),
        )
        return [
            {
                "question_id": r[0],
                "task_id": r[1],
                "questions": json.loads(r[2]),
                "answers": json.loads(r[3]) if r[3] else None,
                "asked_at": r[4],
                "answered_at": r[5],
            }
            for r in rows
        ]

    def close(self) -> None:
        """Close the underlying database connection."""
        self._db.close()


# -- Module-level constant for the default checkpoint directory ----------------
#    Used by DurableExecutionEngine when no explicit directory is provided.
DEFAULT_CHECKPOINT_DIR: Path = _PROJECT_ROOT / "vetinari_checkpoints"
