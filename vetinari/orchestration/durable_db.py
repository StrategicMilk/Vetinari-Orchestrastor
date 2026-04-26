"""Durable execution database layer — SQLite types and thread-safe wrapper.

Contains the data types and database manager used by DurableExecutionEngine:
- ExecutionEvent: immutable event record
- Checkpoint: mutable checkpoint snapshot
- _SCHEMA_SQL: DDL for standalone (test) databases
- _DatabaseManager: thread-safe SQLite wrapper with WAL mode

This module is separated from the main engine so each module stays within
the 550-line ceiling while keeping the DB layer independently testable.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# SQL schema for durable execution state persistence.
# Used in standalone (test) mode; production uses vetinari.database._UNIFIED_SCHEMA.
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
class ExecutionEventRecord:
    """An immutable event in the execution history.

    Attributes:
        event_id: Unique UUID for this event.
        event_type: Category string (e.g. task_started, task_completed, task_failed).
        task_id: The task this event relates to.
        timestamp: ISO-8601 UTC creation timestamp.
        data: Additional event payload.
    """

    event_id: str
    event_type: str
    task_id: str
    timestamp: str
    data: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ExecutionEventRecord(event_id={self.event_id!r}, event_type={self.event_type!r}, "
            f"task_id={self.task_id!r}, timestamp={self.timestamp!r})"
        )


@dataclass
class CheckpointSnapshot:
    """A snapshot of execution state for crash recovery.

    Attributes:
        checkpoint_id: Unique checkpoint identifier.
        plan_id: The plan this checkpoint belongs to.
        created_at: ISO-8601 creation timestamp.
        graph_state: Full serialized ExecutionGraph dict.
        completed_tasks: IDs of tasks that have finished.
        running_tasks: IDs of tasks currently in-flight.
        metadata: Optional extra data for debugging.
    """

    checkpoint_id: str
    plan_id: str
    created_at: str
    graph_state: dict[str, Any]
    completed_tasks: list[str]
    running_tasks: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"CheckpointSnapshot(checkpoint_id={self.checkpoint_id!r}, plan_id={self.plan_id!r}, "
            f"completed_tasks={len(self.completed_tasks)}, "
            f"running_tasks={len(self.running_tasks)})"
        )


class _DatabaseManager:
    """Thread-safe SQLite wrapper for durable execution checkpoints.

    When an explicit ``db_path`` is provided (e.g. in tests), uses a
    direct per-thread connection to that path.  When ``db_path`` is
    ``None``, delegates to the unified ``vetinari.database`` module so
    production data lands in the consolidated database (ADR-0072).

    All reads and writes go through a single shared connection protected
    by ``_write_lock`` to avoid ``database is locked`` errors from multiple
    concurrent threads hitting SQLite's single-writer lock.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path
        self._shared_conn: sqlite3.Connection | None = None
        # Write lock serializes INSERT/UPDATE across threads for this engine
        self._write_lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Return the shared SQLite connection, creating it if needed.

        Uses one connection with ``check_same_thread=False``, protected
        by the engine's ``_write_lock``. When ``_db_path is None``
        (production), connects to the unified database. When ``_db_path``
        is set (tests), connects to that specific file.

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

        For the unified database path, applies the full unified schema
        (including ``execution_events``). For standalone paths, applies
        only the durable execution schema from ``_SCHEMA_SQL``.
        """
        conn = self._get_conn()
        if self._db_path is None:
            from vetinari.database import _UNIFIED_SCHEMA

            conn.executescript(_UNIFIED_SCHEMA)
            conn.commit()
        else:
            conn.executescript(_SCHEMA_SQL)
            conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Execute a SQL statement with thread-safe write serialization.

        All operations go through the write lock to prevent concurrent
        threads hitting ``database is locked`` errors.

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

    def execute_in_transaction(self, statements: list[tuple[str, tuple]]) -> None:
        """Execute multiple SQL statements atomically.

        Raises:
            sqlite3.Error: If any statement fails and the transaction rolls back.
        """
        with self._write_lock:
            conn = self._get_conn()
            try:
                conn.execute("BEGIN")
                for sql, params in statements:
                    conn.execute(sql, params)
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
