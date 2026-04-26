"""Unified SQLite database module — single connection pool for all stores.

Consolidates 10+ separate SQLite databases into one unified database with
thread-local connections, WAL mode, and centralized schema initialization.
Credential vault remains encrypted and separate (~/.vetinari/vault/).

Usage:
    from vetinari.database import get_connection, init_schema

    conn = get_connection()  # Thread-local, WAL mode, auto-initialized
    cursor = conn.execute("SELECT * FROM quality_scores WHERE task_id = ?", (task_id,))

Environment:
    VETINARI_DB_PATH: Path to the unified database file.
        Defaults to ``<PROJECT_ROOT>/.vetinari/vetinari.db``.

Decision: Consolidate 10 SQLite stores into 1 (ADR-0072).
"""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

from vetinari.constants import _PROJECT_ROOT, get_user_dir
from vetinari.database_schema import _UNIFIED_SCHEMA

logger = logging.getLogger(__name__)
_safe_log_lock = threading.Lock()

# ── Default database location ────────────────────────────────────────────────
_DEFAULT_DB_DIR = _PROJECT_ROOT / ".vetinari"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "vetinari.db"

# Thread-local storage for connections
_thread_local = threading.local()


def _reachable_handlers(log: logging.Logger) -> list[logging.Handler]:
    """Return handlers that may receive records from *log* via propagation."""
    handlers: list[logging.Handler] = []
    current: logging.Logger | None = log
    while current is not None:
        for handler in current.handlers:
            if handler not in handlers:
                handlers.append(handler)
        if not current.propagate:
            break
        current = current.parent
    if logging.lastResort is not None and logging.lastResort not in handlers:
        handlers.append(logging.lastResort)
    return handlers


def _safe_log(level: int, message: str, *args: object) -> None:
    """Log non-critical database lifecycle messages without teardown noise."""
    if level < logging.WARNING:
        return
    with _safe_log_lock:
        handlers = _reachable_handlers(logger)
        original_handle_errors = [
            (handler, "handleError" in handler.__dict__, handler.__dict__.get("handleError"))
            for handler in handlers
        ]
        try:
            for handler, _had_instance_override, _original in original_handle_errors:
                handler.handleError = lambda _record: None  # type: ignore[method-assign]
            with contextlib.suppress(OSError, ValueError):
                logger.log(level, message, *args)
        finally:
            for handler, had_instance_override, original in original_handle_errors:
                if had_instance_override:
                    handler.handleError = original  # type: ignore[method-assign]
                else:
                    with contextlib.suppress(AttributeError):
                        del handler.handleError


# Lock for schema initialization (one-time operation)
_schema_init_lock = threading.Lock()
_schema_initialized = False
_schema_initialized_paths: set[Path] = set()


def _get_db_path() -> Path:
    """Resolve the database file path from env var or default.

    Returns:
        Absolute path to the unified SQLite database file.
    """
    env_path = os.environ.get("VETINARI_DB_PATH")
    if env_path:
        return Path(env_path)
    return get_user_dir() / "vetinari.db"


def _schema_path_key() -> Path:
    """Return a stable key for the currently configured physical DB path."""
    return _get_db_path().expanduser().resolve(strict=False)


def get_connection() -> sqlite3.Connection:
    """Return a thread-local SQLite connection to the unified database.

    Creates the connection on first call per thread, sets WAL mode,
    and ensures the schema is initialized. Subsequent calls on the
    same thread return the cached connection. If VETINARI_DB_PATH changes,
    closes and recreates the connection to the new path.

    Returns:
        A ``sqlite3.Connection`` with WAL mode and row factory enabled.
    """
    conn: sqlite3.Connection | None = getattr(_thread_local, "connection", None)
    cached_db_path = getattr(_thread_local, "db_path", None)
    db_path = _get_db_path()

    # If we have a cached connection but the DB path changed, close and reconnect
    if conn is not None and cached_db_path is not None and cached_db_path != db_path:
        _safe_log(
            logging.WARNING,
            "Database path changed from %s to %s — closing and reconnecting",
            cached_db_path,
            db_path,
        )
        conn.close()
        conn = None
        _thread_local.connection = None
        _thread_local.db_path = None

    if conn is not None:
        return conn

    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA wal_autocheckpoint=1000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")  # 5 second wait on lock contention
    conn.execute("PRAGMA cache_size=-32768")  # 32 MB page cache
    conn.execute("PRAGMA mmap_size=268435456")  # 256 MB memory-mapped I/O
    conn.row_factory = sqlite3.Row

    _thread_local.connection = conn
    _thread_local.db_path = db_path

    # Ensure schema is created (thread-safe, one-time)
    init_schema(conn)

    _safe_log(
        logging.DEBUG,
        "Database connection established for thread %s at %s",
        threading.current_thread().name,
        db_path,
    )
    return conn


def close_connection() -> None:
    """Close the thread-local connection if it exists.

    Call this during thread shutdown or test cleanup. The next call
    to ``get_connection()`` on this thread will create a fresh connection.
    """
    conn: sqlite3.Connection | None = getattr(_thread_local, "connection", None)
    if conn is not None:
        conn.close()
        _thread_local.connection = None
        _safe_log(logging.DEBUG, "Database connection closed for thread %s", threading.current_thread().name)


# ── Unified schema ───────────────────────────────────────────────────────────
def _schema_migration_statements(conn: sqlite3.Connection) -> list[str]:
    """Return startup schema repair statements for the current database.

    ``CREATE TABLE IF NOT EXISTS`` silently skips creation when a table
    already exists with the old column set, then subsequent ``CREATE INDEX``
    statements fail because expected columns are missing. The returned
    statements are executed in the same transaction as schema creation.
    """
    statements: list[str] = []
    # Map: table_name -> column that MUST exist in the current schema.
    # If the column is missing the table predates the unified schema and
    # must be recreated.
    _required_columns: dict[str, str] = {
        "benchmark_results": "run_id",
        "benchmark_runs": "suite_name",
        # defect_occurrences was added in session-2B — drop and recreate if stale
        "defect_occurrences": "occurred_at",
        "improvements": "hypothesis",
        "improvement_observations": "observation_id",
    }
    for table, required_col in _required_columns.items():
        try:
            cursor = conn.execute(f"PRAGMA table_info({table})")
            columns = {row[1] for row in cursor.fetchall()}
        except sqlite3.OperationalError:
            logger.warning("Table %s not yet created; skipping column migration check", table)
            continue  # table doesn't exist yet — nothing to migrate
        if columns and required_col not in columns:
            _safe_log(
                logging.INFO,
                "Migrating stale table %s (missing column %s) — dropping and recreating",
                table,
                required_col,
            )
            statements.append(f"DROP TABLE IF EXISTS {table};")

    # Additive column migrations: ALTER TABLE ADD COLUMN when column absent.
    # Used when recreating the table would lose data (memories, memory_episodes).
    _additive_columns: dict[str, list[tuple[str, str]]] = {
        "memories": [
            ("scope", "TEXT NOT NULL DEFAULT 'global'"),
            ("recall_count", "INTEGER DEFAULT 0"),
            ("supersedes_id", "TEXT"),
            ("relationship_type", "TEXT"),
            ("last_accessed", "INTEGER DEFAULT 0"),
        ],
        "embeddings": [("dimensions", "INTEGER NOT NULL DEFAULT 0")],
        "memory_episodes": [("scope", "TEXT NOT NULL DEFAULT 'global'")],
        "execution_state": [("terminal_status", "TEXT")],
        "PlanHistory": [("plan_explanation_json", "TEXT")],
        "SubtaskMemory": [
            ("subtask_explanation_json", "TEXT"),
            ("quality_score", "REAL DEFAULT 0.0"),
        ],
    }
    for table, cols in _additive_columns.items():
        try:
            cursor = conn.execute(f"PRAGMA table_info({table})")
            existing = {row[1] for row in cursor.fetchall()}
        except sqlite3.OperationalError:
            logger.warning("Table %s not yet created; skipping additive column migration", table)
            continue  # table doesn't exist yet — CREATE TABLE below will add it with the column
        for col_name, col_def in cols:
            if existing and col_name not in existing:
                _safe_log(logging.INFO, "Adding column %s to %s", col_name, table)
                statements.append(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def};")
    return statements


def init_schema(conn: sqlite3.Connection | None = None) -> None:
    """Create all tables in the unified schema if they don't exist.

    Thread-safe — only executes once per process. Subsequent calls are no-ops.

    Args:
        conn: Connection to use. Required on first call (provided by
            ``get_connection()`` automatically). If None and schema is
            already initialized, this is a no-op.

    Raises:
        sqlite3.Error: If schema migration or creation fails.
    """
    global _schema_initialized
    schema_key = _schema_path_key()
    if schema_key in _schema_initialized_paths:
        return

    with _schema_init_lock:
        if schema_key in _schema_initialized_paths:
            return

        if conn is None:
            # Schema not initialized and no connection provided — caller
            # must use get_connection() which will pass conn to us.
            return

        migration_sql = "\n".join(_schema_migration_statements(conn))
        script = f"BEGIN IMMEDIATE;\n{migration_sql}\n{_UNIFIED_SCHEMA}\nCOMMIT;"
        try:
            conn.executescript(script)
        except sqlite3.Error:
            with contextlib.suppress(sqlite3.Error):
                conn.execute("ROLLBACK")
            raise
        _schema_initialized_paths.add(schema_key)
        _schema_initialized = True
        _safe_log(logging.INFO, "Unified database schema initialized at %s", schema_key)


def reset_for_testing() -> None:
    """Reset module state for test isolation.

    Closes any thread-local connection and resets the schema-initialized
    flag so the next ``get_connection()`` call creates a fresh database.
    """
    global _schema_initialized
    close_connection()
    with _schema_init_lock:
        _schema_initialized = False
        _schema_initialized_paths.clear()


def execute_query(sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    """Execute a read query and return all rows.

    Convenience wrapper that acquires the thread-local connection,
    executes the query, and returns results.

    Args:
        sql: SQL SELECT statement.
        params: Query parameters.

    Returns:
        List of sqlite3.Row objects (dict-like access by column name).
    """
    conn = get_connection()
    cursor = conn.execute(sql, params)
    return cursor.fetchall()


def execute_write(sql: str, params: tuple[Any, ...] = ()) -> int:
    """Execute a write statement (INSERT/UPDATE/DELETE) and return rowcount.

    Commits the transaction after execution.

    Args:
        sql: SQL write statement.
        params: Statement parameters.

    Returns:
        Number of rows affected.
    """
    conn = get_connection()
    cursor = conn.execute(sql, params)
    conn.commit()
    return cursor.rowcount
