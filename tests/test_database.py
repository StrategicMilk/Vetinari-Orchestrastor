"""Tests for vetinari.database — unified SQLite database module."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import threading

import pytest

from vetinari.database import (
    close_connection,
    execute_query,
    execute_write,
    get_connection,
    init_schema,
    reset_for_testing,
)


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Use a temp database for each test to ensure isolation."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("VETINARI_DB_PATH", str(db_path))
    reset_for_testing()
    yield
    reset_for_testing()


class TestGetConnection:
    """Tests for get_connection() thread-local connection management."""

    def test_returns_connection(self) -> None:
        """get_connection returns a valid sqlite3 Connection."""
        conn = get_connection()
        assert isinstance(conn, sqlite3.Connection)

    def test_same_thread_returns_same_connection(self) -> None:
        """Repeated calls on the same thread return the identical connection."""
        conn1 = get_connection()
        conn2 = get_connection()
        assert conn1 is conn2

    def test_different_threads_get_different_connections(self) -> None:
        """Different threads get their own connections."""
        main_conn = get_connection()
        other_conn = [None]

        def _worker():
            other_conn[0] = get_connection()
            close_connection()

        t = threading.Thread(target=_worker)
        t.start()
        t.join()

        assert other_conn[0] is not None
        assert other_conn[0] is not main_conn

    def test_wal_mode_enabled(self) -> None:
        """Connection uses WAL journal mode."""
        conn = get_connection()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_foreign_keys_enabled(self) -> None:
        """Connection has foreign keys enforcement enabled."""
        conn = get_connection()
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1


class TestInitSchema:
    """Tests for schema initialization."""

    def test_creates_all_tables(self) -> None:
        """init_schema creates the expected set of tables."""
        conn = get_connection()
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'"
            ).fetchall()
        }
        expected = {
            "execution_state",
            "task_checkpoints",
            "paused_questions",
            "execution_events",
            "quality_scores",
            "episodes",
            "alerts",
            "plans",
            "memory_entries",
            "training_data",
            "benchmark_results",
            "improvement_log",
            "adrs",
            "knowledge_chunks",
        }
        assert expected.issubset(tables), f"Missing tables: {expected - tables}"

    def test_idempotent(self) -> None:
        """Calling init_schema twice does not raise."""
        get_connection()
        # Schema already initialized by get_connection; call again manually
        reset_for_testing()
        conn2 = get_connection()  # Re-initializes
        init_schema(conn2)  # Should be no-op
        tables = conn2.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
        assert tables >= 14


class TestCloseConnection:
    """Tests for close_connection()."""

    def test_close_and_reconnect(self) -> None:
        """After close, get_connection creates a new connection."""
        conn1 = get_connection()
        close_connection()
        conn2 = get_connection()
        assert conn2 is not conn1

    def test_close_when_no_connection(self) -> None:
        """close_connection is a no-op if no connection exists."""
        close_connection()  # Ensure any existing connection is removed first
        close_connection()  # Second call: no connection exists, must not raise
        # After calling close on an already-closed state, a new get_connection
        # call must succeed — confirming the module is in a valid state.
        conn = get_connection()
        assert isinstance(conn, sqlite3.Connection)


class TestExecuteHelpers:
    """Tests for execute_query and execute_write convenience functions."""

    def test_write_and_read(self) -> None:
        """execute_write inserts data that execute_query can read back."""
        execute_write(
            "INSERT INTO alerts (alert_type, severity, message, source) VALUES (?, ?, ?, ?)",
            ("test", "info", "Hello from test", "test_suite"),
        )
        rows = execute_query("SELECT alert_type, message FROM alerts WHERE source = ?", ("test_suite",))
        assert len(rows) == 1
        assert rows[0]["alert_type"] == "test"
        assert rows[0]["message"] == "Hello from test"

    def test_write_returns_rowcount(self) -> None:
        """execute_write returns the number of affected rows."""
        count = execute_write(
            "INSERT INTO alerts (alert_type, severity, message) VALUES (?, ?, ?)",
            ("x", "warning", "y"),
        )
        assert count == 1


class TestEnvVar:
    """Tests for VETINARI_DB_PATH environment variable."""

    def test_custom_path(self, tmp_path, monkeypatch) -> None:
        """VETINARI_DB_PATH controls where the database is created."""
        custom = tmp_path / "custom" / "my.db"
        monkeypatch.setenv("VETINARI_DB_PATH", str(custom))
        reset_for_testing()
        get_connection()
        assert custom.exists()
