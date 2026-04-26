"""Tests for vetinari/migrations/runner.py — migration runner utility."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

from vetinari.migrations.runner import run_migrations


class TestRunMigrations:
    """Tests for run_migrations()."""

    def test_creates_database_when_absent(self, tmp_path: Path) -> None:
        """run_migrations creates the database file when it does not exist."""
        db = tmp_path / "sub" / "new.db"
        run_migrations(db)
        assert db.exists()

    def test_returns_zero_when_no_migrations(self, tmp_path: Path) -> None:
        """run_migrations returns 0 when no numbered migration files are present."""
        db = tmp_path / "test.db"
        applied = run_migrations(db)
        assert applied == 0

    def test_creates_migration_history_table(self, tmp_path: Path) -> None:
        """run_migrations creates the _migration_history tracking table."""
        db = tmp_path / "test.db"
        run_migrations(db)
        conn = sqlite3.connect(str(db))
        try:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            assert "_migration_history" in tables
        finally:
            conn.close()

    def test_applies_numbered_migration_file(self, tmp_path: Path) -> None:
        """run_migrations applies a numbered .sql file and records it in history."""
        # Write a numbered migration file into a fake migrations dir
        mig_sql = tmp_path / "001_add_table.sql"
        mig_sql.write_text("CREATE TABLE IF NOT EXISTS test_tbl (id INTEGER PRIMARY KEY);", encoding="utf-8")

        # Use sys.modules so we patch the same module object that
        # run_migrations.__globals__ references (dual-module guard).
        runner_mod = sys.modules[run_migrations.__module__]
        original_dir = runner_mod._MIGRATIONS_DIR
        original_schema = runner_mod._SCHEMA_FILE
        runner_mod._MIGRATIONS_DIR = tmp_path
        runner_mod._SCHEMA_FILE = tmp_path / "schema.sql"  # doesn't exist, will be skipped

        try:
            db = tmp_path / "applied.db"
            applied = run_migrations(db)
            assert applied == 1

            # Verify migration is tracked
            conn = sqlite3.connect(str(db))
            try:
                rows = conn.execute("SELECT filename FROM _migration_history").fetchall()
                filenames = {r[0] for r in rows}
                assert "001_add_table.sql" in filenames
            finally:
                conn.close()
        finally:
            runner_mod._MIGRATIONS_DIR = original_dir
            runner_mod._SCHEMA_FILE = original_schema

    def test_idempotent_on_second_run(self, tmp_path: Path) -> None:
        """run_migrations returns 0 when called again with an up-to-date database."""
        runner_mod = sys.modules[run_migrations.__module__]
        original_dir = runner_mod._MIGRATIONS_DIR
        original_schema = runner_mod._SCHEMA_FILE
        mig_sql = tmp_path / "001_create.sql"
        mig_sql.write_text("CREATE TABLE IF NOT EXISTS idempotent_tbl (x TEXT);", encoding="utf-8")
        runner_mod._MIGRATIONS_DIR = tmp_path
        runner_mod._SCHEMA_FILE = tmp_path / "schema.sql"

        try:
            db = tmp_path / "idem.db"
            first = run_migrations(db)
            second = run_migrations(db)
            assert first == 1
            assert second == 0  # already applied
        finally:
            runner_mod._MIGRATIONS_DIR = original_dir
            runner_mod._SCHEMA_FILE = original_schema

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """run_migrations accepts a string db_path, not just a Path object."""
        db = str(tmp_path / "string_path.db")
        applied = run_migrations(db)
        assert applied == 0
        assert Path(db).exists()

    def test_base_schema_applied_when_present(self, tmp_path: Path) -> None:
        """run_migrations applies schema.sql when it exists."""
        schema_sql = tmp_path / "schema.sql"
        schema_sql.write_text(
            "CREATE TABLE IF NOT EXISTS plans (id TEXT PRIMARY KEY, status TEXT);",
            encoding="utf-8",
        )

        # Use __globals__ so we patch the module run_migrations actually references,
        # not whatever sys.modules holds after _restore_sys_modules cleanup.
        runner_mod = sys.modules[run_migrations.__module__]
        original_dir = runner_mod._MIGRATIONS_DIR
        original_schema = runner_mod._SCHEMA_FILE
        runner_mod._MIGRATIONS_DIR = tmp_path
        runner_mod._SCHEMA_FILE = schema_sql

        try:
            db = tmp_path / "schema_applied.db"
            run_migrations(db)
            conn = sqlite3.connect(str(db))
            try:
                tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                assert "plans" in tables
            finally:
                conn.close()
        finally:
            runner_mod._MIGRATIONS_DIR = original_dir
            runner_mod._SCHEMA_FILE = original_schema
