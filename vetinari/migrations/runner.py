"""Migration runner for Vetinari storage schemas.

Applies the plan-mode schema (``schema.sql``) and any future numbered
migration scripts to the SQLite databases used by subsystems.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import re
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_MIGRATIONS_DIR = Path(__file__).resolve().parent
_SCHEMA_FILE = _MIGRATIONS_DIR / "schema.sql"
_MIGRATION_FILENAME_RE = re.compile(r"^(?P<number>\d{3})_.*\.sql$")


def _run_script_transaction(conn: sqlite3.Connection, sql: str, trailer_sql: str = "") -> None:
    """Run a SQL script inside one SQLite transaction."""
    if trailer_sql and not trailer_sql.rstrip().endswith(";"):
        trailer_sql = f"{trailer_sql};"
    script = f"BEGIN IMMEDIATE;\n{sql}\n{trailer_sql}\nCOMMIT;"
    try:
        conn.executescript(script)
    except sqlite3.Error:
        with contextlib.suppress(sqlite3.Error):
            conn.execute("ROLLBACK")
        raise


def run_migrations(db_path: str | Path) -> int:
    """Initialise or upgrade the SQLite database at *db_path*.

    Applies ``schema.sql`` (idempotent ``CREATE TABLE IF NOT EXISTS``)
    and then any numbered migration files (``001_*.sql``, ``002_*.sql``, …)
    that have not yet been applied.

    Args:
        db_path: Filesystem path to the target SQLite database file.
            Created automatically if it does not exist.

    Returns:
        Number of migration files applied (0 when already up-to-date).

    Raises:
        sqlite3.Error: If schema initialization or a migration fails.
        OSError: If migration files cannot be read.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        # Ensure migration-tracking table exists
        conn.execute(
            "CREATE TABLE IF NOT EXISTS _migration_history ("
            "  filename TEXT PRIMARY KEY,"
            "  migration_id TEXT,"
            "  sha256 TEXT,"
            "  path TEXT,"
            "  applied_at TEXT NOT NULL DEFAULT (datetime('now'))"
            ")"
        )
        existing_history_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(_migration_history)").fetchall()
        }
        for column, definition in {
            "migration_id": "TEXT",
            "sha256": "TEXT",
            "path": "TEXT",
        }.items():
            if column not in existing_history_cols:
                conn.execute(f"ALTER TABLE _migration_history ADD COLUMN {column} {definition}")

        applied = 0

        # 1. Apply base schema (always idempotent)
        if _SCHEMA_FILE.exists():
            schema_sql = _SCHEMA_FILE.read_text(encoding="utf-8")
            _run_script_transaction(conn, schema_sql)
            logger.info("Base schema applied from %s", _SCHEMA_FILE.name)

        # 2. Apply numbered migration files in order
        migration_files = sorted(_MIGRATIONS_DIR.glob("[0-9][0-9][0-9]_*.sql"))
        already_applied = {
            row[0]: row[1]
            for row in conn.execute("SELECT filename, sha256 FROM _migration_history").fetchall()
        }
        previous_migration_number = 0

        for mfile in migration_files:
            match = _MIGRATION_FILENAME_RE.match(mfile.name)
            if match is None:
                logger.warning("Skipping migration with unexpected filename: %s", mfile.name)
                continue
            migration_number = int(match.group("number"))
            if migration_number != previous_migration_number + 1:
                logger.warning(
                    "Stopping migrations at %s because migration %03d is missing before %03d",
                    db_path,
                    previous_migration_number + 1,
                    migration_number,
                )
                break
            previous_migration_number = migration_number
            sql = mfile.read_text(encoding="utf-8")
            digest = hashlib.sha256(sql.encode("utf-8")).hexdigest()
            if mfile.name in already_applied:
                recorded_digest = already_applied[mfile.name]
                if recorded_digest and recorded_digest != digest:
                    raise RuntimeError(
                        f"Migration {mfile.name} has changed since it was applied; "
                        "refusing to treat filename-only history as clean."
                    )
                continue
            migration_id = f"{mfile.name}:{digest[:16]}"
            conn.create_function("_vetinari_migration_filename", 0, lambda name=mfile.name: name)
            conn.create_function("_vetinari_migration_id", 0, lambda value=migration_id: value)
            conn.create_function("_vetinari_migration_sha256", 0, lambda value=digest: value)
            conn.create_function(
                "_vetinari_migration_path",
                0,
                lambda value=str(mfile.resolve(strict=False)): value,
            )
            trailer = (
                "INSERT INTO _migration_history (filename, migration_id, sha256, path) "
                "VALUES (_vetinari_migration_filename(), _vetinari_migration_id(), "
                "_vetinari_migration_sha256(), _vetinari_migration_path())"
            )
            _run_script_transaction(conn, sql, trailer)
            applied += 1
            logger.info("Applied migration: %s", mfile.name)

        if applied == 0:
            logger.info("Database at %s is up-to-date", db_path)
        else:
            logger.info("Applied %d migration(s) to %s", applied, db_path)

        return applied
    finally:
        conn.close()
