#!/usr/bin/env python3
"""Migrate legacy SQLite databases to the unified vetinari.db.

Consolidates 10+ separate SQLite stores into a single database with
foreign keys and proper indexing. The credential vault remains encrypted
and separate. ADR JSON files are kept as human-readable backups.

Usage:
    python scripts/maintenance/migrate_to_unified_db.py              # Full migration
    python scripts/maintenance/migrate_to_unified_db.py --dry-run    # Preview only
    python scripts/maintenance/migrate_to_unified_db.py --backup     # Backup before migrate

Environment:
    VETINARI_DB_PATH: Target unified database (default: .vetinari/vetinari.db)
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from vetinari.database import get_connection, init_schema  # noqa: E402 - late import is required after bootstrap setup

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Known legacy database locations relative to project root
_LEGACY_STORES: dict[str, list[str]] = {
    "durable_execution": [
        ".vetinari/durable_execution.db",
        ".vetinari/executions.db",
    ],
    "quality_scores": [
        ".vetinari/quality_scores.db",
        ".vetinari/quality.db",
    ],
    "episode_memory": [
        ".vetinari/episodes.db",
        ".vetinari/episode_memory.db",
    ],
    "alerts": [
        ".vetinari/alerts.db",
        ".vetinari/dashboard_alerts.db",
    ],
    "plan_tracking": [
        ".vetinari/memory/plans.db",
        ".vetinari/vetinari_memory.db",
    ],
    "unified_memory": [
        ".vetinari/memory/unified.db",
    ],
    "benchmarks": [
        ".vetinari/benchmarks.db",
    ],
    "improvement_log": [
        ".vetinari/kaizen.db",
        ".vetinari/improvement_log.db",
    ],
    "knowledge_base": [
        ".vetinari/rag/knowledge.db",
    ],
}

# Legacy env vars that will be consolidated into VETINARI_DB_PATH
_LEGACY_ENV_VARS = [
    "VETINARI_QUALITY_DB",
    "VETINARI_EPISODE_DB",
    "VETINARI_ALERTS_DB",
    "VETINARI_MEMORY_DB",
    "VETINARI_EXECUTION_DB",
    "VETINARI_BENCHMARK_DB",
    "VETINARI_RAG_DB",
]


def _find_legacy_dbs() -> dict[str, Path]:
    """Find existing legacy database files on disk.

    Returns:
        Dict mapping store name to the first found path.
    """
    found: dict[str, Path] = {}
    for store_name, candidates in _LEGACY_STORES.items():
        for candidate in candidates:
            path = _PROJECT_ROOT / candidate
            if path.exists():
                found[store_name] = path
                break
    return found


def _migrate_table(
    source_conn: sqlite3.Connection,
    target_conn: sqlite3.Connection,
    source_table: str,
    target_table: str,
    *,
    dry_run: bool = False,
) -> int:
    """Migrate rows from a source table to the target unified table.

    Uses INSERT OR IGNORE for idempotency — safe to re-run.

    Args:
        source_conn: Connection to the legacy database.
        target_conn: Connection to the unified database.
        source_table: Table name in the source database.
        target_table: Table name in the target database.
        dry_run: If True, count rows but don't copy.

    Returns:
        Number of rows migrated (or would be migrated in dry-run).
    """
    try:
        source_columns = [row[1] for row in source_conn.execute(f"PRAGMA table_info({source_table})").fetchall()]
    except sqlite3.OperationalError:
        logger.warning("  Source table %s does not exist, skipping", source_table)
        return 0

    # Find matching columns in target
    try:
        target_columns = [row[1] for row in target_conn.execute(f"PRAGMA table_info({target_table})").fetchall()]
    except sqlite3.OperationalError:
        logger.warning("  Target table %s does not exist, skipping", target_table)
        return 0

    common_columns = [c for c in source_columns if c in target_columns]
    if not common_columns:
        logger.warning("  No common columns between %s and %s", source_table, target_table)
        return 0

    rows = source_conn.execute(
        f"SELECT {', '.join(common_columns)} FROM {source_table}"  # noqa: S608 - SQL identifiers are constrained while values stay parameterized
    ).fetchall()

    if dry_run:
        logger.info("  [DRY RUN] Would migrate %d rows from %s -> %s", len(rows), source_table, target_table)
        return len(rows)

    if not rows:
        return 0

    placeholders = ", ".join(["?"] * len(common_columns))
    col_names = ", ".join(common_columns)
    target_conn.executemany(
        f"INSERT OR IGNORE INTO {target_table} ({col_names}) VALUES ({placeholders})",  # noqa: S608 - SQL identifiers are constrained while values stay parameterized
        rows,
    )
    target_conn.commit()
    logger.info("  Migrated %d rows: %s -> %s", len(rows), source_table, target_table)
    return len(rows)


def migrate_all(
    *, dry_run: bool = False, backup: bool = False
) -> tuple[dict[str, int], list[str]]:
    """Run the full migration from all legacy stores.

    Args:
        dry_run: Preview mode — report what would be migrated.
        backup: Create backups of legacy DBs before migration.

    Returns:
        Tuple of (results, errors) where ``results`` maps store name to rows
        migrated and ``errors`` is a list of store names that failed.  A
        non-empty ``errors`` list means the caller should exit nonzero.
    """
    results: dict[str, int] = {}
    errors: list[str] = []

    if dry_run:
        logger.info("=== DRY RUN — no data will be written ===")

    # Ensure target schema exists
    target_conn = get_connection()
    init_schema(target_conn)

    legacy_dbs = _find_legacy_dbs()
    if not legacy_dbs:
        logger.info("No legacy databases found — nothing to migrate")
        return results, errors

    logger.info("Found %d legacy database(s):", len(legacy_dbs))
    for name, path in legacy_dbs.items():
        logger.info("  %s: %s", name, path)

    # Table mapping: source table -> target table
    _table_map: dict[str, list[tuple[str, str]]] = {
        "durable_execution": [
            ("execution_state", "execution_state"),
            ("task_checkpoints", "task_checkpoints"),
            ("paused_questions", "paused_questions"),
        ],
        "quality_scores": [("quality_scores", "quality_scores")],
        "episode_memory": [("episodes", "episodes")],
        "alerts": [("alerts", "alerts")],
        "plan_tracking": [("plans", "plans")],
        "unified_memory": [("memory_entries", "memory_entries")],
        "benchmarks": [("benchmark_results", "benchmark_results")],
        "improvement_log": [("improvement_log", "improvement_log")],
        "knowledge_base": [("knowledge_chunks", "knowledge_chunks")],
    }

    for store_name, db_path in legacy_dbs.items():
        logger.info("\nMigrating store: %s (%s)", store_name, db_path)

        if backup and not dry_run:
            backup_path = db_path.with_suffix(f".bak.{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
            shutil.copy2(db_path, backup_path)
            logger.info("  Backed up to %s", backup_path)

        try:
            source_conn = sqlite3.connect(str(db_path))
            table_pairs = _table_map.get(store_name, [])
            total = 0
            for source_table, target_table in table_pairs:
                total += _migrate_table(source_conn, target_conn, source_table, target_table, dry_run=dry_run)
            source_conn.close()
            results[store_name] = total
        except sqlite3.Error as e:
            logger.error("  FAILED to migrate %s: %s — store left in original location", store_name, e)
            errors.append(store_name)
            # Do not record a fake 0-row success; omit from results so callers can
            # distinguish "migrated 0 rows" from "migration failed entirely".

    logger.info("\n=== Migration summary ===")
    for name, count in results.items():
        logger.info("  %s: %d rows", name, count)
    for name in errors:
        logger.error("  %s: FAILED", name)
    total_rows = sum(results.values())
    logger.info("  TOTAL: %d rows migrated, %d store(s) failed", total_rows, len(errors))

    return results, errors


def main() -> int:
    """CLI entry point for the migration script.

    Returns:
        Exit code: 0 on full success, 1 if any store migration failed.
    """
    parser = argparse.ArgumentParser(description="Migrate legacy SQLite databases to unified vetinari.db")
    parser.add_argument("--dry-run", action="store_true", help="Preview what would be migrated without writing")
    parser.add_argument("--backup", action="store_true", help="Create backups of legacy databases before migration")
    args = parser.parse_args()

    _results, errors = migrate_all(dry_run=args.dry_run, backup=args.backup)
    if errors:
        logger.error("Migration completed with %d failure(s): %s", len(errors), ", ".join(errors))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
