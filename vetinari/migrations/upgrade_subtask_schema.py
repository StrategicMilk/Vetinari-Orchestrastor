"""Migration: upgrade Subtask schema to include ponder auditing fields.

This script migrates existing Subtask records to include:
- ponder_ranking: []
- ponder_scores: {}
- ponder_used: false
- schema_version: 1 (or higher if extended)

Usage:
    python vetinari/migrations/upgrade_subtask_schema_v1_to_v2.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def migrate(storage_root: str | None = None):
    """Migrate existing subtask files to include ponder fields."""
    if storage_root is None:
        storage_root = Path.home() / ".lmstudio" / "projects" / "Vetinari" / "subtasks"
    else:
        storage_root = Path(storage_root)

    if not storage_root.exists():
        logger.warning(f"Storage path does not exist: {storage_root}")
        return

    migrated_count = 0
    error_count = 0

    for file in storage_root.glob("*.json"):
        try:
            with open(file) as f:
                data = json.load(f)

            subtasks = data.get("subtasks", [])
            changed = False

            for st in subtasks:
                if "ponder_ranking" not in st:
                    st["ponder_ranking"] = []
                    changed = True
                if "ponder_scores" not in st:
                    st["ponder_scores"] = {}
                    changed = True
                if "ponder_used" not in st:
                    st["ponder_used"] = False
                    changed = True
                if "schema_version" not in st:
                    st["schema_version"] = 1
                    changed = True

            if changed:
                with open(file, "w") as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Migrated: {file.name}")
                migrated_count += 1
            else:
                logger.debug(f"Skipped (up to date): {file.name}")

        except Exception as e:
            logger.error(f"Error migrating {file.name}: {e}")
            error_count += 1

    logger.info(f"Migration complete: Migrated: {migrated_count} files, Errors: {error_count} files")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        migrate(sys.argv[1])
    else:
        migrate()
