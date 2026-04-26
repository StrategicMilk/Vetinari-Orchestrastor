"""Module using an absolute path derived from the project root.

This fixture satisfies VET230: constructs the database path using
Path(__file__).parent anchoring rather than a bare relative string.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Anchor at the project root so this path is valid regardless of cwd.
_PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = _PROJECT_ROOT / "data" / "metrics.db"


def get_db_path() -> str:
    """Return the absolute path to the metrics database.

    Returns:
        Absolute string path to the SQLite database file.
    """
    return str(DB_PATH)
