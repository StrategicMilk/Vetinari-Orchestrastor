"""Module with a relative path to a database file.

This fixture intentionally violates VET230: uses a bare relative string path
to a data file instead of a constant from vetinari.constants.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DB_PATH = "data/metrics.db"  # Bare relative string — should use an absolute path anchored to the project root


def get_db_path() -> str:
    """Return the path to the metrics database.

    Returns:
        String path to the SQLite database file.
    """
    return DB_PATH
