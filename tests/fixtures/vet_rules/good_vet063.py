"""Module using pathlib for path operations."""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_data_path(base: str, filename: str) -> str:
    """Get path to data file.

    Args:
        base: Base directory.
        filename: File name.

    Returns:
        Full path string.
    """
    return str(Path(base) / filename)
