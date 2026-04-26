"""Module using os.path.join instead of pathlib."""
import logging
import os

logger = logging.getLogger(__name__)


def get_data_path(base: str, filename: str) -> str:
    """Get path to data file.

    Args:
        base: Base directory.
        filename: File name.

    Returns:
        Full path string.
    """
    return os.path.join(base, filename)
