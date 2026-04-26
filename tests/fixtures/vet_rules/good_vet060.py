"""Module that opens files with encoding."""
import logging

logger = logging.getLogger(__name__)


def read_config(path: str) -> str:
    """Read a config file.

    Args:
        path: Path to the file.

    Returns:
        File contents.
    """
    with open(path, encoding="utf-8") as f:
        return f.read()
