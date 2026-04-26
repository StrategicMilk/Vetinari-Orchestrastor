"""Module that opens files without encoding."""
import logging

logger = logging.getLogger(__name__)


def read_config(path: str) -> str:
    """Read a config file.

    Args:
        path: Path to the file.

    Returns:
        File contents.
    """
    with open(path) as f:
        return f.read()
