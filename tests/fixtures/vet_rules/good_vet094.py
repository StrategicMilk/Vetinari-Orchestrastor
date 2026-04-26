"""Module with docstring that includes Raises section."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def parse_config(path: str) -> dict:
    """Parse configuration file.

    Args:
        path: Path to the config file.

    Returns:
        Parsed configuration dict.

    Raises:
        ValueError: If path is empty.
    """
    if not path:
        raise ValueError("Path cannot be empty")
    logger.info("Parsing config from %s", path)
    return {}
