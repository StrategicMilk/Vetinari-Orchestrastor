"""Module with docstring that includes Returns section."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_value(key: str) -> str:
    """Look up value by key.

    Args:
        key: The lookup key.

    Returns:
        Uppercase version of the key.
    """
    logger.info("Looking up %s", key)
    return key.upper()
