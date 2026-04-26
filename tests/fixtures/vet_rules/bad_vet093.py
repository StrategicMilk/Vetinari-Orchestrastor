"""Module with docstring missing Returns section."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_value(key: str) -> str:
    """Look up value by key.

    Args:
        key: The lookup key.
    """
    logger.info("Looking up %s", key)
    return key.upper()
