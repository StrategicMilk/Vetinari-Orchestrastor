"""Module with properly documented public function."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def public_function(value: str) -> str:
    """Convert value to uppercase.

    Args:
        value: The string to convert.

    Returns:
        Uppercase version of value.
    """
    result = value.upper()
    logger.info("Processed %s", value)
    return result
