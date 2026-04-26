"""Module with entry/exit logging noise — triggers VET107."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def process(data: str) -> str:
    """Normalise data to uppercase.

    Args:
        data: Input string.

    Returns:
        Uppercased string.
    """
    logger.info("entering function process")
    result = data.upper()
    return result
