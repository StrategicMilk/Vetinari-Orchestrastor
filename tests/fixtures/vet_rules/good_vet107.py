"""Module with meaningful state-transition logging — clean for VET107."""
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
    result = data.upper()
    logger.info("Processed %d chars of data", len(data))
    return result
