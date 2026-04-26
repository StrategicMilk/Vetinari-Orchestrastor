"""Module with correct future annotations import."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def process(value: str) -> None:
    """Process a value.

    Args:
        value: The value to process.
    """
    logger.info("Processing %s", value)
