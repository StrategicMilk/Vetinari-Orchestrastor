"""Module with adequately long docstring."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def process(value: str) -> str:
    """Transform the input value into its canonical form.

    Args:
        value: Input string to process.

    Returns:
        Processed output string.
    """
    logger.info("Processing %s", value)
    return value
