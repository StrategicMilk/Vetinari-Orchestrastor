"""Module with too-short docstring."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def process(value: str) -> str:
    """Do it."""
    logger.info("Processing %s", value)
    return value
