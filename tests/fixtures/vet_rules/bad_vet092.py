"""Module with docstring missing Args section for multi-param function."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def configure(host: str, port: int) -> None:
    """Configure the connection settings."""
    logger.info("Connecting to %s:%d", host, port)
