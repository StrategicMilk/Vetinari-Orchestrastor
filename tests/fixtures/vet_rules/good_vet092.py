"""Module with docstring that includes Args section for multi-param function."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def configure(host: str, port: int) -> None:
    """Configure the connection settings.

    Args:
        host: Hostname to connect to.
        port: Port number.
    """
    logger.info("Connecting to %s:%d", host, port)
