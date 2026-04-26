"""Module with implemented function body (no bare pass)."""
import logging

logger = logging.getLogger(__name__)


def implemented() -> None:
    """Implemented function."""
    logger.info("Working")
