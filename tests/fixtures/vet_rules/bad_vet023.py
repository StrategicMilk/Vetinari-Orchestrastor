"""Module that logs real failures at DEBUG level."""
import logging

logger = logging.getLogger(__name__)


def fetch_data() -> str:
    """Fetch data from a source.

    Returns:
        Retrieved data string.
    """
    try:
        return "data"
    except RuntimeError:
        logger.debug("RuntimeError during fetch — suppressed")
        return ""
