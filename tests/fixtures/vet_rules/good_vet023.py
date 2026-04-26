"""Module that logs real failures at WARNING level."""
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
        logger.warning("RuntimeError during fetch — returning empty string")
        return ""
