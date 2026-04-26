"""Module with hardcoded localhost URL."""
import logging

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8080"


def get_url() -> str:
    """Get the base URL.

    Returns:
        The base URL string.
    """
    return BASE_URL
