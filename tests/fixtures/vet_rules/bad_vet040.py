"""Module with hardcoded credentials."""
import logging

logger = logging.getLogger(__name__)

api_key = "sk-1234567890abcdef"


def get_client() -> None:
    """Get API client."""
    logger.info("Getting client")
