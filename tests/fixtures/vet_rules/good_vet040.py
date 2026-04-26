"""Module that reads credentials from environment."""
import logging
import os

logger = logging.getLogger(__name__)

api_key = os.environ.get("API_KEY", "")


def get_client() -> None:
    """Get API client."""
    logger.info("Getting client")
