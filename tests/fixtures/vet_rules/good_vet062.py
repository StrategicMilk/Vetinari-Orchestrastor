"""Module with reasonable sleep duration."""
import logging
import time

logger = logging.getLogger(__name__)


def wait_for_ready() -> None:
    """Wait for service to be ready."""
    time.sleep(2)
    logger.info("Service ready")
