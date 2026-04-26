"""Module with excessively long sleep."""
import logging
import time

logger = logging.getLogger(__name__)


def wait_for_ready() -> None:
    """Wait for service to be ready."""
    time.sleep(10)
    logger.info("Service ready")
