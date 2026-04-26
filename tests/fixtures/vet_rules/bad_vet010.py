"""Module that omits the mandatory future annotations import."""
import logging

logger = logging.getLogger(__name__)


def process(value: str) -> None:
    """Process a value.

    Args:
        value: The value to process.
    """
    logger.info("Processing %s", value)
