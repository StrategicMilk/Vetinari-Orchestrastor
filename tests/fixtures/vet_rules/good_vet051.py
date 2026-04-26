"""Module using %-style in logger call."""
import logging

logger = logging.getLogger(__name__)


def process(item: str) -> None:
    """Process an item.

    Args:
        item: Item to process.
    """
    logger.info("Processing %s", item)
