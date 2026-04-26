"""Module with debug breakpoint."""
import logging

logger = logging.getLogger(__name__)


def process(item: str) -> None:
    """Process an item.

    Args:
        item: Item to process.
    """
    breakpoint()
    logger.info("Processing %s", item)
