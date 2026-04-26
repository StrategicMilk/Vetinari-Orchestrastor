"""Module using root logger instead of module logger."""
import logging


def process(item: str) -> None:
    """Process an item.

    Args:
        item: Item to process.
    """
    logging.info("Processing %s", item)
