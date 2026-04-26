"""Module with early-exit except block that includes logging."""
import logging

logger = logging.getLogger(__name__)


def process_item(item: str) -> str:
    """Process an item.

    Args:
        item: Item to process.

    Returns:
        Processed result.
    """
    try:
        return item.upper()
    except AttributeError:
        logger.warning("Item %r has no upper() — returning empty string", item)
        return ""
