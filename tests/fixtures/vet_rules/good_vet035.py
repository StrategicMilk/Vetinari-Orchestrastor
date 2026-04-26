"""Module using logging instead of print()."""
import logging

logger = logging.getLogger(__name__)


def show_result(result: str) -> None:
    """Display a result.

    Args:
        result: The result to display.
    """
    logger.info("Result: %s", result)
