"""Module with non-empty except block."""
import logging

logger = logging.getLogger(__name__)


def risky_operation() -> None:
    """Perform a risky operation."""
    try:
        pass
    except Exception:
        logger.warning("Operation failed — continuing")
