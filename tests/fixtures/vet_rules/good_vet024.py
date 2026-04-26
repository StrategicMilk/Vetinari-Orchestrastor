"""Module with correct exception handling (no success masking)."""
import logging

logger = logging.getLogger(__name__)


def verify_task() -> bool:
    """Verify a task completed successfully.

    Returns:
        True if verification passed.
    """
    try:
        _check()
        return True
    except Exception:
        logger.warning("Check failed — returning False")
        return False


def _check() -> None:
    """Internal check."""
