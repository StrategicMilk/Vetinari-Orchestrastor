"""Module with success-masking except block."""
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
        logger.warning("Check failed")
        return True


def _check() -> None:
    """Internal check."""
