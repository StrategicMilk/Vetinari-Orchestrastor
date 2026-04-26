"""Module returning a generic error to the API client — clean for VET116."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def handle_request(data: str) -> str:
    """Process an API request and return a response.

    Args:
        data: Raw request data to process.

    Returns:
        Processed response string, or a generic error message on failure.
    """
    try:
        return data.upper()
    except Exception:  # noqa: BLE001 - fixture intentionally catches broad exceptions for rule coverage
        logger.exception("Failed to process request data — returning generic error")
        return "An error occurred while processing your request"
