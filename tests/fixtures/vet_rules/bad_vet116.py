"""Module that leaks str(e) to the API client — triggers VET116."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def handle_request(data: str) -> str:
    """Process an API request and return a response.

    Args:
        data: Raw request data to process.

    Returns:
        Processed response string.
    """
    try:
        return data.upper()
    except Exception as e:  # noqa: BLE001 - fixture intentionally catches broad exceptions for rule coverage
        # Leaks internal exception message to the caller — VET116 fires.
        return (str(e))
