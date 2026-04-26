"""Module using timezone-aware datetime — clean for VET103."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def get_timestamp() -> str:
    """Return the current UTC time as an ISO-format string.

    Uses datetime.now(timezone.utc) so the value is unambiguous across
    timezone boundaries and compatible with Python 3.12+ deprecations.

    Returns:
        ISO format timestamp string with UTC offset.
    """
    return datetime.now(timezone.utc).isoformat()
