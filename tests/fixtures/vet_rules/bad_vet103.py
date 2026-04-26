"""Module using naive datetime without timezone — triggers VET103."""
from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def get_timestamp() -> str:
    """Return the current time as an ISO-format string.

    Uses naive datetime.now() which omits timezone info, making the value
    ambiguous in any context that crosses timezone boundaries.

    Returns:
        ISO format timestamp string without timezone offset.
    """
    return datetime.now().isoformat()
