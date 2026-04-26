"""Module with proper exception chaining — clean for VET104."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def parse_value(raw: str) -> int:
    """Parse a string value as an integer.

    Args:
        raw: Raw string to parse.

    Returns:
        Parsed integer value.

    Raises:
        ValueError: If parsing fails, with the original parse error chained.
    """
    try:
        return int(raw)
    except Exception as exc:  # noqa: BLE001 - fixture intentionally catches broad exceptions for rule coverage
        raise ValueError(f"Could not parse value: {raw!r}") from exc
