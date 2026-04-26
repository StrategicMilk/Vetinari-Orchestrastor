"""Module with broken exception chain — triggers VET104."""
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
        ValueError: If parsing fails.
    """
    try:
        return int(raw)
    except Exception as exc:  # noqa: BLE001 - fixture intentionally catches broad exceptions for rule coverage
        # Missing 'from exc' loses the original traceback — VET104 fires here.
        raise ValueError(f"Could not parse value: {raw!r}")
