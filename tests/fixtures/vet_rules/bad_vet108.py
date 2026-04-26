"""Module with redundant boolean ternary return — triggers VET108."""
from __future__ import annotations


def is_valid(value: str) -> bool:
    """Check whether a value is non-empty.

    Args:
        value: Value to check.

    Returns:
        True if the value is non-empty, False otherwise.
    """
    return True if value else False
