"""Module with direct boolean return — clean for VET108."""
from __future__ import annotations


def is_valid(value: str) -> bool:
    """Check whether a value is non-empty.

    Args:
        value: Value to check.

    Returns:
        True if the value is non-empty, False otherwise.
    """
    return bool(value)
