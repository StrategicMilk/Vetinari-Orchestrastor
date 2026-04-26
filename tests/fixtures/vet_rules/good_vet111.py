"""Module that returns directly without an intermediate variable — clean for VET111."""
from __future__ import annotations


def get_uppercase(value: str) -> str:
    """Return the uppercase version of a string.

    Args:
        value: Input string.

    Returns:
        Uppercased string.
    """
    return value.upper()
