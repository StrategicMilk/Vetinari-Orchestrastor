"""Module with redundant intermediate variable — triggers VET111."""
from __future__ import annotations


def get_uppercase(value: str) -> str:
    """Return the uppercase version of a string.

    Args:
        value: Input string.

    Returns:
        Uppercased string.
    """
    result = value.upper()
    return result
