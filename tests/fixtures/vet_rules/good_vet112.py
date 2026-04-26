"""Module without unnecessary defensive or-empty — clean for VET112."""
from __future__ import annotations


def normalise_label(label: str) -> str:
    """Return the label unchanged for display.

    Args:
        label: A non-optional label string.

    Returns:
        The label as-is.
    """
    return label
