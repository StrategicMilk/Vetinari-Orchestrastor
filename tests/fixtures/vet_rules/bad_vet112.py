"""Module with defensive or-empty on a non-optional local variable — triggers VET112."""
from __future__ import annotations


def normalise_label(label: str) -> str:
    """Normalise a label for display, falling back to an empty string.

    The defensive fallback is unnecessary — label is typed str, not str | None.

    Args:
        label: A non-optional label string.

    Returns:
        The label, or an empty string (redundant guard).
    """
    # label is typed str, so the 'or ""' guard is never needed — VET112 fires.
    display = label or ""
    return display
