"""Module with singleton missing double-checked locking.

This fixture intentionally violates VET210: a raw _instance singleton pattern
is used without a threading.Lock guard.
"""

from __future__ import annotations

_instance = None


def get_instance() -> object:
    """Return the module-level singleton, creating it on first call.

    Returns:
        The singleton object instance.
    """
    global _instance
    if _instance is None:
        _instance = object()
    return _instance
