"""Module with properly locked singleton.

This fixture satisfies VET210: the singleton uses a threading.Lock with
double-checked locking to prevent race conditions.
"""

from __future__ import annotations

import threading

_instance = None
_lock = threading.Lock()


def get_instance() -> object:
    """Return the module-level singleton, creating it on first call with a lock.

    Uses double-checked locking so the lock is only acquired on the first call.

    Returns:
        The singleton object instance.
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = object()
    return _instance
