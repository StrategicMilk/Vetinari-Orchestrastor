"""Thread-safe singleton utilities for the Vetinari codebase.

Provides both a decorator (``thread_safe_singleton``) and a metaclass
(``SingletonMeta``, re-exported from ``vetinari.utils``) for creating
singletons with double-checked locking.  Prefer the decorator for
module-level factory functions; use the metaclass for class-level
singletons that need ``reset_instance()`` in tests.
"""

from __future__ import annotations

import functools
import threading
from typing import Any, TypeVar

T = TypeVar("T")

__all__ = ["thread_safe_singleton"]

# ── Sentinel for uninitialised state ────────────────────────────────────────
_UNSET: object = object()


def thread_safe_singleton(func: Any) -> Any:
    """Decorator that turns a factory function into a thread-safe singleton.

    Wraps *func* with double-checked locking so the first call creates and
    caches the instance, and all subsequent calls return the cached object.

    The wrapper exposes a ``reset()`` method that clears the cached instance
    (useful for tests).

    Example::

        @thread_safe_singleton
        def get_database():
            return Database(url="sqlite:///vetinari.db")

        db1 = get_database()
        db2 = get_database()
        assert db1 is db2

        get_database.reset()  # Clear for testing

    Args:
        func: A callable (typically a factory function) whose return value
            should be cached as a singleton.

    Returns:
        A wrapper with the same signature that returns the cached instance.
    """
    lock = threading.Lock()
    instance: list[Any] = [_UNSET]  # Mutable container for nonlocal mutation

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Return the cached singleton instance, creating it on first call.

        Returns:
            The singleton instance created by the wrapped factory function.
        """
        if instance[0] is _UNSET:
            with lock:
                if instance[0] is _UNSET:
                    instance[0] = func(*args, **kwargs)
        return instance[0]

    def reset() -> None:
        """Clear the cached singleton instance (for tests)."""
        with lock:
            instance[0] = _UNSET

    wrapper.reset = reset  # type: ignore[attr-defined]
    return wrapper
