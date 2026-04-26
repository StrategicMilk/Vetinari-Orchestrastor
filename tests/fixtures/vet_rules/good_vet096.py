"""Module with meaningful class and function docstrings."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ValueStore:
    """Thread-safe container for a single mutable string value."""

    def __init__(self, initial: str) -> None:
        """Initialize with an initial value."""
        self._value = initial

    def retrieve(self) -> str:
        """Return the currently stored value."""
        return self._value
