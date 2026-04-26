"""Module with class docstring that just restates the class name."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ValueStore:
    """ValueStore."""

    def __init__(self, initial: str) -> None:
        """Initialize with an initial value."""
        self._value = initial

    def retrieve(self) -> str:
        """Retrieve the stored value."""
        return self._value
