"""Module with zero-logic property — triggers VET106."""
from __future__ import annotations


class DataStore:
    """A simple key-value store with a guarded getter."""

    def __init__(self, value: str) -> None:
        """Store the initial value.

        Args:
            value: The string value to store.
        """
        self._value = value

    @property
    def value(self) -> str:
        return self._value
