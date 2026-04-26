"""Module with a property that does actual work — clean for VET106."""
from __future__ import annotations


class DataStore:
    """A simple key-value store that normalises values on read."""

    def __init__(self, value: str) -> None:
        """Store the initial value, preserving leading/trailing whitespace.

        Args:
            value: The string value to store (stored as-is).
        """
        self._value = value

    @property
    def value(self) -> str:
        """Return the stored value stripped of leading/trailing whitespace.

        Stripping is deferred to read time so callers always receive a
        normalised string regardless of how the value was originally set.

        Returns:
            Stripped value.
        """
        return self._value.strip()
