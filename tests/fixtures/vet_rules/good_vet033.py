"""Module using abstractmethod correctly for NotImplementedError."""
from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    """Abstract base processor."""

    @abstractmethod
    def process(self) -> str:
        """Process something.

        Returns:
            Result string.
        """
        raise NotImplementedError
