"""Module with a mutable value-type dataclass — triggers VET114."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TaskConfig:
    """Configuration for a single task execution.

    Value types like Config should be immutable — use frozen=True.
    """

    max_retries: int = 3  # Maximum number of retry attempts
    timeout_seconds: float = 30.0  # Per-attempt timeout in seconds
