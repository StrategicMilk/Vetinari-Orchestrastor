"""Module with a properly frozen value-type dataclass — clean for VET114."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    """Configuration for a single task execution.

    Frozen so callers cannot mutate shared config instances and introduce
    subtle ordering-dependent bugs in concurrent execution paths.
    """

    max_retries: int = 3  # Maximum number of retry attempts
    timeout_seconds: float = 30.0  # Per-attempt timeout in seconds
