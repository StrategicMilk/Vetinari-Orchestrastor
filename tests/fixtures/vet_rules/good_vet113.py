"""Module with a large dataclass that has __repr__ — clean for VET113."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TaskSummary:
    """Summary of a task execution captured for reporting."""

    task_id: str
    agent_type: str
    status: str
    duration_ms: float

    def __repr__(self) -> str:
        """Show the key identifying fields for debugging.

        Returns:
            Compact string with task_id and status.
        """
        return f"TaskSummary(task_id={self.task_id!r}, status={self.status!r})"
