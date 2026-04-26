"""Module with manual to_dict on a dataclass — triggers VET105."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TaskRecord:
    """A persisted record of a completed task."""

    task_id: str
    status: str

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for serialization.

        Returns:
            Dictionary with task_id and status — mechanical copy of fields.
        """
        return {"task_id": self.task_id, "status": self.status}
