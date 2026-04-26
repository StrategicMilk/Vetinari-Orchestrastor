"""Module with a large dataclass missing __repr__ — triggers VET113."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TaskSummary:
    """Summary of a task execution captured for reporting."""

    task_id: str
    agent_type: str
    status: str
    duration_ms: float
