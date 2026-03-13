"""Data types for the two-layer orchestration system.

Contains enums and dataclasses shared across execution_graph,
plan_generator, durable_engine, and two_layer modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Import canonical enums from single source of truth (P2.2)
from vetinari.types import PlanStatus, TaskStatus  # noqa: F401

logger = logging.getLogger(__name__)


@dataclass
class TaskNode:
    """A single task node in the execution graph."""

    id: str
    description: str
    task_type: str = "general"

    # Dependencies
    depends_on: list[str] = field(default_factory=list)
    depended_by: list[str] = field(default_factory=list)

    # Execution
    status: TaskStatus = TaskStatus.PENDING
    assigned_model: str = ""

    # Results
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    retry_count: int = 0
    max_retries: int = 3

    # Checkpoint
    checkpoint_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "task_type": self.task_type,
            "depends_on": self.depends_on,
            "depended_by": self.depended_by,
            "status": self.status.value,
            "assigned_model": self.assigned_model,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "checkpoint_id": self.checkpoint_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TaskNode:
        node = cls(
            id=data["id"],
            description=data["description"],
            task_type=data.get("task_type", "general"),
            depends_on=data.get("depends_on", []),
            depended_by=data.get("depended_by", []),
            status=TaskStatus(data.get("status", "pending")),
            assigned_model=data.get("assigned_model", ""),
            input_data=data.get("input_data", {}),
            output_data=data.get("output_data", {}),
            error=data.get("error", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            checkpoint_id=data.get("checkpoint_id", ""),
        )
        return node


@dataclass
class ExecutionEvent:
    """An event in the execution history."""

    event_id: str
    event_type: str  # task_started, task_completed, task_failed, etc.
    task_id: str
    timestamp: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Checkpoint:
    """A checkpoint for durable execution."""

    checkpoint_id: str
    plan_id: str
    created_at: str
    graph_state: dict[str, Any]
    completed_tasks: list[str]
    running_tasks: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
