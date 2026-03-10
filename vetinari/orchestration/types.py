"""
Data types for the two-layer orchestration system.

Contains enums and dataclasses shared across execution_graph,
plan_generator, durable_engine, and two_layer modules.
"""

import os
import json
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import threading

# Import canonical enums from single source of truth (P2.2)
from vetinari.types import TaskStatus, PlanStatus  # noqa: F401

logger = logging.getLogger(__name__)


@dataclass
class TaskNode:
    """A single task node in the execution graph."""
    id: str
    description: str
    task_type: str = "general"

    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    depended_by: List[str] = field(default_factory=list)

    # Execution
    status: TaskStatus = TaskStatus.PENDING
    assigned_model: str = ""

    # Results
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    # Checkpoint
    checkpoint_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict) -> 'TaskNode':
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
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Checkpoint:
    """A checkpoint for durable execution."""
    checkpoint_id: str
    plan_id: str
    created_at: str
    graph_state: Dict[str, Any]
    completed_tasks: List[str]
    running_tasks: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
