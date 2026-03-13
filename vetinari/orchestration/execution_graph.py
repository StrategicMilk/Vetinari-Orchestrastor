"""Execution Graph — Layer 1 data structures for the Two-Layer Orchestration System.

Contains the DAG (Directed Acyclic Graph) of tasks used for plan execution.
This is distinct from :class:`~vetinari.orchestration.agent_graph.TaskNode`
which represents agent-level graph nodes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from vetinari.types import PlanStatus, TaskStatus


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
        return cls(
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


@dataclass
class ExecutionGraph:
    """Directed Acyclic Graph (DAG) of tasks for execution.

    Supports:
    - Parallel execution of independent tasks
    - Dependency resolution
    - Execution ordering
    - Checkpoint management
    """

    plan_id: str
    goal: str

    # Nodes in the graph
    nodes: dict[str, TaskNode] = field(default_factory=dict)

    # Graph metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Execution state
    status: PlanStatus = PlanStatus.DRAFT
    current_layer: int = 0
    completed_count: int = 0
    failed_count: int = 0

    def add_task(
        self,
        task_id: str,
        description: str,
        task_type: str = "general",
        depends_on: list[str] | None = None,
        input_data: dict[str, Any] | None = None,
    ) -> TaskNode:
        """Add a task to the graph."""
        node = TaskNode(
            id=task_id,
            description=description,
            task_type=task_type,
            depends_on=depends_on or [],
            input_data=input_data or {},
        )

        # Update dependency links
        for dep_id in node.depends_on:
            if dep_id in self.nodes:
                self.nodes[dep_id].depended_by.append(task_id)

        self.nodes[task_id] = node
        self.updated_at = datetime.now().isoformat()
        return node

    def get_ready_tasks(self) -> list[TaskNode]:
        """Get tasks that are ready to execute (all dependencies completed)."""
        ready = []
        for node in self.nodes.values():
            if node.status != TaskStatus.PENDING:
                continue
            deps_met = all(
                self.nodes.get(dep_id, TaskNode(id=dep_id, description="")).status == TaskStatus.COMPLETED
                for dep_id in node.depends_on
            )
            if deps_met:
                ready.append(node)
        return ready

    def get_next_layer(self) -> list[list[TaskNode]]:
        """Get the full execution schedule as layers of parallel tasks.

        Uses a simulation pass: tasks are "virtually completed" after being
        placed in a layer so that subsequent layers correctly detect their
        dependencies as satisfied.
        """
        layers: list[list[TaskNode]] = []
        simulated_completed: set[str] = {nid for nid, n in self.nodes.items() if n.status == TaskStatus.COMPLETED}
        remaining = {nid: n for nid, n in self.nodes.items() if n.status in (TaskStatus.PENDING, TaskStatus.BLOCKED)}

        while remaining:
            current_layer: list[TaskNode] = []
            to_remove: list[str] = []

            for task_id, node in remaining.items():
                deps_met = all(dep_id in simulated_completed for dep_id in node.depends_on)
                if deps_met:
                    current_layer.append(node)
                    to_remove.append(task_id)

            if not current_layer:
                break  # circular dependency or all blocked

            layers.append(current_layer)
            for task_id in to_remove:
                del remaining[task_id]
                simulated_completed.add(task_id)

        return layers

    def get_execution_order(self) -> list[list[TaskNode]]:
        """Get full execution order as layers."""
        return self.get_next_layer()

    def can_retry(self, task_id: str) -> bool:
        """Check if a failed task can be retried."""
        if task_id not in self.nodes:
            return False
        node = self.nodes[task_id]
        return node.retry_count < node.max_retries

    def get_blocked_tasks(self) -> list[TaskNode]:
        """Get tasks that are blocked waiting for dependencies."""
        return [n for n in self.nodes.values() if n.status == TaskStatus.BLOCKED]

    def get_failed_tasks(self) -> list[TaskNode]:
        """Get all failed tasks."""
        return [n for n in self.nodes.values() if n.status == TaskStatus.FAILED]

    def get_completed_tasks(self) -> list[TaskNode]:
        """Get all completed tasks."""
        return [n for n in self.nodes.values() if n.status == TaskStatus.COMPLETED]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status.value,
            "current_layer": self.current_layer,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
        }

    def to_dag_json(self) -> str:
        """Export graph in DAG format for visualization."""
        return json.dumps(self.to_dict(), indent=2)
