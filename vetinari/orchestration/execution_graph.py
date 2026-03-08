"""
Execution graph (DAG) for the two-layer orchestration system.

Provides the ExecutionGraph class that models tasks as a directed acyclic
graph with dependency resolution and parallel execution scheduling.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from vetinari.orchestration.types import TaskStatus, PlanStatus, TaskNode

logger = logging.getLogger(__name__)


@dataclass
class ExecutionGraph:
    """
    Directed Acyclic Graph (DAG) of tasks for execution.

    Supports:
    - Parallel execution of independent tasks
    - Dependency resolution
    - Execution ordering
    - Checkpoint management
    """

    plan_id: str
    goal: str

    # Nodes in the graph
    nodes: Dict[str, TaskNode] = field(default_factory=dict)

    # Graph metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Execution state
    status: PlanStatus = PlanStatus.DRAFT
    current_layer: int = 0
    completed_count: int = 0
    failed_count: int = 0

    def add_task(self,
                task_id: str,
                description: str,
                task_type: str = "general",
                depends_on: List[str] = None,
                input_data: Dict[str, Any] = None) -> TaskNode:
        """Add a task to the graph."""
        # Create node
        node = TaskNode(
            id=task_id,
            description=description,
            task_type=task_type,
            depends_on=depends_on or [],
            input_data=input_data or {}
        )

        # Update dependency links
        for dep_id in node.depends_on:
            if dep_id in self.nodes:
                self.nodes[dep_id].depended_by.append(task_id)

        self.nodes[task_id] = node
        self.updated_at = datetime.now().isoformat()

        return node

    def get_ready_tasks(self) -> List[TaskNode]:
        """Get tasks that are ready to execute (all dependencies completed)."""
        ready = []
        for node in self.nodes.values():
            if node.status != TaskStatus.PENDING:
                continue

            # Check if all dependencies are completed
            deps_met = all(
                self.nodes.get(dep_id, TaskNode(id=dep_id, description="")).status == TaskStatus.COMPLETED
                for dep_id in node.depends_on
            )

            if deps_met:
                ready.append(node)

        return ready

    def get_next_layer(self) -> List[List["TaskNode"]]:
        """
        Get the full execution schedule as layers of tasks that can run in parallel.

        Uses a simulation pass: tasks are "virtually completed" after being placed in
        a layer so that subsequent layers correctly detect their dependencies as satisfied.

        Returns list of layers (each layer is a list of tasks that can run in parallel).
        """
        layers = []
        # Work on copies of status so we don't mutate real node state
        simulated_completed: set = {
            nid for nid, n in self.nodes.items()
            if n.status == TaskStatus.COMPLETED
        }
        remaining = {
            nid: n for nid, n in self.nodes.items()
            if n.status in (TaskStatus.PENDING, TaskStatus.BLOCKED)
        }

        while remaining:
            current_layer = []
            to_remove = []

            for task_id, node in remaining.items():
                # A task is ready when ALL its dependencies are in the simulated-completed set
                deps_met = all(dep_id in simulated_completed for dep_id in node.depends_on)

                if deps_met:
                    current_layer.append(node)
                    to_remove.append(task_id)

            if not current_layer:
                # No progress possible -- circular dependency or all remaining tasks blocked
                break

            layers.append(current_layer)

            for task_id in to_remove:
                del remaining[task_id]
                simulated_completed.add(task_id)

        return layers

    def get_execution_order(self) -> List[List[TaskNode]]:
        """Get full execution order as layers."""
        return self.get_next_layer()

    def can_retry(self, task_id: str) -> bool:
        """Check if a failed task can be retried."""
        if task_id not in self.nodes:
            return False
        node = self.nodes[task_id]
        return node.retry_count < node.max_retries

    def get_blocked_tasks(self) -> List[TaskNode]:
        """Get tasks that are blocked waiting for dependencies."""
        blocked = []
        for node in self.nodes.values():
            if node.status == TaskStatus.BLOCKED:
                blocked.append(node)
        return blocked

    def get_failed_tasks(self) -> List[TaskNode]:
        """Get all failed tasks."""
        return [n for n in self.nodes.values() if n.status == TaskStatus.FAILED]

    def get_completed_tasks(self) -> List[TaskNode]:
        """Get all completed tasks."""
        return [n for n in self.nodes.values() if n.status == TaskStatus.COMPLETED]

    def to_dict(self) -> Dict[str, Any]:
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
