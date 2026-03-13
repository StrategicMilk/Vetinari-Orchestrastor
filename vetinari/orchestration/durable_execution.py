"""Durable Execution Engine — Layer 2 of the Two-Layer Orchestration System.

Provides stateful task execution with checkpointing, retry policies,
event sourcing, crash recovery, and deterministic replay.
Inspired by Temporal workflow patterns.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from vetinari.orchestration.execution_graph import ExecutionGraph, TaskNode
from vetinari.types import PlanStatus, TaskStatus

logger = logging.getLogger(__name__)


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


class DurableExecutionEngine:
    """Durable execution engine inspired by Temporal.

    Features:
    - State persistence with checkpoints
    - Retry policies
    - Event sourcing
    - Crash recovery
    - Deterministic replay
    """

    def __init__(
        self,
        checkpoint_dir: str | None = None,
        max_concurrent: int = 4,
        default_timeout: float = 300.0,
    ):
        self.checkpoint_dir = Path(checkpoint_dir or "./vetinari_checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout

        # Active executions
        self._active_executions: dict[str, ExecutionGraph] = {}
        self._execution_lock = threading.Lock()

        # Event history
        self._event_history: list[ExecutionEvent] = []

        # Task handlers
        self._task_handlers: dict[str, Callable] = {}

        # Callbacks
        self._on_task_start: Callable | None = None
        self._on_task_complete: Callable | None = None
        self._on_task_fail: Callable | None = None

        logger.info(f"DurableExecutionEngine initialized (checkpoint_dir={self.checkpoint_dir})")

    def register_handler(self, task_type: str, handler: Callable):
        """Register a handler for a task type."""
        self._task_handlers[task_type] = handler
        logger.debug("Registered handler for task type: %s", task_type)

    def set_callbacks(
        self,
        on_task_start: Callable | None = None,
        on_task_complete: Callable | None = None,
        on_task_fail: Callable | None = None,
    ):
        """Set execution callbacks."""
        self._on_task_start = on_task_start
        self._on_task_complete = on_task_complete
        self._on_task_fail = on_task_fail

    def create_execution(self, graph: ExecutionGraph) -> str:
        """Create a new execution from a graph."""
        plan_id = graph.plan_id
        with self._execution_lock:
            self._active_executions[plan_id] = graph
        self._save_checkpoint(plan_id, graph)
        logger.info("Created execution for plan: %s", plan_id)
        return plan_id

    def execute_plan(
        self,
        graph: ExecutionGraph,
        task_handler: Callable | None = None,
    ) -> dict[str, Any]:
        """Execute a plan with durable semantics.

        Args:
            graph: The execution graph
            task_handler: Handler function for executing tasks

        Returns:
            Execution results
        """
        plan_id = graph.plan_id
        graph.status = PlanStatus.RUNNING

        if task_handler:
            self._task_handlers["default"] = task_handler

        self.create_execution(graph)
        layers = graph.get_execution_order()

        results: dict[str, Any] = {
            "plan_id": plan_id,
            "total_tasks": len(graph.nodes),
            "completed": 0,
            "failed": 0,
            "task_results": {},
        }

        for layer_idx, layer in enumerate(layers):
            logger.info(f"Executing layer {layer_idx + 1}/{len(layers)} with {len(layer)} tasks")
            layer_results = self._execute_layer(graph, layer)

            for task_id, result in layer_results.items():
                results["task_results"][task_id] = result
                if result.get("status") == "completed":
                    results["completed"] += 1
                else:
                    results["failed"] += 1

            failed_tasks = [t for t in layer if t.status == TaskStatus.FAILED]
            if failed_tasks:
                self._handle_layer_failure(graph, failed_tasks)

        graph.status = PlanStatus.COMPLETED if results["failed"] == 0 else PlanStatus.FAILED
        self._save_checkpoint(plan_id, graph)
        return results

    def _execute_layer(self, graph: ExecutionGraph, layer: list[TaskNode]) -> dict[str, Any]:
        """Execute a layer of tasks in parallel."""
        results: dict[str, Any] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(layer), self.max_concurrent)) as executor:
            future_to_task = {executor.submit(self._execute_task, graph, task): task for task in layer}
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results[task.id] = result
                except Exception as e:
                    logger.error("Task %s failed with exception: %s", task.id, e)
                    results[task.id] = {"status": "failed", "error": str(e)}

        return results

    def _execute_task(self, graph: ExecutionGraph, task: TaskNode) -> dict[str, Any]:
        """Execute a single task with retry logic."""
        task_id = task.id

        self._emit_event("task_started", task_id, {"description": task.description})
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now().isoformat()

        if self._on_task_start:
            try:
                self._on_task_start(task)
            except Exception as e:
                logger.warning("Task start callback failed: %s", e)

        handler = self._task_handlers.get(task.task_type) or self._task_handlers.get("default")

        if not handler:
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            task.output_data = {"warning": "No handler registered"}
            self._emit_event("task_completed", task_id, {"status": "completed"})
            return {"status": "completed", "output": task.output_data}

        max_attempts = task.max_retries + 1
        last_error = None

        for attempt in range(max_attempts):
            try:
                output = handler(task)

                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now().isoformat()
                task.output_data = output if isinstance(output, dict) else {"output": output}

                self._emit_event(
                    "task_completed",
                    task_id,
                    {"status": "completed", "attempts": attempt + 1},
                )

                # Wire learning & analytics
                self._record_learning(task, task_id, output)

                if self._on_task_complete:
                    try:
                        self._on_task_complete(task)
                    except Exception as e:
                        logger.warning("Task complete callback failed: %s", e)

                self._save_checkpoint(graph.plan_id, graph)
                return {"status": "completed", "output": task.output_data}

            except Exception as e:
                last_error = str(e)
                task.retry_count = attempt + 1
                logger.warning(f"Task {task_id} attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2**attempt)

        # All attempts failed
        task.status = TaskStatus.FAILED
        task.error = last_error
        task.completed_at = datetime.now().isoformat()

        self._emit_event(
            "task_failed",
            task_id,
            {"status": "failed", "error": last_error, "attempts": max_attempts},
        )

        if self._on_task_fail:
            try:
                self._on_task_fail(task)
            except Exception as e:
                logger.warning("Task fail callback failed: %s", e)

        self._save_checkpoint(graph.plan_id, graph)
        return {"status": "failed", "error": last_error}

    @staticmethod
    def _record_learning(task: TaskNode, task_id: str, output: Any) -> None:
        """Record task outcome for the learning pipeline (non-fatal)."""
        try:
            output_str = output if isinstance(output, str) else str(output)[:800]
            model_id = task.input_data.get("assigned_model") or task.assigned_model or "default"
            task_type_str = task.task_type.lower() if hasattr(task, "task_type") and task.task_type else "general"

            from vetinari.learning.quality_scorer import get_quality_scorer

            scorer = get_quality_scorer()
            q_score = scorer.score(
                task_id=task_id,
                model_id=model_id,
                task_type=task_type_str,
                task_description=task.description or "",
                output=output_str,
                use_llm=False,
            )

            from vetinari.learning.feedback_loop import get_feedback_loop

            get_feedback_loop().record_outcome(
                task_id=task_id,
                model_id=model_id,
                task_type=task_type_str,
                quality_score=q_score.overall_score,
                success=True,
            )

            from vetinari.learning.model_selector import get_thompson_selector

            get_thompson_selector().update(model_id, task_type_str, q_score.overall_score, True)
        except Exception as _learn_err:
            logger.debug("Learning hook failed (non-fatal): %s", _learn_err)

    def _handle_layer_failure(self, graph: ExecutionGraph, failed_tasks: list[TaskNode]) -> None:
        """Handle failure in a layer — cancel dependent tasks transitively."""
        cancelled_ids: set[str] = {t.id for t in failed_tasks}

        changed = True
        while changed:
            changed = False
            for node in graph.nodes.values():
                if node.status in (
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                ):
                    continue
                if any(dep in cancelled_ids for dep in node.depends_on):  # noqa: SIM102
                    if node.id not in cancelled_ids:
                        node.status = TaskStatus.CANCELLED
                        cancelled_ids.add(node.id)
                        self._emit_event(
                            "task_cancelled",
                            node.id,
                            {
                                "reason": "dependency_failed",
                                "failed_dependencies": [dep for dep in node.depends_on if dep in cancelled_ids],
                            },
                        )
                        changed = True

    def _emit_event(self, event_type: str, task_id: str, data: dict[str, Any]) -> None:
        """Emit an execution event."""
        event = ExecutionEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            task_id=task_id,
            timestamp=datetime.now().isoformat(),
            data=data,
        )
        self._event_history.append(event)
        logger.debug("Event: %s - %s", event_type, task_id)

    def _save_checkpoint(self, plan_id: str, graph: ExecutionGraph) -> None:
        """Save a checkpoint of the execution state."""
        checkpoint = Checkpoint(
            checkpoint_id=str(uuid.uuid4()),
            plan_id=plan_id,
            created_at=datetime.now().isoformat(),
            graph_state=graph.to_dict(),
            completed_tasks=[t.id for t in graph.get_completed_tasks()],
            running_tasks=[t.id for t in graph.nodes.values() if t.status == TaskStatus.RUNNING],
            metadata={"event_count": len(self._event_history)},
        )

        checkpoint_file = self.checkpoint_dir / f"{plan_id}_checkpoint.json"
        with open(checkpoint_file, "w") as f:
            json.dump(
                {
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "plan_id": checkpoint.plan_id,
                    "created_at": checkpoint.created_at,
                    "graph_state": checkpoint.graph_state,
                    "completed_tasks": checkpoint.completed_tasks,
                    "running_tasks": checkpoint.running_tasks,
                    "metadata": checkpoint.metadata,
                },
                f,
                indent=2,
            )

    def load_checkpoint(self, plan_id: str) -> ExecutionGraph | None:
        """Load a checkpoint to resume execution."""
        checkpoint_file = self.checkpoint_dir / f"{plan_id}_checkpoint.json"

        if not checkpoint_file.exists():
            logger.warning("No checkpoint found for plan: %s", plan_id)
            return None

        with open(checkpoint_file) as f:
            data = json.load(f)

        graph_data = data["graph_state"]
        graph = ExecutionGraph(
            plan_id=graph_data["plan_id"],
            goal=graph_data["goal"],
            created_at=graph_data["created_at"],
            updated_at=graph_data["updated_at"],
            status=PlanStatus(graph_data["status"]),
            current_layer=graph_data.get("current_layer", 0),
            completed_count=graph_data.get("completed_count", 0),
            failed_count=graph_data.get("failed_count", 0),
        )

        for node_id, node_data in graph_data["nodes"].items():
            graph.nodes[node_id] = TaskNode.from_dict(node_data)

        with self._execution_lock:
            self._active_executions[plan_id] = graph

        logger.info("Loaded checkpoint for plan: %s", plan_id)
        return graph

    def recover_execution(self, plan_id: str) -> dict[str, Any]:
        """Recover and continue an execution from checkpoint."""
        graph = self.load_checkpoint(plan_id)

        if not graph:
            return {"status": "error", "message": "No checkpoint found"}

        # Reset failed tasks for retry
        for node in graph.nodes.values():
            if node.status == TaskStatus.FAILED and node.retry_count < node.max_retries:
                node.status = TaskStatus.PENDING
                node.error = ""

        incomplete = [
            n for n in graph.nodes.values() if n.status in (TaskStatus.PENDING, TaskStatus.BLOCKED, TaskStatus.FAILED)
        ]
        logger.info(f"Recovering {len(incomplete)} incomplete tasks for plan: {plan_id}")

        return self.execute_plan(graph)

    def get_execution_status(self, plan_id: str) -> dict[str, Any] | None:
        """Get the status of an execution."""
        with self._execution_lock:
            graph = self._active_executions.get(plan_id)

        if not graph:
            graph = self.load_checkpoint(plan_id)

        if not graph:
            return None

        return {
            "plan_id": plan_id,
            "status": graph.status.value,
            "total_tasks": len(graph.nodes),
            "completed": len(graph.get_completed_tasks()),
            "failed": len(graph.get_failed_tasks()),
            "blocked": len(graph.get_blocked_tasks()),
            "progress": (len(graph.get_completed_tasks()) / len(graph.nodes) if graph.nodes else 0),
        }

    def list_checkpoints(self) -> list[str]:
        """List all available checkpoints."""
        return [f.stem.replace("_checkpoint", "") for f in self.checkpoint_dir.glob("*_checkpoint.json")]
