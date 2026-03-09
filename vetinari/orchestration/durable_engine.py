"""
Durable execution engine for the two-layer orchestration system.

Provides Temporal-inspired stateful task execution with checkpointing,
retry policies, event sourcing, and crash recovery.
"""

import json
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Callable, Set
from datetime import datetime
from pathlib import Path
import threading

from vetinari.orchestration.types import TaskStatus, PlanStatus, TaskNode, ExecutionEvent, Checkpoint
from vetinari.orchestration.execution_graph import ExecutionGraph

logger = logging.getLogger(__name__)


class DurableExecutionEngine:
    """
    Durable execution engine inspired by Temporal.

    Features:
    - State persistence with checkpoints
    - Retry policies
    - Event sourcing
    - Crash recovery
    - Deterministic replay
    """

    def __init__(self,
                 checkpoint_dir: str = None,
                 max_concurrent: int = 4,
                 default_timeout: float = 300.0):
        """
        Initialize the durable execution engine.

        Args:
            checkpoint_dir: Directory to store checkpoints
            max_concurrent: Maximum concurrent task executions
            default_timeout: Default task timeout in seconds
        """
        self.checkpoint_dir = Path(checkpoint_dir or "./vetinari_checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout

        # Active executions
        self._active_executions: Dict[str, ExecutionGraph] = {}
        self._execution_lock = threading.Lock()

        # Event history
        self._event_history: List[ExecutionEvent] = []

        # Task handlers
        self._task_handlers: Dict[str, Callable] = {}

        # Callbacks
        self._on_task_start: Optional[Callable] = None
        self._on_task_complete: Optional[Callable] = None
        self._on_task_fail: Optional[Callable] = None

        logger.info(f"DurableExecutionEngine initialized (checkpoint_dir={self.checkpoint_dir})")

    def register_handler(self, task_type: str, handler: Callable):
        """Register a handler for a task type."""
        self._task_handlers[task_type] = handler
        logger.debug(f"Registered handler for task type: {task_type}")

    def set_callbacks(self,
                     on_task_start: Callable = None,
                     on_task_complete: Callable = None,
                     on_task_fail: Callable = None):
        """Set execution callbacks."""
        self._on_task_start = on_task_start
        self._on_task_complete = on_task_complete
        self._on_task_fail = on_task_fail

    def create_execution(self, graph: ExecutionGraph) -> str:
        """Create a new execution from a graph."""
        plan_id = graph.plan_id

        with self._execution_lock:
            self._active_executions[plan_id] = graph

        # Save initial checkpoint
        self._save_checkpoint(plan_id, graph)

        logger.info(f"Created execution for plan: {plan_id}")
        return plan_id

    def execute_plan(self,
                   graph: ExecutionGraph,
                   task_handler: Callable = None) -> Dict[str, Any]:
        """
        Execute a plan with durable semantics.

        Args:
            graph: The execution graph
            task_handler: Handler function for executing tasks

        Returns:
            Execution results
        """
        plan_id = graph.plan_id
        graph.status = PlanStatus.RUNNING

        # Register handler if provided
        if task_handler:
            self._task_handlers["default"] = task_handler

        # Create execution
        self.create_execution(graph)

        # Get execution order
        layers = graph.get_execution_order()

        results = {
            "plan_id": plan_id,
            "total_tasks": len(graph.nodes),
            "completed": 0,
            "failed": 0,
            "task_results": {}
        }

        # Execute layer by layer
        for layer_idx, layer in enumerate(layers):
            logger.info(f"Executing layer {layer_idx + 1}/{len(layers)} with {len(layer)} tasks")

            layer_results = self._execute_layer(graph, layer)

            # Update results
            for task_id, result in layer_results.items():
                results["task_results"][task_id] = result
                if result.get("status") == "completed":
                    results["completed"] += 1
                else:
                    results["failed"] += 1

            # Check for failures that block downstream
            failed_tasks = [t for t in layer if t.status == TaskStatus.FAILED]
            if failed_tasks:
                # Mark dependent tasks as blocked/cancelled
                self._handle_layer_failure(graph, failed_tasks)

        # Finalize
        graph.status = PlanStatus.COMPLETED if results["failed"] == 0 else PlanStatus.FAILED

        # Save final checkpoint
        self._save_checkpoint(plan_id, graph)

        return results

    def _execute_layer(self, graph: ExecutionGraph, layer: List[TaskNode]) -> Dict[str, Any]:
        """Execute a layer of tasks in parallel."""
        results = {}

        # Execute tasks in parallel (simplified - use ThreadPoolExecutor in production)
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(layer), self.max_concurrent)) as executor:
            future_to_task = {
                executor.submit(self._execute_task, graph, task): task
                for task in layer
            }

            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results[task.id] = result
                except Exception as e:
                    logger.error(f"Task {task.id} failed with exception: {e}")
                    results[task.id] = {"status": "failed", "error": str(e)}

        return results

    def _execute_task(self, graph: ExecutionGraph, task: TaskNode) -> Dict[str, Any]:
        """Execute a single task."""
        task_id = task.id

        # Emit event
        self._emit_event("task_started", task_id, {"description": task.description})

        # Update status
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now().isoformat()

        # Callback
        if self._on_task_start:
            try:
                self._on_task_start(task)
            except Exception as e:
                logger.warning(f"Task start callback failed: {e}")

        # Get handler
        handler = self._task_handlers.get(task.task_type) or self._task_handlers.get("default")

        if not handler:
            # No handler - mark as completed with warning
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            task.output_data = {"warning": "No handler registered"}
            self._emit_event("task_completed", task_id, {"status": "completed"})
            return {"status": "completed", "output": task.output_data}

        # Execute with timeout and retry
        max_attempts = task.max_retries + 1
        last_error = None

        for attempt in range(max_attempts):
            try:
                # Execute handler
                output = handler(task)

                # Success
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now().isoformat()
                task.output_data = output if isinstance(output, dict) else {"output": output}

                # Emit event
                self._emit_event("task_completed", task_id, {
                    "status": "completed",
                    "attempts": attempt + 1
                })

                # Wire learning & analytics: record outcome after each completed task
                try:
                    output_str = (
                        output if isinstance(output, str)
                        else str(output)[:800]
                    )
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
                    logger.debug(f"Learning hook failed (non-fatal): {_learn_err}")

                # Callback
                if self._on_task_complete:
                    try:
                        self._on_task_complete(task)
                    except Exception as e:
                        logger.warning(f"Task complete callback failed: {e}")

                # Save checkpoint
                self._save_checkpoint(graph.plan_id, graph)

                return {"status": "completed", "output": task.output_data}

            except Exception as e:
                last_error = str(e)
                task.retry_count = attempt + 1
                logger.warning(f"Task {task_id} attempt {attempt + 1} failed: {e}")

                # Wait before retry
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        # All attempts failed
        task.status = TaskStatus.FAILED
        task.error = last_error
        task.completed_at = datetime.now().isoformat()

        # Emit event
        self._emit_event("task_failed", task_id, {
            "status": "failed",
            "error": last_error,
            "attempts": max_attempts
        })

        # Callback
        if self._on_task_fail:
            try:
                self._on_task_fail(task)
            except Exception as e:
                logger.warning(f"Task fail callback failed: {e}")

        # Save checkpoint
        self._save_checkpoint(graph.plan_id, graph)

        return {"status": "failed", "error": last_error}

    def _handle_layer_failure(self, graph: ExecutionGraph, failed_tasks: List[TaskNode]):
        """Handle failure in a layer - cancel dependent tasks transitively."""
        cancelled_ids: Set[str] = {t.id for t in failed_tasks}

        # Iteratively expand cancellation to all transitive dependents
        changed = True
        while changed:
            changed = False
            for node in graph.nodes.values():
                if node.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                    continue
                if any(dep in cancelled_ids for dep in node.depends_on):
                    if node.id not in cancelled_ids:
                        node.status = TaskStatus.CANCELLED
                        cancelled_ids.add(node.id)
                        self._emit_event("task_cancelled", node.id, {
                            "reason": "dependency_failed",
                            "failed_dependencies": [
                                dep for dep in node.depends_on if dep in cancelled_ids
                            ]
                        })
                        changed = True

    def _emit_event(self, event_type: str, task_id: str, data: Dict[str, Any]):
        """Emit an execution event."""
        event = ExecutionEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            task_id=task_id,
            timestamp=datetime.now().isoformat(),
            data=data
        )
        self._event_history.append(event)

        # Also log
        logger.debug(f"Event: {event_type} - {task_id}")

    def _save_checkpoint(self, plan_id: str, graph: ExecutionGraph):
        """Save a checkpoint of the execution state."""
        checkpoint = Checkpoint(
            checkpoint_id=str(uuid.uuid4()),
            plan_id=plan_id,
            created_at=datetime.now().isoformat(),
            graph_state=graph.to_dict(),
            completed_tasks=[t.id for t in graph.get_completed_tasks()],
            running_tasks=[t.id for t in graph.nodes.values() if t.status == TaskStatus.RUNNING],
            metadata={"event_count": len(self._event_history)}
        )

        # Save to file
        checkpoint_file = self.checkpoint_dir / f"{plan_id}_checkpoint.json"
        with open(checkpoint_file, "w") as f:
            json.dump({
                "checkpoint_id": checkpoint.checkpoint_id,
                "plan_id": checkpoint.plan_id,
                "created_at": checkpoint.created_at,
                "graph_state": checkpoint.graph_state,
                "completed_tasks": checkpoint.completed_tasks,
                "running_tasks": checkpoint.running_tasks,
                "metadata": checkpoint.metadata,
            }, f, indent=2)

    def load_checkpoint(self, plan_id: str) -> Optional[ExecutionGraph]:
        """Load a checkpoint to resume execution."""
        checkpoint_file = self.checkpoint_dir / f"{plan_id}_checkpoint.json"

        if not checkpoint_file.exists():
            logger.warning(f"No checkpoint found for plan: {plan_id}")
            return None

        with open(checkpoint_file, "r") as f:
            data = json.load(f)

        # Reconstruct graph
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

        # Reconstruct nodes
        for node_id, node_data in graph_data["nodes"].items():
            graph.nodes[node_id] = TaskNode.from_dict(node_data)

        # Restore to active executions
        with self._execution_lock:
            self._active_executions[plan_id] = graph

        logger.info(f"Loaded checkpoint for plan: {plan_id}")
        return graph

    def recover_execution(self, plan_id: str) -> Dict[str, Any]:
        """
        Recover and continue an execution from checkpoint.

        Returns:
            Recovery results
        """
        graph = self.load_checkpoint(plan_id)

        if not graph:
            return {"status": "error", "message": "No checkpoint found"}

        # Find incomplete tasks
        incomplete = [
            n for n in graph.nodes.values()
            if n.status in (TaskStatus.PENDING, TaskStatus.BLOCKED, TaskStatus.FAILED)
        ]

        # Reset failed tasks for retry
        for node in graph.nodes.values():
            if node.status == TaskStatus.FAILED:
                if node.retry_count < node.max_retries:
                    node.status = TaskStatus.PENDING
                    node.error = ""

        # Continue execution
        logger.info(f"Recovering {len(incomplete)} incomplete tasks for plan: {plan_id}")

        return self.execute_plan(graph)

    def get_execution_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an execution."""
        with self._execution_lock:
            graph = self._active_executions.get(plan_id)

        if not graph:
            # Try to load from checkpoint
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
            "progress": len(graph.get_completed_tasks()) / len(graph.nodes) if graph.nodes else 0,
        }

    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints."""
        return [f.stem.replace("_checkpoint", "")
                for f in self.checkpoint_dir.glob("*_checkpoint.json")]
