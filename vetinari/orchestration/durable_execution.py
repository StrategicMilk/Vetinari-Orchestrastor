"""Durable Execution Engine — Layer 2 of the Two-Layer Orchestration System.

Checkpoint-based execution for long-running plans.

When to use this module:
    Use ``DurableExecutionEngine`` when a plan must survive crashes and be
    resumable.  Every task transition is written to SQLite before it happens,
    enabling deterministic replay on restart.  This is the right execution path
    for plans that span minutes or hours, or that must not repeat work already
    completed.

Pipeline role: Plan → **DurableExecution** (checkpoint) → Verify → Learn.
Compare with ``pipeline_engine.py`` (in-memory, no persistence) and
``async_executor.py`` (async wrapper for wave-based plans).

Database types and the SQLite wrapper live in ``durable_db`` and are
re-exported here so existing callers do not need to change their imports.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import random
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vetinari.constants import _PROJECT_ROOT
from vetinari.orchestration.checkpoint_store import Checkpoint, ExecutionEvent
from vetinari.orchestration.durable_db import _DatabaseManager
from vetinari.orchestration.durable_execution_recovery import (
    answer_paused_questions as _answer_paused_questions,
)
from vetinari.orchestration.durable_execution_recovery import (
    cleanup_completed as _cleanup_completed,
)
from vetinari.orchestration.durable_execution_recovery import (
    emit_event as _emit_event_fn,
)
from vetinari.orchestration.durable_execution_recovery import (
    get_execution_status as _get_execution_status,
)
from vetinari.orchestration.durable_execution_recovery import (
    get_paused_questions as _get_paused_questions,
)
from vetinari.orchestration.durable_execution_recovery import (
    handle_layer_failure as _handle_layer_failure_fn,
)
from vetinari.orchestration.durable_execution_recovery import (
    list_checkpoints as _list_checkpoints,
)
from vetinari.orchestration.durable_execution_recovery import (
    load_checkpoint as _load_checkpoint,
)
from vetinari.orchestration.durable_execution_recovery import (
    record_learning as _record_learning,
)
from vetinari.orchestration.durable_execution_recovery import (
    recover_execution as _recover_execution,
)
from vetinari.orchestration.durable_execution_recovery import (
    recover_incomplete_executions as _recover_incomplete_executions,
)
from vetinari.orchestration.durable_execution_recovery import (
    save_checkpoint as _save_checkpoint,
)
from vetinari.orchestration.durable_execution_recovery import (
    save_paused_questions as _save_paused_questions,
)
from vetinari.orchestration.execution_graph import ExecutionGraph, ExecutionTaskNode
from vetinari.types import PlanStatus, StatusEnum

logger = logging.getLogger(__name__)

__all__ = [
    "Checkpoint",
    "DurableExecutionEngine",
    "ExecutionEvent",
]


class DurableExecutionEngine:
    """Durable execution engine inspired by Temporal.

    Features:
    - State persistence with SQLite + WAL (crash-safe, atomic)
    - Retry policies with exponential backoff and jitter
    - Event sourcing via execution_events table
    - Crash recovery via checkpoint resume
    - Deterministic replay
    - Pause/resume for user clarification questions
    - Circuit breaker to prevent cascading failures
    - Cycle detection to prevent infinite retry loops
    """

    def __init__(
        self,
        checkpoint_dir: str | None = None,
        max_concurrent: int = 4,
        default_timeout: float = 300.0,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else _PROJECT_ROOT / "vetinari_checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout

        # SQLite database for crash-safe checkpointing. When checkpoint_dir
        # is provided (e.g. in tests), use a standalone db file in that
        # directory. Otherwise (production), pass None to delegate to the
        # unified vetinari.database module (ADR-0072).
        db_path = (self.checkpoint_dir / "execution_state.db") if checkpoint_dir else None
        self._db = _DatabaseManager(db_path)

        # Checkpoint store facade — higher-level persistence operations that
        # delegate to the same SQLite database as _db.  Used by the recovery
        # functions to call named methods (save_event, load_checkpoint_graph_json,
        # find_incomplete_ids, etc.) rather than raw SQL strings.
        from vetinari.orchestration.checkpoint_store import CheckpointStore

        self._checkpoint_store = CheckpointStore(checkpoint_dir=self.checkpoint_dir if checkpoint_dir else None)

        # Active executions indexed by plan_id
        self._active_executions: dict[str, ExecutionGraph] = {}
        self._execution_lock = threading.Lock()

        # Cycle detection — prevents infinite retry/rework loops (max 10 executions per task)
        from vetinari.orchestration.graph_types import CycleDetector, HumanCheckpoint

        self._cycle_detector = CycleDetector(max_iterations=10)

        # Human checkpoint registry — tasks added here require explicit approval
        # before their results propagate to downstream dependents.
        self._human_checkpoint = HumanCheckpoint()

        # Circuit breaker — prevents cascading failures when models are unavailable.
        # Logged at ERROR when unavailable because cascading-failure protection is
        # a security-relevant safety property — operators must know it is absent.
        self._circuit_breaker = None
        self._circuit_breaker_degraded = False  # True when CB import failed; results carry degraded marker
        try:
            from vetinari.resilience import CircuitBreaker

            self._circuit_breaker = CircuitBreaker("durable_execution")
        except (ImportError, AttributeError):
            self._circuit_breaker_degraded = True
            logger.error(
                "Circuit breaker unavailable for durable execution — "
                "tasks will proceed without cascading-failure protection; "
                "results will carry '_circuit_breaker_degraded=True'"
            )

        # In-memory event history (last 1000 events) for fast access
        self._event_history: deque[ExecutionEvent] = deque(maxlen=1000)

        # Task handlers keyed by task_type; "default" is the fallback
        self._task_handlers: dict[str, Callable] = {}

        # Lifecycle callbacks — optional, called on each task state change
        self._on_task_start: Callable | None = None
        self._on_task_complete: Callable | None = None
        self._on_task_fail: Callable | None = None

        # Heartbeat tracking — detect stuck tasks that stop reporting progress
        self._heartbeats: dict[str, float] = {}  # task_id -> last heartbeat time
        self._heartbeat_timeout = default_timeout

        # Shared thread pool — created once, reused across all layer executions
        # to avoid the overhead of spawning and destroying workers per layer.
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent)

        logger.info(
            "DurableExecutionEngine initialized (checkpoint_dir=%s, backend=sqlite+wal)",
            self.checkpoint_dir,
        )

    def record_heartbeat(self, task_id: str) -> None:
        """Record that a task is still making progress.

        Call this periodically during long-running operations (e.g. LLM
        inference) to prevent the task from being considered stuck.

        Args:
            task_id: The task reporting activity.
        """
        self._heartbeats[task_id] = time.time()

    def is_task_stuck(self, task_id: str) -> bool:
        """Check if a task has missed its heartbeat deadline.

        Args:
            task_id: The task to check.

        Returns:
            True if the task has not sent a heartbeat within the timeout
            window. False if no heartbeat has been registered yet (task
            has not started).
        """
        last_beat = self._heartbeats.get(task_id)
        if last_beat is None:
            return False  # No heartbeat registered yet — task hasn't started
        return (time.time() - last_beat) > self._heartbeat_timeout

    def register_handler(self, task_type: str, handler: Callable) -> None:
        """Register a callable to handle tasks of a specific type.

        Args:
            task_type: The task type string to route to this handler.
            handler: Callable that accepts an ExecutionTaskNode and returns
                a result dict.
        """
        self._task_handlers[task_type] = handler
        logger.debug("Registered handler for task type: %s", task_type)

    def set_callbacks(
        self,
        on_task_start: Callable | None = None,
        on_task_complete: Callable | None = None,
        on_task_fail: Callable | None = None,
    ) -> None:
        """Set lifecycle callbacks for task state transitions.

        Args:
            on_task_start: Called when a task transitions to RUNNING.
            on_task_complete: Called when a task transitions to COMPLETED.
            on_task_fail: Called when a task exhausts all retries and fails.
        """
        self._on_task_start = on_task_start
        self._on_task_complete = on_task_complete
        self._on_task_fail = on_task_fail

    def create_execution(self, graph: ExecutionGraph) -> str:
        """Register a new execution and write its initial checkpoint.

        Args:
            graph: The execution graph to register.

        Returns:
            The plan_id that identifies this execution, used to load
            checkpoints or query status later.
        """
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
        """Execute a plan with durable semantics, layer by layer.

        Registers the execution, iterates over topologically sorted layers,
        runs each layer in parallel, and handles layer failures by cancelling
        dependent tasks transitively.

        Args:
            graph: The execution graph to run.
            task_handler: Optional default handler for all task types.
                Overrides any previously registered "default" handler.

        Returns:
            Dict with plan_id, total_tasks, completed count, failed count,
            and per-task results.
        """
        plan_id = graph.plan_id
        graph.status = PlanStatus.EXECUTING

        # Annotate the correlation context with this plan_id so all structured
        # log lines emitted during execution carry the plan identifier.
        try:
            from vetinari.structured_logging import CorrelationContext
            from vetinari.structured_logging import get_plan_id as _get_pid

            _ctx_plan_id = _get_pid()
            if not _ctx_plan_id:
                # No plan_id set yet — bind to the graph's plan_id so downstream
                # log processors can include it in every emitted log event.
                CorrelationContext.set_plan_id(plan_id)
            elif _ctx_plan_id != plan_id:
                logger.debug(
                    "CorrelationContext plan_id=%s differs from execution plan_id=%s — using execution plan_id",
                    _ctx_plan_id,
                    plan_id,
                )
        except Exception:
            logger.warning(
                "CorrelationContext plan_id annotation unavailable for plan %s — skipping", plan_id, exc_info=False
            )

        if task_handler:
            self._task_handlers["default"] = task_handler

        self.create_execution(graph)
        layers = graph.get_execution_order()

        results: dict[str, Any] = {
            "plan_id": plan_id,
            "total_tasks": len(graph.nodes),
            StatusEnum.COMPLETED.value: 0,
            StatusEnum.FAILED.value: 0,
            "task_results": {},
        }

        try:
            for layer_idx, layer in enumerate(layers):
                logger.info("Executing layer %s/%s with %s tasks", layer_idx + 1, len(layers), len(layer))
                layer_results = self._execute_layer(graph, layer)

                for task_id, result in layer_results.items():
                    results["task_results"][task_id] = result
                    if result.get("status") == StatusEnum.COMPLETED.value:
                        results[StatusEnum.COMPLETED.value] += 1
                    else:
                        results[StatusEnum.FAILED.value] += 1

                failed_tasks = [t for t in layer if t.status == StatusEnum.FAILED]
                if failed_tasks:
                    self._handle_layer_failure(graph, failed_tasks)

            graph.status = PlanStatus.COMPLETED if results[StatusEnum.FAILED.value] == 0 else PlanStatus.FAILED
            self._save_checkpoint(plan_id, graph)
        finally:
            # Remove the completed (or errored) graph from active executions so it does
            # not accumulate indefinitely — the checkpoint on disk is the durable record.
            with self._execution_lock:
                self._active_executions.pop(plan_id, None)

        return results

    def _execute_layer(self, graph: ExecutionGraph, layer: list[ExecutionTaskNode]) -> dict[str, Any]:
        """Execute a layer of tasks in parallel using the shared thread pool.

        Args:
            graph: The execution graph that owns the task layer.
            layer: Tasks in this dependency layer, all of which can run concurrently.

        Returns:
            Mapping of task ID to result dict for every task in the layer.
        """
        results: dict[str, Any] = {}
        future_to_task = {self._executor.submit(self._execute_task, graph, task): task for task in layer}
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results[task.id] = result
            except Exception as e:
                logger.error("Task %s failed with exception: %s", task.id, e)
                results[task.id] = {"status": StatusEnum.FAILED.value, "error": str(e)}
        return results

    def shutdown(self, *, wait: bool = True) -> None:
        """Shut down the shared thread pool.

        Call this when the engine will no longer be used to ensure clean
        resource release.  After ``shutdown()`` the engine must not be
        used to execute further tasks.

        Args:
            wait: If True (default), block until all pending futures
                complete.  Pass False for non-blocking teardown (e.g.
                in test fixtures or ``__del__``).
        """
        self._executor.shutdown(wait=wait)

    def __del__(self) -> None:
        """Release the thread pool on garbage collection as a safety net.

        Production code should call ``shutdown()`` explicitly; this
        prevents zombie threads when callers forget.
        """
        with contextlib.suppress(Exception):
            self._executor.shutdown(wait=False)

    def _execute_task(self, graph: ExecutionGraph, task: ExecutionTaskNode) -> dict[str, Any]:
        """Execute a single task with retry logic and cycle detection.

        Args:
            graph: The parent execution graph (needed to save checkpoints).
            task: The task node to execute.

        Returns:
            Result dict with status, output, and optional error/tokens_used.
        """
        task_id = task.id

        # Skip cancelled tasks immediately — _handle_layer_failure marks downstream
        # tasks CANCELLED when their dependency fails, but those tasks are already in
        # the pre-computed layer list.  Running them would produce spurious output and
        # inflate the handler call count (fail-closed principle: no phantom completions).
        if task.status == StatusEnum.CANCELLED:
            logger.info("Task %s is CANCELLED — skipping execution", task_id)
            return {"status": StatusEnum.CANCELLED.value, "reason": "dependency_failed"}

        # Cycle detection: prevent infinite re-execution of the same task.
        # Log the current execution count before recording so we can surface
        # high-iteration tasks before they hit the hard limit.
        _prior_count = self._cycle_detector.get_count(task_id)
        if _prior_count > 0:
            logger.debug(
                "Task %s has been attempted %d time(s) — cycle detector count before this run",
                task_id,
                _prior_count,
            )
        try:
            self._cycle_detector.record_execution(task_id)
        except RuntimeError as cycle_err:
            logger.error("Cycle detected for task %s: %s", task_id, cycle_err)
            task.status = StatusEnum.FAILED
            task.error = str(cycle_err)
            task.completed_at = datetime.now(timezone.utc).isoformat()
            self._emit_event("task_failed", task_id, {"error": str(cycle_err), "reason": "cycle_detected"})
            return {"status": StatusEnum.FAILED.value, "error": str(cycle_err)}

        # Human-in-the-loop checkpoint: if this task requires approval and has
        # not yet been approved, block execution and return a waiting status.
        if self._human_checkpoint.is_checkpoint(task_id) and not self._human_checkpoint.is_approved(task_id):
            logger.info(
                "Task %s is a human checkpoint and has not been approved — deferring execution",
                task_id,
            )
            return {
                "status": StatusEnum.WAITING.value,
                "waiting_for": "human_approval",
                "task_id": task_id,
            }

        self._emit_event("task_started", task_id, {"description": task.description})
        task.status = StatusEnum.RUNNING
        task.started_at = datetime.now(timezone.utc).isoformat()

        if self._on_task_start:
            try:
                self._on_task_start(task)
            except Exception as e:
                logger.warning("Task start callback failed: %s", e)

        handler = self._task_handlers.get(task.task_type) or self._task_handlers.get("default")

        if not handler:
            # No handler means the task cannot be executed — mark FAILED, not COMPLETED.
            # A completed task with no handler inflates the success count and hides
            # misconfiguration (fail-closed principle).
            task.status = StatusEnum.FAILED
            task.error = "No handler registered for task type"
            task.completed_at = datetime.now(timezone.utc).isoformat()
            self._emit_event(
                "task_failed",
                task_id,
                {"status": StatusEnum.FAILED.value, "error": task.error},
            )
            return {"status": StatusEnum.FAILED.value, "error": task.error}

        max_attempts = task.max_retries + 1
        last_error = None

        for attempt in range(max_attempts):
            # Heartbeat: record that this task is still actively running so
            # is_task_stuck() returns False while execution is in progress.
            self.record_heartbeat(task_id)

            # Circuit breaker check — skip if circuit is open
            if self._circuit_breaker is not None and not self._circuit_breaker.allow_request():
                logger.warning(
                    "Circuit breaker OPEN for task %s — skipping attempt %d",
                    task_id,
                    attempt + 1,
                )
                last_error = "Circuit breaker open — too many recent failures"
                break

            try:
                output = handler(task)

                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_success()

                # A structured-failure dict like {"success": False, "error": "..."} is
                # truthy but explicitly signals failure — treat it the same as empty
                # output so the retry mechanism engages (fail-closed principle).
                _structured_failure = isinstance(output, dict) and output.get("success") is False
                _has_output = bool(output) and output != {"output": ""} and not _structured_failure

                if not _has_output:
                    # Empty output means the handler returned nothing useful — treat as
                    # failure so the retry mechanism can attempt recovery rather than
                    # silently recording a zero-content completion (fail-closed principle).
                    task.status = StatusEnum.FAILED
                    task.error = "Task produced empty output"
                    task.completed_at = datetime.now(timezone.utc).isoformat()
                    logger.warning(
                        "Task %s produced empty output — marking as FAILED",
                        task_id,
                    )
                    self._emit_event(
                        "task_failed",
                        task_id,
                        {"status": StatusEnum.FAILED.value, "error": task.error},
                    )
                    # Count as a failed attempt so retry logic fires on next loop iteration
                    last_error = task.error
                    if self._circuit_breaker is not None:
                        self._circuit_breaker.record_failure()
                    if attempt < max_attempts - 1:
                        base_delay = 2**attempt
                        jitter = random.uniform(0, 1) * base_delay  # noqa: S311 - deterministic randomness is non-cryptographic
                        time.sleep(base_delay + jitter)
                    continue

                task.status = StatusEnum.COMPLETED
                task.completed_at = datetime.now(timezone.utc).isoformat()
                task.output_data = output if isinstance(output, dict) else {"output": output}

                self._emit_event(
                    "task_completed",
                    task_id,
                    {"status": StatusEnum.COMPLETED.value, "attempts": attempt + 1},
                )

                self._record_learning(task, task_id, output)

                if self._on_task_complete:
                    try:
                        self._on_task_complete(task)
                    except Exception as e:
                        logger.warning("Task complete callback failed: %s", e)

                # Update PlanManager task status so plan-level progress tracking
                # reflects the completed task without requiring a separate callback.
                _wave_id = task.input_data.get("wave_id", "") if task.input_data else ""
                try:
                    from vetinari.planning import get_plan_manager

                    get_plan_manager().update_task_status(
                        plan_id=graph.plan_id,
                        wave_id=_wave_id,
                        task_id=task_id,
                        status=StatusEnum.COMPLETED.value,
                        result=task.output_data,
                    )
                except Exception as _pm_exc:
                    logger.warning(
                        "PlanManager task status update skipped for %s — plan not tracked: %s",
                        task_id,
                        _pm_exc,
                    )

                self._save_checkpoint(graph.plan_id, graph)

                # Propagate tokens_used from agent metadata into result dict
                # so the TLO pipeline can aggregate total token usage.
                _tokens = 0
                if isinstance(output, dict):
                    _tokens = output.get("tokens_used", 0)
                    if not _tokens and "metadata" in output:
                        _tokens = output["metadata"].get("tokens_used", 0)
                _result: dict[str, Any] = {
                    "status": StatusEnum.COMPLETED.value,
                    "output": task.output_data,
                    "tokens_used": _tokens,
                    "metadata": task.output_data.get("metadata", {}) if isinstance(task.output_data, dict) else {},
                }
                if self._circuit_breaker_degraded:
                    # Propagate degraded safety marker so downstream consumers know this
                    # task ran without circuit-breaker protection.
                    _result["_circuit_breaker_degraded"] = True
                return _result

            except Exception as e:
                last_error = str(e)
                task.retry_count = attempt + 1
                logger.warning("Task %s attempt %s failed: %s", task_id, attempt + 1, e)

                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_failure()

                if attempt < max_attempts - 1:
                    # Exponential backoff with jitter to prevent thundering herd
                    base_delay = 2**attempt
                    jitter = random.uniform(0, 1) * base_delay  # noqa: S311 - deterministic randomness is non-cryptographic
                    time.sleep(base_delay + jitter)

        # All attempts exhausted
        task.status = StatusEnum.FAILED
        task.error = last_error
        task.completed_at = datetime.now(timezone.utc).isoformat()

        self._emit_event(
            "task_failed",
            task_id,
            {"status": StatusEnum.FAILED.value, "error": last_error, "attempts": max_attempts},
        )

        if self._on_task_fail:
            try:
                self._on_task_fail(task)
            except Exception as e:
                logger.warning("Task fail callback failed: %s", e)

        # Update PlanManager task status so plan-level failure tracking is accurate.
        _wave_id_fail = task.input_data.get("wave_id", "") if task.input_data else ""
        try:
            from vetinari.planning import get_plan_manager

            get_plan_manager().update_task_status(
                plan_id=graph.plan_id,
                wave_id=_wave_id_fail,
                task_id=task_id,
                status=StatusEnum.FAILED.value,
                error=last_error or "unknown error",
            )
        except Exception as _pm_fail_exc:
            logger.warning(
                "PlanManager task status update (failed) skipped for %s — plan not tracked: %s",
                task_id,
                _pm_fail_exc,
            )

        self._save_checkpoint(graph.plan_id, graph)
        return {"status": StatusEnum.FAILED.value, "error": last_error}

    @staticmethod
    def _record_learning(task: ExecutionTaskNode, task_id: str, output: Any) -> None:
        """Record task outcome for learning pipeline. See ``durable_execution_recovery``."""
        _record_learning(task, task_id, output)

    def _handle_layer_failure(self, graph: ExecutionGraph, failed_tasks: list[ExecutionTaskNode]) -> None:
        """Cancel transitive dependants of failed tasks. See ``durable_execution_recovery``."""
        _handle_layer_failure_fn(self, graph, failed_tasks)

    def _emit_event(
        self,
        event_type: str,
        task_id: str,
        data: dict[str, Any],
        execution_id: str = "",
    ) -> None:
        """Emit and persist an execution event. See ``durable_execution_recovery``."""
        if not execution_id:
            with self._execution_lock:
                for plan_id, graph in self._active_executions.items():
                    if task_id in graph.nodes:
                        execution_id = plan_id
                        break
        _emit_event_fn(self, event_type, task_id, data, execution_id)

    # ------------------------------------------------------------------
    # Checkpoint / recovery — implementation in durable_execution_recovery.py
    # ------------------------------------------------------------------

    def _save_checkpoint(self, plan_id: str, graph: ExecutionGraph) -> None:
        """Persist execution state to SQLite. See ``durable_execution_recovery``."""
        _save_checkpoint(self, plan_id, graph)

    def load_checkpoint(self, plan_id: str) -> ExecutionGraph | None:
        """Load persisted execution graph. See ``durable_execution_recovery``."""
        return _load_checkpoint(self, plan_id)

    def save_paused_questions(
        self,
        execution_id: str,
        questions: list[str],
        task_id: str | None = None,
    ) -> str:
        """Pause pipeline and persist user questions. Returns question_id."""
        return _save_paused_questions(self, execution_id, questions, task_id)

    def answer_paused_questions(self, question_id: str, answers: list[str]) -> None:
        """Store answers for paused questions, enabling pipeline resume.

        Args:
            question_id: The question set identifier returned by ``pause_for_input``.
            answers: Ordered list of answer strings matching the original questions.
        """
        _answer_paused_questions(self, question_id, answers)

    def get_paused_questions(self, execution_id: str) -> list[dict[str, Any]]:
        """Return all unanswered questions for an execution."""
        return _get_paused_questions(self, execution_id)

    def recover_execution(self, plan_id: str) -> dict[str, Any]:
        """Resume an execution from its last checkpoint. See ``durable_execution_recovery``."""
        return _recover_execution(self, plan_id)

    def get_execution_status(self, plan_id: str) -> dict[str, Any] | None:
        """Return current status dict for an active or checkpointed execution."""
        return _get_execution_status(self, plan_id)

    def list_checkpoints(self) -> list[str]:
        """Return sorted list of all persisted execution IDs."""
        return _list_checkpoints(self)

    def recover_incomplete_executions(
        self,
        task_handler: Callable | None = None,
    ) -> list[dict[str, Any]]:
        """Find and resume all incomplete executions. See ``durable_execution_recovery``."""
        return _recover_incomplete_executions(self, task_handler)

    def cleanup_completed(self, max_age_days: int = 30) -> int:
        """Delete completed executions older than *max_age_days* days."""
        return _cleanup_completed(self, max_age_days)

    def list_retention_candidates(self, older_than_seconds: float) -> list[str]:
        """Return execution IDs that finished more than *older_than_seconds* ago.

        Delegates to ``CheckpointStore.list_retention_candidates`` which queries
        ``completed_at`` and ``terminal_status`` to identify executions eligible
        for deletion.  Only executions that reached a terminal state (COMPLETED
        or FAILED) are returned — in-progress executions are never included.

        Args:
            older_than_seconds: Age threshold in seconds.

        Returns:
            List of execution IDs eligible for retention cleanup.
        """
        return self._checkpoint_store.list_retention_candidates(older_than_seconds)
