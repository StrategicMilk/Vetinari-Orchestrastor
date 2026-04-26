"""A2A task executor: routes incoming A2A tasks to the Vetinari pipeline.

The :class:`VetinariA2AExecutor` is the bridge between the external A2A
protocol world and Vetinari's internal three-agent factory pipeline.
Incoming A2A tasks carry a ``task_type`` string; the executor maps that
string to an ``(AgentType, mode)`` pair and dispatches accordingly.

All execution is synchronous and side-effect-free in this module — the
actual agent invocation is delegated to the appropriate internal agent via
the routing table.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from vetinari.types import AgentType, StatusEnum

logger = logging.getLogger(__name__)

# ── Optional orchestrator import ─────────────────────────────────────────────
# Imported at module level so tests can patch vetinari.a2a.executor.get_two_layer_orchestrator.
# Falls back to None-returning stub when the orchestration module is unavailable.

try:
    from vetinari.agents.contracts import AgentTask
    from vetinari.orchestration.two_layer import get_two_layer_orchestrator
except ImportError:  # orchestration layer not available (e.g. stripped install)
    logger.debug("TwoLayerOrchestrator not available — A2A executor will run in acknowledgement-only mode")
    AgentTask = None  # type: ignore[assignment,misc]

    def get_two_layer_orchestrator():  # type: ignore[misc]
        """Stub used when the orchestration module cannot be imported."""
        return None


# ── Status constants ─────────────────────────────────────────────────────────

STATUS_COMPLETED = StatusEnum.COMPLETED.value  # Task finished successfully via real execution
STATUS_FAILED = StatusEnum.FAILED.value  # Task encountered an unrecoverable error
STATUS_PENDING = StatusEnum.PENDING.value  # Task has been accepted, not yet started
STATUS_RUNNING = StatusEnum.RUNNING.value  # Task is actively being processed
STATUS_ACKNOWLEDGED = (
    StatusEnum.ACKNOWLEDGED.value
)  # Task accepted in degraded/standalone mode; orchestrator unavailable

# A2A-local recovery terminal state for tasks that were ACKNOWLEDGED on a
# previous run and cannot be re-executed on restart (orchestrator still
# unavailable).  This state is intentionally NOT added to StatusEnum in
# vetinari/types.py — it is an A2A internal recovery concept only.  The row
# is preserved (not deleted) so the audit trail remains intact.
STATUS_ORPHANED = "orphaned"


# ── Data types ───────────────────────────────────────────────────────────────


@dataclass
class A2ATask:
    """An incoming A2A task received from an external agent or caller.

    Attributes:
        task_id: Unique identifier for this task. Auto-generated if not
            provided.
        task_type: A2A task type string (e.g. ``"plan"``, ``"build"``).
        input_data: Arbitrary input payload for the task.
        metadata: Optional caller-supplied metadata (headers, trace IDs, etc.).
        status: Current lifecycle status of the task.
    """

    task_type: str
    input_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = STATUS_PENDING

    def __repr__(self) -> str:
        return f"A2ATask(task_id={self.task_id!r}, task_type={self.task_type!r}, status={self.status!r})"


@dataclass
class A2AResult:
    """The result produced by executing an :class:`A2ATask`.

    Attributes:
        task_id: Identifier of the task that produced this result.
        status: Final status — ``"completed"`` (real execution succeeded),
            ``"failed"`` (execution raised or task type unknown), or
            ``"acknowledged"`` (accepted in degraded/standalone mode without
            real execution).
        output_data: Structured output from the agent.
        error: Human-readable error description when ``status == StatusEnum.FAILED.value``.
    """

    task_id: str
    status: str
    output_data: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def __repr__(self) -> str:
        return f"A2AResult(task_id={self.task_id!r}, status={self.status!r})"

    def to_dict(self) -> dict:
        """Serialise this result to a plain dict for JSON transport.

        Returns:
            Dictionary representation of the result.
        """
        return {
            "taskId": self.task_id,
            "status": self.status,
            "outputData": self.output_data,
            "error": self.error,
        }


# ── Routing table type alias ─────────────────────────────────────────────────

# Maps A2A task type string → (AgentType, mode_string)
_RouteEntry = tuple[AgentType, str]
_RoutingTable = dict[str, _RouteEntry]


# ── Executor ─────────────────────────────────────────────────────────────────


class VetinariA2AExecutor:
    """Routes incoming A2A tasks to Vetinari's internal pipeline.

    Each recognised ``task_type`` string is mapped to a specific
    ``(AgentType, mode)`` pair via the routing table built at construction
    time.  Unknown task types receive a graceful ``failed`` result with a
    descriptive error rather than raising an exception.

    Execution modes and status semantics
    -------------------------------------
    **Normal mode** (orchestrator available):
        :meth:`_dispatch` invokes the ``TwoLayerOrchestrator`` for real
        execution.  The result is ``STATUS_COMPLETED`` on success or
        ``STATUS_FAILED`` if the orchestrator raises.

    **Degraded / standalone mode** (orchestrator unavailable):
        When the orchestrator is ``None`` or import fails, :meth:`_dispatch`
        returns a structured acknowledgement dict flagged with
        ``"_is_acknowledgement_only": True``.  :meth:`execute` detects this
        flag and marks the task ``STATUS_ACKNOWLEDGED`` — meaning the task
        was accepted and recorded but NOT yet executed.  Callers can
        distinguish real execution from acceptance via:

        - ``result.status == STATUS_ACKNOWLEDGED`` (not completed, not failed)
        - ``result.output_data.get("_is_acknowledgement_only") is True``

    **Recovery on restart:**
        At startup, :meth:`recover_pending_tasks` loads all PENDING, RUNNING,
        and ACKNOWLEDGED tasks from the database and attempts to re-execute
        each one.  Tasks that were ACKNOWLEDGED (orchestrator unavailable on
        the previous run) are given a second chance.  If the orchestrator is
        still unavailable, the task is transitioned to ``STATUS_ORPHANED`` —
        a terminal A2A-local state meaning "accepted but never executed across
        at least two process lifetimes."  The database row is preserved (not
        deleted) for audit purposes.

    Example::

        executor = VetinariA2AExecutor()
        task = A2ATask(task_type="build", input_data={"goal": "implement feature X"})
        result = executor.execute(task)
        # In normal mode:  result.status == STATUS_COMPLETED
        # In degraded mode: result.status == STATUS_ACKNOWLEDGED
    """

    def __init__(self, recover_on_init: bool = True) -> None:
        """Initialise the executor, build the routing table, and run startup recovery.

        Args:
            recover_on_init: Whether to attempt recovery of interrupted tasks
                from the database on startup.  Set to ``False`` in tests that
                pre-populate the database and want to control recovery timing.

        Side effects:
          - Creates a2a_tasks table in the unified database if it doesn't exist.
          - If ``recover_on_init`` is True, loads PENDING/RUNNING/ACKNOWLEDGED
            tasks from DB and re-executes them (or marks them orphaned).
          - Sets ``_recovery_run = True`` after the first recovery pass so
            subsequent constructor calls in the same process skip re-recovery.
        """
        self._routing_table: _RoutingTable = self._build_routing_table()
        self._init_persistence()

        if recover_on_init:
            self._run_startup_recovery()

        logger.info(
            "VetinariA2AExecutor initialised with %d route entries",
            len(self._routing_table),
        )

    def _run_startup_recovery(self) -> None:
        """Attempt to re-execute tasks that were interrupted before the previous shutdown.

        Recovery policy:
        - PENDING / RUNNING tasks are re-executed normally.
        - ACKNOWLEDGED tasks (orchestrator was unavailable last time) get one
          retry.  If the orchestrator is still unavailable they become ORPHANED
          (terminal state, row preserved for audit).
        - If re-execution raises, the task is marked FAILED and the error is
          persisted.
        - If the database is unavailable, recovery is skipped silently.

        Idempotency: this method sets ``_recovery_run = True`` on first call.
        If it is called again on the same executor instance (which should not
        happen in normal operation), it is a no-op.
        """
        # Idempotency guard: prevent double-recovery within the same process.
        if getattr(self, "_recovery_run", False):
            return
        self._recovery_run = True

        pending = self.recover_pending_tasks()
        if not pending:
            return

        logger.warning(
            "Recovered %d interrupted A2A task(s) from previous run — attempting re-execution",
            len(pending),
        )

        for task in pending:
            prior_status = task.status
            try:
                result = self.execute(task)
                if result.status == STATUS_ACKNOWLEDGED:
                    # Orchestrator still unavailable after restart — this task
                    # cannot make progress.  Transition to orphaned so it does
                    # not accumulate across further restarts.
                    logger.warning(
                        "A2A task id=%s (previously %s) is still unexecutable after restart — marking as orphaned",
                        task.task_id,
                        prior_status,
                    )
                    self._persist_orphaned(task.task_id)
                else:
                    logger.info(
                        "A2A task id=%s recovered with status=%s",
                        task.task_id,
                        result.status,
                    )
            except Exception as exc:
                logger.exception(
                    "A2A task id=%s raised during recovery re-execution — marking as failed: %s",
                    task.task_id,
                    exc,
                )
                failed_result = A2AResult(
                    task_id=task.task_id,
                    status=STATUS_FAILED,
                    error=f"Recovery re-execution failed: {exc}",
                )
                self._persist_result(task.task_id, failed_result)

    def _persist_orphaned(self, task_id: str) -> None:
        """Transition a task row to STATUS_ORPHANED in the database.

        The row is updated in place rather than deleted so the audit trail
        is preserved for the full task lifecycle.

        Args:
            task_id: Identifier of the task to transition.
        """
        if not getattr(self, "_db_available", False):
            return
        try:
            from vetinari.database import get_connection

            now = datetime.now(timezone.utc).isoformat()
            conn = get_connection()
            conn.execute(
                "UPDATE a2a_tasks SET status = ?, updated_at = ? WHERE task_id = ?",
                (STATUS_ORPHANED, now, task_id),
            )
            conn.commit()
        except Exception:
            logger.warning(
                "Failed to persist orphaned status for A2A task %s — status inconsistency possible",
                task_id,
            )

    def _init_persistence(self) -> None:
        """Create the a2a_tasks table for durable task state.

        Uses the unified database connection if available. Falls back
        to in-memory tracking if the database is unavailable.
        """
        try:
            from vetinari.database import get_connection

            conn = get_connection()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS a2a_tasks (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    input_json TEXT,
                    output_json TEXT,
                    error TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()
            self._db_available = True
        except Exception:
            logger.warning("A2A task persistence unavailable — tasks will not survive restart")
            self._db_available = False

    def _persist_task(self, task: A2ATask) -> None:
        """Persist task state to SQLite (best-effort, non-blocking)."""
        if not getattr(self, "_db_available", False):
            return
        try:
            import json

            from vetinari.database import get_connection

            now = datetime.now(timezone.utc).isoformat()
            conn = get_connection()
            conn.execute(
                """INSERT OR REPLACE INTO a2a_tasks
                   (task_id, task_type, status, input_json, error, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    task.task_id,
                    task.task_type,
                    task.status,
                    json.dumps(task.input_data),
                    "",
                    now,
                    now,
                ),
            )
            conn.commit()
        except Exception:
            logger.warning(
                "Failed to persist A2A task %s — task will not survive restart",
                task.task_id,
            )

    def _persist_result(self, task_id: str, result: A2AResult) -> None:
        """Persist task result to SQLite (best-effort, non-blocking)."""
        if not getattr(self, "_db_available", False):
            return
        try:
            import json

            from vetinari.database import get_connection

            now = datetime.now(timezone.utc).isoformat()
            conn = get_connection()
            conn.execute(
                """UPDATE a2a_tasks
                   SET status = ?, output_json = ?, error = ?, updated_at = ?
                   WHERE task_id = ?""",
                (
                    result.status,
                    json.dumps(result.output_data),
                    result.error,
                    now,
                    task_id,
                ),
            )
            conn.commit()
        except Exception:
            logger.warning(
                "Failed to persist A2A result for task %s — result lost on restart",
                task_id,
            )

    def recover_pending_tasks(self) -> list[A2ATask]:
        """Recover tasks in PENDING, RUNNING, or ACKNOWLEDGED state from the database.

        Called at startup to resume interrupted work.  ACKNOWLEDGED tasks are
        included because they were accepted but never executed (orchestrator was
        unavailable) — they deserve a retry on restart.

        Returns:
            List of A2ATask objects that need to be re-executed.
        """
        if not getattr(self, "_db_available", False):
            return []
        try:
            import json

            from vetinari.database import get_connection

            conn = get_connection()
            rows = conn.execute(
                "SELECT task_id, task_type, status, input_json FROM a2a_tasks WHERE status IN (?, ?, ?)",
                (STATUS_PENDING, STATUS_RUNNING, STATUS_ACKNOWLEDGED),
            ).fetchall()
            tasks = [
                A2ATask(
                    task_id=row[0],
                    task_type=row[1],
                    status=row[2],
                    input_data=json.loads(row[3]) if row[3] else {},
                )
                for row in rows
            ]
            if tasks:
                logger.info("Recovered %d pending A2A tasks from database", len(tasks))
            return tasks
        except Exception:
            logger.warning("A2A task recovery failed — no tasks will be resumed")
            return []

    # ── Public API ────────────────────────────────────────────────────────────

    def execute(self, task: A2ATask) -> A2AResult:
        """Execute an A2A task by routing it to the appropriate pipeline agent.

        Args:
            task: The incoming :class:`A2ATask` to execute.

        Returns:
            :class:`A2AResult` with ``status="completed"`` on success or
            ``status="failed"`` if the task type is unknown or execution
            raises an exception.
        """
        logger.info("Executing A2A task id=%s type=%s", task.task_id, task.task_type)
        task.status = STATUS_RUNNING
        self._persist_task(task)

        route = self._route_to_agent(task.task_type)
        if route is None:
            logger.warning("No route found for A2A task type '%s'", task.task_type)
            task.status = STATUS_FAILED
            result = A2AResult(
                task_id=task.task_id,
                status=STATUS_FAILED,
                error=f"Unknown task type: '{task.task_type}'. Supported types: {sorted(self._routing_table.keys())}",
            )
            self._persist_result(task.task_id, result)
            return result

        agent_type, mode = route
        logger.debug(
            "Task id=%s routed to agent=%s mode=%s",
            task.task_id,
            agent_type.value,
            mode,
        )

        try:
            output = self._dispatch(agent_type, mode, task)
            if output.get("_is_acknowledgement_only"):
                # Orchestrator unavailable — task accepted but not yet executed.
                # Callers must not treat this as successful execution.
                task.status = STATUS_ACKNOWLEDGED
                result = A2AResult(
                    task_id=task.task_id,
                    status=STATUS_ACKNOWLEDGED,
                    output_data=output,
                )
            else:
                task.status = STATUS_COMPLETED
                result = A2AResult(
                    task_id=task.task_id,
                    status=STATUS_COMPLETED,
                    output_data=output,
                )
            self._persist_result(task.task_id, result)
            return result
        except Exception as exc:
            logger.exception("A2A task id=%s failed during dispatch: %s", task.task_id, exc)
            task.status = STATUS_FAILED
            result = A2AResult(
                task_id=task.task_id,
                status=STATUS_FAILED,
                error=str(exc),
            )
            self._persist_result(task.task_id, result)
            return result

    def _route_to_agent(self, task_type: str) -> _RouteEntry | None:
        """Look up the (AgentType, mode) pair for a given task type string.

        Args:
            task_type: The A2A task type string to look up.

        Returns:
            A ``(AgentType, mode)`` tuple if the task type is recognised,
            or ``None`` if it is not in the routing table.
        """
        return self._routing_table.get(task_type)

    def _build_routing_table(self) -> _RoutingTable:
        """Construct the full A2A task-type → agent/mode routing table.

        Returns:
            Mapping from task type string to ``(AgentType, mode)`` tuples.
        """
        table: _RoutingTable = {
            # ── Foreman tasks ────────────────────────────────────────────
            "plan": (AgentType.FOREMAN, "plan"),
            "clarify": (AgentType.FOREMAN, "clarify"),
            "consolidate": (AgentType.FOREMAN, "consolidate"),
            "summarise": (AgentType.FOREMAN, "summarise"),
            "summarize": (AgentType.FOREMAN, "summarise"),  # US spelling alias
            "prune": (AgentType.FOREMAN, "prune"),
            "extract": (AgentType.FOREMAN, "extract"),
            # ── Worker — research group ──────────────────────────────────
            "research": (AgentType.WORKER, "code_discovery"),
            "code_discovery": (AgentType.WORKER, "code_discovery"),
            "domain_research": (AgentType.WORKER, "domain_research"),
            "api_lookup": (AgentType.WORKER, "api_lookup"),
            "lateral_thinking": (AgentType.WORKER, "lateral_thinking"),
            "ui_design": (AgentType.WORKER, "ui_design"),
            "database": (AgentType.WORKER, "database"),
            "devops": (AgentType.WORKER, "devops"),
            "git_workflow": (AgentType.WORKER, "git_workflow"),
            # ── Worker — architecture group ──────────────────────────────
            "architecture": (AgentType.WORKER, "architecture"),
            "risk_assessment": (AgentType.WORKER, "risk_assessment"),
            "ontological_analysis": (AgentType.WORKER, "ontological_analysis"),
            "contrarian_review": (AgentType.WORKER, "contrarian_review"),
            "suggest": (AgentType.WORKER, "suggest"),
            # ── Worker — build group ─────────────────────────────────────
            "build": (AgentType.WORKER, "build"),
            "implement": (AgentType.WORKER, "build"),  # common synonym
            "image_generation": (AgentType.WORKER, "image_generation"),
            # ── Worker — operations group ────────────────────────────────
            "documentation": (AgentType.WORKER, "documentation"),
            "creative_writing": (AgentType.WORKER, "creative_writing"),
            "cost_analysis": (AgentType.WORKER, "cost_analysis"),
            "experiment": (AgentType.WORKER, "experiment"),
            "error_recovery": (AgentType.WORKER, "error_recovery"),
            "synthesis": (AgentType.WORKER, "synthesis"),
            "improvement": (AgentType.WORKER, "improvement"),
            "monitor": (AgentType.WORKER, "monitor"),
            "devops_ops": (AgentType.WORKER, "devops_ops"),
            # ── Inspector tasks ──────────────────────────────────────────
            "review": (AgentType.INSPECTOR, "code_review"),
            "code_review": (AgentType.INSPECTOR, "code_review"),
            "security_audit": (AgentType.INSPECTOR, "security_audit"),
            "test_generation": (AgentType.INSPECTOR, "test_generation"),
            "simplification": (AgentType.INSPECTOR, "simplification"),
        }
        logger.debug("A2A routing table built with %d entries", len(table))
        return table

    def _dispatch(self, agent_type: AgentType, mode: str, task: A2ATask) -> dict[str, Any]:
        """Dispatch a task to the internal Vetinari pipeline.

        Attempts to invoke the ``TwoLayerOrchestrator`` for real execution.
        When the orchestrator is ``None`` (standalone/test mode), returns a
        structured acknowledgement dict flagged with
        ``"_is_acknowledgement_only": True`` so that :meth:`execute` can set
        the correct ``STATUS_ACKNOWLEDGED`` status rather than incorrectly
        marking the task completed.

        Exceptions from the orchestrator are NOT caught here — they propagate
        to :meth:`execute` which handles them and sets ``STATUS_FAILED``.

        Args:
            agent_type: Which pipeline agent should handle the task.
            mode: The specific mode within that agent.
            task: The original :class:`A2ATask` being executed.

        Returns:
            Output data dict to be embedded in the :class:`A2AResult`.
            When ``"_is_acknowledgement_only"`` is ``True`` the task was
            accepted but not executed (degraded mode).

        Raises:
            Exception: Any exception raised by the orchestrator propagates
                to the caller so it can be recorded as ``STATUS_FAILED``.
        """
        logger.info(
            "Dispatching task id=%s to agent=%s mode=%s",
            task.task_id,
            agent_type.value,
            mode,
        )
        orch = get_two_layer_orchestrator()
        if orch is not None:
            task_description = str(task.input_data.get("description", task.task_type))
            agent_task = AgentTask(
                task_id=task.task_id,
                description=task_description,
                prompt=task.input_data.get("goal", task_description),
                agent_type=agent_type,
                context=task.input_data,
            )
            result = orch.execute_task(agent_task)
            return {
                "agent": agent_type.value,
                "mode": mode,
                "task_id": task.task_id,
                "output": result.output if hasattr(result, "output") else str(result),
                "success": result.success if hasattr(result, "success") else True,
            }

        # Orchestrator unavailable (standalone/test mode) — return a structured
        # acknowledgement.  The "_is_acknowledgement_only" flag tells execute()
        # to use STATUS_ACKNOWLEDGED instead of STATUS_COMPLETED so callers
        # are never misled into thinking real execution occurred.
        logger.info(
            "Orchestrator unavailable for task id=%s — returning acknowledgement only (degraded mode)",
            task.task_id,
        )
        return {
            "agent": agent_type.value,
            "mode": mode,
            "task_id": task.task_id,
            "status": STATUS_ACKNOWLEDGED,
            "_is_acknowledgement_only": True,
            "input_summary": {k: str(v)[:100] for k, v in task.input_data.items()},
        }

    @property
    def supported_task_types(self) -> list[str]:
        """Return the sorted list of A2A task type strings this executor handles.

        Returns:
            Sorted list of recognised task type strings.
        """
        return sorted(self._routing_table.keys())
