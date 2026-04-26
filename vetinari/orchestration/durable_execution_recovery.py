"""Checkpoint and recovery helpers for DurableExecutionEngine.

Extracted from ``durable_execution.py`` to keep that file under 550 lines.

All functions accept ``engine: DurableExecutionEngine`` as their first argument
and operate on its instance state (``_db``, ``_active_executions``,
``_execution_lock``, ``_task_handlers``).

Pipeline role: Plan → DurableExecution → **Recovery/Checkpoint** → Verify.

Pause/question functions (``save_paused_questions``, ``answer_paused_questions``,
``get_paused_questions``) provide **metadata-only storage** for the pause-for-
clarification pattern.  No engine code path currently consumes answered
questions to resume execution.  Callers that need a fully resumable pause must
implement their own consumer, or treat these functions as audit-only.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vetinari.orchestration.durable_execution import DurableExecutionEngine

from vetinari.orchestration.execution_graph import ExecutionGraph, ExecutionTaskNode
from vetinari.types import PlanStatus, StatusEnum

# Re-export for use in durable_execution.py stubs
__all__ = [
    "answer_paused_questions",
    "cleanup_completed",
    "emit_event",
    "get_execution_status",
    "get_paused_questions",
    "handle_layer_failure",
    "list_checkpoints",
    "load_checkpoint",
    "record_learning",
    "recover_execution",
    "recover_incomplete_executions",
    "save_checkpoint",
    "save_paused_questions",
]

logger = logging.getLogger(__name__)


def record_learning(task: ExecutionTaskNode, task_id: str, output: Any) -> None:
    """Record task outcome for the learning pipeline (non-fatal side effect).

    Scores the output, records the outcome in the feedback loop, and
    updates Thompson Sampling arms. Silently no-ops on any exception so
    that learning failures never interrupt execution.

    Args:
        task: The completed task node.
        task_id: String task identifier for logging.
        output: Raw handler output for quality scoring.
    """
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

        if q_score.overall_score < 0.5:
            logger.warning(
                "[DurableExec] Low quality score %.2f for task %s (model=%s, type=%s) — review output quality",
                q_score.overall_score,
                task_id,
                model_id,
                task_type_str,
            )
    except Exception as _learn_err:
        logger.warning(
            "Learning hook failed for task %s — execution result unaffected: %s",
            task_id,
            _learn_err,
        )


def emit_event(
    engine: DurableExecutionEngine,
    event_type: str,
    task_id: str,
    data: dict[str, Any],
    execution_id: str = "",
) -> None:
    """Emit an execution event and persist it to SQLite.

    Stores events in both the in-memory deque (fast access) and the
    ``execution_events`` table (crash recovery and audit trail).

    Args:
        engine: The DurableExecutionEngine instance owning the database.
        event_type: Type of event (e.g. ``task_started``).
        task_id: The task this event relates to.
        data: Additional event payload.
        execution_id: Optional execution ID for foreign key linkage.
    """
    import json
    import uuid

    from vetinari.orchestration.durable_db import ExecutionEventRecord

    event = ExecutionEventRecord(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        task_id=task_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        data=data,
    )
    engine._event_history.append(event)
    logger.debug("Event: %s - %s", event_type, task_id)

    # Delegate persistence to CheckpointStore so event storage goes through
    # the named facade rather than raw SQL here.  Falls back to the direct
    # _db path if the store is not yet initialised (e.g. during engine startup).
    _store = getattr(engine, "_checkpoint_store", None)
    if _store is not None:
        _store.save_event(event, execution_id)
    else:
        try:
            engine._db.execute(
                """INSERT OR IGNORE INTO execution_events
                   (event_id, execution_id, event_type, task_id, timestamp, data_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    event.event_id,
                    execution_id,
                    event_type,
                    task_id,
                    event.timestamp,
                    json.dumps(data),
                ),
            )
        except Exception:
            logger.warning(
                "Failed to persist event %s to SQLite — execution continues",
                event.event_id,
                exc_info=True,
            )


def handle_layer_failure(
    engine: DurableExecutionEngine,
    graph: ExecutionGraph,
    failed_tasks: list[ExecutionTaskNode],
) -> None:
    """Cancel tasks that depend (transitively) on any failed task.

    Args:
        engine: The DurableExecutionEngine instance used to emit cancellation events.
        graph: The execution graph to update in-place.
        failed_tasks: Tasks that failed in the current layer.
    """
    cancelled_ids: set[str] = {t.id for t in failed_tasks}

    changed = True
    while changed:
        changed = False
        for node in graph.nodes.values():
            if node.status in (StatusEnum.COMPLETED, StatusEnum.FAILED, StatusEnum.CANCELLED):
                continue
            if any(dep in cancelled_ids for dep in node.depends_on) and node.id not in cancelled_ids:
                node.status = StatusEnum.CANCELLED
                cancelled_ids.add(node.id)
                emit_event(
                    engine,
                    "task_cancelled",
                    node.id,
                    {
                        "reason": "dependency_failed",
                        "failed_dependencies": [dep for dep in node.depends_on if dep in cancelled_ids],
                    },
                )
                changed = True


def save_checkpoint(engine: DurableExecutionEngine, plan_id: str, graph: ExecutionGraph) -> None:
    """Save a checkpoint of the execution state to SQLite (atomic, crash-safe).

    When the graph has reached a terminal state (COMPLETED or FAILED), also
    writes ``completed_at`` and ``terminal_status`` so retention/cleanup queries
    can find and act on finished executions.

    Args:
        engine: The DurableExecutionEngine instance owning the database.
        plan_id: The plan identifier.
        graph: The execution graph to persist.
    """
    now = datetime.now(timezone.utc).isoformat()
    graph_dict = graph.to_dict()
    completed = [t.id for t in graph.get_completed_tasks()]
    running = [t.id for t in graph.nodes.values() if t.status == StatusEnum.RUNNING]

    # Populate finished_at / terminal_status only when the plan has truly ended.
    # While still executing these stay NULL so cleanup queries skip in-progress rows.
    _is_terminal = graph.status in (PlanStatus.COMPLETED, PlanStatus.FAILED)
    _completed_at_val = now if _is_terminal else None
    _terminal_status_val = graph.status.value if _is_terminal else None

    task_rows = [
        (
            node.id,
            plan_id,
            getattr(node, "task_type", ""),
            "",
            node.status.value,
            json.dumps(node.input_data) if hasattr(node, "input_data") and node.input_data else None,
            json.dumps(node.output_data) if hasattr(node, "output_data") and node.output_data else None,
            None,
            getattr(node, "started_at", None),
            getattr(node, "completed_at", None),
            getattr(node, "retry_count", 0),
        )
        for node in graph.nodes.values()
    ]

    # Route all writes through _checkpoint_store (single connection) to prevent
    # "database is locked" errors when engine._db and _checkpoint_store._db both
    # try to write the same SQLite file concurrently (ADR-0073).
    _store = getattr(engine, "_checkpoint_store", None)
    if _store is not None:
        _store.save_checkpoint(
            plan_id,
            graph_dict,
            graph.status.value,
            task_rows,
            now,
            completed_at=_completed_at_val,
            terminal_status=_terminal_status_val,
        )
    else:
        # Fallback for engines without _checkpoint_store (should not occur in production).
        engine._db.execute(
            """INSERT INTO execution_state
                   (execution_id, goal, pipeline_state, task_dag_json, created_at, updated_at,
                    completed_at, terminal_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(execution_id) DO UPDATE SET
                   pipeline_state = excluded.pipeline_state,
                   task_dag_json = excluded.task_dag_json,
                   updated_at = excluded.updated_at,
                   completed_at = COALESCE(execution_state.completed_at, excluded.completed_at),
                   terminal_status = COALESCE(execution_state.terminal_status, excluded.terminal_status)""",
            (
                plan_id,
                graph_dict.get("goal", ""),
                graph.status.value,
                json.dumps(graph_dict),
                graph_dict.get("created_at", now),
                now,
                _completed_at_val,
                _terminal_status_val,
            ),
        )
        if task_rows:
            engine._db.executemany(
                """INSERT INTO task_checkpoints
                   (task_id, execution_id, agent_type, mode, status, input_json, output_json,
                    manifest_hash, started_at, completed_at, retry_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(task_id) DO UPDATE SET
                       status = excluded.status,
                       output_json = excluded.output_json,
                       completed_at = excluded.completed_at,
                       retry_count = excluded.retry_count""",
                task_rows,
            )

    logger.debug(
        "Checkpoint saved: plan=%s, completed=%d, running=%d",
        plan_id,
        len(completed),
        len(running),
    )


def load_checkpoint(engine: DurableExecutionEngine, plan_id: str) -> ExecutionGraph | None:
    """Load a checkpoint from SQLite to resume execution.

    Args:
        engine: The DurableExecutionEngine instance owning the database.
        plan_id: The plan identifier to resume.

    Returns:
        The restored ExecutionGraph, or None if no checkpoint exists.
    """
    _store = getattr(engine, "_checkpoint_store", None)
    if _store is not None:
        raw_json = _store.load_checkpoint_graph_json(plan_id)
    else:
        rows = engine._db.execute(
            "SELECT task_dag_json FROM execution_state WHERE execution_id = ?",
            (plan_id,),
        )
        raw_json = rows[0][0] if rows and rows[0][0] else None

    if not raw_json:
        logger.warning("No checkpoint found for plan: %s", plan_id)
        return None

    try:
        graph_data = json.loads(raw_json)
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
            graph.nodes[node_id] = ExecutionTaskNode.from_dict(node_data)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.warning(
            "Checkpoint for plan %s is corrupt or has an invalid schema — returning None"
            " (cannot recover this checkpoint; a fresh run is required). Detail: %s",
            plan_id,
            exc,
        )
        return None

    with engine._execution_lock:
        engine._active_executions[plan_id] = graph

    logger.info("Loaded checkpoint from SQLite for plan: %s", plan_id)
    return graph


def save_paused_questions(
    engine: DurableExecutionEngine,
    execution_id: str,
    questions: list[str],
    task_id: str | None = None,
) -> str:
    """Persist questions that require user answers, recording the pause event.

    **Metadata-only storage.** No engine code path currently consumes answered
    questions to resume execution. Callers that need resumable pause must
    implement their own consumer or treat this as audit-only.

    Args:
        engine: The DurableExecutionEngine instance owning the database.
        execution_id: The execution being paused.
        questions: List of questions for the user.
        task_id: Optional task that triggered the pause.

    Returns:
        The question_id for later answer retrieval via ``answer_paused_questions()``.
    """
    question_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    engine._db.execute(
        """INSERT OR IGNORE INTO execution_state
           (execution_id, goal, pipeline_state, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?)""",
        (execution_id, "", "paused", now, now),
    )
    engine._db.execute(
        """INSERT INTO paused_questions
           (question_id, execution_id, task_id, questions_json, asked_at)
           VALUES (?, ?, ?, ?, ?)""",
        (question_id, execution_id, task_id, json.dumps(questions), now),
    )
    logger.info("Pipeline paused: execution=%s, %d questions", execution_id, len(questions))
    return question_id


def answer_paused_questions(engine: DurableExecutionEngine, question_id: str, answers: list[str]) -> None:
    """Store answers for a set of paused questions.

    **Metadata-only storage.** No engine code path currently consumes answered
    questions to resume execution. Callers that need resumable pause must
    implement their own consumer or treat this as audit-only.

    Args:
        engine: The DurableExecutionEngine instance owning the database.
        question_id: The question set to answer.
        answers: User-provided answers in the same order as the questions.
    """
    now = datetime.now(timezone.utc).isoformat()
    engine._db.execute(
        "UPDATE paused_questions SET answers_json = ?, answered_at = ? WHERE question_id = ?",
        (json.dumps(answers), now, question_id),
    )
    logger.info("Questions answered: %s", question_id)


def get_paused_questions(engine: DurableExecutionEngine, execution_id: str) -> list[dict[str, Any]]:
    """Return all paused questions for an execution, with any stored answers.

    **Metadata-only storage.** No engine code path currently consumes answered
    questions to resume execution. Callers that need resumable pause must
    implement their own consumer or treat this as audit-only.

    Args:
        engine: The DurableExecutionEngine instance owning the database.
        execution_id: The execution to query.

    Returns:
        List of question dicts with id, questions, answers, and timestamps.
    """
    rows = engine._db.execute(
        "SELECT question_id, task_id, questions_json, answers_json, asked_at, answered_at "
        "FROM paused_questions WHERE execution_id = ?",
        (execution_id,),
    )
    return [
        {
            "question_id": r[0],
            "task_id": r[1],
            "questions": json.loads(r[2]),
            "answers": json.loads(r[3]) if r[3] else None,
            "asked_at": r[4],
            "answered_at": r[5],
        }
        for r in rows
    ]


def recover_execution(engine: DurableExecutionEngine, plan_id: str) -> dict[str, Any]:
    """Recover and continue an execution from its last checkpoint.

    Resets retryable failed tasks (retry_count < max_retries) to PENDING
    and re-executes the plan from the recovered state.

    Args:
        engine: The DurableExecutionEngine instance owning execution state.
        plan_id: The plan identifier to recover.

    Returns:
        Execution result dict identical to what ``execute_plan`` returns,
        or ``{"status": "error", "plan_id": plan_id, "message": "..."}`` if
        no checkpoint exists or the checkpoint is corrupt.
    """
    graph = load_checkpoint(engine, plan_id)

    if not graph:
        return {"status": "error", "plan_id": plan_id, "message": "No checkpoint found"}

    for node in graph.nodes.values():
        if node.status == StatusEnum.RUNNING:
            # RUNNING at recovery time means the process crashed mid-task.
            # Reset to PENDING so the task is retried rather than left stuck.
            node.status = StatusEnum.PENDING
            node.error = "Reset from RUNNING state during recovery — process likely crashed"
        elif node.status == StatusEnum.FAILED and node.retry_count < node.max_retries:
            node.status = StatusEnum.PENDING
            node.error = ""

    incomplete = [
        n for n in graph.nodes.values() if n.status in (StatusEnum.PENDING, StatusEnum.BLOCKED, StatusEnum.FAILED)
    ]
    logger.info("Recovering %s incomplete tasks for plan: %s", len(incomplete), plan_id)

    return engine.execute_plan(graph)


def get_execution_status(engine: DurableExecutionEngine, plan_id: str) -> dict[str, Any] | None:
    """Get the current status of an active or checkpointed execution.

    Args:
        engine: The DurableExecutionEngine instance owning execution state.
        plan_id: The plan identifier to query.

    Returns:
        Dict with plan_id, status, total_tasks, completed count, failed
        count, blocked count, and progress ratio (completed/total).
        Returns None if no active execution or checkpoint is found.
    """
    with engine._execution_lock:
        graph = engine._active_executions.get(plan_id)

    if not graph:
        graph = load_checkpoint(engine, plan_id)

    if not graph:
        return None

    return {
        "plan_id": plan_id,
        "status": graph.status.value,
        "total_tasks": len(graph.nodes),
        StatusEnum.COMPLETED.value: len(graph.get_completed_tasks()),
        StatusEnum.FAILED.value: len(graph.get_failed_tasks()),
        StatusEnum.BLOCKED.value: len(graph.get_blocked_tasks()),
        "progress": (len(graph.get_completed_tasks()) / len(graph.nodes) if graph.nodes else 0),
    }


def list_checkpoints(engine: DurableExecutionEngine) -> list[str]:
    """List all plan IDs with persisted checkpoints.

    Args:
        engine: The DurableExecutionEngine instance owning the database.

    Returns:
        Sorted list of execution IDs from the execution_state table.
    """
    _store = getattr(engine, "_checkpoint_store", None)
    if _store is not None:
        return _store.list_checkpoint_ids()
    rows = engine._db.execute("SELECT execution_id FROM execution_state")
    return sorted({r[0] for r in rows})


def recover_incomplete_executions(
    engine: DurableExecutionEngine,
    task_handler: Callable | None = None,
) -> list[dict[str, Any]]:
    """Find and resume all incomplete executions from persisted checkpoints.

    Queries SQLite for executions whose ``pipeline_state`` is neither
    ``completed`` nor ``failed``, loads each checkpoint, resets retryable
    failed tasks, and re-executes them. Called at startup to ensure
    crash-interrupted work is resumed automatically.

    Args:
        engine: The DurableExecutionEngine instance owning execution state.
        task_handler: Optional default handler. If not provided, tasks
            rely on previously registered handlers.

    Returns:
        List of per-execution result dicts. Empty list when nothing
        needs recovery.
    """
    _store = getattr(engine, "_checkpoint_store", None)
    if _store is not None:
        plan_ids = _store.find_incomplete_ids(StatusEnum.COMPLETED.value, StatusEnum.FAILED.value)
    else:
        rows = engine._db.execute(
            "SELECT execution_id FROM execution_state WHERE pipeline_state NOT IN (?, ?)",
            (StatusEnum.COMPLETED.value, StatusEnum.FAILED.value),
        )
        plan_ids = [r[0] for r in rows] if rows else []

    if not plan_ids:
        logger.info("No incomplete executions found — nothing to recover")
        return []
    logger.info("Found %d incomplete execution(s) to recover: %s", len(plan_ids), plan_ids)

    results: list[dict[str, Any]] = []
    for plan_id in plan_ids:
        try:
            if task_handler:
                engine._task_handlers.setdefault("default", task_handler)

            # Heartbeat staleness is in-process only; across restart the
            # in-memory heartbeat dict is always empty, so is_task_stuck()
            # would always return False and the check would be a no-op.
            # RUNNING-task reset is handled inside recover_execution() which
            # resets every persisted RUNNING node to PENDING unconditionally.

            result = recover_execution(engine, plan_id)
            results.append(result)
            logger.info(
                "Recovered execution %s: completed=%s, failed=%s",
                plan_id,
                result.get(StatusEnum.COMPLETED.value, 0),
                result.get(StatusEnum.FAILED.value, 0),
            )
        except Exception as exc:
            logger.error(
                "Failed to recover execution %s — skipping, other executions will still be attempted: %s",
                plan_id,
                exc,
            )
            results.append({
                "plan_id": plan_id,
                "status": "error",
                "message": f"Recovery failed: {exc}",
            })
    return results


def cleanup_completed(engine: DurableExecutionEngine, max_age_days: int = 30) -> int:
    """Delete completed executions older than max_age_days from SQLite.

    Args:
        engine: The DurableExecutionEngine instance owning the database.
        max_age_days: Remove executions completed more than this many days ago.

    Returns:
        Number of executions deleted.
    """
    from datetime import timedelta

    cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()

    _store = getattr(engine, "_checkpoint_store", None)
    if _store is not None:
        ids = _store.find_completed_before(cutoff)
        for exec_id in ids:
            _store.delete_execution(exec_id)
    else:
        rows = engine._db.execute(
            "SELECT execution_id FROM execution_state WHERE completed_at IS NOT NULL AND completed_at <= ?",
            (cutoff,),
        )
        ids = [r[0] for r in rows] if rows else []
        for exec_id in ids:
            engine._db.execute_in_transaction([
                ("DELETE FROM execution_events WHERE execution_id = ?", (exec_id,)),
                ("DELETE FROM task_checkpoints WHERE execution_id = ?", (exec_id,)),
                ("DELETE FROM paused_questions WHERE execution_id = ?", (exec_id,)),
                ("DELETE FROM execution_state WHERE execution_id = ?", (exec_id,)),
            ])

    if not ids:
        return 0
    logger.info("Cleaned up %d completed executions older than %d days", len(ids), max_age_days)
    return len(ids)
