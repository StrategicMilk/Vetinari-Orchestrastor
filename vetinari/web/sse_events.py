"""Typed SSE Event Dataclasses and dual-delivery contract.

Replaces 30+ ad-hoc dict constructions with frozen dataclasses
for each SSE event type emitted by the Vetinari web layer.

Each event has a ``to_sse()`` method that returns the dict expected
by ``_push_sse_event(project_id, event_type, data)``.

Usage::

    from vetinari.web.sse_events import TaskStartEvent
    event = TaskStartEvent(task_id="t1", description="Build auth")
    _push_sse_event(project_id, event.event_type, event.to_sse())

Dual SSE Delivery Contract
--------------------------
Every SSE event published via ``_push_sse_event`` travels two paths:

**Live queue** (ephemeral):
    An in-memory ``queue.Queue`` in ``vetinari.web.shared`` delivers events
    in real-time to connected SSE clients.  The queue is **ephemeral** — its
    contents are lost on process restart or client disconnect.  It exists
    solely to minimise latency for live subscribers.

**sse_event_log** (durable):
    Simultaneously, every event is written to the ``sse_event_log`` SQLite
    table (columns: ``id``, ``project_id``, ``event_type``, ``payload_json``,
    ``sequence_num``, ``emitted_at``).  This store **survives process
    restarts** and is the source of truth for replay.

**Replay endpoint**:
    Reconnecting clients that missed events while disconnected call
    ``GET /api/v1/projects/{project_id}/events/replay?after_sequence=N``
    to fetch all persisted events with ``sequence_num > N``.  The
    ``N`` value comes from the SSE ``id:`` field sent with every live event
    (browsers expose it as ``EventSource.lastEventId``).

**Log-type events** (ThinkingEvent, DecisionEvent, ErrorEvent):
    These are persisted to ``sse_event_log`` via the same
    ``_push_sse_event`` path and are therefore fully replayable, even
    though they are not tied to task lifecycle milestones.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from itertools import count
from typing import Any

logger = logging.getLogger(__name__)
_SSE_EVENT_SEQUENCE = count(1)


def _next_sse_sequence() -> int:
    """Return the next monotonically increasing SSE event sequence number."""
    return next(_SSE_EVENT_SEQUENCE)


# -- Lifecycle events -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StatusEvent:
    """Pipeline status update (running, idle, etc.)."""

    event_type: str = "status"
    status: str = ""
    total_tasks: int = 0

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {"status": self.status, "total_tasks": self.total_tasks}


@dataclass(frozen=True, slots=True)
class PlanningStartEvent:
    """Plan generation has begun."""

    event_type: str = "planning_started"
    goal: str = ""
    plan_id: str = ""

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {"goal": self.goal, "plan_id": self.plan_id}


@dataclass(frozen=True, slots=True)
class PausedEvent:
    """Pipeline execution paused."""

    event_type: str = "paused"
    project_id: str = ""

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {"project_id": self.project_id, "status": "paused"}


@dataclass(frozen=True, slots=True)
class ResumedEvent:
    """Pipeline execution resumed."""

    event_type: str = "resumed"
    project_id: str = ""

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {"project_id": self.project_id, "status": "resumed"}


@dataclass(frozen=True, slots=True)
class CancelledEvent:
    """Pipeline execution cancelled."""

    event_type: str = "cancelled"
    project_id: str = ""
    reason: str = ""

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {"project_id": self.project_id, "reason": self.reason}


# -- Task events ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TaskStartEvent:
    """A task has started execution."""

    event_type: str = "task_started"
    sequence: int = field(default_factory=_next_sse_sequence)
    task_id: str = ""
    description: str = ""
    agent_type: str = ""
    task_index: int = 0
    total_tasks: int = 0

    def __repr__(self) -> str:
        return (
            f"TaskStartEvent(task_id={self.task_id!r}, agent={self.agent_type!r}, {self.task_index}/{self.total_tasks})"
        )

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {
            "sequence": self.sequence,
            "task_id": self.task_id,
            "description": self.description,
            "agent_type": self.agent_type,
            "task_index": self.task_index,
            "total_tasks": self.total_tasks,
        }


@dataclass(frozen=True, slots=True)
class TaskCompleteEvent:
    """A task has completed successfully."""

    event_type: str = "task_completed"
    sequence: int = field(default_factory=_next_sse_sequence)
    task_id: str = ""
    output_summary: str = ""
    task_index: int = 0
    total_tasks: int = 0

    def __repr__(self) -> str:
        return f"TaskCompleteEvent(task_id={self.task_id!r}, {self.task_index}/{self.total_tasks})"

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {
            "sequence": self.sequence,
            "task_id": self.task_id,
            "output_summary": self.output_summary,
            "task_index": self.task_index,
            "total_tasks": self.total_tasks,
        }


@dataclass(frozen=True, slots=True)
class TaskFailedEvent:
    """A task has failed."""

    event_type: str = "task_failed"
    sequence: int = field(default_factory=_next_sse_sequence)
    task_id: str = ""
    error: str = ""
    task_index: int = 0
    total_tasks: int = 0

    def __repr__(self) -> str:
        return f"TaskFailedEvent(task_id={self.task_id!r}, error={self.error!r})"

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {
            "sequence": self.sequence,
            "task_id": self.task_id,
            "error": self.error,
            "task_index": self.task_index,
            "total_tasks": self.total_tasks,
        }


@dataclass(frozen=True, slots=True)
class TaskCancelledEvent:
    """A task has been cancelled."""

    event_type: str = "task_cancelled"
    sequence: int = field(default_factory=_next_sse_sequence)
    task_id: str = ""
    reason: str = ""

    def __repr__(self) -> str:
        return f"TaskCancelledEvent(task_id={self.task_id!r}, reason={self.reason!r})"

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {"sequence": self.sequence, "task_id": self.task_id, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class TaskRerunEvent:
    """A task is being re-run (retry)."""

    event_type: str = "task_rerun"
    sequence: int = field(default_factory=_next_sse_sequence)
    task_id: str = ""
    attempt: int = 1

    def __repr__(self) -> str:
        return f"TaskRerunEvent(task_id={self.task_id!r}, attempt={self.attempt})"

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {"sequence": self.sequence, "task_id": self.task_id, "attempt": self.attempt}


# -- Stage events -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StageStartEvent:
    """A pipeline stage has started."""

    event_type: str = "stage_started"
    sequence: int = field(default_factory=_next_sse_sequence)
    stage: str = ""
    stage_index: int = 0
    total_stages: int = 0

    def __repr__(self) -> str:
        return f"StageStartEvent(stage={self.stage!r}, {self.stage_index}/{self.total_stages})"

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {
            "sequence": self.sequence,
            "stage": self.stage,
            "stage_index": self.stage_index,
            "total_stages": self.total_stages,
        }


@dataclass(frozen=True, slots=True)
class StageProgressEvent:
    """Progress update within a pipeline stage."""

    event_type: str = "stage_progress"
    sequence: int = field(default_factory=_next_sse_sequence)
    stage: str = ""
    progress: float = 0.0
    message: str = ""

    def __repr__(self) -> str:
        return f"StageProgressEvent(stage={self.stage!r}, progress={self.progress:.0%})"

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {
            "sequence": self.sequence,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class StageCompleteEvent:
    """A pipeline stage has completed."""

    event_type: str = "stage_completed"
    sequence: int = field(default_factory=_next_sse_sequence)
    stage: str = ""
    output_summary: str = ""

    def __repr__(self) -> str:
        return f"StageCompleteEvent(stage={self.stage!r}, summary={self.output_summary!r})"

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {"sequence": self.sequence, "stage": self.stage, "output_summary": self.output_summary}


@dataclass(frozen=True, slots=True)
class PipelineStageEvent:
    """Pipeline stage status snapshot for dashboard visualization."""

    event_type: str = "pipeline_stage"
    stage: str = ""
    status: str = "idle"  # idle | active | complete | failed
    entry_count: int = 0
    exit_count: int = 0

    def __repr__(self) -> str:
        return f"PipelineStageEvent(stage={self.stage!r}, status={self.status!r}, entry={self.entry_count}, exit={self.exit_count})"

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {
            "stage": self.stage,
            "status": self.status,
            "entry_count": self.entry_count,
            "exit_count": self.exit_count,
        }


# -- Agent/model events -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThinkingEvent:
    """Agent thinking/reasoning status."""

    event_type: str = "thinking"
    agent_type: str = ""
    message: str = ""

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {"agent_type": self.agent_type, "message": self.message}


@dataclass(frozen=True, slots=True)
class DecisionEvent:
    """Agent decision notification."""

    event_type: str = "decision"
    decision_type: str = ""
    summary: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"DecisionEvent(type={self.decision_type!r}, summary={self.summary!r})"

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {
            "decision_type": self.decision_type,
            "summary": self.summary,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class ModelLoadingEvent:
    """A model is being loaded for inference."""

    event_type: str = "model_loading"
    model_id: str = ""
    status: str = "loading"

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {"model_id": self.model_id, "status": self.status}


@dataclass(frozen=True, slots=True)
class ModelRecommendationEvent:
    """Model selection recommendation."""

    event_type: str = "model_recommendation"
    recommended_model: str = ""
    reason: str = ""

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {"recommended_model": self.recommended_model, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class EtaUpdateEvent:
    """Estimated time of arrival update."""

    event_type: str = "eta_update"
    remaining_seconds: float = 0.0
    message: str = ""

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {"remaining_seconds": self.remaining_seconds, "message": self.message}


# -- Error events -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    """General error notification."""

    event_type: str = "error"
    error: str = ""
    error_type: str = ""
    recoverable: bool = True

    def __repr__(self) -> str:
        return f"ErrorEvent(type={self.error_type!r}, recoverable={self.recoverable}, error={self.error!r})"

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {
            "error": self.error,
            "error_type": self.error_type,
            "recoverable": self.recoverable,
        }


# -- Quality events ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QualityResultEvent:
    """Quality scoring result from Inspector review."""

    event_type: str = "quality_result"
    project_id: str = ""
    quality_score: float = 0.0
    passed: bool = False
    issues_count: int = 0
    confidence: float = 0.0

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return (
            f"QualityResultEvent(project_id={self.project_id!r}, score={self.quality_score!r}, passed={self.passed!r})"
        )

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {
            "project_id": self.project_id,
            "quality_score": self.quality_score,
            "passed": self.passed,
            "issues_count": self.issues_count,
            "confidence": self.confidence,
        }


# -- Training events --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainingCompleteEvent:
    """Training run completed successfully."""

    event_type: str = "training_completed"
    run_id: str = ""
    summary: str = ""

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {"run_id": self.run_id, "summary": self.summary}


@dataclass(frozen=True, slots=True)
class TrainingFailedEvent:
    """Training run failed."""

    event_type: str = "training_failed"
    error: str = ""

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {"error": self.error}


# -- Notification events ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class NotificationEvent:
    """A notification dispatched via the notification manager."""

    event_type: str = "notification"
    notification_id: str = ""
    title: str = ""
    body: str = ""
    priority: str = ""
    action_type: str = ""

    def __repr__(self) -> str:
        return "NotificationEvent(...)"

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {
            "notification_id": self.notification_id,
            "title": self.title,
            "body": self.body,
            "priority": self.priority,
            "action_type": self.action_type,
        }


@dataclass(frozen=True, slots=True)
class ApprovalRequestEvent:
    """An action has been queued for human approval."""

    event_type: str = "approval_requested"
    action_id: str = ""
    action_type: str = ""
    confidence: float = 0.0

    def __repr__(self) -> str:
        return "ApprovalRequestEvent(...)"

    def to_sse(self) -> dict[str, Any]:
        """Serialize to SSE data payload."""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "confidence": self.confidence,
        }


# -- Registry ---------------------------------------------------------------

# All event classes keyed by their event_type for programmatic lookup
SSE_EVENT_REGISTRY: dict[str, type] = {
    "status": StatusEvent,
    "planning_started": PlanningStartEvent,
    "paused": PausedEvent,
    "resumed": ResumedEvent,
    "cancelled": CancelledEvent,
    "task_started": TaskStartEvent,
    "task_completed": TaskCompleteEvent,
    "task_failed": TaskFailedEvent,
    "task_cancelled": TaskCancelledEvent,
    "task_rerun": TaskRerunEvent,
    "stage_started": StageStartEvent,
    "stage_progress": StageProgressEvent,
    "stage_completed": StageCompleteEvent,
    "pipeline_stage": PipelineStageEvent,
    "thinking": ThinkingEvent,
    "decision": DecisionEvent,
    "model_loading": ModelLoadingEvent,
    "model_recommendation": ModelRecommendationEvent,
    "eta_update": EtaUpdateEvent,
    "error": ErrorEvent,
    "quality_result": QualityResultEvent,
    "training_completed": TrainingCompleteEvent,
    "training_failed": TrainingFailedEvent,
    "notification": NotificationEvent,
    "approval_requested": ApprovalRequestEvent,
}


# -- Persistence helpers --------------------------------------------------------


def _persist_sse_event(project_id: str, event_type: str, payload: dict[str, Any]) -> None:
    """Write an SSE event to the ``sse_event_log`` table (best-effort).

    Called immediately after an event is delivered live so the event is also
    available for replay.  Failures are logged at WARNING and swallowed so live
    delivery is never interrupted.

    Args:
        project_id: The project the event belongs to.
        event_type: The SSE event type string (e.g. ``"task_started"``).
        payload: The event data dict to store as JSON.
    """
    conn = None
    try:
        from vetinari.database import get_connection

        conn = get_connection()
        conn.execute(
            "INSERT INTO sse_event_log (project_id, event_type, payload_json) VALUES (?, ?, ?)",
            (project_id, event_type, json.dumps(payload, ensure_ascii=False)),
        )
        conn.commit()
    except Exception:
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                logger.warning(
                    "Rollback also failed during SSE event persistence — database may be in inconsistent state"
                )
        logger.warning(
            "Could not persist SSE event %s for project %s — event delivered but not stored",
            event_type,
            project_id,
        )


def get_recent_sse_events(
    project_id: str,
    limit: int = 100,
    since: str | None = None,
) -> list[dict[str, Any]]:
    """Return persisted SSE events for *project_id* in ascending order.

    Args:
        project_id: Project to query.
        limit: Maximum number of rows to return.
        since: ISO-format timestamp string; only events emitted after this
            timestamp are returned.  Pass None to return all events.

    Returns:
        List of event dicts with keys ``id``, ``project_id``, ``event_type``,
        ``payload`` (parsed JSON dict), and ``emitted_at``.  Events with
        unparseable JSON return ``{"_raw": <original string>}`` as payload.
    """
    from vetinari.database import get_connection

    conn = get_connection()
    if since is not None:
        rows = conn.execute(
            "SELECT id, project_id, event_type, payload_json, emitted_at"
            " FROM sse_event_log"
            " WHERE project_id = ? AND emitted_at > ?"
            " ORDER BY id ASC LIMIT ?",
            (project_id, since, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, project_id, event_type, payload_json, emitted_at"
            " FROM sse_event_log"
            " WHERE project_id = ?"
            " ORDER BY id ASC LIMIT ?",
            (project_id, limit),
        ).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(row["payload_json"])
        except (json.JSONDecodeError, TypeError):
            payload = {"_raw": row["payload_json"]}
        results.append({
            "id": row["id"],
            "project_id": row["project_id"],
            "event_type": row["event_type"],
            "payload": payload,
            "emitted_at": row["emitted_at"],
        })
    return results


def cleanup_old_sse_events(hours: int = 168) -> int:
    """Delete SSE event log entries older than *hours* hours.

    Args:
        hours: Retention window in hours.  Events emitted more than this many
            hours ago are deleted.  Defaults to 168 (7 days).  Must be >= 1.

    Returns:
        Number of rows deleted.

    Raises:
        ValueError: If hours is less than 1.
    """
    if hours < 1:
        raise ValueError("hours must be >= 1")

    from vetinari.database import get_connection

    conn = get_connection()
    cursor = conn.execute(
        "DELETE FROM sse_event_log WHERE emitted_at < datetime('now', ? || ' hours')",
        (f"-{hours}",),
    )
    conn.commit()
    deleted = cursor.rowcount
    logger.info("cleanup_old_sse_events: deleted %d rows older than %d hours", deleted, hours)
    return deleted


# Alias used by cli_startup.py scheduler callback
cleanup_stale_sse_events = cleanup_old_sse_events
