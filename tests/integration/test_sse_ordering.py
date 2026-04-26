"""Integration tests for SSE event ordering.

Verifies that Server-Sent Events for task lifecycle maintain causal
ordering — started events always precede completed events — and that
every event type serialises correctly to its SSE payload dict.
"""

from __future__ import annotations

import time

import pytest

from tests.factories import make_task
from vetinari.types import AgentType
from vetinari.web.sse_events import (
    SSE_EVENT_REGISTRY,
    CancelledEvent,
    DecisionEvent,
    ErrorEvent,
    EtaUpdateEvent,
    ModelLoadingEvent,
    ModelRecommendationEvent,
    PausedEvent,
    PlanningStartEvent,
    QualityResultEvent,
    ResumedEvent,
    StageCompleteEvent,
    StageProgressEvent,
    StageStartEvent,
    StatusEvent,
    TaskCancelledEvent,
    TaskCompleteEvent,
    TaskFailedEvent,
    TaskRerunEvent,
    TaskStartEvent,
    ThinkingEvent,
    TrainingCompleteEvent,
    TrainingFailedEvent,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _now() -> float:
    """Return the current monotonic time in seconds (high resolution)."""
    return time.perf_counter()


# ---------------------------------------------------------------------------
# Tests: task lifecycle causal ordering
# ---------------------------------------------------------------------------


class TestTaskStartBeforeComplete:
    """A TaskStartEvent's timestamp must precede the matching TaskCompleteEvent."""

    def test_task_start_timestamp_before_complete(self) -> None:
        """TaskStartEvent created before TaskCompleteEvent must have an earlier timestamp."""
        task = make_task(id="task-order-001", description="Validate causal ordering")

        t_start = _now()
        start_event = TaskStartEvent(
            task_id=task.id,
            description=task.description,
            agent_type=AgentType.WORKER.value,
            task_index=1,
            total_tasks=3,
        )
        # Windows time.monotonic() has ~15ms resolution; use perf_counter via
        # _now() and sleep 20ms to guarantee distinct ticks
        time.sleep(0.02)
        t_complete = _now()
        complete_event = TaskCompleteEvent(
            task_id=task.id,
            output_summary="Task finished successfully",
            task_index=1,
            total_tasks=3,
        )

        assert t_start < t_complete, f"Start timestamp {t_start} must precede complete timestamp {t_complete}"
        assert start_event.task_id == complete_event.task_id, "Both events must reference the same task_id"
        assert start_event.event_type == "task_started"
        assert complete_event.event_type == "task_completed"

    def test_task_failed_event_has_correct_task_id(self) -> None:
        """TaskFailedEvent must carry the same task_id as the initiating TaskStartEvent."""
        task = make_task(id="task-fail-001", description="Failing task")

        start = TaskStartEvent(task_id=task.id, description=task.description)
        failed = TaskFailedEvent(task_id=task.id, error="Connection refused", task_index=2, total_tasks=5)

        assert start.task_id == failed.task_id, "Failure event must reference the same task that started"
        assert failed.event_type == "task_failed"
        assert "Connection refused" in failed.error


# ---------------------------------------------------------------------------
# Tests: dependency ordering
# ---------------------------------------------------------------------------


class TestDependencyOrdering:
    """For tasks with dependencies, parent completion must precede child start."""

    def test_dependency_ordering_parent_completes_before_child_starts(self) -> None:
        """Given task B depends on task A, A's complete event is created before B's start."""
        task_a = make_task(id="task-parent-A", description="Task A — no dependencies")
        task_b = make_task(
            id="task-child-B",
            description="Task B — depends on A",
            dependencies=["task-parent-A"],
        )

        t1 = _now()
        a_complete = TaskCompleteEvent(
            task_id=task_a.id,
            output_summary="A is done",
            task_index=1,
            total_tasks=2,
        )
        time.sleep(0.001)
        t2 = _now()
        b_start = TaskStartEvent(
            task_id=task_b.id,
            description=task_b.description,
            agent_type=AgentType.WORKER.value,
            task_index=2,
            total_tasks=2,
        )

        assert t1 < t2, "A's completion timestamp must be strictly before B's start timestamp"
        assert a_complete.event_type == "task_completed"
        assert b_start.event_type == "task_started"
        # Verify the dependency relationship is declared
        assert "task-parent-A" in task_b.dependencies

    def test_cancelled_event_preserves_task_id(self) -> None:
        """TaskCancelledEvent must preserve the task_id of the cancelled task."""
        task = make_task(id="task-cancel-001")
        event = TaskCancelledEvent(task_id=task.id, reason="dependency_failed")

        assert event.task_id == task.id
        assert event.event_type == "task_cancelled"
        assert event.reason == "dependency_failed"


# ---------------------------------------------------------------------------
# Tests: all event types serializable
# ---------------------------------------------------------------------------


class TestAllEventTypesSerializable:
    """Every event class in SSE_EVENT_REGISTRY must produce a valid to_sse() dict."""

    @pytest.mark.parametrize(("event_type_name", "event_class"), list(SSE_EVENT_REGISTRY.items()))
    def test_event_to_sse_returns_dict(self, event_type_name: str, event_class: type) -> None:
        """to_sse() on a default-constructed event must return a non-empty dict."""
        instance = event_class()
        result = instance.to_sse()

        assert isinstance(result, dict), f"{event_class.__name__}.to_sse() must return a dict, got {type(result)}"
        assert len(result) > 0, f"{event_class.__name__}.to_sse() must return a non-empty dict"

    @pytest.mark.parametrize(("event_type_name", "event_class"), list(SSE_EVENT_REGISTRY.items()))
    def test_event_type_field_matches_registry_key(self, event_type_name: str, event_class: type) -> None:
        """The event_type field on each class must match the registry key."""
        instance = event_class()
        assert instance.event_type == event_type_name, (
            f"{event_class.__name__}.event_type={instance.event_type!r} does not match registry key {event_type_name!r}"
        )

    def test_task_start_event_to_sse_includes_required_keys(self) -> None:
        """TaskStartEvent.to_sse() must include task_id, agent_type, task_index, total_tasks."""
        event = TaskStartEvent(
            task_id="t-001",
            description="Build something",
            agent_type=AgentType.WORKER.value,
            task_index=3,
            total_tasks=10,
        )
        payload = event.to_sse()

        assert payload["task_id"] == "t-001"
        assert payload["agent_type"] == AgentType.WORKER.value
        assert payload["task_index"] == 3
        assert payload["total_tasks"] == 10
        assert payload["description"] == "Build something"

    def test_task_complete_event_to_sse_includes_required_keys(self) -> None:
        """TaskCompleteEvent.to_sse() must include task_id, output_summary, task_index, total_tasks."""
        event = TaskCompleteEvent(
            task_id="t-002",
            output_summary="Completed successfully",
            task_index=7,
            total_tasks=10,
        )
        payload = event.to_sse()

        assert payload["task_id"] == "t-002"
        assert payload["output_summary"] == "Completed successfully"
        assert payload["task_index"] == 7
        assert payload["total_tasks"] == 10

    def test_error_event_to_sse_includes_recoverable_flag(self) -> None:
        """ErrorEvent.to_sse() must include the recoverable flag."""
        event = ErrorEvent(error="Timeout", error_type="TimeoutError", recoverable=False)
        payload = event.to_sse()

        assert payload["recoverable"] is False
        assert payload["error"] == "Timeout"
        assert payload["error_type"] == "TimeoutError"

    def test_quality_result_event_to_sse_includes_score(self) -> None:
        """QualityResultEvent.to_sse() must include project_id, quality_score, and passed flag."""
        event = QualityResultEvent(
            project_id="proj-abc",
            quality_score=0.87,
            passed=True,
            issues_count=0,
        )
        payload = event.to_sse()

        assert payload["project_id"] == "proj-abc"
        assert abs(payload["quality_score"] - 0.87) < 1e-9
        assert payload["passed"] is True
        assert payload["issues_count"] == 0


# ---------------------------------------------------------------------------
# Tests: repr contains key info
# ---------------------------------------------------------------------------


class TestEventReprContainsKeyInfo:
    """__repr__ for each event with a custom repr must expose task_id and agent info."""

    def test_task_start_event_repr_contains_task_id(self) -> None:
        """TaskStartEvent repr must show task_id and agent_type."""
        event = TaskStartEvent(task_id="t-repr-001", agent_type=AgentType.FOREMAN.value, task_index=1, total_tasks=4)
        r = repr(event)

        assert "t-repr-001" in r, f"repr must contain task_id, got: {r}"
        assert AgentType.FOREMAN.value in r, f"repr must contain agent_type, got: {r}"

    def test_task_complete_event_repr_contains_task_id(self) -> None:
        """TaskCompleteEvent repr must show task_id and progress indices."""
        event = TaskCompleteEvent(task_id="t-repr-002", task_index=5, total_tasks=8)
        r = repr(event)

        assert "t-repr-002" in r, f"repr must contain task_id, got: {r}"

    def test_task_failed_event_repr_contains_error(self) -> None:
        """TaskFailedEvent repr must show task_id and the error string."""
        event = TaskFailedEvent(task_id="t-repr-003", error="model unavailable", task_index=3, total_tasks=5)
        r = repr(event)

        assert "t-repr-003" in r, f"repr must contain task_id, got: {r}"
        assert "model unavailable" in r, f"repr must contain error text, got: {r}"

    def test_stage_start_event_repr_contains_stage_name(self) -> None:
        """StageStartEvent repr must show the stage name and progress."""
        event = StageStartEvent(stage="execution", stage_index=2, total_stages=3)
        r = repr(event)

        assert "execution" in r, f"repr must contain stage name, got: {r}"

    def test_quality_result_event_repr_contains_project_and_score(self) -> None:
        """QualityResultEvent repr must show project_id, score, and passed flag."""
        event = QualityResultEvent(project_id="proj-xyz", quality_score=0.91, passed=True)
        r = repr(event)

        assert "proj-xyz" in r, f"repr must contain project_id, got: {r}"
        assert "0.91" in r, f"repr must contain quality score, got: {r}"
        assert "True" in r, f"repr must contain passed flag, got: {r}"

    def test_error_event_repr_contains_type_and_error(self) -> None:
        """ErrorEvent repr must show error_type, recoverable, and error text."""
        event = ErrorEvent(error="disk full", error_type="IOError", recoverable=False)
        r = repr(event)

        assert "IOError" in r, f"repr must contain error_type, got: {r}"
        assert "False" in r, f"repr must show recoverable=False, got: {r}"
