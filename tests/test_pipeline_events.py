"""Tests for vetinari.orchestration.pipeline_events module."""

from __future__ import annotations

import time

import pytest

from vetinari.orchestration.pipeline_events import (
    CompositeEventHandler,
    EventSeverity,
    LoggingEventHandler,
    NullEventHandler,
    PipelineEvent,
    PipelineEventHandler,
    PipelineStage,
)


class TestPipelineStage:
    """Tests for the PipelineStage enum."""

    def test_all_nine_stages_exist(self) -> None:
        """Verify all 9 pipeline stages are defined."""
        expected = {
            "intake",
            "prevention",
            "plan_gen",
            "model_assign",
            "execution",
            "refinement",
            "review",
            "verification",
            "assembly",
        }
        actual = {s.value for s in PipelineStage}
        assert actual == expected

    def test_stage_count(self) -> None:
        """Pipeline has exactly 9 stages."""
        assert len(PipelineStage) == 9


class TestPipelineEvent:
    """Tests for the PipelineEvent frozen dataclass."""

    def test_creation_with_defaults(self) -> None:
        """Event can be created with just stage and event_type."""
        event = PipelineEvent(
            stage=PipelineStage.INTAKE,
            event_type="stage_start",
        )
        assert event.stage == PipelineStage.INTAKE
        assert event.event_type == "stage_start"
        assert event.data == {}
        assert event.severity == EventSeverity.INFO
        assert event.timestamp > 0

    def test_creation_with_all_fields(self) -> None:
        """Event accepts all fields explicitly."""
        ts = time.time()
        event = PipelineEvent(
            stage=PipelineStage.EXECUTION,
            event_type="task_complete",
            data={"task_id": "t1", "success": True},
            timestamp=ts,
            severity=EventSeverity.WARNING,
        )
        assert event.stage == PipelineStage.EXECUTION
        assert event.data["task_id"] == "t1"
        assert event.timestamp == ts
        assert event.severity == EventSeverity.WARNING

    def test_frozen(self) -> None:
        """PipelineEvent is immutable."""
        event = PipelineEvent(stage=PipelineStage.REVIEW, event_type="start")
        with pytest.raises(AttributeError):
            event.stage = PipelineStage.ASSEMBLY  # type: ignore[misc]

    def test_repr(self) -> None:
        """Repr includes stage, event_type, and severity."""
        event = PipelineEvent(stage=PipelineStage.PLAN_GEN, event_type="start")
        r = repr(event)
        assert "plan_gen" in r
        assert "start" in r
        assert "info" in r


class TestNullEventHandler:
    """Tests for the NullEventHandler no-op handler."""

    def test_satisfies_protocol(self) -> None:
        """NullEventHandler implements PipelineEventHandler protocol."""
        handler = NullEventHandler()
        assert isinstance(handler, PipelineEventHandler)

    def test_on_event_does_not_raise(self) -> None:
        """Calling on_event does nothing and raises no error."""
        handler = NullEventHandler()
        event = PipelineEvent(stage=PipelineStage.INTAKE, event_type="test")
        raised = False
        try:
            handler.on_event(event)
        except Exception:
            raised = True
        assert not raised, "NullEventHandler.on_event must not raise"


class TestLoggingEventHandler:
    """Tests for the LoggingEventHandler."""

    def test_satisfies_protocol(self) -> None:
        """LoggingEventHandler implements PipelineEventHandler protocol."""
        handler = LoggingEventHandler()
        assert isinstance(handler, PipelineEventHandler)

    def test_on_event_logs_without_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """Event is logged without raising."""
        import logging

        handler = LoggingEventHandler()
        event = PipelineEvent(
            stage=PipelineStage.EXECUTION,
            event_type="task_start",
            data={"task_id": "t1"},
            severity=EventSeverity.DEBUG,
        )
        with caplog.at_level(logging.DEBUG, logger="vetinari.orchestration.pipeline_events"):
            handler.on_event(event)
        assert any("task_start" in record.message for record in caplog.records), (
            "LoggingEventHandler.on_event must emit a log record containing the event_type"
        )


class TestCompositeEventHandler:
    """Tests for the CompositeEventHandler fan-out handler."""

    def test_satisfies_protocol(self) -> None:
        """CompositeEventHandler implements PipelineEventHandler protocol."""
        handler = CompositeEventHandler()
        assert isinstance(handler, PipelineEventHandler)

    def test_fan_out_to_multiple_handlers(self) -> None:
        """Events are forwarded to all child handlers."""
        received: list[PipelineEvent] = []

        class _Collector:
            def on_event(self, event: PipelineEvent) -> None:
                received.append(event)

        composite = CompositeEventHandler(_Collector(), _Collector())
        event = PipelineEvent(stage=PipelineStage.REVIEW, event_type="start")
        composite.on_event(event)
        assert len(received) == 2
        assert all(e is event for e in received)

    def test_add_handler(self) -> None:
        """Handlers added after construction also receive events."""
        received: list[PipelineEvent] = []

        class _Collector:
            def on_event(self, event: PipelineEvent) -> None:
                received.append(event)

        composite = CompositeEventHandler()
        composite.add_handler(_Collector())
        event = PipelineEvent(stage=PipelineStage.ASSEMBLY, event_type="done")
        composite.on_event(event)
        assert len(received) == 1

    def test_child_exception_does_not_break_others(self) -> None:
        """One failing child does not prevent others from receiving."""
        received: list[PipelineEvent] = []

        class _Failing:
            def on_event(self, event: PipelineEvent) -> None:
                msg = "boom"
                raise RuntimeError(msg)

        class _Collector:
            def on_event(self, event: PipelineEvent) -> None:
                received.append(event)

        composite = CompositeEventHandler(_Failing(), _Collector())
        event = PipelineEvent(stage=PipelineStage.INTAKE, event_type="test")
        composite.on_event(event)  # Should not raise
        assert len(received) == 1
