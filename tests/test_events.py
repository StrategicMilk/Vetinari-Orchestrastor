"""Tests for the vetinari.events module (US-033)."""

from __future__ import annotations

import threading
import time

import pytest

from vetinari.events import (
    Event,
    EventBus,
    HumanApprovalNeeded,
    QualityGateResult,
    ResourceRequest,
    TaskCompleted,
    TaskStarted,
    get_event_bus,
    reset_event_bus,
)
from vetinari.types import AgentType


@pytest.fixture(autouse=True)
def _clean_event_bus() -> None:
    """Reset the singleton event bus before and after each test."""
    reset_event_bus()
    yield  # type: ignore[misc]
    reset_event_bus()


# ---------------------------------------------------------------------------
# Event dataclass tests
# ---------------------------------------------------------------------------


class TestEventDataclasses:
    """Verify that all event dataclasses set their discriminator correctly."""

    def test_task_started_event_type(self) -> None:
        """Test that TaskStarted sets event_type to 'TaskStarted'."""
        event = TaskStarted(
            event_type="",
            timestamp=time.time(),
            task_id="t1",
            agent_type=AgentType.WORKER.value,
        )
        assert event.event_type == "TaskStarted"
        assert event.task_id == "t1"
        assert event.agent_type == AgentType.WORKER.value

    def test_task_completed_event_type(self) -> None:
        """Test that TaskCompleted sets event_type and carries result fields."""
        event = TaskCompleted(
            event_type="",
            timestamp=time.time(),
            task_id="t2",
            agent_type=AgentType.INSPECTOR.value,
            success=True,
            duration_ms=123.4,
        )
        assert event.event_type == "TaskCompleted"
        assert event.success is True
        assert event.duration_ms == 123.4

    def test_quality_gate_result_event_type(self) -> None:
        """Test that QualityGateResult carries score and issues."""
        event = QualityGateResult(
            event_type="",
            timestamp=time.time(),
            task_id="t3",
            passed=False,
            score=0.42,
            issues=["missing docstring"],
        )
        assert event.event_type == "QualityGateResult"
        assert event.passed is False
        assert event.score == 0.42
        assert event.issues == ["missing docstring"]

    def test_resource_request_event_type(self) -> None:
        """Test that ResourceRequest carries agent and resource metadata."""
        event = ResourceRequest(
            event_type="",
            timestamp=time.time(),
            agent_type=AgentType.FOREMAN.value,
            resource_type="model",
            details={"model": "gpt-4"},
        )
        assert event.event_type == "ResourceRequest"
        assert event.details == {"model": "gpt-4"}

    def test_human_approval_needed_event_type(self) -> None:
        """Test that HumanApprovalNeeded carries reason and context."""
        event = HumanApprovalNeeded(
            event_type="",
            timestamp=time.time(),
            task_id="t5",
            reason="Destructive operation",
            context={"action": "delete_db"},
        )
        assert event.event_type == "HumanApprovalNeeded"
        assert event.reason == "Destructive operation"

    def test_all_events_inherit_from_event(self) -> None:
        """Test that every concrete event is an instance of Event."""
        ts = time.time()
        events = [
            TaskStarted(event_type="", timestamp=ts, task_id="x", agent_type="A"),
            TaskCompleted(event_type="", timestamp=ts, task_id="x", agent_type="A", success=True, duration_ms=0.0),
            QualityGateResult(event_type="", timestamp=ts, task_id="x", passed=True, score=1.0),
            ResourceRequest(event_type="", timestamp=ts, agent_type="A", resource_type="tool"),
            HumanApprovalNeeded(event_type="", timestamp=ts, task_id="x", reason="r"),
        ]
        for e in events:
            assert isinstance(e, Event)


# ---------------------------------------------------------------------------
# EventBus core behaviour
# ---------------------------------------------------------------------------


class TestEventBusSubscribePublish:
    """Verify synchronous publish/subscribe mechanics."""

    def test_subscribe_and_publish(self) -> None:
        """Test that a subscriber receives the published event."""
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(TaskStarted, received.append)
        event = TaskStarted(event_type="", timestamp=time.time(), task_id="a", agent_type=AgentType.WORKER.value)
        bus.publish(event)
        assert len(received) == 1
        assert received[0] is event

    def test_subscriber_only_receives_matching_type(self) -> None:
        """Test that a TaskStarted subscriber does not receive TaskCompleted events."""
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(TaskStarted, received.append)
        bus.publish(
            TaskCompleted(
                event_type="", timestamp=time.time(), task_id="b", agent_type="Q", success=True, duration_ms=0
            )
        )
        assert len(received) == 0

    def test_subscribe_to_base_event_receives_all(self) -> None:
        """Test that subscribing to Event receives all event subtypes."""
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(Event, received.append)
        bus.publish(TaskStarted(event_type="", timestamp=time.time(), task_id="c", agent_type="A"))
        bus.publish(
            TaskCompleted(
                event_type="", timestamp=time.time(), task_id="d", agent_type="B", success=False, duration_ms=1.0
            )
        )
        assert len(received) == 2

    def test_multiple_subscribers(self) -> None:
        """Test that multiple subscribers all receive the same event."""
        bus = EventBus()
        r1: list[Event] = []
        r2: list[Event] = []
        bus.subscribe(TaskStarted, r1.append)
        bus.subscribe(TaskStarted, r2.append)
        bus.publish(TaskStarted(event_type="", timestamp=time.time(), task_id="e", agent_type="X"))
        assert len(r1) == 1
        assert len(r2) == 1


class TestEventBusUnsubscribe:
    """Verify subscription removal."""

    def test_unsubscribe_stops_delivery(self) -> None:
        """Test that unsubscribing prevents future event delivery."""
        bus = EventBus()
        received: list[Event] = []
        sub_id = bus.subscribe(TaskStarted, received.append)
        bus.unsubscribe(sub_id)
        bus.publish(TaskStarted(event_type="", timestamp=time.time(), task_id="f", agent_type="Y"))
        assert len(received) == 0

    def test_unsubscribe_unknown_id_raises(self) -> None:
        """Test that unsubscribing with an unknown ID raises KeyError."""
        bus = EventBus()
        with pytest.raises(KeyError, match="Subscription not found"):
            bus.unsubscribe("nonexistent-id")


class TestEventBusFaultTolerance:
    """Verify that bad subscribers do not crash the publisher."""

    def test_bad_subscriber_does_not_crash_publish(self) -> None:
        """Test that an exception in one subscriber does not prevent others."""
        bus = EventBus()
        received: list[Event] = []

        def bad_callback(_event: Event) -> None:
            raise RuntimeError("subscriber exploded")

        bus.subscribe(TaskStarted, bad_callback)
        bus.subscribe(TaskStarted, received.append)

        event = TaskStarted(event_type="", timestamp=time.time(), task_id="g", agent_type="Z")
        bus.publish(event)
        assert len(received) == 1


class TestEventBusHistory:
    """Verify event history queries."""

    def test_history_records_events(self) -> None:
        """Test that published events appear in history."""
        bus = EventBus()
        bus.publish(TaskStarted(event_type="", timestamp=time.time(), task_id="h1", agent_type="A"))
        bus.publish(
            TaskCompleted(
                event_type="", timestamp=time.time(), task_id="h2", agent_type="B", success=True, duration_ms=0
            )
        )
        history = bus.get_history()
        assert len(history) == 2

    def test_history_filter_by_type(self) -> None:
        """Test that history can be filtered to a specific event type."""
        bus = EventBus()
        bus.publish(TaskStarted(event_type="", timestamp=time.time(), task_id="i1", agent_type="A"))
        bus.publish(
            TaskCompleted(
                event_type="", timestamp=time.time(), task_id="i2", agent_type="B", success=True, duration_ms=0
            )
        )
        history = bus.get_history(event_type=TaskStarted)
        assert len(history) == 1
        assert isinstance(history[0], TaskStarted)

    def test_history_respects_limit(self) -> None:
        """Test that history respects the limit parameter."""
        bus = EventBus()
        for n in range(10):
            bus.publish(TaskStarted(event_type="", timestamp=time.time(), task_id=f"j{n}", agent_type="A"))
        history = bus.get_history(limit=3)
        assert len(history) == 3
        # Should be the last 3 (newest)
        assert history[-1].task_id == "j9"  # type: ignore[attr-defined]

    def test_history_bounded_by_max_length(self) -> None:
        """Test that history does not grow beyond _HISTORY_MAX_LENGTH."""
        from vetinari.events import _HISTORY_MAX_LENGTH

        bus = EventBus()
        for n in range(_HISTORY_MAX_LENGTH + 50):
            bus.publish(Event(event_type="test", timestamp=float(n)))
        assert len(bus.get_history(limit=_HISTORY_MAX_LENGTH + 100)) == _HISTORY_MAX_LENGTH


class TestEventBusClear:
    """Verify the clear method for test isolation."""

    def test_clear_removes_subscriptions_and_history(self) -> None:
        """Test that clear empties both subscriptions and history."""
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(TaskStarted, received.append)
        bus.publish(TaskStarted(event_type="", timestamp=time.time(), task_id="k", agent_type="A"))
        assert len(received) == 1
        assert len(bus.get_history()) == 1

        bus.clear()
        bus.publish(TaskStarted(event_type="", timestamp=time.time(), task_id="l", agent_type="B"))
        # Subscription was cleared, so received should not grow
        assert len(received) == 1
        # History was cleared and only the new event should be there
        assert len(bus.get_history()) == 1


class TestEventBusAsync:
    """Verify asynchronous publishing."""

    def test_publish_async_delivers_events(self) -> None:
        """Test that publish_async eventually delivers events in a background thread."""
        bus = EventBus()
        received: list[Event] = []
        barrier = threading.Event()

        def callback(event: Event) -> None:
            received.append(event)
            barrier.set()

        bus.subscribe(TaskStarted, callback)
        event = TaskStarted(event_type="", timestamp=time.time(), task_id="m", agent_type="A")
        bus.publish_async(event)

        barrier.wait(timeout=5.0)
        assert len(received) == 1
        assert received[0] is event


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------


class TestSingleton:
    """Verify singleton access and reset."""

    def test_get_event_bus_returns_same_instance(self) -> None:
        """Test that get_event_bus always returns the same instance."""
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2

    def test_reset_event_bus_creates_new_instance(self) -> None:
        """Test that reset_event_bus causes a new instance on next call."""
        bus1 = get_event_bus()
        reset_event_bus()
        bus2 = get_event_bus()
        assert bus1 is not bus2
