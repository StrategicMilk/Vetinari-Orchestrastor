"""Tests for vetinari.workflow.wiring — WIP + Andon dispatch gate."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from vetinari.types import AgentType
from vetinari.workflow.andon import reset_andon_system
from vetinari.workflow.wip import WIPConfig, WIPTracker

# -- Helpers ----------------------------------------------------------------


def _reset_wiring_singleton() -> None:
    """Reset the module-level WIPTracker singleton between tests."""
    import vetinari.workflow.wiring as wiring_mod

    wiring_mod._wip_tracker = None


@pytest.fixture(autouse=True)
def isolated_wiring():
    """Ensure each test starts with a fresh WIPTracker and Andon state."""
    _reset_wiring_singleton()
    reset_andon_system()
    yield
    _reset_wiring_singleton()
    reset_andon_system()


# -- _get_wip_tracker -------------------------------------------------------


class TestGetWipTracker:
    def test_returns_wip_tracker_instance(self):
        from vetinari.workflow.wiring import _get_wip_tracker

        tracker = _get_wip_tracker()
        assert isinstance(tracker, WIPTracker)

    def test_singleton_same_object_on_repeated_calls(self):
        from vetinari.workflow.wiring import _get_wip_tracker

        first = _get_wip_tracker()
        second = _get_wip_tracker()
        assert first is second

    def test_new_instance_after_singleton_reset(self):
        from vetinari.workflow.wiring import _get_wip_tracker

        first = _get_wip_tracker()
        _reset_wiring_singleton()
        second = _get_wip_tracker()
        assert first is not second


# -- dispatch_or_queue ------------------------------------------------------


class TestDispatchOrQueue:
    def test_starts_task_when_capacity_available(self):
        from vetinari.workflow.wiring import dispatch_or_queue

        result = dispatch_or_queue(AgentType.WORKER.value, "task-1")
        assert result is True

    def test_queues_task_when_wip_limit_reached(self):
        from vetinari.workflow.wiring import _get_wip_tracker, dispatch_or_queue

        agent = AgentType.FOREMAN.value  # limit = 1
        # Fill the single slot
        assert dispatch_or_queue(agent, "task-A") is True
        # Next task must be queued
        assert dispatch_or_queue(agent, "task-B") is False
        # Confirm it landed in the queue
        assert _get_wip_tracker().get_queue_depth() == 1

    def test_multiple_tasks_within_limit_all_start(self):
        from vetinari.workflow.wiring import _get_wip_tracker, dispatch_or_queue

        agent = AgentType.WORKER.value  # limit = 3
        for i in range(3):
            assert dispatch_or_queue(agent, f"task-{i}") is True
        assert _get_wip_tracker().get_active_count(agent) == 3

    def test_fourth_worker_task_is_queued(self):
        from vetinari.workflow.wiring import _get_wip_tracker, dispatch_or_queue

        agent = AgentType.WORKER.value  # limit = 3
        for i in range(3):
            dispatch_or_queue(agent, f"task-{i}")
        assert dispatch_or_queue(agent, "task-overflow") is False
        assert _get_wip_tracker().get_queue_depth() == 1


# -- complete_and_pull ------------------------------------------------------


class TestCompleteAndPull:
    def test_returns_none_when_queue_is_empty(self):
        from vetinari.workflow.wiring import complete_and_pull, dispatch_or_queue

        agent = AgentType.WORKER.value
        dispatch_or_queue(agent, "task-solo")
        result = complete_and_pull(agent, "task-solo")
        assert result is None

    def test_returns_queued_task_after_completion(self):
        from vetinari.workflow.wiring import complete_and_pull, dispatch_or_queue

        agent = AgentType.FOREMAN.value  # limit = 1
        dispatch_or_queue(agent, "task-first")
        dispatch_or_queue(agent, "task-second")  # queued

        pulled = complete_and_pull(agent, "task-first")
        assert pulled is not None
        assert pulled["task_id"] == "task-second"
        assert pulled["agent_type"] == agent

    def test_pulled_task_is_auto_started(self):
        from vetinari.workflow.wiring import _get_wip_tracker, complete_and_pull, dispatch_or_queue

        agent = AgentType.FOREMAN.value
        dispatch_or_queue(agent, "task-A")
        dispatch_or_queue(agent, "task-B")

        complete_and_pull(agent, "task-A")
        # task-B should now be active, not queued
        assert _get_wip_tracker().get_active_count(agent) == 1
        assert _get_wip_tracker().get_queue_depth() == 0

    def test_unknown_completion_does_not_pull_queued_task(self):
        """Completing a non-active task must not consume capacity or start queued work."""
        tracker = WIPTracker(WIPConfig(limits={AgentType.WORKER.value: 1, "default": 1}))
        agent = AgentType.WORKER.value
        assert tracker.start_task(agent, "active") is True
        tracker.enqueue(agent, "queued")

        pulled = tracker.complete_task(agent, "missing")

        assert pulled is None
        assert tracker.get_active_count(agent) == 1
        assert tracker.get_queue_depth() == 1

    def test_release_task_does_not_pull_queued_task(self):
        """Deferring an active task must release capacity without starting queued work."""
        tracker = WIPTracker(WIPConfig(limits={AgentType.WORKER.value: 1, "default": 1}))
        agent = AgentType.WORKER.value
        assert tracker.start_task(agent, "active") is True
        tracker.enqueue(agent, "queued")

        assert tracker.release_task(agent, "active") is True

        assert tracker.get_active_count(agent) == 0
        assert tracker.get_queue_depth() == 1

    def test_concurrent_start_does_not_exceed_capacity(self):
        """WIP admission is atomic under concurrent callers."""
        tracker = WIPTracker(WIPConfig(limits={AgentType.WORKER.value: 1, "default": 1}))
        agent = AgentType.WORKER.value
        barrier = threading.Barrier(12)
        results: list[bool] = []
        results_lock = threading.Lock()

        def _claim(index: int) -> None:
            barrier.wait()
            result = tracker.start_task(agent, f"task-{index}")
            with results_lock:
                results.append(result)

        threads = [threading.Thread(target=_claim, args=(i,)) for i in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)

        assert sum(1 for result in results if result) == 1
        assert tracker.get_active_count(agent) == 1


class TestAgentGraphWipAdmission:
    def test_execute_task_node_uses_atomic_start_task(self):
        """AgentGraph admission must queue when the atomic start claim fails."""
        from vetinari.orchestration.graph_task_runner import GraphTaskRunnerMixin

        class _Tracker:
            def __init__(self) -> None:
                self.start_calls: list[tuple[str, str]] = []
                self.enqueued: list[tuple[str, str]] = []

            def can_start(self, _agent_type: str) -> bool:
                raise AssertionError("AgentGraph must not use split can_start/start_task admission")

            def start_task(self, agent_type: str, task_id: str) -> bool:
                self.start_calls.append((agent_type, task_id))
                return False

            def enqueue(self, agent_type: str, task_id: str) -> None:
                self.enqueued.append((agent_type, task_id))

        tracker = _Tracker()
        runner = GraphTaskRunnerMixin()
        runner._wip_tracker = tracker
        node = SimpleNamespace(task=SimpleNamespace(id="task-queued", assigned_agent=AgentType.WORKER))

        result = runner._execute_task_node(node)

        assert result.success is False
        assert tracker.start_calls == [(AgentType.WORKER.value, "task-queued")]
        assert tracker.enqueued == [(AgentType.WORKER.value, "task-queued")]

    def test_execute_task_node_releases_wip_when_operator_paused(self):
        """A paused task must not leak the WIP slot it claimed before the pause check."""
        from unittest.mock import patch

        from vetinari.orchestration.graph_task_runner import GraphTaskRunnerMixin

        tracker = WIPTracker(WIPConfig(limits={AgentType.WORKER.value: 1, "default": 1}))
        runner = GraphTaskRunnerMixin()
        runner._wip_tracker = tracker
        node = SimpleNamespace(task=SimpleNamespace(id="task-paused", assigned_agent=AgentType.WORKER))
        control = {"paused": {AgentType.WORKER.value: {"reason": "maintenance"}}}

        with patch("vetinari.web.litestar_agents_api.get_agent_control_state", return_value=control):
            result = runner._execute_task_node(node)

        assert result.success is False
        assert tracker.get_active_count(AgentType.WORKER.value) == 0
        assert tracker.get_queue_depth() == 0


# -- check_andon_before_dispatch --------------------------------------------


class TestCheckAndonBeforeDispatch:
    def test_returns_true_when_andon_not_paused(self):
        from vetinari.workflow.wiring import check_andon_before_dispatch

        assert check_andon_before_dispatch() is True

    def test_returns_false_when_andon_is_paused(self):
        from vetinari.workflow.andon import get_andon_system
        from vetinari.workflow.wiring import check_andon_before_dispatch

        get_andon_system().raise_signal("test-gate", "critical", "deliberate halt")
        assert check_andon_before_dispatch() is False

    def test_returns_true_after_signal_acknowledged(self):
        from vetinari.workflow.andon import get_andon_system
        from vetinari.workflow.wiring import check_andon_before_dispatch

        andon = get_andon_system()
        andon.raise_signal("test-gate", "critical", "deliberate halt")
        andon.acknowledge(0)
        assert check_andon_before_dispatch() is True


# -- raise_quality_andon ----------------------------------------------------


class TestRaiseQualityAndon:
    def test_returns_andon_signal(self):
        from vetinari.workflow.andon import AndonSignal
        from vetinari.workflow.wiring import raise_quality_andon

        signal = raise_quality_andon("inspector-agent", "output below threshold")
        assert isinstance(signal, AndonSignal)

    def test_signal_has_critical_severity(self):
        from vetinari.workflow.wiring import raise_quality_andon

        signal = raise_quality_andon("cost-gate", "budget exceeded")
        assert signal.severity == "critical"

    def test_signal_records_source_and_message(self):
        from vetinari.workflow.wiring import raise_quality_andon

        signal = raise_quality_andon("spc-monitor", "Nelson rule 1 violation", ["t1", "t2"])
        assert signal.source == "spc-monitor"
        assert signal.message == "Nelson rule 1 violation"
        assert signal.affected_tasks == ["t1", "t2"]

    def test_andon_system_enters_paused_state(self):
        from vetinari.workflow.andon import get_andon_system
        from vetinari.workflow.wiring import raise_quality_andon

        raise_quality_andon("quality-gate", "fatal defect detected")
        assert get_andon_system().is_paused() is True

    def test_affected_tasks_defaults_to_empty_list(self):
        from vetinari.workflow.wiring import raise_quality_andon

        signal = raise_quality_andon("gate", "msg")
        assert signal.affected_tasks == []


# -- get_dispatch_status ----------------------------------------------------


class TestGetDispatchStatus:
    def test_returns_expected_keys(self):
        from vetinari.workflow.wiring import get_dispatch_status

        status = get_dispatch_status()
        assert "wip_utilization" in status
        assert "andon_paused" in status
        assert "queue_depth" in status
        assert "active_andon_signals" in status

    def test_andon_paused_false_when_clear(self):
        from vetinari.workflow.wiring import get_dispatch_status

        assert get_dispatch_status()["andon_paused"] is False

    def test_andon_paused_true_after_critical_signal(self):
        from vetinari.workflow.wiring import get_dispatch_status, raise_quality_andon

        raise_quality_andon("test", "halt")
        assert get_dispatch_status()["andon_paused"] is True

    def test_queue_depth_reflects_queued_tasks(self):
        from vetinari.workflow.wiring import dispatch_or_queue, get_dispatch_status

        agent = AgentType.FOREMAN.value
        dispatch_or_queue(agent, "t1")
        dispatch_or_queue(agent, "t2")  # queued

        assert get_dispatch_status()["queue_depth"] == 1

    def test_active_andon_signals_lists_sources(self):
        from vetinari.workflow.wiring import get_dispatch_status, raise_quality_andon

        raise_quality_andon("gate-A", "problem")
        status = get_dispatch_status()
        assert "gate-A" in status["active_andon_signals"]

    def test_wip_utilization_contains_agent_types(self):
        from vetinari.workflow.wiring import dispatch_or_queue, get_dispatch_status

        dispatch_or_queue(AgentType.WORKER.value, "task-x")
        utilization = get_dispatch_status()["wip_utilization"]
        assert AgentType.WORKER.value in utilization


# -- wire_workflow_subsystem ------------------------------------------------


class TestWireWorkflowSubsystem:
    def test_initialises_singleton(self):
        import vetinari.workflow.wiring as wiring_mod
        from vetinari.workflow.wiring import _get_wip_tracker, wire_workflow_subsystem

        assert wiring_mod._wip_tracker is None
        wire_workflow_subsystem()
        assert wiring_mod._wip_tracker is not None

    def test_idempotent_second_call(self):
        from vetinari.workflow.wiring import _get_wip_tracker, wire_workflow_subsystem

        wire_workflow_subsystem()
        first = _get_wip_tracker()
        wire_workflow_subsystem()
        second = _get_wip_tracker()
        assert first is second
