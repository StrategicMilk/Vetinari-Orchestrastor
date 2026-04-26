"""Tests for WIPTracker integration with AgentGraph (Dept 4.6)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.types import AgentType
from vetinari.workflow import WIPConfig, WIPTracker


class TestWIPTrackerInAgentGraph:
    """Tests for AgentGraph WIPTracker integration."""

    @pytest.fixture
    def make_graph(self):
        """Return a factory that creates an AgentGraph with an optional WIPTracker."""

        def _build(wip_tracker=None):
            with patch("vetinari.orchestration.agent_graph.AgentGraph.initialize"):
                from vetinari.orchestration.agent_graph import AgentGraph, ExecutionStrategy

                graph = AgentGraph(
                    strategy=ExecutionStrategy.SEQUENTIAL,
                    wip_tracker=wip_tracker,
                )
                graph._initialized = True
                return graph

        return _build

    def test_graph_has_wip_tracker(self, make_graph):
        """AgentGraph creates a WIPTracker by default."""
        graph = make_graph()
        assert graph._wip_tracker is not None
        assert isinstance(graph._wip_tracker, WIPTracker)

    def test_graph_accepts_custom_tracker(self, make_graph):
        """AgentGraph accepts a custom WIPTracker."""
        custom = WIPTracker(WIPConfig(limits={AgentType.WORKER.value: 1, "default": 2}))
        graph = make_graph(wip_tracker=custom)
        assert graph._wip_tracker is custom

    def test_wip_blocking(self, make_graph):
        """Task is queued when WIP limit is reached."""
        from vetinari.agents.contracts import Task

        # WORKER limit of 1
        tracker = WIPTracker(WIPConfig(limits={AgentType.WORKER.value: 1, "default": 2}))
        tracker.start_task(AgentType.WORKER.value, "existing-task")  # fill the slot

        graph = make_graph(wip_tracker=tracker)
        # Register a WORKER agent so the WIP check is reached
        graph._agents[AgentType.WORKER] = MagicMock()

        # Create a task node that should be blocked
        from vetinari.orchestration.agent_graph import TaskNode

        task = Task(id="blocked-task", description="test", assigned_agent=AgentType.WORKER)
        node = TaskNode(task=task)

        result = graph._execute_task_node(node, {})
        assert not result.success
        assert "WIP limit" in result.errors[0]

    def test_wip_allows_when_capacity(self, make_graph):
        """Task proceeds when WIP capacity is available."""
        tracker = WIPTracker(WIPConfig(limits={AgentType.WORKER.value: 5, "default": 5}))
        make_graph(wip_tracker=tracker)

        # WIP should allow start (no active tasks)
        assert tracker.can_start(AgentType.WORKER.value) is True

    def test_wip_release_on_completion(self):
        """_emit_task_done releases WIP slot."""
        tracker = WIPTracker(WIPConfig(limits={AgentType.WORKER.value: 2, "default": 2}))
        tracker.start_task(AgentType.WORKER.value, "task-1")

        assert tracker.get_active_count(AgentType.WORKER.value) == 1
        tracker.complete_task(AgentType.WORKER.value, "task-1")
        assert tracker.get_active_count(AgentType.WORKER.value) == 0

    def test_pull_on_complete(self):
        """Completing a task pulls next queued task."""
        tracker = WIPTracker(WIPConfig(limits={AgentType.WORKER.value: 1, "default": 1}))
        tracker.start_task(AgentType.WORKER.value, "task-1")
        tracker.enqueue(AgentType.WORKER.value, "task-2")

        pulled = tracker.complete_task(AgentType.WORKER.value, "task-1")
        assert pulled is not None
        assert pulled["task_id"] == "task-2"
        # task-2 should now be active
        assert tracker.get_active_count(AgentType.WORKER.value) == 1

    def test_queue_ordering(self):
        """Tasks are pulled in FIFO order."""
        tracker = WIPTracker(WIPConfig(limits={AgentType.WORKER.value: 1, "default": 1}))
        tracker.start_task(AgentType.WORKER.value, "task-1")
        tracker.enqueue(AgentType.WORKER.value, "task-2")
        tracker.enqueue(AgentType.WORKER.value, "task-3")

        pulled = tracker.complete_task(AgentType.WORKER.value, "task-1")
        assert pulled["task_id"] == "task-2"


class TestConstraintsYAML:
    """Tests for wip_limits section in constraints.yaml."""

    def test_wip_limits_section_exists(self):
        """constraints.yaml has wip_limits section."""
        from pathlib import Path

        import yaml

        config_path = Path(__file__).parent.parent / "vetinari" / "config" / "standards" / "constraints.yaml"
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert "wip_limits" in config
        assert "foreman" in config["wip_limits"]
        assert "worker" in config["wip_limits"]
        assert "inspector" in config["wip_limits"]
        assert config["wip_limits"]["global_max"] == 8
