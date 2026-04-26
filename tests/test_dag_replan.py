"""Tests for the Mid-Execution DAG Replanning Hook (Dept 7.9)."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from tests.factories import make_agent_result, make_task
from vetinari.agents.contracts import AgentResult, Task
from vetinari.orchestration.agent_graph import (
    AgentGraph,
    ExecutionPlan,
    ReplanResult,
    TaskNode,
)
from vetinari.types import AgentType, StatusEnum


@pytest.fixture
def graph():
    """Create an AgentGraph for testing without initialization."""
    g = AgentGraph.__new__(AgentGraph)
    g._agents = {}
    g._strategy = MagicMock()
    g._strategy.value = "adaptive"
    g._max_workers = 2
    g._execution_plans = {}
    g._initialized = False
    g._goal_tracker = None
    g._milestone_manager = None
    g._quality_reviewed_agents = set()
    g._wip_tracker = None
    return g


class TestShouldReplan:
    """Test _should_replan heuristics for each metadata flag."""

    def test_researcher_scope_changed(self, graph):
        """Worker with scope_changed metadata triggers replan."""
        task = make_task(id="r1", assigned_agent=AgentType.WORKER)
        result = make_agent_result(output="test output", metadata={"scope_changed": True})
        assert graph._should_replan(task, result) is True

    def test_oracle_architecture_concern(self, graph):
        """Worker with architecture_concern metadata triggers replan."""
        task = make_task(id="o1", assigned_agent=AgentType.WORKER)
        result = make_agent_result(output="test output", metadata={"architecture_concern": True})
        assert graph._should_replan(task, result) is True

    def test_builder_complexity_exceeded(self, graph):
        """Worker with complexity_exceeded metadata triggers replan."""
        task = make_task(id="b1", assigned_agent=AgentType.WORKER)
        result = make_agent_result(output="test output", metadata={"complexity_exceeded": True})
        assert graph._should_replan(task, result) is True

    def test_delegate_to_triggers_replan(self, graph):
        """Any agent with delegate_to metadata triggers replan."""
        task = make_task(id="t1", assigned_agent=AgentType.WORKER)
        result = make_agent_result(output="test output", metadata={"delegate_to": "RESEARCHER"})
        assert graph._should_replan(task, result) is True

    def test_no_flags_no_replan(self, graph):
        """No metadata flags means no replan needed."""
        task = make_task(id="t1", assigned_agent=AgentType.WORKER)
        result = make_agent_result(output="test output", metadata={})
        assert graph._should_replan(task, result) is False

    def test_empty_metadata_no_replan(self, graph):
        """Missing metadata attribute means no replan needed."""
        task = make_task(id="t1", assigned_agent=AgentType.WORKER)
        result = make_agent_result(output="test output")
        assert graph._should_replan(task, result) is False

    def test_researcher_without_scope_change_no_replan(self, graph):
        """Researcher without scope_changed flag doesn't trigger replan."""
        task = make_task(id="r1", assigned_agent=AgentType.WORKER)
        result = make_agent_result(output="test output", metadata={"some_other_flag": True})
        assert graph._should_replan(task, result) is False


class TestReplaceRemainingTasks:
    """Test _replace_remaining_tasks swaps pending tasks."""

    def test_replaces_pending_keeps_completed(self, graph):
        """Pending tasks are replaced, completed tasks are kept."""
        plan = ExecutionPlan(
            plan_id="test",
            original_plan=MagicMock(),
        )
        # Add a completed task and two pending tasks
        completed_task = make_task(id="done")
        pending1 = make_task(id="p1")
        pending2 = make_task(id="p2")

        plan.nodes = {
            "done": TaskNode(task=completed_task, status=StatusEnum.COMPLETED),
            "p1": TaskNode(task=pending1, status=StatusEnum.PENDING),
            "p2": TaskNode(task=pending2, status=StatusEnum.PENDING),
        }
        plan.execution_order = ["done", "p1", "p2"]

        # Replace with one new task
        new_task = make_task(id="new1")
        graph._replace_remaining_tasks(plan, [new_task])

        assert "done" in plan.nodes
        assert "p1" not in plan.nodes
        assert "p2" not in plan.nodes
        assert "new1" in plan.nodes
        assert plan.nodes["new1"].status == StatusEnum.PENDING

    def test_replace_with_empty_removes_pending(self, graph):
        """Replacing with empty list removes all pending tasks."""
        plan = ExecutionPlan(plan_id="test", original_plan=MagicMock())
        plan.nodes = {
            "p1": TaskNode(task=make_task(id="p1"), status=StatusEnum.PENDING),
        }
        plan.execution_order = ["p1"]

        graph._replace_remaining_tasks(plan, [])

        assert len(plan.nodes) == 0


class TestReplanResult:
    """Test the ReplanResult dataclass."""

    def test_default_values(self):
        """ReplanResult has sensible defaults."""
        result = ReplanResult()
        assert result.new_tasks == []
        assert result.replan_output == ""

    def test_with_tasks(self):
        """ReplanResult stores tasks correctly."""
        tasks = [make_task(id="t1"), make_task(id="t2")]
        result = ReplanResult(new_tasks=tasks, replan_output="Adjusted scope")
        assert len(result.new_tasks) == 2
        assert result.replan_output == "Adjusted scope"
