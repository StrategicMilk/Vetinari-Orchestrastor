"""Tests for vetinari.orchestration.execution_graph state-query methods.

Covers: can_retry, get_blocked_tasks, get_failed_tasks, get_execution_order,
to_dag_json — methods that were previously untested.
"""

from __future__ import annotations

import json

import pytest

from vetinari.orchestration.execution_graph import ExecutionGraph
from vetinari.types import StatusEnum


@pytest.fixture
def abc_graph() -> ExecutionGraph:
    """Build a simple A→B, A→C DAG for testing state-query methods."""
    g = ExecutionGraph(plan_id="test-plan", goal="test goal")
    g.add_task(task_id="A", description="first task")
    g.add_task(task_id="B", description="depends on A", depends_on=["A"])
    g.add_task(task_id="C", description="also depends on A", depends_on=["A"])
    return g


class TestCanRetry:
    """Tests for ExecutionGraph.can_retry()."""

    def test_unknown_task_returns_false(self, abc_graph: ExecutionGraph) -> None:
        assert abc_graph.can_retry("nonexistent") is False

    def test_fresh_task_can_retry(self, abc_graph: ExecutionGraph) -> None:
        """A task with retry_count < max_retries is retryable."""
        assert abc_graph.can_retry("A") is True

    def test_exhausted_retries_returns_false(self, abc_graph: ExecutionGraph) -> None:
        """A task that has used all retries cannot be retried."""
        node = abc_graph.nodes["A"]
        node.retry_count = node.max_retries
        assert abc_graph.can_retry("A") is False


class TestStateQueryMethods:
    """Tests for get_blocked_tasks, get_failed_tasks, get_completed_tasks."""

    def test_get_blocked_tasks(self, abc_graph: ExecutionGraph) -> None:
        abc_graph.nodes["B"].status = StatusEnum.BLOCKED
        blocked = abc_graph.get_blocked_tasks()
        assert len(blocked) == 1
        assert blocked[0].id == "B"

    def test_get_failed_tasks(self, abc_graph: ExecutionGraph) -> None:
        abc_graph.nodes["A"].status = StatusEnum.FAILED
        failed = abc_graph.get_failed_tasks()
        assert len(failed) == 1
        assert failed[0].id == "A"

    def test_get_completed_tasks(self, abc_graph: ExecutionGraph) -> None:
        abc_graph.nodes["A"].status = StatusEnum.COMPLETED
        abc_graph.nodes["C"].status = StatusEnum.COMPLETED
        completed = abc_graph.get_completed_tasks()
        assert len(completed) == 2
        ids = {t.id for t in completed}
        assert ids == {"A", "C"}

    def test_empty_graph_returns_empty_lists(self) -> None:
        g = ExecutionGraph(plan_id="empty", goal="nothing")
        assert g.get_blocked_tasks() == []
        assert g.get_failed_tasks() == []
        assert g.get_completed_tasks() == []


class TestGetExecutionOrder:
    """Tests for get_execution_order (layered topological ordering)."""

    def test_returns_layers(self, abc_graph: ExecutionGraph) -> None:
        """Execution order returns list of layers (each a list of tasks)."""
        layers = abc_graph.get_execution_order()
        assert isinstance(layers, list)
        # Should have at least 1 layer
        assert len(layers) >= 1


class TestToDagJson:
    """Tests for to_dag_json serialization."""

    def test_produces_valid_json(self, abc_graph: ExecutionGraph) -> None:
        result = abc_graph.to_dag_json()
        parsed = json.loads(result)
        assert "plan_id" in parsed
        assert parsed["plan_id"] == "test-plan"
        assert "nodes" in parsed

    def test_empty_graph_serializes(self) -> None:
        g = ExecutionGraph(plan_id="empty", goal="nothing")
        result = g.to_dag_json()
        parsed = json.loads(result)
        assert parsed["plan_id"] == "empty"
