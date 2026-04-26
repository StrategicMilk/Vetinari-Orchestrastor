"""Tests for AgentGraph.build_pipeline() and express lane metrics (Dept 4.1)."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from vetinari.exceptions import PlanningError
from vetinari.types import AgentType


class TestBuildPipeline:
    """Tests for AgentGraph.build_pipeline()."""

    @pytest.fixture
    def agent_graph(self):
        """Create an AgentGraph without initializing real agents."""
        with patch("vetinari.orchestration.agent_graph.AgentGraph.initialize"):
            from vetinari.orchestration.agent_graph import AgentGraph, ExecutionStrategy

            graph = AgentGraph(strategy=ExecutionStrategy.SEQUENTIAL)
            graph._initialized = True
            return graph

    def test_build_pipeline_express(self, agent_graph):
        """Express tier creates 2-node DAG: WORKER -> INSPECTOR."""
        plan = agent_graph.build_pipeline([AgentType.WORKER, AgentType.INSPECTOR])

        assert len(plan.nodes) == 2
        # First node has no dependencies
        first_id = plan.execution_order[0]
        assert len(plan.nodes[first_id].dependencies) == 0
        assert plan.nodes[first_id].task.assigned_agent == AgentType.WORKER
        # Second node depends on first
        second_id = plan.execution_order[1]
        assert first_id in plan.nodes[second_id].dependencies
        assert plan.nodes[second_id].task.assigned_agent == AgentType.INSPECTOR

    def test_build_pipeline_standard(self, agent_graph):
        """Standard tier creates 3-node DAG: FOREMAN -> WORKER -> INSPECTOR."""
        agents = [AgentType.FOREMAN, AgentType.WORKER, AgentType.INSPECTOR]
        plan = agent_graph.build_pipeline(agents)

        assert len(plan.nodes) == 3
        assert len(plan.execution_order) == 3
        # Verify linear chain: each node depends on previous
        for i in range(1, len(plan.execution_order)):
            current = plan.nodes[plan.execution_order[i]]
            prev_id = plan.execution_order[i - 1]
            assert prev_id in current.dependencies

    def test_build_pipeline_custom_full(self, agent_graph):
        """Custom tier creates 3-node DAG with full 3-agent pipeline."""
        agents = [
            AgentType.FOREMAN,
            AgentType.WORKER,
            AgentType.INSPECTOR,
        ]
        plan = agent_graph.build_pipeline(agents)

        assert len(plan.nodes) == 3
        assert len(plan.execution_order) == 3

    def test_build_pipeline_single_agent(self, agent_graph):
        """Single-agent pipeline creates 1-node DAG with no dependencies."""
        plan = agent_graph.build_pipeline([AgentType.WORKER])

        assert len(plan.nodes) == 1
        only_id = plan.execution_order[0]
        assert len(plan.nodes[only_id].dependencies) == 0

    def test_build_pipeline_empty_raises(self, agent_graph):
        """Empty agent list raises PlanningError."""
        with pytest.raises(PlanningError, match="at least one agent"):
            agent_graph.build_pipeline([])

    def test_build_pipeline_plan_id_format(self, agent_graph):
        """Plan ID starts with 'pipeline-'."""
        plan = agent_graph.build_pipeline([AgentType.WORKER])
        assert plan.plan_id.startswith("pipeline-")

    def test_build_pipeline_agent_types_preserved(self, agent_graph):
        """Agent types in the pipeline match the input list."""
        agents = [AgentType.FOREMAN, AgentType.WORKER, AgentType.INSPECTOR]
        plan = agent_graph.build_pipeline(agents)

        for i, tid in enumerate(plan.execution_order):
            node = plan.nodes[tid]
            assert node.task.assigned_agent == agents[i]


class TestExpressMetrics:
    """Tests for express lane metrics recording."""

    def test_record_express_success(self):
        """Successful express execution increments success counter."""
        with patch("vetinari.orchestration.two_layer.TwoLayerOrchestrator.__init__", return_value=None):
            from vetinari.orchestration.two_layer import TwoLayerOrchestrator

            orch = TwoLayerOrchestrator.__new__(TwoLayerOrchestrator)
            start = time.time()
            orch._record_express_metrics(True, start)

            assert orch._express_metrics["total"] == 1
            assert orch._express_metrics["success"] == 1
            assert orch._express_metrics["failed"] == 0

    def test_record_express_failure(self):
        """Failed express execution increments failed counter (not promoted — WO-13 fix)."""
        with patch("vetinari.orchestration.two_layer.TwoLayerOrchestrator.__init__", return_value=None):
            from vetinari.orchestration.two_layer import TwoLayerOrchestrator

            orch = TwoLayerOrchestrator.__new__(TwoLayerOrchestrator)
            start = time.time()
            orch._record_express_metrics(False, start)

            assert orch._express_metrics["total"] == 1
            assert orch._express_metrics["success"] == 0
            assert orch._express_metrics["failed"] == 1

    def test_record_express_accumulates(self):
        """Multiple express executions accumulate correctly."""
        with patch("vetinari.orchestration.two_layer.TwoLayerOrchestrator.__init__", return_value=None):
            from vetinari.orchestration.two_layer import TwoLayerOrchestrator

            orch = TwoLayerOrchestrator.__new__(TwoLayerOrchestrator)
            start = time.time()
            orch._record_express_metrics(True, start)
            orch._record_express_metrics(True, start)
            orch._record_express_metrics(False, start)

            assert orch._express_metrics["total"] == 3
            assert orch._express_metrics["success"] == 2
            assert orch._express_metrics["failed"] == 1

    def test_record_express_success_rate(self):
        """Success rate is correctly computed."""
        with patch("vetinari.orchestration.two_layer.TwoLayerOrchestrator.__init__", return_value=None):
            from vetinari.orchestration.two_layer import TwoLayerOrchestrator

            orch = TwoLayerOrchestrator.__new__(TwoLayerOrchestrator)
            start = time.time()
            for _ in range(7):
                orch._record_express_metrics(True, start)
            for _ in range(3):
                orch._record_express_metrics(False, start)

            rate = orch._express_metrics["success"] / orch._express_metrics["total"]
            assert abs(rate - 0.7) < 0.01
