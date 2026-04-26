"""
Phase 1 Tests - End-to-End Agent Orchestration

These tests demonstrate the completion of Phase 1 requirements:
- Librarian and Researcher integrated into orchestration
- End-to-end plans with 3+ agents executing successfully
- Dependency validation and result aggregation
"""

import socket
import sys

import pytest


def _local_inference_available() -> bool:
    """Check if local inference is available (llama-cpp-python loaded)."""
    try:
        import llama_cpp

        return True
    except ImportError:
        return False


_skip_no_local_inference = pytest.mark.skipif(
    not _local_inference_available(),
    reason="llama-cpp-python not available for local inference",
)

# Remove incomplete stubs left by earlier test files so real modules load
for _stubname in (
    "vetinari.agents.contracts",
    "vetinari.orchestration",
    "vetinari.orchestration.agent_graph",
    "vetinari.orchestration.two_layer",
    "vetinari.orchestration.plan_generator",
    "vetinari.orchestration.durable_execution",
    "vetinari.orchestration.execution_graph",
    "vetinari.orchestration.types",
):
    sys.modules.pop(_stubname, None)

from vetinari.agents.contracts import ExecutionPlan, Task
from vetinari.orchestration import ExecutionStrategy, get_agent_graph
from vetinari.types import AgentType


@_skip_no_local_inference
class Phase1EndToEndTests:
    """End-to-end tests for Phase 1."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.graph = get_agent_graph(strategy=ExecutionStrategy.ADAPTIVE)
        self.graph.initialize()

    def test_explorer_librarian_pipeline(self):
        """Test: Explorer -> Librarian pipeline (2 agents)."""
        plan = ExecutionPlan.create_new("Explore web framework patterns and look up documentation")

        # Task 1: Explorer searches for patterns
        plan.tasks.append(
            Task(
                id="t1",
                description="Search for web framework patterns",
                inputs=["goal"],
                outputs=["patterns", "references"],
                dependencies=[],
                assigned_agent=AgentType.WORKER,
                depth=0,
            )
        )

        # Task 2: Librarian looks up documentation based on findings
        plan.tasks.append(
            Task(
                id="t2",
                description="Look up documentation for found frameworks",
                inputs=["patterns"],
                outputs=["documentation", "fit_assessment"],
                dependencies=["t1"],
                assigned_agent=AgentType.WORKER,
                depth=1,
            )
        )

        # Execute plan
        results = self.graph.execute_plan(plan)

        # Verify
        assert len(results) >= 2
        assert "t1" in results
        assert "t2" in results

        # Check task order (t1 should complete before t2)
        exec_plan = self.graph.get_execution_plan(plan.plan_id)
        assert exec_plan.execution_order[0] == "t1"
        assert exec_plan.execution_order[1] == "t2"

    def test_researcher_analysis_pipeline(self):
        """Test: Explorer -> Researcher pipeline (2 agents)."""
        plan = ExecutionPlan.create_new("Research technology feasibility for web application")

        # Task 1: Explorer finds patterns
        plan.tasks.append(
            Task(
                id="t1",
                description="Explore available web technologies",
                inputs=["goal"],
                outputs=["tech_options"],
                dependencies=[],
                assigned_agent=AgentType.WORKER,
                depth=0,
            )
        )

        # Task 2: Researcher analyzes feasibility
        plan.tasks.append(
            Task(
                id="t2",
                description="Research feasibility of each technology option",
                inputs=["tech_options"],
                outputs=["feasibility_report", "recommendations"],
                dependencies=["t1"],
                assigned_agent=AgentType.WORKER,
                depth=1,
            )
        )

        # Execute plan
        results = self.graph.execute_plan(plan)

        # Verify
        assert len(results) >= 2
        for task_id in ["t1", "t2"]:
            assert task_id in results
            assert results[task_id].success

    def test_three_agent_pipeline(self):
        """Test: Explorer -> Librarian -> Researcher (3+ agents requirement)."""
        plan = ExecutionPlan.create_new("Complete technology research pipeline")

        # Task 1: Explorer
        plan.tasks.append(
            Task(
                id="t1",
                description="Explore code patterns",
                inputs=["goal"],
                outputs=["patterns"],
                dependencies=[],
                assigned_agent=AgentType.WORKER,
                depth=0,
            )
        )

        # Task 2: Librarian
        plan.tasks.append(
            Task(
                id="t2",
                description="Look up related documentation",
                inputs=["patterns"],
                outputs=["docs"],
                dependencies=["t1"],
                assigned_agent=AgentType.WORKER,
                depth=1,
            )
        )

        # Task 3: Researcher
        plan.tasks.append(
            Task(
                id="t3",
                description="Research feasibility",
                inputs=["patterns", "docs"],
                outputs=["feasibility"],
                dependencies=["t1", "t2"],
                assigned_agent=AgentType.WORKER,
                depth=2,
            )
        )

        # Execute plan
        results = self.graph.execute_plan(plan)

        # Verify
        assert len(results) >= 3

        # Verify all tasks completed
        for task_id in ["t1", "t2", "t3"]:
            assert task_id in results
            assert results[task_id].success, f"Task {task_id} failed: {results[task_id].errors}"

        # Verify execution order
        exec_plan = self.graph.get_execution_plan(plan.plan_id)
        t1_idx = exec_plan.execution_order.index("t1")
        t2_idx = exec_plan.execution_order.index("t2")
        t3_idx = exec_plan.execution_order.index("t3")

        assert t1_idx < t2_idx  # t1 before t2
        assert t2_idx < t3_idx  # t2 before t3

    def test_parallel_exploration_branches(self):
        """Test: Parallel independent exploration tasks."""
        plan = ExecutionPlan.create_new("Parallel research on two independent topics")

        # Task 1: Explore topic A
        plan.tasks.append(
            Task(
                id="t1",
                description="Explore database technologies",
                inputs=["goal"],
                outputs=["db_options"],
                dependencies=[],
                assigned_agent=AgentType.WORKER,
                depth=0,
            )
        )

        # Task 2: Explore topic B (independent of t1)
        plan.tasks.append(
            Task(
                id="t2",
                description="Explore frontend frameworks",
                inputs=["goal"],
                outputs=["frontend_options"],
                dependencies=[],
                assigned_agent=AgentType.WORKER,
                depth=0,
            )
        )

        # Task 3: Research findings from both (depends on t1 and t2)
        plan.tasks.append(
            Task(
                id="t3",
                description="Research integration approach",
                inputs=["db_options", "frontend_options"],
                outputs=["integration_plan"],
                dependencies=["t1", "t2"],
                assigned_agent=AgentType.WORKER,
                depth=1,
            )
        )

        # Execute plan
        results = self.graph.execute_plan(plan)

        # Verify
        assert len(results) >= 3

        # Verify all succeeded
        for task_id in results:
            assert results[task_id].success

        # Verify execution order (t1 and t2 should both come before t3)
        exec_plan = self.graph.get_execution_plan(plan.plan_id)
        t1_idx = exec_plan.execution_order.index("t1")
        t2_idx = exec_plan.execution_order.index("t2")
        t3_idx = exec_plan.execution_order.index("t3")

        assert t1_idx < t3_idx
        assert t2_idx < t3_idx

    def test_librarian_researcher_combined(self):
        """Test: Both Librarian and Researcher in same plan."""
        plan = ExecutionPlan.create_new("Research with documentation and domain analysis")

        # Task 1: Librarian looks up docs
        plan.tasks.append(
            Task(
                id="t1",
                description="Look up API documentation",
                inputs=["goal"],
                outputs=["api_docs"],
                dependencies=[],
                assigned_agent=AgentType.WORKER,
                depth=0,
            )
        )

        # Task 2: Researcher analyzes market
        plan.tasks.append(
            Task(
                id="t2",
                description="Analyze market feasibility",
                inputs=["goal"],
                outputs=["market_analysis"],
                dependencies=[],
                assigned_agent=AgentType.WORKER,
                depth=0,
            )
        )

        # Task 3: Synthesize results (different agent)
        plan.tasks.append(
            Task(
                id="t3",
                description="Synthesize research findings",
                inputs=["api_docs", "market_analysis"],
                outputs=["synthesis"],
                dependencies=["t1", "t2"],
                assigned_agent=AgentType.WORKER,
                depth=1,
            )
        )

        # Execute plan
        results = self.graph.execute_plan(plan)

        # Verify all tasks completed successfully
        assert len(results) >= 3
        for task_id in results:
            assert results[task_id].success


class Phase1AgentCapabilityTests:
    """Test agent capabilities for Phase 1."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.graph = get_agent_graph()
        self.graph.initialize()

    def test_explorer_capabilities(self):
        """Test Explorer agent (now ConsolidatedResearcher) has capabilities."""
        explorer = self.graph.get_agent(AgentType.WORKER)
        capabilities = explorer.get_capabilities()

        assert "code_discovery" in capabilities
        assert "domain_research" in capabilities
        assert len(capabilities) > 0

    def test_librarian_capabilities(self):
        """Test Librarian agent (now Worker) has capabilities."""
        librarian = self.graph.get_agent(AgentType.WORKER)
        capabilities = librarian.get_capabilities()

        assert "api_lookup" in capabilities
        assert len(capabilities) > 0

    def test_researcher_capabilities(self):
        """Test Researcher agent (now Worker) has capabilities."""
        researcher = self.graph.get_agent(AgentType.WORKER)
        capabilities = researcher.get_capabilities()

        assert "domain_research" in capabilities
        assert len(capabilities) > 0

    def test_all_agents_have_capabilities(self):
        """Test that all agents have at least one capability."""
        for agent_type in AgentType:
            agent = self.graph.get_agent(agent_type)
            if agent:
                capabilities = agent.get_capabilities()
                assert len(capabilities) > 0, f"{agent_type.value} has no capabilities"


class Phase1IntegrationTests:
    """Integration tests for Phase 1 components."""

    def test_plan_creation_and_execution(self):
        """Test creating and executing a plan."""
        graph = get_agent_graph()
        graph.initialize()

        plan = ExecutionPlan.create_new("Test plan")
        plan.tasks = [
            Task(id="t1", description="Task 1", assigned_agent=AgentType.WORKER),
        ]

        exec_plan = graph.create_execution_plan(plan)
        assert exec_plan is not None
        assert exec_plan.plan_id == plan.plan_id

    def test_agent_initialization_and_metadata(self):
        """Test agent initialization and metadata retrieval."""
        graph = get_agent_graph()
        graph.initialize()

        for agent_type in [AgentType.WORKER, AgentType.WORKER, AgentType.WORKER]:
            agent = graph.get_agent(agent_type)
            assert agent is not None
            assert agent.is_initialized

            metadata = agent.get_metadata()
            assert "agent_type" in metadata
            assert "name" in metadata
            assert "capabilities" in metadata


@_skip_no_local_inference
class Phase1AcceptanceCriteria:
    """Acceptance criteria tests for Phase 1."""

    def test_acceptance_criterion_3_plus_agents(self):
        """Acceptance Criterion: End-to-end plan with 3+ agents executes successfully."""
        graph = get_agent_graph(strategy=ExecutionStrategy.ADAPTIVE)
        graph.initialize()

        plan = ExecutionPlan.create_new("Phase 1 acceptance test with 3+ agents")

        # Create a 3-agent plan with dependencies
        plan.tasks = [
            Task(id="t1", description="Explore", assigned_agent=AgentType.WORKER),
            Task(id="t2", description="Research", assigned_agent=AgentType.WORKER, dependencies=["t1"]),
            Task(id="t3", description="Consult", assigned_agent=AgentType.WORKER, dependencies=["t2"]),
        ]

        # Execute the plan
        results = graph.execute_plan(plan)

        # Verify acceptance criteria
        assert len(results) >= 3, "Plan must have 3 or more agents"

        # All tasks must execute successfully
        for task_id, result in results.items():
            assert result.success, f"All agents must execute successfully. {task_id} failed: {result.errors}"

        # Execution order must respect dependencies
        exec_plan = graph.get_execution_plan(plan.plan_id)
        t1_idx = exec_plan.execution_order.index("t1")
        t2_idx = exec_plan.execution_order.index("t2")
        t3_idx = exec_plan.execution_order.index("t3")

        assert t1_idx < t2_idx
        assert t2_idx < t3_idx
