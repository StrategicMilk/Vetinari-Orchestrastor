"""
Unit tests for Vetinari agent contracts and data models.
"""

import unittest

from vetinari.agents.contracts import (
    AGENT_REGISTRY,
    AgentResult,
    AgentSpec,
    AgentTask,
    Plan,
    Task,
    VerificationResult,
    get_agent_spec,
)
from vetinari.types import AgentType, TaskStatus


class TestAgentType(unittest.TestCase):
    """Test the AgentType enumeration."""

    def test_all_agent_types_defined(self):
        """Test that all 15 agent types are defined."""
        agent_types = [
            AgentType.PLANNER,
            AgentType.EXPLORER,
            AgentType.LIBRARIAN,
            AgentType.ORACLE,
            AgentType.RESEARCHER,
            AgentType.EVALUATOR,
            AgentType.SYNTHESIZER,
            AgentType.BUILDER,
            AgentType.UI_PLANNER,
            AgentType.SECURITY_AUDITOR,
            AgentType.DATA_ENGINEER,
            AgentType.DOCUMENTATION_AGENT,
            AgentType.COST_PLANNER,
            AgentType.TEST_AUTOMATION,
            AgentType.EXPERIMENTATION_MANAGER
        ]

        assert len(agent_types) == 15
        for agent_type in agent_types:
            assert agent_type.value is not None


class TestAgentSpec(unittest.TestCase):
    """Test the AgentSpec data class."""

    def test_agent_spec_creation(self):
        """Test creating an AgentSpec."""
        spec = AgentSpec(
            agent_type=AgentType.EXPLORER,
            name="Explorer",
            description="Fast code discovery",
            default_model="gpt-4",
            thinking_variant="high",
            enabled=True
        )

        assert spec.agent_type == AgentType.EXPLORER
        assert spec.name == "Explorer"
        assert spec.enabled

    def test_agent_spec_serialization(self):
        """Test AgentSpec to_dict and from_dict."""
        original = AgentSpec(
            agent_type=AgentType.EXPLORER,
            name="Explorer",
            description="Test",
            default_model="gpt-4"
        )

        # Serialize to dict
        spec_dict = original.to_dict()
        assert spec_dict["name"] == "Explorer"

        # Deserialize from dict
        restored = AgentSpec.from_dict(spec_dict)
        assert restored.name == original.name
        assert restored.agent_type == original.agent_type


class TestTask(unittest.TestCase):
    """Test the Task data class."""

    def test_task_creation(self):
        """Test creating a Task."""
        task = Task(
            id="t1",
            description="Test task",
            assigned_agent=AgentType.EXPLORER,
            dependencies=[]
        )

        assert task.id == "t1"
        assert task.assigned_agent == AgentType.EXPLORER
        assert task.status == TaskStatus.PENDING

    def test_task_with_dependencies(self):
        """Test Task with dependencies."""
        task = Task(
            id="t2",
            description="Dependent task",
            assigned_agent=AgentType.BUILDER,
            dependencies=["t1"]
        )

        assert task.dependencies == ["t1"]

    def test_task_serialization(self):
        """Test Task serialization."""
        original = Task(
            id="t1",
            description="Test task",
            assigned_agent=AgentType.EXPLORER,
            dependencies=["t0"]
        )

        task_dict = original.to_dict()
        restored = Task.from_dict(task_dict)

        assert restored.id == original.id
        assert restored.description == original.description
        assert restored.dependencies == original.dependencies


class TestPlan(unittest.TestCase):
    """Test the Plan data class."""

    def test_plan_creation(self):
        """Test creating a Plan."""
        plan = Plan(
            plan_id="plan_001",
            goal="Test goal",
            phase=0
        )

        assert plan.plan_id == "plan_001"
        assert plan.goal == "Test goal"
        assert len(plan.tasks) == 0

    def test_plan_with_tasks(self):
        """Test Plan with tasks."""
        task = Task(
            id="t1",
            description="Task 1",
            assigned_agent=AgentType.EXPLORER
        )

        plan = Plan.create_new("Test goal")
        plan.tasks.append(task)

        assert len(plan.tasks) == 1
        assert plan.tasks[0].id == "t1"

    def test_plan_serialization(self):
        """Test Plan serialization."""
        plan = Plan.create_new("Test goal")
        plan.phase = 1
        plan.notes = "Test notes"

        plan_dict = plan.to_dict()
        restored = Plan.from_dict(plan_dict)

        assert restored.plan_id == plan.plan_id
        assert restored.goal == plan.goal
        assert restored.phase == plan.phase


class TestAgentTask(unittest.TestCase):
    """Test the AgentTask data class."""

    def test_agent_task_creation(self):
        """Test creating an AgentTask."""
        task = AgentTask(
            task_id="t1",
            agent_type=AgentType.EXPLORER,
            description="Explore patterns",
            prompt="Find patterns for X"
        )

        assert task.task_id == "t1"
        assert task.agent_type == AgentType.EXPLORER
        assert task.status == TaskStatus.PENDING

    def test_agent_task_from_task(self):
        """Test creating AgentTask from Task."""
        task = Task(
            id="t1",
            description="Explore code",
            assigned_agent=AgentType.EXPLORER,
            dependencies=[]
        )

        agent_task = AgentTask.from_task(task, "Find patterns")

        assert agent_task.task_id == "t1"
        assert agent_task.agent_type == AgentType.EXPLORER
        assert agent_task.prompt == "Find patterns"


class TestAgentResult(unittest.TestCase):
    """Test the AgentResult data class."""

    def test_successful_result(self):
        """Test creating a successful AgentResult."""
        result = AgentResult(
            success=True,
            output={"findings": ["pattern1", "pattern2"]},
            metadata={"count": 2}
        )

        assert result.success
        assert result.output is not None
        assert len(result.errors) == 0

    def test_failed_result(self):
        """Test creating a failed AgentResult."""
        result = AgentResult(
            success=False,
            output=None,
            errors=["Error message"]
        )

        assert not result.success
        assert len(result.errors) == 1


class TestVerificationResult(unittest.TestCase):
    """Test the VerificationResult data class."""

    def test_passing_verification(self):
        """Test a passing verification result."""
        result = VerificationResult(
            passed=True,
            issues=[],
            score=0.95
        )

        assert result.passed
        assert result.score == 0.95

    def test_failing_verification(self):
        """Test a failing verification result."""
        result = VerificationResult(
            passed=False,
            issues=[{"type": "error", "message": "Invalid output"}],
            score=0.3
        )

        assert not result.passed
        assert len(result.issues) == 1


class TestAgentRegistry(unittest.TestCase):
    """Test the agent registry."""

    def test_registry_has_all_agents(self):
        """Test that registry contains the expected number of agents (grows as new agents are added)."""
        # Original 15 agents + IMPROVEMENT + USER_INTERACTION + DEVOPS + VERSION_CONTROL + ERROR_RECOVERY + CONTEXT_MANAGER
        assert len(AGENT_REGISTRY) >= 15
        # Verify the original 15 are still present
        original_types = [
            AgentType.PLANNER, AgentType.EXPLORER, AgentType.LIBRARIAN, AgentType.ORACLE,
            AgentType.RESEARCHER, AgentType.EVALUATOR, AgentType.SYNTHESIZER, AgentType.BUILDER,
            AgentType.UI_PLANNER, AgentType.SECURITY_AUDITOR, AgentType.DATA_ENGINEER,
            AgentType.DOCUMENTATION_AGENT, AgentType.COST_PLANNER, AgentType.TEST_AUTOMATION,
            AgentType.EXPERIMENTATION_MANAGER,
        ]
        for at in original_types:
            assert at in AGENT_REGISTRY, f"{at} missing from registry"

    def test_get_agent_spec(self):
        """Test getting an agent spec from registry."""
        spec = get_agent_spec(AgentType.EXPLORER)

        assert spec is not None
        assert spec.agent_type == AgentType.EXPLORER
        assert spec.name == "Explorer"

    def test_all_agent_specs_valid(self):
        """Test that all agent specs in registry are valid."""
        for agent_type in AgentType:
            spec = get_agent_spec(agent_type)
            assert spec is not None
            assert spec.agent_type == agent_type
            assert spec.name is not None
            assert spec.description is not None


if __name__ == "__main__":
    unittest.main()
