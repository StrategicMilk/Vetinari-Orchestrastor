"""
Unit tests for Vetinari agent contracts and data models.
"""

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
from vetinari.types import AgentType, StatusEnum


class TestAgentType:
    """Test the AgentType enumeration."""

    def test_all_active_agent_types_defined(self):
        """Test that the 3 active agent types are defined."""
        agent_types = [
            AgentType.FOREMAN,
            AgentType.WORKER,
            AgentType.INSPECTOR,
        ]

        assert len(agent_types) == 3
        for agent_type in agent_types:
            assert isinstance(agent_type.value, str)
            assert len(agent_type.value) > 0

    def test_active_values_exist(self):
        """The enum should have the 3 active values (legacy values may also exist)."""
        assert AgentType.FOREMAN.value == "FOREMAN"
        assert AgentType.WORKER.value == "WORKER"
        assert AgentType.INSPECTOR.value == "INSPECTOR"


class TestAgentSpec:
    """Test the AgentSpec data class."""

    def test_agent_spec_creation(self):
        """Test creating an AgentSpec."""
        spec = AgentSpec(
            agent_type=AgentType.WORKER,
            name="Worker",
            description="Research and discovery tasks",
            default_model="qwen2.5-72b",
            thinking_variant="high",
            enabled=True,
        )

        assert spec.agent_type == AgentType.WORKER
        assert spec.name == "Worker"
        assert spec.enabled

    def test_agent_spec_serialization(self):
        """Test AgentSpec to_dict and from_dict."""
        original = AgentSpec(
            agent_type=AgentType.WORKER, name="Worker", description="Test", default_model="qwen2.5-72b"
        )

        # Serialize to dict
        spec_dict = original.to_dict()
        assert spec_dict["name"] == "Worker"

        # Deserialize from dict
        restored = AgentSpec.from_dict(spec_dict)
        assert restored.name == original.name
        assert restored.agent_type == original.agent_type


class TestTask:
    """Test the Task data class."""

    def test_task_creation(self):
        """Test creating a Task."""
        task = Task(id="t1", description="Test task", assigned_agent=AgentType.WORKER, dependencies=[])

        assert task.id == "t1"
        assert task.assigned_agent == AgentType.WORKER
        assert task.status == StatusEnum.PENDING

    def test_task_with_dependencies(self):
        """Test Task with dependencies."""
        task = Task(id="t2", description="Dependent task", assigned_agent=AgentType.WORKER, dependencies=["t1"])

        assert task.dependencies == ["t1"]

    def test_task_serialization(self):
        """Test Task serialization."""
        original = Task(id="t1", description="Test task", assigned_agent=AgentType.WORKER, dependencies=["t0"])

        task_dict = original.to_dict()
        restored = Task.from_dict(task_dict)

        assert restored.id == original.id
        assert restored.description == original.description
        assert restored.dependencies == original.dependencies


class TestPlan:
    """Test the Plan data class."""

    def test_plan_creation(self):
        """Test creating a Plan."""
        plan = Plan(plan_id="plan_001", goal="Test goal", phase=0)

        assert plan.plan_id == "plan_001"
        assert plan.goal == "Test goal"
        assert len(plan.tasks) == 0

    def test_plan_with_tasks(self):
        """Test Plan with tasks."""
        task = Task(id="t1", description="Task 1", assigned_agent=AgentType.WORKER)

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


class TestAgentTask:
    """Test the AgentTask data class."""

    def test_agent_task_creation(self):
        """Test creating an AgentTask."""
        task = AgentTask(
            task_id="t1", agent_type=AgentType.WORKER, description="Research patterns", prompt="Find patterns for X"
        )

        assert task.task_id == "t1"
        assert task.agent_type == AgentType.WORKER
        assert task.status == StatusEnum.PENDING

    def test_agent_task_from_task(self):
        """Test creating AgentTask from Task."""
        task = Task(id="t1", description="Research code", assigned_agent=AgentType.WORKER, dependencies=[])

        agent_task = AgentTask.from_task(task, "Find patterns")

        assert agent_task.task_id == "t1"
        assert agent_task.agent_type == AgentType.WORKER
        assert agent_task.prompt == "Find patterns"


class TestAgentResult:
    """Test the AgentResult data class."""

    def test_successful_result(self):
        """Test creating a successful AgentResult."""
        result = AgentResult(success=True, output={"findings": ["pattern1", "pattern2"]}, metadata={"count": 2})

        assert result.success
        assert result.output is not None
        assert len(result.errors) == 0

    def test_failed_result(self):
        """Test creating a failed AgentResult."""
        result = AgentResult(success=False, output=None, errors=["Error message"])

        assert not result.success
        assert len(result.errors) == 1


class TestVerificationResult:
    """Test the VerificationResult data class."""

    def test_passing_verification(self):
        """Test a passing verification result."""
        result = VerificationResult(passed=True, issues=[], score=0.95)

        assert result.passed
        assert result.score == 0.95

    def test_failing_verification(self):
        """Test a failing verification result."""
        result = VerificationResult(passed=False, issues=[{"type": "error", "message": "Invalid output"}], score=0.3)

        assert not result.passed
        assert len(result.issues) == 1


class TestAgentRegistry:
    """Test the agent registry."""

    def test_registry_has_agents(self):
        """Registry must contain agent specs."""
        assert len(AGENT_REGISTRY) > 0

    def test_registry_contains_active_types(self):
        """Active agent types are present in the registry."""
        active_types = [
            AgentType.FOREMAN,
            AgentType.WORKER,
            AgentType.INSPECTOR,
        ]
        for at in active_types:
            assert at in AGENT_REGISTRY, f"{at} missing from registry"

    def test_get_agent_spec_foreman(self):
        """Test getting the Foreman spec from the registry."""
        spec = get_agent_spec(AgentType.FOREMAN)

        assert isinstance(spec, AgentSpec)
        assert spec.agent_type == AgentType.FOREMAN

    def test_all_agent_specs_valid(self):
        """Test that all agent specs in registry are valid."""
        for agent_type, spec in AGENT_REGISTRY.items():
            assert isinstance(spec, AgentSpec)
            assert spec.agent_type == agent_type
            assert isinstance(spec.name, str)
            assert len(spec.name) > 0
            assert isinstance(spec.description, str)
            assert len(spec.description) > 0
