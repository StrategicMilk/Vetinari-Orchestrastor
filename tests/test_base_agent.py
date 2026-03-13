"""
Unit tests for Vetinari base agent class.
"""

import unittest
from typing import Any

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import AgentResult, AgentTask, VerificationResult
from vetinari.types import AgentType


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(AgentType.EXPLORER, config)

    def get_system_prompt(self) -> str:
        return "Mock system prompt"

    def execute(self, task: AgentTask) -> AgentResult:
        """Mock execute method."""
        task = self.prepare_task(task)

        if "error" in task.description.lower():
            return AgentResult(
                success=False,
                output=None,
                errors=["Mock error"]
            )

        result = AgentResult(
            success=True,
            output={"result": "mock output"}
        )
        task = self.complete_task(task, result)
        return result

    def verify(self, output: Any) -> VerificationResult:
        """Mock verify method."""
        if output is None:
            return VerificationResult(passed=False, issues=[], score=0.0)

        return VerificationResult(passed=True, issues=[], score=0.95)

    def get_capabilities(self) -> list[str]:
        return ["search", "analyze"]


class TestBaseAgent(unittest.TestCase):
    """Test the BaseAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent = MockAgent()

    def test_agent_initialization(self):
        """Test agent initialization."""
        assert not self.agent.is_initialized
        assert self.agent.agent_type == AgentType.EXPLORER
        assert self.agent.name == "Explorer"

    def test_agent_initialize(self):
        """Test agent initialization with context."""
        context = {"model": "gpt-4", "temperature": 0.7}
        self.agent.initialize(context)

        assert self.agent.is_initialized

    def test_agent_properties(self):
        """Test agent properties."""
        assert self.agent.name is not None
        assert self.agent.description is not None
        assert self.agent.default_model is not None
        assert self.agent.thinking_variant is not None

    def test_agent_metadata(self):
        """Test agent metadata."""
        metadata = self.agent.get_metadata()

        assert "agent_type" in metadata
        assert "name" in metadata
        assert "description" in metadata
        assert "capabilities" in metadata
        assert metadata["agent_type"] == AgentType.EXPLORER.value

    def test_agent_capabilities(self):
        """Test agent capabilities."""
        capabilities = self.agent.get_capabilities()

        assert len(capabilities) == 2
        assert "search" in capabilities
        assert "analyze" in capabilities

    def test_system_prompt(self):
        """Test system prompt retrieval."""
        prompt = self.agent.get_system_prompt()

        assert prompt is not None
        assert isinstance(prompt, str)

    def test_validate_task(self):
        """Test task validation."""
        # Valid task
        valid_task = AgentTask(
            task_id="t1",
            agent_type=AgentType.EXPLORER,
            description="Test",
            prompt="Test"
        )
        assert self.agent.validate_task(valid_task)

        # Invalid task
        invalid_task = AgentTask(
            task_id="t2",
            agent_type=AgentType.BUILDER,
            description="Test",
            prompt="Test"
        )
        assert not self.agent.validate_task(invalid_task)

    def test_task_lifecycle(self):
        """Test task preparation and completion."""
        task = AgentTask(
            task_id="t1",
            agent_type=AgentType.EXPLORER,
            description="Test",
            prompt="Test"
        )

        # Prepare task
        prepared = self.agent.prepare_task(task)
        assert prepared.started_at is not None

        # Complete task
        result = AgentResult(success=True, output={"data": "test"})
        completed = self.agent.complete_task(prepared, result)
        assert completed.completed_at is not None
        assert completed.result == result.output

    def test_execute_success(self):
        """Test successful task execution."""
        task = AgentTask(
            task_id="t1",
            agent_type=AgentType.EXPLORER,
            description="Find patterns",
            prompt="Find patterns"
        )

        result = self.agent.execute(task)

        assert result.success
        assert result.output is not None

    def test_execute_failure(self):
        """Test failed task execution."""
        task = AgentTask(
            task_id="t1",
            agent_type=AgentType.EXPLORER,
            description="Error task",
            prompt="Error task"
        )

        result = self.agent.execute(task)

        assert not result.success
        assert len(result.errors) == 1

    def test_verify_success(self):
        """Test successful verification."""
        output = {"data": "test"}
        result = self.agent.verify(output)

        assert result.passed
        assert result.score > 0.9

    def test_verify_failure(self):
        """Test failed verification."""
        output = None
        result = self.agent.verify(output)

        assert not result.passed
        assert result.score == 0.0

    def test_agent_repr(self):
        """Test agent string representation."""
        repr_str = repr(self.agent)

        assert "MockAgent" in repr_str
        assert "EXPLORER" in repr_str


class TestAgentConfiguration(unittest.TestCase):
    """Test agent configuration."""

    def test_agent_with_config(self):
        """Test agent initialization with configuration."""
        config = {"max_results": 20, "timeout": 30}
        agent = MockAgent(config)

        assert agent._config["max_results"] == 20
        assert agent._config["timeout"] == 30

    def test_agent_default_config(self):
        """Test agent with default configuration."""
        agent = MockAgent()

        assert agent._config is not None
        assert isinstance(agent._config, dict)


if __name__ == "__main__":
    unittest.main()
