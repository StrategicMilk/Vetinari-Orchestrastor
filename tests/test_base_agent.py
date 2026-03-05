"""
Unit tests for Vetinari base agent class.
"""

import unittest
from typing import Any, Dict, List

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    TaskStatus,
    VerificationResult
)


class MockAgent(BaseAgent):
    """Mock agent for testing."""
    
    def __init__(self, config: Dict[str, Any] = None):
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
    
    def get_capabilities(self) -> List[str]:
        return ["search", "analyze"]


class TestBaseAgent(unittest.TestCase):
    """Test the BaseAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = MockAgent()
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertFalse(self.agent.is_initialized)
        self.assertEqual(self.agent.agent_type, AgentType.EXPLORER)
        self.assertEqual(self.agent.name, "Explorer")
    
    def test_agent_initialize(self):
        """Test agent initialization with context."""
        context = {"model": "gpt-4", "temperature": 0.7}
        self.agent.initialize(context)
        
        self.assertTrue(self.agent.is_initialized)
    
    def test_agent_properties(self):
        """Test agent properties."""
        self.assertIsNotNone(self.agent.name)
        self.assertIsNotNone(self.agent.description)
        self.assertIsNotNone(self.agent.default_model)
        self.assertIsNotNone(self.agent.thinking_variant)
    
    def test_agent_metadata(self):
        """Test agent metadata."""
        metadata = self.agent.get_metadata()
        
        self.assertIn("agent_type", metadata)
        self.assertIn("name", metadata)
        self.assertIn("description", metadata)
        self.assertIn("capabilities", metadata)
        self.assertEqual(metadata["agent_type"], AgentType.EXPLORER.value)
    
    def test_agent_capabilities(self):
        """Test agent capabilities."""
        capabilities = self.agent.get_capabilities()
        
        self.assertEqual(len(capabilities), 2)
        self.assertIn("search", capabilities)
        self.assertIn("analyze", capabilities)
    
    def test_system_prompt(self):
        """Test system prompt retrieval."""
        prompt = self.agent.get_system_prompt()
        
        self.assertIsNotNone(prompt)
        self.assertIsInstance(prompt, str)
    
    def test_validate_task(self):
        """Test task validation."""
        # Valid task
        valid_task = AgentTask(
            task_id="t1",
            agent_type=AgentType.EXPLORER,
            description="Test",
            prompt="Test"
        )
        self.assertTrue(self.agent.validate_task(valid_task))
        
        # Invalid task
        invalid_task = AgentTask(
            task_id="t2",
            agent_type=AgentType.BUILDER,
            description="Test",
            prompt="Test"
        )
        self.assertFalse(self.agent.validate_task(invalid_task))
    
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
        self.assertIsNotNone(prepared.started_at)
        
        # Complete task
        result = AgentResult(success=True, output={"data": "test"})
        completed = self.agent.complete_task(prepared, result)
        self.assertIsNotNone(completed.completed_at)
        self.assertEqual(completed.result, result.output)
    
    def test_execute_success(self):
        """Test successful task execution."""
        task = AgentTask(
            task_id="t1",
            agent_type=AgentType.EXPLORER,
            description="Find patterns",
            prompt="Find patterns"
        )
        
        result = self.agent.execute(task)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.output)
    
    def test_execute_failure(self):
        """Test failed task execution."""
        task = AgentTask(
            task_id="t1",
            agent_type=AgentType.EXPLORER,
            description="Error task",
            prompt="Error task"
        )
        
        result = self.agent.execute(task)
        
        self.assertFalse(result.success)
        self.assertEqual(len(result.errors), 1)
    
    def test_verify_success(self):
        """Test successful verification."""
        output = {"data": "test"}
        result = self.agent.verify(output)
        
        self.assertTrue(result.passed)
        self.assertGreater(result.score, 0.9)
    
    def test_verify_failure(self):
        """Test failed verification."""
        output = None
        result = self.agent.verify(output)
        
        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.0)
    
    def test_agent_repr(self):
        """Test agent string representation."""
        repr_str = repr(self.agent)
        
        self.assertIn("MockAgent", repr_str)
        # Canonical type values are lower_case after WS0.4 enum consolidation
        self.assertIn("explorer", repr_str)


class TestAgentConfiguration(unittest.TestCase):
    """Test agent configuration."""
    
    def test_agent_with_config(self):
        """Test agent initialization with configuration."""
        config = {"max_results": 20, "timeout": 30}
        agent = MockAgent(config)
        
        self.assertEqual(agent._config["max_results"], 20)
        self.assertEqual(agent._config["timeout"], 30)
    
    def test_agent_default_config(self):
        """Test agent with default configuration."""
        agent = MockAgent()
        
        self.assertIsNotNone(agent._config)
        self.assertIsInstance(agent._config, dict)


if __name__ == "__main__":
    unittest.main()
