"""Shared test fixtures for Vetinari test suite."""
import pytest
import tempfile
import os
from unittest.mock import MagicMock, patch
from pathlib import Path


@pytest.fixture
def mock_adapter():
    """Mock LM Studio adapter that returns predictable responses."""
    adapter = MagicMock()
    adapter.generate.return_value = {"content": "Mock LLM response", "model": "test-model"}
    adapter.is_available.return_value = True
    adapter.get_loaded_models.return_value = [{"id": "test-model", "name": "Test Model"}]
    return adapter


@pytest.fixture
def tmp_project(tmp_path):
    """Create a temporary project directory with basic structure."""
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "main.py").write_text("def hello():\n    return 'world'\n")
    (tmp_path / "tests" / "test_main.py").write_text("def test_hello():\n    assert True\n")
    return tmp_path


@pytest.fixture
def sample_agent_task():
    """Create a sample AgentTask for testing."""
    from vetinari.agents.contracts import AgentTask, AgentType
    return AgentTask(
        task_id="test-task-001",
        agent_type=AgentType.BUILDER,
        description="Test task for unit testing",
        prompt="Write a hello world function",
        context={"test": True},
    )


@pytest.fixture
def sample_execution_context():
    """Create a sample ExecutionContext for testing."""
    from vetinari.execution_context import ExecutionContext, ExecutionMode
    ctx = ExecutionContext()
    ctx.mode = ExecutionMode.EXECUTION
    return ctx


@pytest.fixture
def training_data_path(tmp_path):
    """Create a temporary path for training data JSONL."""
    return str(tmp_path / "training_data.jsonl")


@pytest.fixture
def mock_web_search():
    """Mock web search that returns predictable results."""
    search = MagicMock()
    search.search.return_value = [
        {"title": "Test Result", "url": "https://example.com", "snippet": "Test snippet"}
    ]
    return search
