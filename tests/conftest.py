"""Shared test fixtures for Vetinari test suite."""
import pytest
import tempfile
import os
from unittest.mock import MagicMock, patch
from pathlib import Path


# ─── Autouse: singleton resets ────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset all global singletons after each test."""
    yield
    _safe_reset("vetinari.telemetry", "reset_telemetry")
    _safe_reset("vetinari.dashboard.api", "reset_dashboard")
    _safe_reset("vetinari.dashboard.alerts", "reset_alert_engine")
    _safe_reset("vetinari.dashboard.log_aggregator", "reset_log_aggregator")
    _safe_reset("vetinari.analytics.cost", "reset_cost_tracker")
    _safe_reset("vetinari.analytics.sla", "reset_sla_tracker")
    _safe_reset("vetinari.analytics.anomaly", "reset_anomaly_detector")
    _safe_reset("vetinari.analytics.forecasting", "reset_forecaster")


def _safe_reset(module_path: str, func_name: str):
    """Import and call a reset function, swallowing ImportError."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        getattr(mod, func_name)()
    except (ImportError, AttributeError):
        pass


# ─── Shared fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_adapter():
    """Mock LM Studio adapter that returns predictable responses."""
    adapter = MagicMock()
    adapter.generate.return_value = {"content": "Mock LLM response", "model": "test-model"}
    adapter.infer.return_value = "mocked response"
    adapter.chat.return_value = {"content": "mocked"}
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
def tmp_db(tmp_path):
    """Shared temp database path."""
    return str(tmp_path / "test.db")


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


@pytest.fixture(scope="session")
def flask_test_app():
    """Session-scoped Flask test app (created once, reused across all tests)."""
    try:
        from vetinari.web_ui import create_app
        app = create_app(testing=True)
        app.config["TESTING"] = True
        return app
    except (ImportError, Exception):
        return None


@pytest.fixture
def flask_client(flask_test_app):
    """Per-test Flask test client."""
    if flask_test_app is None:
        pytest.skip("Flask app not available")
    with flask_test_app.test_client() as client:
        yield client
