"""Integration tests for WorkerAgent.execute() across all mode groups.

Tests that each mode group (research, architecture, build, operations) routes
correctly and returns a well-formed AgentResult.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.agents.consolidated.worker_agent import WorkerAgent
from vetinari.agents.contracts import AgentResult, AgentTask
from vetinari.types import AgentType, StatusEnum


@pytest.fixture
def worker():
    """Create a WorkerAgent instance with inference mocked."""
    return WorkerAgent()


@pytest.fixture
def _make_task():
    """Factory for creating AgentTask instances with a given mode."""

    def _factory(mode: str, prompt: str = "Test task") -> AgentTask:
        return AgentTask(
            task_id="test-001",
            agent_type=AgentType.WORKER,
            description=f"Test {mode} mode",
            prompt=prompt,
            context={"mode": mode},
        )

    return _factory


class TestWorkerModeRouting:
    """Verify that WorkerAgent.execute() routes to correct internal handler."""

    def test_all_24_modes_are_registered(self, worker):
        """Every documented mode has a handler registered in MODES dict."""
        assert len(worker.MODES) == 24

    def test_unknown_mode_returns_error(self, worker, _make_task):
        """Unknown mode returns AgentResult with success=False."""
        task = _make_task("nonexistent_mode_xyz")
        result = worker.execute(task)
        assert isinstance(result, AgentResult)
        assert result.success is False
        assert result.errors

    @pytest.mark.parametrize(
        "mode",
        [
            "code_discovery",
            "domain_research",
            "api_lookup",
            "lateral_thinking",
            "ui_design",
            "database",
            "devops",
            "git_workflow",
        ],
    )
    def test_research_modes_dispatch(self, worker, _make_task, mode):
        """Research modes route to _dispatch_research."""
        assert worker.MODES[mode] == "_dispatch_research"

    @pytest.mark.parametrize(
        "mode",
        [
            "architecture",
            "risk_assessment",
            "ontological_analysis",
            "contrarian_review",
            "suggest",
        ],
    )
    def test_architecture_modes_dispatch(self, worker, _make_task, mode):
        """Architecture modes route to _dispatch_oracle."""
        assert worker.MODES[mode] == "_dispatch_oracle"

    @pytest.mark.parametrize("mode", ["build", "image_generation"])
    def test_build_modes_dispatch(self, worker, _make_task, mode):
        """Build modes route to _dispatch_builder."""
        assert worker.MODES[mode] == "_dispatch_builder"

    @pytest.mark.parametrize(
        "mode",
        [
            "documentation",
            "creative_writing",
            "cost_analysis",
            "experiment",
            "error_recovery",
            "synthesis",
            "improvement",
            "monitor",
            "devops_ops",
        ],
    )
    def test_operations_modes_dispatch(self, worker, _make_task, mode):
        """Operations modes route to _dispatch_operations."""
        assert worker.MODES[mode] == "_dispatch_operations"


class TestWorkerExecuteIntegration:
    """Integration tests that actually call execute() with mocked inference."""

    def test_build_mode_returns_agent_result(self, worker, _make_task):
        """Build mode produces an AgentResult via _dispatch_builder."""
        task = _make_task("build", prompt="Write a hello world function")
        task._quality_gate = False  # Skip quality gate to test routing only, not quality scoring
        mock_result = AgentResult(
            success=True,
            output={"code": "def hello(): return 'hello'"},
            metadata={"mode": "build"},
        )
        with patch.object(worker, "_dispatch_builder", return_value=mock_result):
            result = worker.execute(task)
        assert isinstance(result, AgentResult)
        assert result.success is True

    def test_documentation_mode_returns_agent_result(self, worker, _make_task):
        """Operations documentation mode produces an AgentResult."""
        task = _make_task("documentation", prompt="Document the API")
        task._quality_gate = False  # Skip quality gate to test routing only, not quality scoring
        mock_result = AgentResult(
            success=True,
            output={"title": "API Docs", "sections": []},
            metadata={"mode": "documentation"},
        )
        with patch.object(worker, "_dispatch_operations", return_value=mock_result):
            result = worker.execute(task)
        assert isinstance(result, AgentResult)
        assert result.success is True

    def test_suggest_mode_returns_agent_result(self, worker, _make_task):
        """Oracle suggest mode produces an AgentResult."""
        task = _make_task("suggest", prompt="Suggest improvements")
        task._quality_gate = False  # Skip quality gate to test routing only, not quality scoring
        mock_result = AgentResult(
            success=True,
            output={"suggestions": []},
            metadata={"mode": "suggest"},
        )
        with patch.object(worker, "_dispatch_oracle", return_value=mock_result):
            result = worker.execute(task)
        assert isinstance(result, AgentResult)
        assert result.success is True

    def test_code_discovery_returns_agent_result(self, worker, _make_task):
        """Research code_discovery mode produces an AgentResult."""
        task = _make_task("code_discovery", prompt="Find authentication code")
        task._quality_gate = False  # Skip quality gate to test routing only, not quality scoring
        mock_result = AgentResult(
            success=True,
            output={"findings": ["auth.py line 42"]},
            metadata={"mode": "code_discovery"},
        )
        with patch.object(worker, "_dispatch_research", return_value=mock_result):
            result = worker.execute(task)
        assert isinstance(result, AgentResult)
        assert result.success is True

    def test_default_mode_is_build(self, worker):
        """When no mode is specified, worker defaults to 'build'."""
        assert worker.DEFAULT_MODE == "build"

    def test_execute_handles_handler_exception(self, worker, _make_task):
        """If a handler raises, execute returns success=False with error."""
        task = _make_task("build", prompt="This will fail")
        with patch.object(worker, "_dispatch_builder", side_effect=RuntimeError("boom")):
            result = worker.execute(task)
        assert isinstance(result, AgentResult)
        assert result.success is False
        assert any("boom" in e for e in result.errors)
