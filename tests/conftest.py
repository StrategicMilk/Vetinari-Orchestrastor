"""
Shared test fixtures for Vetinari test suite (tests/ scope).

Provides common fixtures to reduce setup duplication across 60+ test files.
The root conftest.py handles: mock_adapter, mock_adapter_manager, tmp_config_dir,
flask_app, flask_client, fresh_shared_memory, _quiet_logging.

Fixtures here add: environment isolation, deterministic seeding, project_dir,
config_dir, temp_db.
"""

import os
import random

import pytest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Ignore test files from main that reference modules not present in v0.4.0
# ---------------------------------------------------------------------------
collect_ignore = [
    "test_architect_executor.py",
    "test_ast_indexer.py",
    "test_benchmark_learning_integration.py",
    "test_benchmark_seeder.py",
    "test_benchmarks.py",
    "test_context_compression.py",
    "test_cost_planner_skill.py",
    "test_creative_suggestions.py",
    "test_data_engineer_skill.py",
    "test_documentation_skill.py",
    "test_experimentation_manager_skill.py",
    "test_git_workflow.py",
    "test_goal_tracker.py",
    "test_grep_context.py",
    "test_integration_wiring.py",
    "test_manufacturing_workflow.py",
    "test_mcp_server.py",
    "test_milestones.py",
    "test_missing_skills.py",
    "test_model_switching.py",
    "test_phase_coordinator.py",
    "test_plan_cache.py",
    "test_planner_intelligence.py",
    "test_quality_gates.py",
    "test_security_auditor_skill.py",
    "test_skill_contract.py",
    "test_sse_streaming.py",
    "test_ui_preferences.py",
    "test_watch_mode.py",
    "test_planning.py",
    "test_planning_coverage.py",
]


# ---------------------------------------------------------------------------
# Environment Safety
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolate_env(monkeypatch, tmp_path):
    """Redirect DB and state paths so tests never write to real files.

    Consumed by:
      - vetinari/learning/quality_scorer.py  -> VETINARI_QUALITY_DB
      - vetinari/learning/model_selector.py  -> VETINARI_STATE_DIR
    """
    monkeypatch.setenv("VETINARI_QUALITY_DB", str(tmp_path / "test_quality.db"))
    monkeypatch.setenv("VETINARI_STATE_DIR", str(tmp_path / "state"))


@pytest.fixture(autouse=True)
def deterministic_seeds():
    """Seed Python's random module for reproducible test runs."""
    random.seed(42)
    yield


# ---------------------------------------------------------------------------
# Temporary Directories & Files
# ---------------------------------------------------------------------------

@pytest.fixture
def project_dir(tmp_path):
    """Temporary project directory for tests that need file I/O."""
    proj = tmp_path / "test_project"
    proj.mkdir()
    (proj / "manifest").mkdir()
    return proj


@pytest.fixture
def config_dir(tmp_path):
    """Temporary config directory (distinct from root conftest's tmp_config_dir)."""
    cfg = tmp_path / "config"
    cfg.mkdir()
    return cfg


# ---------------------------------------------------------------------------
# Database Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db(tmp_path):
    """Temporary SQLite database path string."""
    return str(tmp_path / "test.db")


# ---------------------------------------------------------------------------
# Shared Agent / Orchestrator / Planning Mocks
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_base_agent():
    """Reusable BaseAgent mock with standard interface."""
    agent = MagicMock()
    agent.execute.return_value = MagicMock(success=True, output="mock result")
    agent.verify.return_value = MagicMock(passed=True, score=0.9)
    agent.get_system_prompt.return_value = "You are a test agent."
    agent.get_capabilities.return_value = ["test_capability"]
    return agent


@pytest.fixture
def mock_orchestrator():
    """Reusable orchestrator mock."""
    orch = MagicMock()
    orch.execute_goal.return_value = {"status": "completed", "output": "mock"}
    orch.model_pool.models = [
        {"id": "test-model", "name": "Test Model", "capabilities": ["code_gen"],
         "context_len": 4096, "memory_gb": 4, "version": "1.0"}
    ]
    orch.adapter.chat.return_value = {
        "output": "mock response", "latency_ms": 10, "tokens_used": 5,
    }
    return orch


@pytest.fixture
def mock_plan_engine():
    """Reusable PlanModeEngine mock."""
    engine = MagicMock()
    mock_plan = MagicMock()
    mock_plan.subtasks = []
    mock_plan.plan_candidates = []
    mock_plan.plan_justification = "test justification"
    mock_plan.to_dict.return_value = {
        "subtasks": [], "plan_candidates": [], "goal": "test",
    }
    engine.generate_plan.return_value = mock_plan
    return engine


@pytest.fixture
def mock_adapter_response():
    """Factory fixture for adapter inference responses."""
    def _make(output="mock output", latency_ms=10, tokens_used=5):
        return {"output": output, "latency_ms": latency_ms, "tokens_used": tokens_used}
    return _make
