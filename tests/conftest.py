"""
Shared test fixtures for Vetinari test suite (tests/ scope).

Provides common fixtures to reduce setup duplication across 60+ test files.
The root conftest.py handles: mock_adapter, mock_adapter_manager, tmp_config_dir,
flask_app, flask_client, fresh_shared_memory, _quiet_logging.

Fixtures here add: environment isolation, deterministic seeding, project_dir,
config_dir, temp_db.
"""

import random

import pytest
from unittest.mock import MagicMock


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
