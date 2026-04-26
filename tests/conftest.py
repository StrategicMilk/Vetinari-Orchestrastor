"""
Shared test fixtures for Vetinari test suite (tests/ scope).

Provides common fixtures to reduce setup duplication across 60+ test files.
The root conftest.py handles: mock_adapter, mock_adapter_manager, tmp_config_dir,
flask_app, flask_client, _quiet_logging.

Fixtures here add: environment isolation, deterministic seeding, project_dir,
config_dir, temp_db.
"""

from __future__ import annotations

import logging
import random
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from vetinari.analytics.wiring import reset_all as reset_analytics
from vetinari.autonomy.governor import reset_governor
from vetinari.events import reset_event_bus
from vetinari.observability.checkpoints import reset_checkpoint_store
from vetinari.observability.otel_genai import reset_genai_tracer
from vetinari.telemetry import reset_telemetry
from vetinari.workflow.wiring import reset_wip_tracker

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# sys.modules Pollution Guard
# ---------------------------------------------------------------------------
# Snapshot taken before pytest collects any test files. Some test files do
# module-level ``sys.modules["vetinari.X"] = stub`` which permanently
# replaces real modules with stubs. Without cleanup, later tests that import
# the real module get the stub instead.
#
# _CLEAN_MODULES_KEYS     – frozenset of ALL module names at startup
# _CLEAN_VETINARI_MODULES – vetinari.* name→module for value restoration

_CLEAN_MODULES_KEYS: frozenset[str] = frozenset(sys.modules.keys())

_CLEAN_VETINARI_MODULES: dict[str, Any] = {k: v for k, v in sys.modules.items() if k.startswith("vetinari")}

# Packages with non-reentrant C extensions that crash on re-import.
# Never remove or replace these — even a caught re-import attempt can
# leave internal C state corrupted (e.g. torch._has_torch_function
# docstring re-registration).
_NEVER_TOUCH_PREFIXES: tuple[str, ...] = (
    "torch",
    "pyarrow",
    "numpy",
    "scipy",
    "pandas",
    "datasets",  # depends on pyarrow C extensions; re-import triggers ArrowKeyError
    "tests",  # test modules are loaded after the snapshot; never remove them
)


def _is_protected_module(key: str) -> bool:
    """Return True if *key* belongs to a package that must not be removed."""
    top = key.split(".")[0]
    return top in _NEVER_TOUCH_PREFIXES


def _cleanup_sys_modules() -> None:
    """Remove test-injected stubs and restore real vetinari modules."""
    # 0. Learn about real vetinari modules loaded since the last cleanup.
    #    Modules loaded during test collection aren't in the initial snapshot,
    #    so without this step a stub replacement would delete them and the
    #    next import would create a *new* module object — causing the
    #    dual-module problem (patches hit module A, code uses module B).
    #    Check isinstance(mod, types.ModuleType) to avoid capturing MagicMock
    #    stubs, which report hasattr(__file__) == True.
    for key, mod in list(sys.modules.items()):
        if (
            key.startswith("vetinari")
            and key not in _CLEAN_VETINARI_MODULES
            and isinstance(mod, types.ModuleType)
            and hasattr(mod, "__file__")
        ):
            _CLEAN_VETINARI_MODULES[key] = mod

    # 1. Restore any real vetinari modules that were overwritten with stubs
    for key, original_mod in _CLEAN_VETINARI_MODULES.items():
        if sys.modules.get(key) is not original_mod:
            sys.modules[key] = original_mod
    # 2. Remove ALL stubs injected by tests (vetinari AND third-party).
    #    A "stub" is a module with no __file__ that wasn't present at startup.
    #    Stubs with real child modules get replaced (not deleted) to avoid
    #    re-import crashes from non-reentrant C extensions (torch, pyarrow).
    #    Protected native packages are left alone only when a real child module
    #    is already loaded. Plain stubs like ``sys.modules["torch"] =
    #    ModuleType("torch")`` must be removed or SciPy/array-api probes treat
    #    them as broken real installs.
    stubs = [
        k
        for k in sys.modules
        if k not in _CLEAN_MODULES_KEYS
        and sys.modules[k] is not None  # None = Python import sentinel, leave alone
        and not hasattr(sys.modules[k], "__file__")
    ]
    for key in stubs:
        if _is_protected_module(key) and _has_real_child_modules(key):
            continue
        if _has_real_child_modules(key):
            _replace_stub_with_real(key)
        else:
            del sys.modules[key]


@pytest.fixture(autouse=True, scope="module")
def _restore_sys_modules():
    """Restore sys.modules to pre-collection state around each test module.

    Runs before and after each module to remove stubs injected both at
    collection time (module-level assignments) and at runtime (test body
    assignments).
    """
    _cleanup_sys_modules()
    yield
    _cleanup_sys_modules()


@pytest.fixture(autouse=True)
def _ensure_current_test_module_registered(request: pytest.FixtureRequest) -> None:
    """Keep the active test module visible to runtime annotation resolution.

    Some tests declare dataclasses inside test functions.  On Python 3.12,
    ``dataclasses.dataclass`` resolves string annotations via
    ``sys.modules[cls.__module__]``.  The suite's sys.modules isolation can
    remove importlib-loaded test modules before the function body executes, so
    re-register the current module defensively before each test.
    """
    module = getattr(request, "module", None)
    if isinstance(module, types.ModuleType):
        sys.modules[module.__name__] = module


def _has_real_child_modules(module_name: str) -> bool:
    """Check if any child of this module in sys.modules is a real (non-stub) module.

    Removing a parent package from sys.modules forces Python to re-execute
    its __init__.py on next import, which re-imports children.  If a child
    loaded a non-reentrant C extension (e.g. torch), the re-import crashes.
    """
    prefix = module_name + "."
    return any(k.startswith(prefix) and hasattr(sys.modules[k], "__file__") for k in sys.modules)


def _replace_stub_with_real(module_name: str) -> None:
    """Replace a stub module with its real counterpart via fresh import.

    When a stub has real child modules already loaded, we can't just delete
    it — re-importing the parent could crash non-reentrant C extensions.
    Instead, temporarily remove the stub and attempt a fresh import.  If
    the fresh import succeeds, the real module replaces the stub.  If it
    fails (e.g. torch C extension re-registration), restore the stub.
    """
    import importlib

    stub = sys.modules.pop(module_name)
    try:
        importlib.import_module(module_name)
    except Exception:
        # Re-import failed (C extension crash, ImportError, etc.) — restore stub
        sys.modules[module_name] = stub


# ---------------------------------------------------------------------------
# Magic Test Data Constants
# ---------------------------------------------------------------------------
# Use these instead of scattering hardcoded strings across test files.
# When a test checks a value, import from here so changes stay in one place.

TEST_TASK_ID = "task_1"  # Generic task identifier for unit tests
TEST_MODEL_ID = "test-model"  # Generic model identifier
TEST_BASE_URL = "http://example.com"  # Generic base URL for tests
TEST_ENDPOINT = "http://ep"  # Short provider endpoint used in adapter tests
TEST_API_KEY = "test-key"  # Placeholder API key (not a real credential)

# ---------------------------------------------------------------------------
# Tiered Test Timeouts
# ---------------------------------------------------------------------------

# Marker-based timeout overrides (default 30s set in pyproject.toml).
_MARKER_TIMEOUTS = {
    "integration": 60,
    "slow": 300,
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply tiered timeouts based on test markers.

    - @pytest.mark.integration → 60s
    - @pytest.mark.slow → 300s
    - default (via pyproject.toml) → 30s
    """
    for item in items:
        for marker_name, timeout_val in _MARKER_TIMEOUTS.items():
            if marker_name in item.keywords:
                # Only override if no explicit timeout marker already set
                if "timeout" not in item.keywords:
                    item.add_marker(pytest.mark.timeout(timeout_val))
                break  # first match wins (slow > integration if both present)


# ---------------------------------------------------------------------------
# Environment Safety
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch, tmp_path):
    """Redirect DB and state paths so tests never write to real files.

    Consumed by:
      - vetinari.constants.get_user_dir()     -> VETINARI_USER_DIR
      - vetinari/learning/quality_scorer.py  -> VETINARI_QUALITY_DB
      - vetinari/learning/model_selector.py  -> VETINARI_STATE_DIR
    """
    monkeypatch.setenv("VETINARI_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("VETINARI_QUALITY_DB", str(tmp_path / "test_quality.db"))
    monkeypatch.setenv("VETINARI_STATE_DIR", str(tmp_path / "state"))


@pytest.fixture(autouse=True)
def deterministic_seeds():
    """Seed Python and numpy RNGs for reproducible test runs."""
    random.seed(42)
    if _HAS_NUMPY:
        np.random.seed(42)  # Legacy API; seeds global state used by code calling np.random.*


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


@pytest.fixture(autouse=True)
def reset_preferences(monkeypatch):
    """Reset the global PreferencesManager singleton between tests.

    Prevents state leakage when tests that interact with user preferences
    run in the same process. Uses ``reset_preferences_manager()`` which sets
    the module-level singleton back to ``None`` so the next ``get_preferences_manager()``
    call creates a fresh instance.
    """
    yield
    try:
        from vetinari.web.preferences import reset_preferences_manager

        reset_preferences_manager()
    except ImportError:
        logger.debug("preferences module not available — reset_preferences_manager skipped safely")


@pytest.fixture(autouse=True)
def reset_governor_singleton():
    """Reset the global AutonomyGovernor singleton between tests.

    Prevents state leakage when tests that call governor methods (directly or
    via RollbackRegistry.check_quality_regression) run in the same process.
    """
    yield
    reset_governor()


@pytest.fixture(autouse=True)
def reset_wip_tracker_singleton():
    """Reset the global WIPTracker singleton between tests.

    Prevents state leakage when tests that call dispatch functions
    (which lazily init the singleton) run before tests that assert
    the singleton starts as None.
    """
    yield
    reset_wip_tracker()


@pytest.fixture(autouse=True)
def reset_shared_singletons():
    """Reset commonly-leaked singletons between tests.

    Covers observability, analytics, and telemetry singletons that cause
    test-ordering failures when left initialized across test boundaries.
    """
    yield
    reset_genai_tracer()
    reset_analytics()
    reset_telemetry()
    reset_checkpoint_store()


@pytest.fixture(autouse=True, scope="session")
def shutdown_event_bus_at_exit():
    """Shut down the EventBus async worker thread when the test session ends.

    Without this, the daemon thread polling the async queue keeps running
    after all tests complete, causing the pytest process to hang at exit.
    """
    yield
    reset_event_bus()


# ---------------------------------------------------------------------------
# Shared Agent / Orchestrator / Planning Mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_base_agent() -> Any:
    """Factory for BaseAgent mocks with standard interface.

    Returns a callable that creates configured agent mocks with optional overrides.

    Example:
        agent = mock_base_agent(success=False, output="failed")
    """

    def _make(
        success: bool = True,
        output: str = "mock result",
        passed: bool = True,
        score: float = 0.9,
    ) -> MagicMock:
        agent = MagicMock()
        agent.execute.return_value = MagicMock(success=success, output=output)
        agent.verify.return_value = MagicMock(passed=passed, score=score)
        agent.get_system_prompt.return_value = "You are a test agent."
        agent.get_capabilities.return_value = ["test_capability"]
        return agent

    return _make


@pytest.fixture
def mock_orchestrator() -> Any:
    """Factory for orchestrator mocks with configurable model pool.

    Returns a callable that creates configured orchestrator mocks.

    Example:
        orch = mock_orchestrator(models=[{"id": "custom-model", ...}])
    """

    def _make(
        status: str = "completed",
        output: str = "mock",
        models: list[dict[str, Any]] | None = None,
        chat_output: str = "mock response",
    ) -> MagicMock:
        orch = MagicMock()
        orch.execute_goal.return_value = {"status": status, "output": output}
        orch.model_pool.models = models or [
            {
                "id": "test-model",
                "name": "Test Model",
                "capabilities": ["code_gen"],
                "context_len": 4096,
                "memory_gb": 4,
                "version": "1.0",
            },
        ]
        orch.adapter.chat.return_value = {
            "output": chat_output,
            "latency_ms": 10,
            "tokens_used": 5,
        }
        return orch

    return _make


@pytest.fixture
def mock_plan_engine() -> Any:
    """Factory for PlanModeEngine mocks with configurable plan data.

    Returns a callable that creates configured plan engine mocks.

    Example:
        engine = mock_plan_engine(goal="deploy v2", subtasks=["build", "test"])
    """

    def _make(
        goal: str = "test",
        subtasks: list[Any] | None = None,
        plan_candidates: list[Any] | None = None,
        justification: str = "test justification",
    ) -> MagicMock:
        engine = MagicMock()
        mock_plan = MagicMock()
        mock_plan.subtasks = subtasks or []
        mock_plan.plan_candidates = plan_candidates or []
        mock_plan.plan_justification = justification
        mock_plan.to_dict.return_value = {
            "subtasks": mock_plan.subtasks,
            "plan_candidates": mock_plan.plan_candidates,
            "goal": goal,
        }
        engine.generate_plan.return_value = mock_plan
        return engine

    return _make


@pytest.fixture
def mock_adapter_response():
    """Factory fixture for adapter inference responses."""

    def _make(output="mock output", latency_ms=10, tokens_used=5):
        return {"output": output, "latency_ms": latency_ms, "tokens_used": tokens_used}

    return _make
