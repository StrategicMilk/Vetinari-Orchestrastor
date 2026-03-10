"""
Reusable module stub factories for test isolation.

Many test files need to install sys.modules stubs BEFORE importing
vetinari modules.  This file centralises the common patterns so each
test file can call a one-liner instead of 80-250 lines of inline setup.

Usage (at module level, before importing vetinari.*):

    from tests.conftest_stubs import (
        make_module_stub, install_stub, install_blueprint_stubs,
        install_orchestrator_stubs, install_telemetry_stubs,
    )
    install_orchestrator_stubs()
    install_blueprint_stubs()
    # ... now safe to import vetinari.web_ui etc.
"""

import os
import sys
import types
from unittest.mock import MagicMock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Core factory helpers
# ---------------------------------------------------------------------------

def make_module_stub(name: str, **attrs) -> types.ModuleType:
    """Create a stub module with optional attributes.

    Does NOT install into sys.modules — use ``install_stub()`` for that.
    """
    mod = types.ModuleType(name)
    mod.__path__ = []
    mod.__package__ = name
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def install_stub(name: str, **attrs) -> types.ModuleType:
    """Create a stub module, install it in sys.modules, and return it.

    Intermediate package stubs are created automatically.
    If the module already exists in sys.modules, its attributes are
    updated (not overwritten) with *attrs*.
    """
    parts = name.split(".")
    # Ensure parent packages exist
    for i in range(1, len(parts)):
        pkg = ".".join(parts[:i])
        if pkg not in sys.modules:
            pkg_mod = types.ModuleType(pkg)
            pkg_mod.__path__ = []
            pkg_mod.__package__ = pkg
            sys.modules[pkg] = pkg_mod

    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = []
        mod.__package__ = name
        sys.modules[name] = mod
    else:
        mod = sys.modules[name]

    for k, v in attrs.items():
        if not hasattr(mod, k):
            setattr(mod, k, v)
    return mod


# ---------------------------------------------------------------------------
# Blueprint stub factory (for web_ui tests)
# ---------------------------------------------------------------------------

def make_blueprint_stub(dotted_name: str, url_prefix: str | None = None):
    """Create a stub module containing a real Flask Blueprint.

    Installs intermediate package stubs automatically.
    Returns the module.
    """
    from flask import Blueprint

    short = dotted_name.split(".")[-1].replace("_routes", "")
    bp = Blueprint(short, dotted_name, url_prefix=url_prefix)
    mod = types.ModuleType(dotted_name)
    mod.bp = bp

    parts = dotted_name.split(".")
    for i in range(1, len(parts)):
        pkg_name = ".".join(parts[:i])
        if pkg_name not in sys.modules:
            pkg_mod = types.ModuleType(pkg_name)
            pkg_mod.__path__ = []
            sys.modules[pkg_name] = pkg_mod
    sys.modules[dotted_name] = mod
    return mod


def install_blueprint_stubs():
    """Install the standard set of web route blueprint stubs."""
    make_blueprint_stub("vetinari.web.adr_routes", "/api/adr")
    make_blueprint_stub("vetinari.web.decomposition_routes", "/api/decomposition")
    make_blueprint_stub("vetinari.web.ponder_routes", "/api/ponder")
    make_blueprint_stub("vetinari.web.rules_routes", "/api/rules")
    make_blueprint_stub("vetinari.web.admin_routes", "/api/admin")
    make_blueprint_stub("vetinari.web.training_routes", "/api/training")


# ---------------------------------------------------------------------------
# Common dependency stubs
# ---------------------------------------------------------------------------

def install_orchestrator_stubs():
    """Install orchestrator + plan mode + telemetry stubs.

    Returns a dict of the created mocks for assertion use:
    ``{"orchestrator": ..., "plan_engine": ..., "telemetry": ...}``
    """
    mocks = {}

    # Orchestrator
    mock_orch = MagicMock(name="Orchestrator_instance")
    mock_orch.model_pool.models = [
        {
            "id": "test-model",
            "name": "Test Model",
            "capabilities": ["code_gen"],
            "context_len": 4096,
            "memory_gb": 4,
            "version": "1.0",
        }
    ]
    mock_orch.adapter.chat.return_value = {
        "output": "mock response",
        "latency_ms": 10,
        "tokens_used": 5,
    }
    mock_orch.upgrader.check_for_upgrades.return_value = []
    mock_orch.executor._parse_code_blocks.return_value = {}
    mock_cls = MagicMock(return_value=mock_orch)
    install_stub("vetinari.orchestrator", Orchestrator=mock_cls)
    mocks["orchestrator"] = mock_orch

    # Plan mode engine
    mock_plan = MagicMock()
    mock_plan.subtasks = []
    mock_plan.plan_candidates = []
    mock_plan.to_dict.return_value = {
        "subtasks": [],
        "plan_candidates": [],
        "goal": "test",
    }
    mock_engine_cls = MagicMock(
        return_value=MagicMock(
            generate_plan=MagicMock(return_value=mock_plan)
        )
    )
    install_stub("vetinari.plan_mode", PlanModeEngine=mock_engine_cls)
    install_stub("vetinari.plan_types", PlanGenerationRequest=MagicMock)
    install_stub("vetinari.planning_engine", PlanningEngine=MagicMock())
    mocks["plan_engine"] = mock_plan

    return mocks


def install_telemetry_stubs():
    """Install telemetry and analytics cost tracker stubs."""
    mock_telemetry = MagicMock()
    mock_telemetry.get_summary.return_value = {
        "total_tokens_used": 100,
        "total_cost_usd": 0.5,
        "by_model": {},
        "by_provider": {},
        "session_requests": 5,
    }
    install_stub(
        "vetinari.telemetry",
        get_telemetry_collector=MagicMock(return_value=mock_telemetry),
    )

    mock_cost = MagicMock()
    mock_cost.get_summary.return_value = {
        "total_cost_usd": 1.23,
        "by_model": {"test": 1.0},
    }
    install_stub("vetinari.analytics")
    install_stub(
        "vetinari.analytics.cost",
        get_cost_tracker=MagicMock(return_value=mock_cost),
    )
    return {"telemetry": mock_telemetry, "cost_tracker": mock_cost}


def install_model_stubs():
    """Install model discovery and model pool stubs."""
    install_stub(
        "vetinari.model_discovery",
        ModelDiscovery=MagicMock(),
        ModelSearchEngine=MagicMock(),
        LiveModelSearchAdapter=MagicMock(),
    )
    install_stub("vetinari.model_pool", ModelPool=MagicMock())


def install_scheduler_stubs():
    """Install APScheduler and related stubs."""
    install_stub("apscheduler")
    install_stub("apscheduler.schedulers")
    install_stub("apscheduler.schedulers.background", BackgroundScheduler=MagicMock())
    install_stub("apscheduler.triggers")
    install_stub("apscheduler.triggers.interval", IntervalTrigger=MagicMock())
