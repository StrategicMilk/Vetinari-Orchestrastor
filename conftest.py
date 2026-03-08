"""Shared test fixtures for Vetinari test suite.

This file is automatically loaded by pytest before any test modules.
Fixtures defined here are available to all tests without explicit import.
"""

import json
import logging
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# Pre-import commonly stubbed third-party modules so that test files using
# ``sys.modules.setdefault("yaml", stub)`` (like test_cli.py) cannot replace
# the real module once it's already loaded.
import yaml  # noqa: F401 — keep real PyYAML in sys.modules

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

# Ensure vetinari package is importable from project root
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Prevent APScheduler from starting real background threads during tests.
# vetinari.web_ui starts a BackgroundScheduler at import time; if the module
# is re-imported (due to sys.modules isolation below) a second scheduler
# starts and may cause threading deadlocks.  Mocking it here (before any
# test file imports web_ui) avoids the problem entirely.
# ---------------------------------------------------------------------------

class _NoOpScheduler:
    """Drop-in replacement for BackgroundScheduler that does nothing."""
    def start(self): pass
    def shutdown(self, **kw): pass
    def add_job(self, *a, **kw): pass
    def remove_all_jobs(self): pass

import types as _types
for _apmod in ("apscheduler", "apscheduler.schedulers",
               "apscheduler.schedulers.background",
               "apscheduler.triggers", "apscheduler.triggers.interval"):
    _m = _types.ModuleType(_apmod)
    _m.__path__ = []
    _m.__package__ = _apmod
    sys.modules[_apmod] = _m          # force-override even if installed
sys.modules["apscheduler.schedulers.background"].BackgroundScheduler = _NoOpScheduler


# ---------------------------------------------------------------------------
# Prevent daemon threads from accumulating during test runs.
# When sys.modules isolation resets singletons, re-importing production code
# spawns NEW background threads (auto-tuner, TrainingDataWriter) for each
# test file.  Over ~80 files this grows to 50+ leaked threads that eventually
# deadlock or timeout the suite.  We patch Thread.start() to silently skip
# these known daemon workers.
# ---------------------------------------------------------------------------

import threading as _threading

_ORIG_THREAD_START = _threading.Thread.start

# Only block auto-tuner threads.  They accumulate across test files (one per
# cli.py re-import) and deadlock on import locks due to sys.modules isolation.
# TrainingDataWriter threads are harmless daemons that drain a queue; blocking
# them causes flush()/queue.join() to deadlock.
_BLOCKED_THREAD_NAMES = frozenset({"auto-tuner"})


def _safe_thread_start(self):
    """Skip starting known daemon threads that leak across test files."""
    if getattr(self, "name", "") in _BLOCKED_THREAD_NAMES:
        return  # silently skip — these are production helpers, not needed in tests
    return _ORIG_THREAD_START(self)


_threading.Thread.start = _safe_thread_start


# ---------------------------------------------------------------------------
# sys.modules isolation between test modules
# ---------------------------------------------------------------------------
# Many test files install lightweight stubs into sys.modules at import time.
# Without isolation each file's stubs leak into files imported later, causing
# hundreds of cross-test failures.  The strategy:
#
#  1. ``pytest_make_collect_report`` hookwrapper saves/restores vetinari
#     entries in sys.modules around each test-file import so files cannot
#     pollute each other during *collection*.
#  2. Before each test *runs*, the fixture ``_isolate_vetinari_modules``
#     restores the exact vetinari-module state that existed right after that
#     file was imported — but ONLY when switching between files (not every
#     test) to avoid O(tests × modules) overhead and memory leaks.
#
# Memory-efficiency: we only cache state for files that actually CHANGED
# vetinari modules (installed stubs).  Files that don't modify modules
# are not tracked, saving memory.
# ---------------------------------------------------------------------------

import gc as _gc

# Map  test-file path -> {module_name: module_object}  (only for files that
# install stubs / modify vetinari modules during import)
_MODULE_VETINARI_STATES: dict = {}

# The "clean" vetinari module state (captured once, before any test file
# modifies it).  Used as the baseline for files that don't install stubs.
_CLEAN_VETINARI_STATE: dict | None = None

# Track which file was last restored so we skip redundant swaps
_LAST_RESTORED_FILE: list = [None]  # mutable container for closure


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    """Save/restore vetinari sys.modules around each test file import."""
    global _CLEAN_VETINARI_STATE
    from _pytest.python import Module

    if not isinstance(collector, Module):
        yield
        return

    # Snapshot vetinari modules BEFORE this test file is imported
    pre_import = {
        k: v for k, v in sys.modules.items()
        if k == "vetinari" or k.startswith("vetinari.")
    }

    # Capture the clean baseline once
    if _CLEAN_VETINARI_STATE is None:
        _CLEAN_VETINARI_STATE = dict(pre_import)

    yield  # ← collection (import) happens here

    # Snapshot vetinari modules AFTER import (includes this file's stubs)
    post_import = {
        k: v for k, v in sys.modules.items()
        if k == "vetinari" or k.startswith("vetinari.")
    }

    # Only store state if this file actually CHANGED vetinari modules.
    # Compare keys and object identities (not deep equality).
    changed = (set(pre_import.keys()) != set(post_import.keys()) or
               any(pre_import.get(k) is not post_import[k]
                   for k in post_import))
    if changed:
        _MODULE_VETINARI_STATES[str(collector.path)] = post_import

    # Restore pre-import state so the NEXT file sees clean modules
    for k in list(sys.modules):
        if k == "vetinari" or k.startswith("vetinari."):
            sys.modules.pop(k, None)
    sys.modules.update(pre_import)


@pytest.fixture(autouse=True)
def _isolate_vetinari_modules(request):
    """Restore this test file's vetinari modules before each test runs.

    Optimised: only performs the swap when switching to a different file,
    not on every single test.  This reduces 6000+ swaps to ~80.
    """
    fspath = str(request.fspath)

    # Skip swap if we're still in the same file as the last test
    if fspath == _LAST_RESTORED_FILE[0]:
        yield
        return

    _LAST_RESTORED_FILE[0] = fspath

    state = _MODULE_VETINARI_STATES.get(fspath)
    if state is not None:
        # This file installed stubs — restore its exact module state
        for k in list(sys.modules):
            if k == "vetinari" or k.startswith("vetinari."):
                sys.modules.pop(k, None)
        sys.modules.update(state)
        _gc.collect()
    elif _CLEAN_VETINARI_STATE is not None:
        # This file didn't install stubs — restore the clean baseline
        # (in case the previous file DID install stubs)
        for k in list(sys.modules):
            if k == "vetinari" or k.startswith("vetinari."):
                sys.modules.pop(k, None)
        sys.modules.update(_CLEAN_VETINARI_STATE)
        _gc.collect()
    yield


# Suppress noisy loggers during tests
@pytest.fixture(autouse=True)
def _quiet_logging():
    """Suppress INFO/DEBUG logging noise during tests."""
    logging.getLogger("vetinari").setLevel(logging.WARNING)
    yield
    logging.getLogger("vetinari").setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Mock adapters (avoid real LM Studio connections)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_lmstudio_response() -> Dict[str, Any]:
    """Standard mock response from LM Studio adapter."""
    return {
        "output": "Mock LLM response for testing",
        "latency_ms": 100,
        "tokens_used": 42,
        "status": "ok",
        "error": None,
    }


@pytest.fixture
def mock_adapter(mock_lmstudio_response):
    """A mock LMStudioAdapter that returns canned responses."""
    adapter = MagicMock()
    adapter.chat.return_value = mock_lmstudio_response
    adapter.is_healthy.return_value = True
    adapter.list_loaded_models.return_value = [
        {"id": "test-model-7b", "object": "model"},
    ]
    return adapter


@pytest.fixture
def mock_adapter_manager(mock_lmstudio_response):
    """A mock AdapterManager for testing orchestration without real inference."""
    manager = MagicMock()
    resp = MagicMock()
    resp.output = mock_lmstudio_response["output"]
    resp.status = "ok"
    resp.latency_ms = 100
    resp.tokens_used = 42
    resp.error = None
    manager.infer.return_value = resp
    manager.health_check.return_value = {"healthy": True}
    return manager


# ---------------------------------------------------------------------------
# Temporary directories
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_memory_dir(tmp_path) -> Path:
    """Temporary directory for memory storage during tests."""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    return mem_dir


@pytest.fixture
def tmp_checkpoint_dir(tmp_path) -> Path:
    """Temporary directory for execution checkpoints during tests."""
    cp_dir = tmp_path / "checkpoints"
    cp_dir.mkdir()
    return cp_dir


@pytest.fixture
def tmp_config_dir(tmp_path) -> Path:
    """Temporary directory for config files during tests."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    return cfg_dir


# ---------------------------------------------------------------------------
# Shared memory fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_shared_memory(tmp_memory_dir):
    """Create an isolated SharedMemory instance for testing."""
    from vetinari.shared_memory import SharedMemory

    # Reset singleton
    old_instance = SharedMemory._instance
    SharedMemory._instance = None

    mem = SharedMemory(storage_path=str(tmp_memory_dir))
    yield mem

    # Restore singleton
    SharedMemory._instance = old_instance


# ---------------------------------------------------------------------------
# Flask test client
# ---------------------------------------------------------------------------

@pytest.fixture
def flask_app():
    """Create a Flask test app with dashboard blueprints registered."""
    try:
        from vetinari.web import create_app
        app = create_app()
    except ImportError:
        from flask import Flask
        app = Flask(__name__)

    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test-secret-key"
    return app


@pytest.fixture
def flask_client(flask_app):
    """Flask test client for API endpoint testing."""
    return flask_app.test_client()


# ---------------------------------------------------------------------------
# Test markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: requires LM Studio running")
    config.addinivalue_line("markers", "security: security-related tests")


# ---------------------------------------------------------------------------
# Block real network calls (prevent tests from hitting LM Studio)
# ---------------------------------------------------------------------------

_ORIG_CONNECT = socket.socket.connect


@pytest.fixture(autouse=True)
def _block_real_http(request):
    """Prevent any test from making real network connections.

    Tests that genuinely need network (marked ``@pytest.mark.integration``)
    are exempted.  Everything else gets a clear error if it accidentally
    tries to connect to LM Studio or any other host.
    """
    if "integration" in {m.name for m in request.node.iter_markers()}:
        yield
        return

    def _guarded_connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        # Allow loopback connections (asyncio event loops on Windows use
        # internal 127.0.0.1 sockets; Flask test client also needs them)
        if isinstance(address, tuple) and host in ("127.0.0.1", "::1", "localhost"):
            return _ORIG_CONNECT(self, address)
        # Allow Unix-domain sockets
        if isinstance(address, str):
            return _ORIG_CONNECT(self, address)
        raise OSError(
            f"[conftest] Real network connection to {address!r} blocked during tests. "
            "Use unittest.mock to mock network calls."
        )

    with patch.object(socket.socket, "connect", _guarded_connect):
        yield
