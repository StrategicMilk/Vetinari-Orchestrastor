"""Shared test fixtures for Vetinari test suite.

This file is automatically loaded by pytest before any test modules.
Fixtures defined here are available to all tests without explicit import.
"""

from __future__ import annotations

import contextlib
import logging
import socket
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any
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

# Pre-create the coverage data directory used by pytest-cov on Windows so the
# final coverage file rename does not fail on a missing or ACL-misaligned path.
(PROJECT_ROOT / ".vetinari" / "coverage").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Prevent APScheduler from starting real background threads during tests.
# vetinari.web_ui starts a BackgroundScheduler at import time; if the module
# is re-imported (due to sys.modules isolation below) a second scheduler
# starts and may cause threading deadlocks.  Mocking it here (before any
# test file imports web_ui) avoids the problem entirely.
# ---------------------------------------------------------------------------


class _NoOpScheduler:
    """Drop-in replacement for BackgroundScheduler that keeps job metadata."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self._jobs: list[dict[str, Any]] = []
        self.is_running = False

    def start(self) -> None:
        self.is_running = True

    def shutdown(self, **_kwargs: Any) -> None:
        self.is_running = False

    def add_job(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        job = {
            "args": args,
            "kwargs": kwargs,
            "id": kwargs.get("id"),
        }
        self._jobs.append(job)
        return job

    def get_jobs(self) -> list[dict[str, Any]]:
        return list(self._jobs)

    def remove_all_jobs(self) -> None:
        self._jobs.clear()


import types as _types

for _apmod in (
    "apscheduler",
    "apscheduler.schedulers",
    "apscheduler.schedulers.background",
    "apscheduler.triggers",
    "apscheduler.triggers.interval",
):
    _m = _types.ModuleType(_apmod)
    _m.__path__ = []
    _m.__package__ = _apmod
    sys.modules[_apmod] = _m  # force-override even if installed
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

# Block daemon threads that accumulate across test files (one per re-import
# of production code due to sys.modules isolation).  Over ~80 files this
# grows to 50+ leaked threads that eventually deadlock or exhaust resources.
# Note: eventbus-async is NOT blocked because it was refactored to use a
# single reusable worker thread (not one per publish). The _cleanup_singletons
# function shuts it down properly between test files.
_BLOCKED_THREAD_NAMES = frozenset({
    "auto-tuner",  # vetinari/learning/auto_tuner.py
    "TrainingDataWriter",  # vetinari/learning/training_data.py
})


def _safe_thread_start(self: _threading.Thread) -> None:
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
import os as _os
import tempfile as _tempfile

import _pytest.pathlib as _pytest_pathlib
import _pytest.tmpdir as _pytest_tmpdir

# Map  test-file path -> {module_name: module_object}  (only for files that
# install stubs / modify vetinari modules during import)
_MODULE_VETINARI_STATES: dict = {}


if sys.platform == "win32":
    # Repo-local temp roots are not reliably deletable in this Windows sandbox.
    # Keep both pytest's tmp_path factory and plain tempfile.* calls inside
    # LocalAppData\Temp so tests exercise a realistic, writable location.
    _WINDOWS_TEMP_PARENT = Path(_tempfile.gettempdir()).resolve() / "vetinari-pytest"
    _WINDOWS_TEMP_PARENT.mkdir(mode=0o755, parents=True, exist_ok=True)

    for _temp_env in ("PYTEST_DEBUG_TEMPROOT", "TMP", "TEMP", "TMPDIR"):
        _os.environ[_temp_env] = str(_WINDOWS_TEMP_PARENT)
    _tempfile.tempdir = str(_WINDOWS_TEMP_PARENT)

    _ORIG_TEMPFLE_MKDTEMP = _tempfile.mkdtemp

    def _windows_safe_mkdtemp(
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,
    ) -> str:
        """Create temp dirs with a Windows-safe ACL instead of Python's 0o700 default."""
        root = Path(dir or _tempfile.gettempdir()).resolve()
        prefix = prefix or "tmp"
        suffix = suffix or ""

        for _ in range(_tempfile.TMP_MAX):
            candidate = root / f"{prefix}{next(_tempfile._get_candidate_names())}{suffix}"
            try:
                candidate.mkdir(mode=0o755)
                return str(candidate)
            except FileExistsError:
                continue

        raise FileExistsError("No usable temporary directory name found")

    _tempfile.mkdtemp = _windows_safe_mkdtemp

    _ORIG_PYTEST_MAKE_NUMBERED_DIR = _pytest_pathlib.make_numbered_dir

    def _pytest_safe_mode(mode: int) -> int:
        """Avoid Python 3.12 Windows ACLs that make 0o700 temp dirs unreadable."""
        return 0o755 if mode == 0o700 else mode

    def _pytest_make_numbered_dir(root: Path, prefix: str, mode: int = 0o700) -> Path:
        """Create pytest temp dirs with a Windows-safe mode."""
        return _ORIG_PYTEST_MAKE_NUMBERED_DIR(root, prefix, _pytest_safe_mode(mode))

    def _pytest_getbasetemp(self: _pytest_tmpdir.TempPathFactory) -> Path:
        """Return a readable pytest temp root on Windows/Python 3.12."""
        if self._basetemp is not None:
            return self._basetemp

        if self._given_basetemp is not None:
            basetemp = self._given_basetemp
            if basetemp.exists():
                _pytest_tmpdir.rm_rf(basetemp)
            basetemp.mkdir(mode=_pytest_safe_mode(0o700))
            basetemp = basetemp.resolve()
        else:
            from_env = _os.environ.get("PYTEST_DEBUG_TEMPROOT")
            temproot = Path(from_env or _tempfile.gettempdir()).resolve()
            user = _pytest_tmpdir.get_user() or "unknown"
            rootdir = temproot.joinpath(f"pytest-of-{user}")
            try:
                rootdir.mkdir(mode=_pytest_safe_mode(0o700), exist_ok=True)
            except OSError:
                rootdir = temproot.joinpath("pytest-of-unknown")
                rootdir.mkdir(mode=_pytest_safe_mode(0o700), exist_ok=True)

            uid = _pytest_tmpdir.get_user_id()
            if uid is not None:
                rootdir_stat = rootdir.stat()
                if rootdir_stat.st_uid != uid:
                    raise OSError(
                        f"The temporary directory {rootdir} is not owned by the current user. Fix this and try again."
                    )
                if (rootdir_stat.st_mode & 0o077) != 0:
                    _os.chmod(rootdir, rootdir_stat.st_mode & ~0o077)

            keep = self._retention_count
            if self._retention_policy == "none":
                keep = 0
            basetemp = _pytest_tmpdir.make_numbered_dir_with_cleanup(
                prefix="pytest-",
                root=rootdir,
                keep=keep,
                lock_timeout=_pytest_tmpdir.LOCK_TIMEOUT,
                mode=_pytest_safe_mode(0o700),
            )

        self._basetemp = basetemp
        self._trace("new basetemp", basetemp)
        return basetemp

    def _pytest_mktemp(self: _pytest_tmpdir.TempPathFactory, basename: str, numbered: bool = True) -> Path:
        """Create pytest child temp dirs with a Windows-safe mode."""
        basename = self._ensure_relative_to_basetemp(basename)
        if not numbered:
            p = self.getbasetemp().joinpath(basename)
            p.mkdir(mode=_pytest_safe_mode(0o700))
        else:
            p = _pytest_tmpdir.make_numbered_dir(
                root=self.getbasetemp(),
                prefix=basename,
                mode=_pytest_safe_mode(0o700),
            )
            self._trace("mktemp", p)
        return p

    _pytest_pathlib.make_numbered_dir = _pytest_make_numbered_dir
    _pytest_tmpdir.make_numbered_dir = _pytest_make_numbered_dir
    _pytest_tmpdir.TempPathFactory.getbasetemp = _pytest_getbasetemp
    _pytest_tmpdir.TempPathFactory.mktemp = _pytest_mktemp

# The "clean" vetinari module state (captured once, before any test file
# modifies it).  Used as the baseline for files that don't install stubs.
_CLEAN_VETINARI_STATE: dict | None = None

# Track which file was last restored so we skip redundant swaps
_LAST_RESTORED_FILE: list = [None]  # mutable container for closure


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector: Any) -> Generator[None, None, None]:
    """Save/restore vetinari sys.modules around each test file import."""
    global _CLEAN_VETINARI_STATE
    from _pytest.python import Module

    if not isinstance(collector, Module):
        yield
        return

    # Snapshot vetinari modules BEFORE this test file is imported
    pre_import = {k: v for k, v in sys.modules.items() if k == "vetinari" or k.startswith("vetinari.")}

    # Capture the clean baseline once
    if _CLEAN_VETINARI_STATE is None:
        _CLEAN_VETINARI_STATE = dict(pre_import)

    yield  # ← collection (import) happens here

    # Snapshot vetinari modules AFTER import (includes this file's stubs)
    post_import = {k: v for k, v in sys.modules.items() if k == "vetinari" or k.startswith("vetinari.")}

    # Only store state if this file actually CHANGED vetinari modules.
    # Compare keys and object identities (not deep equality).
    changed = set(pre_import.keys()) != set(post_import.keys()) or any(
        pre_import.get(k) is not post_import[k] for k in post_import
    )
    if changed:
        _MODULE_VETINARI_STATES[str(collector.path)] = post_import

    # Restore pre-import state so the NEXT file sees clean modules
    for k in list(sys.modules):
        if k == "vetinari" or k.startswith("vetinari."):
            sys.modules.pop(k, None)
    sys.modules.update(pre_import)


def _cleanup_singletons() -> None:
    """Explicitly close resources held by module-level singletons.

    When sys.modules isolation discards vetinari modules, their singletons
    become orphaned — SQLite connections stay open, threads keep running.
    This function closes known resource-holding singletons BEFORE the
    module swap, preventing resource exhaustion over 80+ test files.
    """
    # SQLite connections in memory stores
    _mem_mod = sys.modules.get("vetinari.memory.unified")
    if _mem_mod:
        _store = getattr(_mem_mod, "_unified_store", None)
        if _store is not None:
            with contextlib.suppress(Exception):
                _store.close()
            _mem_mod._unified_store = None

    _mem_init = sys.modules.get("vetinari.memory")
    if _mem_init:
        _store = getattr(_mem_init, "_memory_store", None)
        if _store is not None:
            with contextlib.suppress(Exception):
                if hasattr(_store, "close"):
                    _store.close()
            _mem_init._memory_store = None

    # TrainingDataCollector background thread
    _td_mod = sys.modules.get("vetinari.learning.training_data")
    if _td_mod:
        _cls = getattr(_td_mod, "TrainingDataCollector", None)
        if _cls and getattr(_cls, "_instance", None) is not None:
            with contextlib.suppress(Exception):
                _cls._instance._shutdown.set()
            _cls._instance = None

    # TelemetryPersistence workers can outlive sys.modules swaps.  The module
    # keeps a process-global registry so test cleanup can stop older instances.
    import builtins as _builtins

    _telemetry_instances = getattr(_builtins, "_vetinari_telemetry_persistence_instances", None)
    if _telemetry_instances is not None:
        for _instance in list(_telemetry_instances):
            with contextlib.suppress(Exception):
                _instance.stop()

    # Audit logger file handle
    _audit_mod = sys.modules.get("vetinari.audit")
    if _audit_mod:
        _logger = getattr(_audit_mod, "_audit_logger", None)
        if _logger is not None:
            with contextlib.suppress(Exception):
                if hasattr(_logger, "close"):
                    _logger.close()
            _audit_mod._audit_logger = None

    # EventBus — clear subscriptions to prevent stale callbacks
    _events_mod = sys.modules.get("vetinari.events")
    if _events_mod:
        _bus = getattr(_events_mod, "_event_bus", None)
        if _bus is not None:
            with contextlib.suppress(Exception):
                _bus.clear()
            _events_mod._event_bus = None

    # Analytics singletons (hold state, not resources, but prevent cross-test pollution)
    for _mod_name, _attr in [
        ("vetinari.analytics", "_cost_tracker"),
        ("vetinari.analytics", "_sla_tracker"),
        ("vetinari.analytics", "_anomaly_detector"),
        ("vetinari.analytics", "_forecaster"),
        ("vetinari.drift.contract_registry", "_registry"),
        ("vetinari.drift.capability_auditor", "_auditor"),
        ("vetinari.drift.schema_validator", "_validator"),
        ("vetinari.safety.guardrails", "_guardrails_manager"),
    ]:
        _mod = sys.modules.get(_mod_name)
        if _mod and hasattr(_mod, _attr):
            setattr(_mod, _attr, None)

    # Module-level singleton reset helpers
    for _mod_name, _reset_fn in [
        ("vetinari.adapters.batch_processor", "reset_batch_processor"),
        ("vetinari.analytics.telemetry_persistence", "reset_telemetry_persistence"),
        ("vetinari.cascade_router", "reset_cascade_router"),
        ("vetinari.learning.model_selector", "reset_thompson_selector"),
        ("vetinari.mcp.worker_bridge", "reset_worker_mcp_bridge"),
        ("vetinari.safety.llm_guard_scanner", "reset_llm_guard_scanner"),
        ("vetinari.shutdown", "reset"),
        ("vetinari.web.preferences", "reset_preferences_manager"),
    ]:
        _mod = sys.modules.get(_mod_name)
        if _mod:
            _fn = getattr(_mod, _reset_fn, None)
            if _fn:
                with contextlib.suppress(Exception):
                    _fn()


@pytest.fixture(autouse=True)
def _isolate_vetinari_modules(request: pytest.FixtureRequest) -> Generator[None, None, None]:
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

    # Close resources held by singletons BEFORE discarding modules
    _cleanup_singletons()

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


# ---------------------------------------------------------------------------
# Database isolation — every test gets its own temp SQLite database (ADR-0072).
# Without this, tests that call get_connection() share the project's real
# .vetinari/vetinari.db, leading to stale-schema errors and cross-test leaks.
# ---------------------------------------------------------------------------

import os as _os
import tempfile as _tempfile


@pytest.fixture(autouse=True)
def _isolate_database(tmp_path: Path) -> Generator[None, None, None]:
    """Point the unified database at a per-test temp file."""
    db_path = str(tmp_path / "test_vetinari.db")
    old = _os.environ.get("VETINARI_DB_PATH")
    _os.environ["VETINARI_DB_PATH"] = db_path
    try:
        # Reset the cached connection and schema flag so the next
        # get_connection() call creates a fresh database at db_path.
        try:
            from vetinari.database import reset_for_testing

            reset_for_testing()
        except ImportError:
            pass  # database module not yet available in all test contexts
        yield
    finally:
        if old is None:
            _os.environ.pop("VETINARI_DB_PATH", None)
        else:
            _os.environ["VETINARI_DB_PATH"] = old
        try:
            from vetinari.database import reset_for_testing

            reset_for_testing()
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# ExecutionContext isolation — reset ContextVar between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_execution_context() -> Generator[None, None, None]:
    """Reset ExecutionContext singleton before each test.

    ExecutionContext maintains a global singleton and a ContextVar stack per thread/task.
    Without resetting the singleton, state from one test can leak into the next.

    Strategy: Clear the global _context_manager singleton before and after each test.
    Each fresh ContextManager instance will create its own stack in the ContextVar.
    """
    try:
        import sys

        # BEFORE each test: reset the global singleton so fresh instances get fresh stacks
        try:
            if "vetinari.execution_context" in sys.modules:
                mod = sys.modules["vetinari.execution_context"]
                if hasattr(mod, "_context_manager"):
                    mod._context_manager = None
        except (ImportError, AttributeError):
            pass

    except Exception:
        pass

    yield

    # AFTER the test: reset the singleton again
    try:
        import sys

        try:
            if "vetinari.execution_context" in sys.modules:
                mod = sys.modules["vetinari.execution_context"]
                if hasattr(mod, "_context_manager"):
                    mod._context_manager = None
        except (ImportError, AttributeError):
            pass

    except Exception:
        pass


# Suppress noisy loggers during tests
@pytest.fixture(autouse=True)
def _quiet_logging() -> Generator[None, None, None]:
    """Suppress INFO/DEBUG logging noise during tests."""
    logging.getLogger("vetinari").setLevel(logging.WARNING)
    yield
    logging.getLogger("vetinari").setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Mock adapters (avoid real LM Studio connections)
# ---------------------------------------------------------------------------


@pytest.fixture
def reset_singletons() -> Generator[None, None, None]:
    """Explicitly reset all known singletons before and after a test.

    Use this fixture in tests that create or modify singleton instances
    to prevent cross-test pollution. The autouse ``_isolate_vetinari_modules``
    fixture only resets when switching files — this fixture resets per-test.
    """
    _cleanup_singletons()
    yield
    _cleanup_singletons()


@pytest.fixture
def mock_lmstudio_response() -> Any:
    """Factory for mock LM Studio adapter responses.

    Returns a callable that creates response dicts with optional overrides.

    Example:
        response = mock_lmstudio_response(output="custom", tokens_used=100)
    """

    def _make(
        output: str = "Mock LLM response for testing",
        latency_ms: int = 100,
        tokens_used: int = 42,
        status: str = "ok",
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "output": output,
            "latency_ms": latency_ms,
            "tokens_used": tokens_used,
            "status": status,
            "error": error,
        }

    return _make


@pytest.fixture
def mock_adapter(mock_lmstudio_response: Any) -> Any:
    """Factory for mock LMStudioAdapter instances with canned responses.

    Returns a callable that creates configured MagicMock adapters.

    Example:
        adapter = mock_adapter(healthy=False, models=[{"id": "custom"}])
    """

    def _make(
        healthy: bool = True,
        models: list[dict[str, str]] | None = None,
        response_overrides: dict[str, Any] | None = None,
    ) -> MagicMock:
        adapter = MagicMock()
        adapter.chat.return_value = mock_lmstudio_response(**(response_overrides or {}))
        adapter.is_healthy.return_value = healthy
        adapter.list_loaded_models.return_value = models or [
            {"id": "test-model-7b", "object": "model"},
        ]
        return adapter

    return _make


@pytest.fixture
def mock_adapter_manager(mock_lmstudio_response: Any) -> Any:
    """Factory for mock AdapterManager instances for orchestration tests.

    Returns a callable that creates configured MagicMock managers.

    Example:
        manager = mock_adapter_manager(output="custom response")
    """

    def _make(
        output: str = "Mock LLM response for testing",
        latency_ms: int = 100,
        tokens_used: int = 42,
        healthy: bool = True,
    ) -> MagicMock:
        manager = MagicMock()
        resp = MagicMock()
        resp.output = output
        resp.status = "ok"
        resp.latency_ms = latency_ms
        resp.tokens_used = tokens_used
        resp.error = None
        manager.infer.return_value = resp
        manager.health_check.return_value = {"healthy": healthy}
        return manager

    return _make


# ---------------------------------------------------------------------------
# Temporary directories
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_memory_dir(tmp_path: Path) -> Path:
    """Temporary directory for memory storage during tests."""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    return mem_dir


@pytest.fixture
def tmp_checkpoint_dir(tmp_path: Path) -> Path:
    """Temporary directory for execution checkpoints during tests."""
    cp_dir = tmp_path / "checkpoints"
    cp_dir.mkdir()
    return cp_dir


@pytest.fixture
def tmp_config_dir(tmp_path: Path) -> Path:
    """Temporary directory for config files during tests."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    return cfg_dir


# ---------------------------------------------------------------------------
# Shared memory fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_shared_memory(tmp_memory_dir: Path) -> Generator[Any, None, None]:
    """Create an isolated UnifiedMemoryStore backed by a temp database."""
    from unittest.mock import patch

    from vetinari.memory.unified import UnifiedMemoryStore

    db_path = str(tmp_memory_dir / "shared.db")

    with patch("vetinari.memory.unified._embed_via_local_inference", return_value=None):
        store = UnifiedMemoryStore(db_path=db_path, max_entries=100, session_max=10)
        yield store
        store.close()


# ---------------------------------------------------------------------------
# Single-instance guard — kills orphaned pytest runs, then acquires mutex
# ---------------------------------------------------------------------------

# Arbitrary loopback port held as a process mutex for the duration of the run.
# The OS releases the socket automatically on exit (including crashes), so no
# stale lock files to clean up.
_PYTEST_MUTEX_PORT = 47236
_MUTEX_KILL_WAIT_S = 5  # seconds to wait after killing before giving up
_PYTEST_MUTEX_KILL_ENV = "VETINARI_KILL_ORPHAN_PYTEST"


def _find_pid_on_port(port: int) -> int | None:
    """Return the PID listening on the given loopback port, or None."""
    import shutil
    import subprocess  # local import — only needed for orphan detection

    netstat = shutil.which("netstat") or "netstat"
    with contextlib.suppress(Exception):
        result = subprocess.run(  # noqa: S603 - argv is controlled and shell interpolation is not used
            [netstat, "-ano"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if f"127.0.0.1:{port}" in line and "LISTENING" in line:
                parts = line.split()
                if parts:
                    return int(parts[-1])
    return None


def _get_process_command_line(pid: int) -> str | None:
    """Return a process command line for orphan verification, if available."""
    import shutil
    import subprocess

    with contextlib.suppress(Exception):
        if sys.platform == "win32":
            powershell = shutil.which("powershell") or "powershell"
            result = subprocess.run(  # noqa: S603 - argv is controlled and shell interpolation is not used
                [
                    powershell,
                    "-NoProfile",
                    "-Command",
                    f"(Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\").CommandLine",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
        else:
            ps = shutil.which("ps") or "ps"
            result = subprocess.run(  # noqa: S603 - argv is controlled and shell interpolation is not used
                [ps, "-p", str(pid), "-o", "args="],
                capture_output=True,
                text=True,
                timeout=5,
            )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


def _is_vetinari_pytest_process(pid: int) -> bool:
    """Return True only for a prior Vetinari pytest wrapper/session."""
    command_line = _get_process_command_line(pid)
    if not command_line:
        return False
    normalized = command_line.replace("\\", "/").lower()
    return (
        "vetinari" in normalized
        and ("pytest" in normalized or "scripts/dev/run_tests.py" in normalized)
        and ("python" in normalized or "pytest" in normalized)
    )


def _kill_orphan_pytest(port: int) -> None:
    """Kill a verified orphaned pytest process holding the mutex port.

    Uses netstat to find the PID and taskkill to terminate it.  After the
    kill we poll until the port is free (up to _MUTEX_KILL_WAIT_S seconds).
    """
    import shutil
    import subprocess
    import time

    pid = _find_pid_on_port(port)
    if pid is None:
        return

    if not _is_vetinari_pytest_process(pid):
        raise SystemExit(
            f"\n[vetinari] Refusing to kill PID={pid} on pytest mutex port {port} "
            "because it does not look like a Vetinari pytest process.\n"
        )

    print(f"\n[vetinari] Killing orphaned pytest process (PID={pid}) before starting new run.")  # noqa: T201 - test harness emits operator-facing cleanup status
    taskkill = shutil.which("taskkill") or "taskkill"
    with contextlib.suppress(Exception):
        subprocess.run([taskkill, "/F", "/PID", str(pid)], capture_output=True, timeout=5)  # noqa: S603 - argv is controlled and shell interpolation is not used

    # Poll until the port is free or we time out
    deadline = time.monotonic() + _MUTEX_KILL_WAIT_S
    while time.monotonic() < deadline:
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            probe.bind(("127.0.0.1", port))
            probe.close()
            return  # port is free
        except OSError:
            time.sleep(0.3)


def _pytest_mutex_busy_message(port: int) -> str:
    """Build a clear error when another process owns the pytest mutex."""
    pid = _find_pid_on_port(port)
    owner = "unknown"
    if pid is not None:
        owner = _get_process_command_line(pid) or f"PID={pid}"
    return (
        f"\n[vetinari] Could not acquire pytest mutex on 127.0.0.1:{port}.\n"
        f"Owner: {owner}\n"
        f"Stop that process, or set {_PYTEST_MUTEX_KILL_ENV}=1 to let the test "
        "bootstrap kill a verified Vetinari pytest process.\n"
    )


# ---------------------------------------------------------------------------
# Test markers
# ---------------------------------------------------------------------------


def pytest_configure(config: Any) -> None:
    """Acquire the single-instance pytest mutex and register markers."""
    # Allow CI / concurrent developer runs to skip the mutex via env var.
    # Set VETINARI_SKIP_MUTEX=1 to bypass the single-instance guard.
    if _os.environ.get("VETINARI_SKIP_MUTEX") == "1":
        pass  # skip mutex acquisition — caller is responsible for avoiding collisions
    else:
        # Bind the mutex socket. If the port is busy, fail clearly by default;
        # explicit opt-in is required before killing another process.
        try:
            _mutex = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            _mutex.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            _mutex.bind(("127.0.0.1", _PYTEST_MUTEX_PORT))
        except OSError:
            if _os.environ.get(_PYTEST_MUTEX_KILL_ENV) == "1":
                _kill_orphan_pytest(_PYTEST_MUTEX_PORT)
            else:
                raise SystemExit(_pytest_mutex_busy_message(_PYTEST_MUTEX_PORT)) from None
            try:
                _mutex = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                _mutex.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
                _mutex.bind(("127.0.0.1", _PYTEST_MUTEX_PORT))
            except OSError:
                raise SystemExit(_pytest_mutex_busy_message(_PYTEST_MUTEX_PORT)) from None

        _mutex.listen(1)
        config._pytest_mutex = _mutex  # keep alive for session duration

    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: requires LM Studio running")
    config.addinivalue_line("markers", "security: security-related tests")
    config.addinivalue_line("markers", "eval: LLM evaluation quality tests")


def pytest_unconfigure(config: Any) -> None:
    """Release the single-instance run mutex."""
    mutex = getattr(config, "_pytest_mutex", None)
    if mutex is not None:
        with contextlib.suppress(Exception):
            mutex.close()


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    """Clean up resources after tests complete.

    Explicitly close singletons one final time so daemon threads and
    SQLite connections don't prevent clean shutdown.
    """
    _cleanup_singletons()
    _gc.collect()


# ---------------------------------------------------------------------------
# Block real network calls (prevent tests from hitting LM Studio)
# ---------------------------------------------------------------------------

_ORIG_CONNECT = socket.socket.connect


@pytest.fixture(autouse=True)
def _block_real_http(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Prevent any test from making real network connections.

    Tests that genuinely need network (marked ``@pytest.mark.integration``)
    are exempted.  Everything else gets a clear error if it accidentally
    tries to connect to LM Studio or any other host.
    """
    if "integration" in {m.name for m in request.node.iter_markers()}:
        yield
        return

    def _guarded_connect(self: socket.socket, address: Any) -> None:
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


@pytest.fixture(autouse=True)
def _mock_embedding_probe(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Prevent embedding endpoint probes from hanging during unit tests.

    ``_embed_via_local_inference`` is called by ``UnifiedMemoryStore.embeddings_available()``
    and by ``SharedMemory.embeddings_available()``.  In unit tests the local
    inference server is not running, so the probe would time out after
    ``EMBEDDING_API_TIMEOUT`` seconds.  We patch it to return ``None`` (no
    endpoint available) so tests run at full speed.

    Integration tests (``@pytest.mark.integration``) are exempted so they can
    test real embedding behaviour when the server is up.
    """
    if "integration" in {m.name for m in request.node.iter_markers()}:
        yield
        return

    # Only patch if the real module (with _embed_via_local_inference defined) is
    # loaded.  Some test modules (e.g. test_plan_api.py, test_two_layer.py) stub
    # out vetinari.memory.unified without that function; we skip patching for
    # those so as not to break their setup.
    import importlib
    import sys as _sys

    try:
        _mod = _sys.modules.get("vetinari.memory.unified") or importlib.import_module("vetinari.memory.unified")
        if not hasattr(_mod, "_embed_via_local_inference"):
            yield
            return
    except Exception:
        yield
        return

    # Patch the function object directly on the module rather than using a
    # dotted string target.  Python 3.10's mock._importer resolves dotted
    # targets via getattr traversal (vetinari -> memory -> unified -> func),
    # which fails when sys.modules isolation has removed intermediate packages.
    with patch.object(_mod, "_embed_via_local_inference", return_value=None):
        yield


# ---------------------------------------------------------------------------
# Block real llama.cpp model loading (prevent tests from doing real inference)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _block_real_llama_loading(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Prevent tests from loading real GGUF models into llama.cpp.

    The ``llama_cpp.Llama`` constructor is wrapped so that if the
    ``model_path`` points to a file that actually exists on disk AND is
    larger than 1 MB (i.e. not a test stub), the call raises immediately
    instead of spending minutes loading layers into VRAM.

    Tests that mock ``llama_cpp`` (the standard pattern) are unaffected
    because the mock replaces the constructor entirely.  Integration
    tests marked ``@pytest.mark.integration`` are exempted.
    """
    if "integration" in {m.name for m in request.node.iter_markers()}:
        yield
        return

    try:
        import llama_cpp as _lc

        _original_init = _lc.Llama.__init__

        def _guarded_init(self: Any, *args: Any, **kwargs: Any) -> None:
            from pathlib import Path as _P

            model_path = kwargs.get("model_path") or (args[0] if args else None)
            if model_path is not None:
                _mp = _P(str(model_path))
                # Block only real model files (>1 MB); tiny test stubs pass through
                _ONE_MB = 1_048_576
                if _mp.exists() and _mp.stat().st_size > _ONE_MB:
                    raise RuntimeError(
                        f"[conftest] Real model loading blocked in tests: {model_path} "
                        f"({_mp.stat().st_size / 1_048_576:.1f} MB). "
                        "Mock llama_cpp or mark @pytest.mark.integration."
                    )
            return _original_init(self, *args, **kwargs)

        with patch.object(_lc.Llama, "__init__", _guarded_init):
            yield
    except ImportError:
        # llama_cpp not installed — nothing to guard
        yield


# ---------------------------------------------------------------------------
# Block sentence-transformers model downloads (prevent HuggingFace network I/O)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _block_huggingface_downloads(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent HuggingFace Hub from downloading models during tests.

    ``SentenceTransformerEmbedder.__init__`` calls ``_try_load()`` which
    invokes ``SentenceTransformer(model_name, device="cpu")``.  If the model
    isn't cached locally, this triggers a multi-hundred-MB download from
    HuggingFace Hub, causing test timeouts in CI and offline environments.

    Setting ``HF_HUB_OFFLINE=1`` forces the hub client to use only cached
    models.  If a model is already downloaded, it loads normally; if not,
    it fails fast instead of hanging on a network request.

    Integration tests marked ``@pytest.mark.integration`` are exempted.
    """
    if "integration" in {m.name for m in request.node.iter_markers()}:
        return

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


# ---------------------------------------------------------------------------
# Reset semantic cache singleton between tests (prevent cross-test leakage)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_semantic_cache_singleton() -> Generator[None, None, None]:
    """Clear the SemanticCache singleton before and after each test.

    The semantic cache stores inference results keyed by prompt text.  When
    the sentence-transformer embedder is available, semantically similar keys
    (e.g. ``"model-a:test"`` vs ``"model-b:test"``) can match via cosine
    similarity, causing one test's cached results to leak into another.

    Resetting the singleton ensures every test starts with an empty cache.
    """
    try:
        from vetinari.optimization.semantic_cache import reset_semantic_cache

        reset_semantic_cache()
        yield
        reset_semantic_cache()
    except ImportError:
        yield
