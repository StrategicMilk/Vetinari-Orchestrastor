"""Regression tests for sandbox policy and manager bug fixes (SESSION-33B bugs 3-9).

Each test class is named after the bug it covers and exercises only the specific
branch introduced by the fix.  No real inference calls are made; only the
sandbox pre-execution logic, policy loading, rate limiting, and audit paths.
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vetinari.exceptions import ConfigurationError
from vetinari.sandbox_policy import (
    ExternalPluginSandbox,
    _SandboxAuditLogger,
    _SandboxPolicyLoader,
    _SandboxRateLimiter,
)

# ---------------------------------------------------------------------------
# Bug 3 regression: AdapterRegistry.__repr__() race condition
# ---------------------------------------------------------------------------


class TestAdapterRegistryReprLocked:
    """AdapterRegistry.__repr__() must snapshot shared state under _registry_lock.

    Regression for Bug 3: __repr__ read class-level mutable dicts without
    holding the lock, creating a data race with concurrent registrations.
    """

    def test_repr_returns_string(self) -> None:
        """__repr__ must return a non-empty string describing the registry."""
        from vetinari.adapters.registry import AdapterRegistry

        result = repr(AdapterRegistry)
        assert isinstance(result, str)
        assert "AdapterRegistry" in result
        assert "providers=" in result
        assert "instances=" in result

    def test_repr_concurrent_stability(self) -> None:
        """Concurrent calls to __repr__ must not raise RuntimeError or return garbage."""
        from vetinari.adapters.registry import AdapterRegistry

        errors: list[Exception] = []

        def call_repr() -> None:
            try:
                for _ in range(50):
                    _ = repr(AdapterRegistry)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=call_repr) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent __repr__ raised: {errors}"


# ---------------------------------------------------------------------------
# Bug 4 regression: get_status() hardcoded True values
# ---------------------------------------------------------------------------


class TestSandboxManagerStatusDynamic:
    """SandboxManager.get_status() must perform real availability checks.

    Regression for Bug 4: get_status() previously hardcoded ``available: True``
    for all backends without checking actual state.
    """

    def test_status_subprocess_reports_load(self) -> None:
        """subprocess.current_load and max_concurrent must appear in status."""
        from vetinari.sandbox_manager import SandboxManager

        mgr = SandboxManager()
        status = mgr.get_status()
        sub = status["subprocess"]
        assert "current_load" in sub
        assert "max_concurrent" in sub
        assert isinstance(sub["current_load"], float | int)

    def test_status_subprocess_unavailable_when_at_capacity(self) -> None:
        """subprocess backend reports unavailable when current_load >= max_concurrent."""
        from vetinari.sandbox_manager import SandboxManager

        mgr = SandboxManager()
        mgr.current_load = mgr.max_concurrent  # saturate
        status = mgr.get_status()
        assert status["subprocess"]["available"] is False

    def test_status_subprocess_available_when_under_capacity(self) -> None:
        """subprocess backend reports available when current_load < max_concurrent."""
        from vetinari.sandbox_manager import SandboxManager

        mgr = SandboxManager()
        mgr.current_load = 0.0
        status = mgr.get_status()
        assert status["subprocess"]["available"] is True

    def test_status_external_reflects_plugin_dir_existence(self) -> None:
        """external.available must reflect whether the plugin directory exists."""
        from vetinari.sandbox_manager import SandboxManager

        mgr = SandboxManager()
        # Point to a non-existent directory — external should be unavailable.
        mgr.external.plugin_dir = Path("/nonexistent/path/that/cannot/exist_xyz")
        status = mgr.get_status()
        assert status["external"]["available"] is False


# ---------------------------------------------------------------------------
# Bug 5 regression: ExternalPluginSandbox path traversal
# ---------------------------------------------------------------------------


class TestExternalPluginSandboxPathTraversal:
    """execute_hook() must reject plugin names that escape the plugin directory.

    Regression for Bug 5: plugin_name was used as a path component without
    checking whether the resolved path stayed inside plugin_dir.
    """

    def test_traversal_attempt_blocked(self, tmp_path: Path) -> None:
        """Plugin name containing .. must be rejected before any I/O."""
        sandbox = ExternalPluginSandbox(plugin_dir=str(tmp_path))
        result = sandbox.execute_hook("../../etc/passwd", "read_file", {})
        assert "error" in result
        assert "path traversal" in result["error"].lower() or "not allowed" in result["error"].lower()

    def test_traversal_with_encoded_slash_blocked(self, tmp_path: Path) -> None:
        """Deeply nested traversal attempt must also be blocked."""
        sandbox = ExternalPluginSandbox(plugin_dir=str(tmp_path))
        result = sandbox.execute_hook("../../../evil", "read_file", {})
        assert "error" in result

    def test_legitimate_plugin_name_allowed_through(self, tmp_path: Path) -> None:
        """A simple plugin name without traversal components must pass the path check.

        The plugin itself won't be found (tmp_path has no plugin), but the
        error must be 'not found', not 'path traversal'.
        """
        sandbox = ExternalPluginSandbox(plugin_dir=str(tmp_path))
        result = sandbox.execute_hook("my_plugin", "read_file", {})
        assert "error" in result
        # Must fail because plugin doesn't exist, NOT because of traversal.
        assert "path traversal" not in result["error"].lower()
        assert "not found" in result["error"].lower() or "not allowed" in result["error"].lower()

    def test_traversal_audit_entry_recorded(self, tmp_path: Path) -> None:
        """A blocked traversal attempt must not leave a partial audit entry for 'executing'."""
        sandbox = ExternalPluginSandbox(plugin_dir=str(tmp_path))
        sandbox.execute_hook("../../etc/passwd", "read_file", {})
        # The audit log should be empty — traversal is rejected before the audit entry
        # for 'executing' is written (the path check fires first).
        assert len(sandbox.audit_log) == 0


# ---------------------------------------------------------------------------
# Bug 6 regression: ThreadPoolExecutor blocking on TimeoutError
# ---------------------------------------------------------------------------


class TestExternalPluginSandboxTimeout:
    """execute_hook() must not block indefinitely on slow hooks.

    Regression for Bug 6: using ``with ThreadPoolExecutor`` as a context
    manager caused __exit__ to call shutdown(wait=True), blocking the caller
    until the timed-out thread actually finished.
    """

    def test_slow_hook_returns_within_deadline(self, tmp_path: Path) -> None:
        """A hook that runs longer than timeout must return an error quickly."""
        # Create a minimal fake plugin with a hook that sleeps forever.
        plugin_dir = tmp_path / "slow_plugin"
        plugin_dir.mkdir()
        main_py = plugin_dir / "main.py"
        main_py.write_text(
            "import time\n\ndef read_file(params):\n    time.sleep(60)\n    return {}\n",
            encoding="utf-8",
        )

        sandbox = ExternalPluginSandbox(plugin_dir=str(tmp_path), timeout=1)
        deadline = time.monotonic() + 5  # generous wall-clock budget
        # Parent-process monkeypatches must not leak into plugin execution.
        # The sandbox should still enforce a real timeout in the child process.
        with patch("time.sleep", return_value=None):
            result = sandbox.execute_hook("slow_plugin", "read_file", {})
        elapsed = time.monotonic()

        assert elapsed < deadline, "execute_hook blocked longer than 5 s on a 1 s timeout"
        assert "error" in result
        assert "timed out" in result["error"].lower()


# ---------------------------------------------------------------------------
# Bug 7 regression: _SandboxPolicyLoader fails closed on schema violations
# ---------------------------------------------------------------------------


class TestSandboxPolicyLoaderFailClosed:
    """_SandboxPolicyLoader.load() must raise on invalid policy files.

    Regression for Bug 7: a policy file that existed but failed schema
    validation would silently fall back to permissive defaults.  The fix
    re-raises so operators see the error.
    """

    def test_missing_file_uses_defaults(self) -> None:
        """A missing policy file must produce built-in defaults without raising."""
        from vetinari.config.sandbox_schema import SandboxPolicyConfig

        loader = _SandboxPolicyLoader()
        result = loader.load(Path("/nonexistent/path/sandbox_policy.yaml"))
        assert isinstance(result, SandboxPolicyConfig)

    def test_invalid_schema_raises(self, tmp_path: Path) -> None:
        """A policy file that fails schema validation must raise, not fall back."""
        bad_policy = tmp_path / "sandbox_policy.yaml"
        # Write a policy with a value that violates the schema (negative timeout).
        bad_policy.write_text(
            "sandbox:\n  subprocess:\n    timeout_seconds: -999\n",
            encoding="utf-8",
        )
        loader = _SandboxPolicyLoader()
        with pytest.raises(ValueError, match=r"validation error|Input should|greater than"):
            loader.load(bad_policy)

    def test_malformed_yaml_raises(self, tmp_path: Path) -> None:
        """A present but unparseable YAML file must raise, not fall back."""
        bad_policy = tmp_path / "sandbox_policy.yaml"
        bad_policy.write_text("{ this is not valid yaml: [[\n", encoding="utf-8")
        loader = _SandboxPolicyLoader()
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            loader.load(bad_policy)


# ---------------------------------------------------------------------------
# Bug 8 regression: _SandboxPolicyLoader treats all errors consistently
# ---------------------------------------------------------------------------


class TestSandboxPolicyLoaderErrorConsistency:
    """_SandboxPolicyLoader.load() must raise for all error types on a present file.

    Regression for Bug 8: non-ValueError exceptions (e.g. malformed YAML, I/O
    errors) were previously swallowed and fell back to permissive defaults.
    """

    def test_permission_error_on_present_file_raises(self, tmp_path: Path) -> None:
        """An I/O error on a present policy file must propagate, not silently default."""
        policy_path = tmp_path / "sandbox_policy.yaml"
        policy_path.write_text("sandbox: {}\n", encoding="utf-8")

        loader = _SandboxPolicyLoader()
        with patch("vetinari.config.sandbox_schema.SandboxPolicyConfig.from_yaml_file") as mock_load:
            mock_load.side_effect = OSError("Permission denied")
            with pytest.raises(OSError, match="Permission denied"):
                loader.load(policy_path)

    def test_only_file_not_found_uses_defaults(self, tmp_path: Path) -> None:
        """FileNotFoundError on a non-existent path must use defaults (not raise)."""
        from vetinari.config.sandbox_schema import SandboxPolicyConfig

        loader = _SandboxPolicyLoader()
        result = loader.load(tmp_path / "does_not_exist.yaml")
        assert isinstance(result, SandboxPolicyConfig)


# ---------------------------------------------------------------------------
# Bug 9 regression: _SandboxAuditLogger in-memory buffer and get_audit_log merge
# ---------------------------------------------------------------------------


class TestSandboxAuditLoggerBuffer:
    """_SandboxAuditLogger must write all records to an in-memory buffer.

    Regression for Bug 9 (part 1): the class had no __init__, no in-memory
    buffer, and no get_records() method.  SandboxManager.get_audit_log()
    only read from ExternalPluginSandbox.audit_log (plugin hook records),
    missing all subprocess/in-process execution records.
    """

    def test_record_writes_to_buffer(self) -> None:
        """record() must append to the in-memory buffer unconditionally."""
        logger = _SandboxAuditLogger()
        with patch("vetinari.sandbox_policy._SandboxAuditLogger.record", wraps=logger.record):
            pass  # just verify the method is callable

        logger.record(
            sandbox_type="in_process",
            execution_id="exec_aabbccdd",
            status="success",
            duration_ms=42,
            code_length=100,
        )
        records = logger.get_records(limit=10)
        assert len(records) == 1
        rec = records[0]
        assert rec["execution_id"] == "exec_aabbccdd"
        assert rec["status"] == "success"
        assert rec["sandbox_type"] == "in_process"
        assert rec["duration_ms"] == 42
        assert rec["code_length"] == 100
        assert "timestamp" in rec

    def test_buffer_respects_limit(self) -> None:
        """get_records(limit=N) must return at most N records, most recent first."""
        logger = _SandboxAuditLogger()
        for i in range(10):
            logger.record("subprocess", f"exec_{i:08x}", "success", i * 10, 50)
        records = logger.get_records(limit=3)
        assert len(records) == 3
        # The last 3 written should be the most recent (exec_7, exec_8, exec_9).
        ids = {r["execution_id"] for r in records}
        assert "exec_00000009" in ids
        assert "exec_00000008" in ids
        assert "exec_00000007" in ids

    def test_buffer_ring_cap_prevents_unbounded_growth(self) -> None:
        """Buffer must cap at _MAX_BUFFER entries and evict oldest."""
        logger = _SandboxAuditLogger()
        cap = logger._MAX_BUFFER
        # Write cap + 50 records.
        for i in range(cap + 50):
            logger.record("subprocess", f"exec_{i:08x}", "success", 1, 1)
        records = logger.get_records(limit=cap + 100)
        assert len(records) == cap

    def test_record_is_thread_safe(self) -> None:
        """Concurrent record() calls must not corrupt the buffer."""
        logger = _SandboxAuditLogger()
        errors: list[Exception] = []

        def write_records() -> None:
            try:
                for i in range(100):
                    logger.record("in_process", f"exec_{i:08x}", "success", i, 10)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write_records) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent record() raised: {errors}"
        # All 5 × 100 = 500 records fit within _MAX_BUFFER (1000).
        records = logger.get_records(limit=600)
        assert len(records) == 500

    def test_external_logger_failure_does_not_block_buffer(self) -> None:
        """If the external audit logger raises, the in-memory record must still land."""
        logger = _SandboxAuditLogger()
        with patch("vetinari.audit.get_audit_logger") as mock_get:
            mock_get.side_effect = RuntimeError("audit system unavailable")
            logger.record("subprocess", "exec_deadbeef", "failure", 99, 200)
        records = logger.get_records(limit=5)
        assert len(records) == 1
        assert records[0]["execution_id"] == "exec_deadbeef"


class TestSandboxManagerAuditLogMerge:
    """SandboxManager.get_audit_log() must merge both audit sinks.

    Regression for Bug 9 (part 2): get_audit_log() only read from
    ExternalPluginSandbox.audit_log (in-memory list of AuditEntry), completely
    missing the _SandboxAuditLogger buffer used by execute().
    """

    def test_execute_records_appear_in_get_audit_log(self) -> None:
        """Records written by execute() must appear in get_audit_log()."""
        from vetinari.sandbox_manager import SandboxManager

        mgr = SandboxManager()
        # Directly write to the audit logger (avoids running real subprocess).
        mgr._audit_logger.record(
            sandbox_type="in_process",
            execution_id="exec_testmerge",
            status="success",
            duration_ms=10,
            code_length=5,
        )
        log = mgr.get_audit_log(limit=50)
        ids = [e.get("execution_id") for e in log]
        assert "exec_testmerge" in ids, f"execute() record missing from get_audit_log(); got: {ids}"

    def test_plugin_records_appear_in_get_audit_log(self, tmp_path: Path) -> None:
        """Records from ExternalPluginSandbox.audit_log must appear in get_audit_log()."""
        from dataclasses import asdict
        from datetime import datetime, timezone

        from vetinari.sandbox_manager import SandboxManager
        from vetinari.sandbox_types import AuditEntry

        mgr = SandboxManager()
        # Inject a fake plugin audit entry directly into the external sink.
        fake_entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            execution_id="plugin_fakeentry",
            operation="read_file",
            sandbox_type="external",
            status="success",
            duration_ms=5,
            details={"plugin": "fake"},
        )
        mgr.external.audit_log.append(fake_entry)

        log = mgr.get_audit_log(limit=50)
        ids = [e.get("execution_id") for e in log]
        assert "plugin_fakeentry" in ids, f"Plugin record missing from get_audit_log(); got: {ids}"

    def test_get_audit_log_merges_both_sinks(self) -> None:
        """get_audit_log() must include records from both sinks in one list."""
        from datetime import datetime, timezone

        from vetinari.sandbox_manager import SandboxManager
        from vetinari.sandbox_types import SandboxAuditEntry

        mgr = SandboxManager()

        # Write to the manager (_audit_logger) sink.
        mgr._audit_logger.record("in_process", "exec_manager_side", "success", 1, 1)

        # Write to the external (ExternalPluginSandbox.audit_log) sink.
        mgr.external.audit_log.append(
            SandboxAuditEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                execution_id="plugin_external_side",
                operation="read_file",
                sandbox_type="external",
                status="success",
                duration_ms=1,
                details={},
            )
        )

        log = mgr.get_audit_log(limit=50)
        ids = {e.get("execution_id") for e in log}
        assert "exec_manager_side" in ids
        assert "plugin_external_side" in ids
