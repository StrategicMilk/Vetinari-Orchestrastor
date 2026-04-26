"""Tests for persistent audit logging system (US-043)."""

from __future__ import annotations

import json
import tempfile
import threading

import pytest

from vetinari.audit import AuditEntry, AuditLogger, get_audit_logger, reset_audit_logger
from vetinari.types import AgentType


class TestAuditEntry:
    """Tests for AuditEntry dataclass."""

    def test_default_timestamp_populated(self) -> None:
        """AuditEntry timestamp is auto-populated on creation."""
        entry = AuditEntry(event_type="test")
        assert entry.timestamp  # should be auto-populated

    def test_all_fields_serializable(self) -> None:
        """AuditEntry fields can be serialised to and from JSON."""
        entry = AuditEntry(
            event_type="permission_check",
            agent_type=AgentType.WORKER.value,
            task_id="task-001",
            action="file_write",
            resource="/src/main.py",
            outcome="allowed",
            details={"reason": "in jurisdiction"},
            trace_id="trace-abc",
        )
        data = json.loads(json.dumps(entry.__dict__, default=str))
        assert data["event_type"] == "permission_check"
        assert data["outcome"] == "allowed"


class TestAuditLogger:
    """Tests for AuditLogger file persistence."""

    def test_writes_entries_to_file(self) -> None:
        """A logged event is persisted and readable back from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log = AuditLogger(audit_dir=tmpdir)
            log.log_event(
                event_type="test_event",
                action="test_action",
                outcome="allowed",
            )
            entries = log.read_entries()
            assert len(entries) == 1
            assert entries[0]["event_type"] == "test_event"

    def test_multiple_entries_persisted(self) -> None:
        """Multiple logged events all appear in read_entries()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log = AuditLogger(audit_dir=tmpdir)
            for i in range(5):
                log.log_event(
                    event_type=f"event_{i}",
                    action="action",
                    outcome="allowed",
                )
            entries = log.read_entries()
            assert len(entries) == 5

    def test_thread_safe_concurrent_writes(self) -> None:
        """Concurrent writes from multiple threads do not corrupt the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit = AuditLogger(audit_dir=tmpdir)
            errors: list[Exception] = []

            def write_entries(thread_id: int) -> None:
                try:
                    for i in range(20):
                        audit.log_event(
                            event_type=f"thread_{thread_id}",
                            action=f"action_{i}",
                            outcome="allowed",
                        )
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=write_entries, args=(t,)) for t in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors, f"Concurrent write errors: {errors}"
            entries = audit.read_entries(limit=200)
            assert len(entries) == 100  # 5 threads * 20 entries

    def test_jsonl_format_valid(self) -> None:
        """Each line of the audit file is valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit = AuditLogger(audit_dir=tmpdir)
            audit.log_event(event_type="test", action="act", outcome="ok")
            with open(audit.file_path, encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    assert isinstance(data, dict)

    def test_read_entries_limit(self) -> None:
        """read_entries(limit=N) returns the most recent N entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit = AuditLogger(audit_dir=tmpdir)
            for i in range(10):
                audit.log_event(event_type=f"e{i}", action="a", outcome="ok")
            entries = audit.read_entries(limit=3)
            assert len(entries) == 3
            # Should be the last 3
            assert entries[0]["event_type"] == "e7"

    def test_read_entries_empty_file(self) -> None:
        """read_entries() returns an empty list when no entries exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit = AuditLogger(audit_dir=tmpdir)
            entries = audit.read_entries()
            assert entries == []

    def test_file_path_property(self) -> None:
        """file_path property reflects the configured directory and filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit = AuditLogger(audit_dir=tmpdir, audit_file="custom.jsonl")
            assert audit.file_path.name == "custom.jsonl"
            assert str(tmpdir) in str(audit.file_path)


class TestAuditLoggerConvenienceMethods:
    """Tests for specialised logging helpers."""

    def test_log_permission_check_denied(self) -> None:
        """Denied permission check is recorded with correct event_type and outcome."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit = AuditLogger(audit_dir=tmpdir)
            audit.log_permission_check(
                agent_type=AgentType.FOREMAN.value,
                permission="file_write",
                outcome="denied",
            )
            entries = audit.read_entries()
            assert entries[0]["event_type"] == "permission_check"
            assert entries[0]["outcome"] == "denied"

    def test_log_permission_check_allowed(self) -> None:
        """Allowed permission check is recorded with outcome=allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit = AuditLogger(audit_dir=tmpdir)
            audit.log_permission_check(
                agent_type=AgentType.WORKER.value,
                permission="file_read",
                outcome="allowed",
            )
            entries = audit.read_entries()
            assert entries[0]["outcome"] == "allowed"
            assert entries[0]["agent_type"] == AgentType.WORKER.value

    def test_log_guardrail_violation(self) -> None:
        """Denied guardrail check uses event_type=guardrail_violation and records violations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit = AuditLogger(audit_dir=tmpdir)
            audit.log_guardrail_check(
                check_type="input",
                outcome="denied",
                violations=["jailbreak attempt detected"],
            )
            entries = audit.read_entries()
            assert entries[0]["event_type"] == "guardrail_violation"
            assert "jailbreak" in entries[0]["details"]["violations"][0]

    def test_log_guardrail_check_allowed(self) -> None:
        """Allowed guardrail check uses event_type=guardrail_check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit = AuditLogger(audit_dir=tmpdir)
            audit.log_guardrail_check(
                check_type="output",
                outcome="allowed",
            )
            entries = audit.read_entries()
            assert entries[0]["event_type"] == "guardrail_check"
            assert entries[0]["outcome"] == "allowed"

    def test_log_sandbox_execution(self) -> None:
        """Sandbox execution log captures event_type, outcome, and timing details."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit = AuditLogger(audit_dir=tmpdir)
            audit.log_sandbox_execution(
                sandbox_type="in_process",
                execution_id="exec-001",
                status="success",
                duration_ms=42.5,
                code_length=150,
            )
            entries = audit.read_entries()
            assert entries[0]["event_type"] == "sandbox_execution"
            assert entries[0]["details"]["duration_ms"] == 42.5
            assert entries[0]["details"]["code_length"] == 150
            assert entries[0]["outcome"] == "success"


class TestAuditLoggerSingleton:
    """Tests for the global singleton helpers."""

    def test_get_audit_logger_returns_singleton(self) -> None:
        """Two consecutive calls to get_audit_logger() return the same object."""
        reset_audit_logger()
        a = get_audit_logger()
        b = get_audit_logger()
        assert a is b

    def test_reset_clears_singleton(self) -> None:
        """After reset_audit_logger(), get_audit_logger() returns a new instance."""
        a = get_audit_logger()
        reset_audit_logger()
        b = get_audit_logger()
        assert a is not b

    def teardown_method(self, _method) -> None:
        """Reset singleton after each test to avoid state leakage."""
        reset_audit_logger()
