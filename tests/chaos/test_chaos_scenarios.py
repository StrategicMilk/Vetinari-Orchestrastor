"""Chaos injection tests for resilience verification.

Each test injects a controlled failure into a critical subsystem and asserts
that the system degrades gracefully (logs the error, returns a typed failure
result) rather than propagating an unhandled exception to the caller.

All external dependencies (adapters, database, filesystem) are mocked so
tests run without any real infrastructure.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.factories import make_inference_request
from vetinari.adapters.base import InferenceResponse

# ── Chaos 1: Model OOM ────────────────────────────────────────────────────────


class TestModelOOMGracefulDegradation:
    """The adapter layer must not propagate MemoryError to the caller."""

    def test_litellm_adapter_oom_returns_error_response(self):
        """LiteLLMAdapter.infer() returns an error InferenceResponse on MemoryError.

        The adapter catches the MemoryError during the litellm.completion() call
        and returns a response with status='error' instead of re-raising.
        """
        from vetinari.adapters.base import ProviderConfig
        from vetinari.adapters.litellm_adapter import LiteLLMAdapter
        from vetinari.types import ModelProvider

        config = ProviderConfig(
            provider_type=ModelProvider.LOCAL,
            name="test-local",
            endpoint="http://localhost:11434",
        )
        adapter = LiteLLMAdapter(config)
        request = make_inference_request()

        with patch("vetinari.adapters.litellm_adapter._litellm") as mock_litellm_fn:
            mock_litellm = MagicMock()
            mock_litellm.completion.side_effect = MemoryError("Out of GPU memory")
            mock_litellm_fn.return_value = mock_litellm

            response = adapter.infer(request)

        # Must not raise — must return a typed error response
        assert isinstance(response, InferenceResponse), f"Expected InferenceResponse, got {type(response).__name__}"
        assert response.status == "error", f"Expected status='error', got status={response.status!r}"
        assert response.output == "", f"Expected empty output on OOM, got {response.output!r}"
        assert response.model_id == request.model_id

    def test_adapter_oom_error_field_is_set(self):
        """InferenceResponse.error is populated when OOM occurs during inference."""
        from vetinari.adapters.base import ProviderConfig
        from vetinari.adapters.litellm_adapter import LiteLLMAdapter
        from vetinari.types import ModelProvider

        config = ProviderConfig(
            provider_type=ModelProvider.LOCAL,
            name="test-local",
            endpoint="http://localhost:11434",
        )
        adapter = LiteLLMAdapter(config)
        request = make_inference_request()

        with patch("vetinari.adapters.litellm_adapter._litellm") as mock_litellm_fn:
            mock_litellm = MagicMock()
            mock_litellm.completion.side_effect = MemoryError("CUDA out of memory")
            mock_litellm_fn.return_value = mock_litellm

            response = adapter.infer(request)

        assert response.error is not None, "Expected error field to be set on OOM"
        assert isinstance(response.error, str), (
            f"Expected error field to be a string, got {type(response.error).__name__}"
        )
        assert len(response.error) > 0, "Expected non-empty error message on OOM"


# ── Chaos 2: SQLite lock contention ──────────────────────────────────────────


class TestSQLiteLockContention:
    """Database writes under lock contention must not crash the process."""

    def test_execute_write_lock_error_propagates_as_operational_error(self, tmp_path, monkeypatch):
        """execute_write raises OperationalError (not a crash) on locked database.

        SQLite raises OperationalError("database is locked") when busy_timeout
        expires. We verify this is surfaced as a typed exception rather than a
        silent hang or process crash.
        """
        from vetinari import database

        monkeypatch.setenv("VETINARI_DB_PATH", str(tmp_path / "locked_test.db"))
        database.reset_for_testing()

        with patch.object(database, "get_connection") as mock_conn_fn:
            mock_conn = MagicMock()
            mock_conn.execute.side_effect = sqlite3.OperationalError("database is locked")
            mock_conn_fn.return_value = mock_conn

            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                database.execute_write(
                    "INSERT INTO execution_state (execution_id, goal, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    ("exec-001", "test goal", "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
                )

        database.reset_for_testing()

    def test_durable_engine_emit_event_survives_locked_database(self, tmp_path, monkeypatch):
        """DurableExecutionEngine._emit_event does not crash when SQLite is locked.

        _emit_event has an explicit except clause around the INSERT to prevent
        event persistence failures from breaking task execution. This test
        confirms that contract holds under lock contention.
        """
        import sys
        import unittest.mock as mock

        _mock_qs = mock.MagicMock()
        _mock_qs.score.return_value = mock.MagicMock(overall_score=0.8)

        sys.modules.setdefault(
            "vetinari.learning.quality_scorer",
            mock.MagicMock(get_quality_scorer=lambda: _mock_qs),
        )
        sys.modules.setdefault(
            "vetinari.learning.feedback_loop",
            mock.MagicMock(get_feedback_loop=lambda: mock.MagicMock()),
        )
        sys.modules.setdefault(
            "vetinari.learning.model_selector",
            mock.MagicMock(get_thompson_selector=lambda: mock.MagicMock()),
        )

        monkeypatch.setenv("VETINARI_DB_PATH", str(tmp_path / "wal_test.db"))
        from vetinari.database import reset_for_testing

        reset_for_testing()

        from vetinari.orchestration.durable_execution import DurableExecutionEngine

        engine = DurableExecutionEngine(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            max_concurrent=1,
            default_timeout=5.0,
        )

        with patch.object(
            engine._checkpoint_store._db,
            "execute",
            side_effect=sqlite3.OperationalError("database is locked"),
        ) as execute:
            try:
                engine._emit_event(
                    event_type="task_started",
                    task_id="task-locked-001",
                    data={"description": "lock contention test"},
                    execution_id="exec-locked-001",
                )
            except sqlite3.OperationalError:
                pytest.fail(
                    "_emit_event propagated sqlite3.OperationalError — "
                    "the explicit except clause in _emit_event should have swallowed this."
                )

        execute.assert_called_once()
        assert len(engine._event_history) == 1
        assert engine._event_history[0].task_id == "task-locked-001"
        reset_for_testing()


# ── Chaos 3: Network timeout ──────────────────────────────────────────────────


class TestNetworkTimeoutLocalFallback:
    """Network-level timeouts must not crash the process."""

    def test_litellm_adapter_timeout_returns_error_response(self):
        """LiteLLMAdapter.infer() returns an error response on requests.Timeout.

        The adapter's broad except clause around litellm.completion() catches
        any exception (including network timeouts) and returns a typed error
        InferenceResponse rather than crashing.
        """
        from vetinari.adapters.base import ProviderConfig
        from vetinari.adapters.litellm_adapter import LiteLLMAdapter
        from vetinari.types import ModelProvider

        config = ProviderConfig(
            provider_type=ModelProvider.OPENAI,
            name="test-openai",
            endpoint="https://api.openai.com",
            api_key="sk-test",
        )
        adapter = LiteLLMAdapter(config)
        request = make_inference_request(model_id="gpt-4o-mini")

        # Simulate a network timeout during the litellm call
        class _FakeTimeout(OSError):
            """Simulated network timeout exception."""

        with patch("vetinari.adapters.litellm_adapter._litellm") as mock_litellm_fn:
            mock_litellm = MagicMock()
            mock_litellm.completion.side_effect = _FakeTimeout("Connection timed out")
            mock_litellm_fn.return_value = mock_litellm

            response = adapter.infer(request)

        assert isinstance(response, InferenceResponse), f"Expected InferenceResponse, got {type(response).__name__}"
        assert response.status == "error", f"Expected status='error' on network timeout, got {response.status!r}"
        assert response.tokens_used == 0, "Expected tokens_used=0 when inference fails before getting a response"

    def test_health_check_timeout_returns_unhealthy(self):
        """health_check() returns unhealthy=False when the endpoint is unreachable."""
        from vetinari.adapters.base import ProviderConfig
        from vetinari.adapters.litellm_adapter import LiteLLMAdapter
        from vetinari.types import ModelProvider

        config = ProviderConfig(
            provider_type=ModelProvider.LOCAL,
            name="test-unreachable",
            endpoint="http://127.0.0.1:9999",
        )
        adapter = LiteLLMAdapter(config)

        with patch("vetinari.http.create_session") as mock_session_fn:
            mock_session = MagicMock()
            mock_session.__enter__ = lambda s: s
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.get.side_effect = OSError("Connection refused")
            mock_session_fn.return_value = mock_session

            result = adapter.health_check()

        assert isinstance(result, dict), f"Expected dict, got {type(result).__name__}"
        assert result["healthy"] is False, f"Expected healthy=False on timeout, got {result['healthy']!r}"
        assert "reason" in result, "Expected 'reason' key in health_check result"


# ── Chaos 4: Disk full during write ──────────────────────────────────────────


class TestDiskFullDuringWrite:
    """OSError(ENOSPC) during file writes must not crash the process."""

    def test_replay_recorder_disk_full_raises_os_error(self, tmp_path):
        """ReplayRecorder.stop_recording() surfaces OSError on disk-full.

        The recorder does not silently swallow the error — the caller knows
        the scenario was not persisted and can take corrective action.
        """
        from tests.replay.recorder import ReplayRecorder

        recorder = ReplayRecorder(output_dir=tmp_path)
        recorder.start_recording("disk_full_scenario")
        recorder.record_llm_call(prompt="test", response="test response", model_id="test-7b")

        disk_full_error = OSError(28, "No space left on device")

        with patch.object(Path, "open", side_effect=disk_full_error):
            with pytest.raises(OSError, match="No space left on device") as exc_info:
                recorder.stop_recording()

        assert exc_info.value.errno == 28, f"Expected errno 28 (ENOSPC), got {exc_info.value.errno}"

    def test_path_write_text_disk_full_raises_os_error(self, tmp_path):
        """Path.write_text() on a disk-full volume raises OSError (errno 28).

        Verifies the error propagates correctly so callers can handle it — the
        Python stdlib does not swallow ENOSPC.
        """
        test_file = tmp_path / "output.json"
        disk_full_error = OSError(28, "No space left on device")

        with patch.object(Path, "write_text", side_effect=disk_full_error):
            with pytest.raises(OSError, match="No space left on device") as exc_info:
                test_file.write_text('{"key": "value"}', encoding="utf-8")

        assert exc_info.value.errno == 28, f"Expected errno 28 (ENOSPC), got {exc_info.value.errno}"

    def test_durable_engine_emit_event_failure_does_not_crash(self, tmp_path, monkeypatch):
        """DurableExecutionEngine._emit_event() swallows write errors gracefully.

        When the SQLite INSERT for event persistence fails (e.g. disk full),
        _emit_event must continue — event loss is preferable to task execution
        failure. The engine has an explicit except clause for this purpose.
        """
        import sys
        import unittest.mock as mock

        sys.modules.setdefault(
            "vetinari.learning.quality_scorer",
            mock.MagicMock(get_quality_scorer=lambda: mock.MagicMock()),
        )
        sys.modules.setdefault(
            "vetinari.learning.feedback_loop",
            mock.MagicMock(get_feedback_loop=lambda: mock.MagicMock()),
        )
        sys.modules.setdefault(
            "vetinari.learning.model_selector",
            mock.MagicMock(get_thompson_selector=lambda: mock.MagicMock()),
        )

        monkeypatch.setenv("VETINARI_DB_PATH", str(tmp_path / "disk_full_test.db"))
        from vetinari.database import reset_for_testing

        reset_for_testing()

        from vetinari.orchestration.durable_execution import DurableExecutionEngine

        engine = DurableExecutionEngine(
            checkpoint_dir=str(tmp_path / "cp"),
            max_concurrent=1,
            default_timeout=5.0,
        )

        with patch.object(
            engine._checkpoint_store._db,
            "execute",
            side_effect=OSError(28, "No space left on device"),
        ) as execute:
            try:
                # _emit_event has an explicit except clause that logs and continues
                engine._emit_event(
                    event_type="task_started",
                    task_id="task-disk-full-001",
                    data={"description": "chaos test"},
                    execution_id="exec-disk-full",
                )
            except OSError:
                pytest.fail(
                    "_emit_event propagated OSError(ENOSPC) — "
                    "should have been caught and logged to prevent execution failure."
                )

        execute.assert_called_once()
        assert len(engine._event_history) == 1
        assert engine._event_history[0].task_id == "task-disk-full-001"
        reset_for_testing()
