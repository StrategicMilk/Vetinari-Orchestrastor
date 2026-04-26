"""Tests for tests/replay/hooks.py — the replay recording integration layer.

Verifies that:
- Recording is off by default (VETINARI_REPLAY_RECORD not set)
- record_inference() and record_tool_call() no-op when disabled
- start_session() / stop_session() lifecycle works when enabled
- Auto-session is created on first record call when enabled
- Multiple concurrent calls are thread-safe
- stop_session() returns the JSONL path and the file contains events
"""

from __future__ import annotations

import importlib
import sys
import threading
from pathlib import Path
from typing import Any

from tests.factories import make_mock_inference_request, make_mock_inference_response

# ---------------------------------------------------------------------------
# Helpers to patch the module with REPLAY_RECORDING_ENABLED=True
# ---------------------------------------------------------------------------


def _reload_hooks_with_flag(flag: str) -> Any:
    """Reload tests.replay.hooks with the env var patched to the given value.

    Returns the freshly reloaded module.
    """
    import os

    original = os.environ.get("VETINARI_REPLAY_RECORD")
    os.environ["VETINARI_REPLAY_RECORD"] = flag

    # Force a fresh import so the module-level flag is re-evaluated
    mod_name = "tests.replay.hooks"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    module = importlib.import_module(mod_name)

    # Restore env var
    if original is None:
        os.environ.pop("VETINARI_REPLAY_RECORD", None)
    else:
        os.environ["VETINARI_REPLAY_RECORD"] = original

    return module


# ---------------------------------------------------------------------------
# Tests: disabled by default
# ---------------------------------------------------------------------------


class TestHooksDisabledByDefault:
    """When VETINARI_REPLAY_RECORD is not set, all hooks must no-op."""

    def test_replay_recording_enabled_is_false_by_default(self):
        """REPLAY_RECORDING_ENABLED flag is False when env var is absent."""
        import os

        os.environ.pop("VETINARI_REPLAY_RECORD", None)
        mod = _reload_hooks_with_flag("0")
        assert mod.REPLAY_RECORDING_ENABLED is False

    def test_record_inference_noop_when_disabled(self):
        """record_inference() returns immediately when recording is disabled."""
        mod = _reload_hooks_with_flag("0")
        # Should not raise and should not start a session
        mod.record_inference(make_mock_inference_request(), make_mock_inference_response())
        assert not mod.is_recording()

    def test_record_tool_call_noop_when_disabled(self):
        """record_tool_call() returns immediately when recording is disabled."""
        mod = _reload_hooks_with_flag("0")
        mod.record_tool_call("my_tool", {"x": 1}, "result")
        assert not mod.is_recording()

    def test_start_session_noop_when_disabled(self, tmp_path):
        """start_session() is a no-op when recording is disabled."""
        mod = _reload_hooks_with_flag("0")
        mod.start_session("test_scenario", output_dir=tmp_path)
        assert not mod.is_recording()

    def test_stop_session_returns_none_when_disabled(self):
        """stop_session() returns None when recording is disabled."""
        mod = _reload_hooks_with_flag("0")
        result = mod.stop_session()
        assert result is None


# ---------------------------------------------------------------------------
# Tests: enabled via env var
# ---------------------------------------------------------------------------


class TestHooksEnabled:
    """When VETINARI_REPLAY_RECORD=1, sessions are created and events written."""

    def test_replay_recording_enabled_is_true_when_flag_set(self):
        """REPLAY_RECORDING_ENABLED is True when env var is '1'."""
        mod = _reload_hooks_with_flag("1")
        assert mod.REPLAY_RECORDING_ENABLED is True

    def test_start_stop_session_creates_jsonl_file(self, tmp_path):
        """start_session + stop_session writes a JSONL file to the output dir."""
        mod = _reload_hooks_with_flag("1")
        mod.start_session("my_scenario", output_dir=tmp_path)
        assert mod.is_recording()

        path = mod.stop_session()
        assert path is not None
        assert isinstance(path, Path)
        assert path.exists()
        assert path.suffix == ".jsonl"
        assert not mod.is_recording()

    def test_record_inference_writes_llm_event(self, tmp_path):
        """record_inference() writes an llm_call event to the active session."""
        import json

        mod = _reload_hooks_with_flag("1")
        mod.start_session("infer_test", output_dir=tmp_path)
        mod.record_inference(
            make_mock_inference_request("What is 2+2?", "test-model"),
            make_mock_inference_response("4", "test-model"),
            latency_ms=30,
        )
        path = mod.stop_session()

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1, f"Expected 1 line in JSONL, got {len(lines)}"

        event = json.loads(lines[0])
        assert event["event_type"] == "llm_call"
        assert event["prompt"] == "What is 2+2?"
        assert event["response"] == "4"
        assert event["model_id"] == "test-model"
        assert event["latency_ms"] == 30

    def test_record_tool_call_writes_tool_event(self, tmp_path):
        """record_tool_call() writes a tool_call event to the active session."""
        import json

        mod = _reload_hooks_with_flag("1")
        mod.start_session("tool_test", output_dir=tmp_path)
        mod.record_tool_call("file_reader", {"path": "/etc/hosts"}, "127.0.0.1 localhost")
        path = mod.stop_session()

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

        event = json.loads(lines[0])
        assert event["event_type"] == "tool_call"
        assert event["tool_name"] == "file_reader"
        assert event["output"] == "127.0.0.1 localhost"

    def test_auto_session_started_on_first_record_call(self, tmp_path, monkeypatch):
        """If no session is active, record_* auto-starts one in the recordings dir."""
        mod = _reload_hooks_with_flag("1")
        # Point the default recordings dir to tmp_path so we don't pollute the repo
        monkeypatch.setattr(mod, "_RECORDINGS_DIR", tmp_path)

        assert not mod.is_recording()
        mod.record_tool_call("greet", {}, "hello")
        assert mod.is_recording()

        # Clean up
        mod.stop_session()

    def test_double_start_raises_runtime_error(self, tmp_path):
        """start_session() raises RuntimeError if called while a session is active."""
        import pytest

        mod = _reload_hooks_with_flag("1")
        mod.start_session("first", output_dir=tmp_path)
        try:
            with pytest.raises(RuntimeError, match="already active"):
                mod.start_session("second", output_dir=tmp_path)
        finally:
            mod.stop_session()

    def test_stop_when_no_session_returns_none(self):
        """stop_session() returns None when no session is active."""
        mod = _reload_hooks_with_flag("1")
        # Ensure no session is active
        result = mod.stop_session()
        assert result is None

    def test_mixed_inference_and_tool_events_both_written(self, tmp_path):
        """Both llm_call and tool_call events are written to the same JSONL file."""
        import json

        mod = _reload_hooks_with_flag("1")
        mod.start_session("mixed", output_dir=tmp_path)
        mod.record_inference(make_mock_inference_request("hi"), make_mock_inference_response("hello"))
        mod.record_tool_call("calc", {"expr": "1+1"}, 2)
        mod.record_inference(make_mock_inference_request("bye"), make_mock_inference_response("goodbye"))
        path = mod.stop_session()

        events = [json.loads(line) for line in path.read_text(encoding="utf-8").strip().splitlines()]
        assert len(events) == 3
        types = [e["event_type"] for e in events]
        assert types == ["llm_call", "tool_call", "llm_call"]


# ---------------------------------------------------------------------------
# Tests: thread safety
# ---------------------------------------------------------------------------


class TestHooksThreadSafety:
    """Concurrent record calls must not corrupt the session."""

    def test_concurrent_record_calls_all_written(self, tmp_path):
        """50 concurrent record_tool_call() calls all appear in the JSONL file."""
        import json

        mod = _reload_hooks_with_flag("1")
        mod.start_session("concurrent_tools", output_dir=tmp_path)

        errors: list[Exception] = []

        def _record(i: int) -> None:
            try:
                mod.record_tool_call(f"tool_{i}", {"i": i}, f"result_{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_record, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        path = mod.stop_session()

        assert not errors, f"Errors during concurrent recording: {errors}"
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 50, f"Expected 50 lines, got {len(lines)}"

        # Verify all events are valid JSON
        for line in lines:
            event = json.loads(line)
            assert event["event_type"] == "tool_call"
