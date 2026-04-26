"""Round-trip tests for the replay recording and replay infrastructure.

Verifies that events recorded by ReplayRecorder can be loaded by
ReplayReplayer and produce bit-identical responses, and that the
exhaustion and completeness checks work correctly.
"""

from __future__ import annotations

import pytest

from tests.replay.recorder import ReplayRecorder
from tests.replay.replayer import ReplayExhaustedError, ReplayReplayer

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def recorder(tmp_path):
    """A ReplayRecorder writing to a temporary directory."""
    return ReplayRecorder(output_dir=tmp_path)


@pytest.fixture
def replayer():
    """A fresh ReplayReplayer with no scenario loaded."""
    return ReplayReplayer()


# ── Happy path ────────────────────────────────────────────────────────────────


class TestReplayRoundtrip:
    """End-to-end record → save → load → replay tests."""

    def test_llm_calls_survive_roundtrip(self, recorder, replayer, tmp_path):
        """LLM responses recorded and then replayed are bit-identical to originals."""
        recorder.start_recording("llm_basic")
        recorder.record_llm_call(
            prompt="What is the capital of France?",
            response="Paris",
            model_id="test-model-7b",
            latency_ms=42,
        )
        recorder.record_llm_call(
            prompt="What is 2 + 2?",
            response="4",
            model_id="test-model-7b",
            latency_ms=10,
        )
        scenario_path = recorder.stop_recording()

        replayer.start_replay(scenario_path)
        first = replayer.get_next_llm_response("What is the capital of France?")
        second = replayer.get_next_llm_response("What is 2 + 2?")

        assert first == "Paris", f"Expected 'Paris', got {first!r}"
        assert second == "4", f"Expected '4', got {second!r}"

    def test_tool_calls_survive_roundtrip(self, recorder, replayer, tmp_path):
        """Tool outputs recorded and then replayed are equal to originals."""
        recorder.start_recording("tool_basic")
        recorder.record_tool_call(
            tool_name="file_reader",
            input_data={"path": "/etc/hosts"},
            output_data="127.0.0.1 localhost",
        )
        recorder.record_tool_call(
            tool_name="calculator",
            input_data={"expr": "3 * 7"},
            output_data=21,
        )
        scenario_path = recorder.stop_recording()

        replayer.start_replay(scenario_path)
        first = replayer.get_next_tool_response("file_reader")
        second = replayer.get_next_tool_response("calculator")

        assert first == "127.0.0.1 localhost"
        assert second == 21

    def test_mixed_llm_and_tool_calls(self, recorder, replayer, tmp_path):
        """LLM and tool event streams are partitioned and served independently."""
        recorder.start_recording("mixed_scenario")
        recorder.record_llm_call(
            prompt="Summarise the file",
            response="A short summary",
            model_id="test-7b",
            latency_ms=200,
        )
        recorder.record_tool_call(
            tool_name="read_file",
            input_data={"path": "README.md"},
            output_data="# Vetinari",
        )
        recorder.record_llm_call(
            prompt="Translate to Spanish",
            response="Una breve resumen",
            model_id="test-7b",
            latency_ms=180,
        )
        scenario_path = recorder.stop_recording()

        replayer.start_replay(scenario_path)
        # LLM events served independently of tool events
        llm_1 = replayer.get_next_llm_response("Summarise the file")
        tool_1 = replayer.get_next_tool_response("read_file")
        llm_2 = replayer.get_next_llm_response("Translate to Spanish")

        assert llm_1 == "A short summary"
        assert tool_1 == "# Vetinari"
        assert llm_2 == "Una breve resumen"

    def test_assert_all_replayed_passes_when_fully_consumed(self, recorder, replayer, tmp_path):
        """assert_all_replayed() raises nothing when all events were consumed."""
        recorder.start_recording("full_consume")
        recorder.record_llm_call(prompt="hi", response="hello", model_id="m1")
        scenario_path = recorder.stop_recording()

        replayer.start_replay(scenario_path)
        replayer.get_next_llm_response("hi")
        # Should not raise — all events consumed
        result = replayer.assert_all_replayed()
        assert result is None

    def test_assert_all_replayed_fails_when_events_remain(self, recorder, replayer, tmp_path):
        """assert_all_replayed() raises AssertionError when events were not consumed."""
        recorder.start_recording("partial_consume")
        recorder.record_llm_call(prompt="ignored", response="not read", model_id="m1")
        recorder.record_llm_call(prompt="also ignored", response="also not read", model_id="m1")
        scenario_path = recorder.stop_recording()

        replayer.start_replay(scenario_path)
        # Consume only the first event
        replayer.get_next_llm_response("ignored")

        with pytest.raises(AssertionError) as exc_info:
            replayer.assert_all_replayed()

        error_message = str(exc_info.value)
        assert "LLM" in error_message
        assert "1" in error_message  # one remaining

    def test_replay_exhausted_error_raised_on_extra_llm_call(self, recorder, replayer, tmp_path):
        """ReplayExhaustedError is raised when more LLM calls are made than recorded."""
        recorder.start_recording("overflow_llm")
        recorder.record_llm_call(prompt="only one", response="answer", model_id="m1")
        scenario_path = recorder.stop_recording()

        replayer.start_replay(scenario_path)
        replayer.get_next_llm_response("only one")

        with pytest.raises(ReplayExhaustedError) as exc_info:
            replayer.get_next_llm_response("extra call not in recording")

        assert exc_info.value.event_type == "llm_call"
        assert exc_info.value.calls_made == 1

    def test_replay_exhausted_error_raised_on_extra_tool_call(self, recorder, replayer, tmp_path):
        """ReplayExhaustedError is raised when more tool calls are made than recorded."""
        recorder.start_recording("overflow_tool")
        recorder.record_tool_call(
            tool_name="once",
            input_data={},
            output_data="result",
        )
        scenario_path = recorder.stop_recording()

        replayer.start_replay(scenario_path)
        replayer.get_next_tool_response("once")

        with pytest.raises(ReplayExhaustedError) as exc_info:
            replayer.get_next_tool_response("extra_tool")

        assert exc_info.value.event_type == "tool_call"
        assert exc_info.value.calls_made == 1

    def test_scenario_file_is_valid_jsonl(self, recorder, tmp_path):
        """Every line in the written JSONL file is valid JSON with required fields."""
        import json

        recorder.start_recording("jsonl_check")
        recorder.record_llm_call(
            prompt="test prompt",
            response="test response",
            model_id="m1",
            latency_ms=5,
        )
        recorder.record_tool_call(
            tool_name="grep",
            input_data={"pattern": "foo"},
            output_data=["match1"],
        )
        path = recorder.stop_recording()

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2, f"Expected 2 JSONL lines, got {len(lines)}"

        for line in lines:
            event = json.loads(line)
            assert "seq" in event, "Event missing 'seq' field"
            assert "event_type" in event, "Event missing 'event_type' field"
            assert "timestamp" in event, "Event missing 'timestamp' field"

    def test_sequence_numbers_are_monotonically_increasing(self, recorder, tmp_path):
        """Recorded events carry monotonically increasing sequence numbers."""
        import json

        recorder.start_recording("seq_check")
        recorder.record_llm_call(prompt="a", response="1", model_id="m")
        recorder.record_tool_call(tool_name="t", input_data={}, output_data="ok")
        recorder.record_llm_call(prompt="b", response="2", model_id="m")
        path = recorder.stop_recording()

        events = [json.loads(line) for line in path.read_text(encoding="utf-8").strip().splitlines()]
        seqs = [e["seq"] for e in events]
        assert seqs == sorted(seqs), f"Sequence numbers not monotonic: {seqs}"
        assert len(set(seqs)) == len(seqs), f"Duplicate sequence numbers: {seqs}"


# ── Error path ────────────────────────────────────────────────────────────────


class TestRecorderErrors:
    """Tests for ReplayRecorder error conditions."""

    def test_double_start_raises(self, recorder):
        """Starting a second recording session without stopping raises RuntimeError."""
        recorder.start_recording("first")
        with pytest.raises(RuntimeError, match="already in progress"):
            recorder.start_recording("second")
        recorder.stop_recording()

    def test_record_without_start_raises(self, recorder):
        """Calling record_llm_call without start_recording raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No active recording session"):
            recorder.record_llm_call(prompt="x", response="y", model_id="m")

    def test_stop_without_start_raises(self, recorder):
        """Calling stop_recording without a prior start raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No active recording session"):
            recorder.stop_recording()


class TestReplayerErrors:
    """Tests for ReplayReplayer error conditions."""

    def test_get_llm_without_start_raises(self, replayer):
        """Calling get_next_llm_response before start_replay raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Call start_replay"):
            replayer.get_next_llm_response("hello")

    def test_get_tool_without_start_raises(self, replayer):
        """Calling get_next_tool_response before start_replay raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Call start_replay"):
            replayer.get_next_tool_response("my_tool")

    def test_start_replay_missing_file_raises(self, replayer, tmp_path):
        """start_replay raises FileNotFoundError for a non-existent path."""
        missing = tmp_path / "does_not_exist.jsonl"
        with pytest.raises(FileNotFoundError):
            replayer.start_replay(missing)

    def test_empty_scenario_passes_assert_all_replayed(self, recorder, replayer, tmp_path):
        """An empty recording satisfies assert_all_replayed with no calls made."""
        recorder.start_recording("empty")
        path = recorder.stop_recording()

        replayer.start_replay(path)
        # Should not raise — nothing to consume
        result = replayer.assert_all_replayed()
        assert result is None
