"""Tests for TrainingDataCollector structured trace storage (Gap 5.14)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from vetinari.learning.training_data import TrainingDataCollector


@pytest.fixture
def collector(tmp_path: Path) -> TrainingDataCollector:
    """Return a sync-mode collector writing to a temp directory."""
    return TrainingDataCollector(
        output_path=str(tmp_path / "training_data.jsonl"),
        sync=True,
    )


class TestStoreTrace:
    def test_creates_trace_directory(self, collector: TrainingDataCollector, tmp_path: Path) -> None:
        """store_trace creates a subdirectory under traces/."""
        trace_dir = collector.store_trace(
            task_id="task_abc",
            prompt="Write a function",
            output="def foo(): pass",
        )
        assert trace_dir.is_dir()
        assert trace_dir.name == "task_abc"

    def test_writes_prompt_and_output(self, collector: TrainingDataCollector) -> None:
        """store_trace writes prompt.txt and output.txt."""
        collector.store_trace(
            task_id="task_001",
            prompt="Hello",
            output="World",
        )
        traces_dir = collector._output_path.parent / "traces" / "task_001"
        assert (traces_dir / "prompt.txt").read_text(encoding="utf-8") == "Hello"
        assert (traces_dir / "output.txt").read_text(encoding="utf-8") == "World"

    def test_writes_inspector_verdict_json(self, collector: TrainingDataCollector) -> None:
        """store_trace serialises inspector_verdict to inspector_verdict.json."""
        verdict = {"passed": False, "issues": ["incomplete output"]}
        collector.store_trace(
            task_id="task_002",
            prompt="p",
            output="o",
            inspector_verdict=verdict,
        )
        verdict_path = collector._output_path.parent / "traces" / "task_002" / "inspector_verdict.json"
        assert verdict_path.exists()
        loaded = json.loads(verdict_path.read_text(encoding="utf-8"))
        assert loaded["passed"] is False
        assert loaded["issues"] == ["incomplete output"]

    def test_skips_verdict_file_when_none(self, collector: TrainingDataCollector) -> None:
        """store_trace does not create inspector_verdict.json when verdict is None."""
        collector.store_trace(task_id="task_003", prompt="p", output="o")
        verdict_path = collector._output_path.parent / "traces" / "task_003" / "inspector_verdict.json"
        assert not verdict_path.exists()

    def test_writes_errors_log(self, collector: TrainingDataCollector) -> None:
        """store_trace writes each error as a line in errors.log."""
        collector.store_trace(
            task_id="task_004",
            prompt="p",
            output="o",
            errors=["KeyError: 'inputs'", "TimeoutError"],
        )
        errors_path = collector._output_path.parent / "traces" / "task_004" / "errors.log"
        lines = errors_path.read_text(encoding="utf-8").splitlines()
        assert lines == ["KeyError: 'inputs'", "TimeoutError"]

    def test_truncates_long_prompt_and_output(self, collector: TrainingDataCollector) -> None:
        """store_trace truncates prompt and output to 10 000 characters each."""
        long_text = "x" * 20000
        collector.store_trace(task_id="task_trunc", prompt=long_text, output=long_text)
        traces_dir = collector._output_path.parent / "traces" / "task_trunc"
        assert len((traces_dir / "prompt.txt").read_text(encoding="utf-8")) == 10000
        assert len((traces_dir / "output.txt").read_text(encoding="utf-8")) == 10000

    def test_returns_path_to_trace_dir(self, collector: TrainingDataCollector) -> None:
        """store_trace returns the Path of the created trace directory."""
        result = collector.store_trace(task_id="task_ret", prompt="p", output="o")
        assert isinstance(result, Path)
        assert result.is_dir()


class TestLoadTrace:
    def test_returns_none_for_missing_task(self, collector: TrainingDataCollector) -> None:
        """load_trace returns None when no trace directory exists."""
        assert collector.load_trace("nonexistent") is None

    def test_round_trips_all_fields(self, collector: TrainingDataCollector) -> None:
        """load_trace returns a dict matching what store_trace wrote."""
        collector.store_trace(
            task_id="task_rt",
            prompt="prompt text",
            output="output text",
            inspector_verdict={"passed": True},
            errors=["err1"],
        )
        trace = collector.load_trace("task_rt")
        assert trace is not None
        assert trace["prompt"] == "prompt text"
        assert trace["output"] == "output text"
        assert trace["inspector_verdict"] == {"passed": True}
        assert trace["errors"] == ["err1"]

    def test_handles_missing_optional_files(self, collector: TrainingDataCollector) -> None:
        """load_trace succeeds when only prompt.txt and output.txt exist."""
        collector.store_trace(task_id="task_min", prompt="p", output="o")
        trace = collector.load_trace("task_min")
        assert trace is not None
        assert "inspector_verdict" not in trace
        assert "errors" not in trace

    def test_handles_corrupt_verdict_json(self, collector: TrainingDataCollector, tmp_path: Path) -> None:
        """load_trace falls back to empty dict when inspector_verdict.json is corrupt."""
        collector.store_trace(task_id="task_corrupt", prompt="p", output="o")
        verdict_path = collector._output_path.parent / "traces" / "task_corrupt" / "inspector_verdict.json"
        verdict_path.write_text("not-valid-json", encoding="utf-8")
        trace = collector.load_trace("task_corrupt")
        assert trace is not None
        assert trace.get("inspector_verdict") == {}


class TestGetRecentTraces:
    def test_returns_empty_when_no_traces_dir(self, collector: TrainingDataCollector) -> None:
        """get_recent_traces returns [] before any traces are stored."""
        result = collector.get_recent_traces()
        assert result == []

    def test_returns_traces_with_task_id(self, collector: TrainingDataCollector) -> None:
        """get_recent_traces includes task_id in each returned dict."""
        collector.store_trace(task_id="t1", prompt="p", output="o")
        collector.store_trace(task_id="t2", prompt="p", output="o")
        traces = collector.get_recent_traces(limit=10)
        task_ids = {t["task_id"] for t in traces}
        assert {"t1", "t2"}.issubset(task_ids)

    def test_respects_limit(self, collector: TrainingDataCollector) -> None:
        """get_recent_traces returns at most limit entries."""
        for i in range(5):
            collector.store_trace(task_id=f"task_{i}", prompt="p", output="o")
        result = collector.get_recent_traces(limit=2)
        assert len(result) <= 2

    def test_failed_only_filters_passing_verdicts(self, collector: TrainingDataCollector) -> None:
        """failed_only=True excludes traces whose inspector_verdict.passed is True."""
        collector.store_trace(
            task_id="pass_task",
            prompt="p",
            output="o",
            inspector_verdict={"passed": True},
        )
        collector.store_trace(
            task_id="fail_task",
            prompt="p",
            output="o",
            inspector_verdict={"passed": False},
        )
        failed = collector.get_recent_traces(limit=10, failed_only=True)
        task_ids = {t["task_id"] for t in failed}
        assert "fail_task" in task_ids
        assert "pass_task" not in task_ids
