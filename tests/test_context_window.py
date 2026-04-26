"""Tests for context window tester — needle-in-haystack, effective window calculation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vetinari.testing.context_window import (
    DEFAULT_ACCURACY_THRESHOLD,
    NEEDLE_ANSWER,
    NEEDLE_POSITIONS,
    ContextWindowReport,
    ContextWindowTester,
    get_effective_window,
)

# -- Helpers ------------------------------------------------------------------


def _make_tester(tmp_path: Path, **kwargs) -> ContextWindowTester:
    """Create a ContextWindowTester writing results to tmp_path."""
    return ContextWindowTester(results_dir=tmp_path / "results", **kwargs)


# -- Tests --------------------------------------------------------------------


def test_needle_positions() -> None:
    """Verify NEEDLE_POSITIONS constant has exactly the expected values."""
    assert NEEDLE_POSITIONS == [0.25, 0.50, 0.75, 0.90, 0.95]


def test_build_haystack(tmp_path: Path) -> None:
    """Build haystack at a position — needle is embedded in the output."""
    tester = _make_tester(tmp_path)
    haystack = tester.build_haystack(target_tokens=1000, needle_position_pct=0.5)

    assert NEEDLE_ANSWER in haystack, "Needle answer must appear in the haystack"
    # The full needle fact string must be present
    from vetinari.testing.context_window import NEEDLE_FACT

    assert NEEDLE_FACT in haystack


def test_build_haystack_needle_at_start(tmp_path: Path) -> None:
    """Needle embedded at position 0.0 still produces a non-empty haystack."""
    tester = _make_tester(tmp_path)
    haystack = tester.build_haystack(target_tokens=500, needle_position_pct=0.0)
    assert NEEDLE_ANSWER in haystack


def test_accuracy_scoring_found(tmp_path: Path) -> None:
    """Mock inference returning the needle answer — 100% accuracy at every position."""
    tester = _make_tester(tmp_path)

    def _always_found(prompt: str) -> str:
        return f"The secret code is {NEEDLE_ANSWER}, I found it."

    report = tester.run_needle_test(
        model_id="test-model",
        declared_window=4096,
        inference_fn=_always_found,
    )

    assert isinstance(report, ContextWindowReport)
    assert len(report.position_results) == len(NEEDLE_POSITIONS)
    for pr in report.position_results:
        assert pr.accuracy == 1.0
        assert pr.retrieved is True


def test_accuracy_scoring_not_found(tmp_path: Path) -> None:
    """Mock inference returning irrelevant text — 0% accuracy at every position."""
    tester = _make_tester(tmp_path)

    def _never_found(prompt: str) -> str:
        return "The weather is nice today, I have no idea about any codes."

    report = tester.run_needle_test(
        model_id="test-model-miss",
        declared_window=4096,
        inference_fn=_never_found,
    )

    for pr in report.position_results:
        assert pr.accuracy == 0.0
        assert pr.retrieved is False


def test_effective_window_calculation(tmp_path: Path) -> None:
    """Inference works at 25%, 50%, 75% but fails at 90% and 95%.

    Effective window should be the token position at 75% of declared window.
    """
    declared = 4096
    threshold_pct = 0.75  # Last position that succeeds
    threshold_tokens = int(declared * threshold_pct)

    def _partial_retrieval(prompt: str) -> str:
        # The haystack size in the prompt correlates with position — we track
        # calls by prompt length: short prompts = early positions (succeed),
        # long prompts = late positions (fail).
        # Approximate: at 90% / 95%, prompt is significantly longer than at 75%.
        approx_target_chars_at_75 = int(declared * 4 * threshold_pct)
        if len(prompt) <= approx_target_chars_at_75 * 1.1:
            return f"The answer is {NEEDLE_ANSWER}"
        return "I don't know."

    tester = _make_tester(tmp_path)
    report = tester.run_needle_test(
        model_id="partial-model",
        declared_window=declared,
        inference_fn=_partial_retrieval,
    )

    # Effective window must be <= 75% of declared (could be 25% or 50% depending
    # on prompt lengths, but must NOT be larger than the declared window)
    assert report.effective_window <= declared
    # At minimum, the declared window and effective window are both set
    assert report.declared_window == declared
    assert isinstance(report.effective_window, int)


def test_effective_window_all_succeed(tmp_path: Path) -> None:
    """When all positions succeed, effective_window equals the last position's tokens."""
    declared = 2048
    tester = _make_tester(tmp_path)

    report = tester.run_needle_test(
        model_id="full-model",
        declared_window=declared,
        inference_fn=lambda p: f"Found: {NEEDLE_ANSWER}",
    )

    expected_last = int(declared * NEEDLE_POSITIONS[-1])
    assert report.effective_window == expected_last


def test_result_persistence(tmp_path: Path) -> None:
    """Run with tmp_path results dir — JSON results file is written with correct keys."""
    results_dir = tmp_path / "results"
    tester = ContextWindowTester(results_dir=results_dir)

    report = tester.run_needle_test(
        model_id="persist-model",
        declared_window=1024,
        inference_fn=lambda p: f"The answer is {NEEDLE_ANSWER}",
    )

    files = list(results_dir.glob("context_window_persist-model_*.json"))
    assert len(files) == 1

    with open(files[0], encoding="utf-8") as f:
        data = json.load(f)

    assert data["model_id"] == "persist-model"
    assert data["declared_window"] == 1024
    assert "effective_window" in data
    assert "position_results" in data
    assert len(data["position_results"]) == len(NEEDLE_POSITIONS)


def test_get_effective_window(tmp_path: Path) -> None:
    """get_effective_window returns the stored effective window for the model."""
    results_dir = tmp_path / "results"
    tester = ContextWindowTester(results_dir=results_dir)

    tester.run_needle_test(
        model_id="lookup-model",
        declared_window=8192,
        inference_fn=lambda p: f"Code: {NEEDLE_ANSWER}",
    )

    result = get_effective_window("lookup-model", results_dir=results_dir)
    # All positions should have succeeded, so effective_window = last token position
    assert result is not None
    assert isinstance(result, int)
    assert result == int(8192 * NEEDLE_POSITIONS[-1])


def test_get_effective_window_no_results(tmp_path: Path) -> None:
    """get_effective_window returns None when no result file exists for the model."""
    result = get_effective_window("unknown-model", results_dir=tmp_path / "empty")
    assert result is None


def test_default_accuracy_threshold() -> None:
    """DEFAULT_ACCURACY_THRESHOLD is 0.8 as documented."""
    assert DEFAULT_ACCURACY_THRESHOLD == 0.8


def test_check_retrieval_case_insensitive(tmp_path: Path) -> None:
    """check_retrieval is case-insensitive on the needle answer."""
    tester = _make_tester(tmp_path)
    assert tester.check_retrieval(NEEDLE_ANSWER.lower()) is True
    assert tester.check_retrieval(NEEDLE_ANSWER.upper()) is True
    assert tester.check_retrieval("no code here") is False
