"""Tests for canary test suite — corpus loading, similarity scoring, alerts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from vetinari.testing.canary_tests import (
    DEFAULT_SIMILARITY_THRESHOLD,
    CanaryPair,
    CanaryReport,
    CanaryTestSuite,
)


def _make_corpus_yaml(pairs: list[dict]) -> dict:
    """Build a minimal canary corpus dict."""
    return {"canary_pairs": pairs}


def _write_corpus(path: Path, pairs: list[dict]) -> None:
    path.write_text(yaml.dump(_make_corpus_yaml(pairs)), encoding="utf-8")


_SAMPLE_PAIRS = [
    {
        "id": "c001",
        "category": "math",
        "prompt": "What is 2+2?",
        "expected_output": "4",
    },
    {
        "id": "c002",
        "category": "code",
        "prompt": "Write a hello world function.",
        "expected_output": "def hello(): print('Hello, world!')",
    },
]


# -- test_corpus_loading ------------------------------------------------------


def test_corpus_loading(tmp_path: Path) -> None:
    """load_corpus() returns CanaryPair objects matching the YAML file."""
    corpus_file = tmp_path / "canary_corpus.yaml"
    _write_corpus(corpus_file, _SAMPLE_PAIRS)

    suite = CanaryTestSuite(corpus_path=corpus_file, results_dir=tmp_path / "results")
    pairs = suite.load_corpus()

    assert len(pairs) == 2
    assert all(isinstance(p, CanaryPair) for p in pairs)
    assert pairs[0].id == "c001"
    assert pairs[1].category == "code"
    assert pairs[0].similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD


# -- test_similarity_scoring_identical ----------------------------------------


def test_similarity_scoring_identical(tmp_path: Path) -> None:
    """Identical strings score 1.0."""
    suite = CanaryTestSuite(results_dir=tmp_path / "results")
    score = suite.compute_similarity("hello world", "hello world")
    assert score == pytest.approx(1.0)


# -- test_similarity_scoring_different ----------------------------------------


def test_similarity_scoring_different(tmp_path: Path) -> None:
    """Completely different strings score well below 0.5."""
    suite = CanaryTestSuite(results_dir=tmp_path / "results")
    score = suite.compute_similarity("The quick brown fox", "ZZZZZ QQQQQ XXXXX PPPPP")
    assert score < 0.5


# -- test_alert_threshold_triggered -------------------------------------------


def test_alert_threshold_triggered(tmp_path: Path) -> None:
    """When inference returns low-similarity output, overall_pass is False."""
    corpus_file = tmp_path / "corpus.yaml"
    _write_corpus(corpus_file, _SAMPLE_PAIRS)
    results_dir = tmp_path / "results"

    suite = CanaryTestSuite(corpus_path=corpus_file, results_dir=results_dir)
    suite.load_corpus()

    # Return something completely unrelated so similarity is below threshold
    report = suite.run(inference_fn=lambda prompt: "TOTALLY UNRELATED GARBAGE XYZ")

    assert report.overall_pass is False
    assert report.failed > 0
    failed_results = [r for r in report.scores if not r.passed]
    assert len(failed_results) > 0


# -- test_alert_threshold_not_triggered ---------------------------------------


def test_alert_threshold_not_triggered(tmp_path: Path) -> None:
    """When inference returns exact expected output, overall_pass is True."""
    corpus_file = tmp_path / "corpus.yaml"
    _write_corpus(corpus_file, _SAMPLE_PAIRS)
    results_dir = tmp_path / "results"

    # Map prompts to expected outputs
    expected_map = {p["prompt"]: p["expected_output"] for p in _SAMPLE_PAIRS}
    suite = CanaryTestSuite(corpus_path=corpus_file, results_dir=results_dir)
    suite.load_corpus()

    report = suite.run(inference_fn=lambda prompt: expected_map.get(prompt, ""))

    assert report.overall_pass is True
    assert report.failed == 0
    assert report.passed == len(_SAMPLE_PAIRS)


# -- test_result_persistence --------------------------------------------------


def test_result_persistence(tmp_path: Path) -> None:
    """run() writes a timestamped JSON file with correct structure."""
    corpus_file = tmp_path / "corpus.yaml"
    _write_corpus(corpus_file, _SAMPLE_PAIRS[:1])
    results_dir = tmp_path / "results"

    suite = CanaryTestSuite(corpus_path=corpus_file, results_dir=results_dir)
    suite.load_corpus()
    suite.run(inference_fn=lambda prompt: "4")

    result_files = list(results_dir.glob("canary_*.json"))
    assert len(result_files) == 1

    data = json.loads(result_files[0].read_text(encoding="utf-8"))
    assert "pairs_tested" in data
    assert "passed" in data
    assert "failed" in data
    assert "overall_pass" in data
    assert "scores" in data
    assert data["pairs_tested"] == 1


# -- test_canary_pair_dataclass -----------------------------------------------


def test_canary_pair_dataclass() -> None:
    """CanaryPair fields are correct and default threshold is applied."""
    pair = CanaryPair(
        id="x001",
        category="general",
        prompt="Prompt text",
        expected_output="Expected",
    )
    assert pair.id == "x001"
    assert pair.category == "general"
    assert pair.prompt == "Prompt text"
    assert pair.expected_output == "Expected"
    assert pair.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD

    # Custom threshold is respected
    custom = CanaryPair(
        id="x002",
        category="strict",
        prompt="P",
        expected_output="E",
        similarity_threshold=0.95,
    )
    assert custom.similarity_threshold == pytest.approx(0.95)
