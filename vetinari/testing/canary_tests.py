"""Canary test suite — fixed prompt/expected-output pairs for model regression detection.

Loads canonical pairs from config/testing/canary_corpus.yaml and runs them
against a model, measuring output similarity. Alerts when similarity drops
below a configurable threshold.

Pipeline role: Quality Gate — runs weekly to detect model behavior drift.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml

from vetinari.constants import _PROJECT_ROOT

logger = logging.getLogger(__name__)

# -- Configuration --
_CORPUS_PATH = _PROJECT_ROOT / "config" / "testing" / "canary_corpus.yaml"
_DEFAULT_RESULTS_DIR = Path(".vetinari") / "testing" / "canary_results"
DEFAULT_SIMILARITY_THRESHOLD = 0.7  # Below this triggers an alert


@dataclass(frozen=True, slots=True)
class CanaryPair:
    """A single canary test case: prompt → expected output."""

    id: str
    category: str
    prompt: str
    expected_output: str
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD

    def __repr__(self) -> str:
        return f"CanaryPair(id={self.id!r}, category={self.category!r})"


@dataclass(frozen=True, slots=True)
class PairResult:
    """Result from running one canary pair."""

    pair_id: str
    category: str
    similarity: float
    passed: bool
    actual_output: str

    def __repr__(self) -> str:
        return f"PairResult(pair_id={self.pair_id!r}, passed={self.passed!r}, similarity={self.similarity!r})"


@dataclass(slots=True)
class CanaryReport:
    """Aggregated results from a canary test run."""

    pairs_tested: int = 0
    passed: int = 0
    failed: int = 0
    scores: list[PairResult] = field(default_factory=list)
    overall_pass: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __repr__(self) -> str:
        return f"CanaryReport(tested={self.pairs_tested}, passed={self.passed}, failed={self.failed})"


class CanaryTestSuite:
    """Load and run canary test pairs for regression detection.

    Loads prompt/expected-output pairs from a YAML corpus file and
    evaluates model output similarity using SequenceMatcher. Results
    are stored as JSON for historical tracking.

    Args:
        corpus_path: Path to the canary corpus YAML. Defaults to config/testing/canary_corpus.yaml.
        results_dir: Directory for result JSON files. Defaults to .vetinari/testing/canary_results/.
        default_threshold: Global similarity threshold if pair doesn't specify one.
    """

    def __init__(
        self,
        corpus_path: Path | None = None,
        results_dir: Path | None = None,
        default_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self._corpus_path = corpus_path or _CORPUS_PATH
        self._results_dir = results_dir or _DEFAULT_RESULTS_DIR
        self._default_threshold = default_threshold
        self._pairs: list[CanaryPair] = []

    def load_corpus(self) -> list[CanaryPair]:
        """Load canary pairs from the YAML corpus file.

        Returns:
            List of CanaryPair objects loaded from the corpus.

        Raises:
            FileNotFoundError: If the corpus file does not exist.
            ValueError: If the corpus file has invalid structure.
        """
        if not self._corpus_path.exists():
            raise FileNotFoundError(f"Canary corpus not found: {self._corpus_path}")

        with open(self._corpus_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        raw_pairs = data.get("canary_pairs")
        if not raw_pairs or not isinstance(raw_pairs, list):
            raise ValueError(f"Canary corpus must have a 'canary_pairs' list, got: {type(raw_pairs)}")

        self._pairs = [
            CanaryPair(
                id=p["id"],
                category=p["category"],
                prompt=p["prompt"],
                expected_output=p["expected_output"],
                similarity_threshold=p.get("similarity_threshold", self._default_threshold),
            )
            for p in raw_pairs
        ]
        logger.info("Loaded %d canary pairs from %s", len(self._pairs), self._corpus_path)
        return self._pairs

    def compute_similarity(self, expected: str, actual: str) -> float:
        """Compute text similarity between expected and actual output.

        Uses SequenceMatcher ratio for a 0.0-1.0 similarity score.

        Args:
            expected: The expected output string.
            actual: The actual model output string.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        if not expected and not actual:
            return 1.0
        if not expected or not actual:
            return 0.0
        return SequenceMatcher(None, expected.lower(), actual.lower()).ratio()

    def run(
        self,
        inference_fn: Callable[[str], str],
    ) -> CanaryReport:
        """Execute all canary pairs and produce a report.

        Args:
            inference_fn: Callable that takes a prompt string and returns model output.

        Returns:
            CanaryReport with per-pair results and overall pass/fail.
        """
        if not self._pairs:
            self.load_corpus()

        report = CanaryReport()
        for pair in self._pairs:
            try:
                actual = inference_fn(pair.prompt)
            except Exception:
                logger.warning(
                    "Inference failed for canary %s — marking as failed",
                    pair.id,
                )
                actual = ""

            similarity = self.compute_similarity(pair.expected_output, actual)
            passed = similarity >= pair.similarity_threshold
            result = PairResult(
                pair_id=pair.id,
                category=pair.category,
                similarity=similarity,
                passed=passed,
                actual_output=actual,
            )
            report.scores.append(result)
            report.pairs_tested += 1
            if passed:
                report.passed += 1
            else:
                report.failed += 1
                report.overall_pass = False
                logger.warning(
                    "Canary %s FAILED: similarity %.3f < threshold %.3f",
                    pair.id,
                    similarity,
                    pair.similarity_threshold,
                )

        self._store_results(report)
        return report

    def _store_results(self, report: CanaryReport) -> None:
        """Persist canary results as a timestamped JSON file.

        Args:
            report: The CanaryReport to store.
        """
        self._results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = self._results_dir / f"canary_{ts}.json"

        data: dict[str, Any] = {
            "timestamp": report.timestamp,
            "pairs_tested": report.pairs_tested,
            "passed": report.passed,
            "failed": report.failed,
            "overall_pass": report.overall_pass,
            "scores": [
                {
                    "pair_id": s.pair_id,
                    "category": s.category,
                    "similarity": round(s.similarity, 4),
                    "passed": s.passed,
                }
                for s in report.scores
            ],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Canary results written to %s", out_path)
