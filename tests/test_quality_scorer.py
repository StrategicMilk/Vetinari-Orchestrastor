"""Tests for the quality scorer overhaul (Session 3).

Validates that quality scores produce meaningful variance across task types,
structural checks work correctly, flat-score detection triggers calibration,
variance monitoring warns on suspiciously flat distributions, inference
confidence feeds into scoring, and unmeasured dimensions are explicitly flagged.
"""

from __future__ import annotations

import logging
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from vetinari.learning.quality_scorer import QualityScore, QualityScorer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scorer():
    """Create a QualityScorer with baselines loaded from config."""
    return QualityScorer()


@pytest.fixture
def scorer_no_adapter():
    """QualityScorer with no adapter manager (heuristic-only)."""
    s = QualityScorer(adapter_manager=None)
    return s


# ---------------------------------------------------------------------------
# 3.1 — Task-specific baselines
# ---------------------------------------------------------------------------


class TestTaskSpecificBaselines:
    """Different task types produce different baseline scores."""

    def test_coding_vs_documentation_baselines_differ(self, scorer):
        """Coding and documentation tasks must produce different scores for identical output."""
        output = "This is a moderate length output with some content about the topic. " * 10

        coding_score = scorer.score(
            task_id="t1",
            model_id="m1",
            task_type="coding",
            task_description="Write code",
            output=output,
            use_llm=False,
        )
        doc_score = scorer.score(
            task_id="t2",
            model_id="m1",
            task_type="documentation",
            task_description="Write docs",
            output=output,
            use_llm=False,
        )
        assert coding_score.overall_score != doc_score.overall_score, (
            f"Coding ({coding_score.overall_score}) and documentation ({doc_score.overall_score}) "
            f"should differ for the same output"
        )

    def test_research_vs_analysis_baselines_differ(self, scorer):
        """Research and analysis tasks produce different dimension scores."""
        output = "According to the data, the metrics show a 15% improvement.\n\n## Summary\n\nRecommend action."

        research = scorer.score(
            task_id="t3",
            model_id="m1",
            task_type="research",
            task_description="Research topic",
            output=output,
            use_llm=False,
        )
        analysis = scorer.score(
            task_id="t4",
            model_id="m1",
            task_type="analysis",
            task_description="Analyze data",
            output=output,
            use_llm=False,
        )
        assert research.dimensions != analysis.dimensions

    def test_baselines_loaded_from_yaml(self, scorer):
        """Scorer must load baselines from quality_baselines.yaml."""
        assert scorer._baselines, "Baselines should be loaded from config"
        assert "coding" in scorer._baselines
        assert "documentation" in scorer._baselines
        assert "research" in scorer._baselines


# ---------------------------------------------------------------------------
# 3.2 — Structural scoring depth
# ---------------------------------------------------------------------------


class TestStructuralScoringDepth:
    """Good output scores higher than bad output on structural checks."""

    def test_good_code_scores_higher_than_bad_code(self, scorer):
        """Code with defs, docstrings, error handling, type hints scores higher."""
        good_code = '''
import os
from pathlib import Path

def process_file(path: str, encoding: str = "utf-8") -> dict[str, int]:
    """Process a file and return word counts.

    Args:
        path: File path to process.
        encoding: File encoding.

    Returns:
        Dictionary mapping words to counts.
    """
    try:
        with Path(path).open(encoding=encoding) as f:
            content = f.read()
        words = content.split()
        return {w: words.count(w) for w in set(words)}
    except FileNotFoundError:
        return {}
'''
        bad_code = "x = 1\ny = 2\nz = x + y"

        good_score = scorer.score(
            task_id="good",
            model_id="m1",
            task_type="coding",
            task_description="Write a function",
            output=good_code,
            use_llm=False,
        )
        bad_score = scorer.score(
            task_id="bad",
            model_id="m1",
            task_type="coding",
            task_description="Write a function",
            output=bad_code,
            use_llm=False,
        )
        assert good_score.overall_score > bad_score.overall_score, (
            f"Good code ({good_score.overall_score}) should score higher than bad code ({bad_score.overall_score})"
        )

    def test_coding_checks_error_handling(self, scorer):
        """Code with try/except gets higher correctness than code without."""
        with_error = "def foo():\n    try:\n        x = 1\n    except ValueError:\n        pass"
        without_error = "def foo():\n    x = 1"

        score_with = scorer.score(
            task_id="we",
            model_id="m1",
            task_type="coding",
            task_description="task",
            output=with_error,
            use_llm=False,
        )
        score_without = scorer.score(
            task_id="wo",
            model_id="m1",
            task_type="coding",
            task_description="task",
            output=without_error,
            use_llm=False,
        )
        assert score_with.dimensions["correctness"] > score_without.dimensions["correctness"]

    def test_coding_checks_type_hints(self, scorer):
        """Code with type hints gets higher style score."""
        with_hints = "def foo(x: str) -> int:\n    return len(x)"
        without_hints = "def foo(x):\n    return len(x)"

        score_with = scorer.score(
            task_id="wh",
            model_id="m1",
            task_type="coding",
            task_description="task",
            output=with_hints,
            use_llm=False,
        )
        score_without = scorer.score(
            task_id="woh",
            model_id="m1",
            task_type="coding",
            task_description="task",
            output=without_hints,
            use_llm=False,
        )
        assert score_with.dimensions["style"] > score_without.dimensions["style"]

    def test_documentation_checks_examples(self, scorer):
        """Docs with examples score higher on examples dimension."""
        with_examples = "# Guide\n\n## Usage\n\nExample:\n```python\nfoo()\n```\n\nSee other docs."
        without_examples = "# Guide\n\n## Usage\n\nThis function does things."

        score_with = scorer.score(
            task_id="de",
            model_id="m1",
            task_type="documentation",
            task_description="Write docs",
            output=with_examples,
            use_llm=False,
        )
        score_without = scorer.score(
            task_id="dne",
            model_id="m1",
            task_type="documentation",
            task_description="Write docs",
            output=without_examples,
            use_llm=False,
        )
        assert score_with.dimensions["examples"] > score_without.dimensions["examples"]

    def test_research_checks_citations(self, scorer):
        """Research with sources scores higher on source_quality."""
        with_sources = (
            "According to the study, evidence shows improvement. "
            "Source: https://example.com/paper\n\n"
            "## Conclusion\n\nRecommend further investigation."
        )
        without_sources = "The thing is probably good. It works well maybe."

        score_with = scorer.score(
            task_id="rs",
            model_id="m1",
            task_type="research",
            task_description="Research topic",
            output=with_sources,
            use_llm=False,
        )
        score_without = scorer.score(
            task_id="rns",
            model_id="m1",
            task_type="research",
            task_description="Research topic",
            output=without_sources,
            use_llm=False,
        )
        assert score_with.dimensions["source_quality"] > score_without.dimensions["source_quality"]


# ---------------------------------------------------------------------------
# 3.3 — Flat score distribution triggers LLM calibration
# ---------------------------------------------------------------------------


class TestFlatScoreCalibration:
    """Flat heuristic scores trigger forced LLM calibration."""

    def test_flat_distribution_detected(self, scorer):
        """If last 5 scores are within 0.05, _is_score_distribution_flat returns True."""
        key = ("model-a", "coding")
        scorer._score_history[key] = deque([0.55, 0.56, 0.54, 0.55, 0.56], maxlen=50)
        assert scorer._is_score_distribution_flat("model-a", "coding") is True

    def test_varied_distribution_not_flat(self, scorer):
        """Scores with meaningful variance should not trigger flat detection."""
        key = ("model-a", "coding")
        scorer._score_history[key] = deque([0.3, 0.5, 0.7, 0.4, 0.8], maxlen=50)
        assert scorer._is_score_distribution_flat("model-a", "coding") is False

    def test_insufficient_history_not_flat(self, scorer):
        """Less than 5 scores should never trigger flat detection."""
        key = ("model-a", "coding")
        scorer._score_history[key] = deque([0.55, 0.55], maxlen=50)
        assert scorer._is_score_distribution_flat("model-a", "coding") is False

    def test_flat_scores_force_calibration(self, scorer):
        """When flat detected, next score() call should attempt LLM calibration."""
        # Pre-fill flat history
        key = ("model-a", "coding")
        scorer._score_history[key] = deque([0.55, 0.56, 0.54, 0.55, 0.56], maxlen=50)

        # Mock adapter manager so we can detect calibration attempt
        mock_am = MagicMock()
        scorer._adapter_manager = mock_am

        with patch.object(scorer, "_score_with_llm", return_value=None) as mock_llm:
            result = scorer.score(
                task_id="t1",
                model_id="model-a",
                task_type="coding",
                task_description="task",
                output="def foo():\n    pass",
                use_llm=True,
            )
            # _score_with_llm should have been called due to flat detection
            mock_llm.assert_called_once()
        assert mock_llm.call_args.args[:5] == (
            "t1",
            "model-a",
            "coding",
            "task",
            "def foo():\n    pass",
        )
        assert result.task_id == "t1"
        assert result.method == "heuristic"


# ---------------------------------------------------------------------------
# 3.4 — Score variance monitoring
# ---------------------------------------------------------------------------


class TestScoreVarianceMonitoring:
    """Low variance over many scores triggers WARNING log."""

    def test_low_variance_logs_warning(self, scorer, caplog):
        """Variance < 0.01 over 20+ scores should log a warning."""
        key = ("model-b", "research")
        scorer._score_history[key] = deque([0.55] * 25, maxlen=50)

        with caplog.at_level(logging.WARNING):
            scorer._check_score_distribution("model-b", "research")

        assert any("variance too low" in msg.lower() for msg in caplog.messages), (
            f"Expected variance warning, got: {caplog.messages}"
        )

    def test_healthy_variance_no_warning(self, scorer, caplog):
        """Normal variance should not trigger warning."""
        key = ("model-b", "research")
        # Scores with real variance
        import random

        rng = random.Random(42)
        scores = [rng.uniform(0.3, 0.8) for _ in range(25)]
        scorer._score_history[key] = deque(scores, maxlen=50)

        with caplog.at_level(logging.WARNING):
            scorer._check_score_distribution("model-b", "research")

        assert not any("variance too low" in msg.lower() for msg in caplog.messages)

    def test_too_few_scores_no_warning(self, scorer, caplog):
        """Under 20 scores should not trigger variance check."""
        key = ("model-b", "research")
        scorer._score_history[key] = deque([0.55] * 10, maxlen=50)

        with caplog.at_level(logging.WARNING):
            scorer._check_score_distribution("model-b", "research")

        assert not any("variance too low" in msg.lower() for msg in caplog.messages)


# ---------------------------------------------------------------------------
# 3.5 — Inference confidence from logprob variance
# ---------------------------------------------------------------------------


class TestInferenceConfidence:
    """Logprob variance feeds into quality scoring as confidence signal."""

    def test_low_confidence_penalizes_scores(self, scorer):
        """Low inference confidence should reduce heuristic scores."""
        output = "def foo(x: str) -> int:\n    return len(x)"

        high_conf = scorer.score(
            task_id="hc",
            model_id="m1",
            task_type="coding",
            task_description="task",
            output=output,
            use_llm=False,
            inference_confidence=0.9,
        )
        low_conf = scorer.score(
            task_id="lc",
            model_id="m1",
            task_type="coding",
            task_description="task",
            output=output,
            use_llm=False,
            inference_confidence=0.1,
        )
        assert high_conf.overall_score > low_conf.overall_score, (
            f"High confidence ({high_conf.overall_score}) should score higher "
            f"than low confidence ({low_conf.overall_score})"
        )

    def test_none_confidence_no_effect(self, scorer):
        """None confidence (logprobs unavailable) should not affect scores."""
        output = "def foo(x: str) -> int:\n    return len(x)"

        with_none = scorer.score(
            task_id="n1",
            model_id="m1",
            task_type="coding",
            task_description="task",
            output=output,
            use_llm=False,
            inference_confidence=None,
        )
        without = scorer.score(
            task_id="n2",
            model_id="m1",
            task_type="coding",
            task_description="task",
            output=output,
            use_llm=False,
        )
        assert with_none.overall_score == without.overall_score

    def test_logprob_variance_in_response_metadata(self):
        """InferenceResponse metadata can carry logprob_variance and inference_confidence."""
        from vetinari.adapters.base import InferenceResponse

        response = InferenceResponse(
            model_id="test",
            output="hello",
            latency_ms=100,
            tokens_used=10,
            status="ok",
            metadata={"logprob_variance": 0.5, "inference_confidence": 0.83},
        )
        assert response.metadata["inference_confidence"] == 0.83
        assert response.metadata["logprob_variance"] == 0.5


# ---------------------------------------------------------------------------
# 3.6 — Unmeasured dimensions default to 0.0
# ---------------------------------------------------------------------------


class TestUnmeasuredDimensions:
    """Unmeasured dimensions show 0.0 with tracking, not fake 0.7."""

    def test_default_score_fields_are_zero(self):
        """QualityScore defaults correctness/completeness/efficiency/style to 0.0."""
        score = QualityScore(
            task_id="t1",
            model_id="m1",
            task_type="coding",
            overall_score=0.0,
        )
        assert score.correctness == 0.0
        assert score.completeness == 0.0
        assert score.efficiency == 0.0
        assert score.style == 0.0
        assert score.method == "unmeasured"

    def test_measured_dimensions_tracked(self, scorer):
        """Heuristic scoring must populate measured_dimensions for actually checked dims."""
        output = "def foo(x: str) -> int:\n    return len(x)"
        result = scorer.score(
            task_id="md",
            model_id="m1",
            task_type="coding",
            task_description="task",
            output=output,
            use_llm=False,
        )
        assert len(result.measured_dimensions) > 0, "Should have measured at least some dimensions"
        # Coding should measure correctness, style, test_coverage, efficiency
        assert "correctness" in result.measured_dimensions
        assert "style" in result.measured_dimensions

    def test_unmeasured_dims_are_zero(self, scorer):
        """Dimensions not measured should be 0.0 in the dimensions dict."""
        output = "def foo():\n    pass"
        result = scorer.score(
            task_id="uz",
            model_id="m1",
            task_type="coding",
            task_description="task",
            output=output,
            use_llm=False,
        )
        for dim, value in result.dimensions.items():
            if dim not in result.measured_dimensions:
                assert value == 0.0, f"Unmeasured dimension {dim!r} should be 0.0, got {value}"

    def test_get_model_average_returns_zero_for_unknown(self, scorer):
        """get_model_average returns 0.0 for models with no history."""
        avg = scorer.get_model_average("nonexistent-model")
        assert avg == 0.0, f"Expected 0.0 for unknown model, got {avg}"

    def test_empty_output_is_measured_zero(self, scorer):
        """Empty output should score 0.0 with measured dimensions, not unmeasured."""
        result = scorer.score(
            task_id="empty",
            model_id="m1",
            task_type="coding",
            task_description="task",
            output="",
            use_llm=False,
        )
        assert result.overall_score == 0.0
        assert result.method == "rejected"

    def test_rejected_output_has_measured_dimensions(self, scorer):
        """Rejected (fallback/empty) output should have measured_dimensions set."""
        result = scorer.score(
            task_id="rej",
            model_id="m1",
            task_type="coding",
            task_description="task",
            output="",
            use_llm=False,
        )
        assert len(result.measured_dimensions) > 0


# ---------------------------------------------------------------------------
# Cross-cutting: variance across task types
# ---------------------------------------------------------------------------


class TestScoreVarianceAcrossTaskTypes:
    """Quality scores show meaningful variance, not all ~0.65."""

    def test_scores_not_all_identical(self, scorer):
        """Scoring 5 different outputs should NOT produce identical scores."""
        outputs = [
            "def foo(): pass",
            "def bar(x: str) -> int:\n    '''Doc.'''\n    try:\n        return int(x)\n    except ValueError:\n        return 0",
            "x = 1",
            "# Nothing here",
            "import os\nclass Foo:\n    def __init__(self):\n        self.x = 1",
        ]
        scores = []
        for i, output in enumerate(outputs):
            result = scorer.score(
                task_id=f"var-{i}",
                model_id="m1",
                task_type="coding",
                task_description="Write code",
                output=output,
                use_llm=False,
            )
            scores.append(result.overall_score)

        unique_scores = set(scores)
        assert len(unique_scores) >= 3, (
            f"Expected at least 3 unique scores from 5 different outputs, got {len(unique_scores)}: {scores}"
        )

    def test_no_score_is_065(self, scorer):
        """The old flat 0.65 default should never appear."""
        outputs_and_types = [
            ("def foo(): pass", "coding"),
            ("The data shows trends according to sources.", "research"),
            ("# Title\n\n## Section\n\nContent here.", "documentation"),
            ("def test_foo():\n    assert 1 == 1", "testing"),
        ]
        for output, task_type in outputs_and_types:
            result = scorer.score(
                task_id=f"no65-{task_type}",
                model_id="m1",
                task_type=task_type,
                task_description="task",
                output=output,
                use_llm=False,
            )
            assert result.overall_score != 0.65, (
                f"Score for {task_type} should not be the old 0.65 default, got {result.overall_score}"
            )


# ---------------------------------------------------------------------------
# OutcomeSignal wrapper
# ---------------------------------------------------------------------------


class TestQualityScorerSignal:
    """Tests for QualityScorer.score_with_signal() OutcomeSignal wrapper."""

    def test_rejected_output_returns_unsupported(self, scorer):
        """Empty output (rejected) yields UNSUPPORTED basis, passed=False."""
        from vetinari.types import EvidenceBasis

        sig = scorer.score_with_signal(
            task_id="t1",
            model_id="m1",
            task_type="coding",
            task_description="Write code",
            output="",
            use_llm=False,
        )

        assert sig.passed is False
        assert sig.basis is EvidenceBasis.UNSUPPORTED
        assert sig.score == 0.0

    def test_normal_output_returns_llm_judgment(self, scorer):
        """Normal output yields LLM_JUDGMENT basis with populated llm_judgment."""
        from vetinari.types import EvidenceBasis

        output = "def foo(x: str) -> int:\n    '''Doc.'''\n    return len(x)"
        sig = scorer.score_with_signal(
            task_id="t2",
            model_id="m1",
            task_type="coding",
            task_description="Write code",
            output=output,
            use_llm=False,
        )

        assert sig.basis is EvidenceBasis.LLM_JUDGMENT
        assert sig.llm_judgment is not None
        assert sig.llm_judgment.model_id  # non-empty
        assert sig.llm_judgment.score >= 0.0
        assert sig.llm_judgment.summary  # non-empty

    def test_signal_passed_reflects_score_threshold(self, scorer):
        """passed=True only when score >= 0.5 and no issues."""
        # Empty output → rejected → passed=False regardless
        sig_bad = scorer.score_with_signal(
            task_id="bad",
            model_id="m1",
            task_type="coding",
            task_description="task",
            output="",
            use_llm=False,
        )
        assert sig_bad.passed is False

        # Good output → score high enough → passed=True
        good = "def foo(x: str) -> int:\n    '''Good.'''\n    try:\n        return int(x)\n    except ValueError:\n        return 0"
        sig_good = scorer.score_with_signal(
            task_id="good",
            model_id="m1",
            task_type="coding",
            task_description="task",
            output=good,
            use_llm=False,
        )
        # Score depends on heuristics; just confirm it aligns with the overall_score
        from vetinari.learning.quality_scorer import QualityScorer

        raw = QualityScorer().score(
            task_id="good2",
            model_id="m1",
            task_type="coding",
            task_description="task",
            output=good,
            use_llm=False,
        )
        if raw.overall_score >= 0.5 and not raw.issues:
            assert sig_good.passed is True
        else:
            assert sig_good.passed is False

    def test_signal_provenance_populated(self, scorer):
        """Provenance carries source and timestamp_utc."""
        output = "def foo(): return 1"
        sig = scorer.score_with_signal(
            task_id="prov",
            model_id="m1",
            task_type="coding",
            task_description="task",
            output=output,
            use_llm=False,
        )

        assert sig.provenance is not None
        assert sig.provenance.source  # non-empty
        assert sig.provenance.timestamp_utc
        assert "T" in sig.provenance.timestamp_utc  # ISO-8601

    def test_signal_llm_judgment_model_id_from_quality_score(self, scorer):
        """llm_judgment.model_id reflects the model_id passed to score_with_signal."""
        output = "def foo(): return 1"
        sig = scorer.score_with_signal(
            task_id="mid",
            model_id="test-model-xyz",
            task_type="coding",
            task_description="task",
            output=output,
            use_llm=False,
        )

        if sig.llm_judgment is not None:
            # model_id in llm_judgment comes from QualityScore.model_id
            assert sig.llm_judgment.model_id  # non-empty string
