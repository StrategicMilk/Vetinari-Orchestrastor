"""Tests for vetinari.awareness.confidence — canonical confidence computation."""

from __future__ import annotations

import pytest

from vetinari.awareness.confidence import (
    ConfidenceComputer,
    ConfidenceResult,
    UnknownSituation,
    UnknownSituationProtocol,
)
from vetinari.types import ConfidenceAction, ConfidenceLevel

# -- ConfidenceResult ---------------------------------------------------------


class TestConfidenceResult:
    """ConfidenceResult properties and __repr__."""

    def test_is_actionable_true_for_proceed(self) -> None:
        """is_actionable is True when action is PROCEED."""
        result = ConfidenceResult(
            score=-0.2,
            level=ConfidenceLevel.HIGH,
            action=ConfidenceAction.PROCEED,
            explanation="test",
        )
        assert result.is_actionable is True

    @pytest.mark.parametrize(
        "action",
        [ConfidenceAction.REFINE, ConfidenceAction.BEST_OF_N, ConfidenceAction.DEFER_TO_HUMAN],
    )
    def test_is_actionable_false_for_non_proceed(self, action: ConfidenceAction) -> None:
        """is_actionable is False for all non-PROCEED actions."""
        result = ConfidenceResult(
            score=-2.0,
            level=ConfidenceLevel.MEDIUM,
            action=action,
            explanation="test",
        )
        assert result.is_actionable is False

    def test_needs_human_true_for_defer(self) -> None:
        """needs_human is True when action is DEFER_TO_HUMAN."""
        result = ConfidenceResult(
            score=-10.0,
            level=ConfidenceLevel.VERY_LOW,
            action=ConfidenceAction.DEFER_TO_HUMAN,
            explanation="test",
        )
        assert result.needs_human is True

    @pytest.mark.parametrize(
        "action",
        [ConfidenceAction.PROCEED, ConfidenceAction.REFINE, ConfidenceAction.BEST_OF_N],
    )
    def test_needs_human_false_for_non_defer(self, action: ConfidenceAction) -> None:
        """needs_human is False for non-DEFER_TO_HUMAN actions."""
        result = ConfidenceResult(
            score=-0.5,
            level=ConfidenceLevel.HIGH,
            action=action,
            explanation="test",
        )
        assert result.needs_human is False

    def test_repr_without_unknown_situation(self) -> None:
        """__repr__ includes score, level, and action when no unknown situation."""
        result = ConfidenceResult(
            score=-0.3,
            level=ConfidenceLevel.HIGH,
            action=ConfidenceAction.PROCEED,
            explanation="test",
        )
        r = repr(result)
        assert "score=-0.300" in r
        assert "'high'" in r
        assert "'proceed'" in r
        assert "unknown" not in r

    def test_repr_with_unknown_situation(self) -> None:
        """__repr__ includes the unknown situation when set."""
        result = ConfidenceResult(
            score=-999.0,
            level=ConfidenceLevel.VERY_LOW,
            action=ConfidenceAction.DEFER_TO_HUMAN,
            explanation="test",
            unknown_situation=UnknownSituation.NO_DATA,
        )
        r = repr(result)
        assert "unknown=no_data" in r


# -- ConfidenceComputer.compute -----------------------------------------------


class TestConfidenceComputerCompute:
    """ConfidenceComputer.compute() — logprob-based classification."""

    @pytest.mark.parametrize(
        ("logprobs", "expected_level", "expected_action"),
        [
            ([-0.1, -0.2, -0.3], ConfidenceLevel.HIGH, ConfidenceAction.PROCEED),
            ([-0.8, -1.0, -1.2], ConfidenceLevel.MEDIUM, ConfidenceAction.REFINE),
            ([-1.8, -2.0, -2.2], ConfidenceLevel.LOW, ConfidenceAction.BEST_OF_N),
            ([-4.0, -5.0, -6.0], ConfidenceLevel.VERY_LOW, ConfidenceAction.DEFER_TO_HUMAN),
        ],
    )
    def test_level_and_action_for_logprob_ranges(
        self,
        logprobs: list[float],
        expected_level: ConfidenceLevel,
        expected_action: ConfidenceAction,
    ) -> None:
        """Each logprob range maps to the correct level and action."""
        computer = ConfidenceComputer()
        result = computer.compute(logprobs)
        assert result.level == expected_level
        assert result.action == expected_action

    def test_empty_logprobs_triggers_unknown_protocol(self) -> None:
        """Empty logprobs produce VERY_LOW with NO_DATA unknown situation."""
        computer = ConfidenceComputer()
        result = computer.compute([])
        assert result.level == ConfidenceLevel.VERY_LOW
        assert result.action == ConfidenceAction.DEFER_TO_HUMAN
        assert result.unknown_situation == UnknownSituation.NO_DATA

    def test_high_variance_downgrades_medium_to_low(self) -> None:
        """std > 2.0 with MEDIUM base level downgrades to LOW."""
        computer = ConfidenceComputer()
        # Mean ≈ -0.69 (MEDIUM), std ≈ 1.86 — close to threshold
        # Use higher-variance set: mean ≈ -1.42 (MEDIUM), std ≈ 2.34
        vals = [-0.6] * 9 + [-8.0]
        result = computer.compute(vals)
        assert result.level == ConfidenceLevel.LOW
        assert result.factors.get("variance_downgrade") == 1.0

    def test_factors_populated(self) -> None:
        """compute() always populates mean, min, std, and token_count factors."""
        computer = ConfidenceComputer()
        result = computer.compute([-0.5, -1.0, -1.5])
        assert "mean_logprob" in result.factors
        assert "min_logprob" in result.factors
        assert "std_logprob" in result.factors
        assert "token_count" in result.factors
        assert result.factors["token_count"] == 3.0

    def test_score_equals_mean_logprob(self) -> None:
        """score field is the mean of the input logprobs."""
        computer = ConfidenceComputer()
        logprobs = [-1.0, -2.0, -3.0]
        result = computer.compute(logprobs)
        assert abs(result.score - (-2.0)) < 1e-9

    def test_source_is_logprob(self) -> None:
        """compute() sets source='logprob'."""
        computer = ConfidenceComputer()
        result = computer.compute([-0.5])
        assert result.source == "logprob"

    def test_task_type_in_metadata(self) -> None:
        """task_type is stored in metadata."""
        computer = ConfidenceComputer()
        result = computer.compute([-0.5], task_type="code_generation")
        assert result.metadata.get("task_type") == "code_generation"

    def test_custom_thresholds(self) -> None:
        """Custom thresholds override the defaults."""
        # Move HIGH threshold way down so everything becomes HIGH
        computer = ConfidenceComputer(threshold_high=-100.0)
        result = computer.compute([-50.0])
        assert result.level == ConfidenceLevel.HIGH


# -- ConfidenceComputer.compute_from_responses --------------------------------


class TestConfidenceComputerFromResponses:
    """compute_from_responses() — semantic entropy based assessment."""

    def test_identical_responses_give_high_confidence(self) -> None:
        """Identical responses produce HIGH confidence."""
        computer = ConfidenceComputer()
        responses = ["the cat sat on the mat"] * 3
        result = computer.compute_from_responses(responses)
        assert result.level == ConfidenceLevel.HIGH
        assert result.source == "semantic_entropy"

    def test_diverse_responses_give_low_confidence(self) -> None:
        """Completely different responses produce LOW or VERY_LOW confidence."""
        computer = ConfidenceComputer()
        responses = [
            "alpha beta gamma delta epsilon",
            "zulu yankee xray whiskey victor",
            "north south east west central",
        ]
        result = computer.compute_from_responses(responses)
        assert result.level in (ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW)

    def test_single_response_triggers_unknown_protocol(self) -> None:
        """Fewer than 2 responses produce VERY_LOW with LOW_EVIDENCE unknown."""
        computer = ConfidenceComputer()
        result = computer.compute_from_responses(["only one"])
        assert result.level == ConfidenceLevel.VERY_LOW
        assert result.unknown_situation == UnknownSituation.LOW_EVIDENCE

    def test_factors_populated(self) -> None:
        """compute_from_responses populates mean_similarity, response_count, pair_count."""
        computer = ConfidenceComputer()
        responses = ["hello world", "hello world", "hello world"]
        result = computer.compute_from_responses(responses)
        assert "mean_similarity" in result.factors
        assert result.factors["response_count"] == 3.0
        assert result.factors["pair_count"] == 3.0  # 3 pairs from 3 responses

    def test_custom_similarity_threshold(self) -> None:
        """A very low threshold classifies moderate similarity as HIGH."""
        computer = ConfidenceComputer()
        # Two partially similar responses
        responses = ["hello world foo", "hello world bar"]
        # With a low threshold, moderate similarity should be HIGH
        result_low_threshold = computer.compute_from_responses(responses, similarity_threshold=0.1)
        assert result_low_threshold.level == ConfidenceLevel.HIGH


# -- UnknownSituationProtocol -------------------------------------------------


class TestUnknownSituationProtocol:
    """UnknownSituationProtocol.declare_unknown() — explicit uncertainty handling."""

    @pytest.mark.parametrize("situation", list(UnknownSituation))
    def test_all_situations_produce_very_low_defer(self, situation: UnknownSituation) -> None:
        """Every unknown situation produces VERY_LOW level with DEFER_TO_HUMAN action."""
        result = UnknownSituationProtocol.declare_unknown(situation, "test explanation")
        assert result.level == ConfidenceLevel.VERY_LOW
        assert result.action == ConfidenceAction.DEFER_TO_HUMAN
        assert result.unknown_situation == situation

    def test_score_is_sentinel_value(self) -> None:
        """Unknown results have score=-999.0 as sentinel."""
        result = UnknownSituationProtocol.declare_unknown(UnknownSituation.NO_DATA, "no data available")
        assert result.score == -999.0

    def test_explanation_preserved(self) -> None:
        """The explanation is stored verbatim."""
        explanation = "specific explanation about why this is unknown"
        result = UnknownSituationProtocol.declare_unknown(UnknownSituation.CONTRADICTORY, explanation)
        assert result.explanation == explanation

    def test_custom_source(self) -> None:
        """Custom source parameter is stored correctly."""
        result = UnknownSituationProtocol.declare_unknown(
            UnknownSituation.OUT_OF_DOMAIN,
            "out of domain",
            source="domain_classifier",
        )
        assert result.source == "domain_classifier"

    def test_default_source(self) -> None:
        """Default source is 'unknown_protocol'."""
        result = UnknownSituationProtocol.declare_unknown(UnknownSituation.STALE_EVIDENCE, "stale")
        assert result.source == "unknown_protocol"
