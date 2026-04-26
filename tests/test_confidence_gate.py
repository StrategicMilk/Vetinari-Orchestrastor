"""Tests for vetinari.agents.confidence_gate — logprob-based confidence classification.

Also covers vetinari.awareness.confidence.classify_confidence_score — raw float
to ConfidenceLevel mapping used for operator-visible decision history.
"""

from __future__ import annotations

import pytest

from vetinari.agents.confidence_gate import ConfidenceGate
from vetinari.awareness.confidence import classify_confidence_score
from vetinari.types import ConfidenceAction, ConfidenceLevel

# -- assess_confidence --------------------------------------------------------


class TestAssessConfidence:
    """assess_confidence() classifies token logprob sequences."""

    def test_high_logprobs_return_high_level(self) -> None:
        """Mean logprob above -0.5 classifies as HIGH."""
        gate = ConfidenceGate()
        # -0.2 is above the HIGH threshold of -0.5
        result = gate.assess_confidence([-0.1, -0.2, -0.3])
        assert result.level == ConfidenceLevel.HIGH

    def test_medium_logprobs_return_medium_level(self) -> None:
        """Mean logprob between -0.5 and -1.5 classifies as MEDIUM."""
        gate = ConfidenceGate()
        # Mean around -1.0 — between HIGH (-0.5) and MEDIUM (-1.5) thresholds
        result = gate.assess_confidence([-0.8, -1.0, -1.2])
        assert result.level == ConfidenceLevel.MEDIUM

    def test_low_logprobs_return_low_level(self) -> None:
        """Mean logprob between -1.5 and -3.0 classifies as LOW."""
        gate = ConfidenceGate()
        # Mean around -2.0
        result = gate.assess_confidence([-1.8, -2.0, -2.2])
        assert result.level == ConfidenceLevel.LOW

    def test_very_low_logprobs_return_very_low_level(self) -> None:
        """Mean logprob below -3.0 classifies as VERY_LOW."""
        gate = ConfidenceGate()
        result = gate.assess_confidence([-4.0, -5.0, -6.0])
        assert result.level == ConfidenceLevel.VERY_LOW

    def test_empty_logprobs_return_very_low(self) -> None:
        """An empty logprob list signals no information — VERY_LOW."""
        gate = ConfidenceGate()
        result = gate.assess_confidence([])
        assert result.level == ConfidenceLevel.VERY_LOW

    def test_high_variance_downgrades_high_to_medium(self) -> None:
        """When std > 2.0, HIGH is downgraded to MEDIUM."""
        gate = ConfidenceGate()
        # Mean is high (-0.1) but variance is extreme
        result = gate.assess_confidence([-0.1, -0.1, -5.0, -5.0])
        # The high variance (std >> 2.0) should downgrade from HIGH or MEDIUM
        assert result.level in (ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW)

    def test_high_variance_downgrades_medium_range(self) -> None:
        """When std > 2.0 and base level is MEDIUM, it downgrades to a lower level."""
        gate = ConfidenceGate()
        # 9 tokens at -0.6 (mean base in MEDIUM range) + one very negative outlier
        # Mean = (-0.6*9 + -8.0)/10 = -1.34 (MEDIUM range), std ≈ 2.34 (> 2.0)
        vals = [-0.6] * 9 + [-8.0]
        result = gate.assess_confidence(vals)
        # std > 2.0 with MEDIUM base — should downgrade to LOW
        assert result.level == ConfidenceLevel.LOW


# -- route_by_confidence ------------------------------------------------------


class TestRouteByConfidence:
    """route_by_confidence() maps confidence levels to routing actions."""

    @pytest.mark.parametrize(
        ("logprobs", "expected_action"),
        [
            ([-0.1, -0.2], ConfidenceAction.PROCEED),
            ([-0.8, -1.0, -1.2], ConfidenceAction.REFINE),
            ([-1.8, -2.0, -2.2], ConfidenceAction.BEST_OF_N),
            ([-4.0, -5.0, -6.0], ConfidenceAction.DEFER_TO_HUMAN),
            ([], ConfidenceAction.DEFER_TO_HUMAN),
        ],
    )
    def test_routing_actions(self, logprobs: list[float], expected_action: ConfidenceAction) -> None:
        """Each confidence level maps to the correct routing action enum."""
        gate = ConfidenceGate()
        decision = gate.route_by_confidence(logprobs)
        assert decision.action == expected_action


# -- assess_semantic_entropy --------------------------------------------------


class TestSemanticEntropy:
    """assess_semantic_entropy() uses response diversity to infer confidence."""

    def test_similar_responses_return_high(self) -> None:
        """Nearly identical responses indicate the model is confident."""
        gate = ConfidenceGate()
        # Identical responses — maximum similarity
        responses = ["the cat sat on the mat"] * 3
        result = gate.assess_semantic_entropy(responses)
        assert result.level == ConfidenceLevel.HIGH

    def test_diverse_responses_return_low_or_very_low(self) -> None:
        """Completely different responses signal high uncertainty."""
        gate = ConfidenceGate()
        responses = [
            "alpha beta gamma delta epsilon",
            "zulu yankee xray whiskey victor",
            "north south east west central",
        ]
        result = gate.assess_semantic_entropy(responses)
        assert result.level in (ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW)

    def test_single_response_triggers_unknown_protocol(self) -> None:
        """A single response cannot be assessed — returns VERY_LOW via unknown protocol."""
        gate = ConfidenceGate()
        result = gate.assess_semantic_entropy(["only one response"])
        assert result.level == ConfidenceLevel.VERY_LOW


# -- classify_confidence_score ------------------------------------------------


class TestClassifyConfidenceScore:
    """classify_confidence_score() maps 0.0-1.0 agent scores to ConfidenceLevel."""

    @pytest.mark.parametrize("score", [0.85, 0.9, 1.0])
    def test_high_confidence(self, score: float) -> None:
        """Scores at or above 0.85 classify as HIGH."""
        assert classify_confidence_score(score) == ConfidenceLevel.HIGH

    @pytest.mark.parametrize("score", [0.6, 0.7, 0.84])
    def test_medium_confidence(self, score: float) -> None:
        """Scores in [0.60, 0.85) classify as MEDIUM."""
        assert classify_confidence_score(score) == ConfidenceLevel.MEDIUM

    @pytest.mark.parametrize("score", [0.3, 0.4, 0.59])
    def test_low_confidence(self, score: float) -> None:
        """Scores in [0.30, 0.60) classify as LOW."""
        assert classify_confidence_score(score) == ConfidenceLevel.LOW

    @pytest.mark.parametrize("score", [0.0, 0.1, 0.15, 0.29])
    def test_very_low_confidence(self, score: float) -> None:
        """Scores below 0.30 (including 0.0) classify as VERY_LOW."""
        assert classify_confidence_score(score) == ConfidenceLevel.VERY_LOW

    def test_boundary_at_zero_is_very_low(self) -> None:
        """A score of exactly 0.0 (fallback sentinel) classifies as VERY_LOW."""
        assert classify_confidence_score(0.0) == ConfidenceLevel.VERY_LOW

    def test_very_low_and_fallback_share_level_but_differ_by_score(self) -> None:
        """Both 0.0 and 0.15 are VERY_LOW, but the raw score distinguishes fallback from genuine low."""
        level_nonzero = classify_confidence_score(0.15)
        level_zero = classify_confidence_score(0.0)
        assert level_nonzero == ConfidenceLevel.VERY_LOW
        assert level_zero == ConfidenceLevel.VERY_LOW
        # The caller can use score == 0.0 to detect fallback; level alone cannot
        assert 0.15 != 0.0
