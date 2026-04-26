"""Tests for quality-conditioned rework routing in vetinari.orchestration.pipeline_rework."""

from __future__ import annotations

import pytest

from vetinari.orchestration.pipeline_rework import PipelineReworkMixin, ReworkDecision

# -- Helpers ------------------------------------------------------------------


class _ConcreteRework(PipelineReworkMixin):
    """Minimal concrete class to exercise PipelineReworkMixin without orchestrator deps."""


# -- _select_rework_by_quality_score ------------------------------------------


class TestSelectReworkByQualityScore:
    """_select_rework_by_quality_score() maps score bands to decisions."""

    def setup_method(self) -> None:
        """Create a fresh mixin instance for each test."""
        self.mixin = _ConcreteRework()

    @pytest.mark.parametrize(
        ("score", "expected_decision"),
        [
            (0.75, ReworkDecision.RETRY_SAME_AGENT),
            (0.60, ReworkDecision.RETRY_DIFFERENT_MODEL),
            (0.40, ReworkDecision.RESEARCH_THEN_RETRY),
            (0.20, ReworkDecision.ESCALATE_TO_USER),
        ],
    )
    def test_score_maps_to_correct_decision(self, score: float, expected_decision: ReworkDecision) -> None:
        """Mid-range score values map to their documented decision."""
        result = self.mixin._select_rework_by_quality_score("task_001", score)
        assert result == expected_decision

    def test_boundary_0_70_returns_retry_same_agent(self) -> None:
        """Score of exactly 0.70 lands in the RETRY_SAME_AGENT band (>= 0.70)."""
        result = self.mixin._select_rework_by_quality_score("task_002", 0.70)
        assert result == ReworkDecision.RETRY_SAME_AGENT

    def test_boundary_0_50_returns_retry_different_model(self) -> None:
        """Score of exactly 0.50 lands in the RETRY_DIFFERENT_MODEL band (>= 0.50)."""
        result = self.mixin._select_rework_by_quality_score("task_003", 0.50)
        assert result == ReworkDecision.RETRY_DIFFERENT_MODEL

    def test_boundary_0_30_returns_research_then_retry(self) -> None:
        """Score of exactly 0.30 lands in the RESEARCH_THEN_RETRY band (>= 0.30)."""
        result = self.mixin._select_rework_by_quality_score("task_004", 0.30)
        assert result == ReworkDecision.RESEARCH_THEN_RETRY

    def test_score_just_below_0_30_escalates(self) -> None:
        """A score of 0.29 is below the last band — escalates to user."""
        result = self.mixin._select_rework_by_quality_score("task_005", 0.29)
        assert result == ReworkDecision.ESCALATE_TO_USER

    def test_perfect_score_returns_retry_same_agent(self) -> None:
        """Score of 1.0 (perfect) uses the lightest intervention."""
        result = self.mixin._select_rework_by_quality_score("task_006", 1.0)
        assert result == ReworkDecision.RETRY_SAME_AGENT

    def test_zero_score_escalates(self) -> None:
        """Score of 0.0 (no quality at all) escalates to user."""
        result = self.mixin._select_rework_by_quality_score("task_007", 0.0)
        assert result == ReworkDecision.ESCALATE_TO_USER
