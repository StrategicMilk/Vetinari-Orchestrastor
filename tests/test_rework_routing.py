"""Tests for the Reflect/Replan Feedback Loop — ReworkDecision routing."""

from __future__ import annotations

import pytest

from vetinari.orchestration.two_layer import ReworkDecision, TwoLayerOrchestrator


@pytest.fixture
def orchestrator():
    """Create a TwoLayerOrchestrator for testing rework routing."""
    return TwoLayerOrchestrator.__new__(TwoLayerOrchestrator)


class TestReworkDecisionRouting:
    """Test that _handle_quality_rejection routes correctly per DefectCategory."""

    def test_bad_spec_routes_to_replan(self, orchestrator):
        """BAD_SPEC defects should trigger replanning."""
        result = {"root_cause": {"category": "bad_spec"}}
        decision = orchestrator._handle_quality_rejection("t1", result, 0)
        assert decision == ReworkDecision.REPLAN

    def test_complexity_routes_to_replan(self, orchestrator):
        """COMPLEXITY_UNDERESTIMATE defects should trigger replanning."""
        result = {"root_cause": {"category": "complexity"}}
        decision = orchestrator._handle_quality_rejection("t1", result, 0)
        assert decision == ReworkDecision.REPLAN

    def test_wrong_model_routes_to_retry_different(self, orchestrator):
        """WRONG_MODEL defects should retry with a different model."""
        result = {"root_cause": {"category": "wrong_model"}}
        decision = orchestrator._handle_quality_rejection("t1", result, 0)
        assert decision == ReworkDecision.RETRY_DIFFERENT_MODEL

    def test_hallucination_routes_to_retry_same(self, orchestrator):
        """HALLUCINATION defects should retry the same agent with stricter instructions."""
        result = {"root_cause": {"category": "hallucination"}}
        decision = orchestrator._handle_quality_rejection("t1", result, 0)
        assert decision == ReworkDecision.RETRY_SAME_AGENT

    def test_prompt_weakness_routes_to_retry_same(self, orchestrator):
        """PROMPT_WEAKNESS defects should retry the same agent."""
        result = {"root_cause": {"category": "prompt"}}
        decision = orchestrator._handle_quality_rejection("t1", result, 0)
        assert decision == ReworkDecision.RETRY_SAME_AGENT

    def test_context_routes_to_research_then_retry(self, orchestrator):
        """INSUFFICIENT_CONTEXT defects should research then retry."""
        result = {"root_cause": {"category": "context"}}
        decision = orchestrator._handle_quality_rejection("t1", result, 0)
        assert decision == ReworkDecision.RESEARCH_THEN_RETRY

    def test_integration_routes_to_replan_wider_scope(self, orchestrator):
        """INTEGRATION_ERROR defects should replan with wider scope."""
        result = {"root_cause": {"category": "integration"}}
        decision = orchestrator._handle_quality_rejection("t1", result, 0)
        assert decision == ReworkDecision.REPLAN_WIDER_SCOPE

    def test_escalate_at_max_rework(self, orchestrator):
        """Exceeding max rework count (3) should escalate to user."""
        result = {"root_cause": {"category": "hallucination"}}
        decision = orchestrator._handle_quality_rejection("t1", result, 3)
        assert decision == ReworkDecision.ESCALATE_TO_USER

    def test_no_root_cause_defaults_to_retry_same(self, orchestrator):
        """Missing root_cause in result defaults to RETRY_SAME_AGENT."""
        decision = orchestrator._handle_quality_rejection("t1", {}, 0)
        assert decision == ReworkDecision.RETRY_SAME_AGENT

    def test_unknown_category_defaults_to_retry_same(self, orchestrator):
        """Unknown defect category defaults to RETRY_SAME_AGENT."""
        result = {"root_cause": {"category": "unknown_category"}}
        decision = orchestrator._handle_quality_rejection("t1", result, 0)
        assert decision == ReworkDecision.RETRY_SAME_AGENT
