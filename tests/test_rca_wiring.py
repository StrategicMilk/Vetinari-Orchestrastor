"""Tests for RCA wiring into QualityAgent and TwoLayerOrchestrator.

Verifies that:
- QualityAgent has _perform_root_cause_analysis
- root_cause metadata is added to failed quality results
- TwoLayerOrchestrator._handle_quality_rejection routes based on DefectCategory
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.agents.consolidated.quality_agent import InspectorAgent as QualityAgent
from vetinari.agents.contracts import AgentResult, AgentTask
from vetinari.orchestration.two_layer import TwoLayerOrchestrator
from vetinari.types import AgentType
from vetinari.validation import DefectCategory, RootCauseAnalysis

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def quality_agent() -> QualityAgent:
    """Return a bare QualityAgent with no LLM context required."""
    return QualityAgent(config={})


@pytest.fixture
def orchestrator() -> TwoLayerOrchestrator:
    """Return a TwoLayerOrchestrator with minimal config."""
    return TwoLayerOrchestrator(enable_correction_loop=False)


@pytest.fixture
def sample_task() -> AgentTask:
    """Return a minimal AgentTask for testing."""
    return AgentTask(
        task_id="test-task-1",
        agent_type=AgentType.INSPECTOR,
        description="Review this code for quality",
        prompt="Review this code for quality",
        context={"code": "def foo(): pass", "mode": "code_review"},
    )


# ---------------------------------------------------------------------------
# 1. Structural: QualityAgent has the method
# ---------------------------------------------------------------------------


class TestQualityAgentHasRcaMethod:
    def test_quality_agent_has_rca_method(self, quality_agent: QualityAgent) -> None:
        """QualityAgent must expose _perform_root_cause_analysis."""
        assert hasattr(quality_agent, "_perform_root_cause_analysis"), (
            "QualityAgent is missing _perform_root_cause_analysis"
        )
        assert callable(quality_agent._perform_root_cause_analysis)


# ---------------------------------------------------------------------------
# 2. Metadata wiring: root_cause added on fail
# ---------------------------------------------------------------------------


class TestRcaMetadataAddedOnFail:
    def test_rca_metadata_added_on_fail(self, quality_agent: QualityAgent, sample_task: AgentTask) -> None:
        """_perform_root_cause_analysis adds root_cause key to metadata."""
        failed_result = AgentResult(
            success=True,
            output={
                "score": 0.2,
                "issues": [{"message": "hallucinated import not found", "severity": "high"}],
                "summary": "Code references missing module",
            },
            metadata={"mode": "code_review"},
        )

        updated = quality_agent._perform_root_cause_analysis(sample_task, failed_result)

        assert "root_cause" in updated.metadata
        rc = updated.metadata["root_cause"]
        assert "category" in rc
        assert "confidence" in rc
        assert "corrective_action" in rc
        assert "preventive_action" in rc
        assert "evidence" in rc
        # confidence must be a float in [0, 1]
        assert 0.0 <= rc["confidence"] <= 1.0

    def test_rca_metadata_contains_valid_category(self, quality_agent: QualityAgent, sample_task: AgentTask) -> None:
        """The root_cause category must be a valid DefectCategory value."""
        failed_result = AgentResult(
            success=True,
            output={"score": 0.1, "issues": [], "summary": "ambiguous spec unclear requirements"},
            metadata={"mode": "code_review"},
        )

        updated = quality_agent._perform_root_cause_analysis(sample_task, failed_result)

        valid_categories = {c.value for c in DefectCategory}
        assert updated.metadata["root_cause"]["category"] in valid_categories


# ---------------------------------------------------------------------------
# 3-7. Corrective routing in TwoLayerOrchestrator
# ---------------------------------------------------------------------------


class TestCorrectiveRouting:
    def _result_with_category(self, category_value: str) -> dict:
        """Build a minimal result dict with a root_cause of the given category."""
        return {"root_cause": {"category": category_value, "confidence": 0.85}}

    def test_corrective_routing_bad_spec(self, orchestrator: TwoLayerOrchestrator) -> None:
        """BAD_SPEC category must route to REPLAN."""
        from vetinari.orchestration.two_layer import ReworkDecision

        result = self._result_with_category(DefectCategory.BAD_SPEC.value)
        action = orchestrator._handle_quality_rejection("task-1", result, rework_count=0)
        assert action == ReworkDecision.REPLAN

    def test_corrective_routing_wrong_model(self, orchestrator: TwoLayerOrchestrator) -> None:
        """WRONG_MODEL category must route to RETRY_DIFFERENT_MODEL."""
        from vetinari.orchestration.two_layer import ReworkDecision

        result = self._result_with_category(DefectCategory.WRONG_MODEL.value)
        action = orchestrator._handle_quality_rejection("task-2", result, rework_count=1)
        assert action == ReworkDecision.RETRY_DIFFERENT_MODEL

    def test_corrective_routing_hallucination(self, orchestrator: TwoLayerOrchestrator) -> None:
        """HALLUCINATION category must route to RETRY_SAME_AGENT."""
        from vetinari.orchestration.two_layer import ReworkDecision

        result = self._result_with_category(DefectCategory.HALLUCINATION.value)
        action = orchestrator._handle_quality_rejection("task-3", result, rework_count=0)
        assert action == ReworkDecision.RETRY_SAME_AGENT

    def test_corrective_routing_context(self, orchestrator: TwoLayerOrchestrator) -> None:
        """INSUFFICIENT_CONTEXT category must route to RESEARCH_THEN_RETRY."""
        from vetinari.orchestration.two_layer import ReworkDecision

        result = self._result_with_category(DefectCategory.INSUFFICIENT_CONTEXT.value)
        action = orchestrator._handle_quality_rejection("task-4", result, rework_count=0)
        assert action == ReworkDecision.RESEARCH_THEN_RETRY

    def test_corrective_routing_integration_error(self, orchestrator: TwoLayerOrchestrator) -> None:
        """INTEGRATION_ERROR category must route to REPLAN_WIDER_SCOPE."""
        from vetinari.orchestration.two_layer import ReworkDecision

        result = self._result_with_category(DefectCategory.INTEGRATION_ERROR.value)
        action = orchestrator._handle_quality_rejection("task-5", result, rework_count=0)
        assert action == ReworkDecision.REPLAN_WIDER_SCOPE

    def test_corrective_routing_complexity(self, orchestrator: TwoLayerOrchestrator) -> None:
        """COMPLEXITY_UNDERESTIMATE category must route to REPLAN."""
        from vetinari.orchestration.two_layer import ReworkDecision

        result = self._result_with_category(DefectCategory.COMPLEXITY_UNDERESTIMATE.value)
        action = orchestrator._handle_quality_rejection("task-6", result, rework_count=0)
        assert action == ReworkDecision.REPLAN

    def test_corrective_routing_prompt_weakness(self, orchestrator: TwoLayerOrchestrator) -> None:
        """PROMPT_WEAKNESS category must route to RETRY_SAME_AGENT."""
        from vetinari.orchestration.two_layer import ReworkDecision

        result = self._result_with_category(DefectCategory.PROMPT_WEAKNESS.value)
        action = orchestrator._handle_quality_rejection("task-7", result, rework_count=0)
        assert action == ReworkDecision.RETRY_SAME_AGENT

    def test_corrective_routing_escalate(self, orchestrator: TwoLayerOrchestrator) -> None:
        """rework_count >= 3 must return ESCALATE_TO_USER regardless of category."""
        from vetinari.orchestration.two_layer import ReworkDecision

        result = self._result_with_category(DefectCategory.BAD_SPEC.value)
        action = orchestrator._handle_quality_rejection("task-8", result, rework_count=3)
        assert action == ReworkDecision.ESCALATE_TO_USER

    def test_corrective_routing_escalate_at_exactly_3(self, orchestrator: TwoLayerOrchestrator) -> None:
        """Boundary check: rework_count == 3 is the escalation threshold."""
        from vetinari.orchestration.two_layer import ReworkDecision

        result = self._result_with_category(DefectCategory.HALLUCINATION.value)
        action = orchestrator._handle_quality_rejection("task-9", result, rework_count=3)
        assert action == ReworkDecision.ESCALATE_TO_USER

    def test_corrective_routing_no_root_cause(self, orchestrator: TwoLayerOrchestrator) -> None:
        """Missing root_cause key in result must default to RETRY_SAME_AGENT."""
        from vetinari.orchestration.two_layer import ReworkDecision

        result: dict = {"issues": []}  # No root_cause key, no quality score
        action = orchestrator._handle_quality_rejection("task-10", result, rework_count=0)
        assert action == ReworkDecision.RETRY_SAME_AGENT

    def test_corrective_routing_empty_result(self, orchestrator: TwoLayerOrchestrator) -> None:
        """Empty result dict must default to RETRY_SAME_AGENT."""
        from vetinari.orchestration.two_layer import ReworkDecision

        action = orchestrator._handle_quality_rejection("task-11", {}, rework_count=1)
        assert action == ReworkDecision.RETRY_SAME_AGENT
