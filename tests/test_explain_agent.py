import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vetinari.explain_agent import (
    ExplainAgent,
    ExplanationBlock,
    PlanExplanation,
    SubtaskExplanation,
    get_explain_agent,
)
from vetinari.plan_types import Plan, PlanRiskLevel, Subtask, SubtaskStatus, TaskDomain


class TestExplainAgent:
    """Tests for the ExplainAgent module."""

    def setup_method(self):
        """Set up test fixtures."""
        os.environ["EXPLAINABILITY_ENABLED"] = "true"
        self.agent = ExplainAgent()

    def test_explain_agent_initialization(self):
        """Test ExplainAgent initializes correctly."""
        assert self.agent is not None
        assert self.agent.enabled is True
        assert len(self.agent._domain_templates) > 0

    def test_explain_agent_disabled(self):
        """Test ExplainAgent when disabled."""
        os.environ["EXPLAINABILITY_ENABLED"] = "false"
        agent = ExplainAgent()
        assert agent.enabled is False

    def test_explain_plan_returns_plan_explanation(self):
        """Test that explain_plan returns a PlanExplanation object."""
        plan = Plan(
            plan_id="test_plan_001",
            plan_version=1,
            goal="Build a web application",
            risk_score=0.15,
            risk_level=PlanRiskLevel.LOW,
            chosen_plan_id="candidate_1",
            plan_justification="Low risk, well-scoped"
        )

        explanation = self.agent.explain_plan(plan)

        assert isinstance(explanation, PlanExplanation)
        assert explanation.plan_id == "test_plan_001"
        assert len(explanation.blocks) > 0
        assert explanation.summary != ""

    def test_explanation_blocks_have_required_fields(self):
        """Test that explanation blocks have all required fields."""
        plan = Plan(
            plan_id="test_plan_002",
            plan_version=1,
            goal="Implement API endpoints",
            risk_score=0.25,
            risk_level=PlanRiskLevel.MEDIUM
        )

        explanation = self.agent.explain_plan(plan)

        for block in explanation.blocks:
            assert block.id is not None
            assert block.target_id == "test_plan_002"
            assert block.content != ""
            assert 0.0 <= block.confidence <= 1.0
            assert block.timestamp is not None

    def test_domain_inference_coding(self):
        """Test domain inference for coding tasks."""
        assert self.agent._infer_domain("Build a web app with Python") == "coding"
        assert self.agent._infer_domain("Implement REST API") == "coding"
        assert self.agent._infer_domain("Write unit tests") == "coding"

    def test_domain_inference_data_processing(self):
        """Test domain inference for data processing tasks."""
        assert self.agent._infer_domain("Build ETL pipeline") == "data_processing"
        assert self.agent._infer_domain("Process data with Spark") == "data_processing"

    def test_domain_inference_infra(self):
        """Test domain inference for infrastructure tasks."""
        assert self.agent._infer_domain("Set up CI/CD pipeline") == "infra"
        assert self.agent._infer_domain("Deploy to Kubernetes") == "infra"

    def test_domain_inference_docs(self):
        """Test domain inference for documentation tasks."""
        assert self.agent._infer_domain("Write API documentation") == "docs"
        assert self.agent._infer_domain("Create user guide") == "docs"

    def test_domain_inference_ai_experiments(self):
        """Test domain inference for AI experiment tasks."""
        assert self.agent._infer_domain("Run model comparison experiments") == "ai_experiments"
        assert self.agent._infer_domain("Benchmark model performance") == "ai_experiments"

    def test_domain_inference_research(self):
        """Test domain inference for research tasks."""
        assert self.agent._infer_domain("Research new algorithms") == "research"
        assert self.agent._infer_domain("Study literature on ML") == "research"

    def test_domain_inference_general(self):
        """Test domain inference for general tasks."""
        assert self.agent._infer_domain("Do something") == "general"
        assert self.agent._infer_domain("Complete task") == "general"

    def test_explain_subtask_returns_subtask_explanation(self):
        """Test that explain_subtask returns a SubtaskExplanation object."""
        subtask = Subtask(
            subtask_id="subtask_001",
            plan_id="test_plan_001",
            description="Implement API endpoint",
            domain=TaskDomain.CODING,
            depth=0,
            status=SubtaskStatus.PENDING
        )

        explanation = self.agent.explain_subtask(subtask)

        assert isinstance(explanation, SubtaskExplanation)
        assert explanation.subtask_id == "subtask_001"
        assert len(explanation.blocks) > 0

    def test_sanitize_explanation(self):
        """Test explanation sanitization for public exposure."""
        plan = Plan(
            plan_id="test_plan_003",
            plan_version=1,
            goal="Build a web app"
        )

        explanation = self.agent.explain_plan(plan)

        # Mark one block as unsanitized
        explanation.blocks[0].sanitized = False

        sanitized = self.agent.sanitize_explanation(explanation)

        # All blocks should be marked as sanitized in the result
        for block in sanitized.blocks:
            assert block.sanitized is True

    def test_plan_explanation_json_serialization(self):
        """Test PlanExplanation can be serialized to JSON."""
        plan = Plan(
            plan_id="test_plan_004",
            plan_version=1,
            goal="Test serialization"
        )

        explanation = self.agent.explain_plan(plan)
        json_str = json.dumps(explanation.to_dict())

        assert json_str is not None
        assert "plan_id" in json_str
        assert "blocks" in json_str

    def test_get_explain_agent_singleton(self):
        """Test that get_explain_agent returns a singleton."""
        agent1 = get_explain_agent()
        agent2 = get_explain_agent()

        assert agent1 is agent2


class TestExplanationBlock:
    """Tests for the ExplanationBlock dataclass."""

    def test_explanation_block_to_dict(self):
        """Test ExplanationBlock to_dict conversion."""
        block = ExplanationBlock(
            target_id="plan_123",
            domain="planning",
            content="Test content",
            confidence=0.8,
            sources=["test"],
            sanitized=True
        )

        block_dict = block.to_dict()

        assert block_dict["target_id"] == "plan_123"
        assert block_dict["domain"] == "planning"
        assert block_dict["confidence"] == 0.8
        assert block_dict["sanitized"] is True

    def test_explanation_block_from_dict(self):
        """Test ExplanationBlock from_dict conversion."""
        block_dict = {
            "id": "exp_123",
            "target_id": "plan_456",
            "domain": "coding",
            "content": "Test explanation",
            "confidence": 0.75,
            "sources": ["test"],
            "timestamp": "2024-01-01T00:00:00",
            "sanitized": False
        }

        block = ExplanationBlock.from_dict(block_dict)

        assert block.id == "exp_123"
        assert block.target_id == "plan_456"
        assert block.confidence == 0.75


class TestPlanExplanation:
    """Tests for the PlanExplanation dataclass."""

    def test_plan_explanation_to_dict(self):
        """Test PlanExplanation to_dict conversion."""
        block = ExplanationBlock(
            target_id="plan_123",
            domain="planning",
            content="Test"
        )

        explanation = PlanExplanation(
            plan_id="plan_123",
            plan_version=1,
            blocks=[block],
            summary="Test summary"
        )

        exp_dict = explanation.to_dict()

        assert exp_dict["plan_id"] == "plan_123"
        assert len(exp_dict["blocks"]) == 1
        assert exp_dict["summary"] == "Test summary"

    def test_plan_explanation_from_dict(self):
        """Test PlanExplanation from_dict conversion."""
        exp_dict = {
            "plan_id": "plan_456",
            "plan_version": 1,
            "blocks": [
                {
                    "id": "exp_1",
                    "target_id": "plan_456",
                    "domain": "planning",
                    "content": "Test",
                    "confidence": 0.8,
                    "sources": [],
                    "timestamp": "2024-01-01T00:00:00",
                    "sanitized": False
                }
            ],
            "summary": "Test summary",
            "sources": [],
            "created_at": "2024-01-01T00:00:00"
        }

        explanation = PlanExplanation.from_dict(exp_dict)

        assert explanation.plan_id == "plan_456"
        assert len(explanation.blocks) == 1
        assert explanation.blocks[0].id == "exp_1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
