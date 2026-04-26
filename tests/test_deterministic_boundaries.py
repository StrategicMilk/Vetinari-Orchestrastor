"""Tests for deterministic-vs-model boundary correctness (US-008).

Verifies that bounded mathematical tasks use deterministic logic (no model calls)
and semantic classification tasks use model-backed reasoning with confidence gating.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class TestBoundedValidationNoModelDependency(unittest.TestCase):
    """Cost analysis MUST produce correct results without any LLM call."""

    def test_cost_analysis_no_infer_call(self) -> None:
        """Verify _execute_cost_analysis uses deterministic math, not _infer_json.

        The cost analysis mode computes token estimates and model pricing using
        arithmetic. It MUST NOT call _infer_json() for this bounded computation.
        This test is the specification for the corrected behavior — it will fail
        until the general-path branch of _execute_cost_analysis is made deterministic.
        """
        from vetinari.agents.consolidated.operations_agent import OperationsAgent
        from vetinari.agents.contracts import AgentResult

        agent = OperationsAgent.__new__(OperationsAgent)
        # Ensure _infer_json is NOT called by making it raise if touched
        agent._infer_json = MagicMock(side_effect=AssertionError("_infer_json should not be called for cost analysis"))
        agent._infer = MagicMock(side_effect=AssertionError("_infer should not be called for cost analysis"))

        task = MagicMock()
        task.description = "Estimate the cost of running a 2000-word document through various models"
        task.context = {"analysis_type": "general"}  # Non-model_comparison triggers the general path

        result = agent._execute_cost_analysis(task)

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert "analysis" in result.output
        assert "recommendations" in result.output
        # Must signal deterministic source so callers can distinguish from LLM output
        assert result.output.get("source") == "deterministic"
        # _infer_json must NOT have been called
        agent._infer_json.assert_not_called()

    def test_cost_analysis_produces_valid_estimates(self) -> None:
        """Token estimates and cost calculations MUST be mathematically correct.

        The general path estimates tokens from word count using a fixed multiplier
        (word_count * 1.3) and costs from the _MODEL_PRICING table.
        Recommendations must be sorted cheapest-first.
        """
        from vetinari.agents.consolidated.operations_agent import OperationsAgent

        agent = OperationsAgent.__new__(OperationsAgent)
        agent._infer_json = MagicMock(side_effect=AssertionError("No LLM call expected"))

        task = MagicMock()
        task.description = "Hello world program in Python"
        task.context = {"analysis_type": "general"}

        result = agent._execute_cost_analysis(task)
        assert result.success is True

        # Recommendations must be sorted by ascending estimated cost
        recs = result.output.get("recommendations", [])
        if len(recs) >= 2:
            costs = [r["estimated_cost"] for r in recs]
            assert costs == sorted(costs), f"Recommendations must be sorted by ascending cost; got {costs}"

    def test_model_comparison_path_is_already_deterministic(self) -> None:
        """The model_comparison branch MUST NOT call _infer_json (it already uses arithmetic).

        This documents the existing correct boundary as a regression guard.
        """
        from vetinari.agents.consolidated.operations_agent import OperationsAgent

        agent = OperationsAgent.__new__(OperationsAgent)
        agent._infer_json = MagicMock(side_effect=AssertionError("_infer_json must not be called for model_comparison"))

        task = MagicMock()
        task.description = "Compare models for a 10k token workload"
        task.context = {
            "analysis_type": "model_comparison",
            "estimated_tokens": 10000,
            "models": ["qwen2.5-coder-7b", "qwen2.5-72b"],
        }

        result = agent._execute_cost_analysis(task)
        assert result.success is True
        assert "comparisons" in result.output
        # Sorted cheapest first
        comparisons = result.output["comparisons"]
        assert len(comparisons) == 2
        costs = [c["total_cost"] for c in comparisons]
        assert costs == sorted(costs)
        agent._infer_json.assert_not_called()


class TestSemanticPathUsesModelReasoning(unittest.TestCase):
    """Goal classification MUST use model-backed reasoning, not just keywords."""

    def test_classify_goal_detailed_tries_llm_first(self) -> None:
        """classify_goal_detailed() must attempt LLM classification before keywords.

        The function imports classify_goal_via_llm at call time from
        vetinari.llm_helpers; patching that module ensures the mock is found
        by the dynamic import inside the try block.
        """
        from vetinari.orchestration.request_routing import classify_goal_detailed

        with patch(
            "vetinari.llm_helpers.classify_goal_via_llm",
            return_value="security",
        ) as mock_llm:
            result = classify_goal_detailed("Check our application for vulnerabilities")
            mock_llm.assert_called_once()
            assert result["category"] == "security"
            assert result["source"] == "llm"
            # LLM path reports high confidence
            assert result["confidence"] >= 0.8

    def test_keyword_fallback_distinguishable_from_llm(self) -> None:
        """When LLM is unavailable, keyword fallback MUST NOT claim LLM source.

        Making classify_goal_via_llm raise forces the except branch, which
        falls through to keyword matching.  The result must be distinguishable
        from the LLM path by the absence of ``source='llm'``.
        Cross-cutting goals produce lower confidence due to hit distribution.
        """
        from vetinari.orchestration.request_routing import classify_goal_detailed

        # Use a cross-cutting goal to get realistic keyword behavior
        with patch(
            "vetinari.llm_helpers.classify_goal_via_llm",
            side_effect=RuntimeError("LLM unavailable"),
        ):
            result = classify_goal_detailed("implement a security audit tool for the database schema")
            # Keyword path should match multiple categories (code + security + data)
            assert result["category"] in ("code", "security", "data")
            # Cross-cutting goal confidence is lower because hits are distributed
            assert result["confidence"] < 0.95
            # Must NOT report llm as source when LLM was unavailable
            assert result.get("source") != "llm"
            # Should have cross-cutting categories
            assert len(result.get("cross_cutting", [])) >= 1

    def test_analyze_input_delegates_to_classifier(self) -> None:
        """PipelineHelpersMixin._analyze_input must delegate to classify_goal_detailed.

        classify_goal_detailed is imported at call time from
        vetinari.orchestration.request_routing; patching there ensures the mock
        is intercepted by the dynamic import inside the try block.
        """
        mock_result = {
            "category": "code",
            "confidence": 0.95,
            "complexity": "standard",
            "cross_cutting": ["research"],
            "matched_keywords": [],
            "source": "llm",
        }
        with patch(
            "vetinari.orchestration.request_routing.classify_goal_detailed",
            return_value=mock_result,
        ) as mock_classify:
            from vetinari.orchestration.pipeline_helpers import PipelineHelpersMixin

            helper = PipelineHelpersMixin.__new__(PipelineHelpersMixin)
            result = helper._analyze_input("Build a REST API with authentication", {})

            mock_classify.assert_called_once()
            assert result["needs_code"] is True
            assert result["domain"] == "coding"
            assert result.get("classification_confidence", 0) > 0.5

    def test_paraphrased_goal_classified_correctly_via_llm(self) -> None:
        """Goals using synonyms/paraphrases MUST classify correctly through the LLM path.

        "Write a test suite" contains "write" which the creative keyword list also
        matches. With LLM classification active the correct category is "code".
        This test verifies the LLM path overrides naive keyword matching.
        """
        from vetinari.orchestration.request_routing import classify_goal_detailed

        with patch(
            "vetinari.llm_helpers.classify_goal_via_llm",
            return_value="code",
        ):
            result = classify_goal_detailed("Write a comprehensive test suite for the auth module")
            assert result["category"] == "code"
            assert result["source"] == "llm"

    def test_no_source_in_keyword_fallback_result(self) -> None:
        """Keyword fallback result must NOT include source='llm'.

        When both LLM and TaskClassifier are unavailable, the result comes from
        keyword matching and must be clearly distinguishable from LLM output.
        """
        from vetinari.orchestration.request_routing import classify_goal_detailed

        with patch(
            "vetinari.llm_helpers.classify_goal_via_llm",
            side_effect=Exception("all LLM paths unavailable"),
        ):
            result = classify_goal_detailed("deploy the app to kubernetes")
            # Should fall through to keyword path
            assert result["category"] in ("devops", "general")
            # Must not claim to be LLM-sourced
            assert result.get("source") != "llm"


if __name__ == "__main__":
    unittest.main()
