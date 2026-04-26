"""Tests for the Root Cause Analysis module (vetinari.validation.root_cause)."""

from __future__ import annotations

import pytest

from vetinari.validation import (
    DefectCategory,
    RootCauseAnalysis,
    RootCauseAnalyzer,
)


class TestRootCauseAnalyzer:
    """Tests for RootCauseAnalyzer.analyze() and its heuristic helpers."""

    @pytest.fixture
    def analyzer(self) -> RootCauseAnalyzer:
        """Return a fresh RootCauseAnalyzer instance."""
        return RootCauseAnalyzer()

    # ── Happy-path category detections ────────────────────────────────────

    def test_hallucination_detection(self, analyzer: RootCauseAnalyzer) -> None:
        """Reasons mentioning 'import not found' should classify as HALLUCINATION."""
        result = analyzer.analyze(
            task_description="Write a parser module",
            rejection_reasons=["import not found: vetinari.nonexistent"],
            quality_score=0.5,
        )
        assert result.category is DefectCategory.HALLUCINATION

    def test_hallucination_detected_for_fabricated(self, analyzer: RootCauseAnalyzer) -> None:
        """Reasons containing 'fabricated' should also classify as HALLUCINATION."""
        result = analyzer.analyze(
            task_description="Summarise docs",
            rejection_reasons=["Agent fabricated API endpoints that do not exist"],
            quality_score=0.6,
        )
        assert result.category is DefectCategory.HALLUCINATION

    def test_bad_spec_detection(self, analyzer: RootCauseAnalyzer) -> None:
        """Reasons mentioning 'ambiguous' should classify as BAD_SPEC."""
        result = analyzer.analyze(
            task_description="Build feature X",
            rejection_reasons=["Requirements are ambiguous — no acceptance criteria defined"],
            quality_score=0.6,
        )
        assert result.category is DefectCategory.BAD_SPEC

    def test_bad_spec_detected_for_incomplete(self, analyzer: RootCauseAnalyzer) -> None:
        """Reasons containing 'incomplete' spec should classify as BAD_SPEC."""
        result = analyzer.analyze(
            task_description="Implement auth",
            rejection_reasons=["spec is incomplete — missing error handling requirements"],
            quality_score=0.55,
        )
        assert result.category is DefectCategory.BAD_SPEC

    def test_wrong_model_detection(self, analyzer: RootCauseAnalyzer) -> None:
        """Low score combined with 'capability' mention should classify as WRONG_MODEL."""
        result = analyzer.analyze(
            task_description="Generate 3D mesh",
            rejection_reasons=["model lacks capability for 3D generation"],
            quality_score=0.2,
        )
        assert result.category is DefectCategory.WRONG_MODEL

    def test_low_score_alone_does_not_classify_as_wrong_model(self, analyzer: RootCauseAnalyzer) -> None:
        """A critically low score without capability keywords must NOT classify as WRONG_MODEL.

        Low scores can reflect hallucination, bad spec, or complexity issues —
        WRONG_MODEL requires an explicit keyword signal in the rejection reasons.
        Without a keyword hit, the analyzer falls through to PROMPT_WEAKNESS.
        """
        result = analyzer.analyze(
            task_description="Complex reasoning task",
            rejection_reasons=["Output is mostly incoherent"],
            quality_score=0.15,
        )
        # No capability keyword — must NOT be WRONG_MODEL (falls through to PROMPT_WEAKNESS)
        assert result.category is not DefectCategory.WRONG_MODEL

    def test_insufficient_context(self, analyzer: RootCauseAnalyzer) -> None:
        """Reasons mentioning 'missing information' should classify as INSUFFICIENT_CONTEXT."""
        result = analyzer.analyze(
            task_description="Write migration script",
            rejection_reasons=["missing information about target schema"],
            quality_score=0.5,
        )
        assert result.category is DefectCategory.INSUFFICIENT_CONTEXT

    def test_insufficient_context_not_provided(self, analyzer: RootCauseAnalyzer) -> None:
        """Reasons containing 'not provided' should classify as INSUFFICIENT_CONTEXT."""
        result = analyzer.analyze(
            task_description="Generate report",
            rejection_reasons=["data source credentials not provided"],
            quality_score=0.55,
        )
        assert result.category is DefectCategory.INSUFFICIENT_CONTEXT

    def test_integration_error(self, analyzer: RootCauseAnalyzer) -> None:
        """Reasons mentioning 'breaks integration' should classify as INTEGRATION_ERROR."""
        result = analyzer.analyze(
            task_description="Add new endpoint",
            rejection_reasons=["new endpoint breaks integration with auth middleware"],
            quality_score=0.7,
        )
        assert result.category is DefectCategory.INTEGRATION_ERROR

    def test_integration_error_incompatible(self, analyzer: RootCauseAnalyzer) -> None:
        """Reasons containing 'incompatible' should classify as INTEGRATION_ERROR."""
        result = analyzer.analyze(
            task_description="Update data model",
            rejection_reasons=["schema change is incompatible with downstream consumers"],
            quality_score=0.65,
        )
        assert result.category is DefectCategory.INTEGRATION_ERROR

    def test_complexity_underestimate(self, analyzer: RootCauseAnalyzer) -> None:
        """Reasons mentioning 'too complex' should classify as COMPLEXITY_UNDERESTIMATE."""
        result = analyzer.analyze(
            task_description="Implement caching layer",
            rejection_reasons=["task is too complex — needs to be split into subtasks"],
            quality_score=0.4,
        )
        assert result.category is DefectCategory.COMPLEXITY_UNDERESTIMATE

    def test_complexity_detected_for_decompose(self, analyzer: RootCauseAnalyzer) -> None:
        """Reasons containing 'decompose' should classify as COMPLEXITY_UNDERESTIMATE."""
        result = analyzer.analyze(
            task_description="Refactor entire module",
            rejection_reasons=["output is too large — please decompose into smaller tasks"],
            quality_score=0.45,
        )
        assert result.category is DefectCategory.COMPLEXITY_UNDERESTIMATE

    def test_fallback_prompt_weakness(self, analyzer: RootCauseAnalyzer) -> None:
        """No keyword matches with decent score should fall back to PROMPT_WEAKNESS."""
        result = analyzer.analyze(
            task_description="Write a blog post",
            rejection_reasons=["output did not meet expectations"],
            quality_score=0.55,
        )
        assert result.category is DefectCategory.PROMPT_WEAKNESS

    def test_fallback_prompt_weakness_low_confidence(self, analyzer: RootCauseAnalyzer) -> None:
        """Fallback PROMPT_WEAKNESS confidence should be lower than primary detections."""
        result = analyzer.analyze(
            task_description="Write a blog post",
            rejection_reasons=["output did not meet expectations"],
            quality_score=0.55,
        )
        assert result.confidence < 0.75, "Fallback confidence should be low"

    # ── Structural invariant tests ─────────────────────────────────────────

    def test_confidence_ranges_hallucination(self, analyzer: RootCauseAnalyzer) -> None:
        """Confidence for HALLUCINATION must be in [0.0, 1.0]."""
        result = analyzer.analyze("task", ["import not found"], 0.5)
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_ranges_bad_spec(self, analyzer: RootCauseAnalyzer) -> None:
        """Confidence for BAD_SPEC must be in [0.0, 1.0]."""
        result = analyzer.analyze("task", ["ambiguous requirements"], 0.6)
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_ranges_wrong_model(self, analyzer: RootCauseAnalyzer) -> None:
        """Confidence for WRONG_MODEL must be in [0.0, 1.0]."""
        result = analyzer.analyze("task", ["output is poor"], 0.1)
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_ranges_insufficient_context(self, analyzer: RootCauseAnalyzer) -> None:
        """Confidence for INSUFFICIENT_CONTEXT must be in [0.0, 1.0]."""
        result = analyzer.analyze("task", ["missing information about domain"], 0.5)
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_ranges_integration_error(self, analyzer: RootCauseAnalyzer) -> None:
        """Confidence for INTEGRATION_ERROR must be in [0.0, 1.0]."""
        result = analyzer.analyze("task", ["breaks downstream service"], 0.6)
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_ranges_complexity(self, analyzer: RootCauseAnalyzer) -> None:
        """Confidence for COMPLEXITY_UNDERESTIMATE must be in [0.0, 1.0]."""
        result = analyzer.analyze("task", ["too complex to fit in context"], 0.4)
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_ranges_fallback(self, analyzer: RootCauseAnalyzer) -> None:
        """Confidence for PROMPT_WEAKNESS fallback must be in [0.0, 1.0]."""
        result = analyzer.analyze("task", ["did not meet expectations"], 0.55)
        assert 0.0 <= result.confidence <= 1.0

    def test_corrective_action_not_empty(self, analyzer: RootCauseAnalyzer) -> None:
        """corrective_action must always be a non-empty string."""
        cases = [
            (["import not found"], 0.5),
            (["ambiguous spec"], 0.6),
            (["poor output"], 0.1),
            (["missing information"], 0.5),
            (["breaks integration"], 0.6),
            (["too complex"], 0.4),
            (["did not meet expectations"], 0.55),
        ]
        for reasons, score in cases:
            result = analyzer.analyze("task", reasons, score)
            assert result.corrective_action, f"corrective_action empty for reasons={reasons}"

    def test_preventive_action_not_empty(self, analyzer: RootCauseAnalyzer) -> None:
        """preventive_action must always be a non-empty string."""
        cases = [
            (["import not found"], 0.5),
            (["ambiguous spec"], 0.6),
            (["poor output"], 0.1),
            (["missing information"], 0.5),
            (["breaks integration"], 0.6),
            (["too complex"], 0.4),
            (["did not meet expectations"], 0.55),
        ]
        for reasons, score in cases:
            result = analyzer.analyze("task", reasons, score)
            assert result.preventive_action, f"preventive_action empty for reasons={reasons}"

    def test_evidence_populated(self, analyzer: RootCauseAnalyzer) -> None:
        """evidence list must always be non-empty."""
        cases = [
            (["import not found"], 0.5),
            (["ambiguous spec"], 0.6),
            (["poor output"], 0.1),
            (["missing information"], 0.5),
            (["breaks integration"], 0.6),
            (["too complex"], 0.4),
            (["did not meet expectations"], 0.55),
        ]
        for reasons, score in cases:
            result = analyzer.analyze("task", reasons, score)
            assert result.evidence, f"evidence empty for reasons={reasons}"

    def test_evidence_populated_with_empty_reasons(self, analyzer: RootCauseAnalyzer) -> None:
        """evidence must be non-empty even when no rejection reasons are supplied."""
        result = analyzer.analyze("task", [], 0.55)
        assert result.evidence

    def test_result_type(self, analyzer: RootCauseAnalyzer) -> None:
        """analyze() must always return a RootCauseAnalysis instance."""
        result = analyzer.analyze("task", ["ambiguous"], 0.6)
        assert isinstance(result, RootCauseAnalysis)

    def test_task_mode_accepted(self, analyzer: RootCauseAnalyzer) -> None:
        """analyze() must accept the optional task_mode parameter without error."""
        result = analyzer.analyze(
            task_description="task",
            rejection_reasons=["ambiguous requirements"],
            quality_score=0.6,
            task_mode="planning",
        )
        assert result.category is DefectCategory.BAD_SPEC


class TestCausalGraph:
    """Tests for CausalGraph — single-failure root cause fix (Defect 30)."""

    def test_single_failure_without_caused_by_is_its_own_root_cause(self) -> None:
        """A failure with no caused_by must appear in get_root_causes().

        Before the fix, a single standalone failure created no edges, so
        get_root_causes() returned [] — the failure was silently dropped.
        """
        from vetinari.validation import CausalGraph

        graph = CausalGraph()
        graph.build_from_failures([{"id": "failure-A"}])
        roots = graph.get_root_causes()
        assert "failure-A" in roots, (
            f"Single standalone failure must be its own root cause, got: {roots!r}"
        )

    def test_chained_failures_root_cause_is_upstream(self) -> None:
        """With A -> B, the root cause is A (the upstream node)."""
        from vetinari.validation import CausalGraph

        graph = CausalGraph()
        graph.build_from_failures([
            {"id": "failure-A"},
            {"id": "failure-B", "caused_by": "failure-A"},
        ])
        roots = graph.get_root_causes()
        assert "failure-A" in roots
        assert "failure-B" not in roots

    def test_multiple_isolated_failures_all_appear_as_root_causes(self) -> None:
        """Two unrelated failures with no caused_by are both root causes."""
        from vetinari.validation import CausalGraph

        graph = CausalGraph()
        graph.build_from_failures([{"id": "X"}, {"id": "Y"}])
        roots = graph.get_root_causes()
        assert "X" in roots
        assert "Y" in roots

    def test_failure_with_empty_id_skipped(self) -> None:
        """Failure records with empty or missing id must be skipped gracefully."""
        from vetinari.validation import CausalGraph

        graph = CausalGraph()
        graph.build_from_failures([{"id": ""}, {}])
        # No error, no phantom root causes
        assert graph.get_root_causes() == []

    def test_get_root_causes_returns_sorted_list(self) -> None:
        """get_root_causes() output must be deterministically sorted."""
        from vetinari.validation import CausalGraph

        graph = CausalGraph()
        graph.build_from_failures([{"id": "z-failure"}, {"id": "a-failure"}])
        roots = graph.get_root_causes()
        assert roots == sorted(roots)
