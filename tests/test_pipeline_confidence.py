"""Tests for vetinari.orchestration.pipeline_confidence — confidence-based output routing."""

from __future__ import annotations

from vetinari.awareness.confidence import ConfidenceResult, UnknownSituation
from vetinari.orchestration.pipeline_confidence import apply_confidence_routing
from vetinari.types import ConfidenceAction, ConfidenceLevel


def _make_result(
    level: ConfidenceLevel,
    action: ConfidenceAction,
    score: float = -1.0,
) -> ConfidenceResult:
    """Build a minimal ConfidenceResult for routing tests."""
    return ConfidenceResult(
        score=score,
        level=level,
        action=action,
        explanation=f"{level.value} confidence",
    )


# -- PROCEED ------------------------------------------------------------------


class TestProceed:
    """PROCEED action passes output and confidence through unchanged."""

    def test_proceed_returns_original_output(self) -> None:
        """PROCEED action returns the output and confidence unchanged."""
        confidence = _make_result(ConfidenceLevel.HIGH, ConfidenceAction.PROCEED, score=-0.2)
        output, result_confidence = apply_confidence_routing("my output", confidence)
        assert output == "my output"
        assert result_confidence is confidence

    def test_proceed_ignores_provided_callbacks(self) -> None:
        """PROCEED doesn't call refine_fn or sample_fn even if provided."""
        confidence = _make_result(ConfidenceLevel.HIGH, ConfidenceAction.PROCEED)
        called = []

        def refine(text: str) -> str:
            called.append("refine")
            return text

        apply_confidence_routing("output", confidence, refine_fn=refine)
        assert "refine" not in called


# -- REFINE -------------------------------------------------------------------


class TestRefine:
    """REFINE action triggers self-refinement callback."""

    def test_refine_calls_refine_fn(self) -> None:
        """REFINE invokes refine_fn and returns its output."""
        confidence = _make_result(ConfidenceLevel.MEDIUM, ConfidenceAction.REFINE, score=-1.0)
        refined_output, new_confidence = apply_confidence_routing(
            "original",
            confidence,
            refine_fn=lambda text: "refined: " + text,
        )
        assert refined_output == "refined: original"
        assert new_confidence.action == ConfidenceAction.PROCEED
        assert "refinement" in new_confidence.source

    def test_refine_without_fn_passes_through(self) -> None:
        """REFINE with no refine_fn returns original output and confidence unchanged."""
        confidence = _make_result(ConfidenceLevel.MEDIUM, ConfidenceAction.REFINE)
        output, result = apply_confidence_routing("original", confidence)
        assert output == "original"
        assert result is confidence

    def test_refine_upgrades_confidence_level(self) -> None:
        """After refinement, confidence level is upgraded (LOW → MEDIUM, MEDIUM → HIGH)."""
        medium_confidence = _make_result(ConfidenceLevel.MEDIUM, ConfidenceAction.REFINE)
        _, after_medium = apply_confidence_routing("x", medium_confidence, refine_fn=lambda t: t)
        assert after_medium.level == ConfidenceLevel.HIGH

        low_confidence = _make_result(ConfidenceLevel.LOW, ConfidenceAction.REFINE)
        _, after_low = apply_confidence_routing("x", low_confidence, refine_fn=lambda t: t)
        assert after_low.level == ConfidenceLevel.MEDIUM

    def test_refine_records_pre_refinement_level(self) -> None:
        """Metadata contains pre_refinement_level from original confidence."""
        confidence = _make_result(ConfidenceLevel.MEDIUM, ConfidenceAction.REFINE)
        _, new_confidence = apply_confidence_routing("x", confidence, refine_fn=lambda t: t)
        assert new_confidence.metadata.get("pre_refinement_level") == ConfidenceLevel.MEDIUM.value


# -- BEST_OF_N ----------------------------------------------------------------


class TestBestOfN:
    """BEST_OF_N action samples candidates and picks best by score, not length."""

    def test_best_of_n_calls_sample_fn(self) -> None:
        """BEST_OF_N invokes sample_fn."""
        confidence = _make_result(ConfidenceLevel.LOW, ConfidenceAction.BEST_OF_N, score=-2.5)
        sample_called = []

        def sample() -> list[tuple[str, list[float]]]:
            sample_called.append(True)
            return [("candidate", [-0.2, -0.3])]

        apply_confidence_routing("original", confidence, sample_fn=sample)
        assert sample_called

    def test_best_of_n_picks_by_highest_score_not_length(self) -> None:
        """BEST_OF_N selects the candidate with highest logprob score, not longest text."""
        confidence = _make_result(ConfidenceLevel.LOW, ConfidenceAction.BEST_OF_N, score=-2.5)

        # Short output with HIGH confidence vs long output with LOW confidence
        def sample() -> list[tuple[str, list[float]]]:
            return [
                ("short but good", [-0.2, -0.1]),  # HIGH confidence
                ("this is a very long output that goes on and on", [-4.0, -5.0, -6.0]),  # LOW confidence
            ]

        output, new_confidence = apply_confidence_routing("original", confidence, sample_fn=sample)
        # Should pick "short but good" (higher score), not the longer low-confidence output
        assert output == "short but good"
        assert new_confidence.level == ConfidenceLevel.HIGH

    def test_best_of_n_without_fn_passes_through(self) -> None:
        """BEST_OF_N with no sample_fn returns original output and confidence."""
        confidence = _make_result(ConfidenceLevel.LOW, ConfidenceAction.BEST_OF_N)
        output, result = apply_confidence_routing("original", confidence)
        assert output == "original"
        assert result is confidence

    def test_best_of_n_empty_candidates_uses_original(self) -> None:
        """BEST_OF_N with empty candidate list falls back to original output."""
        confidence = _make_result(ConfidenceLevel.LOW, ConfidenceAction.BEST_OF_N, score=-2.5)
        output, result = apply_confidence_routing("original", confidence, sample_fn=list)
        assert output == "original"
        assert result is confidence


# -- DEFER_TO_HUMAN -----------------------------------------------------------


class TestDeferToHuman:
    """DEFER_TO_HUMAN action escalates to human via defer_fn callback."""

    def test_defer_calls_defer_fn(self) -> None:
        """DEFER_TO_HUMAN invokes defer_fn with output and confidence."""
        confidence = _make_result(ConfidenceLevel.VERY_LOW, ConfidenceAction.DEFER_TO_HUMAN, score=-5.0)
        received = []

        def defer(text: str, conf: ConfidenceResult) -> str | None:
            received.append((text, conf))
            return "human approved output"

        apply_confidence_routing("original", confidence, defer_fn=defer)
        assert len(received) == 1
        assert received[0][0] == "original"
        assert received[0][1] is confidence

    def test_defer_with_approval_returns_high_confidence(self) -> None:
        """When defer_fn returns approved output, confidence is set to HIGH."""
        confidence = _make_result(ConfidenceLevel.VERY_LOW, ConfidenceAction.DEFER_TO_HUMAN)
        output, new_confidence = apply_confidence_routing(
            "original",
            confidence,
            defer_fn=lambda t, c: "approved by human",
        )
        assert output == "approved by human"
        assert new_confidence.level == ConfidenceLevel.HIGH
        assert new_confidence.action == ConfidenceAction.PROCEED
        assert new_confidence.score == 1.0

    def test_defer_with_none_marks_deferred(self) -> None:
        """When defer_fn returns None, output is marked as deferred to queue."""
        confidence = _make_result(ConfidenceLevel.VERY_LOW, ConfidenceAction.DEFER_TO_HUMAN, score=-5.0)
        output, new_confidence = apply_confidence_routing(
            "original",
            confidence,
            defer_fn=lambda t, c: None,
        )
        assert output == "original"
        assert new_confidence.action == ConfidenceAction.DEFER_TO_HUMAN
        assert new_confidence.metadata.get("deferred") is True

    def test_defer_without_fn_passes_through(self) -> None:
        """DEFER_TO_HUMAN with no defer_fn returns original output and confidence."""
        confidence = _make_result(ConfidenceLevel.VERY_LOW, ConfidenceAction.DEFER_TO_HUMAN)
        output, result = apply_confidence_routing("original", confidence)
        assert output == "original"
        assert result is confidence

    def test_defer_preserves_unknown_situation(self) -> None:
        """When deferred to queue, the unknown_situation from original confidence is preserved."""
        confidence = ConfidenceResult(
            score=-999.0,
            level=ConfidenceLevel.VERY_LOW,
            action=ConfidenceAction.DEFER_TO_HUMAN,
            explanation="no data",
            unknown_situation=UnknownSituation.NO_DATA,
        )
        _, new_confidence = apply_confidence_routing("x", confidence, defer_fn=lambda t, c: None)
        assert new_confidence.unknown_situation == UnknownSituation.NO_DATA
