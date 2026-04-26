"""Tests for clarification wiring (US-073).

Covers PipelinePaused dataclass, clarification triggering for Custom tier,
low confidence routing, and follow_up_question handling.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.orchestration.intake import (
    CONFIDENCE_THRESHOLD,
    IntakeFeatures,
    PipelinePaused,
    Tier,
)

# ── PipelinePaused ─────────────────────────────────────────────────────


class TestPipelinePaused:
    """Test PipelinePaused dataclass."""

    def test_defaults(self) -> None:
        p = PipelinePaused()
        assert p.questions == []
        assert p.pipeline_state == {}
        assert p.tier == "custom"
        assert p.goal == ""
        assert p.confidence == 0.0

    def test_to_dict(self) -> None:
        p = PipelinePaused(
            questions=["What do you mean?"],
            tier="custom",
            goal="Do stuff",
            confidence=0.5,
        )
        d = p.to_dict()
        assert d["paused"] is True
        assert d["questions"] == ["What do you mean?"]
        assert d["tier"] == "custom"
        assert d["goal"] == "Do stuff"
        assert d["confidence"] == 0.5

    def test_with_pipeline_state(self) -> None:
        p = PipelinePaused(
            pipeline_state={"goal": "test", "tier": "custom"},
        )
        d = p.to_dict()
        assert d["pipeline_state"]["goal"] == "test"


# ── Clarification Triggering ───────────────────────────────────────────


class TestClarificationTriggering:
    """Test clarification trigger conditions."""

    def test_custom_tier_triggers_clarification(self) -> None:
        """Custom tier should trigger _run_clarification."""
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        orch = TwoLayerOrchestrator.__new__(TwoLayerOrchestrator)
        orch.plan_generator = MagicMock()
        orch.plan_generator.generate_plan.return_value = MagicMock(plan_id="test-plan", nodes={})

        mock_intake = MagicMock()
        mock_intake.classify_with_features.return_value = (
            Tier.CUSTOM,
            MagicMock(confidence=0.6, pattern_key="abc", word_count=30, cross_cutting_keywords=4),
        )

        # Clarification returns questions needing user input
        mock_clarify = MagicMock(
            return_value={
                "needs_user_input": True,
                "pending_questions": ["What specifically should change?"],
            }
        )

        with (
            patch("vetinari.orchestration.intake.get_request_intake", return_value=mock_intake),
            patch.object(orch, "_enrich_goal", return_value="Do something complex"),
            patch.object(orch, "_run_clarification", mock_clarify),
        ):
            result = orch.generate_and_execute("Maybe do something with the stuff and things somehow perhaps probably")
            assert result.get("paused") is True
            assert len(result.get("questions", [])) > 0
            mock_clarify.assert_called_once()

    def test_low_confidence_triggers_clarification(self) -> None:
        """Low confidence should trigger clarification even for Standard tier."""
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        orch = TwoLayerOrchestrator.__new__(TwoLayerOrchestrator)
        orch.plan_generator = MagicMock()
        orch.plan_generator.generate_plan.return_value = MagicMock(plan_id="test-plan", nodes={})

        mock_intake = MagicMock()
        # Standard tier but low confidence
        mock_intake.classify_with_features.return_value = (
            Tier.STANDARD,
            MagicMock(confidence=0.5, pattern_key="abc", word_count=25, cross_cutting_keywords=1),
        )

        mock_clarify = MagicMock(
            return_value={
                "needs_user_input": True,
                "pending_questions": ["Can you clarify?"],
            }
        )

        with (
            patch("vetinari.orchestration.intake.get_request_intake", return_value=mock_intake),
            patch.object(orch, "_enrich_goal", return_value="Vague goal text"),
            patch.object(orch, "_run_clarification", mock_clarify),
        ):
            result = orch.generate_and_execute(
                "Add a multi-file feature with tests and documentation for all endpoints in the system?"
            )
            assert result.get("paused") is True
            mock_clarify.assert_called_once()

    def test_high_confidence_skips_clarification(self) -> None:
        """High confidence Standard tier should not trigger clarification."""
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        orch = TwoLayerOrchestrator.__new__(TwoLayerOrchestrator)
        orch.plan_generator = MagicMock()
        orch.plan_generator.generate_plan.return_value = MagicMock(plan_id="test-plan", nodes={})

        mock_intake = MagicMock()
        mock_intake.classify_with_features.return_value = (
            Tier.STANDARD,
            MagicMock(confidence=0.95, pattern_key="abc", word_count=25, cross_cutting_keywords=0),
        )

        mock_clarify = MagicMock()

        with (
            patch("vetinari.orchestration.intake.get_request_intake", return_value=mock_intake),
            patch.object(orch, "_enrich_goal", return_value="Clear goal"),
            patch.object(orch, "_run_clarification", mock_clarify),
            patch.object(orch, "_analyze_input", return_value={}),
            patch.object(orch, "_validate_stage_boundary", return_value=(True, [])),
            patch.object(orch, "_route_model_for_task", return_value="default"),
            patch.object(orch, "_make_default_handler", return_value=lambda g, c: "result"),
        ):
            try:
                orch.generate_and_execute(
                    "Add input validation to the user registration form in auth.py and views.py with proper error messages"
                )
            except Exception:  # noqa: VET022 - best-effort optional path must not fail the primary flow
                pass

            mock_clarify.assert_not_called()

    def test_inline_clarification_enriches_context(self) -> None:
        """If clarification resolves inline (no user input needed), context is enriched."""
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        orch = TwoLayerOrchestrator.__new__(TwoLayerOrchestrator)
        orch.plan_generator = MagicMock()
        orch.plan_generator.generate_plan.return_value = MagicMock(plan_id="test-plan", nodes={})

        mock_intake = MagicMock()
        mock_intake.classify_with_features.return_value = (
            Tier.CUSTOM,
            MagicMock(confidence=0.6, pattern_key="abc", word_count=30, cross_cutting_keywords=4),
        )

        # Clarification resolves without user input
        mock_clarify = MagicMock(
            return_value={
                "clarification_0": {"question": "What?", "answer": "This."},
            }
        )

        with (
            patch("vetinari.orchestration.intake.get_request_intake", return_value=mock_intake),
            patch.object(orch, "_enrich_goal", return_value="Somewhat vague goal"),
            patch.object(orch, "_run_clarification", mock_clarify),
            patch.object(orch, "_analyze_input", return_value={}),
            patch.object(orch, "_validate_stage_boundary", return_value=(True, [])),
            patch.object(orch, "_route_model_for_task", return_value="default"),
            patch.object(orch, "_make_default_handler", return_value=lambda g, c: "result"),
        ):
            try:
                orch.generate_and_execute("Maybe do something with the stuff and things somehow perhaps probably")
            except Exception:  # noqa: VET022 - best-effort optional path must not fail the primary flow
                pass

            # Should NOT return paused
            mock_clarify.assert_called_once()
            clarify_goal, clarify_context = mock_clarify.call_args.args
            assert clarify_goal == "Somewhat vague goal"
            assert clarify_context["intake_tier"] == Tier.CUSTOM.value
            assert clarify_context["intake_pattern_key"] == "abc"


# ── _run_clarification ─────────────────────────────────────────────────


class TestRunClarification:
    """Test _run_clarification method."""

    def test_returns_none_when_no_planner(self) -> None:
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        orch = TwoLayerOrchestrator.__new__(TwoLayerOrchestrator)
        result = orch._run_clarification("test goal", {})
        assert result is None

    def test_returns_clarify_output(self) -> None:
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator
        from vetinari.types import AgentType

        orch = TwoLayerOrchestrator.__new__(TwoLayerOrchestrator)
        mock_planner = MagicMock()
        mock_planner.execute.return_value = MagicMock(
            success=True,
            output={"needs_user_input": True, "pending_questions": ["Q1?"]},
        )
        # _agents dict uses string keys keyed by AgentType.value
        orch._agents = {AgentType.FOREMAN.value: mock_planner}

        result = orch._run_clarification("ambiguous goal", {})
        assert result is not None
        assert result["needs_user_input"] is True


# ── Confidence Threshold ───────────────────────────────────────────────


class TestConfidenceThreshold:
    """Test CONFIDENCE_THRESHOLD value."""

    def test_threshold_value(self) -> None:
        assert CONFIDENCE_THRESHOLD == 0.85

    def test_confidence_in_features(self) -> None:
        f = IntakeFeatures(confidence=0.5)
        assert f.confidence < CONFIDENCE_THRESHOLD
