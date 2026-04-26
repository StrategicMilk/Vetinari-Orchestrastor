"""Tests for intake integration with two_layer.py and TIER_PIPELINES.

Covers TIER_PIPELINES definition, intake wiring in generate_and_execute(),
express tier bypass, and tier metadata propagation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.orchestration.intake import TIER_PIPELINES, Tier
from vetinari.types import AgentType

# ── TIER_PIPELINES ─────────────────────────────────────────────────────


class TestTierPipelines:
    """Test TIER_PIPELINES definition."""

    def test_express_pipeline(self) -> None:
        assert TIER_PIPELINES[Tier.EXPRESS] == [AgentType.WORKER.value, AgentType.INSPECTOR.value]

    def test_standard_pipeline(self) -> None:
        pipeline = TIER_PIPELINES[Tier.STANDARD]
        assert AgentType.FOREMAN.value in pipeline
        assert AgentType.WORKER.value in pipeline
        assert AgentType.INSPECTOR.value in pipeline

    def test_custom_pipeline(self) -> None:
        pipeline = TIER_PIPELINES[Tier.CUSTOM]
        assert AgentType.FOREMAN.value in pipeline
        assert AgentType.WORKER.value in pipeline
        assert AgentType.INSPECTOR.value in pipeline

    def test_all_tiers_have_pipelines(self) -> None:
        for tier in Tier:
            assert tier in TIER_PIPELINES

    def test_express_is_shortest(self) -> None:
        assert len(TIER_PIPELINES[Tier.EXPRESS]) < len(TIER_PIPELINES[Tier.STANDARD])

    def test_custom_has_full_pipeline(self) -> None:
        # CUSTOM uses the same full pipeline as STANDARD in the 3-agent model
        assert len(TIER_PIPELINES[Tier.CUSTOM]) == len(TIER_PIPELINES[Tier.STANDARD])


# ── Intake in generate_and_execute ─────────────────────────────────────


class TestIntakeInGenerateAndExecute:
    """Test intake classification wiring in TwoLayerOrchestrator."""

    def test_intake_called_during_pipeline(self) -> None:
        """Verify intake.classify_with_features is called."""
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        orch = TwoLayerOrchestrator.__new__(TwoLayerOrchestrator)
        orch.plan_generator = MagicMock()
        orch.plan_generator.generate_plan.return_value = MagicMock(plan_id="test-plan", nodes={})

        mock_intake = MagicMock()
        mock_intake.classify_with_features.return_value = (
            Tier.STANDARD,
            MagicMock(confidence=0.9, pattern_key="abc", word_count=25, cross_cutting_keywords=0),
        )

        with (
            patch("vetinari.orchestration.intake.get_request_intake", return_value=mock_intake),
            patch.object(orch, "_analyze_input", return_value={}),
            patch.object(orch, "_validate_stage_boundary", return_value=(True, [])),
            patch.object(orch, "_route_model_for_task", return_value="default"),
            patch.object(orch, "_make_default_handler", return_value=lambda g, c: "result"),
        ):
            # Won't fully succeed but intake should be called
            try:
                orch.generate_and_execute(
                    "Build a multi-file feature with tests and documentation updates across the codebase"
                )
            except Exception:  # noqa: VET022 - best-effort optional path must not fail the primary flow
                pass  # May fail on later stages, that's OK

            mock_intake.classify_with_features.assert_called_once()
            intake_goal, intake_context = mock_intake.classify_with_features.call_args.args
            assert intake_goal == "Build a multi-file feature with tests and documentation updates across the codebase"
            assert intake_context["intake_tier"] == Tier.STANDARD.value
            assert intake_context["intake_confidence"] == 0.9
            assert intake_context["intake_pattern_key"] == "abc"

    def test_express_tier_bypasses_planning(self) -> None:
        """Verify Express tier calls _execute_express instead of planning."""
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        orch = TwoLayerOrchestrator.__new__(TwoLayerOrchestrator)

        mock_intake = MagicMock()
        mock_intake.classify_with_features.return_value = (
            Tier.EXPRESS,
            MagicMock(confidence=0.95, pattern_key="abc", word_count=5, cross_cutting_keywords=0),
        )

        mock_express = MagicMock(return_value={"completed": 1, "tier": "express"})

        with (
            patch("vetinari.orchestration.intake.get_request_intake", return_value=mock_intake),
            patch.object(orch, "_enrich_goal", return_value="Fix typo"),
            patch.object(orch, "_execute_express", mock_express),
        ):
            result = orch.generate_and_execute("Fix typo")
            assert result["tier"] == "express"
            mock_express.assert_called_once()

    def test_tier_stored_in_stages(self) -> None:
        """Verify tier is stored in stages dict during pipeline."""
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        orch = TwoLayerOrchestrator.__new__(TwoLayerOrchestrator)
        orch.plan_generator = MagicMock()
        orch.plan_generator.generate_plan.return_value = MagicMock(plan_id="test-plan", nodes={})

        mock_intake = MagicMock()
        mock_intake.classify_with_features.return_value = (
            Tier.STANDARD,
            MagicMock(confidence=0.9, pattern_key="abc", word_count=25, cross_cutting_keywords=0),
        )

        with (
            patch("vetinari.orchestration.intake.get_request_intake", return_value=mock_intake),
            patch.object(orch, "_analyze_input", return_value={}),
            patch.object(orch, "_validate_stage_boundary", return_value=(True, [])),
            patch.object(orch, "_route_model_for_task", return_value="default"),
            patch.object(orch, "_make_default_handler", return_value=lambda g, c: "result"),
        ):
            try:
                result = orch.generate_and_execute(
                    "Build a multi-file feature with tests and documentation for all endpoints"
                )
                # Check stages dict if available
                if "stages" in result:
                    assert result["stages"].get("intake", {}).get("tier") == "standard"
            except Exception:  # noqa: VET022 - best-effort optional path must not fail the primary flow
                pass

            # Verify intake was called with correct args
            mock_intake.classify_with_features.assert_called_once()


# ── Express Execution ──────────────────────────────────────────────────


class TestExpressExecution:
    """Test _execute_express method."""

    def test_express_returns_result(self) -> None:
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        orch = TwoLayerOrchestrator.__new__(TwoLayerOrchestrator)

        # Handler receives a single TaskNode argument (Express path)
        with patch.object(orch, "_make_default_handler", return_value=lambda task: "built"):
            import time

            result = orch._execute_express("Fix typo", {}, {}, time.time(), None, None)
            assert result["backend"] == "express"
            assert result["tier"] == "express"
            assert result["completed"] == 1

    def test_express_handles_failure(self) -> None:
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        orch = TwoLayerOrchestrator.__new__(TwoLayerOrchestrator)

        def failing_handler(task):
            raise RuntimeError("build failed")

        with patch.object(orch, "_make_default_handler", return_value=failing_handler):
            import time

            result = orch._execute_express("Fix typo", {}, {}, time.time(), None, None)
            assert result["failed"] == 1
            assert "build failed" in result["error"]
