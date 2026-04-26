"""Tests for Worker self-reflection loop (Session 23, US-004).

Covers the self-reflection module (strategy resolution, draft-refine loop,
tree-of-thought candidate selection, kaizen reporting) and the wiring into
MultiModeAgent.execute().
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from tests.factories import make_agent_result, make_agent_task
from vetinari.agents.contracts import AgentResult, VerificationResult
from vetinari.agents.self_reflection import (
    KAIZEN_ITERATION_THRESHOLD,
    MAX_REFLECTION_ITERATIONS,
    ReflectionStrategy,
    get_reflection_strategy,
    reflect,
)

# -- Strategy resolution -----------------------------------------------------


class TestGetReflectionStrategy:
    """Reflection strategy resolution from task context."""

    def test_default_is_simple(self):
        task = make_agent_task()
        assert get_reflection_strategy(task) == ReflectionStrategy.SIMPLE

    def test_draft_refine_from_context(self):
        task = make_agent_task(context={"reflection_strategy": "draft_refine"})
        assert get_reflection_strategy(task) == ReflectionStrategy.DRAFT_REFINE

    def test_tree_of_thought_from_context(self):
        task = make_agent_task(context={"reflection_strategy": "tree_of_thought"})
        assert get_reflection_strategy(task) == ReflectionStrategy.TREE_OF_THOUGHT

    def test_unknown_strategy_falls_back_to_simple(self):
        task = make_agent_task(context={"reflection_strategy": "nonexistent"})
        assert get_reflection_strategy(task) == ReflectionStrategy.SIMPLE

    @pytest.mark.parametrize(
        "strategy_value,expected",
        [
            ("simple", ReflectionStrategy.SIMPLE),
            ("draft_refine", ReflectionStrategy.DRAFT_REFINE),
            ("tree_of_thought", ReflectionStrategy.TREE_OF_THOUGHT),
        ],
    )
    def test_all_strategies_configurable(self, strategy_value, expected):
        """AC #2, #6: all strategies are configurable via task context."""
        task = make_agent_task(context={"reflection_strategy": strategy_value})
        assert get_reflection_strategy(task) == expected


# -- SIMPLE strategy ---------------------------------------------------------


class TestSimpleReflection:
    """SIMPLE strategy is a no-op passthrough."""

    def test_returns_original_unchanged(self):
        task = make_agent_task()
        result = make_agent_result(output="original output")
        evaluator = MagicMock()
        refiner = MagicMock()

        reflection = reflect(task, result, ReflectionStrategy.SIMPLE, evaluator, refiner)

        assert reflection.original_output == "original output"
        assert reflection.refined_output == "original output"
        assert reflection.iterations_used == 0
        assert reflection.strategy == ReflectionStrategy.SIMPLE
        assert reflection.is_improved is False

    def test_evaluator_and_refiner_not_called(self):
        task = make_agent_task()
        result = make_agent_result()
        evaluator = MagicMock()
        refiner = MagicMock()

        reflect(task, result, ReflectionStrategy.SIMPLE, evaluator, refiner)

        evaluator.assert_not_called()
        refiner.assert_not_called()


# -- DRAFT_REFINE strategy ---------------------------------------------------


class TestDraftRefineReflection:
    """DRAFT_REFINE strategy: evaluate-refine loop."""

    def test_accepts_on_first_evaluation(self):
        """If evaluator says acceptable immediately, no refinement needed."""
        task = make_agent_task()
        result = make_agent_result(output="good output")
        evaluator = MagicMock(return_value=(True, []))
        refiner = MagicMock()

        reflection = reflect(task, result, ReflectionStrategy.DRAFT_REFINE, evaluator, refiner)

        assert reflection.refined_output == "good output"
        assert reflection.iterations_used == 1
        assert reflection.is_improved is False
        refiner.assert_not_called()

    def test_refines_and_accepts(self):
        """Evaluator rejects first draft, accepts after one refinement."""
        task = make_agent_task()
        initial = make_agent_result(output="draft")
        refined = make_agent_result(output="refined")

        evaluator = MagicMock(
            side_effect=[
                (False, ["needs improvement"]),
                (True, []),
            ]
        )
        refiner = MagicMock(return_value=refined)

        reflection = reflect(task, initial, ReflectionStrategy.DRAFT_REFINE, evaluator, refiner)

        assert reflection.refined_output == "refined"
        assert reflection.original_output == "draft"
        assert reflection.iterations_used == 2
        assert reflection.is_improved is True
        assert "needs improvement" in reflection.evaluation_notes

    def test_caps_at_max_iterations(self):
        """Stops after MAX_REFLECTION_ITERATIONS even if still unacceptable."""
        task = make_agent_task()
        initial = make_agent_result(output="never good enough")
        still_bad = make_agent_result(output="still bad")

        evaluator = MagicMock(return_value=(False, ["still wrong"]))
        refiner = MagicMock(return_value=still_bad)

        reflection = reflect(task, initial, ReflectionStrategy.DRAFT_REFINE, evaluator, refiner)

        assert reflection.iterations_used == MAX_REFLECTION_ITERATIONS
        assert len(reflection.evaluation_notes) == MAX_REFLECTION_ITERATIONS

    def test_multiple_refinement_rounds(self):
        """Three rounds of feedback before acceptance."""
        task = make_agent_task()
        initial = make_agent_result(output="v1")
        v2 = make_agent_result(output="v2")
        v3 = make_agent_result(output="v3")

        evaluator = MagicMock(
            side_effect=[
                (False, ["issue A"]),
                (False, ["issue B"]),
                (True, []),
            ]
        )
        refiner = MagicMock(side_effect=[v2, v3])

        reflection = reflect(task, initial, ReflectionStrategy.DRAFT_REFINE, evaluator, refiner)

        assert reflection.iterations_used == 3
        assert reflection.is_improved is True
        assert reflection.evaluation_notes == ["issue A", "issue B"]


# -- TREE_OF_THOUGHT strategy ------------------------------------------------


class TestTreeOfThoughtReflection:
    """TREE_OF_THOUGHT strategy: generate candidates, pick best."""

    def test_accepts_initial_if_good(self):
        """If initial output passes evaluation, skip candidate generation."""
        task = make_agent_task()
        result = make_agent_result(output="already great")
        evaluator = MagicMock(return_value=(True, []))
        refiner = MagicMock()

        reflection = reflect(task, result, ReflectionStrategy.TREE_OF_THOUGHT, evaluator, refiner)

        assert reflection.refined_output == "already great"
        assert reflection.iterations_used == 1
        assert reflection.is_improved is False
        refiner.assert_not_called()

    def test_picks_best_candidate(self):
        """Generates candidates and picks the one with fewest issues."""
        task = make_agent_task()
        initial = make_agent_result(output="mediocre")
        better = make_agent_result(output="better")
        ok = make_agent_result(output="ok")

        # Initial: 3 issues, candidate 1: 1 issue (best), candidate 2: 2 issues
        evaluator = MagicMock(
            side_effect=[
                (False, ["issue1", "issue2", "issue3"]),  # initial
                (False, ["minor issue"]),  # candidate 1 — best
                (False, ["issue1", "issue2"]),  # candidate 2
            ]
        )
        refiner = MagicMock(side_effect=[better, ok])

        reflection = reflect(task, initial, ReflectionStrategy.TREE_OF_THOUGHT, evaluator, refiner)

        assert reflection.refined_output == "better"
        assert reflection.is_improved is True

    def test_stops_early_on_acceptable_candidate(self):
        """If a candidate is acceptable, stop generating more."""
        task = make_agent_task()
        initial = make_agent_result(output="needs work")
        perfect = make_agent_result(output="perfect")

        evaluator = MagicMock(
            side_effect=[
                (False, ["needs work"]),  # initial — rejected
                (True, []),  # first candidate — accepted
            ]
        )
        refiner = MagicMock(return_value=perfect)

        reflection = reflect(task, initial, ReflectionStrategy.TREE_OF_THOUGHT, evaluator, refiner)

        assert reflection.refined_output == "perfect"
        assert refiner.call_count == 1  # stopped early after first candidate


# -- Kaizen reporting --------------------------------------------------------


class TestKaizenReporting:
    """AC #3: Self-correction iteration count tracked and reported to kaizen."""

    def test_high_iterations_logged_as_warning(self, caplog):
        """Iterations >= KAIZEN_ITERATION_THRESHOLD trigger under-qualified warning."""
        task = make_agent_task()
        initial = make_agent_result(output="draft")
        still_bad = make_agent_result(output="still bad")

        # Reject for (threshold - 1) iterations, then accept on the threshold iteration
        side_effects = [(False, ["issue"])] * (KAIZEN_ITERATION_THRESHOLD - 1) + [(True, [])]
        evaluator = MagicMock(side_effect=side_effects)
        refiner = MagicMock(return_value=still_bad)

        with caplog.at_level(logging.WARNING, logger="vetinari.agents.self_reflection"):
            reflect(task, initial, ReflectionStrategy.DRAFT_REFINE, evaluator, refiner)

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("under-qualified" in msg for msg in warning_msgs)

    def test_low_iterations_no_warning(self, caplog):
        """Iterations below threshold do not trigger under-qualified warning."""
        task = make_agent_task()
        initial = make_agent_result(output="good enough")
        evaluator = MagicMock(return_value=(True, []))
        refiner = MagicMock()

        with caplog.at_level(logging.WARNING, logger="vetinari.agents.self_reflection"):
            reflect(task, initial, ReflectionStrategy.DRAFT_REFINE, evaluator, refiner)

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("under-qualified" in msg for msg in warning_msgs)


# -- MultiModeAgent integration ----------------------------------------------


class TestMultiModeAgentSelfReflection:
    """Integration: self-reflection wired into MultiModeAgent.execute()."""

    def _make_test_agent_class(self):
        """Build a minimal MultiModeAgent subclass for testing."""
        from vetinari.agents.multi_mode_agent import MultiModeAgent
        from vetinari.types import AgentType

        class _ReflectionTestAgent(MultiModeAgent):
            MODES = {"test_mode": "_execute_test"}
            DEFAULT_MODE = "test_mode"
            MODE_KEYWORDS = {}

            def __init__(self):
                super().__init__(AgentType.WORKER)

            def _execute_test(self, task):
                if task.context.get("reflection_feedback"):
                    return make_agent_result(output="refined output")
                return make_agent_result(output="draft output")

            def verify(self, output):
                if output == "draft output":
                    return VerificationResult(
                        passed=False,
                        issues=[{"message": "needs work"}],
                        score=0.3,
                    )
                return VerificationResult(passed=True, score=0.9)

        return _ReflectionTestAgent

    def test_worker_self_reflection_produces_refined_output(self):
        """AC #1, #5: Worker implements draft -> evaluate -> refine -> submit."""
        cls = self._make_test_agent_class()
        agent = cls()
        task = make_agent_task(
            context={"reflection_strategy": "draft_refine", "mode": "test_mode"},
        )

        result = agent.execute(task)

        assert result.success is True
        assert result.output == "refined output"
        assert result.metadata.get("self_reflection", {}).get("is_improved") is True
        assert result.metadata["self_reflection"]["strategy"] == "draft_refine"

    def test_worker_self_reflection_configurable_strategy(self):
        """AC #6: Different strategies produce different reflection behavior."""
        cls = self._make_test_agent_class()

        # SIMPLE — no reflection, returns draft
        agent_simple = cls()
        task_simple = make_agent_task(context={"mode": "test_mode"})
        result_simple = agent_simple.execute(task_simple)
        assert result_simple.output == "draft output"
        assert "self_reflection" not in (result_simple.metadata or {})

        # DRAFT_REFINE — reflects and refines
        agent_refine = cls()
        task_refine = make_agent_task(
            context={"reflection_strategy": "draft_refine", "mode": "test_mode"},
        )
        result_refine = agent_refine.execute(task_refine)
        assert result_refine.output == "refined output"
        assert result_refine.metadata["self_reflection"]["strategy"] == "draft_refine"

    def test_skips_reflection_on_failed_result(self):
        """Failed handler results bypass self-reflection entirely."""
        from vetinari.agents.multi_mode_agent import MultiModeAgent
        from vetinari.types import AgentType

        class _FailingAgent(MultiModeAgent):
            MODES = {"fail_mode": "_execute_fail"}
            DEFAULT_MODE = "fail_mode"
            MODE_KEYWORDS = {}

            def __init__(self):
                super().__init__(AgentType.WORKER)

            def _execute_fail(self, task):
                return make_agent_result(success=False, output=None, errors=["handler failed"])

        agent = _FailingAgent()
        task = make_agent_task(
            context={"reflection_strategy": "draft_refine", "mode": "fail_mode"},
        )

        result = agent.execute(task)

        assert result.success is False
        assert "self_reflection" not in (result.metadata or {})

    def test_skips_reflection_for_simple_strategy(self):
        """Default SIMPLE strategy means no reflection overhead."""
        cls = self._make_test_agent_class()
        agent = cls()
        task = make_agent_task(context={"mode": "test_mode"})

        result = agent.execute(task)

        assert result.output == "draft output"
        assert "self_reflection" not in (result.metadata or {})
