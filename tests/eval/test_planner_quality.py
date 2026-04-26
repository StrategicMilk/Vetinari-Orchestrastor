"""Evaluation tests for planner agent output quality.

These tests use the scoring protocol in ``vetinari.evaluation`` to verify
that the ``evaluate_plan_quality`` function correctly scores well-formed,
malformed, and degenerate plan structures.
"""

from __future__ import annotations

import pytest

from vetinari.evaluation import EvalResult, evaluate_plan_quality


@pytest.mark.eval
class TestPlannerQuality:
    """Evaluation tests for plan structure quality scoring."""

    def test_good_plan_scores_high(self, mock_plan_response: dict) -> None:
        """A well-formed plan with 3-5 tasks, valid deps, and a clear goal scores above 0.7."""
        result: EvalResult = evaluate_plan_quality(mock_plan_response)

        assert result.score > 0.7, (
            f"Expected score > 0.7 for a well-formed plan, got {result.score:.2f}. Reasoning: {result.reasoning}"
        )
        assert result.passed is True
        assert "task_count" in result.metrics
        assert result.metrics["task_count"] == 4

    def test_empty_plan_fails(self) -> None:
        """A plan with no tasks fails evaluation despite valid goal and deps."""
        empty_plan = {
            "goal": "Do something",
            "tasks": [],
            "dependencies": {},
        }
        result: EvalResult = evaluate_plan_quality(empty_plan)

        # Task dimension scores 0 → composite is 0.6 (dep 0.3 + goal 0.3)
        # Below a well-formed plan (1.0) but not zero due to valid goal/deps
        assert result.score < 0.7, (
            f"Expected score < 0.7 for an empty plan, got {result.score:.2f}. Reasoning: {result.reasoning}"
        )
        assert result.passed is False
        assert result.metrics["task_count"] == 0
        assert result.metrics["task_score"] == 0.0

    def test_circular_deps_detected(self) -> None:
        """A plan with circular task dependencies receives a dependency score penalty."""
        circular_plan = {
            "goal": "Complete a set of interdependent tasks",
            "tasks": [
                {"id": "task-a", "name": "Step A"},
                {"id": "task-b", "name": "Step B"},
                {"id": "task-c", "name": "Step C"},
            ],
            "dependencies": {
                "task-a": ["task-c"],  # A depends on C
                "task-b": ["task-a"],  # B depends on A
                "task-c": ["task-b"],  # C depends on B — cycle: A -> C -> B -> A
            },
        }
        result: EvalResult = evaluate_plan_quality(circular_plan)

        # Dependency dimension should be scored at 0 due to the cycle
        assert result.metrics["dependency_score"] == 0.0
        # Overall score should be penalised below a well-formed plan
        assert result.score < 0.8, f"Expected score below 0.8 for a circular-dep plan, got {result.score:.2f}"
        # The reasoning or issues should mention the cycle
        dep_issues: list[str] = result.metrics.get("dependency_issues", [])
        assert any("circular" in issue for issue in dep_issues), (
            f"Expected 'circular' in dependency issues, got: {dep_issues}"
        )

    def test_plan_missing_goal_penalised(self) -> None:
        """A plan with tasks but no goal string receives a goal-dimension penalty."""
        no_goal_plan = {
            "goal": "",
            "tasks": [
                {"id": "t1", "name": "Task one"},
                {"id": "t2", "name": "Task two"},
            ],
            "dependencies": {},
        }
        result: EvalResult = evaluate_plan_quality(no_goal_plan)

        assert result.metrics["goal_score"] == 0.0
        # With missing goal the max achievable composite is 0.70 (task 0.40 + dep 0.30)
        assert result.score <= 0.70 + 1e-9

    def test_plan_with_unknown_dep_ids_penalised(self) -> None:
        """A plan whose dependencies reference non-existent task IDs is penalised."""
        bad_dep_plan = {
            "goal": "Accomplish the objective",
            "tasks": [
                {"id": "step-1", "name": "First step"},
            ],
            "dependencies": {
                "step-1": ["step-99"],  # step-99 does not exist
            },
        }
        result: EvalResult = evaluate_plan_quality(bad_dep_plan)

        assert result.metrics["dependency_score"] == 0.0
        dep_issues = result.metrics.get("dependency_issues", [])
        assert any("unknown" in issue for issue in dep_issues), (
            f"Expected 'unknown' in dependency issues, got: {dep_issues}"
        )
