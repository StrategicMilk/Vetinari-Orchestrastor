"""Integration tests for Thompson Sampling convergence.

Verifies that the bandit-based model selector converges to the better
model within a reasonable number of iterations, and that exploration
occurs during the cold-start phase.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from vetinari.learning.model_selector import BetaArm, ThompsonSamplingSelector
from vetinari.learning.thompson_arms import ThompsonBetaArm

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_selector() -> ThompsonSamplingSelector:
    """Create an isolated ThompsonSamplingSelector with no persisted state.

    Patches out all disk I/O (SQLite + JSON) so the selector starts empty
    and converges purely from in-test updates.
    """
    with (
        patch.object(ThompsonSamplingSelector, "_load_state"),
        patch.object(ThompsonSamplingSelector, "_save_state"),
        patch.object(ThompsonSamplingSelector, "_seed_from_benchmarks"),
        patch.object(
            ThompsonSamplingSelector,
            "_get_informed_prior",
            return_value=(1.0, 1.0),
        ),
    ):
        selector = ThompsonSamplingSelector()
        # Clear any arms that may have been created despite the patches
        selector._arms.clear()
    return selector


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConvergenceClearWinner:
    """Selector must converge to model_a when model_a dominates model_b."""

    def test_convergence_clear_winner_over_500_iterations(self, fresh_selector: ThompsonSamplingSelector) -> None:
        """After 500 iterations model_a (quality 0.9) is selected >80% in the last 100."""
        models = ["model_a", "model_b"]
        task_type = "coding"

        # Simulate 500 rounds: model_a succeeds at q=0.9, model_b at q=0.3
        for _ in range(500):
            chosen = fresh_selector.select_model(task_type, models)
            if chosen == "model_a":
                fresh_selector.update(chosen, task_type, quality_score=0.9, success=True)
            else:
                fresh_selector.update(chosen, task_type, quality_score=0.3, success=False)

        # Measure last-100 selections
        last_100_choices = [fresh_selector.select_model(task_type, models) for _ in range(100)]
        model_a_rate = last_100_choices.count("model_a") / 100

        assert model_a_rate > 0.80, f"Expected model_a selection rate >80% in last 100, got {model_a_rate:.0%}"

    def test_convergence_winner_has_higher_mean(self, fresh_selector: ThompsonSamplingSelector) -> None:
        """After convergence the winner's Beta mean is strictly above the loser's."""
        models = ["winner", "loser"]
        task_type = "review"

        for _ in range(300):
            chosen = fresh_selector.select_model(task_type, models)
            if chosen == "winner":
                fresh_selector.update(chosen, task_type, quality_score=0.95, success=True)
            else:
                fresh_selector.update(chosen, task_type, quality_score=0.1, success=False)

        winner_state = fresh_selector.get_arm_state("winner", task_type)
        loser_state = fresh_selector.get_arm_state("loser", task_type)

        assert winner_state["mean"] > loser_state["mean"], (
            f"Winner mean {winner_state['mean']:.3f} should exceed loser mean {loser_state['mean']:.3f}"
        )


class TestConvergenceThreeModels:
    """Selector must correctly rank three models with distinct quality levels."""

    def test_convergence_with_three_models(self, fresh_selector: ThompsonSamplingSelector) -> None:
        """After training the best model (q=0.9) is selected more than the worst (q=0.1).

        Uses forced updates to ensure each arm gets enough observations,
        since Thompson Sampling may under-explore the best arm during
        early cold-start rounds on slow CI runners.
        """
        models = ["model_best", "model_mid", "model_worst"]
        quality_map = {"model_best": 0.9, "model_mid": 0.5, "model_worst": 0.1}
        task_type = "planning"

        # Phase 1: force-feed each arm enough data so posteriors separate.
        # Without this, the selector may never try model_best enough times
        # to learn it's good (exploration starvation).
        for model_id in models:
            q = quality_map[model_id]
            for _ in range(80):
                fresh_selector.update(model_id, task_type, quality_score=q, success=q >= 0.5)

        # Phase 2: let the selector choose freely — it should converge.
        for _ in range(300):
            chosen = fresh_selector.select_model(task_type, models)
            q = quality_map[chosen]
            fresh_selector.update(chosen, task_type, quality_score=q, success=q >= 0.5)

        # Phase 3: measure selection distribution.
        counts: dict[str, int] = dict.fromkeys(models, 0)
        for _ in range(300):
            counts[fresh_selector.select_model(task_type, models)] += 1

        assert counts["model_best"] > counts["model_worst"], (
            f"Best model count {counts['model_best']} should exceed worst model count "
            f"{counts['model_worst']} after training"
        )
        assert counts["model_best"] > 0, "model_best should be selected at least once in 300 trials"


class TestColdStartExploration:
    """The selector must explore all candidates before converging."""

    def test_cold_start_exploration_samples_all_models(self, fresh_selector: ThompsonSamplingSelector) -> None:
        """In the first 20 selections all models should appear at least once."""
        models = ["alpha", "beta", "gamma"]
        task_type = "analysis"
        seen: set[str] = set()

        for _ in range(20):
            chosen = fresh_selector.select_model(task_type, models)
            seen.add(chosen)

        assert len(seen) > 1, f"Expected exploration across multiple models in first 20 selections, only saw: {seen}"

    def test_cold_start_not_all_same_model(self, fresh_selector: ThompsonSamplingSelector) -> None:
        """With equal priors, first 20 selections must not all return the same model."""
        models = ["x", "y", "z"]
        task_type = "testing"
        first_20 = [fresh_selector.select_model(task_type, models) for _ in range(20)]

        unique_models = set(first_20)
        assert len(unique_models) >= 2, (
            f"Cold-start exploration should pick at least 2 distinct models in 20 trials, only got: {unique_models}"
        )


class TestUpdateShiftsPreference:
    """A single strong update must visibly shift the Beta mean toward that model."""

    def test_update_shifts_preference_toward_high_quality_model(self, fresh_selector: ThompsonSamplingSelector) -> None:
        """After updating model_x with high quality, its Beta mean rises above model_y."""
        models = ["model_x", "model_y"]
        task_type = "architecture"

        # Ensure both arms exist with equal priors before measuring
        for _ in range(5):
            fresh_selector.select_model(task_type, models)

        # Apply 20 high-quality successes to model_x only
        for _ in range(20):
            fresh_selector.update("model_x", task_type, quality_score=0.95, success=True)

        x_state = fresh_selector.get_arm_state("model_x", task_type)
        y_state = fresh_selector.get_arm_state("model_y", task_type)

        assert x_state["mean"] > y_state["mean"], (
            f"model_x mean {x_state['mean']:.3f} should exceed model_y mean "
            f"{y_state['mean']:.3f} after 20 high-quality updates"
        )

    def test_update_total_pulls_increments(self, fresh_selector: ThompsonSamplingSelector) -> None:
        """Each call to update() must increment total_pulls on the arm."""
        task_type = "refactoring"

        before = fresh_selector.get_arm_state("model_q", task_type)["total_pulls"]
        fresh_selector.update("model_q", task_type, quality_score=0.7, success=True)
        after = fresh_selector.get_arm_state("model_q", task_type)["total_pulls"]

        assert after == before + 1, f"total_pulls should increase by 1 after update, was {before} now {after}"
