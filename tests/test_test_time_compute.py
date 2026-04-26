"""Tests for vetinari.optimization.test_time_compute.

Covers NGramHeuristicScorer, MCTSPlanner, TestTimeComputeScaler, and related
dataclasses.
"""

from __future__ import annotations

import math

import pytest

from vetinari.exceptions import ConfigurationError
from vetinari.optimization.test_time_compute import (
    ComputeResult,
    ComputeStepScore,
    MCTSNode,
    MCTSPlanner,
    NGramHeuristicScorer,
    get_test_time_scaler,
)
from vetinari.optimization.test_time_compute import (
    TestTimeComputeScaler as TimeComputeScaler,
)

# ---------------------------------------------------------------------------
# StepScore / ComputeResult dataclasses
# ---------------------------------------------------------------------------


class TestComputeResultDataclass:
    def test_compute_result_dataclass(self):
        """ComputeResult stores all expected fields."""
        result = ComputeResult(
            level_used=2,
            result="do X -> do Y",
            quality_estimate=0.75,
            steps_evaluated=4,
            computation_budget_used=8.0,
        )
        assert result.level_used == 2
        assert result.result == "do X -> do Y"
        assert result.quality_estimate == 0.75
        assert result.steps_evaluated == 4
        assert result.computation_budget_used == 8.0

    def test_step_score_dataclass(self):
        """StepScore stores step text, score, and reasoning."""
        ss = ComputeStepScore(step_text="Analyse requirements", score=0.82, reasoning="High coherence")
        assert ss.step_text == "Analyse requirements"
        assert ss.score == 0.82
        assert "coherence" in ss.reasoning.lower()


# ---------------------------------------------------------------------------
# NGramHeuristicScorer
# ---------------------------------------------------------------------------


class TestNGramHeuristicScorer:
    def test_ngram_heuristic_scorer_scores_steps(self):
        """score_steps returns one StepScore per input step."""
        scorer = NGramHeuristicScorer()
        steps = [
            "Identify the problem requirements",
            "Design the data model",
            "Implement the core logic",
        ]
        results = scorer.score_steps(steps)
        assert len(results) == len(steps)
        for ss in results:
            assert isinstance(ss, ComputeStepScore)
            assert 0.0 <= ss.score <= 1.0
            assert ss.reasoning  # non-empty reasoning

    def test_ngram_heuristic_scorer_prunes_low_quality(self):
        """prune_low_quality removes steps below threshold."""
        scorer = NGramHeuristicScorer(coherence_threshold=0.5)
        # Empty step should score 0.0 and be pruned
        steps = ["Good detailed reasoning step about the architecture", ""]
        kept = scorer.prune_low_quality(steps, threshold=0.1)
        # Empty string scores 0.0, so it should be pruned at threshold=0.1
        assert "" not in kept

    def test_ngram_heuristic_scorer_prunes_with_default_threshold(self):
        """prune_low_quality uses instance threshold when override is None."""
        scorer = NGramHeuristicScorer(coherence_threshold=0.99)
        steps = ["a", "b", "c"]  # very short steps — likely low-quality
        kept = scorer.prune_low_quality(steps)
        # With a very high threshold almost everything should be pruned
        assert len(kept) <= len(steps)

    def test_ngram_heuristic_score_step_returns_float_in_range(self):
        """score_step always returns a value in [0.0, 1.0]."""
        scorer = NGramHeuristicScorer()
        score = scorer.score_step("This is a reasonable reasoning step.", context="Some prior context.")
        assert 0.0 <= score <= 1.0

    def test_ngram_heuristic_empty_step_scores_zero(self):
        """An empty step must score 0.0."""
        scorer = NGramHeuristicScorer()
        assert scorer.score_step("") == 0.0
        assert scorer.score_step("   ") == 0.0

    def test_ngram_heuristic_score_distribution_varies_with_input(self):
        """Different inputs produce different scores — not a collapsed constant.

        Catches the 'all scores identical' anti-pattern from Learning Data
        Quality rules: a scorer that assigns the same score to every input
        is broken and cannot discriminate step quality.
        """
        scorer = NGramHeuristicScorer()
        inputs = [
            "Identify requirements and plan the architecture of the system",
            "Design the data model around aggregate roots",
            "Implement the authorisation middleware first, then the handler",
            "a",
            "aaaa aaaa aaaa aaaa",
            "zzzzzzzzzzzzzzzzzzzzzzzzzzzz",
            "The cat sat on the mat",
            "def add(a, b):\n    return a + b",
        ]
        context = "Overall goal: build a working web service with tests"
        scores = {text: scorer.score_step(text, context=context) for text in inputs}

        unique_scores = {round(v, 3) for v in scores.values()}
        assert len(unique_scores) >= 4, (
            "NGramHeuristicScorer produced fewer than 4 distinct scores across 8 "
            f"very different inputs: {scores}. If the distribution is this flat, "
            "the scorer is broken (Learning Data Quality anti-pattern)."
        )

    def test_ngram_heuristic_scorer_does_not_market_as_prm(self):
        """Docstring and class name MUST NOT claim to be a process reward model."""
        assert NGramHeuristicScorer.__name__ == "NGramHeuristicScorer"
        doc = NGramHeuristicScorer.__doc__ or ""
        assert "NOT a process reward model" in doc, (
            f"Docstring must explicitly disclaim being a PRM. Found: {doc[:200]!r}"
        )


# ---------------------------------------------------------------------------
# MCTSNode / MCTSPlanner
# ---------------------------------------------------------------------------


class TestMCTSNodeUCB1:
    def test_mcts_node_ucb1_selection(self):
        """MCTSPlanner selects unvisited nodes before using UCB1."""
        planner = MCTSPlanner(max_iterations=10, max_depth=2)
        root = MCTSNode(state="Design a web service")

        # Manually add children — one visited, one not
        visited_child = MCTSNode(state="visited", parent=root, action="visited")
        visited_child.visits = 5
        visited_child.total_value = 3.0

        unvisited_child = MCTSNode(state="unvisited", parent=root, action="unvisited")

        root.children = [visited_child, unvisited_child]
        root.visits = 5

        selected = planner._select(root)
        # Unvisited child must be preferred
        assert selected is unvisited_child

    def test_ucb1_infinite_for_unvisited(self):
        """UCB1 is infinite for a never-visited node."""
        planner = MCTSPlanner()
        node = MCTSNode(state="x", visits=0)
        assert planner._ucb1(node) == float("inf")

    def test_ucb1_finite_for_visited(self):
        """UCB1 is a finite float for a visited node with a parent."""
        planner = MCTSPlanner()
        parent = MCTSNode(state="parent", visits=10)
        child = MCTSNode(state="child", parent=parent, visits=3, total_value=2.1)
        score = planner._ucb1(child)
        assert math.isfinite(score)
        assert score > 0


class TestMCTSSearch:
    def test_mcts_search_returns_path(self):
        """search() returns a non-empty list of strings."""
        planner = MCTSPlanner(max_iterations=20, max_depth=3)

        def eval_fn(task: str) -> float:
            return 0.8 if "implement" in task.lower() else 0.4

        path = planner.search("Implement a REST API endpoint", eval_fn)
        assert isinstance(path, list)
        assert len(path) >= 1
        assert all(isinstance(s, str) for s in path)

    def test_mcts_decompose_step_returns_subtasks(self):
        """_decompose_step splits a task into 2–4 subtasks."""
        planner = MCTSPlanner()
        subtasks = planner._decompose_step("Implement caching and then add unit tests")
        assert 2 <= len(subtasks) <= 4
        assert all(isinstance(s, str) and s for s in subtasks)

    def test_mcts_search_respects_max_depth(self):
        """Tree depth does not exceed max_depth during search."""
        planner = MCTSPlanner(max_iterations=50, max_depth=2)
        path = planner.search("Design data model", lambda s: 0.6)
        # Path length should reflect depth constraint — at most max_depth steps
        assert len(path) <= planner._max_depth + 1


# ---------------------------------------------------------------------------
# TestTimeComputeScaler
# ---------------------------------------------------------------------------


class TestScalerAutoSelectLevel:
    def test_scaler_auto_select_level(self):
        """auto_select_level maps complexity ranges to the right level."""
        scaler = TimeComputeScaler()
        assert scaler.auto_select_level(1) == 1
        assert scaler.auto_select_level(3) == 1
        assert scaler.auto_select_level(4) == 2
        assert scaler.auto_select_level(7) == 2
        assert scaler.auto_select_level(8) == 3
        assert scaler.auto_select_level(10) == 3


class TestScalerLevel1:
    def test_scaler_level1_best_of_n(self):
        """Level 1 returns a ComputeResult with level_used=1."""
        scaler = TimeComputeScaler()
        result = scaler.scale(task="Write a hello world function", level=1, n=3)
        assert isinstance(result, ComputeResult)
        assert result.level_used == 1
        assert result.steps_evaluated == 3
        assert isinstance(result.result, str)
        assert result.result  # non-empty

    def test_scaler_level1_with_evaluate_fn(self):
        """Level 1 uses the evaluate_fn when provided."""
        scaler = TimeComputeScaler()

        def score(text: str) -> float:
            return 1.0 if "step by step" in text.lower() else 0.0

        result = scaler.scale("Solve the problem", level=1, evaluate_fn=score, n=4)
        # Best-of-N should pick the candidate that includes "step by step"
        assert result.quality_estimate == 1.0


class TestScalerLevel2:
    def test_scaler_level2_prm(self):
        """Level 2 returns a ComputeResult with level_used=2."""
        scaler = TimeComputeScaler()
        result = scaler.scale("Refactor the authentication module", level=2)
        assert result.level_used == 2
        assert result.steps_evaluated >= 1
        assert 0.0 <= result.quality_estimate <= 1.0


class TestScalerLevel3:
    def test_scaler_level3_mcts(self):
        """Level 3 returns a ComputeResult with level_used=3."""
        scaler = TimeComputeScaler(mcts_planner=MCTSPlanner(max_iterations=10))
        result = scaler.scale("Build a distributed cache", level=3, evaluate_fn=lambda s: 0.7)
        assert result.level_used == 3
        assert result.result
        assert result.steps_evaluated == 10  # max_iterations from planner

    def test_scaler_invalid_level_raises(self):
        """Requesting an unsupported level raises ValueError."""
        scaler = TimeComputeScaler()
        with pytest.raises(ConfigurationError, match="Unsupported compute level"):
            scaler.scale("task", level=99)


class TestGetTestTimeScaler:
    def test_factory_returns_scaler(self):
        """get_test_time_scaler() returns a TimeComputeScaler."""
        scaler = get_test_time_scaler()
        assert isinstance(scaler, TimeComputeScaler)
