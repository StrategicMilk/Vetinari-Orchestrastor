"""Tests for Session 3A exit criteria: Learning Loop Closure.

Verifies that all learning subsystems are correctly wired and produce
expected behavior when called through the production interfaces.

Exit criteria covered:
1. Thompson Sampling converges to better model after 50+ observations
2. Training data records have non-zero tokens_used and latency_ms
3. FeedbackLoop updates model weights after recording outcomes
4. Prompt variant with higher quality gets promoted
5. Self-refinement attempts fix before escalating on Inspector rejection
6. QualityScorer produces non-degenerate score distribution
7. Training quality gate rejects worse model
8. Shadow testing framework compares metrics correctly
"""

from __future__ import annotations

import os
import tempfile
import threading
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

# Ensure test-mode environment before any vetinari imports so state dirs
# are isolated from production data.
os.environ.setdefault("VETINARI_STATE_DIR", tempfile.mkdtemp())


# ---------------------------------------------------------------------------
# Exit criterion 1: Thompson Sampling converges to better model
# ---------------------------------------------------------------------------


class TestThompsonSamplingConvergence:
    """Thompson Sampling converges to better model after 50+ observations."""

    def test_converges_to_better_model(self) -> None:
        """After 50 observations model_a dominates model_b in selection."""
        from vetinari.learning.model_selector import ThompsonSamplingSelector

        selector = ThompsonSamplingSelector()

        # 50 paired observations: model_a high quality, model_b low quality
        for _ in range(50):
            selector.update("model_a", "coding", quality_score=0.9, success=True)
            selector.update("model_b", "coding", quality_score=0.3, success=False)

        counts: dict[str, int] = {"model_a": 0, "model_b": 0}
        for _ in range(100):
            selected = selector.select_model("coding", ["model_a", "model_b"])
            counts[selected] += 1

        assert counts["model_a"] > 80, (
            f"Expected model_a selected > 80/100 times after 50 quality observations, got {counts['model_a']}"
        )

    def test_arms_update_alpha_on_success(self) -> None:
        """A success with high quality score increases alpha more than beta."""
        from vetinari.learning.model_selector import ThompsonSamplingSelector

        selector = ThompsonSamplingSelector()
        before = selector.get_arm_state("probe_model", "general")
        initial_alpha = before["alpha"]

        selector.update("probe_model", "general", quality_score=0.8, success=True)

        after = selector.get_arm_state("probe_model", "general")
        assert after["alpha"] > initial_alpha, "alpha should increase after a successful high-quality observation"
        assert after["total_pulls"] == before["total_pulls"] + 1

    def test_arms_update_beta_on_failure(self) -> None:
        """A failure with low quality score increases beta more than alpha."""
        from vetinari.learning.model_selector import ThompsonSamplingSelector

        selector = ThompsonSamplingSelector()
        before = selector.get_arm_state("probe_model_b", "general")
        initial_beta = before["beta"]

        selector.update("probe_model_b", "general", quality_score=0.2, success=False)

        after = selector.get_arm_state("probe_model_b", "general")
        assert after["beta"] > initial_beta, "beta should increase after a failed low-quality observation"


# ---------------------------------------------------------------------------
# Exit criterion 2: Training data records have non-zero tokens_used/latency_ms
# ---------------------------------------------------------------------------


class TestTrainingDataQuality:
    """Training data collector enforces non-zero tokens_used and latency_ms."""

    def _fresh_collector(self) -> object:
        """Return an isolated synchronous collector backed by a temp file."""
        from vetinari.learning.training_data import TrainingDataCollector

        path = os.path.join(tempfile.mkdtemp(), "training.jsonl")
        return TrainingDataCollector(output_path=path, sync=True)

    def test_rejects_zero_tokens(self) -> None:
        """Record with tokens_used=0 must be silently dropped."""
        collector = self._fresh_collector()
        collector.record(
            task="test task",
            prompt="test prompt",
            response="test response content",
            score=0.8,
            model_id="test_model",
            tokens_used=0,
            latency_ms=500,
        )
        stats = collector.get_stats()
        assert stats.get("total", 0) == 0, "Record with tokens_used=0 should be rejected"

    def test_rejects_zero_latency(self) -> None:
        """Record with latency_ms=0 must be silently dropped."""
        collector = self._fresh_collector()
        collector.record(
            task="test task",
            prompt="test prompt",
            response="test response content",
            score=0.8,
            model_id="test_model",
            tokens_used=100,
            latency_ms=0,
        )
        stats = collector.get_stats()
        assert stats.get("total", 0) == 0, "Record with latency_ms=0 should be rejected"

    def test_accepts_valid_record(self) -> None:
        """Record with valid tokens and latency must be persisted."""
        collector = self._fresh_collector()
        collector.record(
            task="implement a cache",
            prompt="write a simple LRU cache in Python",
            response="class LRUCache: ...",
            score=0.8,
            model_id="test_model",
            tokens_used=100,
            latency_ms=500,
        )
        stats = collector.get_stats()
        assert stats.get("total", 0) == 1, "Valid record should be accepted and counted"

    def test_rejects_fallback_flag(self) -> None:
        """Record with metadata._is_fallback=True must be dropped."""
        collector = self._fresh_collector()
        collector.record(
            task="test task",
            prompt="test prompt",
            response="some non-empty response text here",
            score=0.8,
            model_id="test_model",
            tokens_used=100,
            latency_ms=500,
            metadata={"_is_fallback": True},
        )
        stats = collector.get_stats()
        assert stats.get("total", 0) == 0, "Record flagged _is_fallback=True should be rejected"

    def test_rejects_empty_response_pattern(self) -> None:
        """Record matching a known fallback response pattern must be dropped."""
        collector = self._fresh_collector()
        collector.record(
            task="test task",
            prompt="test prompt",
            response='{"content":"","sections":[]}',
            score=0.8,
            model_id="test_model",
            tokens_used=100,
            latency_ms=500,
        )
        stats = collector.get_stats()
        assert stats.get("total", 0) == 0, "Record with known fallback response pattern should be rejected"


# ---------------------------------------------------------------------------
# Exit criterion 3: FeedbackLoop updates model weights after recording outcomes
# ---------------------------------------------------------------------------


class TestFeedbackLoopWeightUpdate:
    """FeedbackLoop.record_outcome propagates quality into Thompson Sampling."""

    def test_record_outcome_updates_thompson_arm(self) -> None:
        """Recording a task outcome must increment the Thompson Sampling arm."""
        from vetinari.learning.feedback_loop import FeedbackLoop
        from vetinari.learning.model_selector import get_thompson_selector

        # Use the singleton so both FeedbackLoop and the assertion share state
        selector = get_thompson_selector()
        model_id = "feedback_test_model_x1"
        task_type = "analysis"

        before = selector.get_arm_state(model_id, task_type)
        initial_pulls = before["total_pulls"]

        loop = FeedbackLoop()
        loop.record_outcome(
            task_id="feedback-test-001",
            model_id=model_id,
            task_type=task_type,
            quality_score=0.85,
            success=True,
        )

        after = selector.get_arm_state(model_id, task_type)
        assert after["total_pulls"] > initial_pulls, (
            "FeedbackLoop.record_outcome must update Thompson Sampling arm "
            f"(pulls before={initial_pulls}, after={after['total_pulls']})"
        )

    def test_high_quality_outcome_increases_alpha(self) -> None:
        """A high-quality successful outcome should raise the arm's alpha."""
        from vetinari.learning.feedback_loop import FeedbackLoop
        from vetinari.learning.model_selector import get_thompson_selector

        selector = get_thompson_selector()
        model_id = "feedback_alpha_model_y2"
        task_type = "research"

        before = selector.get_arm_state(model_id, task_type)
        initial_alpha = before["alpha"]

        loop = FeedbackLoop()
        loop.record_outcome(
            task_id="feedback-alpha-002",
            model_id=model_id,
            task_type=task_type,
            quality_score=0.95,
            success=True,
        )

        after = selector.get_arm_state(model_id, task_type)
        assert after["alpha"] > initial_alpha, "High-quality success must increase arm alpha"


# ---------------------------------------------------------------------------
# Exit criterion 4: Prompt variant with higher quality gets promoted
# ---------------------------------------------------------------------------


class TestPromptVariantPromotion:
    """PromptEvolver promotes a variant with statistically better quality."""

    def test_variant_promoted_after_sufficient_trials(self) -> None:
        """Variant with clearly superior quality is promoted after MIN_TRIALS."""
        from vetinari.learning.prompt_evolver import PromptEvolver, PromptVariant
        from vetinari.types import PromptVersionStatus

        evolver = PromptEvolver()
        evolver.register_baseline("worker", "You are a helpful worker agent.")

        # Add a testing variant
        variant = PromptVariant(
            variant_id="worker_test_v1",
            agent_type="worker",
            prompt_text="You are an expert, thorough worker agent.",
            status=PromptVersionStatus.TESTING.value,
        )
        evolver._variants["worker"].append(variant)
        evolver._score_history["worker_test_v1"] = deque(maxlen=500)

        # Also seed the baseline score history so statistical test has data
        baseline = evolver._variants["worker"][0]
        evolver._score_history[baseline.variant_id] = deque(maxlen=500)

        # Mock _validate_variant_with_benchmark to always pass (no LLM needed)
        evolver._validate_variant_with_benchmark = lambda v: True

        # Record clearly separated quality: variant ~0.9, baseline ~0.5
        # Use enough trials to meet MIN_TRIALS=30 and get p < 0.05.
        # Add slight noise to avoid scipy precision-loss warning on identical values.
        import random

        rng = random.Random(42)
        for _ in range(35):
            evolver.record_result("worker", "worker_test_v1", quality=0.9 + rng.uniform(-0.02, 0.02))
            evolver.record_result("worker", baseline.variant_id, quality=0.5 + rng.uniform(-0.02, 0.02))

        statuses = [v.status for v in evolver._variants.get("worker", [])]
        assert PromptVersionStatus.PROMOTED.value in statuses, (
            f"Higher-quality variant should be promoted; statuses found: {statuses}"
        )

    def test_poor_variant_is_not_promoted(self) -> None:
        """Variant with quality below baseline must not be promoted."""
        from vetinari.learning.prompt_evolver import PromptEvolver, PromptVariant
        from vetinari.types import PromptVersionStatus

        evolver = PromptEvolver()
        evolver.register_baseline("inspector", "You are a strict inspector agent.")

        variant = PromptVariant(
            variant_id="inspector_bad_v1",
            agent_type="inspector",
            prompt_text="You are a lazy inspector.",
            status=PromptVersionStatus.TESTING.value,
        )
        evolver._variants["inspector"].append(variant)
        evolver._score_history["inspector_bad_v1"] = deque(maxlen=500)
        baseline = evolver._variants["inspector"][0]
        evolver._score_history[baseline.variant_id] = deque(maxlen=500)

        for _ in range(35):
            evolver.record_result("inspector", "inspector_bad_v1", quality=0.4)
            evolver.record_result("inspector", baseline.variant_id, quality=0.75)

        variant_obj = next(v for v in evolver._variants["inspector"] if v.variant_id == "inspector_bad_v1")
        assert variant_obj.status != PromptVersionStatus.PROMOTED.value, "Poor-quality variant must not be promoted"


# ---------------------------------------------------------------------------
# Exit criterion 5: Self-refinement attempts fix before escalating
# ---------------------------------------------------------------------------


class TestSelfRefinementLoop:
    """SelfRefinementLoop skips when quality is good; runs when quality is low."""

    def test_refinement_skipped_when_quality_good(self) -> None:
        """Output already above threshold bypasses refinement entirely."""
        from vetinari.learning.self_refinement import SelfRefinementLoop

        refiner = SelfRefinementLoop(adapter_manager=None)
        result = refiner.refine(
            task_description="Write a sorting function",
            initial_output="def sort(items): return sorted(items)",
            task_type="coding",
            initial_quality=0.9,
        )

        assert result.rounds_used == 0, "Refinement rounds must be 0 when quality is already above threshold"
        assert not result.improved, "improved must be False when refinement was skipped"
        assert result.output == "def sort(items): return sorted(items)", (
            "Output must be unchanged when refinement is skipped"
        )

    def test_refinement_attempted_on_low_quality_without_adapter(self) -> None:
        """Without an adapter, refiner returns original output without crashing."""
        from vetinari.learning.self_refinement import SelfRefinementLoop

        refiner = SelfRefinementLoop(adapter_manager=None)
        result = refiner.refine(
            task_description="Write a sorting function",
            initial_output="def sort(): pass",
            task_type="coding",
            importance=0.8,
            initial_quality=0.3,
        )

        # No adapter -> cannot improve, but must return a valid result object
        assert result is not None, "refine() must return a RefinementResult"
        assert result.output == "def sort(): pass", "Without adapter, original output must be returned"
        assert result.rounds_used == 0, "No rounds should complete without an adapter"

    def test_refinement_attempted_on_low_quality_with_adapter(self) -> None:
        """With an adapter available, low quality triggers at least one round."""
        from vetinari.learning.self_refinement import SelfRefinementLoop

        mock_adapter = MagicMock()
        # Critique returns an actionable message; revision returns improved text
        mock_adapter.infer.side_effect = [
            "The function is missing a docstring and error handling.",
            'def sort(items):\n    """Sort a list."""\n    return sorted(items)',
        ]

        refiner = SelfRefinementLoop(adapter_manager=mock_adapter)
        # Patch the private helpers that call adapter_manager
        refiner._critique = MagicMock(return_value="Missing docstring and error handling.")
        refiner._revise = MagicMock(return_value='def sort(items):\n    """Sort a list."""\n    return sorted(items)')

        result = refiner.refine(
            task_description="Write a sorting function",
            initial_output="def sort(): pass",
            task_type="coding",
            importance=0.8,
            initial_quality=0.3,
        )

        assert result.rounds_used >= 1, (
            "At least one refinement round should complete when adapter is present and quality is below threshold"
        )


# ---------------------------------------------------------------------------
# Exit criterion 6: QualityScorer produces non-degenerate score distribution
# ---------------------------------------------------------------------------


class TestQualityScorerDistribution:
    """QualityScorer assigns different scores to outputs of varying quality."""

    def test_different_inputs_get_different_scores(self) -> None:
        """Empty, minimal, and complete outputs must receive different scores."""
        from vetinari.learning.quality_scorer import QualityScorer

        scorer = QualityScorer(adapter_manager=None)

        score_minimal = scorer.score(
            task_id="dist_t1",
            model_id="m1",
            task_type="coding",
            task_description="Write a sort function",
            output="def sort(): pass",
            use_llm=False,
        )

        score_good = scorer.score(
            task_id="dist_t2",
            model_id="m1",
            task_type="coding",
            task_description="Write a sort function",
            output=(
                "def sort_list(items):\n"
                '    """Sort items in ascending order.\n\n'
                "    Args:\n        items: List of comparable items.\n\n"
                "    Returns:\n        Sorted list.\n"
                '    """\n'
                "    return sorted(items)\n\n"
                "assert sort_list([3, 1, 2]) == [1, 2, 3]\n"
            ),
            use_llm=False,
        )

        assert score_minimal.overall_score != score_good.overall_score, (
            "Minimal stub and well-documented implementation should score differently; "
            f"both scored {score_minimal.overall_score:.3f}"
        )
        assert score_good.overall_score >= score_minimal.overall_score, (
            "Well-documented implementation should score at least as high as minimal stub"
        )

    def test_fallback_output_gets_zero_score(self) -> None:
        """Known fallback pattern must receive overall_score == 0.0."""
        from vetinari.learning.quality_scorer import QualityScorer

        scorer = QualityScorer(adapter_manager=None)
        score = scorer.score(
            task_id="fallback_t1",
            model_id="m1",
            task_type="general",
            task_description="Do something",
            output='{"content":"","sections":[]}',
            use_llm=False,
        )
        assert score.overall_score == 0.0, "Known fallback pattern must score exactly 0.0"

    def test_empty_output_gets_zero_score(self) -> None:
        """Empty output must receive overall_score == 0.0."""
        from vetinari.learning.quality_scorer import QualityScorer

        scorer = QualityScorer(adapter_manager=None)
        score = scorer.score(
            task_id="empty_t1",
            model_id="m1",
            task_type="coding",
            task_description="Write a function",
            output="",
            use_llm=False,
        )
        assert score.overall_score == 0.0, "Empty output must score exactly 0.0"

    def test_score_distribution_across_multiple_outputs(self) -> None:
        """Scoring 5+ different outputs must produce at least 2 distinct values."""
        from vetinari.learning.quality_scorer import QualityScorer

        scorer = QualityScorer(adapter_manager=None)

        outputs = [
            "",
            "pass",
            "def f(): return 1",
            "def sort(items): return sorted(items)",
            (
                "def sort_list(items: list) -> list:\n"
                '    """Sort items."""\n'
                "    if not items:\n"
                "        return []\n"
                "    return sorted(items)\n"
            ),
        ]

        scores = {
            scorer.score(
                task_id=f"dist_multi_{i}",
                model_id="m1",
                task_type="coding",
                task_description="Sort items",
                output=out,
                use_llm=False,
            ).overall_score
            for i, out in enumerate(outputs)
        }

        assert len(scores) >= 2, f"Expected at least 2 distinct score values across 5 outputs, got: {scores}"


# ---------------------------------------------------------------------------
# Exit criterion 7: Training quality gate rejects worse model
# ---------------------------------------------------------------------------


class TestTrainingQualityGate:
    """TrainingQualityGate issues correct deploy/reject/flag verdicts."""

    def test_rejects_worse_model(self) -> None:
        """Gate must reject candidate when its quality is lower than baseline."""
        from vetinari.training.quality_gate import TrainingQualityGate

        gate = TrainingQualityGate()

        def mock_evaluate(model_id: str, tasks: list) -> list:
            if model_id == "baseline_model":
                return [{"quality": 0.85, "latency_ms": 100.0, "tokens": 50.0}] * 5
            return [{"quality": 0.50, "latency_ms": 110.0, "tokens": 55.0}] * 5

        gate._evaluate_model = mock_evaluate  # type: ignore[method-assign]

        decision = gate.evaluate("candidate_model", "baseline_model")

        assert decision.decision == "reject", (
            f"Expected 'reject' when candidate quality (0.50) < baseline (0.85), got '{decision.decision}'"
        )
        assert decision.quality_delta < 0, "quality_delta must be negative when candidate is worse"

    def test_deploys_better_model(self) -> None:
        """Gate must deploy candidate when its quality clearly exceeds baseline."""
        from vetinari.training.quality_gate import TrainingQualityGate

        gate = TrainingQualityGate()

        def mock_evaluate(model_id: str, tasks: list) -> list:
            if model_id == "baseline_model":
                return [{"quality": 0.60, "latency_ms": 100.0, "tokens": 50.0}] * 5
            return [{"quality": 0.85, "latency_ms": 95.0, "tokens": 48.0}] * 5

        gate._evaluate_model = mock_evaluate  # type: ignore[method-assign]

        decision = gate.evaluate("candidate_model", "baseline_model")

        assert decision.decision == "deploy", (
            f"Expected 'deploy' when candidate quality (0.85) > baseline (0.60) "
            f"with acceptable overhead, got '{decision.decision}'"
        )
        assert decision.quality_delta > 0, "quality_delta must be positive when candidate is better"

    def test_flags_marginal_improvement(self) -> None:
        """Gate must flag for review when improvement is too small to deploy."""
        from vetinari.training.quality_gate import TrainingQualityGate

        gate = TrainingQualityGate()

        def mock_evaluate(model_id: str, tasks: list) -> list:
            if model_id == "baseline_model":
                return [{"quality": 0.70, "latency_ms": 100.0, "tokens": 50.0}] * 5
            # quality_delta = 0.01 — below _QUALITY_DEPLOY_THRESHOLD (0.02)
            return [{"quality": 0.71, "latency_ms": 100.0, "tokens": 50.0}] * 5

        gate._evaluate_model = mock_evaluate  # type: ignore[method-assign]

        decision = gate.evaluate("candidate_model", "baseline_model")

        assert decision.decision == "flag_for_review", (
            f"Marginal improvement (delta=0.01) should be flagged for review, got '{decision.decision}'"
        )

    def test_empty_eval_set_flags_for_review(self) -> None:
        """When no eval tasks are available the gate must flag for review."""
        from vetinari.training.quality_gate import TrainingQualityGate

        gate = TrainingQualityGate()
        decision = gate.evaluate("candidate_model", "baseline_model", eval_tasks=[])

        assert decision.decision == "flag_for_review", (
            "Empty eval set must result in flag_for_review, not a hard decision"
        )


# ---------------------------------------------------------------------------
# Exit criterion 8: Shadow testing framework compares metrics correctly
# ---------------------------------------------------------------------------


class TestShadowTesting:
    """ShadowTestRunner promotes better candidates and rejects worse ones."""

    def test_promotes_better_candidate(self) -> None:
        """Candidate with higher quality and lower latency must be promoted."""
        from vetinari.learning.shadow_testing import ShadowTestRunner

        runner = ShadowTestRunner()
        test_id = runner.create_test(
            "Test prompt improvement",
            {"model": "baseline"},
            {"model": "candidate"},
            min_samples=5,
        )

        for _ in range(10):
            runner.record_production(test_id, quality=0.70, latency_ms=200.0)
            runner.record_candidate(test_id, quality=0.85, latency_ms=180.0)

        result = runner.evaluate(test_id)
        assert result["decision"] == "promote", (
            f"Candidate with better quality (0.85 vs 0.70) and lower latency "
            f"should be promoted; got decision='{result['decision']}'"
        )

    def test_rejects_worse_candidate(self) -> None:
        """Candidate with lower quality and higher latency must be rejected."""
        from vetinari.learning.shadow_testing import ShadowTestRunner

        runner = ShadowTestRunner()
        test_id = runner.create_test(
            "Test bad candidate",
            {"model": "baseline"},
            {"model": "bad_candidate"},
            min_samples=5,
        )

        for _ in range(10):
            runner.record_production(test_id, quality=0.80, latency_ms=100.0)
            runner.record_candidate(test_id, quality=0.50, latency_ms=350.0)

        result = runner.evaluate(test_id)
        assert result["decision"] == "reject", (
            f"Candidate with worse quality (0.50 vs 0.80) should be rejected; got decision='{result['decision']}'"
        )

    def test_insufficient_data_returns_early(self) -> None:
        """Evaluate before min_samples are collected must return insufficient_data."""
        from vetinari.learning.shadow_testing import ShadowTestRunner

        runner = ShadowTestRunner()
        test_id = runner.create_test(
            "Premature evaluation test",
            {"model": "baseline"},
            {"model": "candidate"},
            min_samples=20,
        )

        # Only record 3 observations (below min_samples=20)
        for _ in range(3):
            runner.record_production(test_id, quality=0.7, latency_ms=100.0)
            runner.record_candidate(test_id, quality=0.8, latency_ms=90.0)

        result = runner.evaluate(test_id)
        assert result["decision"] == "insufficient_data", (
            f"evaluate() before min_samples must return 'insufficient_data', got '{result['decision']}'"
        )

    def test_quality_delta_is_correctly_computed(self) -> None:
        """quality_delta in the promote result must equal candidate minus production avg."""
        from vetinari.learning.shadow_testing import ShadowTestRunner

        runner = ShadowTestRunner()
        test_id = runner.create_test(
            "Metric accuracy check",
            {"model": "prod"},
            {"model": "cand"},
            min_samples=5,
        )

        for _ in range(10):
            runner.record_production(test_id, quality=0.60, latency_ms=100.0)
            runner.record_candidate(test_id, quality=0.80, latency_ms=90.0)

        result = runner.evaluate(test_id)
        assert result["decision"] == "promote"
        assert abs(result["quality_delta"] - 0.20) < 0.01, (
            f"quality_delta should be ~0.20 (0.80 - 0.60), got {result['quality_delta']}"
        )
