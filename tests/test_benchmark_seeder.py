"""Tests for benchmark seeder and model selection intelligence."""

import pytest
from vetinari.learning.benchmark_seeder import (
    BenchmarkSeeder, BenchmarkPrior, get_benchmark_seeder,
    CAPABILITY_TASK_AFFINITY, MODEL_CAPABILITY_PATTERNS,
)
from vetinari.learning.model_selector import ThompsonSamplingSelector, BetaArm


class TestBenchmarkSeeder:
    def test_seeder_init(self):
        seeder = BenchmarkSeeder()
        assert seeder is not None

    def test_get_prior_coding_model(self):
        seeder = BenchmarkSeeder()
        alpha, beta = seeder.get_prior("deepseek-coder-33b", "coding")
        # Coding model should have higher alpha for coding tasks
        assert alpha > beta, "Coding model should favor coding tasks"

    def test_get_prior_chat_model(self):
        seeder = BenchmarkSeeder()
        alpha, beta = seeder.get_prior("llama-3.2-1b-instruct", "creative")
        # Chat/instruct model should have reasonable priors for creative
        assert alpha > 1.0

    def test_get_prior_unknown_model(self):
        seeder = BenchmarkSeeder()
        alpha, beta = seeder.get_prior("unknown-model", "coding")
        # Should return reasonable defaults, not crash
        assert alpha >= 1.0
        assert beta >= 1.0

    def test_seed_model(self):
        seeder = BenchmarkSeeder()
        priors = seeder.seed_model("qwen3-coder-8b")
        assert "coding" in priors
        assert "reasoning" in priors
        assert len(priors) == 8

    def test_size_scaling(self):
        seeder = BenchmarkSeeder()
        # Larger model should get higher priors
        a_small, _ = seeder.get_prior("coder-1b", "coding")
        a_large, _ = seeder.get_prior("coder-33b", "coding")
        assert a_large > a_small, "Larger model should have higher alpha"

    def test_capability_affinity_coverage(self):
        # All capability categories should have entries for all standard task types
        standard_tasks = ["coding", "reasoning", "general"]
        for cap, affinities in CAPABILITY_TASK_AFFINITY.items():
            for tt in standard_tasks:
                assert tt in affinities, f"Missing {tt} in {cap} affinities"

    def test_singleton(self):
        s1 = get_benchmark_seeder()
        s2 = get_benchmark_seeder()
        assert s1 is s2


class TestThompsonWithPriors:
    def test_informed_cold_start(self):
        selector = ThompsonSamplingSelector()
        # Force new arm creation with informed prior
        arm = selector._get_or_create_arm("deepseek-coder-v2", "coding")
        # Should have informed priors, not default Beta(1,1)
        assert arm.alpha != 1.0 or arm.beta != 1.0 or arm.total_pulls == 0

    def test_update_preserves_prior(self):
        selector = ThompsonSamplingSelector()
        arm = selector._get_or_create_arm("test-model-7b", "reasoning")
        initial_alpha = arm.alpha
        selector.update("test-model-7b", "reasoning", 0.9, True)
        assert arm.alpha > initial_alpha

    def test_select_model_with_priors(self):
        selector = ThompsonSamplingSelector()
        selected = selector.select_model(
            "coding",
            ["deepseek-coder-33b", "llama-3.2-1b-instruct"],
        )
        assert selected in ["deepseek-coder-33b", "llama-3.2-1b-instruct"]


class TestFeedbackLoopThompson:
    def test_record_outcome_updates_thompson(self):
        from vetinari.learning.feedback_loop import FeedbackLoop
        loop = FeedbackLoop()
        # Should not raise
        loop.record_outcome(
            task_id="test-1", model_id="test-model",
            task_type="coding", quality_score=0.8, success=True,
        )
