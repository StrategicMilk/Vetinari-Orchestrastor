"""
Tests for Task 29: Benchmark -> Self-Learning Integration

Verifies that benchmark results correctly feed back into:
- Quality scorer benchmark blending
- Feedback loop benchmark data acceptance
- Model selector benchmark updates (3x weight)
- Training data benchmark fields
- Episode memory benchmark boost
- Workflow learner benchmark pattern learning
"""

import json
import os
import sqlite3
import tempfile
import threading
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db(tmp_path):
    """Return a temporary SQLite DB path."""
    return str(tmp_path / "test.db")


@pytest.fixture
def temp_jsonl(tmp_path):
    """Return a temporary JSONL path."""
    return str(tmp_path / "training_data.jsonl")


# ===================================================================
# 1. Quality Scorer - Benchmark Blending
# ===================================================================

class TestQualityScorerBenchmarkBlending:
    """Test that benchmark scores blend correctly into quality scores."""

    def test_score_without_benchmark_unchanged(self, temp_db):
        """Score without benchmark_score should behave as before."""
        from vetinari.learning.quality_scorer import QualityScorer

        scorer = QualityScorer(adapter_manager=None, db_path=temp_db)
        result = scorer.score(
            task_id="t1",
            model_id="model-a",
            task_type="coding",
            task_description="Write a function",
            output="def hello(): pass",
            use_llm=False,
            benchmark_score=None,
        )
        # No benchmark blending -- method should be pure heuristic
        assert "benchmark" not in result.method
        assert "benchmark_score" not in result.dimensions

    def test_score_with_benchmark_blends(self, temp_db):
        """When benchmark_score is provided, final = computed*0.7 + benchmark*0.3."""
        from vetinari.learning.quality_scorer import QualityScorer

        scorer = QualityScorer(adapter_manager=None, db_path=temp_db)

        # First get the heuristic score without benchmark
        baseline = scorer.score(
            task_id="t2a",
            model_id="model-a",
            task_type="coding",
            task_description="Write a function",
            output="def hello(): pass",
            use_llm=False,
        )
        heuristic_score = baseline.overall_score

        # Now score with a high benchmark
        result = scorer.score(
            task_id="t2b",
            model_id="model-a",
            task_type="coding",
            task_description="Write a function",
            output="def hello(): pass",
            use_llm=False,
            benchmark_score=1.0,
        )

        expected = round(heuristic_score * 0.7 + 1.0 * 0.3, 3)
        assert result.overall_score == expected
        assert "benchmark" in result.method
        assert result.dimensions.get("benchmark_score") == 1.0

    def test_score_with_benchmark_zero(self, temp_db):
        """Benchmark score of 0.0 should pull the final score down."""
        from vetinari.learning.quality_scorer import QualityScorer

        scorer = QualityScorer(adapter_manager=None, db_path=temp_db)

        baseline = scorer.score(
            task_id="t3a",
            model_id="model-a",
            task_type="coding",
            task_description="Write a function",
            output="def hello(): pass",
            use_llm=False,
        )
        heuristic_score = baseline.overall_score

        result = scorer.score(
            task_id="t3b",
            model_id="model-a",
            task_type="coding",
            task_description="Write a function",
            output="def hello(): pass",
            use_llm=False,
            benchmark_score=0.0,
        )

        expected = round(heuristic_score * 0.7 + 0.0 * 0.3, 3)
        assert result.overall_score == expected

    def test_score_with_benchmark_convenience_method(self, temp_db):
        """score_with_benchmark() should be equivalent to score() with benchmark_score."""
        from vetinari.learning.quality_scorer import QualityScorer

        scorer = QualityScorer(adapter_manager=None, db_path=temp_db)
        result = scorer.score_with_benchmark(
            task_id="t4",
            model_id="model-a",
            task_type="coding",
            task_description="Write a function",
            output="def hello(): pass",
            benchmark_score=0.9,
            use_llm=False,
        )
        assert "benchmark" in result.method
        assert result.dimensions.get("benchmark_score") == 0.9

    def test_benchmark_score_clamped(self, temp_db):
        """Benchmark scores outside 0-1 should be clamped."""
        from vetinari.learning.quality_scorer import QualityScorer

        scorer = QualityScorer(adapter_manager=None, db_path=temp_db)
        result = scorer.score(
            task_id="t5",
            model_id="model-a",
            task_type="coding",
            task_description="Write a function",
            output="def hello(): pass",
            use_llm=False,
            benchmark_score=1.5,  # Over 1.0
        )
        # Should be clamped to 1.0
        assert result.dimensions.get("benchmark_score") == 1.0

    def test_blend_weight_constant(self):
        """BENCHMARK_BLEND_WEIGHT should be 0.3."""
        from vetinari.learning.quality_scorer import QualityScorer
        assert QualityScorer.BENCHMARK_BLEND_WEIGHT == 0.3


# ===================================================================
# 2. Feedback Loop - Benchmark Data Acceptance
# ===================================================================

class TestFeedbackLoopBenchmark:
    """Test that FeedbackLoop accepts and processes benchmark results."""

    def test_record_outcome_without_benchmark(self):
        """record_outcome without benchmark_result should work as before."""
        from vetinari.learning.feedback_loop import FeedbackLoop

        loop = FeedbackLoop()
        # Should not raise
        loop.record_outcome(
            task_id="t1",
            model_id="model-a",
            task_type="coding",
            quality_score=0.8,
            success=True,
        )

    def test_record_outcome_with_benchmark_result(self):
        """record_outcome with benchmark_result should process it."""
        from vetinari.learning.feedback_loop import FeedbackLoop

        loop = FeedbackLoop()
        benchmark_result = {
            "pass_rate": 0.85,
            "task_type": "coding",
            "suite_name": "toolbench",
            "n_trials": 20,
            "avg_score": 0.82,
        }

        # Should not raise -- _process_benchmark_result will try to import
        # Thompson selector etc., which may fail in test env, but should
        # handle exceptions gracefully
        loop.record_outcome(
            task_id="t2",
            model_id="model-a",
            task_type="coding",
            quality_score=0.8,
            success=True,
            benchmark_result=benchmark_result,
        )

    def test_record_benchmark_outcome_standalone(self):
        """record_benchmark_outcome should work as a standalone call."""
        from vetinari.learning.feedback_loop import FeedbackLoop

        loop = FeedbackLoop()
        benchmark_result = {
            "pass_rate": 0.9,
            "task_type": "research",
            "suite_name": "taskbench",
            "n_trials": 10,
            "avg_score": 0.88,
        }

        # Should not raise
        loop.record_benchmark_outcome(
            model_id="model-b",
            benchmark_result=benchmark_result,
        )

    def test_benchmark_result_propagates_to_thompson(self):
        """Benchmark result should call update_from_benchmark on Thompson selector."""
        from vetinari.learning.feedback_loop import FeedbackLoop

        loop = FeedbackLoop()
        benchmark_result = {
            "pass_rate": 0.75,
            "task_type": "coding",
            "suite_name": "swe-bench",
            "n_trials": 40,
            "avg_score": 0.70,
        }

        with patch("vetinari.learning.feedback_loop.get_feedback_loop") as _:
            with patch("vetinari.learning.model_selector.get_thompson_selector") as mock_ts:
                mock_selector = MagicMock()
                mock_ts.return_value = mock_selector

                loop._process_benchmark_result("model-c", "coding", benchmark_result)

                mock_selector.update_from_benchmark.assert_called_once_with(
                    model_id="model-c",
                    pass_rate=0.75,
                    n_trials=40,
                    task_type="coding",
                )


# ===================================================================
# 3. Model Selector - Benchmark Updates (3x Weight)
# ===================================================================

class TestModelSelectorBenchmarkUpdates:
    """Test that Thompson Sampling selector handles benchmark updates with 3x weight."""

    def _make_selector(self):
        """Create a fresh ThompsonSamplingSelector with no persisted state."""
        from vetinari.learning.model_selector import ThompsonSamplingSelector

        selector = ThompsonSamplingSelector.__new__(ThompsonSamplingSelector)
        selector._arms = {}
        return selector

    def test_update_from_benchmark_creates_arm(self):
        """update_from_benchmark should create an arm if it does not exist."""
        selector = self._make_selector()
        selector._save_state = MagicMock()  # Don't persist in tests

        selector.update_from_benchmark(
            model_id="model-x",
            pass_rate=0.8,
            n_trials=10,
            task_type="coding",
        )

        arm = selector._arms.get("model-x:coding")
        assert arm is not None
        assert arm.total_pulls == 10

    def test_update_from_benchmark_3x_weight(self):
        """Benchmark updates should apply 3x weight to alpha and beta."""
        selector = self._make_selector()
        selector._save_state = MagicMock()

        # Start with fresh arm: alpha=1, beta=1
        from vetinari.learning.model_selector import BetaArm
        arm = BetaArm(model_id="model-y", task_type="coding", alpha=1.0, beta=1.0)
        selector._arms["model-y:coding"] = arm

        initial_alpha = arm.alpha
        initial_beta = arm.beta

        # 80% pass rate on 10 trials
        selector.update_from_benchmark(
            model_id="model-y",
            pass_rate=0.8,
            n_trials=10,
            task_type="coding",
        )

        # successes = 0.8 * 10 = 8, failures = 2
        # 3x weight: alpha += 8*3 = 24, beta += 2*3 = 6
        expected_alpha = initial_alpha + 8 * 3
        expected_beta = initial_beta + 2 * 3
        assert arm.alpha == expected_alpha
        assert arm.beta == expected_beta

    def test_update_from_benchmark_perfect_pass_rate(self):
        """100% pass rate should only increase alpha."""
        selector = self._make_selector()
        selector._save_state = MagicMock()

        from vetinari.learning.model_selector import BetaArm
        arm = BetaArm(model_id="model-z", task_type="research", alpha=1.0, beta=1.0)
        selector._arms["model-z:research"] = arm

        selector.update_from_benchmark(
            model_id="model-z",
            pass_rate=1.0,
            n_trials=5,
            task_type="research",
        )

        # successes=5, failures=0, 3x: alpha += 15, beta += 0
        assert arm.alpha == 1.0 + 15.0
        assert arm.beta == 1.0  # Unchanged

    def test_update_from_benchmark_zero_pass_rate(self):
        """0% pass rate should only increase beta."""
        selector = self._make_selector()
        selector._save_state = MagicMock()

        from vetinari.learning.model_selector import BetaArm
        arm = BetaArm(model_id="model-w", task_type="coding", alpha=1.0, beta=1.0)
        selector._arms["model-w:coding"] = arm

        selector.update_from_benchmark(
            model_id="model-w",
            pass_rate=0.0,
            n_trials=5,
            task_type="coding",
        )

        # successes=0, failures=5, 3x: alpha += 0, beta += 15
        assert arm.alpha == 1.0  # Unchanged
        assert arm.beta == 1.0 + 15.0

    def test_benchmark_weight_multiplier_constant(self):
        """BENCHMARK_WEIGHT_MULTIPLIER should be 3."""
        from vetinari.learning.model_selector import ThompsonSamplingSelector
        assert ThompsonSamplingSelector.BENCHMARK_WEIGHT_MULTIPLIER == 3

    def test_update_from_benchmark_updates_total_pulls(self):
        """update_from_benchmark should increase total_pulls by n_trials."""
        selector = self._make_selector()
        selector._save_state = MagicMock()

        selector.update_from_benchmark(
            model_id="model-p",
            pass_rate=0.5,
            n_trials=20,
            task_type="coding",
        )

        arm = selector._arms["model-p:coding"]
        assert arm.total_pulls == 20


# ===================================================================
# 4. Training Data - Benchmark Fields
# ===================================================================

class TestTrainingDataBenchmarkFields:
    """Test that TrainingRecord has benchmark fields and record() accepts them."""

    def test_training_record_has_benchmark_fields(self):
        """TrainingRecord should have benchmark_suite, benchmark_pass, benchmark_score."""
        from vetinari.learning.training_data import TrainingRecord

        rec = TrainingRecord(
            record_id="test1",
            timestamp="2025-01-01",
            task="test task",
            prompt="test prompt",
            response="test response",
            score=0.8,
            model_id="model-a",
            task_type="coding",
        )

        # Default values
        assert rec.benchmark_suite == ""
        assert rec.benchmark_pass is False
        assert rec.benchmark_score == 0.0

    def test_training_record_with_benchmark_values(self):
        """TrainingRecord should accept benchmark field values."""
        from vetinari.learning.training_data import TrainingRecord

        rec = TrainingRecord(
            record_id="test2",
            timestamp="2025-01-01",
            task="test task",
            prompt="test prompt",
            response="test response",
            score=0.9,
            model_id="model-a",
            task_type="coding",
            benchmark_suite="toolbench",
            benchmark_pass=True,
            benchmark_score=0.85,
        )

        assert rec.benchmark_suite == "toolbench"
        assert rec.benchmark_pass is True
        assert rec.benchmark_score == 0.85

    def test_training_record_to_dict_includes_benchmark(self):
        """to_dict() should include benchmark fields."""
        from vetinari.learning.training_data import TrainingRecord

        rec = TrainingRecord(
            record_id="test3",
            timestamp="2025-01-01",
            task="test",
            prompt="p",
            response="r",
            score=0.8,
            model_id="m",
            task_type="coding",
            benchmark_suite="swe-bench",
            benchmark_pass=True,
            benchmark_score=0.92,
        )
        d = rec.to_dict()

        assert d["benchmark_suite"] == "swe-bench"
        assert d["benchmark_pass"] is True
        assert d["benchmark_score"] == 0.92

    def test_collector_record_accepts_benchmark_params(self, temp_jsonl):
        """TrainingDataCollector.record() should accept benchmark parameters."""
        from vetinari.learning.training_data import TrainingDataCollector

        collector = TrainingDataCollector(output_path=temp_jsonl)
        try:
            # Should not raise
            collector.record(
                task="test task",
                prompt="test prompt",
                response="test response",
                score=0.85,
                model_id="model-a",
                task_type="coding",
                benchmark_suite="toolbench",
                benchmark_pass=True,
                benchmark_score=0.88,
            )
            collector.flush()

            # Verify the record was written with benchmark fields
            with open(temp_jsonl, "r") as f:
                line = f.readline()
                data = json.loads(line)
                assert data["benchmark_suite"] == "toolbench"
                assert data["benchmark_pass"] is True
                assert data["benchmark_score"] == 0.88
        finally:
            collector.shutdown()

    def test_training_record_roundtrip_with_benchmark(self, temp_jsonl):
        """TrainingRecord should survive JSON roundtrip with benchmark fields."""
        from vetinari.learning.training_data import TrainingRecord

        rec = TrainingRecord(
            record_id="rt1",
            timestamp="2025-01-01",
            task="roundtrip test",
            prompt="p",
            response="r",
            score=0.75,
            model_id="m",
            task_type="coding",
            benchmark_suite="tau-bench",
            benchmark_pass=False,
            benchmark_score=0.45,
        )
        serialized = json.dumps(rec.to_dict())
        deserialized = json.loads(serialized)
        restored = TrainingRecord(**deserialized)

        assert restored.benchmark_suite == "tau-bench"
        assert restored.benchmark_pass is False
        assert restored.benchmark_score == 0.45


# ===================================================================
# 5. Episode Memory - Benchmark Boost
# ===================================================================

class TestEpisodeMemoryBenchmarkBoost:
    """Test that episodes with high benchmark_score get 1.5x relevance boost."""

    def test_benchmark_boost_constants(self):
        """EpisodeMemory should have boost threshold and factor."""
        from vetinari.learning.episode_memory import EpisodeMemory

        assert EpisodeMemory.BENCHMARK_BOOST_THRESHOLD == 0.8
        assert EpisodeMemory.BENCHMARK_BOOST_FACTOR == 1.5

    def test_high_benchmark_episode_boosted(self, temp_db):
        """Episode with benchmark_score > 0.8 should be ranked higher."""
        from vetinari.learning.episode_memory import EpisodeMemory

        mem = EpisodeMemory(db_path=temp_db)

        # Record two similar episodes -- one with high benchmark, one without
        ep1_id = mem.record(
            task_description="Build a cache layer for API responses",
            agent_type="BUILDER",
            task_type="coding",
            output_summary="Created CacheWrapper class",
            quality_score=0.7,
            success=True,
            model_id="model-a",
            metadata={"benchmark_score": 0.9},  # HIGH benchmark
        )

        ep2_id = mem.record(
            task_description="Build a cache system for API endpoints",
            agent_type="BUILDER",
            task_type="coding",
            output_summary="Created CacheManager class",
            quality_score=0.75,  # Slightly higher base quality
            success=True,
            model_id="model-b",
            metadata={"benchmark_score": 0.3},  # LOW benchmark
        )

        # Recall with a query similar to both
        results = mem.recall("Implement a cache layer for API", k=2)

        assert len(results) == 2
        # The high-benchmark episode should be ranked first due to 1.5x boost
        # even though ep2 has slightly higher quality_score
        assert results[0].episode_id == ep1_id

    def test_no_benchmark_no_boost(self, temp_db):
        """Episodes without benchmark_score in metadata should not be boosted."""
        from vetinari.learning.episode_memory import EpisodeMemory

        mem = EpisodeMemory(db_path=temp_db)

        mem.record(
            task_description="Write a sorting algorithm",
            agent_type="BUILDER",
            task_type="coding",
            output_summary="Implemented quicksort",
            quality_score=0.8,
            success=True,
            model_id="model-a",
            metadata={},  # No benchmark_score
        )

        results = mem.recall("sorting algorithm", k=1)
        assert len(results) == 1
        # Should still work without benchmark metadata

    def test_benchmark_below_threshold_no_boost(self, temp_db):
        """Episodes with benchmark_score <= 0.8 should NOT get boosted."""
        from vetinari.learning.episode_memory import EpisodeMemory

        mem = EpisodeMemory(db_path=temp_db)

        ep1_id = mem.record(
            task_description="Design a database schema",
            agent_type="BUILDER",
            task_type="data",
            output_summary="Schema v1",
            quality_score=0.8,
            success=True,
            metadata={"benchmark_score": 0.8},  # Exactly at threshold, not above
        )

        ep2_id = mem.record(
            task_description="Design a database schema for users",
            agent_type="BUILDER",
            task_type="data",
            output_summary="Schema v2",
            quality_score=0.85,
            success=True,
            metadata={"benchmark_score": 0.79},  # Below threshold
        )

        results = mem.recall("database schema design", k=2)
        assert len(results) == 2
        # Neither should be boosted -- normal similarity ordering applies


# ===================================================================
# 6. Workflow Learner - Benchmark Pattern Learning
# ===================================================================

class TestWorkflowLearnerBenchmark:
    """Test that WorkflowLearner learns from benchmark results."""

    def _make_learner(self):
        """Create a fresh WorkflowLearner with no persisted state."""
        from vetinari.learning.workflow_learner import WorkflowLearner

        learner = WorkflowLearner.__new__(WorkflowLearner)
        learner._patterns = {}
        return learner

    def test_learn_from_benchmark_creates_pattern(self):
        """learn_from_benchmark should create/update a pattern for the domain."""
        learner = self._make_learner()
        learner._save_patterns = MagicMock()

        benchmark_result = {
            "suite_name": "toolbench",
            "task_type": "coding",
            "pass_rate": 0.85,
            "avg_score": 0.82,
            "total_cases": 20,
            "passed_cases": 17,
            "metadata": {
                "depth": 4,
                "breadth": 3,
                "agents_used": ["BUILDER", "EVALUATOR", "TEST_AUTOMATION"],
            },
        }

        learner.learn_from_benchmark(benchmark_result)

        # Should have created a pattern for "coding" domain
        assert "coding" in learner._patterns
        pattern = learner._patterns["coding"]
        assert pattern.sample_count == 1

    def test_learn_from_benchmark_skips_low_pass_rate(self):
        """learn_from_benchmark should skip results with pass_rate < 0.3."""
        learner = self._make_learner()
        learner._save_patterns = MagicMock()

        benchmark_result = {
            "suite_name": "hard-bench",
            "task_type": "coding",
            "pass_rate": 0.1,  # Too low
            "avg_score": 0.05,
            "total_cases": 20,
            "passed_cases": 2,
        }

        learner.learn_from_benchmark(benchmark_result)

        # Should NOT create a pattern
        assert "coding" not in learner._patterns

    def test_learn_from_benchmark_updates_existing_pattern(self):
        """learn_from_benchmark should update existing patterns via EMA."""
        learner = self._make_learner()
        learner._save_patterns = MagicMock()

        # First benchmark
        learner.learn_from_benchmark({
            "suite_name": "bench1",
            "task_type": "coding",
            "pass_rate": 0.8,
            "avg_score": 0.75,
            "total_cases": 10,
            "passed_cases": 8,
            "metadata": {"depth": 3, "breadth": 2, "agents_used": ["BUILDER"]},
        })

        first_count = learner._patterns["coding"].sample_count

        # Second benchmark
        learner.learn_from_benchmark({
            "suite_name": "bench2",
            "task_type": "coding",
            "pass_rate": 0.9,
            "avg_score": 0.88,
            "total_cases": 10,
            "passed_cases": 9,
            "metadata": {"depth": 5, "breadth": 4, "agents_used": ["EVALUATOR"]},
        })

        assert learner._patterns["coding"].sample_count == first_count + 1

    def test_learn_from_benchmark_extracts_case_patterns(self):
        """Detailed per-case results should influence preferred_agents."""
        learner = self._make_learner()
        learner._save_patterns = MagicMock()

        # First create a base pattern
        learner.learn_from_benchmark({
            "suite_name": "bench1",
            "task_type": "coding",
            "pass_rate": 0.8,
            "avg_score": 0.75,
            "total_cases": 2,
            "passed_cases": 2,
            "metadata": {"depth": 3, "breadth": 2, "agents_used": ["BUILDER"]},
        })

        # Now add detailed results
        learner.learn_from_benchmark({
            "suite_name": "bench2",
            "task_type": "coding",
            "pass_rate": 0.9,
            "avg_score": 0.85,
            "total_cases": 3,
            "passed_cases": 3,
            "results": [
                {"case_id": "c1", "passed": True, "score": 0.9, "metadata": {"agents_used": ["BUILDER", "SECURITY_AUDITOR"]}},
                {"case_id": "c2", "passed": True, "score": 0.85, "metadata": {"agents_used": ["BUILDER", "TEST_AUTOMATION"]}},
                {"case_id": "c3", "passed": True, "score": 0.8, "metadata": {"agents_used": ["BUILDER"]}},
            ],
            "metadata": {"depth": 4, "breadth": 3, "agents_used": ["BUILDER"]},
        })

        pattern = learner._patterns["coding"]
        # BUILDER should be the most frequent agent
        assert "BUILDER" in pattern.preferred_agents

    def test_domain_inference_from_task_type(self):
        """_infer_domain_from_task_type should map benchmark task types to domains."""
        learner = self._make_learner()

        assert learner._infer_domain_from_task_type("coding") == "coding"
        assert learner._infer_domain_from_task_type("research") == "research"
        assert learner._infer_domain_from_task_type("documentation") == "docs"
        assert learner._infer_domain_from_task_type("data") == "data"
        assert learner._infer_domain_from_task_type("testing") == "coding"


# ===================================================================
# 7. Prompt Evolver - Benchmark Validation
# ===================================================================

class TestPromptEvolverBenchmarkValidation:
    """Test the benchmark validation gate for prompt promotion."""

    def test_benchmark_pass_threshold_constant(self):
        """BENCHMARK_PASS_THRESHOLD should be 0.5."""
        from vetinari.learning.prompt_evolver import PromptEvolver
        assert PromptEvolver.BENCHMARK_PASS_THRESHOLD == 0.5

    def test_validate_variant_passes_when_no_benchmark_suite(self):
        """Validation should pass (return True) when benchmarks are not available."""
        from vetinari.learning.prompt_evolver import PromptEvolver, PromptVariant

        evolver = PromptEvolver(adapter_manager=None)
        variant = PromptVariant(
            variant_id="test_v1",
            agent_type="NONEXISTENT_AGENT",
            prompt_text="Test prompt",
        )

        # Should return True when benchmark suite can't run for this agent
        mock_suite = MagicMock()
        mock_result = MagicMock()
        mock_result.cases_run = 0  # No cases for this agent
        mock_suite.return_value.run_agent.return_value = mock_result

        import vetinari.benchmarks as _bm
        with patch.dict("sys.modules", {"vetinari.benchmarks.suite": mock_suite}):
            _bm.suite = mock_suite  # Python 3.9/3.10 need attribute on parent
            try:
                with patch("vetinari.benchmarks.suite.BenchmarkSuite", mock_suite, create=True):
                    result = evolver._validate_variant_with_benchmark(variant)
                    assert result is True
            finally:
                if hasattr(_bm, "suite"):
                    del _bm.suite

    def test_validate_variant_fails_below_threshold(self):
        """Validation should fail when benchmark pass rate < threshold."""
        from vetinari.learning.prompt_evolver import PromptEvolver, PromptVariant

        evolver = PromptEvolver(adapter_manager=None)
        variant = PromptVariant(
            variant_id="test_v2",
            agent_type="BUILDER",
            prompt_text="Test prompt",
        )

        mock_suite_cls = MagicMock()
        mock_result = MagicMock()
        mock_result.cases_run = 10
        mock_result.cases_passed = 3  # 30% pass rate -- below 50% threshold
        mock_result.avg_score = 0.3
        mock_suite_cls.return_value.run_agent.return_value = mock_result

        mock_module = MagicMock()
        mock_module.BenchmarkSuite = mock_suite_cls

        with patch.dict("sys.modules", {"vetinari.benchmarks.suite": mock_module}):
            result = evolver._validate_variant_with_benchmark(variant)
            assert result is False

    def test_validate_variant_passes_above_threshold(self):
        """Validation should pass when benchmark pass rate >= threshold."""
        from vetinari.learning.prompt_evolver import PromptEvolver, PromptVariant

        evolver = PromptEvolver(adapter_manager=None)
        variant = PromptVariant(
            variant_id="test_v3",
            agent_type="BUILDER",
            prompt_text="Test prompt",
        )

        mock_suite_cls = MagicMock()
        mock_result = MagicMock()
        mock_result.cases_run = 10
        mock_result.cases_passed = 8  # 80% pass rate -- above threshold
        mock_result.avg_score = 0.82
        mock_suite_cls.return_value.run_agent.return_value = mock_result

        mock_module = MagicMock()
        mock_module.BenchmarkSuite = mock_suite_cls

        with patch.dict("sys.modules", {"vetinari.benchmarks.suite": mock_module}):
            result = evolver._validate_variant_with_benchmark(variant)
            assert result is True

    def test_validate_variant_handles_import_error(self):
        """Should return True (fail open) if BenchmarkSuite import fails."""
        from vetinari.learning.prompt_evolver import PromptEvolver, PromptVariant

        evolver = PromptEvolver(adapter_manager=None)
        variant = PromptVariant(
            variant_id="test_v4",
            agent_type="BUILDER",
            prompt_text="Test prompt",
        )

        # Force ImportError on the from-import inside the method
        with patch.dict("sys.modules", {"vetinari.benchmarks.suite": None}):
            result = evolver._validate_variant_with_benchmark(variant)
            # Should fail open (return True) on import error
            assert result is True


# ===================================================================
# 8. Integration: End-to-End Flow
# ===================================================================

class TestEndToEndIntegration:
    """Test the full benchmark -> learning integration flow."""

    def test_benchmark_result_flows_through_system(self, temp_db):
        """A benchmark result should be processable by all subsystems."""
        # Simulate a benchmark result dict (as produced by BenchmarkReport.summary_dict())
        benchmark_result = {
            "suite_name": "toolbench",
            "task_type": "coding",
            "pass_rate": 0.85,
            "avg_score": 0.82,
            "total_cases": 20,
            "passed_cases": 17,
            "n_trials": 20,
            "metadata": {
                "depth": 4,
                "breadth": 3,
                "agents_used": ["BUILDER", "EVALUATOR"],
            },
        }

        # 1. Quality scorer can blend benchmark
        from vetinari.learning.quality_scorer import QualityScorer
        scorer = QualityScorer(adapter_manager=None, db_path=temp_db)
        qs = scorer.score(
            task_id="e2e_1",
            model_id="model-a",
            task_type="coding",
            task_description="Build feature",
            output="def feature(): pass",
            use_llm=False,
            benchmark_score=benchmark_result["avg_score"],
        )
        assert "benchmark" in qs.method

        # 2. Training record can store benchmark fields
        from vetinari.learning.training_data import TrainingRecord
        rec = TrainingRecord(
            record_id="e2e_tr",
            timestamp="2025-01-01",
            task="test",
            prompt="p",
            response="r",
            score=qs.overall_score,
            model_id="model-a",
            task_type="coding",
            benchmark_suite=benchmark_result["suite_name"],
            benchmark_pass=benchmark_result["pass_rate"] >= 0.5,
            benchmark_score=benchmark_result["avg_score"],
        )
        assert rec.benchmark_suite == "toolbench"
        assert rec.benchmark_pass is True

        # 3. Workflow learner can learn from benchmark
        from vetinari.learning.workflow_learner import WorkflowLearner
        learner = WorkflowLearner.__new__(WorkflowLearner)
        learner._patterns = {}
        learner._save_patterns = MagicMock()
        learner.learn_from_benchmark(benchmark_result)
        assert "coding" in learner._patterns
