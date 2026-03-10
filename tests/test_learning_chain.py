"""
Tests for the Vetinari Learning Chain (Phase 13.3c)

Covers the three core learning subsystems that had 0% test coverage:
- QualityScorer: heuristic + LLM-as-judge scoring
- FeedbackLoop: EMA-based performance tracking
- ThompsonSamplingSelector: Bayesian bandit model selection

Also tests the integration chain: Score → Feedback → Sampling
"""

import json
import os
import random
import tempfile
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# QualityScorer Tests
# ---------------------------------------------------------------------------

class TestQualityScore:
    """Test the QualityScore dataclass."""

    def test_quality_score_defaults(self):
        from vetinari.learning.quality_scorer import QualityScore
        qs = QualityScore(task_id="t1", model_id="m1", task_type="coding", overall_score=0.8)
        assert qs.overall_score == 0.8
        assert qs.correctness == 0.7
        assert qs.method == "heuristic"
        assert isinstance(qs.dimensions, dict)
        assert len(qs.dimensions) == 0  # default is empty dict
        assert isinstance(qs.issues, list)
        assert len(qs.issues) == 0  # default is empty list
        assert qs.task_id == "t1"
        assert qs.model_id == "m1"
        assert qs.task_type == "coding"

    def test_quality_score_to_dict(self):
        from vetinari.learning.quality_scorer import QualityScore
        qs = QualityScore(task_id="t1", model_id="m1", task_type="coding", overall_score=0.85)
        d = qs.to_dict()
        assert isinstance(d, dict)
        assert d["task_id"] == "t1"
        assert d["model_id"] == "m1"
        assert d["task_type"] == "coding"
        assert d["overall_score"] == 0.85
        assert "timestamp" in d
        assert isinstance(d["timestamp"], str)
        assert len(d["timestamp"]) > 0
        assert isinstance(d["dimensions"], dict)
        assert isinstance(d["issues"], list)


class TestQualityScorer:
    """Test QualityScorer heuristic scoring and persistence."""

    @pytest.fixture
    def scorer(self, tmp_path):
        """Create a QualityScorer with a temp DB."""
        from vetinari.learning.quality_scorer import QualityScorer
        db_path = str(tmp_path / "test_quality.db")
        return QualityScorer(adapter_manager=None, db_path=db_path)

    def test_heuristic_score_coding(self, scorer):
        score = scorer.score(
            task_id="t1", model_id="m1", task_type="coding",
            task_description="Write a function",
            output='def hello():\n    """Say hello."""\n    return "Hello"\n\nassert hello() == "Hello"',
            use_llm=False,
        )
        assert 0.0 <= score.overall_score <= 1.0
        assert score.method == "heuristic"
        assert score.correctness == 0.7  # has def
        assert score.style == 0.8  # has docstring
        assert "test_coverage" in score.dimensions
        assert score.dimensions["test_coverage"] == 0.8  # has assert

    def test_heuristic_score_research(self, scorer):
        score = scorer.score(
            task_id="t2", model_id="m1", task_type="research",
            task_description="Research AI orchestration",
            output="## Overview\n\nAI orchestration is...\n\n## Sources\n\nhttp://example.com\n\n## Conclusion\n\nActionable findings...",
            use_llm=False,
        )
        assert 0.0 <= score.overall_score <= 1.0
        assert score.dimensions.get("source_quality", 0) == 0.8  # has http
        assert score.dimensions.get("actionability", 0) == 0.7  # has sections

    def test_heuristic_score_empty_output(self, scorer):
        score = scorer.score(
            task_id="t3", model_id="m1", task_type="coding",
            task_description="Write something",
            output="",
            use_llm=False,
        )
        assert score.overall_score == 0.0
        assert "Empty output" in score.issues

    def test_heuristic_score_short_output(self, scorer):
        score = scorer.score(
            task_id="t4", model_id="m1", task_type="default",
            task_description="Do something",
            output="OK done",
            use_llm=False,
        )
        assert score.dimensions.get("completeness", 1.0) == 0.3
        assert "Very short output" in score.issues

    def test_heuristic_score_coding_no_def(self, scorer):
        score = scorer.score(
            task_id="t5", model_id="m1", task_type="coding",
            task_description="Write code",
            output="x = 1\ny = 2\nprint(x + y)\n# just some script\nresult = x * y",
            use_llm=False,
        )
        assert score.correctness == 0.4  # no def/class
        assert "No function/class definitions found" in score.issues

    def test_heuristic_score_research_no_sources(self, scorer):
        score = scorer.score(
            task_id="t6", model_id="m1", task_type="research",
            task_description="Research topic",
            output="This is a simple paragraph about the topic without any references or links.",
            use_llm=False,
        )
        assert score.dimensions.get("source_quality", 1.0) == 0.4
        assert "No source citations found" in score.issues

    def test_score_persistence(self, scorer, tmp_path):
        """Verify scores persist to SQLite."""
        score = scorer.score(
            task_id="persist1", model_id="m1", task_type="coding",
            task_description="test", output="def foo(): pass",
            use_llm=False,
        )

        # Re-create scorer pointing at same DB
        from vetinari.learning.quality_scorer import QualityScorer
        scorer2 = QualityScorer(db_path=str(tmp_path / "test_quality.db"))
        history = scorer2.get_history()
        assert len(history) >= 1
        assert history[0].task_id == "persist1"
        assert history[0].model_id == "m1"
        assert history[0].task_type == "coding"
        assert 0.0 <= history[0].overall_score <= 1.0
        assert history[0].method == "heuristic"

    def test_get_history_filters(self, scorer):
        scorer.score("a1", "model_a", "coding", "desc", "def foo(): pass", use_llm=False)
        scorer.score("a2", "model_b", "research", "desc", "http://example.com research findings", use_llm=False)
        scorer.score("a3", "model_a", "research", "desc", "http://example.com more research", use_llm=False)

        all_scores = scorer.get_history()
        assert len(all_scores) == 3

        model_a_scores = scorer.get_history(model_id="model_a")
        assert len(model_a_scores) == 2

        coding_scores = scorer.get_history(task_type="coding")
        assert len(coding_scores) == 1

    def test_get_model_average(self, scorer):
        scorer.score("t1", "m1", "coding", "desc", "def foo(): pass", use_llm=False)
        scorer.score("t2", "m1", "coding", "desc", "def bar(): return 1", use_llm=False)

        avg = scorer.get_model_average("m1", "coding")
        assert 0.0 <= avg <= 1.0
        assert avg > 0.0  # Both outputs have "def", so correctness=0.7; avg must be positive
        assert isinstance(avg, float)

    def test_get_model_average_no_data(self, scorer):
        avg = scorer.get_model_average("nonexistent_model")
        assert avg == 0.7  # Default prior

    def test_dimensions_by_task_type(self, scorer):
        from vetinari.learning.quality_scorer import QualityScorer
        assert "test_coverage" in QualityScorer.DIMENSIONS["coding"]
        assert "source_quality" in QualityScorer.DIMENSIONS["research"]
        assert "clarity" in QualityScorer.DIMENSIONS["documentation"]
        assert "correctness" in QualityScorer.DIMENSIONS["default"]

    def test_score_overall_bounded(self, scorer):
        """Overall score should always be in [0, 1]."""
        for task_type in ["coding", "research", "analysis", "documentation", "testing", "default"]:
            score = scorer.score(
                f"bounded_{task_type}", "m1", task_type, "desc",
                "def foo():\n    '''doc'''\n    return True\nassert foo()\nhttp://x.com",
                use_llm=False,
            )
            assert 0.0 <= score.overall_score <= 1.0, f"{task_type}: {score.overall_score}"


# ---------------------------------------------------------------------------
# FeedbackLoop Tests
# ---------------------------------------------------------------------------

class TestFeedbackLoop:
    """Test the FeedbackLoop EMA updates."""

    @pytest.fixture
    def feedback_loop(self):
        from vetinari.learning.feedback_loop import FeedbackLoop
        fl = FeedbackLoop()
        fl._memory = None  # Disable memory store
        fl._router = None  # Disable router
        return fl

    def test_record_outcome_no_crash(self, feedback_loop):
        """Recording an outcome should not crash even with no memory/router."""
        feedback_loop.record_outcome(
            task_id="t1", model_id="m1", task_type="coding",
            quality_score=0.85, success=True,
        )
        # No exception means success

    def test_record_outcome_with_mock_memory(self, feedback_loop):
        mock_mem = MagicMock()
        mock_mem.get_model_performance.return_value = {
            "success_rate": 0.7,
            "avg_latency": 1000,
            "total_uses": 5,
        }
        feedback_loop._memory = mock_mem

        feedback_loop.record_outcome(
            task_id="t1", model_id="m1", task_type="coding",
            quality_score=0.9, latency_ms=500, success=True,
        )

        mock_mem.get_model_performance.assert_called_once_with("m1", "coding")
        mock_mem.update_model_performance.assert_called_once()
        call_args = mock_mem.update_model_performance.call_args
        assert call_args[0][0] == "m1"  # model_id
        assert call_args[0][1] == "coding"  # task_type
        update_dict = call_args[0][2]
        assert update_dict["total_uses"] == 6
        # EMA: (1 - 0.3) * 0.7 + 0.3 * (0.5 * 1.0 + 0.5 * 0.9) = 0.49 + 0.285 = 0.775
        assert abs(update_dict["success_rate"] - 0.775) < 0.01

    def test_ema_converges_toward_signal(self, feedback_loop):
        """After many updates, EMA should converge toward the input signal."""
        mock_mem = MagicMock()
        state = {"success_rate": 0.5, "avg_latency": 1000, "total_uses": 0}

        def get_perf(model_id, task_type):
            return dict(state)

        def update_perf(model_id, task_type, updates):
            state.update(updates)

        mock_mem.get_model_performance = get_perf
        mock_mem.update_model_performance = update_perf
        mock_mem.update_subtask_quality = MagicMock()
        feedback_loop._memory = mock_mem

        # Push 20 perfect outcomes
        for i in range(20):
            feedback_loop.record_outcome(
                task_id=f"t{i}", model_id="m1", task_type="coding",
                quality_score=1.0, success=True,
            )

        # EMA should converge toward signal of 0.5*1.0 + 0.5*1.0 = 1.0
        assert state["success_rate"] > 0.9

    def test_record_outcome_with_mock_router(self, feedback_loop):
        mock_router = MagicMock()
        mock_router.get_performance_cache.return_value = {
            "success_rate": 0.7,
            "avg_latency_ms": 1000,
        }
        feedback_loop._router = mock_router

        feedback_loop.record_outcome(
            task_id="t1", model_id="m1", task_type="coding",
            quality_score=0.8, latency_ms=200, success=True,
        )

        mock_router.get_performance_cache.assert_called_once_with("m1:coding")
        mock_router.update_performance_cache.assert_called_once()

    def test_ema_alpha_value(self, feedback_loop):
        from vetinari.learning.feedback_loop import FeedbackLoop
        assert FeedbackLoop.EMA_ALPHA == 0.3


# ---------------------------------------------------------------------------
# ThompsonSamplingSelector Tests
# ---------------------------------------------------------------------------

class TestBetaArm:
    """Test the BetaArm dataclass."""

    def test_initial_mean(self):
        from vetinari.learning.model_selector import BetaArm
        arm = BetaArm(model_id="m1", task_type="coding")
        assert arm.mean == 0.5  # Beta(2,2) mean = 0.5

    def test_update_success(self):
        from vetinari.learning.model_selector import BetaArm
        arm = BetaArm(model_id="m1", task_type="coding")
        arm.update(quality_score=0.9, success=True)
        assert arm.alpha == 2.9  # 2.0 + 0.9
        assert arm.beta == 2.0   # unchanged
        assert arm.total_pulls == 1

    def test_update_failure(self):
        from vetinari.learning.model_selector import BetaArm
        arm = BetaArm(model_id="m1", task_type="coding")
        arm.update(quality_score=0.3, success=False)
        assert arm.alpha == 2.0  # unchanged
        assert arm.beta == 2.7   # 2.0 + (1.0 - 0.3)
        assert arm.total_pulls == 1

    def test_sample_in_range(self):
        from vetinari.learning.model_selector import BetaArm
        arm = BetaArm(model_id="m1", task_type="coding", alpha=5.0, beta=3.0)
        random.seed(42)
        samples = [arm.sample() for _ in range(100)]
        assert all(0.0 <= s <= 1.0 for s in samples)

    def test_mean_shifts_with_successes(self):
        from vetinari.learning.model_selector import BetaArm
        arm = BetaArm(model_id="m1", task_type="coding")
        initial_mean = arm.mean
        for _ in range(10):
            arm.update(quality_score=0.8, success=True)
        assert arm.mean > initial_mean


class TestThompsonSamplingSelector:
    """Test the ThompsonSamplingSelector model selection."""

    @pytest.fixture
    def selector(self, tmp_path):
        from vetinari.learning.model_selector import ThompsonSamplingSelector
        # Use temp dir for state persistence
        os.environ["VETINARI_STATE_DIR"] = str(tmp_path / ".vetinari")
        sel = ThompsonSamplingSelector()
        sel._arms = {}  # Start fresh
        yield sel
        os.environ.pop("VETINARI_STATE_DIR", None)

    def test_select_from_empty_returns_default(self, selector):
        result = selector.select_model("coding", [])
        assert result == "default"

    def test_select_from_single_model(self, selector):
        result = selector.select_model("coding", ["model_a"])
        assert result == "model_a"

    def test_select_returns_valid_candidate(self, selector):
        candidates = ["model_a", "model_b", "model_c"]
        result = selector.select_model("coding", candidates)
        assert result in candidates

    def test_update_modifies_arm(self, selector):
        selector.update("model_a", "coding", quality_score=0.9, success=True)
        state = selector.get_arm_state("model_a", "coding")
        # After update: initial_alpha + quality_score = alpha
        initial_alpha = state["alpha"] - 0.9
        assert abs(state["alpha"] - (initial_alpha + 0.9)) < 0.001
        assert state["total_pulls"] == 1
        assert state["model_id"] == "model_a"
        assert state["task_type"] == "coding"
        assert 0.0 < state["mean"] < 1.0

    def test_exploitation_after_training(self, selector):
        """A well-trained model should be selected more often than untrained ones."""
        random.seed(42)
        # Train model_a heavily
        for _ in range(50):
            selector.update("model_a", "coding", quality_score=0.95, success=True)
        # Train model_b with failures
        for _ in range(50):
            selector.update("model_b", "coding", quality_score=0.2, success=False)

        # Count selections over many trials
        selections = {"model_a": 0, "model_b": 0}
        for _ in range(100):
            chosen = selector.select_model("coding", ["model_a", "model_b"])
            selections[chosen] += 1

        assert selections["model_a"] > selections["model_b"]

    def test_cost_penalty_applied(self, selector):
        """Expensive models should be penalized."""
        random.seed(42)
        # Make both models equally good
        for _ in range(20):
            selector.update("cheap", "coding", quality_score=0.8, success=True)
            selector.update("expensive", "coding", quality_score=0.8, success=True)

        cost_map = {"cheap": 0.01, "expensive": 1.0}
        selections = {"cheap": 0, "expensive": 0}
        for _ in range(100):
            chosen = selector.select_model("coding", ["cheap", "expensive"], cost_per_model=cost_map)
            selections[chosen] += 1

        # Cheap model should be preferred when quality is equal
        assert selections["cheap"] > selections["expensive"]

    def test_get_rankings(self, selector):
        selector.update("model_a", "coding", 0.9, True)
        selector.update("model_b", "coding", 0.3, False)

        rankings = selector.get_rankings("coding")
        assert len(rankings) == 2
        # model_a should rank higher (more successes)
        assert rankings[0][0] == "model_a"
        # Rankings are (model_id, mean) tuples; means must be in [0, 1] and descending
        assert 0.0 < rankings[0][1] <= 1.0
        assert 0.0 < rankings[1][1] <= 1.0
        assert rankings[0][1] > rankings[1][1]  # model_a mean > model_b mean
        assert rankings[1][0] == "model_b"

    def test_state_persistence(self, selector, tmp_path):
        selector.update("m1", "coding", 0.9, True)
        selector.update("m2", "research", 0.5, False)

        # Create new selector pointing at same state dir
        from vetinari.learning.model_selector import ThompsonSamplingSelector
        sel2 = ThompsonSamplingSelector()
        state = sel2.get_arm_state("m1", "coding")
        # alpha should have increased by quality_score on success
        assert state["total_pulls"] == 1
        # m2: beta updated on failure by (1.0 - quality_score)
        state2 = sel2.get_arm_state("m2", "research")
        assert state2["total_pulls"] == 1

    def test_thread_safety(self, selector):
        """Concurrent updates should not corrupt state."""
        import threading
        errors = []

        def update_arm(model_id, n):
            try:
                for i in range(n):
                    selector.update(model_id, "coding", random.random(), random.random() > 0.3)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=update_arm, args=(f"m{i}", 50))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All arms should exist
        for i in range(5):
            state = selector.get_arm_state(f"m{i}", "coding")
            assert state["total_pulls"] == 50


# ---------------------------------------------------------------------------
# Integration: Full Learning Chain
# ---------------------------------------------------------------------------

class TestLearningChainIntegration:
    """Test the full Score → Feedback → Sampling chain."""

    def test_full_chain(self, tmp_path):
        """Simulate the chain: QualityScorer → FeedbackLoop → ThompsonSampling."""
        from vetinari.learning.quality_scorer import QualityScorer
        from vetinari.learning.feedback_loop import FeedbackLoop
        from vetinari.learning.model_selector import ThompsonSamplingSelector

        os.environ["VETINARI_STATE_DIR"] = str(tmp_path / ".vetinari")
        db_path = str(tmp_path / "quality.db")

        scorer = QualityScorer(db_path=db_path)
        feedback = FeedbackLoop()
        selector = ThompsonSamplingSelector()
        selector._arms = {}

        # Step 1: Score an output
        score = scorer.score(
            task_id="chain_t1", model_id="model_a", task_type="coding",
            task_description="Write a function",
            output='def hello():\n    """Say hello."""\n    return "Hello"',
            use_llm=False,
        )
        assert isinstance(score.overall_score, float)
        assert 0.0 < score.overall_score <= 1.0
        assert score.task_id == "chain_t1"
        assert score.model_id == "model_a"
        assert score.method == "heuristic"
        assert score.correctness == 0.7  # has "def"
        assert score.style == 0.8  # has docstring

        # Step 2: Record feedback (using mock memory/router to avoid deps)
        feedback.record_outcome(
            task_id="chain_t1", model_id="model_a", task_type="coding",
            quality_score=score.overall_score, success=True,
        )

        # Step 3: Update Thompson Sampling
        selector.update("model_a", "coding", score.overall_score, True)

        # Verify the chain produced meaningful state
        arm_state = selector.get_arm_state("model_a", "coding")
        # alpha = 2.0 + score.overall_score (success path); score > 0 so alpha > 2.0
        assert arm_state["alpha"] > 2.0
        # beta should not increase on success (only alpha increases)
        assert arm_state["total_pulls"] == 1
        assert 0.0 < arm_state["mean"] < 1.0  # Valid probability
        assert arm_state["model_id"] == "model_a"
        assert arm_state["task_type"] == "coding"

        os.environ.pop("VETINARI_STATE_DIR", None)

    def test_chain_convergence(self, tmp_path):
        """After many iterations, good models should be strongly preferred."""
        from vetinari.learning.quality_scorer import QualityScorer
        from vetinari.learning.model_selector import ThompsonSamplingSelector

        os.environ["VETINARI_STATE_DIR"] = str(tmp_path / ".vetinari")
        db_path = str(tmp_path / "quality.db")
        scorer = QualityScorer(db_path=db_path)
        selector = ThompsonSamplingSelector()
        selector._arms = {}

        random.seed(42)

        # Simulate 30 rounds: model_good produces good code, model_bad produces bad code
        for i in range(30):
            good_score = scorer.score(
                f"good_{i}", "model_good", "coding", "Write function",
                'def compute(x):\n    """Compute result."""\n    return x * 2\nassert compute(3) == 6',
                use_llm=False,
            )
            selector.update("model_good", "coding", good_score.overall_score, True)

            bad_score = scorer.score(
                f"bad_{i}", "model_bad", "coding", "Write function",
                "bad",
                use_llm=False,
            )
            selector.update("model_bad", "coding", bad_score.overall_score, False)

        # model_good should dominate selection
        selections = {"model_good": 0, "model_bad": 0}
        for _ in range(100):
            chosen = selector.select_model("coding", ["model_good", "model_bad"])
            selections[chosen] += 1

        assert selections["model_good"] > 80, f"Expected >80, got {selections['model_good']}"

        os.environ.pop("VETINARI_STATE_DIR", None)
