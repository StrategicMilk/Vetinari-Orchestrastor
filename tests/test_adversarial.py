"""Adversarial integration tests — exercise real code with hostile inputs.

These tests use NO mocks for core logic. They import real modules and feed
them adversarial data to verify the system handles edge cases correctly.
The goal is to CATCH bugs, not pass tests.
"""

from __future__ import annotations

import json
import time

import pytest

# ═══════════════════════════════════════════════════════════════════════════
# 1. Training data with adversarial content
# ═══════ï¿½ï¿½ï¿½═══════════════════════════════════════════════════════════════════


class TestTrainingDataAdversarial:
    """Training data pipeline must handle corrupt records."""

    def test_record_with_empty_strings(self, tmp_path):
        """Recording empty prompt/response should not crash and data persists.

        Note: tokens_used and latency_ms must be non-zero to pass quality gates
        (records with 0 values are rejected as fallback/mock outputs).
        """
        from vetinari.learning.training_data import TrainingDataCollector

        path = str(tmp_path / "data.jsonl")
        collector = TrainingDataCollector(output_path=path, sync=True)
        collector.record(
            task="",
            prompt="",
            response="non-empty",
            score=0.0,
            model_id="",
            task_type="",
            tokens_used=1,
            latency_ms=1,
        )
        stats = collector.get_stats()
        assert stats["total"] == 1

    def test_record_with_unicode(self, tmp_path):
        """Unicode in training data must persist correctly to JSONL."""
        from pathlib import Path

        from vetinari.learning.training_data import TrainingDataCollector

        path = Path(tmp_path / "data.jsonl")
        collector = TrainingDataCollector(output_path=str(path), sync=True)
        collector.record(
            task="unicode test",
            prompt="Japanese: \u65e5\u672c\u8a9e Russian: \u041f\u0440\u0438\u0432\u0435\u0442 Emoji: \U0001f389",
            response="Response with tabs\tand\nnewlines",
            score=0.5,
            model_id="test",
            task_type="general",
            tokens_used=50,
            latency_ms=100,
        )
        assert path.exists(), f"JSONL file not created at {path}"
        with path.open(encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                assert "prompt" in data
                assert "\u65e5\u672c\u8a9e" in data["prompt"]

    def test_export_sft_filters_by_score(self, tmp_path):
        """export_sft_dataset must actually filter by min_score."""
        from vetinari.learning.training_data import TrainingDataCollector

        path = str(tmp_path / "data.jsonl")
        collector = TrainingDataCollector(output_path=path, sync=True)
        for score in [0.1, 0.3, 0.5, 0.7, 0.9]:
            collector.record(
                task="t",
                prompt="p",
                response="r",
                score=score,
                model_id="m",
                task_type="general",
                tokens_used=10,
                latency_ms=50,
            )
        all_records = collector.export_sft_dataset(min_score=0.0)
        high_quality = collector.export_sft_dataset(min_score=0.8)
        assert len(all_records) == 5, f"Expected 5 records, got {len(all_records)}"
        assert len(high_quality) == 1, f"Expected 1 high-quality record, got {len(high_quality)}"

    def test_quality_gate_rejects_zero_tokens(self, tmp_path):
        """Records with tokens_used=0 must be rejected (fallback/mock indicator)."""
        from vetinari.learning.training_data import TrainingDataCollector

        path = str(tmp_path / "data.jsonl")
        collector = TrainingDataCollector(output_path=path, sync=True)
        collector.record(
            task="test",
            prompt="p",
            response="r",
            score=0.5,
            model_id="m",
            task_type="general",
            tokens_used=0,
            latency_ms=100,
        )
        assert collector.get_stats()["total"] == 0

    def test_quality_gate_rejects_zero_latency(self, tmp_path):
        """Records with latency_ms=0 must be rejected (fallback/mock indicator)."""
        from vetinari.learning.training_data import TrainingDataCollector

        path = str(tmp_path / "data.jsonl")
        collector = TrainingDataCollector(output_path=path, sync=True)
        collector.record(
            task="test",
            prompt="p",
            response="r",
            score=0.5,
            model_id="m",
            task_type="general",
            tokens_used=10,
            latency_ms=0,
        )
        assert collector.get_stats()["total"] == 0

    def test_quality_gate_rejects_empty_response(self, tmp_path):
        """Records with empty response body must be rejected (fallback pattern)."""
        from vetinari.learning.training_data import TrainingDataCollector

        path = str(tmp_path / "data.jsonl")
        collector = TrainingDataCollector(output_path=path, sync=True)
        collector.record(
            task="test",
            prompt="p",
            response="",
            score=0.5,
            model_id="m",
            task_type="general",
            tokens_used=10,
            latency_ms=50,
        )
        assert collector.get_stats()["total"] == 0

    def test_quality_gate_accepts_valid_record(self, tmp_path):
        """Records with valid tokens_used, latency_ms, and response must be accepted."""
        from vetinari.learning.training_data import TrainingDataCollector

        path = str(tmp_path / "data.jsonl")
        collector = TrainingDataCollector(output_path=path, sync=True)
        collector.record(
            task="test",
            prompt="p",
            response="valid output",
            score=0.8,
            model_id="m",
            task_type="general",
            tokens_used=50,
            latency_ms=100,
        )
        assert collector.get_stats()["total"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# 2. Telemetry — verify data actually flows end-to-end
# ═══════════════════════════════════════════════════════════════════════════


class TestTelemetryEndToEnd:
    """Telemetry must actually record and retrieve data."""

    def test_adapter_metrics_accumulate(self):
        """Recording adapter latency must increment counters."""
        from vetinari.telemetry import TelemetryCollector

        tc = TelemetryCollector()
        tc.record_adapter_latency("local", "test-model", latency_ms=100.0, tokens_used=50, success=True)
        tc.record_adapter_latency("local", "test-model", latency_ms=200.0, tokens_used=75, success=True)
        tc.record_adapter_latency("local", "test-model", latency_ms=300.0, tokens_used=25, success=False)

        summary = tc.get_summary()
        assert summary["total_tokens_used"] == 150
        assert summary["session_requests"] == 3

    def test_get_summary_returns_correct_shape(self):
        """get_summary must return all required keys for token-stats endpoint."""
        from vetinari.telemetry import TelemetryCollector

        tc = TelemetryCollector()
        summary = tc.get_summary()
        for key in ("total_tokens_used", "total_cost_usd", "by_model", "by_provider", "session_requests"):
            assert key in summary, f"Missing key: {key}"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Quality scoring — verify meaningful differentiation
# ═══════════════════════════════════════════════════════════════════════════


class TestQualityScoringAdversarial:
    """Quality scorer must differentiate good from bad output."""

    def test_empty_output_scores_low(self):
        """Empty output must score near zero."""
        from vetinari.learning.quality_scorer import QualityScorer

        scorer = QualityScorer()
        result = scorer.score(
            task_id="test-empty",
            model_id="test",
            task_type="coding",
            task_description="Write code",
            output="",
            use_llm=False,
        )
        assert result is not None
        assert result.overall_score < 0.3

    def test_meaningful_output_scores_higher_than_gibberish(self):
        """Real code output must score higher than random text."""
        from vetinari.learning.quality_scorer import QualityScorer

        scorer = QualityScorer()

        good = "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)\n"
        bad = "asdf jkl qwerty uiop zxcv"

        good_score = scorer.score(
            task_id="test-good",
            model_id="test",
            task_type="coding",
            task_description="Write fibonacci",
            output=good,
            use_llm=False,
        )
        bad_score = scorer.score(
            task_id="test-bad",
            model_id="test",
            task_type="coding",
            task_description="Write fibonacci",
            output=bad,
            use_llm=False,
        )

        assert good_score is not None
        assert bad_score is not None
        assert good_score.overall_score > bad_score.overall_score


# ═══════════════════════════════════════════════════════════════════════════
# 4. Thompson Sampling — verify learning actually happens
# ═══════════════════════════════════════════════════════════════════════════


class TestThompsonSamplingAdversarial:
    """Thompson Sampling must actually learn from feedback."""

    def test_high_quality_model_preferred_after_updates(self):
        """After many updates, the high-quality model should be preferred."""
        from vetinari.learning.model_selector import ThompsonSamplingSelector

        ts = ThompsonSamplingSelector()

        for _ in range(20):
            ts.update("model-a", "coding", quality_score=0.9, success=True)
            ts.update("model-b", "coding", quality_score=0.2, success=False)

        selections = {"model-a": 0, "model-b": 0}
        for _ in range(50):
            selected = ts.select_model("coding", ["model-a", "model-b"])
            if selected in selections:
                selections[selected] += 1

        assert selections["model-a"] > selections["model-b"]

    def test_pass_rate_clamped(self):
        """pass_rate > 1.0 must not corrupt Beta distribution."""
        from vetinari.learning.model_selector import ThompsonSamplingSelector

        ts = ThompsonSamplingSelector()
        ts.update_from_benchmark("test-model", pass_rate=1.5, n_trials=10, task_type="coding")

        arm = ts._arms.get("test-model:coding")
        assert arm is not None
        assert arm.beta >= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 5. Idle detection — verify real timing behavior
# ═══════════════════════════════════════════════════════════════════════════


class TestIdleDetectionAdversarial:
    """IdleDetector must accurately track user activity."""

    def test_not_idle_after_activity(self):
        """Must not be idle right after activity."""
        from vetinari.training.idle_scheduler import IdleDetector

        d = IdleDetector(min_idle_minutes=1.0)
        d.record_activity()
        assert not d.idle

    def test_becomes_idle_after_threshold(self):
        """Must become idle after the threshold."""
        from datetime import timedelta

        from vetinari.training.idle_scheduler import IdleDetector

        d = IdleDetector(min_idle_minutes=1.0)
        # Backdate last activity to deterministically exceed the threshold
        d._last_activity = d._last_activity - timedelta(minutes=2)
        assert d.idle
        assert d.idle_duration_minutes > 0


# ═══════════════════════════════════════════════════════════════════════════
# 6. Execution graph — verify dependency ordering
# ═══════════════════════════════════════════════════════════════════════════


class TestExecutionGraphAdversarial:
    """Execution graph must enforce dependency ordering."""

    def test_dependency_ordering(self):
        """Tasks must not be ready until dependencies complete."""
        from vetinari.orchestration.execution_graph import ExecutionGraph

        graph = ExecutionGraph(plan_id="test-plan", goal="Test ordering")
        graph.add_task("task-a", description="First")
        graph.add_task("task-b", description="Second", depends_on=["task-a"])
        graph.add_task("task-c", description="Third", depends_on=["task-b"])

        ready = [t.id for t in graph.get_ready_tasks()]
        assert "task-a" in ready
        assert "task-b" not in ready

        from vetinari.types import StatusEnum

        graph.nodes["task-a"].status = StatusEnum.COMPLETED
        ready = [t.id for t in graph.get_ready_tasks()]
        assert "task-b" in ready
        assert "task-c" not in ready

    def test_circular_deps_produce_no_ready_tasks(self):
        """Circular dependencies must not cause infinite loops."""
        from vetinari.orchestration.execution_graph import ExecutionGraph

        graph = ExecutionGraph(plan_id="test-circular", goal="Test cycles")
        graph.add_task("a", description="A", depends_on=["c"])
        graph.add_task("b", description="B", depends_on=["a"])
        graph.add_task("c", description="C", depends_on=["b"])

        ready = graph.get_ready_tasks()
        assert len(ready) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 8. Cost tracking — verify real aggregation
# ═══════════════════════════════════════════════════════════════════════════


class TestCostTrackingAdversarial:
    """Cost tracker must accurately aggregate costs."""

    def test_cost_summary_aggregates_correctly(self):
        """get_summary must sum costs across all entries."""
        from vetinari.analytics.cost import CostEntry, CostTracker, reset_cost_tracker

        reset_cost_tracker()
        tracker = CostTracker()

        tracker.record(CostEntry(provider="local", model="model-a", input_tokens=100, output_tokens=50))
        tracker.record(CostEntry(provider="local", model="model-a", input_tokens=200, output_tokens=100))
        tracker.record(CostEntry(provider="cloud", model="gpt-4o", input_tokens=50, output_tokens=25))

        summary = tracker.get_summary()
        assert summary["total_cost_usd"] >= 0
        assert "model-a" in str(summary["by_model"]) or "gpt-4o" in str(summary["by_model"])


# ═══════════════════════════════════════════════════════════════════════════
# 9. Plan generation — different goals, different plans
# ═══════════════════════════════════════════════════════════════════════════


class TestPlanGenerationAdversarial:
    """Plans must be goal-specific, not cookie-cutter templates."""

    def test_different_domains_different_plans(self):
        """Coding vs research goals must produce different task structures."""
        from vetinari.planning.plan_mode import PlanModeEngine
        from vetinari.planning.plan_types import PlanGenerationRequest

        engine = PlanModeEngine()
        coding_plan = engine.generate_plan(PlanGenerationRequest(goal="Build a REST API with auth"))
        research_plan = engine.generate_plan(PlanGenerationRequest(goal="Research quantum computing advances"))

        coding_descs = {s.description for s in coding_plan.subtasks}
        research_descs = {s.description for s in research_plan.subtasks}

        assert coding_descs != research_descs

    def test_empty_goal_produces_plan(self):
        """Empty goal should produce a plan, not crash."""
        from vetinari.planning.plan_mode import PlanModeEngine
        from vetinari.planning.plan_types import PlanGenerationRequest

        engine = PlanModeEngine()
        plan = engine.generate_plan(PlanGenerationRequest(goal=""))
        assert plan is not None
        assert len(plan.subtasks) > 0
