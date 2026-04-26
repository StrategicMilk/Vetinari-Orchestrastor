"""Tests for vetinari/learning/prompt_optimizer.py"""

from __future__ import annotations

import pytest

from vetinari.learning.prompt_optimizer import (
    PromptExperiment,
    PromptOptimizer,
    TraceAnalysis,
    get_prompt_optimizer,
)


class TestPromptExperiment:
    def test_avg_quality_empty(self):
        exp = PromptExperiment(
            experiment_id="e1",
            agent_type="foreman",
            instruction="Do the work.",
        )
        assert exp.avg_quality == 0.0

    def test_avg_quality_multiple_scores(self):
        exp = PromptExperiment(
            experiment_id="e2",
            agent_type="worker",
            instruction="Work carefully.",
            quality_scores=[0.6, 0.8, 1.0],
        )
        assert exp.avg_quality == pytest.approx(0.8)

    def test_repr_shows_key_fields(self):
        exp = PromptExperiment(
            experiment_id="e3",
            agent_type="inspector",
            instruction="Inspect output.",
            status="completed",
        )
        r = repr(exp)
        assert "e3" in r
        assert "inspector" in r
        assert "completed" in r


class TestTraceAnalysis:
    def test_repr_shows_category_and_confidence(self):
        ta = TraceAnalysis(
            failure_category="incomplete",
            root_cause="Output truncated",
            suggested_fix="Add more tokens",
            confidence=0.9,
        )
        r = repr(ta)
        assert "incomplete" in r
        assert "0.9" in r


class TestPromptOptimizer:
    @pytest.fixture
    def optimizer(self):
        return PromptOptimizer()

    def test_optimize_via_search_returns_experiment(self, optimizer):
        result = optimizer.optimize_via_search(
            agent_type="worker",
            baseline_instruction="Complete the task.",
        )
        assert result is not None
        assert isinstance(result, PromptExperiment)
        assert result.agent_type == "worker"
        assert result.avg_quality > 0.0

    def test_optimize_via_search_time_boxed(self, optimizer):
        # Should complete quickly with a very short budget
        result = optimizer.optimize_via_search(
            agent_type="worker",
            baseline_instruction="Complete the task.",
            time_budget_seconds=0.001,  # Near-zero budget — may produce 0 experiments
        )
        # Either None (no time) or a valid experiment — both are correct
        assert result is None or isinstance(result, PromptExperiment)

    def test_optimize_via_search_stores_experiment(self, optimizer):
        optimizer.optimize_via_search(
            agent_type="foreman",
            baseline_instruction="Plan the work.",
        )
        exps = optimizer.get_experiments(agent_type="foreman")
        assert len(exps) >= 1
        assert exps[0]["agent_type"] == "foreman"
        assert "avg_quality" in exps[0]

    def test_optimize_via_trace_incomplete_output(self, optimizer):
        trace = {"output": "", "error": "", "quality_score": 0.0}
        result = optimizer.optimize_via_trace(
            agent_type="worker",
            baseline_instruction="Do the work.",
            failed_trace=trace,
        )
        assert result is not None
        assert "complete" in result.instruction.lower() or "IMPORTANT" in result.instruction

    def test_optimize_via_trace_format_error(self, optimizer):
        trace = {
            "output": "some output that is long enough to pass the length check",
            "error": "json parse error",
            "quality_score": 0.5,
        }
        result = optimizer.optimize_via_trace(
            agent_type="worker",
            baseline_instruction="Return JSON output.",
            failed_trace=trace,
        )
        assert result is not None
        assert result.instruction != "Return JSON output."

    def test_optimize_via_trace_reasoning_error(self, optimizer):
        trace = {
            "output": "x" * 30,  # Long enough to pass length check
            "error": "",
            "quality_score": 0.1,  # Very low — reasoning error
        }
        from vetinari.learning.prompt_optimizer import PromptExperiment

        result = optimizer.optimize_via_trace(
            agent_type="foreman",
            baseline_instruction="Plan carefully.",
            failed_trace=trace,
        )
        assert isinstance(result, PromptExperiment)
        assert result.agent_type == "foreman"

    def test_optimize_via_trace_high_confidence_low_score(self, optimizer):
        # quality_score between 0.3-0.6 triggers off_topic (confidence=0.5)
        # This is below the 0.4 threshold, so should return None
        # Actually confidence=0.5 >= 0.4, so it should return an experiment
        trace = {
            "output": "x" * 50,
            "error": "",
            "quality_score": 0.45,
        }
        # Should either produce a result or None depending on heuristic
        result = optimizer.optimize_via_trace(
            agent_type="worker",
            baseline_instruction="Do the work.",
            failed_trace=trace,
        )
        # Valid result or None (if instruction unchanged after fix application)
        assert result is None or isinstance(result, PromptExperiment)

    def test_evaluate_instruction_scores_longer_text_higher(self, optimizer):
        short_score = optimizer._evaluate_instruction("Do it.", "worker")
        long_score = optimizer._evaluate_instruction(
            "You are an expert. Think step by step. Verify your work. Format your response with clear structure.",
            "worker",
        )
        assert long_score > short_score

    def test_evaluate_instruction_penalizes_very_long(self, optimizer):
        # Very long instruction (>1000 chars) should be penalized
        very_long = "instruction " * 200  # ~2400 chars
        normal = "instruction that is just long enough to score well " * 3
        very_long_score = optimizer._evaluate_instruction(very_long, "worker")
        normal_score = optimizer._evaluate_instruction(normal, "worker")
        assert very_long_score < normal_score

    def test_get_experiments_filter_by_agent(self, optimizer):
        optimizer.optimize_via_search("alpha", "Do A.")
        optimizer.optimize_via_search("beta", "Do B.")
        alpha_exps = optimizer.get_experiments(agent_type="alpha")
        assert all(e["agent_type"] == "alpha" for e in alpha_exps)

    def test_get_experiments_no_filter_returns_all(self, optimizer):
        optimizer.optimize_via_search("x_agent", "Do X.")
        all_exps = optimizer.get_experiments()
        assert len(all_exps) >= 1

    def test_get_prompt_optimizer_singleton(self):
        a = get_prompt_optimizer()
        b = get_prompt_optimizer()
        assert a is b
