"""Tests for vetinari.observability — GenAI tracing, step evaluation, tracing spans."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from vetinari.observability.otel_genai import (
    GenAITracer,
    SpanContext,
    get_genai_tracer,
    reset_genai_tracer,
    ATTR_AGENT_NAME,
    ATTR_OPERATION,
    ATTR_REQUEST_MODEL,
    ATTR_SPAN_STATUS,
    ATTR_TOOL_NAME,
)
from vetinari.observability.step_evaluator import (
    StepEvaluator,
    PlanQualityMetric,
    PlanAdherenceMetric,
    StepScore,
    EvaluationReport,
    get_step_evaluator,
    reset_step_evaluator,
)
from vetinari.observability.ci_evaluator import (
    CIEvaluator,
    EvalCase,
    CaseResult,
    CIReport,
    get_ci_evaluator,
    reset_ci_evaluator,
)
from vetinari.observability.tracing import (
    start_span,
    pipeline_span,
    stage_span,
    agent_span,
    llm_span,
    NoOpSpan,
    is_otel_available,
)


# ── GenAI Tracer ────────────────────────────────────────────────────────────


class TestGenAITracer:
    """Tests for the GenAITracer class."""

    def setup_method(self) -> None:
        self.tracer = GenAITracer()

    def test_start_and_end_span(self) -> None:
        span = self.tracer.start_agent_span("builder", "chat", model="qwen-32b")
        assert span.is_active
        assert span.agent_name == "builder"
        assert span.operation == "chat"
        assert span.attributes[ATTR_REQUEST_MODEL] == "qwen-32b"

        self.tracer.end_agent_span(span, status="ok", tokens_used=100)
        assert not span.is_active
        assert span.attributes[ATTR_SPAN_STATUS] == "ok"

    def test_record_tool_call(self) -> None:
        span = self.tracer.start_agent_span("researcher", "chat")
        self.tracer.record_tool_call(span, "code_search", '{"q": "foo"}', '"bar"')
        assert len(span.events) == 1
        assert span.events[0]["attributes"][ATTR_TOOL_NAME] == "code_search"
        self.tracer.end_agent_span(span)

    def test_duration_ms(self) -> None:
        span = self.tracer.start_agent_span("planner", "chat")
        assert span.duration_ms >= 0
        self.tracer.end_agent_span(span)
        assert span.duration_ms >= 0

    def test_double_end_warns(self) -> None:
        span = self.tracer.start_agent_span("oracle", "chat")
        self.tracer.end_agent_span(span, status="ok")
        # Second end should be a no-op (logged as warning)
        self.tracer.end_agent_span(span, status="error")
        assert span.attributes[ATTR_SPAN_STATUS] == "ok"  # unchanged

    def test_stats(self) -> None:
        s1 = self.tracer.start_agent_span("a", "chat")
        s2 = self.tracer.start_agent_span("b", "chat")
        self.tracer.end_agent_span(s1, tokens_used=50)

        stats = self.tracer.get_stats()
        assert stats["total_spans"] == 1  # only completed
        assert stats["active_spans"] == 1  # s2 still open
        assert stats["total_tokens"] == 50
        self.tracer.end_agent_span(s2)

    def test_export_traces(self) -> None:
        span = self.tracer.start_agent_span("builder", "chat")
        self.tracer.end_agent_span(span, tokens_used=200)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            count = self.tracer.export_traces(path)
            assert count == 1
            with open(path) as fh:
                data = json.load(fh)
            assert data["convention"] == "opentelemetry-genai-semconv"
            assert len(data["spans"]) == 1
            assert data["spans"][0]["agent_name"] == "builder"
        finally:
            os.unlink(path)

    def test_reset(self) -> None:
        self.tracer.start_agent_span("a", "chat")
        self.tracer.reset()
        stats = self.tracer.get_stats()
        assert stats["total_spans"] == 0
        assert stats["active_spans"] == 0

    def test_span_to_dict(self) -> None:
        span = self.tracer.start_agent_span("quality", "embeddings")
        self.tracer.end_agent_span(span)
        d = span.to_dict()
        assert "trace_id" in d
        assert "span_id" in d
        assert d["agent_name"] == "quality"
        assert d["operation"] == "embeddings"
        assert "duration_ms" in d

    def test_singleton_reset(self) -> None:
        reset_genai_tracer()
        t1 = get_genai_tracer()
        t2 = get_genai_tracer()
        assert t1 is t2
        reset_genai_tracer()


# ── Step Evaluator ──────────────────────────────────────────────────────────


class TestPlanQualityMetric:
    """Tests for the PlanQualityMetric class."""

    def setup_method(self) -> None:
        self.metric = PlanQualityMetric()

    def test_good_plan(self) -> None:
        plan = {
            "tasks": [
                {"id": "t1", "depends_on": []},
                {"id": "t2", "depends_on": ["t1"]},
                {"id": "t3", "depends_on": ["t2"]},
            ]
        }
        score = self.metric.evaluate_plan(plan)
        assert score.passed is True
        assert score.score == 4.0
        assert score.details["has_tasks"] is True
        assert score.details["no_cycles"] is True

    def test_empty_plan(self) -> None:
        plan = {"tasks": []}
        score = self.metric.evaluate_plan(plan)
        assert score.details["has_tasks"] is False

    def test_cyclic_plan(self) -> None:
        plan = {
            "tasks": [
                {"id": "a", "depends_on": ["b"]},
                {"id": "b", "depends_on": ["a"]},
            ]
        }
        score = self.metric.evaluate_plan(plan)
        assert score.details["no_cycles"] is False

    def test_plan_without_dependencies(self) -> None:
        plan = {"tasks": [{"id": "t1"}]}
        score = self.metric.evaluate_plan(plan)
        assert score.details["has_dependencies"] is False


class TestPlanAdherenceMetric:
    """Tests for the PlanAdherenceMetric class."""

    def setup_method(self) -> None:
        self.metric = PlanAdherenceMetric()

    def test_perfect_adherence(self) -> None:
        plan = {"tasks": [{"id": "t1"}, {"id": "t2"}, {"id": "t3"}]}
        log = [{"task_id": "t1"}, {"task_id": "t2"}, {"task_id": "t3"}]
        score = self.metric.evaluate_adherence(plan, log)
        assert score.passed is True
        assert score.score == 3.0

    def test_missing_tasks(self) -> None:
        plan = {"tasks": [{"id": "t1"}, {"id": "t2"}, {"id": "t3"}]}
        log = [{"task_id": "t1"}]
        score = self.metric.evaluate_adherence(plan, log)
        assert score.details["skipped_tasks"] == ["t2", "t3"]

    def test_wrong_order(self) -> None:
        plan = {"tasks": [{"id": "t1"}, {"id": "t2"}, {"id": "t3"}]}
        log = [{"task_id": "t3"}, {"task_id": "t1"}, {"task_id": "t2"}]
        score = self.metric.evaluate_adherence(plan, log)
        assert score.details["order_followed"] is False

    def test_empty_plan_and_log(self) -> None:
        plan = {"tasks": []}
        log = []
        score = self.metric.evaluate_adherence(plan, log)
        assert score.passed is True


class TestStepEvaluator:
    """Tests for the StepEvaluator orchestrator."""

    def setup_method(self) -> None:
        self.evaluator = StepEvaluator()

    def test_evaluate_all_good(self) -> None:
        plan = {
            "tasks": [
                {"id": "t1", "depends_on": []},
                {"id": "t2", "depends_on": ["t1"]},
            ]
        }
        log = [{"task_id": "t1"}, {"task_id": "t2"}]
        report = self.evaluator.evaluate_all(plan, log)
        assert report.passed is True
        assert report.overall_score > 0.8
        assert len(report.recommendations) == 0

    def test_evaluate_all_failing(self) -> None:
        plan = {"tasks": [{"id": "t1"}, {"id": "t2"}]}
        log = []  # nothing executed
        report = self.evaluator.evaluate_all(plan, log)
        assert report.passed is False
        assert len(report.recommendations) > 0

    def test_step_score_ratio(self) -> None:
        s = StepScore(metric_name="test", score=3.0, max_score=4.0, details={}, passed=True)
        assert s.ratio == 0.75

    def test_step_score_ratio_zero_max(self) -> None:
        s = StepScore(metric_name="test", score=0.0, max_score=0.0, details={}, passed=True)
        assert s.ratio == 1.0

    def test_singleton_reset(self) -> None:
        reset_step_evaluator()
        e1 = get_step_evaluator()
        e2 = get_step_evaluator()
        assert e1 is e2
        reset_step_evaluator()


# ── Tracing (NoOpSpan) ─────────────────────────────────────────────────────


class TestTracing:
    """Tests for the tracing context managers and NoOpSpan."""

    def test_start_span_no_otel(self) -> None:
        with start_span("test.span", {"key": "val"}) as span:
            assert isinstance(span, NoOpSpan)
            span.set_attribute("extra", 42)
            assert span.attributes["extra"] == 42

    def test_span_end_logs_duration(self) -> None:
        span = NoOpSpan(name="test")
        span.end()
        assert span.end_time is not None

    def test_span_set_status(self) -> None:
        span = NoOpSpan(name="test")
        span.set_status("ERROR", "something broke")
        assert span.status == "ERROR"
        assert span.attributes["status_description"] == "something broke"

    def test_span_add_event(self) -> None:
        span = NoOpSpan(name="test")
        span.add_event("llm.token", {"count": 100})
        assert len(span.events) == 1
        assert span.events[0]["name"] == "llm.token"

    def test_pipeline_span(self) -> None:
        with pipeline_span("test goal", "plan-1") as span:
            assert "goal" in span.attributes

    def test_stage_span(self) -> None:
        with stage_span("research", 1) as span:
            assert "stage.name" in span.attributes

    def test_agent_span(self) -> None:
        with agent_span("BUILDER", "build", "task-1") as span:
            assert span.attributes["agent.type"] == "BUILDER"

    def test_llm_span(self) -> None:
        with llm_span("qwen-32b", "BUILDER", 2048) as span:
            assert span.attributes["llm.model_id"] == "qwen-32b"

    def test_span_exception_propagates(self) -> None:
        with pytest.raises(ValueError, match="boom"):
            with start_span("failing") as span:
                raise ValueError("boom")

    def test_is_otel_available(self) -> None:
        # Should return a bool regardless of whether OTel is installed
        assert isinstance(is_otel_available(), bool)


# ── Module imports ──────────────────────────────────────────────────────────


# ── CI Assessor ─────────────────────────────────────────────────────────────


class _StubAdapter:
    """Stub adapter for CI assessor tests."""

    def __init__(self, response: str = "The answer is Python.") -> None:
        self._response = response

    def complete(self, prompt: str, model_id: str, max_tokens: int = 512) -> str:
        return self._response


class _FailingAdapter:
    """Adapter that always raises."""

    def complete(self, prompt: str, model_id: str, max_tokens: int = 512) -> str:
        raise RuntimeError("adapter exploded")


class TestCIAssessor:
    """Tests for the CIEvaluator class."""

    def setup_method(self) -> None:
        self.ci = CIEvaluator()

    def test_run_all_pass(self) -> None:
        cases = [
            EvalCase(name="c1", prompt="What is Python?", expected_contains=["Python"]),
            EvalCase(name="c2", prompt="Hello", expected_contains=["answer"]),
        ]
        report = self.ci.run_eval(cases, _StubAdapter(), "test-model")
        assert report.total == 2
        assert report.passed == 2
        assert report.pass_rate == 1.0

    def test_run_with_failure(self) -> None:
        cases = [
            EvalCase(name="c1", prompt="test", expected_contains=["nonexistent_xyz"]),
        ]
        report = self.ci.run_eval(cases, _StubAdapter(), "test-model")
        assert report.failed == 1
        assert report.pass_rate == 0.0

    def test_expected_not_contains(self) -> None:
        cases = [
            EvalCase(name="c1", prompt="test", expected_not_contains=["Java"]),
        ]
        report = self.ci.run_eval(cases, _StubAdapter(), "test-model")
        assert report.passed == 1

    def test_adapter_error_skips(self) -> None:
        cases = [EvalCase(name="c1", prompt="test")]
        report = self.ci.run_eval(cases, _FailingAdapter(), "test-model")
        assert report.skipped == 1

    def test_should_halt_deployment(self) -> None:
        report = CIReport(total=10, passed=8, failed=2, skipped=0, pass_rate=0.8, cases=[])
        assert self.ci.should_halt_deployment(report, min_pass_rate=0.95) is True
        assert self.ci.should_halt_deployment(report, min_pass_rate=0.7) is False

    def test_compare_to_baseline(self) -> None:
        baseline = CIReport(
            total=2, passed=2, failed=0, skipped=0, pass_rate=1.0,
            cases=[
                CaseResult(name="c1", passed=True, latency_ms=100, quality_score=1.0, output_snippet=""),
                CaseResult(name="c2", passed=True, latency_ms=200, quality_score=1.0, output_snippet=""),
            ],
        )
        current = CIReport(
            total=2, passed=1, failed=1, skipped=0, pass_rate=0.5,
            cases=[
                CaseResult(name="c1", passed=True, latency_ms=150, quality_score=1.0, output_snippet=""),
                CaseResult(name="c2", passed=False, latency_ms=300, quality_score=0.0, output_snippet="", failure_reason="missing"),
            ],
        )
        comparison = self.ci.compare_to_baseline(current, baseline)
        assert comparison["regression_detected"] is True
        assert "c2" in comparison["regressed_cases"]
        assert comparison["pass_rate_delta"] == -0.5

    def test_defaults(self) -> None:
        case = EvalCase(name="test", prompt="hello")
        assert case.expected_contains == []
        assert case.max_latency_ms == 10_000
        assert case.min_quality_score == 0.5

    def test_singleton_reset(self) -> None:
        reset_ci_evaluator()
        e1 = get_ci_evaluator()
        e2 = get_ci_evaluator()
        assert e1 is e2
        reset_ci_evaluator()


# ── Module imports ──────────────────────────────────────────────────────────


class TestObservabilityImports:
    """Test that the observability package exports are correct."""

    def test_package_imports(self) -> None:
        from vetinari.observability import (
            get_genai_tracer,
            GenAITracer,
            SpanContext,
            get_step_evaluator,
            StepEvaluator,
            start_span,
            NoOpSpan,
            get_ci_evaluator,
            CIEvaluator,
        )
        assert callable(get_genai_tracer)
        assert callable(get_step_evaluator)
        assert callable(start_span)
        assert callable(get_ci_evaluator)
