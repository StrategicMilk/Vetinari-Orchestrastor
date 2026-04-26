"""Tests for vetinari.observability — GenAI tracing, step evaluation, tracing spans."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from vetinari.observability.ci_evaluator import (
    CaseResult,
    CIEvaluator,
    CIReport,
    EvalCase,
    get_ci_evaluator,
    reset_ci_evaluator,
)
from vetinari.observability.otel_genai import (
    ATTR_REQUEST_MODEL,
    ATTR_SPAN_STATUS,
    ATTR_TOOL_NAME,
    GenAITracer,
    get_genai_tracer,
    reset_genai_tracer,
)
from vetinari.observability.step_evaluator import (
    PlanAdherenceMetric,
    PlanQualityMetric,
    StepEvaluator,
    StepScore,
    get_step_evaluator,
    reset_step_evaluator,
)
from vetinari.observability.tracing import (
    NoOpSpan,
    agent_span,
    is_otel_available,
    llm_span,
    pipeline_span,
    stage_span,
    start_span,
)
from vetinari.types import AgentType

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

    def test_concurrent_callers_get_distinct_trace_ids(self) -> None:
        """D1: ContextVar isolation — two threads sharing the singleton must not see each other's trace IDs."""
        import threading

        results: dict[str, str | None] = {}
        barrier = threading.Barrier(2)

        def _run(label: str) -> None:
            tracer = get_genai_tracer()
            span = tracer.start_agent_span("model-a", "infer")
            barrier.wait()  # Both threads have their spans started before we read trace_id
            results[label] = span.trace_id
            tracer.end_agent_span(span)

        t1 = threading.Thread(target=_run, args=("t1",))
        t2 = threading.Thread(target=_run, args=("t2",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["t1"] is not None, "t1 trace_id must be set"
        assert results["t2"] is not None, "t2 trace_id must be set"
        assert results["t1"] != results["t2"], "Trace IDs must be isolated per thread via ContextVar"


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

    @pytest.fixture(autouse=True)
    def _no_otel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Force NoOpSpan path for all tracing tests regardless of OTel install."""
        import vetinari.observability.tracing as _tracing

        monkeypatch.setattr(_tracing, "_OTEL_AVAILABLE", False)
        monkeypatch.setattr(_tracing, "_tracer", None)

    def test_start_span_no_otel(self) -> None:
        with start_span("test.span", {"key": "val"}) as span:
            assert isinstance(span, NoOpSpan)
            span.set_attribute("extra", 42)
            assert span.attributes["extra"] == 42

    def test_span_end_logs_duration(self) -> None:
        span = NoOpSpan(name="test")
        span.end()
        assert isinstance(span.end_time, float)
        assert span.end_time > 0

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
        with agent_span(AgentType.WORKER.value, "build", "task-1") as span:
            assert span.attributes["agent.type"] == AgentType.WORKER.value

    def test_llm_span(self) -> None:
        with llm_span("qwen-32b", AgentType.WORKER.value, 2048) as span:
            assert span.attributes["llm.model_id"] == "qwen-32b"

    def test_span_exception_propagates(self) -> None:
        with pytest.raises(ValueError, match="boom"), start_span("failing"):
            raise ValueError("boom")

    def test_is_otel_available(self) -> None:
        # Should return a bool regardless of whether OTel is installed
        assert isinstance(is_otel_available(), bool)

    def test_child_span_inherits_parent_trace_id(self) -> None:
        """D2: start_span with parent= propagates trace_id and sets parent_id correctly."""
        with start_span("parent.op") as parent:
            # Assign a known trace_id to the parent so we can assert inheritance
            parent.trace_id = parent.span_id
            with start_span("child.op", parent=parent) as child:
                assert child.parent_id == parent.span_id, (
                    "child.parent_id must equal parent.span_id"
                )
                assert child.trace_id == parent.trace_id, (
                    "child must inherit parent's trace_id"
                )

    def test_root_span_has_no_parent_or_trace_id(self) -> None:
        """D2: A root span (no parent) starts with parent_id=None and trace_id=None."""
        with start_span("root.op") as span:
            assert span.parent_id is None, "root span must have no parent_id"
            assert span.trace_id is None, "root span must have no inherited trace_id"


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
            total=2,
            passed=2,
            failed=0,
            skipped=0,
            pass_rate=1.0,
            cases=[
                CaseResult(name="c1", passed=True, latency_ms=100, quality_score=1.0, output_snippet=""),
                CaseResult(name="c2", passed=True, latency_ms=200, quality_score=1.0, output_snippet=""),
            ],
        )
        current = CIReport(
            total=2,
            passed=1,
            failed=1,
            skipped=0,
            pass_rate=0.5,
            cases=[
                CaseResult(name="c1", passed=True, latency_ms=150, quality_score=1.0, output_snippet=""),
                CaseResult(
                    name="c2",
                    passed=False,
                    latency_ms=300,
                    quality_score=0.0,
                    output_snippet="",
                    failure_reason="missing",
                ),
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
            get_ci_evaluator,
            get_genai_tracer,
            get_step_evaluator,
            start_span,
        )

        assert callable(get_genai_tracer)
        assert callable(get_step_evaluator)
        assert callable(start_span)
        assert callable(get_ci_evaluator)


# ── _ObservabilitySpan (base_agent integration hook) ────────────────────────


class _MockTrace:
    """Minimal stand-in for opentelemetry.trace."""

    def __init__(self) -> None:
        self.mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_span.return_value = self.mock_span
        self._mock_tracer = mock_tracer

    def get_tracer(self, name: str):
        return self._mock_tracer


@pytest.fixture
def otel_modules():
    """Return (otel_mod, trace_mod, mock_trace) ready for patch.dict.

    Returns:
        A tuple of (opentelemetry stub module, opentelemetry.trace stub module,
        _MockTrace instance) suitable for use with patch.dict(sys.modules, ...).
    """
    mock_trace = _MockTrace()
    otel_mod = ModuleType("opentelemetry")
    trace_mod = ModuleType("opentelemetry.trace")
    trace_mod.get_tracer = mock_trace.get_tracer
    return otel_mod, trace_mod, mock_trace


class TestBaseAgentObservabilitySpanNoOp:
    """_ObservabilitySpan is a no-op when opentelemetry is absent."""

    def test_enters_without_error_when_opentelemetry_missing(self):
        """No exception when opentelemetry cannot be imported."""
        from vetinari.agents.base_agent import _ObservabilitySpan

        with patch.dict(sys.modules, {"opentelemetry": None, "opentelemetry.trace": None}):
            with _ObservabilitySpan("test.op") as span:
                pass
        assert span._span is None  # span is a no-op when OTel is absent

    def test_span_is_none_when_opentelemetry_missing(self):
        """Internal _span remains None when import fails."""
        from vetinari.agents.base_agent import _ObservabilitySpan

        with patch.dict(sys.modules, {"opentelemetry": None, "opentelemetry.trace": None}):
            with _ObservabilitySpan("test.op") as s:
                assert s._span is None

    def test_set_attribute_is_noop_when_no_span(self):
        """set_attribute does not raise when _span is None."""
        from vetinari.agents.base_agent import _ObservabilitySpan

        with patch.dict(sys.modules, {"opentelemetry": None, "opentelemetry.trace": None}):
            with _ObservabilitySpan("test.op") as s:
                s.set_attribute("key", "value")  # must not raise
                assert s._span is None  # no span was created

    def test_exception_propagates_without_swallowing(self):
        """Exceptions inside the with-block are not swallowed."""
        from vetinari.agents.base_agent import _ObservabilitySpan

        with patch.dict(sys.modules, {"opentelemetry": None, "opentelemetry.trace": None}):
            with pytest.raises(ValueError, match="boom"):
                with _ObservabilitySpan("test.op"):
                    raise ValueError("boom")

    def test_start_time_set_on_entry(self):
        """_start_time is non-zero after __enter__."""
        from vetinari.agents.base_agent import _ObservabilitySpan

        with patch.dict(sys.modules, {"opentelemetry": None, "opentelemetry.trace": None}):
            span = _ObservabilitySpan("test.op")
            assert span._start_time == 0.0
            with span as s:
                assert s._start_time > 0.0


class TestBaseAgentObservabilitySpanWithOTel:
    """_ObservabilitySpan delegates to OTel when the library is present."""

    def test_creates_span_with_correct_operation_name(self, otel_modules):
        """tracer.start_span receives the operation name."""
        from vetinari.agents.base_agent import _ObservabilitySpan

        otel_mod, trace_mod, mock_trace = otel_modules
        with patch.dict(sys.modules, {"opentelemetry": otel_mod, "opentelemetry.trace": trace_mod}):
            with _ObservabilitySpan("agent.MyAgent.infer"):
                pass
            mock_trace._mock_tracer.start_span.assert_called_once_with("agent.MyAgent.infer")

    def test_span_end_called_on_exit(self, otel_modules):
        """span.end() is called when the context manager exits normally."""
        from vetinari.agents.base_agent import _ObservabilitySpan

        otel_mod, trace_mod, mock_trace = otel_modules
        with patch.dict(sys.modules, {"opentelemetry": otel_mod, "opentelemetry.trace": trace_mod}):
            with _ObservabilitySpan("agent.MyAgent.infer"):
                pass
            mock_trace.mock_span.end.assert_called_once()
        attr_keys = [c.args[0] for c in mock_trace.mock_span.set_attribute.call_args_list]
        assert "duration_ms" in attr_keys

    def test_metadata_attributes_forwarded(self, otel_modules):
        """Metadata dict entries are set on the span."""
        from vetinari.agents.base_agent import _ObservabilitySpan

        otel_mod, trace_mod, mock_trace = otel_modules
        metadata = {"agent_type": "planner", "model_name": "claude-3-haiku"}
        with patch.dict(sys.modules, {"opentelemetry": otel_mod, "opentelemetry.trace": trace_mod}):
            with _ObservabilitySpan("agent.MyAgent.infer", metadata=metadata):
                pass
        calls = {c.args[0]: c.args[1] for c in mock_trace.mock_span.set_attribute.call_args_list}
        assert calls.get("agent_type") == "planner"
        assert calls.get("model_name") == "claude-3-haiku"

    def test_duration_ms_recorded_on_exit(self, otel_modules):
        """duration_ms attribute is set at exit."""
        from vetinari.agents.base_agent import _ObservabilitySpan

        otel_mod, trace_mod, mock_trace = otel_modules
        with patch.dict(sys.modules, {"opentelemetry": otel_mod, "opentelemetry.trace": trace_mod}):
            with _ObservabilitySpan("agent.MyAgent.infer"):
                pass
        attr_keys = [c.args[0] for c in mock_trace.mock_span.set_attribute.call_args_list]
        assert "duration_ms" in attr_keys

    def test_duration_ms_is_non_negative(self, otel_modules):
        """duration_ms value is >= 0."""
        from vetinari.agents.base_agent import _ObservabilitySpan

        otel_mod, trace_mod, mock_trace = otel_modules
        with patch.dict(sys.modules, {"opentelemetry": otel_mod, "opentelemetry.trace": trace_mod}):
            with _ObservabilitySpan("agent.MyAgent.infer"):
                pass
        calls_dict = {c.args[0]: c.args[1] for c in mock_trace.mock_span.set_attribute.call_args_list}
        assert float(calls_dict["duration_ms"]) >= 0.0

    def test_error_attributes_set_on_exception(self, otel_modules):
        """error=True and error.type are set when an exception propagates."""
        from vetinari.agents.base_agent import _ObservabilitySpan

        otel_mod, trace_mod, mock_trace = otel_modules
        with patch.dict(sys.modules, {"opentelemetry": otel_mod, "opentelemetry.trace": trace_mod}):
            with pytest.raises(RuntimeError):
                with _ObservabilitySpan("agent.MyAgent.infer"):
                    raise RuntimeError("inference failed")
        calls_dict = {c.args[0]: c.args[1] for c in mock_trace.mock_span.set_attribute.call_args_list}
        assert calls_dict.get("error") is True
        assert calls_dict.get("error.type") == "RuntimeError"

    def test_set_attribute_forwards_to_span(self, otel_modules):
        """set_attribute proxies key/value to the OTel span as strings."""
        from vetinari.agents.base_agent import _ObservabilitySpan

        otel_mod, trace_mod, mock_trace = otel_modules
        with patch.dict(sys.modules, {"opentelemetry": otel_mod, "opentelemetry.trace": trace_mod}):
            with _ObservabilitySpan("agent.MyAgent.infer") as s:
                s.set_attribute("prompt_tokens", 42)
        calls_dict = {c.args[0]: c.args[1] for c in mock_trace.mock_span.set_attribute.call_args_list}
        assert calls_dict.get("prompt_tokens") == "42"


class TestBackendExporterTruth:
    """33C.2 D1: configure_backend('jaeger') must set _backend='noop' when OTel SDK unavailable."""

    def test_jaeger_without_otel_reports_noop(self):
        """When OTel SDK is unavailable, configure_backend('jaeger') must record 'noop'."""
        from unittest.mock import patch

        import vetinari.observability.otel_genai as og

        with patch.object(og, "_OTEL_AVAILABLE", False):
            og.configure_backend("jaeger", endpoint="http://localhost:4317")
            result = og.get_active_backend()
        og.configure_backend("noop")  # restore
        assert result == "noop", f"Expected 'noop' when OTel unavailable, got {result!r}"

    def test_jaeger_with_otel_reports_jaeger(self):
        """When OTel SDK is available, configure_backend('jaeger') must record 'jaeger'."""
        from unittest.mock import patch

        import vetinari.observability.otel_genai as og

        # Patch only _OTEL_AVAILABLE — configure_backend() uses it directly.
        # The function sets _backend = "jaeger" without instantiating exporter classes
        # when the SDK flag is True.
        with patch.object(og, "_OTEL_AVAILABLE", True):
            og.configure_backend("jaeger", endpoint="http://localhost:4317")
            result = og.get_active_backend()
        og.configure_backend("noop")  # restore
        assert result == "jaeger", f"Expected 'jaeger' when OTel available, got {result!r}"

    def test_env_jaeger_without_otel_active_backend_is_noop(self):
        """_init_backend_from_env with BACKEND=jaeger and no OTel must yield noop backend."""
        from unittest.mock import patch

        import vetinari.observability.otel_genai as og

        with (
            patch.object(og, "_OTEL_AVAILABLE", False),
            patch.dict("os.environ", {"VETINARI_OTEL_BACKEND": "jaeger"}, clear=False),
        ):
            og._init_backend_from_env()
            result = og.get_active_backend()
        og.configure_backend("noop")  # restore
        assert result == "noop", f"Env-driven jaeger with no OTel must yield noop, got {result!r}"


# ---------------------------------------------------------------------------
# Defect 4: OTel parent span context propagated (regression)
# ---------------------------------------------------------------------------


class TestOtelParentContextPropagation:
    """start_agent_span must pass context=otel_ctx to tracer.start_span() when a
    parent span exists, so child spans nest under their parent in the OTel trace."""

    def test_start_span_called_with_context_kwarg(self, otel_modules):
        """start_child_span() passes context= to tracer.start_span() when parent has _otel_span.

        This uses start_child_span (not start_agent_span) because that is the
        code path that propagates parent context.  start_agent_span always creates
        a root span with parent_span_id=None; start_child_span explicitly sets
        parent_span_id and calls set_span_in_context to build the OTel context.
        """
        import vetinari.observability.otel_genai as og

        otel_mod, trace_mod, mock_trace = otel_modules
        sentinel_ctx = object()
        trace_mod.set_span_in_context = MagicMock(return_value=sentinel_ctx)

        with patch.dict(sys.modules, {"opentelemetry": otel_mod, "opentelemetry.trace": trace_mod}):
            with (
                patch.object(og, "_OTEL_AVAILABLE", True),
                patch.object(og, "_otel_trace", trace_mod),
                patch.object(og, "_backend", "file"),
            ):
                reset_genai_tracer()
                tracer = get_genai_tracer()

                # Start a parent span and attach a fake _otel_span so the
                # child propagation path has something to call set_span_in_context on.
                parent = tracer.start_agent_span("builder", "chat")
                parent_otel_mock = MagicMock()
                parent._otel_span = parent_otel_mock  # type: ignore[attr-defined]

                # start_child_span sets parent_span_id and calls
                # _otel_trace.set_span_in_context(parent._otel_span) before
                # calling tracer.start_span(context=otel_ctx).
                mock_trace._mock_tracer.start_span.reset_mock()
                tracer.start_child_span(parent, "builder", "verify")

        # After start_child_span, start_span must have been called with a
        # non-None context= argument (the sentinel returned by set_span_in_context).
        start_span_calls = mock_trace._mock_tracer.start_span.call_args_list
        assert any(
            call.kwargs.get("context") is not None for call in start_span_calls
        ), "start_span() was never called with a non-None context= kwarg"

        reset_genai_tracer()

    def test_tracing_start_span_passes_otel_context_to_parent(self):
        """start_span() in tracing.py passes context= to start_as_current_span when a parent span
        with _otel_span is given, so nested spans appear under their parent in the OTel trace tree.

        This is the regression test for Fix 1: the OTel branch of start_span() was previously
        ignoring the parent argument entirely, causing every span to start a new root trace.

        Because opentelemetry is not installed in this environment, the module-level ``trace``
        name is never bound by the conditional import.  We inject it manually into the module's
        namespace under a try/finally so the module is left clean after the test.
        """
        import vetinari.observability.tracing as tracing_mod
        from vetinari.observability.tracing import start_span

        sentinel_ctx = object()
        mock_otel_span = MagicMock()

        # Build a mock tracer whose start_as_current_span acts as a context manager
        # and records the kwargs it was called with.
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        # Build a mock ``trace`` namespace that returns the sentinel from
        # set_span_in_context.  The tracing module accesses this as a bare name
        # (``trace.set_span_in_context``), so we inject it into the module dict.
        mock_trace_ns = MagicMock()
        mock_trace_ns.set_span_in_context.return_value = sentinel_ctx

        # Build a parent span with a real _otel_span attribute so the propagation
        # path fires (start_span checks `hasattr(parent, "_otel_span")`).
        mock_parent = MagicMock()
        mock_parent._otel_span = mock_otel_span

        # ``trace`` is only bound in the module when opentelemetry is installed; we
        # inject it for the duration of this test and restore the original state.
        had_trace = hasattr(tracing_mod, "trace")
        original_trace = getattr(tracing_mod, "trace", None)
        try:
            tracing_mod.trace = mock_trace_ns  # type: ignore[attr-defined]
            with (
                patch.object(tracing_mod, "_OTEL_AVAILABLE", True),
                patch.object(tracing_mod, "_tracer", mock_tracer),
            ):
                with start_span("test.operation", attributes={"key": "value"}, parent=mock_parent):
                    pass
        finally:
            if had_trace:
                tracing_mod.trace = original_trace  # type: ignore[attr-defined]
            else:
                # Remove the injected name so the module is clean for subsequent tests
                tracing_mod.__dict__.pop("trace", None)

        # set_span_in_context must have been called with the parent's _otel_span
        mock_trace_ns.set_span_in_context.assert_called_once_with(mock_otel_span)

        # start_as_current_span must have received context=sentinel_ctx (not None)
        call_kwargs = mock_tracer.start_as_current_span.call_args
        assert call_kwargs is not None, "start_as_current_span was never called"
        actual_ctx = call_kwargs.kwargs.get("context")
        assert actual_ctx is sentinel_ctx, (
            f"Expected context={sentinel_ctx!r} but got context={actual_ctx!r} — "
            "parent OTel context was not propagated to start_as_current_span"
        )


# ---------------------------------------------------------------------------
# Defect 5: configure_backend() resets tracer singleton (regression)
# ---------------------------------------------------------------------------


class TestConfigureBackendResetsTracer:
    """configure_backend() must call reset_genai_tracer() so the next caller
    gets a fresh GenAITracer that observes the updated _backend value."""

    def test_configure_backend_invalidates_singleton(self):
        """Instance before configure_backend differs from instance after."""
        import vetinari.observability.otel_genai as og

        reset_genai_tracer()
        instance_before = get_genai_tracer()
        og.configure_backend("file")
        instance_after = get_genai_tracer()
        og.configure_backend("noop")  # restore
        reset_genai_tracer()
        assert instance_before is not instance_after, (
            "configure_backend() must reset the singleton so a new instance is created"
        )

    def test_configure_backend_noop_also_resets(self):
        """Even configure_backend('noop') resets the singleton."""
        import vetinari.observability.otel_genai as og

        reset_genai_tracer()
        instance_before = get_genai_tracer()
        og.configure_backend("noop")
        instance_after = get_genai_tracer()
        reset_genai_tracer()
        assert instance_before is not instance_after


# ---------------------------------------------------------------------------
# Defect 6: noop backend suppresses OTel spans (regression)
# ---------------------------------------------------------------------------


class TestNoopBackendSuppressesOtelSpans:
    """When _backend == 'noop', start_agent_span must NOT create OTel spans even
    when the OTel SDK is available, so the noop setting actually disables tracing."""

    def test_no_otel_span_created_when_backend_is_noop(self, otel_modules):
        """tracer.start_span() is NOT called when _backend is 'noop'."""
        import vetinari.observability.otel_genai as og

        otel_mod, trace_mod, mock_trace = otel_modules

        with patch.dict(sys.modules, {"opentelemetry": otel_mod, "opentelemetry.trace": trace_mod}):
            with (
                patch.object(og, "_OTEL_AVAILABLE", True),
                patch.object(og, "_otel_trace", trace_mod),
                patch.object(og, "_backend", "noop"),
            ):
                reset_genai_tracer()
                tracer = get_genai_tracer()
                tracer.start_agent_span("builder", "chat")

        # OTel start_span must NOT have been called because backend is noop.
        assert not mock_trace._mock_tracer.start_span.called, (
            "start_span() must not be called when _backend == 'noop'"
        )
        reset_genai_tracer()

    def test_otel_span_created_when_backend_is_file(self, otel_modules):
        """tracer.start_span() IS called when _backend is 'file' (non-noop)."""
        import vetinari.observability.otel_genai as og

        otel_mod, trace_mod, mock_trace = otel_modules

        with patch.dict(sys.modules, {"opentelemetry": otel_mod, "opentelemetry.trace": trace_mod}):
            with (
                patch.object(og, "_OTEL_AVAILABLE", True),
                patch.object(og, "_otel_trace", trace_mod),
                patch.object(og, "_backend", "file"),
            ):
                reset_genai_tracer()
                tracer = get_genai_tracer()
                tracer.start_agent_span("builder", "chat")

        assert mock_trace._mock_tracer.start_span.called, (
            "start_span() must be called when _backend == 'file'"
        )
        reset_genai_tracer()
