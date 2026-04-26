"""Tests for vetinari.analytics.failure_taxonomy — AgentRx 9-category failure taxonomy."""

from __future__ import annotations

import time

import pytest

from tests.factories import make_failure_record as _make_record
from vetinari.analytics.failure_taxonomy import (
    FailureClassifier,
    FailureRecord,
    FailureTracker,
    get_failure_tracker,
    reset_failure_tracker,
)
from vetinari.exceptions import (
    GuardrailError,
    InferenceError,
    ModelUnavailableError,
    SandboxError,
    SecurityError,
    VetinariTimeoutError,
)
from vetinari.types import AgentRxFailureCategory, AgentType

# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestAgentRxFailureCategory:
    def test_all_nine_categories_exist(self) -> None:
        expected = {
            "plan_adherence_failure",
            "hallucination",
            "invalid_tool_invocation",
            "misinterpretation_of_tool_output",
            "intent_plan_misalignment",
            "under_specified_intent",
            "intent_not_supported",
            "guardrails_triggered",
            "system_failure",
        }
        actual = {cat.value for cat in AgentRxFailureCategory}
        assert actual == expected

    def test_enum_count_is_nine(self) -> None:
        assert len(AgentRxFailureCategory) == 9

    def test_enum_is_str_subclass(self) -> None:
        # str subclass allows direct string comparison without .value
        assert AgentRxFailureCategory.HALLUCINATION == "hallucination"

    @pytest.mark.parametrize(
        ("member", "value"),
        [
            (AgentRxFailureCategory.PLAN_ADHERENCE_FAILURE, "plan_adherence_failure"),
            (AgentRxFailureCategory.HALLUCINATION, "hallucination"),
            (AgentRxFailureCategory.INVALID_TOOL_INVOCATION, "invalid_tool_invocation"),
            (AgentRxFailureCategory.MISINTERPRETATION_OF_TOOL_OUTPUT, "misinterpretation_of_tool_output"),
            (AgentRxFailureCategory.INTENT_PLAN_MISALIGNMENT, "intent_plan_misalignment"),
            (AgentRxFailureCategory.UNDER_SPECIFIED_INTENT, "under_specified_intent"),
            (AgentRxFailureCategory.INTENT_NOT_SUPPORTED, "intent_not_supported"),
            (AgentRxFailureCategory.GUARDRAILS_TRIGGERED, "guardrails_triggered"),
            (AgentRxFailureCategory.SYSTEM_FAILURE, "system_failure"),
        ],
    )
    def test_individual_values(self, member: AgentRxFailureCategory, value: str) -> None:
        assert member.value == value


# ---------------------------------------------------------------------------
# FailureRecord tests
# ---------------------------------------------------------------------------


class TestFailureRecord:
    def test_basic_construction(self) -> None:
        record = _make_record()
        assert record.category == AgentRxFailureCategory.PLAN_ADHERENCE_FAILURE
        assert record.task_id == "task-1"
        assert record.agent_type == AgentType.WORKER.value
        assert record.severity == "error"

    def test_frozen(self) -> None:
        record = _make_record()
        with pytest.raises(AttributeError):
            record.task_id = "other"  # type: ignore[misc]

    def test_invalid_severity_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid severity"):
            FailureRecord(
                category=AgentRxFailureCategory.HALLUCINATION,
                task_id="t",
                agent_type=AgentType.WORKER.value,
                description="x",
                timestamp=time.time(),
                severity="info",  # not valid
            )

    def test_valid_severities(self) -> None:
        for sev in ("warning", "error", "critical"):
            r = _make_record(severity=sev)
            assert r.severity == sev

    def test_context_defaults_to_empty_dict(self) -> None:
        record = _make_record()
        assert record.context == {}

    def test_context_stored(self) -> None:
        ctx = {"model_id": "qwen-7b", "tool": "read_file"}
        record = FailureRecord(
            category=AgentRxFailureCategory.INVALID_TOOL_INVOCATION,
            task_id="t2",
            agent_type=AgentType.WORKER.value,
            description="bad tool call",
            timestamp=time.time(),
            context=ctx,
        )
        assert record.context["model_id"] == "qwen-7b"

    def test_repr_contains_key_fields(self) -> None:
        record = _make_record(category=AgentRxFailureCategory.HALLUCINATION, task_id="abc")
        r = repr(record)
        assert "hallucination" in r
        assert "abc" in r


# ---------------------------------------------------------------------------
# FailureClassifier tests
# ---------------------------------------------------------------------------


class TestFailureClassifier:
    def setup_method(self) -> None:
        self.clf = FailureClassifier()

    def test_timeout_error_maps_to_system_failure(self) -> None:
        result = self.clf.classify(TimeoutError("timed out"), {})
        assert result is AgentRxFailureCategory.SYSTEM_FAILURE

    def test_vetinari_timeout_maps_to_system_failure(self) -> None:
        result = self.clf.classify(VetinariTimeoutError("inference timed out"), {})
        assert result is AgentRxFailureCategory.SYSTEM_FAILURE

    def test_model_unavailable_maps_to_system_failure(self) -> None:
        result = self.clf.classify(ModelUnavailableError("no model"), {})
        assert result is AgentRxFailureCategory.SYSTEM_FAILURE

    @pytest.mark.parametrize(
        "msg",
        ["budget exceeded", "token limit reached", "quota exhausted"],
    )
    def test_inference_error_budget_keywords_map_to_system_failure(self, msg: str) -> None:
        result = self.clf.classify(InferenceError(msg), {})
        assert result is AgentRxFailureCategory.SYSTEM_FAILURE

    def test_inference_error_circuit_breaker_maps_to_system_failure(self) -> None:
        result = self.clf.classify(InferenceError("circuit breaker open"), {})
        assert result is AgentRxFailureCategory.SYSTEM_FAILURE

    def test_guardrail_error_maps_to_guardrails_triggered(self) -> None:
        result = self.clf.classify(GuardrailError("blocked by guardrail"), {})
        assert result is AgentRxFailureCategory.GUARDRAILS_TRIGGERED

    def test_sandbox_error_maps_to_guardrails_triggered(self) -> None:
        result = self.clf.classify(SandboxError("sandbox violation"), {})
        assert result is AgentRxFailureCategory.GUARDRAILS_TRIGGERED

    def test_security_error_maps_to_guardrails_triggered(self) -> None:
        result = self.clf.classify(SecurityError("policy violation"), {})
        assert result is AgentRxFailureCategory.GUARDRAILS_TRIGGERED

    @pytest.mark.parametrize(
        "msg",
        ["unknown tool specified", "function not found", "cannot invoke tool"],
    )
    def test_value_error_with_tool_keyword_maps_to_invalid_tool_invocation(self, msg: str) -> None:
        result = self.clf.classify(ValueError(msg), {})
        assert result is AgentRxFailureCategory.INVALID_TOOL_INVOCATION

    def test_value_error_without_tool_keyword_falls_through_to_default(self) -> None:
        result = self.clf.classify(ValueError("unexpected value"), {})
        assert result is AgentRxFailureCategory.PLAN_ADHERENCE_FAILURE

    def test_key_error_with_tool_output_context_maps_to_misinterpretation(self) -> None:
        ctx: dict = {"tool_output": {"key": "val"}}
        result = self.clf.classify(KeyError("missing_field"), ctx)
        assert result is AgentRxFailureCategory.MISINTERPRETATION_OF_TOOL_OUTPUT

    def test_key_error_without_tool_output_context_falls_to_default(self) -> None:
        result = self.clf.classify(KeyError("missing_field"), {})
        assert result is AgentRxFailureCategory.PLAN_ADHERENCE_FAILURE

    def test_unknown_exception_defaults_to_plan_adherence_failure(self) -> None:
        result = self.clf.classify(RuntimeError("something went wrong"), {})
        assert result is AgentRxFailureCategory.PLAN_ADHERENCE_FAILURE

    def test_generic_inference_error_without_keywords_defaults_to_plan_adherence(self) -> None:
        result = self.clf.classify(InferenceError("LLM returned empty response"), {})
        assert result is AgentRxFailureCategory.PLAN_ADHERENCE_FAILURE


# ---------------------------------------------------------------------------
# FailureTracker tests
# ---------------------------------------------------------------------------


class TestFailureTracker:
    def setup_method(self) -> None:
        reset_failure_tracker()
        self.tracker = FailureTracker()

    def test_record_and_get_by_category(self) -> None:
        record = _make_record(category=AgentRxFailureCategory.HALLUCINATION, task_id="t1")
        self.tracker.record(record)
        results = self.tracker.get_by_category(AgentRxFailureCategory.HALLUCINATION)
        assert len(results) == 1
        assert results[0].task_id == "t1"

    def test_get_by_category_excludes_other_categories(self) -> None:
        self.tracker.record(_make_record(category=AgentRxFailureCategory.HALLUCINATION))
        self.tracker.record(_make_record(category=AgentRxFailureCategory.SYSTEM_FAILURE))
        results = self.tracker.get_by_category(AgentRxFailureCategory.HALLUCINATION)
        assert len(results) == 1
        assert results[0].category is AgentRxFailureCategory.HALLUCINATION

    def test_classify_and_record_returns_failure_record(self) -> None:
        err = VetinariTimeoutError("timed out")
        record = self.tracker.classify_and_record(err, "task-99", AgentType.FOREMAN.value)
        assert isinstance(record, FailureRecord)
        assert record.category is AgentRxFailureCategory.SYSTEM_FAILURE
        assert record.task_id == "task-99"

    def test_classify_and_record_persists_record(self) -> None:
        err = GuardrailError("blocked")
        self.tracker.classify_and_record(err, "task-2", AgentType.INSPECTOR.value)
        results = self.tracker.get_by_category(AgentRxFailureCategory.GUARDRAILS_TRIGGERED)
        assert len(results) == 1

    def test_get_breakdown_returns_all_categories(self) -> None:
        breakdown = self.tracker.get_breakdown()
        assert set(breakdown.keys()) == {cat.value for cat in AgentRxFailureCategory}

    def test_get_breakdown_counts_correctly(self) -> None:
        self.tracker.record(_make_record(category=AgentRxFailureCategory.HALLUCINATION))
        self.tracker.record(_make_record(category=AgentRxFailureCategory.HALLUCINATION))
        self.tracker.record(_make_record(category=AgentRxFailureCategory.SYSTEM_FAILURE))
        breakdown = self.tracker.get_breakdown()
        assert breakdown["hallucination"] == 2
        assert breakdown["system_failure"] == 1
        assert breakdown["plan_adherence_failure"] == 0

    def test_get_trends_within_window(self) -> None:
        now = time.time()
        # A record from 1 hour ago (within 24h window)
        self.tracker.record(
            _make_record(
                category=AgentRxFailureCategory.HALLUCINATION,
                timestamp=now - 3600,
            )
        )
        trends = self.tracker.get_trends(window_hours=24)
        assert trends["hallucination"] == 1

    def test_get_trends_excludes_old_records(self) -> None:
        now = time.time()
        # A record from 2 hours ago, but window is 1 hour
        self.tracker.record(
            _make_record(
                category=AgentRxFailureCategory.HALLUCINATION,
                timestamp=now - 7200,
            )
        )
        trends = self.tracker.get_trends(window_hours=1)
        assert trends["hallucination"] == 0

    def test_get_trends_returns_all_categories(self) -> None:
        trends = self.tracker.get_trends()
        assert set(trends.keys()) == {cat.value for cat in AgentRxFailureCategory}

    def test_reset_clears_records(self) -> None:
        self.tracker.record(_make_record())
        self.tracker.reset()
        assert self.tracker.get_by_category(AgentRxFailureCategory.PLAN_ADHERENCE_FAILURE) == []

    def test_get_by_category_returns_empty_list_when_none_recorded(self) -> None:
        results = self.tracker.get_by_category(AgentRxFailureCategory.UNDER_SPECIFIED_INTENT)
        assert results == []


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------


class TestGetFailureTracker:
    def setup_method(self) -> None:
        reset_failure_tracker()

    def teardown_method(self) -> None:
        reset_failure_tracker()

    def test_returns_same_instance(self) -> None:
        a = get_failure_tracker()
        b = get_failure_tracker()
        assert a is b

    def test_reset_creates_new_instance(self) -> None:
        a = get_failure_tracker()
        reset_failure_tracker()
        b = get_failure_tracker()
        assert a is not b

    def test_singleton_stores_records_across_calls(self) -> None:
        tracker = get_failure_tracker()
        tracker.record(_make_record(category=AgentRxFailureCategory.SYSTEM_FAILURE, task_id="singleton-task"))
        same_tracker = get_failure_tracker()
        results = same_tracker.get_by_category(AgentRxFailureCategory.SYSTEM_FAILURE)
        assert any(r.task_id == "singleton-task" for r in results)


# ---------------------------------------------------------------------------
# Analytics package wiring test
# ---------------------------------------------------------------------------


class TestAnalyticsPackageWiring:
    def test_imports_from_analytics_package(self) -> None:
        from vetinari.analytics import (
            FailureClassifier,
            FailureRecord,
            FailureTracker,
            get_failure_tracker,
            reset_failure_tracker,
        )

        assert FailureClassifier is not None
        assert FailureRecord is not None
        assert FailureTracker is not None
        assert callable(get_failure_tracker)
        assert callable(reset_failure_tracker)
