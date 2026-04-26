"""Tests for vetinari.orchestration.error_escalation.

Covers ErrorClassifier pattern matching, escalation levels, RecoveryMetrics,
the get_error_classifier singleton, and generate_retry_brief (LLM + fallback).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.orchestration.error_escalation import (
    ErrorClassification,
    ErrorClassifier,
    EscalationLevel,
    RecoveryMetrics,
    generate_retry_brief,
    get_error_classifier,
)

# ---------------------------------------------------------------------------
# ErrorClassifier — transient patterns
# ---------------------------------------------------------------------------


class TestErrorClassifierTransient:
    """Classifier correctly identifies transient failures."""

    def test_timeout_is_transient(self) -> None:
        """'timeout' error maps to TRANSIENT level."""
        clf = ErrorClassifier()
        result = clf.classify("Connection timeout after 30s")
        assert result.level == EscalationLevel.TRANSIENT
        assert result.is_retryable is True

    def test_rate_limit_is_transient(self) -> None:
        """'rate limit' error maps to TRANSIENT level."""
        clf = ErrorClassifier()
        result = clf.classify("rate limit exceeded — too many requests")
        assert result.level == EscalationLevel.TRANSIENT
        assert result.is_retryable is True

    def test_oom_is_transient(self) -> None:
        """Out-of-memory maps to TRANSIENT level."""
        clf = ErrorClassifier()
        result = clf.classify("out of memory error during inference")
        assert result.level == EscalationLevel.TRANSIENT

    def test_service_unavailable_is_transient(self) -> None:
        """503 service unavailable maps to TRANSIENT."""
        clf = ErrorClassifier()
        result = clf.classify("503 service unavailable")
        assert result.level == EscalationLevel.TRANSIENT


# ---------------------------------------------------------------------------
# ErrorClassifier — semantic patterns
# ---------------------------------------------------------------------------


class TestErrorClassifierSemantic:
    """Classifier correctly identifies semantic failures."""

    def test_ambiguous_is_semantic(self) -> None:
        """'ambiguous' error maps to SEMANTIC level."""
        clf = ErrorClassifier()
        result = clf.classify("Request is ambiguous — cannot determine intent")
        assert result.level == EscalationLevel.SEMANTIC
        assert result.is_retryable is True

    def test_invalid_format_is_semantic(self) -> None:
        """'invalid JSON' maps to SEMANTIC level."""
        clf = ErrorClassifier()
        result = clf.classify("invalid json in response body")
        assert result.level == EscalationLevel.SEMANTIC

    def test_missing_field_is_semantic(self) -> None:
        """'missing field' maps to SEMANTIC level."""
        clf = ErrorClassifier()
        result = clf.classify("missing field: description in task payload")
        assert result.level == EscalationLevel.SEMANTIC


# ---------------------------------------------------------------------------
# ErrorClassifier — capability patterns
# ---------------------------------------------------------------------------


class TestErrorClassifierCapability:
    """Classifier correctly identifies capability failures."""

    def test_not_capable_is_capability(self) -> None:
        """'not capable' maps to CAPABILITY level."""
        clf = ErrorClassifier()
        result = clf.classify("Agent is not capable of visual reasoning")
        assert result.level == EscalationLevel.CAPABILITY
        assert result.is_retryable is True

    def test_tool_not_available_is_capability(self) -> None:
        """'tool not available' maps to CAPABILITY."""
        clf = ErrorClassifier()
        result = clf.classify("tool not available: code_executor")
        assert result.level == EscalationLevel.CAPABILITY

    def test_permission_denied_is_capability(self) -> None:
        """'permission denied' maps to CAPABILITY."""
        clf = ErrorClassifier()
        result = clf.classify("permission denied: cannot access /etc/passwd")
        assert result.level == EscalationLevel.CAPABILITY


# ---------------------------------------------------------------------------
# ErrorClassifier — fatal patterns
# ---------------------------------------------------------------------------


class TestErrorClassifierFatal:
    """Classifier correctly identifies fatal failures."""

    def test_impossible_is_fatal(self) -> None:
        """'impossible' maps to FATAL level."""
        clf = ErrorClassifier()
        result = clf.classify("This task is impossible to complete")
        assert result.level == EscalationLevel.FATAL
        assert result.is_retryable is False

    def test_data_corruption_is_fatal(self) -> None:
        """'data corruption' maps to FATAL level."""
        clf = ErrorClassifier()
        result = clf.classify("data corruption detected in plan state")
        assert result.level == EscalationLevel.FATAL
        assert result.is_retryable is False

    def test_security_violation_is_fatal(self) -> None:
        """'security violation' maps to FATAL."""
        clf = ErrorClassifier()
        result = clf.classify("security violation: unauthorized action attempted")
        assert result.level == EscalationLevel.FATAL
        assert result.is_retryable is False


# ---------------------------------------------------------------------------
# ErrorClassifier — fallback and context-based
# ---------------------------------------------------------------------------


class TestErrorClassifierFallback:
    """Classifier handles unknown errors and context hints."""

    def test_unknown_error_defaults_to_transient(self) -> None:
        """Unrecognized errors default to TRANSIENT on first occurrence."""
        clf = ErrorClassifier()
        result = clf.classify("some completely novel error XYZ-9999")
        assert result.level == EscalationLevel.TRANSIENT
        assert result.is_retryable is True

    def test_repeated_unknown_error_bumps_to_semantic(self) -> None:
        """After retry_count > 2 the level bumps to SEMANTIC."""
        clf = ErrorClassifier()
        result = clf.classify("some novel error", context={"retry_count": 3})
        assert result.level == EscalationLevel.SEMANTIC

    def test_explicit_escalation_level_overrides_pattern(self) -> None:
        """Explicit escalation_level in context overrides pattern matching."""
        clf = ErrorClassifier()
        result = clf.classify("timeout error", context={"escalation_level": 3})
        assert result.level == EscalationLevel.FATAL

    def test_invalid_explicit_level_falls_through_to_pattern(self) -> None:
        """Invalid escalation_level is ignored; pattern matching proceeds."""
        clf = ErrorClassifier()
        result = clf.classify("timeout error", context={"escalation_level": "bad"})
        assert result.level == EscalationLevel.TRANSIENT

    def test_empty_error_returns_transient(self) -> None:
        """Empty error string defaults to TRANSIENT."""
        clf = ErrorClassifier()
        result = clf.classify("")
        assert result.level == EscalationLevel.TRANSIENT
        assert result.is_retryable is True


# ---------------------------------------------------------------------------
# ErrorClassification
# ---------------------------------------------------------------------------


class TestErrorClassification:
    """ErrorClassification serialization and repr."""

    def test_to_dict_has_required_keys(self) -> None:
        """to_dict returns all expected keys."""
        ec = ErrorClassification(
            level=EscalationLevel.SEMANTIC,
            reason="Test reason",
            suggested_action="Rephrase and retry",
        )
        d = ec.to_dict()
        assert d["level"] == EscalationLevel.SEMANTIC.value
        assert d["level_name"] == "SEMANTIC"
        assert d["reason"] == "Test reason"
        assert d["suggested_action"] == "Rephrase and retry"
        assert d["is_retryable"] is True

    def test_repr_includes_level_and_retryable(self) -> None:
        """__repr__ includes level name and is_retryable."""
        ec = ErrorClassification(level=EscalationLevel.FATAL, reason="irreversible", is_retryable=False)
        r = repr(ec)
        assert "FATAL" in r
        assert "False" in r


# ---------------------------------------------------------------------------
# RecoveryMetrics
# ---------------------------------------------------------------------------


class TestRecoveryMetrics:
    """RecoveryMetrics tracking and resolution rate calculation."""

    def test_initial_resolution_rate_is_zero(self) -> None:
        """Resolution rate is 0.0 with no attempts."""
        m = RecoveryMetrics()
        assert m.resolution_rate() == 0.0

    def test_resolution_rate_reflects_successes(self) -> None:
        """Resolution rate = resolved / attempts."""
        m = RecoveryMetrics()
        m.record(EscalationLevel.TRANSIENT, resolved=True)
        m.record(EscalationLevel.SEMANTIC, resolved=False)
        assert m.resolution_rate() == pytest.approx(0.5)

    def test_by_level_increments_correctly(self) -> None:
        """by_level counter increments per escalation level."""
        m = RecoveryMetrics()
        m.record(EscalationLevel.TRANSIENT, resolved=True)
        m.record(EscalationLevel.TRANSIENT, resolved=False)
        m.record(EscalationLevel.FATAL, resolved=False)
        assert m.by_level[EscalationLevel.TRANSIENT.value] == 2
        assert m.by_level[EscalationLevel.FATAL.value] == 1


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestGetErrorClassifier:
    """get_error_classifier returns a stable singleton."""

    def test_returns_same_instance(self) -> None:
        """Two calls return the same object."""
        a = get_error_classifier()
        b = get_error_classifier()
        assert a is b

    def test_returns_error_classifier_type(self) -> None:
        """Returns an ErrorClassifier instance."""
        assert isinstance(get_error_classifier(), ErrorClassifier)


# ---------------------------------------------------------------------------
# generate_retry_brief
# ---------------------------------------------------------------------------


class TestGenerateRetryBrief:
    """generate_retry_brief returns LLM brief or falls back gracefully."""

    @patch("vetinari.llm_helpers.quick_llm_call")
    def test_returns_llm_brief_when_available(self, mock_quick: MagicMock) -> None:
        """Returns LLM-generated brief when the LLM responds."""
        mock_quick.return_value = (
            "1. Check import paths\n"
            "2. Validate input schema before calling\n"
            "3. Add explicit error handling around LLM call"
        )
        result = generate_retry_brief(
            failure_context="ImportError: cannot import 'Foo' from 'bar'",
            agent_type="worker",
        )
        assert isinstance(result, str)
        assert len(result) > 0
        assert "1." in result or "import" in result.lower()

    @patch("vetinari.llm_helpers.quick_llm_call")
    def test_fallback_when_llm_unavailable(self, mock_quick: MagicMock) -> None:
        """Returns the generic fallback string when LLM returns None."""
        mock_quick.return_value = None
        result = generate_retry_brief(
            failure_context="connection refused",
            agent_type="foreman",
        )
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Review the error" in result or "adjust approach" in result

    @patch("vetinari.llm_helpers.quick_llm_call")
    def test_fallback_when_llm_returns_whitespace(self, mock_quick: MagicMock) -> None:
        """Returns fallback when LLM returns only whitespace."""
        mock_quick.return_value = "   "
        result = generate_retry_brief(
            failure_context="parse error in output",
            agent_type="inspector",
        )
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Review the error" in result or "adjust approach" in result

    @patch("vetinari.llm_helpers.quick_llm_call")
    def test_result_is_never_none(self, mock_quick: MagicMock) -> None:
        """generate_retry_brief always returns a str, never None."""
        mock_quick.return_value = None
        result = generate_retry_brief(failure_context="some failure", agent_type="worker")
        assert result is not None
        assert isinstance(result, str)

    @patch("vetinari.llm_helpers.quick_llm_call")
    def test_agent_type_included_in_context(self, mock_quick: MagicMock) -> None:
        """agent_type is forwarded into the LLM prompt context."""
        mock_quick.return_value = "1. do A\n2. do B\n3. do C"
        generate_retry_brief(
            failure_context="task timed out",
            agent_type="specialist",
        )
        assert mock_quick.called
        # The prompt passed to quick_llm_call must embed the agent_type
        call_kwargs = mock_quick.call_args[1]
        prompt_arg = call_kwargs.get("prompt") or ""
        assert "specialist" in prompt_arg

    @patch("vetinari.llm_helpers.quick_llm_call")
    def test_exception_during_llm_call_returns_fallback(self, mock_quick: MagicMock) -> None:
        """Returns fallback string if the LLM call raises an unexpected exception."""
        import vetinari.orchestration.error_escalation as mod

        mock_quick.side_effect = RuntimeError("unexpected model crash")
        result = generate_retry_brief(failure_context="crash", agent_type="worker")
        assert isinstance(result, str)
        assert result == mod._FALLBACK_RETRY_BRIEF
