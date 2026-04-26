"""Tests for guardrails fail-closed behaviour (SESSION-28).

Covers three changes landed in this session:

1. ``check_output`` with ``RailContext.CODE_EXECUTION`` now runs secrets
   scanning (via ``get_secret_scanner``) instead of blanket-passing.

2. ``get_nemo_provider()`` returning ``None`` while ``is_nemo_init_failed()``
   is ``True`` causes both ``check_input`` and ``check_output`` to surface a
   degraded-safety violation rather than silently skipping the NeMo tier.

3. ``AgentMonitor.register_agent`` rejects duplicate registrations so that
   a mid-run re-registration cannot silently reset the step counter and hide
   runaway execution.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.safety.agent_monitor import AgentMonitor, reset_agent_monitor
from vetinari.safety.guardrails import GuardrailResult, GuardrailsManager, RailContext, reset_guardrails

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_guardrails_singleton() -> None:
    """Destroy the GuardrailsManager singleton before and after every test.

    The singleton caches state; resetting it ensures each test starts clean
    and avoids cross-test contamination.
    """
    reset_guardrails()
    yield
    reset_guardrails()


@pytest.fixture(autouse=True)
def _reset_agent_monitor_singleton() -> None:
    """Destroy the AgentMonitor singleton before and after every test."""
    reset_agent_monitor()
    yield
    reset_agent_monitor()


@pytest.fixture
def guardrails() -> GuardrailsManager:
    """Return a fresh GuardrailsManager instance."""
    return GuardrailsManager()


@pytest.fixture
def monitor() -> AgentMonitor:
    """Return a fresh AgentMonitor instance."""
    return AgentMonitor()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A realistic-looking OpenAI API key that the SecretScanner will detect.
_FAKE_OPENAI_KEY = "sk-FakeKeyForTestingPurposesOnlyABCDEFGH"

# Clean code output that contains no credential-like patterns.
_CLEAN_CODE_OUTPUT = "exit_code = 0\nstdout = 'Hello, world!'\nstderr = ''"


# ---------------------------------------------------------------------------
# 1. CODE_EXECUTION output: secrets scanning
# ---------------------------------------------------------------------------


class TestCodeExecutionSecretsScanning:
    """check_output with CODE_EXECUTION context must scan for secrets."""

    def test_secret_in_code_output_is_redacted(self, guardrails: GuardrailsManager) -> None:
        """Secrets detected in CODE_EXECUTION output are redacted and a violation is raised.

        The result must be allowed=True (content is returned, not blocked) but
        the content must differ from the original (secret replaced) and the
        violations list must be non-empty to surface the detection.
        """
        text_with_secret = f"OPENAI_API_KEY={_FAKE_OPENAI_KEY}\nresult = call_api()"

        # The real SecretScanner will detect the sk- key pattern.
        result: GuardrailResult = guardrails.check_output(text_with_secret, context=RailContext.CODE_EXECUTION)

        # Content is allowed through (not blocked) but sanitised.
        assert result.allowed is True, "CODE_EXECUTION secrets: result must be allowed=True (redacted, not blocked)"
        assert result.content != text_with_secret, "CODE_EXECUTION secrets: content must be redacted"
        assert _FAKE_OPENAI_KEY not in result.content, "CODE_EXECUTION secrets: raw key must not appear in output"
        assert len(result.violations) >= 1, "CODE_EXECUTION secrets: at least one violation must be reported"
        assert result.violations[0].rail == "sensitive_data", (
            "CODE_EXECUTION secrets: violation rail must be 'sensitive_data'"
        )
        assert result.violations[0].severity == "high", "CODE_EXECUTION secrets: violation severity must be 'high'"

    def test_clean_code_output_passes_unchanged(self, guardrails: GuardrailsManager) -> None:
        """Clean CODE_EXECUTION output with no secrets passes through unchanged."""
        result: GuardrailResult = guardrails.check_output(_CLEAN_CODE_OUTPUT, context=RailContext.CODE_EXECUTION)

        assert result.allowed is True, "Clean CODE_EXECUTION: result must be allowed=True"
        assert result.content == _CLEAN_CODE_OUTPUT, "Clean CODE_EXECUTION: content must be unchanged"
        assert result.violations == [], "Clean CODE_EXECUTION: violations must be empty"

    def test_code_execution_uses_secret_scanner_not_sensitive_data_patterns(
        self, guardrails: GuardrailsManager
    ) -> None:
        """CODE_EXECUTION must call the secret scanner, not the regex sensitive-data check.

        The regex-based ``_check_sensitive_data`` is used for USER_FACING output.
        CODE_EXECUTION skips it in favour of ``get_secret_scanner().redact()``.
        This test proves the scanner is actually called by verifying its redact()
        return value is used as the result content.
        """
        redacted_sentinel = "OUTPUT_WITH_SENTINEL_REDACTED"

        mock_scanner = MagicMock()
        # redact() returns something different from the input — signals a hit.
        mock_scanner.redact.return_value = redacted_sentinel

        with patch("vetinari.security.get_secret_scanner", return_value=mock_scanner):
            result = guardrails.check_output("some secret here", context=RailContext.CODE_EXECUTION)

        mock_scanner.redact.assert_called_once_with("some secret here")
        assert result.content == redacted_sentinel, "CODE_EXECUTION: must use scanner redact() output as content"
        assert result.allowed is True, "CODE_EXECUTION secrets via mock: must be allowed=True"
        assert len(result.violations) == 1, "CODE_EXECUTION secrets via mock: exactly one violation expected"

    def test_code_execution_clean_scanner_no_violation(self, guardrails: GuardrailsManager) -> None:
        """When the secret scanner finds nothing (redact returns input unchanged), no violation fires."""
        clean_text = "print('hello')"
        mock_scanner = MagicMock()
        mock_scanner.redact.return_value = clean_text  # unchanged — no secrets

        with patch("vetinari.security.get_secret_scanner", return_value=mock_scanner):
            result = guardrails.check_output(clean_text, context=RailContext.CODE_EXECUTION)

        assert result.allowed is True
        assert result.content == clean_text
        assert result.violations == []


# ---------------------------------------------------------------------------
# 2. AgentMonitor re-registration rejection
# ---------------------------------------------------------------------------


class TestAgentMonitorReRegistrationRejection:
    """Duplicate register_agent calls must be ignored to protect the step counter."""

    def test_duplicate_registration_does_not_reset_step_counter(self, monitor: AgentMonitor) -> None:
        """A second register_agent call for the same ID must not reset the step counter.

        If the counter were reset, a re-registration mid-run would hide runaway
        execution by making it appear the agent just started.
        """
        monitor.register_agent("agent-1", timeout_seconds=60, max_steps=20)

        # Record steps so the counter is non-zero.
        monitor.record_step("agent-1")
        monitor.record_step("agent-1")
        monitor.record_step("agent-1")

        # Attempt re-registration — must be silently ignored.
        monitor.register_agent("agent-1", timeout_seconds=30, max_steps=5)

        # Step counter must still reflect the three steps recorded above.
        stats = monitor.get_stats()
        agent_info = next(a for a in stats["agents"] if a["agent_id"] == "agent-1")
        assert agent_info["step_count"] == 3, "Re-registration must not reset step counter — it should still be 3"

    def test_duplicate_registration_does_not_change_max_steps(self, monitor: AgentMonitor) -> None:
        """The original max_steps must survive a duplicate registration attempt."""
        monitor.register_agent("agent-2", timeout_seconds=60, max_steps=20)
        monitor.register_agent("agent-2", timeout_seconds=60, max_steps=5)  # should be ignored

        stats = monitor.get_stats()
        agent_info = next(a for a in stats["agents"] if a["agent_id"] == "agent-2")
        assert agent_info["max_steps"] == 20, "Re-registration must not overwrite original max_steps"

    def test_first_registration_is_accepted(self, monitor: AgentMonitor) -> None:
        """The first register_agent call for a new ID must succeed."""
        monitor.register_agent("agent-new", timeout_seconds=60, max_steps=10)
        assert "agent-new" in monitor.active_agents, "First registration must be accepted"

    def test_second_registration_agent_remains_in_registry(self, monitor: AgentMonitor) -> None:
        """After a duplicate call, the agent remains registered and functional."""
        monitor.register_agent("agent-3", timeout_seconds=60, max_steps=10)
        monitor.register_agent("agent-3", timeout_seconds=60, max_steps=10)  # duplicate

        # Must still be usable — heartbeat and record_step must not raise.
        monitor.heartbeat("agent-3")
        monitor.record_step("agent-3")
        assert "agent-3" in monitor.active_agents

    def test_reset_agent_allows_fresh_step_tracking(self, monitor: AgentMonitor) -> None:
        """reset_agent() is the correct way to intentionally reset the step counter.

        This tests the documented escape hatch so callers know how to achieve
        an intentional reset without hitting the duplicate-registration guard.
        """
        monitor.register_agent("agent-4", timeout_seconds=60, max_steps=20)
        monitor.record_step("agent-4")
        monitor.record_step("agent-4")

        monitor.reset_agent("agent-4")

        stats = monitor.get_stats()
        agent_info = next(a for a in stats["agents"] if a["agent_id"] == "agent-4")
        assert agent_info["step_count"] == 0, "reset_agent() must bring step_count back to zero"


# ---------------------------------------------------------------------------
# 3. NeMo init-failure degraded-safety violation
# ---------------------------------------------------------------------------


class TestNemoInitFailureDegradedSafety:
    """When NeMo is installed but failed to init, the check fails closed."""

    # Both check_input and check_output share the same NeMo-tier logic, so we
    # parametrize over both paths to prove they both surface the violation.
    @pytest.mark.parametrize("check_method", ["check_input", "check_output"])
    def test_degraded_safety_violation_when_nemo_init_failed(
        self, guardrails: GuardrailsManager, check_method: str
    ) -> None:
        """When NeMo init has failed, each check must block with a fail-closed violation."""
        safe_text = "What is the weather today?"

        with (
            patch(
                "vetinari.safety.nemo_provider.get_nemo_provider",
                return_value=None,
            ),
            patch(
                "vetinari.safety.nemo_provider.is_nemo_init_failed",
                return_value=True,
            ),
        ):
            fn = getattr(guardrails, check_method)
            result: GuardrailResult = fn(safe_text)

        assert result.allowed is False, f"{check_method}: degraded NeMo must fail closed"
        nemo_violations = [v for v in result.violations if v.rail == "nemo_colang"]
        assert len(nemo_violations) == 1, (
            f"{check_method}: exactly one nemo_colang violation must be present when init failed"
        )
        assert nemo_violations[0].severity == "high", (
            f"{check_method}: fail-closed violation must have severity 'high'"
        )
        assert "fail-closed" in nemo_violations[0].description.lower(), (
            f"{check_method}: violation description must mention 'fail-closed'"
        )

    def test_no_degraded_violation_when_nemo_not_installed(self, guardrails: GuardrailsManager) -> None:
        """When NeMo is simply not installed, no violation is added.

        The ImportError path silently skips the tier — that is expected and correct
        because there is nothing broken, the dependency just is not present.
        """
        safe_text = "Tell me a joke."

        with (
            patch(
                "vetinari.safety.nemo_provider.get_nemo_provider",
                return_value=None,
            ),
            patch(
                "vetinari.safety.nemo_provider.is_nemo_init_failed",
                return_value=False,  # not installed — no init failure
            ),
        ):
            result: GuardrailResult = guardrails.check_input(safe_text)

        nemo_violations = [v for v in result.violations if v.rail == "nemo_colang"]
        assert nemo_violations == [], "No violation when NeMo is simply not installed"
        assert result.allowed is True

    def test_no_degraded_violation_when_nemo_works_correctly(self, guardrails: GuardrailsManager) -> None:
        """When a working NeMo provider is present, no degraded-safety violation fires."""
        safe_text = "What is 2 + 2?"

        mock_nemo = MagicMock()
        mock_nemo.check_input.return_value = GuardrailResult(allowed=True, content=safe_text)

        with (
            patch(
                "vetinari.safety.nemo_provider.get_nemo_provider",
                return_value=mock_nemo,
            ),
            patch(
                "vetinari.safety.nemo_provider.is_nemo_init_failed",
                return_value=False,
            ),
        ):
            result: GuardrailResult = guardrails.check_input(safe_text)

        nemo_violations = [v for v in result.violations if v.rail == "nemo_colang"]
        assert nemo_violations == [], "No degraded-safety violation when NeMo is healthy"
        assert result.allowed is True

    def test_degraded_safety_violation_on_check_output(self, guardrails: GuardrailsManager) -> None:
        """check_output also blocks with the fail-closed NeMo violation."""
        safe_output = "The result is 42."

        with (
            patch(
                "vetinari.safety.nemo_provider.get_nemo_provider",
                return_value=None,
            ),
            patch(
                "vetinari.safety.nemo_provider.is_nemo_init_failed",
                return_value=True,
            ),
        ):
            result: GuardrailResult = guardrails.check_output(safe_output)

        nemo_violations = [v for v in result.violations if v.rail == "nemo_colang"]
        assert len(nemo_violations) == 1, "check_output: one nemo_colang violation when init failed"
        assert result.allowed is False
