"""Tests that verify safety and quality infrastructure fails closed.

Every checker, gate, and permission guard in this suite MUST return a
failure/degraded result when a dependency is unavailable — never a silent pass.
These tests are defect detectors: if the production code ever reverts to
fail-open behaviour they will catch it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.factories import make_agent_task
from vetinari.agents.contracts import AgentResult, AgentTask, VerificationResult
from vetinari.types import AgentType

# ---------------------------------------------------------------------------
# Minimal concrete BaseAgent subclass for prepare_task tests
# ---------------------------------------------------------------------------


class _StubAgent:
    """Minimal BaseAgent subclass that exposes prepare_task without inference."""

    def __init__(self) -> None:
        from vetinari.agents.base_agent import BaseAgent

        class _Impl(BaseAgent):
            def get_system_prompt(self) -> str:
                return "stub"

            def execute(self, task: AgentTask) -> AgentResult:
                return AgentResult(success=True, output="ok")

            def verify(self, output: Any) -> VerificationResult:
                return VerificationResult(passed=True, issues=[], score=1.0)

            def get_capabilities(self) -> list[str]:
                return []

        self._agent = _Impl(AgentType.WORKER)

    def prepare_task(self, task: AgentTask) -> AgentTask:
        """Delegate to underlying agent's prepare_task."""
        return self._agent.prepare_task(task)

    @property
    def degraded_safety(self) -> bool:
        """Return whether the agent flagged degraded safety."""
        return getattr(self._agent, "_degraded_safety", False)


# ---------------------------------------------------------------------------
# Problem 1: sandbox._run_tests / _run_lint — already confirmed fixed
# ---------------------------------------------------------------------------


class TestSandboxFailClosed:
    """Verify sandbox verification fails closed when tooling is missing."""

    def test_run_tests_returns_false_when_pytest_missing(self, tmp_path: Path) -> None:
        """_run_tests must return (False, [message]) when pytest is not installed.

        Simulates pytest being absent by making subprocess.run raise
        FileNotFoundError, which happens when the executable cannot be found.
        """
        from vetinari.project.sandbox import _run_tests

        with patch("vetinari.project.sandbox.subprocess.run", side_effect=FileNotFoundError):
            passed, errors = _run_tests(tmp_path, timeout=10)

        assert passed is False, "Missing pytest must not be treated as a pass"
        assert len(errors) > 0, "_run_tests must report at least one error message"
        assert any("pytest" in e.lower() for e in errors), (
            "Error message must mention pytest so the reader knows what is missing"
        )

    def test_run_lint_returns_false_when_ruff_missing(self, tmp_path: Path) -> None:
        """_run_lint must return (False, [message]) when ruff is not installed.

        Simulates ruff being absent by making subprocess.run raise
        FileNotFoundError.
        """
        from vetinari.project.sandbox import _run_lint

        with patch("vetinari.project.sandbox.subprocess.run", side_effect=FileNotFoundError):
            passed, errors = _run_lint(tmp_path, timeout=10)

        assert passed is False, "Missing ruff must not be treated as a pass"
        assert len(errors) > 0, "_run_lint must report at least one error message"
        assert any("ruff" in e.lower() for e in errors), (
            "Error message must mention ruff so the reader knows what is missing"
        )


# ---------------------------------------------------------------------------
# Problem 2: ForemanAgent.validate_agent_output ImportError — fail closed
# ---------------------------------------------------------------------------


class TestValidateAgentOutputFailClosed:
    """Verify validate_agent_output fails closed when practices module is missing."""

    def test_import_error_returns_false(self) -> None:
        """validate_agent_output must return (False, [msg]) when practices cannot be imported.

        This guards against a fail-open path where a missing verification
        requirements module silently certifies every output as valid.
        """
        from vetinari.agents.planner_agent import ForemanAgent

        agent = ForemanAgent()

        with patch.dict(
            "sys.modules",
            {"vetinari.agents.practices": None},
        ):
            # Force re-evaluation of the import inside the function by patching
            # the import machinery to raise ImportError for this module.
            import builtins

            original_import = builtins.__import__

            def _block_practices(name, *args, **kwargs):
                if name == "vetinari.agents.practices":
                    raise ImportError("simulated missing module")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=_block_practices):
                passed, unmet = agent.validate_agent_output(AgentType.WORKER.value, "build", {"verification": {}})

        assert passed is False, "ImportError on practices must produce a failing result, not a pass"
        assert len(unmet) > 0, "At least one error reason must be reported"
        assert any("unavailable" in u.lower() or "practices" in u.lower() for u in unmet), (
            "Error message must identify the missing dependency"
        )


# ---------------------------------------------------------------------------
# Problem 3: BaseAgent.prepare_task — permission system unavailable
# ---------------------------------------------------------------------------


class TestPrepareTaskPermissionDegradedSafety:
    """Verify prepare_task logs a warning and sets _degraded_safety when the
    execution context module cannot be loaded."""

    def test_logs_warning_when_execution_context_unavailable(self, caplog: pytest.LogCaptureFixture) -> None:
        """prepare_task must log a WARNING (not silently pass) when the
        execution-context import fails.  Silent continuation is fail-open
        — the caller has no way to know the permission check was skipped.
        """
        stub = _StubAgent()
        task = make_agent_task(agent_type=AgentType.WORKER)

        def _raise_import(*_args, **_kwargs):
            raise ImportError("simulated missing execution_context")

        with (
            patch(
                "vetinari.agents.base_agent._get_execution_context_mod",
                side_effect=_raise_import,
            ),
            caplog.at_level(logging.WARNING, logger="vetinari.agents.base_agent"),
        ):
            stub.prepare_task(task)

        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("permission" in m.lower() or "degraded" in m.lower() for m in warning_messages), (
            "A WARNING mentioning the permission system or degraded safety must be emitted"
        )

    def test_sets_degraded_safety_flag_when_execution_context_unavailable(self) -> None:
        """prepare_task must set _degraded_safety=True when the execution-context
        module is unavailable so downstream code can inspect the agent state."""
        stub = _StubAgent()
        task = make_agent_task(agent_type=AgentType.WORKER)

        def _raise_import(*_args, **_kwargs):
            raise ImportError("simulated missing execution_context")

        with patch(
            "vetinari.agents.base_agent._get_execution_context_mod",
            side_effect=_raise_import,
        ):
            stub.prepare_task(task)

        assert stub.degraded_safety is True, "_degraded_safety must be True after a permission-system import failure"


# ---------------------------------------------------------------------------
# Problem 4: BaseAgent.prepare_task — conservative defaults when registry fails
# ---------------------------------------------------------------------------


class TestPrepareTaskConservativeConstraintDefaults:
    """Verify prepare_task applies conservative token/retry/timeout caps when
    the constraint registry is unavailable instead of leaving the task uncapped."""

    def test_conservative_defaults_applied_when_registry_unavailable(self) -> None:
        """When _get_agent_constraints returns None, prepare_task must set
        _constraint_max_tokens, _constraint_max_retries, and _constraint_timeout
        on the task to prevent runaway execution."""
        stub = _StubAgent()
        task = make_agent_task(agent_type=AgentType.WORKER)

        # _get_agent_constraints is the private helper — patch it to return None
        # to simulate a registry failure (the function itself already logs and
        # returns None on Exception).
        with patch(
            "vetinari.agents.base_agent._get_agent_constraints",
            return_value=None,
        ):
            prepared = stub.prepare_task(task)

        assert hasattr(prepared, "_constraint_max_tokens"), (
            "_constraint_max_tokens must be set even when the registry is unavailable"
        )
        assert hasattr(prepared, "_constraint_max_retries"), (
            "_constraint_max_retries must be set even when the registry is unavailable"
        )
        assert hasattr(prepared, "_constraint_timeout"), (
            "_constraint_timeout must be set even when the registry is unavailable"
        )
        # The conservative caps must be finite and reasonable
        assert prepared._constraint_max_tokens > 0
        assert prepared._constraint_max_retries > 0
        assert prepared._constraint_timeout > 0

    def test_conservative_defaults_are_restrictive(self) -> None:
        """The conservative fallback caps must be bounded — they exist to prevent
        runaway execution, not to grant unlimited resources."""
        stub = _StubAgent()
        task = make_agent_task(agent_type=AgentType.WORKER)

        with patch(
            "vetinari.agents.base_agent._get_agent_constraints",
            return_value=None,
        ):
            prepared = stub.prepare_task(task)

        # Conservative defaults: max 4096 tokens, 2 retries, 120s timeout
        assert prepared._constraint_max_tokens <= 8192, "Conservative token cap should be modest, not unlimited"
        assert prepared._constraint_max_retries <= 3, "Conservative retry cap should be small"
        assert prepared._constraint_timeout <= 300, "Conservative timeout should be a few minutes at most"

    def test_conservative_defaults_log_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A WARNING must be emitted when conservative defaults are applied
        so operators know the constraint registry was unavailable."""
        stub = _StubAgent()
        task = make_agent_task(agent_type=AgentType.WORKER)

        with (
            patch(
                "vetinari.agents.base_agent._get_agent_constraints",
                return_value=None,
            ),
            caplog.at_level(logging.WARNING, logger="vetinari.agents.base_agent"),
        ):
            stub.prepare_task(task)

        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("constraint" in m.lower() or "conservative" in m.lower() for m in warning_messages), (
            "A WARNING about the constraint registry being unavailable must be emitted"
        )
