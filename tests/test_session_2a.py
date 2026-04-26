"""Tests for Session 2A modules.

Covers: budget_tracker, execution_feedback, agent_circuit_breaker,
agent_pool, debate, error_escalation, task_context, git_checkpoint,
agent_permissions, dag_analyzer, topology_router, and all modified files.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from vetinari.types import AgentType

# -- budget_tracker ----------------------------------------------------------


class TestBudgetTracker:
    def test_default_construction(self):
        from vetinari.agents.budget_tracker import BudgetTracker

        bt = BudgetTracker()
        assert not bt.is_exhausted()
        snap = bt.snapshot()
        assert snap.tokens_used == 0
        assert snap.is_exhausted is False

    def test_record_tokens(self):
        from vetinari.agents.budget_tracker import BudgetTracker

        bt = BudgetTracker(token_budget=100)
        bt.record_tokens(50)
        assert bt.snapshot().tokens_used == 50
        assert not bt.is_exhausted()
        bt.record_tokens(50)
        assert bt.is_exhausted()

    def test_record_tokens_negative_ignored(self):
        from vetinari.agents.budget_tracker import BudgetTracker

        bt = BudgetTracker(token_budget=100)
        bt.record_tokens(-10)
        assert bt.snapshot().tokens_used == 0

    def test_record_iteration(self):
        from vetinari.agents.budget_tracker import BudgetTracker

        bt = BudgetTracker(iteration_cap=3)
        for _ in range(3):
            bt.record_iteration()
        assert bt.is_exhausted()

    def test_record_cost(self):
        from vetinari.agents.budget_tracker import BudgetTracker

        bt = BudgetTracker(cost_budget_usd=1.0)
        bt.record_cost(0.5)
        assert not bt.is_exhausted()
        bt.record_cost(0.5)
        assert bt.is_exhausted()

    def test_record_delegation(self):
        from vetinari.agents.budget_tracker import BudgetTracker

        bt = BudgetTracker(delegation_budget=2)
        bt.record_delegation()
        assert not bt.is_exhausted()
        bt.record_delegation()
        assert bt.is_exhausted()

    def test_child_delegation_budget_conservation(self):
        from vetinari.agents.budget_tracker import BudgetTracker

        bt = BudgetTracker(delegation_budget=5)
        bt.record_delegation()  # used=1, remaining=4
        child_budget = bt.child_delegation_budget()
        assert child_budget == 3  # remaining - 1

    def test_child_delegation_budget_never_negative(self):
        from vetinari.agents.budget_tracker import BudgetTracker

        bt = BudgetTracker(delegation_budget=1)
        bt.record_delegation()  # exhausted
        assert bt.child_delegation_budget() == 0

    def test_from_agent_spec(self):
        from vetinari.agents.budget_tracker import BudgetTracker

        spec = MagicMock()
        spec.token_budget = 10_000
        spec.iteration_cap = 5
        spec.cost_budget_usd = 2.0
        spec.delegation_budget = 3
        spec.name = "test_agent"
        bt = BudgetTracker.from_agent_spec(spec)
        snap = bt.snapshot()
        assert snap.token_budget == 10_000
        assert snap.iteration_cap == 5
        assert snap.delegation_budget == 3

    def test_from_agent_spec_missing_fields(self):
        """AgentSpec without budget fields falls back to defaults."""
        from vetinari.agents.budget_tracker import (
            DEFAULT_DELEGATION_BUDGET,
            DEFAULT_TOKEN_BUDGET,
            BudgetTracker,
        )

        spec = MagicMock(spec=[])  # no budget attrs
        bt = BudgetTracker.from_agent_spec(spec)
        assert bt.snapshot().token_budget == DEFAULT_TOKEN_BUDGET
        assert bt.snapshot().delegation_budget == DEFAULT_DELEGATION_BUDGET

    def test_thread_safety(self):
        from vetinari.agents.budget_tracker import BudgetTracker

        bt = BudgetTracker(token_budget=10_000)
        errors: list[Exception] = []

        def add_tokens():
            try:
                for _ in range(100):
                    bt.record_tokens(10)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=add_tokens) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert bt.snapshot().tokens_used == 10_000

    def test_to_dict(self):
        from vetinari.agents.budget_tracker import BudgetTracker

        bt = BudgetTracker(token_budget=500)
        bt.record_tokens(100)
        d = bt.to_dict()
        assert d["tokens_used"] == 100
        assert d["token_budget"] == 500
        assert "is_exhausted" in d


# -- execution_feedback ------------------------------------------------------


class TestExecutionFeedback:
    def test_parse_success(self):
        from vetinari.agents.execution_feedback import parse_sandbox_output

        fb = parse_sandbox_output("5 passed in 1.23s", "", 0)
        assert fb.success is True
        assert fb.return_code == 0

    def test_parse_pytest_failures(self):
        from vetinari.agents.execution_feedback import parse_sandbox_output

        stdout = "FAILED tests/test_foo.py::test_bar - AssertionError: expected 1\n"
        fb = parse_sandbox_output(stdout, "", 1)
        assert fb.success is False
        assert len(fb.failed_tests) == 1
        assert fb.failed_tests[0].test_id == "tests/test_foo.py::test_bar"
        assert fb.failed_tests[0].kind == "FAILED"

    def test_parse_traceback(self):
        from vetinari.agents.execution_feedback import parse_sandbox_output

        stderr = (
            "Traceback (most recent call last):\n"
            "  File 'foo.py', line 10, in bar\n"
            "    raise ValueError('oops')\n"
            "ValueError: oops\n"
        )
        fb = parse_sandbox_output("", stderr, 1)
        assert len(fb.tracebacks) == 1
        assert "ValueError" in fb.tracebacks[0]

    def test_parse_type_errors(self):
        from vetinari.agents.execution_feedback import parse_sandbox_output

        stderr = "vetinari/foo.py:42: error: Argument 1 has incompatible type\n"
        fb = parse_sandbox_output("", stderr, 1)
        assert len(fb.type_errors) == 1
        assert "42" in fb.type_errors[0]

    def test_parse_lint_errors(self):
        from vetinari.agents.execution_feedback import parse_sandbox_output

        stdout = "vetinari/foo.py:10:5: E501 line too long\n"
        fb = parse_sandbox_output(stdout, "", 1)
        assert len(fb.lint_errors) == 1

    def test_tail_truncation(self):
        from vetinari.agents.execution_feedback import TAIL_LINES, parse_sandbox_output

        long_output = "\n".join(f"line {i}" for i in range(TAIL_LINES + 100))
        fb = parse_sandbox_output(long_output, "", 0)
        lines = fb.raw_stdout.splitlines()
        # Should include truncation notice + TAIL_LINES
        assert len(lines) <= TAIL_LINES + 1

    def test_to_prompt_success(self):
        from vetinari.agents.execution_feedback import ExecutionFeedback

        fb = ExecutionFeedback(success=True)
        prompt = fb.to_prompt()
        assert "Execution Feedback" in prompt

    def test_to_prompt_failures(self):
        from vetinari.agents.execution_feedback import ExecutionFeedback, TestResult

        fb = ExecutionFeedback(
            success=False,
            failed_tests=[TestResult("tests/x.py::test_a", "FAILED", "AssertionError")],
            tracebacks=["Traceback...\nValueError: bad"],
            summary="1 failed in 0.1s",
        )
        prompt = fb.to_prompt()
        assert "test_a" in prompt
        assert "Traceback" in prompt
        assert "1 failed" in prompt

    def test_pytest_summary_parsing(self):
        from vetinari.agents.execution_feedback import parse_sandbox_output

        output = "3 failed, 10 passed in 5.21s"
        fb = parse_sandbox_output(output, "", 1)
        assert "3 failed" in fb.summary
        assert "10 passed" in fb.summary


# -- agent_circuit_breaker ----------------------------------------------------


class TestAgentCircuitBreaker:
    @pytest.fixture(autouse=True)
    def _deterministic_clock(self, request) -> None:
        """Replace time.monotonic in the circuit breaker module for timing-sensitive tests.

        Under full-suite load on Windows, time.sleep(0.05) may not advance the
        OS clock far enough relative to a 10 ms cooldown.  For the two tests
        that rely on the cooldown expiring after a sleep, this fixture provides
        a counter-based monotonic clock: call 0 → 0.0 s, call 1+ → 1.0 s.
        One second is always >= the 0.01 s cooldown, making the transition
        deterministic regardless of load.

        Only active for the two timing-sensitive tests.  All other tests in
        this class use real time so their assertions (e.g. ``cooldown_seconds=60``
        must NOT expire immediately) continue to work correctly.
        """
        _TIMING_TESTS = frozenset({
            "test_half_open_after_cooldown",
            "test_half_open_probe_success_closes",
        })
        if request.node.name not in _TIMING_TESTS:
            yield
            return

        _calls = [-1]

        def _fake_monotonic() -> float:
            _calls[0] += 1
            # First call records _opened_at = 0.0; subsequent calls return 1.0
            # so elapsed = 1.0 - 0.0 = 1.0, which exceeds any cooldown <= 1.0 s.
            return 0.0 if _calls[0] == 0 else 1.0

        with patch(
            "vetinari.agents.agent_circuit_breaker.time.monotonic",
            side_effect=_fake_monotonic,
        ):
            yield

    def test_initial_state_closed(self):
        from vetinari.agents.agent_circuit_breaker import AgentCircuitBreaker, CircuitState

        cb = AgentCircuitBreaker("test", failure_threshold=3, cooldown_seconds=60)
        assert cb.get_status().state == CircuitState.CLOSED

    def test_trips_after_threshold(self):
        from vetinari.agents.agent_circuit_breaker import AgentCircuitBreaker, CircuitState

        cb = AgentCircuitBreaker("test", failure_threshold=3, cooldown_seconds=60)
        cb.record_failure()
        cb.record_failure()
        assert cb.get_status().state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.get_status().state == CircuitState.OPEN

    def test_open_rejects_requests(self):
        from vetinari.agents.agent_circuit_breaker import AgentCircuitBreaker

        cb = AgentCircuitBreaker("test", failure_threshold=1, cooldown_seconds=60)
        cb.record_failure()
        assert not cb.allow_request()
        assert cb.get_status().total_rejections >= 1

    def test_half_open_after_cooldown(self):
        from vetinari.agents.agent_circuit_breaker import AgentCircuitBreaker, CircuitState

        cb = AgentCircuitBreaker("test", failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        time.sleep(0.2)  # generous margin for slow CI runners
        cb.get_status()  # triggers maybe_recover
        assert cb.get_status().state == CircuitState.HALF_OPEN

    def test_half_open_probe_success_closes(self):
        from vetinari.agents.agent_circuit_breaker import AgentCircuitBreaker, CircuitState

        cb = AgentCircuitBreaker("test", failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        time.sleep(0.2)  # generous margin for slow CI runners
        assert cb.allow_request()  # probe
        cb.record_success()
        assert cb.get_status().state == CircuitState.CLOSED

    def test_half_open_probe_failure_reopens(self):
        from vetinari.agents.agent_circuit_breaker import AgentCircuitBreaker, CircuitState

        cb = AgentCircuitBreaker("test", failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        time.sleep(0.2)  # generous margin for slow CI runners
        cb.allow_request()  # probe
        cb._cooldown_seconds = 60  # keep the reopened state stable for this assertion
        cb.record_failure()
        assert cb.get_status().state == CircuitState.OPEN

    def test_success_resets_consecutive_failures(self):
        from vetinari.agents.agent_circuit_breaker import AgentCircuitBreaker

        cb = AgentCircuitBreaker("test", failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.get_status().consecutive_failures == 0

    def test_reset(self):
        from vetinari.agents.agent_circuit_breaker import AgentCircuitBreaker, CircuitState

        cb = AgentCircuitBreaker("test", failure_threshold=1)
        cb.record_failure()
        cb.reset()
        assert cb.get_status().state == CircuitState.CLOSED
        assert cb.get_status().total_failures == 0

    def test_invalid_threshold_raises(self):
        from vetinari.agents.agent_circuit_breaker import AgentCircuitBreaker

        with pytest.raises(ValueError, match="failure_threshold must be >= 1"):
            AgentCircuitBreaker("x", failure_threshold=0)

    def test_invalid_cooldown_raises(self):
        from vetinari.agents.agent_circuit_breaker import AgentCircuitBreaker

        with pytest.raises(ValueError, match="cooldown_seconds must be > 0"):
            AgentCircuitBreaker("x", cooldown_seconds=0)


# -- agent_pool ---------------------------------------------------------------


class TestAgentPool:
    def test_add_and_claim(self):
        from vetinari.orchestration.agent_pool import AgentPool

        pool = AgentPool(max_size=3)
        agent = MagicMock()
        pool.add_agent(agent, AgentType.WORKER.value)
        pa = pool.claim_worker(AgentType.WORKER.value)
        assert pa is not None
        assert pa.is_claimed is True

    def test_claim_miss(self):
        from vetinari.orchestration.agent_pool import AgentPool

        pool = AgentPool(max_size=3)
        pa = pool.claim_worker(AgentType.WORKER.value)
        assert pa is None

    def test_release_general_returns_to_idle(self):
        from vetinari.orchestration.agent_pool import AgentPool

        pool = AgentPool(max_size=3)
        agent = MagicMock()
        pool.add_agent(agent, AgentType.WORKER.value)
        pa = pool.claim_worker(AgentType.WORKER.value)
        pool.release_worker(pa.agent_id)
        assert pool.idle_count() == 1

    def test_release_specialist_reclaimed(self):
        from vetinari.orchestration.agent_pool import AgentPool

        pool = AgentPool(max_size=3)
        agent = MagicMock()
        pa = pool.spawn_specialist(agent, AgentType.INSPECTOR.value)
        assert pa is not None
        pool.release_worker(pa.agent_id)
        assert pool.pool_size() == 0

    def test_spawn_specialist_at_capacity(self):
        from vetinari.orchestration.agent_pool import AgentPool

        pool = AgentPool(max_size=1)
        pool.add_agent(MagicMock(), AgentType.WORKER.value)
        pa = pool.spawn_specialist(MagicMock(), AgentType.INSPECTOR.value)
        assert pa is None

    def test_reclaim_idle_specialists(self):
        from vetinari.orchestration.agent_pool import AgentPool

        pool = AgentPool(max_size=5, idle_reclaim_threshold=1)
        pool.add_agent(MagicMock(), AgentType.WORKER.value)
        # Spawn and immediately release 2 specialists to make them idle
        pa1 = pool.spawn_specialist(MagicMock(), AgentType.INSPECTOR.value)
        pool.release_worker(pa1.agent_id)
        # At this point specialist was reclaimed on release — pool has general worker only
        assert pool.pool_size() == 1

    def test_capacity_raises(self):
        from vetinari.orchestration.agent_pool import AgentPool

        pool = AgentPool(max_size=1)
        pool.add_agent(MagicMock(), AgentType.WORKER.value)
        with pytest.raises(RuntimeError):
            pool.add_agent(MagicMock(), AgentType.WORKER.value)

    def test_status_dict(self):
        from vetinari.orchestration.agent_pool import AgentPool

        pool = AgentPool(max_size=3)
        pool.add_agent(MagicMock(), AgentType.WORKER.value)
        status = pool.status()
        assert status["pool_size"] == 1
        assert status["max_size"] == 3
        assert len(status["agents"]) == 1


# -- debate -------------------------------------------------------------------


class TestDebateProtocol:
    def test_no_convergence_with_one_round(self):
        from vetinari.orchestration.debate import DebatePosition, DebateProtocol

        protocol = DebateProtocol("auth strategy")
        positions = [
            DebatePosition("hawk", "Use mTLS", ["certs", "mutual auth"], 0.9),
            DebatePosition("pragmatist", "Use JWT", ["simple", "stateless"], 0.7),
        ]
        converged = protocol.add_round(positions)
        assert not converged  # need >= 2 rounds

    def test_convergence_with_identical_rounds(self):
        from vetinari.orchestration.debate import DebatePosition, DebateProtocol

        protocol = DebateProtocol("arch")
        points = ["microservices", "event driven", "scalable"]
        pos = [DebatePosition("a", "use microservices", points, 0.8)]
        protocol.add_round(pos)
        converged = protocol.add_round(pos)
        assert converged is True

    def test_finalize_consensus_points(self):
        from vetinari.orchestration.debate import DebatePosition, DebateProtocol

        protocol = DebateProtocol("data model")
        shared_kp = ["use postgres", "normalize schema"]
        pos1 = [
            DebatePosition("arch", "normalised", [*shared_kp, "add indexes"], 0.8),
            DebatePosition("pragmatist", "normalised", [*shared_kp, "denormalize hot paths"], 0.7),
        ]
        protocol.add_round(pos1)
        protocol.add_round(pos1)
        result = protocol.finalize()
        assert "use postgres" in result.consensus_points or "normalize schema" in result.consensus_points

    def test_finalize_empty_returns_not_converged(self):
        from vetinari.orchestration.debate import DebateProtocol

        protocol = DebateProtocol("empty")
        result = protocol.finalize()
        assert not result.converged
        assert result.rounds_taken == 0

    def test_should_trigger_debate_security_low_confidence(self):
        from vetinari.orchestration.debate import should_trigger_debate
        from vetinari.types import GoalCategory

        assert should_trigger_debate(GoalCategory.SECURITY, 0.4) is True

    def test_should_trigger_debate_code_high_confidence(self):
        from vetinari.orchestration.debate import should_trigger_debate
        from vetinari.types import GoalCategory

        assert should_trigger_debate(GoalCategory.CODE, 0.9) is False

    def test_should_trigger_debate_force(self):
        from vetinari.orchestration.debate import should_trigger_debate
        from vetinari.types import GoalCategory

        assert should_trigger_debate(GoalCategory.CREATIVE, 0.9, force=True) is True

    def test_debate_result_to_dict(self):
        from vetinari.orchestration.debate import DebatePosition, DebateProtocol

        protocol = DebateProtocol("test")
        pos = [DebatePosition("a", "x", ["p1", "p2"], 0.8)]
        protocol.add_round(pos)
        protocol.add_round(pos)
        result = protocol.finalize()
        d = result.to_dict()
        assert "converged" in d
        assert "consensus_points" in d


# -- error_escalation ---------------------------------------------------------


class TestErrorClassifier:
    def test_transient_timeout(self):
        from vetinari.orchestration.error_escalation import EscalationLevel, get_error_classifier

        clf = get_error_classifier()
        result = clf.classify("Request timed out after 30s")
        assert result.level == EscalationLevel.TRANSIENT
        assert result.is_retryable

    def test_transient_rate_limit(self):
        from vetinari.orchestration.error_escalation import EscalationLevel, get_error_classifier

        clf = get_error_classifier()
        result = clf.classify("Rate limit exceeded: too many requests")
        assert result.level == EscalationLevel.TRANSIENT

    def test_semantic_ambiguous(self):
        from vetinari.orchestration.error_escalation import EscalationLevel, get_error_classifier

        clf = get_error_classifier()
        result = clf.classify("The request is ambiguous — unclear what output format is expected")
        assert result.level == EscalationLevel.SEMANTIC

    def test_capability_mismatch(self):
        from vetinari.orchestration.error_escalation import EscalationLevel, get_error_classifier

        clf = get_error_classifier()
        result = clf.classify("Task is not capable of being completed by this agent — requires specialist")
        assert result.level == EscalationLevel.CAPABILITY

    def test_fatal_impossible(self):
        from vetinari.orchestration.error_escalation import EscalationLevel, get_error_classifier

        clf = get_error_classifier()
        result = clf.classify("This task is impossible given the constraints")
        assert result.level == EscalationLevel.FATAL
        assert not result.is_retryable

    def test_unknown_error_transient_first(self):
        from vetinari.orchestration.error_escalation import EscalationLevel, get_error_classifier

        clf = get_error_classifier()
        result = clf.classify("some completely unknown error xyz")
        assert result.level == EscalationLevel.TRANSIENT

    def test_unknown_error_semantic_after_retries(self):
        from vetinari.orchestration.error_escalation import EscalationLevel, get_error_classifier

        clf = get_error_classifier()
        result = clf.classify("some unknown error", context={"retry_count": 3})
        assert result.level == EscalationLevel.SEMANTIC

    def test_singleton_same_instance(self):
        from vetinari.orchestration.error_escalation import get_error_classifier

        a = get_error_classifier()
        b = get_error_classifier()
        assert a is b

    def test_recovery_metrics_rate(self):
        from vetinari.orchestration.error_escalation import EscalationLevel, RecoveryMetrics

        m = RecoveryMetrics()
        m.record(EscalationLevel.TRANSIENT, resolved=True)
        m.record(EscalationLevel.SEMANTIC, resolved=False)
        assert m.resolution_rate() == pytest.approx(0.5)

    def test_recovery_metrics_zero_attempts(self):
        from vetinari.orchestration.error_escalation import RecoveryMetrics

        m = RecoveryMetrics()
        assert m.resolution_rate() == 0.0

    def test_classification_to_dict(self):
        from vetinari.orchestration.error_escalation import get_error_classifier

        clf = get_error_classifier()
        result = clf.classify("timeout")
        d = result.to_dict()
        assert "level" in d
        assert "level_name" in d
        assert "is_retryable" in d


# -- task_context -------------------------------------------------------------


class TestTaskContextManifest:
    def test_build_basic(self):
        from vetinari.orchestration.task_context import TaskManifestContext

        manifest = TaskManifestContext.build_for_task(
            task_id="t1",
            task_description="Implement auth module",
            goal="Build a web app",
        )
        assert manifest.task_id == "t1"
        assert manifest.goal == "Build a web app"

    def test_build_with_dependencies(self):
        from vetinari.orchestration.task_context import TaskManifestContext

        completed = {"t0": {"status": "done", "output": "schema defined"}}
        manifest = TaskManifestContext.build_for_task(
            task_id="t1",
            task_description="Implement auth",
            completed_results=completed,
            dependency_ids=["t0"],
        )
        assert len(manifest.dependency_outputs) == 1
        assert manifest.dependency_outputs[0]["task_id"] == "t0"

    def test_missing_dependency_skipped(self):
        from vetinari.orchestration.task_context import TaskManifestContext

        manifest = TaskManifestContext.build_for_task(
            task_id="t1",
            task_description="Build feature",
            completed_results={},
            dependency_ids=["missing_dep"],
        )
        assert len(manifest.dependency_outputs) == 0

    def test_format_for_prompt_contains_task(self):
        from vetinari.orchestration.task_context import TaskManifestContext

        manifest = TaskManifestContext.build_for_task(
            task_id="t1",
            task_description="Write tests for auth module",
            goal="Build a secure system",
        )
        prompt = manifest.format_for_prompt()
        assert "Write tests" in prompt
        assert "Build a secure system" in prompt

    def test_format_includes_constraints(self):
        from vetinari.orchestration.task_context import TaskManifestContext

        manifest = TaskManifestContext.build_for_task(
            task_id="t1",
            task_description="Do something",
            constraints={"max_cost_usd": 0.5, "timeout": 60},
        )
        prompt = manifest.format_for_prompt()
        assert "max_cost_usd" in prompt

    def test_long_output_truncated(self):
        from vetinari.orchestration.task_context import MAX_OUTPUT_PREVIEW_CHARS, TaskManifestContext

        long_output = "x" * (MAX_OUTPUT_PREVIEW_CHARS + 200)
        manifest = TaskManifestContext.build_for_task(
            task_id="t1",
            task_description="Task",
            completed_results={"dep": long_output},
            dependency_ids=["dep"],
        )
        prompt = manifest.format_for_prompt()
        assert "[truncated]" in prompt

    def test_memory_snippets_included(self):
        from vetinari.orchestration.task_context import TaskManifestContext

        memory = [{"type": "decision", "content": "Use PostgreSQL for persistence"}]
        manifest = TaskManifestContext.build_for_task(
            task_id="t1",
            task_description="DB task",
            memory=memory,
        )
        prompt = manifest.format_for_prompt()
        assert "PostgreSQL" in prompt


# -- git_checkpoint -----------------------------------------------------------


class TestGitCheckpoint:
    def test_no_op_when_git_unavailable(self):
        from vetinari.orchestration.git_checkpoint import GitCheckpoint

        cp = GitCheckpoint()
        cp._is_available = False
        result = cp.create_checkpoint("test")
        assert result.success
        assert "no-op" in result.message

    def test_rollback_no_op(self):
        from vetinari.orchestration.git_checkpoint import GitCheckpoint

        cp = GitCheckpoint()
        cp._is_available = False
        result = cp.rollback()
        assert result.success

    def test_commit_no_op(self):
        from vetinari.orchestration.git_checkpoint import GitCheckpoint

        cp = GitCheckpoint()
        cp._is_available = False
        result = cp.commit_accepted("feat: test")
        assert result.success

    def test_rollback_requires_stash_ref(self):
        from vetinari.orchestration.git_checkpoint import GitCheckpoint

        cp = GitCheckpoint()
        cp._is_available = True
        with patch("vetinari.orchestration.git_checkpoint._run_git", return_value=(True, "", "")):
            result = cp.rollback(stash_ref="")
        assert not result.success

    def test_checkpoint_result_repr(self):
        from vetinari.orchestration.git_checkpoint import CheckpointResult

        r = CheckpointResult(success=True, stash_ref="stash@{0}")
        assert "stash@{0}" in repr(r)


# -- agent_permissions --------------------------------------------------------


class TestAgentPermissions:
    def test_foreman_can_read(self):
        from vetinari.security.agent_permissions import AgentAction, AgentPermissions
        from vetinari.types import AgentType

        ap = AgentPermissions()
        assert ap.is_allowed(AgentType.FOREMAN, AgentAction.READ_FILE)

    def test_foreman_cannot_execute_shell(self):
        from vetinari.security.agent_permissions import AgentAction, AgentPermissions
        from vetinari.types import AgentType

        ap = AgentPermissions()
        assert not ap.is_allowed(AgentType.FOREMAN, AgentAction.EXECUTE_SHELL)

    def test_worker_can_write_file(self):
        from vetinari.security.agent_permissions import AgentAction, AgentPermissions
        from vetinari.types import AgentType

        ap = AgentPermissions()
        assert ap.is_allowed(AgentType.WORKER, AgentAction.WRITE_FILE)

    def test_inspector_cannot_write_file(self):
        from vetinari.security.agent_permissions import AgentAction, AgentPermissions
        from vetinari.types import AgentType

        ap = AgentPermissions()
        assert not ap.is_allowed(AgentType.INSPECTOR, AgentAction.WRITE_FILE)

    def test_inspector_can_approve_plan(self):
        from vetinari.security.agent_permissions import AgentAction, AgentPermissions
        from vetinari.types import AgentType

        ap = AgentPermissions()
        assert ap.is_allowed(AgentType.INSPECTOR, AgentAction.APPROVE_PLAN)

    def test_missing_policy_denied(self):
        from vetinari.security.agent_permissions import AgentAction, AgentPermissions
        from vetinari.types import AgentType

        # Use a custom policy dict that omits FOREMAN entirely
        ap = AgentPermissions(policies={AgentType.WORKER: MagicMock(allows=lambda a: True)})
        assert not ap.is_allowed(AgentType.FOREMAN, AgentAction.READ_FILE)

    def test_dual_permission_check(self):
        from vetinari.security.agent_permissions import AgentAction, AgentPermissions
        from vetinari.types import AgentType

        ap = AgentPermissions()
        # FOREMAN can read, WORKER can read — both allow
        assert ap.check_dual_permission(AgentType.FOREMAN, AgentType.WORKER, AgentAction.READ_FILE)

    def test_dual_permission_denied_if_one_denies(self):
        from vetinari.security.agent_permissions import AgentAction, AgentPermissions
        from vetinari.types import AgentType

        ap = AgentPermissions()
        # INSPECTOR cannot WRITE_FILE
        assert not ap.check_dual_permission(AgentType.WORKER, AgentType.INSPECTOR, AgentAction.WRITE_FILE)

    def test_fail_closed_on_exception(self):
        from vetinari.security.agent_permissions import AgentAction, AgentPermissions
        from vetinari.types import AgentType

        ap = AgentPermissions()
        # Make .get() raise to simulate a broken policy store
        broken = MagicMock()
        broken.get = MagicMock(side_effect=RuntimeError("boom"))
        ap._policies = broken
        result = ap.is_allowed(AgentType.FOREMAN, AgentAction.READ_FILE)
        assert result is False

    def test_to_dict_all_agents(self):
        from vetinari.security.agent_permissions import AgentPermissions
        from vetinari.types import AgentType

        ap = AgentPermissions()
        d = ap.to_dict()
        assert AgentType.FOREMAN.value in d
        assert AgentType.WORKER.value in d
        assert AgentType.INSPECTOR.value in d

    def test_get_allowed_actions_non_empty(self):
        from vetinari.security.agent_permissions import AgentPermissions
        from vetinari.types import AgentType

        ap = AgentPermissions()
        actions = ap.get_allowed_actions(AgentType.WORKER)
        assert len(actions) > 0


# -- dag_analyzer -------------------------------------------------------------


class TestDAGAnalyzer:
    def test_empty_tasks(self):
        from vetinari.routing.dag_analyzer import analyze_dag

        shape = analyze_dag([])
        assert shape.task_count == 0

    def test_single_task(self):
        from vetinari.routing.dag_analyzer import analyze_dag

        shape = analyze_dag([{"id": "t1", "dependencies": []}])
        assert shape.task_count == 1
        assert shape.max_depth == 0
        assert shape.independent_tasks == 1

    def test_linear_chain(self):
        from vetinari.routing.dag_analyzer import analyze_dag

        tasks = [
            {"id": "t1", "dependencies": []},
            {"id": "t2", "dependencies": ["t1"]},
            {"id": "t3", "dependencies": ["t2"]},
        ]
        shape = analyze_dag(tasks)
        assert shape.max_depth == 2
        assert shape.independent_tasks == 1

    def test_parallel_tasks(self):
        from vetinari.routing.dag_analyzer import analyze_dag

        tasks = [
            {"id": "t1", "dependencies": []},
            {"id": "t2", "dependencies": []},
            {"id": "t3", "dependencies": []},
        ]
        shape = analyze_dag(tasks)
        assert shape.independent_tasks == 3
        assert shape.parallelism_potential == pytest.approx(1.0)

    def test_fan_out_detection(self):
        from vetinari.routing.dag_analyzer import analyze_dag

        tasks = [
            {"id": "root", "dependencies": []},
            {"id": "t1", "dependencies": ["root"]},
            {"id": "t2", "dependencies": ["root"]},
            {"id": "t3", "dependencies": ["root"]},
            {"id": "t4", "dependencies": ["root"]},
        ]
        shape = analyze_dag(tasks)
        assert shape.max_fan_out == 4
        assert shape.has_bottleneck

    def test_connected_components(self):
        from vetinari.routing.dag_analyzer import analyze_dag

        tasks = [
            {"id": "a1", "dependencies": []},
            {"id": "a2", "dependencies": ["a1"]},
            {"id": "b1", "dependencies": []},  # separate component
        ]
        shape = analyze_dag(tasks)
        assert shape.connected_components == 2

    def test_suggest_topology_express(self):
        from vetinari.routing.dag_analyzer import DAGShape, suggest_topology

        shape = DAGShape(task_count=1)
        assert suggest_topology(shape) == "express"

    def test_suggest_topology_parallel(self):
        from vetinari.routing.dag_analyzer import DAGShape, suggest_topology

        shape = DAGShape(task_count=6, independent_tasks=5, parallelism_potential=0.8)
        assert suggest_topology(shape) == "parallel"

    def test_suggest_topology_scatter_gather(self):
        from vetinari.routing.dag_analyzer import DAGShape, suggest_topology

        shape = DAGShape(task_count=5, connected_components=2)
        assert suggest_topology(shape) == "scatter_gather"

    def test_dag_shape_to_dict(self):
        from vetinari.routing.dag_analyzer import DAGShape

        shape = DAGShape(task_count=3, max_depth=2)
        d = shape.to_dict()
        assert d["task_count"] == 3
        assert d["max_depth"] == 2


# -- topology_router ----------------------------------------------------------


class TestTopologyRouter:
    def test_simple_task_express(self):
        from vetinari.routing.topology_router import Topology, TopologyRouter

        router = TopologyRouter()
        decision = router.route("Fix typo in readme", complexity_hint="simple")
        assert decision.topology == Topology.EXPRESS

    def test_complex_security_debate(self):
        from vetinari.routing.topology_router import Topology, TopologyRouter

        router = TopologyRouter()
        decision = router.route(
            "Design the security architecture and authentication strategy for the platform",
            complexity_hint="complex",
        )
        assert decision.topology == Topology.DEBATE

    def test_parallel_batch(self):
        from vetinari.routing.topology_router import Topology, TopologyRouter

        router = TopologyRouter()
        decision = router.route("Process each item in the batch concurrently", complexity_hint="moderate")
        assert decision.topology == Topology.PARALLEL

    def test_sequential_pipeline(self):
        from vetinari.routing.topology_router import Topology, TopologyRouter

        router = TopologyRouter()
        decision = router.route("Build a step by step data pipeline", complexity_hint="moderate")
        assert decision.topology == Topology.SEQUENTIAL

    def test_dag_shape_overrides_keywords(self):
        from vetinari.routing.dag_analyzer import DAGShape
        from vetinari.routing.topology_router import Topology, TopologyRouter

        router = TopologyRouter()
        shape = DAGShape(task_count=1)  # single task → express
        decision = router.route("Complex architecture task", dag_shape=shape)
        assert decision.topology == Topology.EXPRESS

    def test_decision_to_dict(self):
        from vetinari.routing.topology_router import TopologyRouter

        router = TopologyRouter()
        decision = router.route("simple task", complexity_hint="simple")
        d = decision.to_dict()
        assert "topology" in d
        assert "complexity" in d
        assert "recommended_agents" in d

    def test_recommended_agents_non_empty(self):
        from vetinari.routing.topology_router import TopologyRouter

        router = TopologyRouter()
        decision = router.route("test task")
        assert len(decision.recommended_agents) > 0

    def test_execution_strategy_code_mode_for_express(self):
        from vetinari.routing.topology_router import TopologyRouter

        router = TopologyRouter()
        decision = router.route("tiny task", complexity_hint="simple")
        assert decision.execution_strategy == "code_mode"


# -- types new enums ----------------------------------------------------------


class TestNewEnums:
    def test_inference_status_values(self):
        from vetinari.types import InferenceStatus

        assert InferenceStatus.SUCCESS.value == "success"
        assert InferenceStatus.FALLBACK.value == "fallback"
        assert InferenceStatus.CONTEXT_OVERFLOW.value == "context_overflow"

    def test_failure_category_values(self):
        from vetinari.types import FailureCategory

        assert FailureCategory.TRANSIENT.value == "transient"
        assert FailureCategory.FATAL.value == "fatal"
        assert FailureCategory.BUDGET.value == "budget"

    def test_permission_tier_values(self):
        from vetinari.types import PermissionTier

        assert PermissionTier.TIER_0.value == "tier_0"
        assert PermissionTier.TIER_3.value == "tier_3"


# -- contracts enhancements ---------------------------------------------------


class TestContractsEnhancements:
    def test_agent_spec_budget_fields(self):
        from vetinari.agents.contracts import AgentSpec
        from vetinari.types import AgentType

        spec = AgentSpec(
            agent_type=AgentType.WORKER,
            name="test",
            description="test agent",
            default_model="test-model",
            token_budget=5000,
            delegation_budget=3,
        )
        assert spec.token_budget == 5000
        assert spec.delegation_budget == 3

    def test_agent_spec_budget_defaults(self):
        from vetinari.agents.contracts import AgentSpec
        from vetinari.types import AgentType

        spec = AgentSpec(
            agent_type=AgentType.FOREMAN,
            name="foreman",
            description="orchestrator",
            default_model="llama3",
        )
        assert spec.token_budget == 32_000
        assert spec.iteration_cap == 10
        assert spec.cost_budget_usd == 1.0
        assert spec.delegation_budget == 5

    def test_agent_result_enhanced_fields(self):
        from vetinari.agents.contracts import AgentResult
        from vetinari.types import InferenceStatus

        result = AgentResult(
            success=True,
            output="done",
            task_id="t1",
            status=InferenceStatus.SUCCESS,
            output_type="code",
        )
        assert result.task_id == "t1"
        assert result.status == InferenceStatus.SUCCESS
        assert result.output_type == "code"

    def test_agent_task_context_budget_default(self):
        from vetinari.agents.contracts import AgentTask
        from vetinari.types import AgentType

        task = AgentTask(
            task_id="t1",
            agent_type=AgentType.WORKER,
            description="do something",
            prompt="do something",
        )
        assert task.context_budget_tokens == 4096


# -- circuit_breaker budget_exhaustion ----------------------------------------


class TestCircuitBreakerBudgetExhaustion:
    def test_budget_exhaustion_counts_as_failure(self):
        from vetinari.resilience.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker("test_budget")
        cb.record_budget_exhaustion()
        assert cb.stats.total_failures == 1

    def test_budget_exhaustion_trips_breaker(self):
        from vetinari.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState

        cfg = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test_budget_trip", config=cfg)
        cb.record_budget_exhaustion()
        cb.record_budget_exhaustion()
        assert cb.state == CircuitState.OPEN


# -- stagnation scope param ---------------------------------------------------


class TestStagnationScope:
    def test_scope_stored(self):
        from vetinari.orchestration.stagnation import StagnationDetector

        d = StagnationDetector(scope="plan-123")
        assert d.scope == "plan-123"

    def test_scope_default_empty(self):
        from vetinari.orchestration.stagnation import StagnationDetector

        d = StagnationDetector()
        assert d.scope == ""


# -- decomposition recursive --------------------------------------------------


class TestDecompositionRecursive:
    def test_max_recursive_depth_constant(self):
        from vetinari.planning.decomposition import MAX_RECURSIVE_DEPTH

        assert MAX_RECURSIVE_DEPTH == 3

    def test_decompose_recursive_returns_list(self):
        from vetinari.planning.decomposition import DecompositionEngine

        engine = DecompositionEngine()
        # Patch decompose_task to return simple subtasks
        with patch.object(
            engine,
            "decompose_task",
            return_value=[
                {
                    "subtask_id": "st_001",
                    "description": "Do the thing",
                    "agent_type": AgentType.WORKER.value,
                    "depth": 1,
                    "inputs": [],
                    "outputs": [],
                    "dependencies": [],
                    "acceptance_criteria": "",
                    "parent_task_id": "root",
                }
            ],
        ):
            result = engine.decompose_recursive("Build a web app", recursive_depth=0)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_decompose_recursive_stops_at_max_depth(self):
        from vetinari.planning.decomposition import MAX_RECURSIVE_DEPTH, DecompositionEngine

        engine = DecompositionEngine()
        call_count = {"n": 0}

        def fake_decompose(**kwargs):
            call_count["n"] += 1
            return []

        with patch.object(engine, "decompose_task", side_effect=fake_decompose):
            engine.decompose_recursive("task", recursive_depth=MAX_RECURSIVE_DEPTH)
        # Should call decompose_task exactly once (no recursion at max depth)
        assert call_count["n"] == 1


# -- subtask new fields -------------------------------------------------------


class TestSubtaskTopologyFields:
    def test_subtask_topology_defaults(self):
        from vetinari.planning.plan_types import Subtask

        s = Subtask(description="test task")
        assert s.assigned_plan_id == ""
        assert s.execution_topology == ""

    def test_subtask_topology_set(self):
        from vetinari.planning.plan_types import Subtask

        s = Subtask(description="complex task", execution_topology="hierarchical", assigned_plan_id="plan_abc")
        assert s.execution_topology == "hierarchical"
        assert s.assigned_plan_id == "plan_abc"


# -- code sandbox to_feedback -------------------------------------------------


class TestExecutionResultToFeedback:
    def test_to_feedback_success(self):
        from vetinari.code_sandbox import ExecutionResult

        result = ExecutionResult(
            success=True,
            output="5 passed",
            stdout="5 passed in 1.0s",
            return_code=0,
        )
        fb = result.to_feedback()
        assert fb.success is True

    def test_to_feedback_failure(self):
        from vetinari.code_sandbox import ExecutionResult

        result = ExecutionResult(
            success=False,
            output="",
            stderr="FAILED tests/test_x.py::test_y - AssertionError",
            return_code=1,
        )
        fb = result.to_feedback()
        assert not fb.success
        assert len(fb.failed_tests) >= 1
