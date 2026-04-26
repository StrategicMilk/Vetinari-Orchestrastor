"""Tests verifying the system recovers from various failure scenarios.

Covers five crash/failure recovery paths:
1. Server crash mid-pipeline (durable execution resumes from checkpoint)
2. Model crash mid-inference (circuit breaker opens, subsequent calls fail fast)
3. Disk full during training data write (OSError caught, system continues)
4. Concurrent write to failure registry JSONL (no corruption, no lost entries)
5. Remediation actions that return False (system reports failure, not success)
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stubs for heavy optional modules before importing modules that reference them
# ---------------------------------------------------------------------------
_mock_quality_scorer = MagicMock()
_mock_quality_scorer.score.return_value = MagicMock(overall_score=0.8)

_mock_feedback_loop = MagicMock()
_mock_thompson = MagicMock()

_mock_quality_scorer_mod = MagicMock()
_mock_quality_scorer_mod.get_quality_scorer.return_value = _mock_quality_scorer

_mock_feedback_loop_mod = MagicMock()
_mock_feedback_loop_mod.get_feedback_loop.return_value = _mock_feedback_loop

_mock_model_selector_mod = MagicMock()
_mock_model_selector_mod.get_thompson_selector.return_value = _mock_thompson

sys.modules.setdefault("vetinari.learning.quality_scorer", _mock_quality_scorer_mod)
sys.modules.setdefault("vetinari.learning.feedback_loop", _mock_feedback_loop_mod)
sys.modules.setdefault("vetinari.learning.model_selector", _mock_model_selector_mod)

# ---------------------------------------------------------------------------
# Real imports — safe after stubs are in place
# ---------------------------------------------------------------------------
import pytest

import vetinari.analytics.failure_registry as _fr_module
from tests.factories import make_durable_engine, make_execution_graph, make_execution_task_node
from vetinari.analytics.failure_registry import (
    FailureRegistry,
    get_failure_registry,
    reset_failure_registry,
)
from vetinari.orchestration.durable_execution import DurableExecutionEngine
from vetinari.orchestration.execution_graph import ExecutionGraph, ExecutionTaskNode
from vetinari.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    ResilienceCircuitState,
    reset_circuit_breaker_registry,
)
from vetinari.system.remediation import (
    FailureMode,
    RemediationAction,
    RemediationEngine,
    RemediationPlan,
    RemediationTier,
)
from vetinari.types import PlanStatus, StatusEnum

# ---------------------------------------------------------------------------
# Shared autouse fixture — isolate the unified database per test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Redirect the unified database to a temp directory for each test."""
    _test_db = str(tmp_path / "test_crash.db")
    monkeypatch.setenv("VETINARI_DB_PATH", _test_db)
    from vetinari.database import reset_for_testing

    reset_for_testing()
    yield
    reset_for_testing()


# ---------------------------------------------------------------------------
# 1. TestPipelineResumeAfterCrash
# ---------------------------------------------------------------------------


class TestPipelineResumeAfterCrash:
    """Server crash mid-pipeline — durable execution resumes from checkpoint."""

    def test_resume_starts_from_task_2_not_task_1(self, tmp_path: Path) -> None:
        """Simulate crash after task 1; recovery must skip task 1 and run task 2+.

        Build a 3-task linear pipeline. Simulate a server crash by running the
        first execution with task_b set to max_retries=0 and a handler that
        raises for task_b. After the first run, task_a is COMPLETED in the
        graph, task_b is FAILED, and task_c is BLOCKED. Reset task_b and task_c
        to PENDING (simulating a crash reset), then verify that the recovery
        engine completes all three tasks without re-running task_a.
        """
        # Track which task IDs were actually executed by the handler.
        executed_task_ids: list[str] = []

        # Build a 3-node linear graph: task_a -> task_b -> task_c
        graph = make_execution_graph()
        graph.add_task("task_a", "first task")
        task_b_node = graph.add_task("task_b", "second task", depends_on=["task_a"])
        graph.add_task("task_c", "third task", depends_on=["task_b"])

        # No retries: task_b failing once counts as a definitive failure in the
        # first run, which is what we want to assert (failed >= 1).
        task_b_node.max_retries = 0

        engine = make_durable_engine(str(tmp_path / "checkpoints"))

        # crash_armed controls whether task_b raises.  Armed for the first run,
        # disarmed before the recovery run so the handler succeeds on retry.
        crash_armed: list[bool] = [True]

        def crashing_handler(task: ExecutionTaskNode) -> dict[str, Any]:
            executed_task_ids.append(task.id)
            if task.id == "task_b" and crash_armed[0]:
                raise RuntimeError("simulated server crash mid-pipeline")
            return {"result": "ok", "task_id": task.id}

        # The engine should catch the task failure and record it as failed.
        with patch("time.sleep"):
            result_first = engine.execute_plan(graph, crashing_handler)

        # task_a ran; task_b raised → it is in failed
        assert "task_a" in executed_task_ids, "task_a should have executed before crash"
        assert result_first["failed"] >= 1, "at least one task should be recorded as failed after crash"

        # Disarm the crash before the recovery run.
        crash_armed[0] = False

        # Simulate crash recovery: reset task_b (FAILED) and task_c (BLOCKED)
        # to PENDING so they can be re-executed.  task_a stays COMPLETED so
        # the recovery engine skips it entirely (same graph object, same plan_id).
        graph.nodes["task_b"].status = StatusEnum.PENDING
        graph.nodes["task_b"].error = ""
        graph.nodes["task_b"].retry_count = 0
        graph.nodes["task_c"].status = StatusEnum.PENDING
        graph.nodes["task_c"].error = ""

        executed_task_ids.clear()

        recovery_engine = make_durable_engine(str(tmp_path / "checkpoints"))
        with patch("time.sleep"):
            result_recovery = recovery_engine.execute_plan(graph, crashing_handler)

        assert result_recovery["failed"] == 0, "all tasks should succeed on recovery run"
        # Recovery engine only executes the 2 remaining tasks (task_b, task_c).
        # task_a was already COMPLETED and is skipped by the graph scheduler.
        assert result_recovery["completed"] == 2, (
            "recovery run must complete exactly the 2 remaining tasks (task_b and task_c)"
        )

        # task_a must NOT have been re-executed during recovery
        assert "task_a" not in executed_task_ids, "task_a was already completed and must not run again on recovery"
        assert "task_b" in executed_task_ids, "task_b must run during recovery"
        assert "task_c" in executed_task_ids, "task_c must run during recovery"


# ---------------------------------------------------------------------------
# 2. TestCircuitBreakerOpensOnModelCrash
# ---------------------------------------------------------------------------


class TestCircuitBreakerOpensOnModelCrash:
    """Model crash mid-inference — circuit breaker opens, subsequent calls fail fast."""

    @pytest.fixture(autouse=True)
    def _reset_breaker(self) -> Any:
        """Ensure a clean circuit breaker registry between tests."""
        reset_circuit_breaker_registry()
        yield
        reset_circuit_breaker_registry()

    def test_circuit_opens_after_threshold_failures(self) -> None:
        """Repeated model failures must trip the circuit breaker to OPEN state.

        Configure a breaker with failure_threshold=3. Record three consecutive
        failures. Assert the state transitions to OPEN and subsequent calls
        are rejected with CircuitBreakerOpen rather than executing the function.
        """
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60.0,
            half_open_max_calls=1,
        )
        breaker = CircuitBreaker(name="test_model_breaker", config=config)

        assert breaker.state == ResilienceCircuitState.CLOSED, "breaker must start CLOSED"

        # Simulate three model inference failures
        for _ in range(3):
            breaker.record_failure()

        assert breaker.state == ResilienceCircuitState.OPEN, "breaker must be OPEN after hitting failure_threshold"

    def test_open_circuit_rejects_without_executing_callable(self) -> None:
        """An OPEN circuit must raise CircuitBreakerOpen without calling the function.

        Verify that the wrapped callable is never invoked when the circuit is open,
        and that the exception carries the breaker name.
        """
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=120.0)
        breaker = CircuitBreaker(name="inference_model", config=config)
        breaker.record_failure()  # trip the breaker

        assert breaker.state == ResilienceCircuitState.OPEN

        invocation_count: list[int] = [0]

        def inference_fn() -> str:
            invocation_count[0] += 1
            return "model output"

        with pytest.raises(CircuitBreakerOpen):
            breaker.call(inference_fn)

        assert invocation_count[0] == 0, "inference callable must NOT be invoked when circuit breaker is OPEN"

    def test_stats_record_rejections_separately_from_failures(self) -> None:
        """Rejection count must increment independently of failure count.

        Rejections are requests refused by an open breaker — they are different
        from actual execution failures that caused the breaker to open.
        """
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=120.0)
        breaker = CircuitBreaker(name="stats_test", config=config)
        breaker.record_failure()  # open the breaker

        assert breaker.state == ResilienceCircuitState.OPEN

        # Two attempts while open
        for _ in range(2):
            with pytest.raises(CircuitBreakerOpen):
                breaker.call(lambda: None)

        stats = breaker.stats
        assert stats.total_rejections == 2, "each rejected call must increment total_rejections"
        assert stats.total_failures == 1, "only the original failure counts as a failure"


# ---------------------------------------------------------------------------
# 3. TestDiskFullDuringWrite
# ---------------------------------------------------------------------------


class TestDiskFullDuringWrite:
    """Disk full during training data write — OSError is caught, system continues."""

    @pytest.fixture(autouse=True)
    def _reset_registry(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
        """Redirect failure registry writes to tmp_path, reset singleton between tests."""
        monkeypatch.setattr(_fr_module, "_REGISTRY_DIR", tmp_path)
        reset_failure_registry()
        yield
        reset_failure_registry()

    def test_oserror_during_append_is_caught_and_not_raised(self, caplog: pytest.LogCaptureFixture) -> None:
        """OSError inside _append_entry must be caught internally — must never propagate.

        _append_entry uses Path.open(). Patch pathlib.Path.open to raise OSError
        (simulating "no space left on device") and call _append_entry directly.
        The method must swallow the OSError and log it — the caller must never see
        the exception. This is the graceful-degradation contract.
        """
        import logging

        from vetinari.analytics.failure_registry import FailureRegistryEntry

        registry = get_failure_registry()

        # Build a minimal entry — _append_entry takes a FailureRegistryEntry directly
        test_entry = FailureRegistryEntry(
            failure_id="fail_test000001",
            timestamp="2026-01-01T00:00:00+00:00",
            category="disk_full_test",
            severity="error",
            description="disk full simulation",
        )

        # Patch Path.open to raise OSError — simulates disk-full condition
        with patch("pathlib.Path.open", side_effect=OSError("no space left on device")):
            with caplog.at_level(logging.ERROR, logger="vetinari.analytics.failure_registry"):
                # Must NOT raise — _append_entry catches OSError internally
                registry._append_entry(test_entry)

        # The error must be logged — silent swallowing without logging is forbidden
        assert any("Could not write failure entry" in r.message for r in caplog.records), (
            "_append_entry must log an error when it cannot write — silent swallowing is forbidden"
        )

    def test_oserror_during_append_does_not_corrupt_subsequent_writes(self, tmp_path: Path) -> None:
        """After a failed write, subsequent successful writes must still persist.

        First write: patched to fail with OSError.
        Second write: no patch — must succeed and be readable from the JSONL file.
        """
        registry = get_failure_registry()

        # First write fails — patch Path.open so the real _append_entry runs
        # and catches the OSError internally (testing graceful degradation).
        # Using patch.object on _append_entry would bypass the internal
        # try/except and let OSError escape log_failure uncaught.
        with patch("pathlib.Path.open", side_effect=OSError("disk full")):
            registry.log_failure(
                category="first_write",
                severity="warning",
                description="First write attempt",
            )

        # Second write — no patch — must succeed
        entry2 = registry.log_failure(
            category="second_write",
            severity="error",
            description="Second write attempt after recovery",
        )

        # Reload entries from disk and verify only entry2 is present
        entries = registry._load_all_entries()
        categories = [e["category"] for e in entries]
        assert "second_write" in categories, "successful write after OSError must persist"
        assert entry2.failure_id.startswith("fail_")


# ---------------------------------------------------------------------------
# 4. TestConcurrentFailureRegistryWrites
# ---------------------------------------------------------------------------


class TestConcurrentFailureRegistryWrites:
    """Concurrent write to failure registry JSONL — no corruption, no lost entries."""

    @pytest.fixture(autouse=True)
    def _reset_registry(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
        """Redirect failure registry to tmp_path and reset singleton."""
        # Reset FIRST to clear the cached _REGISTRY_DIR singleton variable
        reset_failure_registry()
        # THEN patch so subsequent calls to _get_registry_dir() see tmp_path
        monkeypatch.setattr(_fr_module, "_REGISTRY_DIR", tmp_path)
        yield
        reset_failure_registry()

    def test_ten_concurrent_threads_produce_no_lost_entries(self) -> None:
        """Ten threads writing simultaneously must produce exactly 10 JSONL entries.

        Each thread logs exactly one failure. After all threads complete, read
        the JSONL file and assert it contains exactly 10 valid entries.
        """
        _thread_count = 10
        registry = get_failure_registry()
        errors: list[Exception] = []

        def write_one(thread_index: int) -> None:
            try:
                registry.log_failure(
                    category="concurrent_test",
                    severity="warning",
                    description=f"Thread {thread_index} failure",
                    root_cause=f"simulated failure from thread {thread_index}",
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write_one, args=(i,)) for i in range(_thread_count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors, f"threads raised exceptions: {errors}"

        entries = registry._load_all_entries()
        assert len(entries) == _thread_count, (
            f"expected {_thread_count} entries, got {len(entries)} — entries may have been lost or duplicated"
        )

        # All entries must be valid JSONL (already parsed — verify required fields present)
        for entry in entries:
            assert "failure_id" in entry, "each entry must have a failure_id"
            assert entry["failure_id"].startswith("fail_"), "failure_id must be prefixed with 'fail_'"
            assert "category" in entry, "each entry must have a category"

    def test_no_jsonl_line_is_malformed_after_concurrent_writes(self, tmp_path: Path) -> None:
        """Concurrent writes must not interleave bytes, producing only complete JSON lines.

        Read the raw JSONL file line by line after concurrent writes and assert
        every non-empty line is valid JSON.
        """
        registry = get_failure_registry()

        def write_batch(thread_index: int) -> None:
            for i in range(5):
                registry.log_failure(
                    category="line_integrity",
                    severity="warning",
                    description=f"Thread {thread_index} write {i}",
                )

        threads = [threading.Thread(target=write_batch, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)

        # Read raw file and verify every line parses as JSON
        import json

        registry_file = tmp_path / "failure-registry.jsonl"
        assert registry_file.exists(), "registry file must be created after writes"

        malformed_lines: list[int] = []
        with registry_file.open(encoding="utf-8") as f:
            for lineno, raw_line in enumerate(f, 1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    json.loads(stripped)
                except json.JSONDecodeError:
                    malformed_lines.append(lineno)

        assert not malformed_lines, f"malformed JSON on lines: {malformed_lines} — concurrent writes corrupted the file"


# ---------------------------------------------------------------------------
# 5. TestRemediationActionStateChange
# ---------------------------------------------------------------------------


class TestRemediationActionStateChange:
    """Remediation actions that return False — system reports failure not success."""

    def _make_failing_plan(self, failure_mode: FailureMode = FailureMode.DISK_FULL) -> RemediationPlan:
        """Build a plan where every action_fn returns False (no state change)."""
        actions = [
            RemediationAction(
                description="Attempt cache flush",
                tier=RemediationTier.AUTO_FIX,
                action_fn=lambda: False,  # no state change
            ),
            RemediationAction(
                description="Alert operator",
                tier=RemediationTier.ALERT,
                action_fn=lambda: False,  # no state change
            ),
        ]
        return RemediationPlan(
            failure_mode=failure_mode,
            diagnosis="Disk full — all remediation paths returning False",
            actions=actions,
            max_tier=RemediationTier.ALERT,
        )

    @pytest.fixture(autouse=True)
    def _reset_registry(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
        """Redirect failure registry and reset singletons for isolation."""
        monkeypatch.setattr(_fr_module, "_REGISTRY_DIR", tmp_path)
        reset_failure_registry()
        yield
        reset_failure_registry()

    def test_all_false_actions_produce_success_false_result(self) -> None:
        """When every action_fn returns False, result.success must be False.

        The engine must not declare success when no action resolved the failure.
        This is the fail-closed contract for the remediation subsystem.
        """
        engine = RemediationEngine()
        plan = self._make_failing_plan()

        result = engine.execute_remediation(plan)

        assert result.success is False, (
            "RemediationEngine must report success=False when all action_fn calls return False"
        )
        assert result.failure_mode == FailureMode.DISK_FULL
        assert result.error is not None, "error field must explain why remediation failed"

    def test_all_false_actions_record_all_attempted_descriptions(self) -> None:
        """All attempted action descriptions must appear in actions_taken.

        Even when every action fails, the engine must record the full audit trail
        of what was attempted — not silently drop the failed actions.
        """
        engine = RemediationEngine()
        plan = self._make_failing_plan()

        result = engine.execute_remediation(plan)

        assert "Attempt cache flush" in result.actions_taken, "first action description must be in actions_taken"
        assert "Alert operator" in result.actions_taken, "second action description must be in actions_taken"

    def test_false_result_does_not_masquerade_as_partial_success(self) -> None:
        """A result with success=False must not have a truthy success field under any coercion.

        Guards against accidental `result.success = 1` or similar.
        """
        engine = RemediationEngine()
        plan = self._make_failing_plan(failure_mode=FailureMode.OOM)

        result = engine.execute_remediation(plan)

        assert result.success is False
        assert bool(result.success) is False, "success must be strictly False, not a truthy non-bool"

    def test_informational_action_without_callable_does_not_count_as_success(self) -> None:
        """An action with action_fn=None is informational only — must not trigger success.

        When the only action has no callable, the engine should exhaust the plan
        and report success=False, not treat the informational record as a resolution.
        """
        engine = RemediationEngine()
        plan = RemediationPlan(
            failure_mode=FailureMode.THERMAL,
            diagnosis="GPU thermal throttling detected",
            actions=[
                RemediationAction(
                    description="Log thermal event for operator review",
                    tier=RemediationTier.ALERT,
                    action_fn=None,  # informational only — no callable
                ),
            ],
            max_tier=RemediationTier.ALERT,
        )

        result = engine.execute_remediation(plan)

        # action_fn=None skips execution — no resolution occurred
        assert result.success is False, "informational-only actions (action_fn=None) must not produce success=True"


# ---------------------------------------------------------------------------
# 6. TestDownstreamCancelledTasksDoNotExecute
# ---------------------------------------------------------------------------


class TestDownstreamCancelledTasksDoNotExecute:
    """When a dependency fails, downstream tasks marked CANCELLED must not run.

    The CANCELLED guard added to ``_execute_task`` prevents phantom completions:
    _handle_layer_failure marks downstream nodes CANCELLED, but since layers are
    pre-computed by ``get_execution_order()``, those tasks would otherwise reach
    the thread pool and call the handler.  This class verifies the guard works.
    """

    def test_cancelled_downstream_task_handler_never_called(self, tmp_path: Path) -> None:
        """Handler must NOT be called for a task whose dependency failed.

        Build a 2-task graph where task_b depends on task_a.  Register a handler
        that always fails for task_a (raising RuntimeError, max_retries=0).
        Assert that after execution, task_b is CANCELLED and the handler was
        NOT invoked for task_b.
        """
        # Track which task IDs the handler was called for.
        handler_call_ids: list[str] = []

        graph = make_execution_graph(plan_id="plan-cancelled-guard")
        task_a = graph.add_task("task_a", "always-failing upstream task")
        task_a.max_retries = 0
        graph.add_task("task_b", "downstream task", depends_on=["task_a"])

        engine = make_durable_engine(str(tmp_path / "checkpoints"))

        def tracking_handler(task: ExecutionTaskNode) -> dict[str, Any]:
            handler_call_ids.append(task.id)
            if task.id == "task_a":
                raise RuntimeError("upstream task intentionally fails")
            return {"result": "ok", "task_id": task.id}

        with patch("time.sleep"):
            results = engine.execute_plan(graph, tracking_handler)

        # task_a failed — must appear in failed count.
        assert results["failed"] >= 1, "task_a must be recorded as failed"

        # The handler must NEVER have been called for task_b.
        assert "task_b" not in handler_call_ids, "handler must not be called for a CANCELLED downstream task"

        # task_b must be CANCELLED (set by _handle_layer_failure), not COMPLETED.
        assert graph.nodes["task_b"].status == StatusEnum.CANCELLED, (
            "downstream task status must be CANCELLED, not COMPLETED or PENDING"
        )

    def test_cancelled_task_does_not_inflate_completed_count(self, tmp_path: Path) -> None:
        """A CANCELLED downstream task must not count toward the completed total.

        Ensures the CANCELLED guard in ``_execute_task`` returns the CANCELLED
        sentinel without incrementing the completed counter, so the caller
        cannot mistake a cancelled task for a successful one.
        """
        graph = make_execution_graph(plan_id="plan-cancelled-count")
        task_a = graph.add_task("task_a", "failing upstream")
        task_a.max_retries = 0
        graph.add_task("task_b", "cancelled downstream", depends_on=["task_a"])

        engine = make_durable_engine(str(tmp_path / "checkpoints"))

        def failing_handler(task: ExecutionTaskNode) -> dict[str, Any]:
            if task.id == "task_a":
                raise RuntimeError("upstream fails")
            return {"result": "ok"}

        with patch("time.sleep"):
            results = engine.execute_plan(graph, failing_handler)

        # Only task_a ran and failed — completed count must be 0.
        assert results[StatusEnum.COMPLETED.value] == 0, "CANCELLED tasks must not inflate the completed count"
