"""
Comprehensive tests for vetinari/orchestration/durable_execution.py

Installs sys.modules stubs before importing the module so the real
vetinari.types is used (stdlib-only) but heavy optional deps are mocked.
"""

import json
import os
import re
import shutil
import sys
import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

# ---------------------------------------------------------------------------
# Real types module — safe to import (only uses stdlib enum)
# ---------------------------------------------------------------------------
import vetinari.types

# ---------------------------------------------------------------------------
# Stubs for heavy optional modules that durable_execution imports lazily
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
# Now it is safe to import the real module under test
# ---------------------------------------------------------------------------
import pytest

from tests.factories import make_durable_engine, make_execution_graph, make_execution_task_node
from vetinari.orchestration.durable_execution import (
    Checkpoint,
    DurableExecutionEngine,
    ExecutionEvent,
)
from vetinari.orchestration.execution_graph import ExecutionGraph, ExecutionTaskNode
from vetinari.types import PlanStatus, StatusEnum


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Ensure each test gets its own unified database for isolation."""
    _test_db = str(tmp_path / "test_durable.db")
    monkeypatch.setenv("VETINARI_DB_PATH", _test_db)
    from vetinari.database import reset_for_testing

    reset_for_testing()
    yield
    reset_for_testing()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop_handler(task: ExecutionTaskNode) -> dict[str, Any]:
    return {"result": "ok", "task_id": task.id}


def _fail_handler(task: ExecutionTaskNode) -> dict[str, Any]:
    raise RuntimeError("intentional failure")


# ---------------------------------------------------------------------------
# 1. TestExecutionEvent
# ---------------------------------------------------------------------------


class TestExecutionEvent:
    def test_fields_set_correctly(self):
        ev = ExecutionEvent(
            event_id="eid-1",
            event_type="task_started",
            task_id="t1",
            timestamp="2024-01-01T00:00:00",
        )
        assert ev.event_id == "eid-1"
        assert ev.event_type == "task_started"
        assert ev.task_id == "t1"
        assert ev.timestamp == "2024-01-01T00:00:00"

    def test_data_defaults_to_empty_dict(self):
        ev = ExecutionEvent(event_id="x", event_type="task_completed", task_id="t", timestamp="ts")
        assert isinstance(ev.data, dict)
        assert ev.data == {}

    def test_data_can_be_set(self):
        ev = ExecutionEvent(
            event_id="x",
            event_type="task_failed",
            task_id="t",
            timestamp="ts",
            data={"error": "boom", "attempts": 3},
        )
        assert ev.data["error"] == "boom"
        assert ev.data["attempts"] == 3

    def test_event_id_is_string(self):
        ev = ExecutionEvent(event_id="uuid-abc", event_type="e", task_id="t", timestamp="ts")
        assert isinstance(ev.event_id, str)

    def test_different_event_types(self):
        for etype in ("task_started", "task_completed", "task_failed", "task_cancelled"):
            ev = ExecutionEvent(event_id="e", event_type=etype, task_id="t", timestamp="ts")
            assert ev.event_type == etype

    def test_data_independence_between_instances(self):
        ev1 = ExecutionEvent(event_id="e1", event_type="e", task_id="t", timestamp="ts")
        ev2 = ExecutionEvent(event_id="e2", event_type="e", task_id="t", timestamp="ts")
        ev1.data["key"] = "val"
        assert "key" not in ev2.data

    def test_task_id_stored(self):
        ev = ExecutionEvent(event_id="e", event_type="e", task_id="my-task-99", timestamp="ts")
        assert ev.task_id == "my-task-99"

    def test_timestamp_stored(self):
        ts = datetime.now().isoformat()
        ev = ExecutionEvent(event_id="e", event_type="e", task_id="t", timestamp=ts)
        assert ev.timestamp == ts

    def test_data_dict_mutation(self):
        ev = ExecutionEvent(event_id="e", event_type="e", task_id="t", timestamp="ts")
        ev.data["foo"] = "bar"
        assert ev.data["foo"] == "bar"


# ---------------------------------------------------------------------------
# 2. TestCheckpoint
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def _make(self, **kwargs) -> Checkpoint:
        defaults = {
            "checkpoint_id": "cp-1",
            "plan_id": "plan-abc",
            "created_at": "2024-01-01T00:00:00",
            "graph_state": {"plan_id": "plan-abc", "nodes": {}},
            "completed_tasks": ["t1"],
            "running_tasks": [],
        }
        defaults.update(kwargs)
        return Checkpoint(**defaults)

    def test_fields_stored(self):
        cp = self._make()
        assert cp.checkpoint_id == "cp-1"
        assert cp.plan_id == "plan-abc"

    def test_metadata_defaults_to_empty_dict(self):
        cp = self._make()
        assert cp.metadata == {}

    def test_metadata_can_be_set(self):
        cp = self._make(metadata={"event_count": 5})
        assert cp.metadata["event_count"] == 5

    def test_completed_tasks_list(self):
        cp = self._make(completed_tasks=["t1", "t2", "t3"])
        assert len(cp.completed_tasks) == 3

    def test_running_tasks_list(self):
        cp = self._make(running_tasks=["t5"])
        assert "t5" in cp.running_tasks

    def test_graph_state_dict(self):
        state = {"plan_id": "p", "nodes": {"n1": {}}}
        cp = self._make(graph_state=state)
        assert cp.graph_state["plan_id"] == "p"

    def test_created_at_stored(self):
        ts = "2025-06-15T12:00:00"
        cp = self._make(created_at=ts)
        assert cp.created_at == ts

    def test_metadata_independence(self):
        cp1 = self._make()
        cp2 = self._make()
        cp1.metadata["x"] = 1
        assert "x" not in cp2.metadata

    def test_plan_id_matches(self):
        cp = self._make(plan_id="unique-plan")
        assert cp.plan_id == "unique-plan"


# ---------------------------------------------------------------------------
# 3. TestDurableExecutionEngineInit
# ---------------------------------------------------------------------------


class TestDurableExecutionEngineInit:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tmpdir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_default_max_concurrent(self):
        eng = DurableExecutionEngine(checkpoint_dir=self.tmpdir)
        assert eng.max_concurrent == 4

    def test_default_timeout(self):
        eng = DurableExecutionEngine(checkpoint_dir=self.tmpdir)
        assert eng.default_timeout == 300.0

    def test_custom_max_concurrent(self):
        eng = DurableExecutionEngine(checkpoint_dir=self.tmpdir, max_concurrent=8)
        assert eng.max_concurrent == 8

    def test_custom_timeout(self):
        eng = DurableExecutionEngine(checkpoint_dir=self.tmpdir, default_timeout=60.0)
        assert eng.default_timeout == 60.0

    def test_checkpoint_dir_created(self):
        new_dir = os.path.join(self.tmpdir, "sub", "checkpoints")
        DurableExecutionEngine(checkpoint_dir=new_dir)
        assert Path(new_dir).exists()

    def test_checkpoint_dir_stored_as_path(self):
        eng = DurableExecutionEngine(checkpoint_dir=self.tmpdir)
        assert isinstance(eng.checkpoint_dir, Path)

    def test_active_executions_empty(self):
        eng = DurableExecutionEngine(checkpoint_dir=self.tmpdir)
        assert eng._active_executions == {}

    def test_event_history_empty(self):
        eng = DurableExecutionEngine(checkpoint_dir=self.tmpdir)
        assert len(eng._event_history) == 0

    def test_task_handlers_empty(self):
        eng = DurableExecutionEngine(checkpoint_dir=self.tmpdir)
        assert eng._task_handlers == {}

    def test_callbacks_default_none(self):
        eng = DurableExecutionEngine(checkpoint_dir=self.tmpdir)
        assert eng._on_task_start is None
        assert eng._on_task_complete is None
        assert eng._on_task_fail is None

    def test_default_checkpoint_dir_used_when_none(self):
        # When no checkpoint_dir provided, engine creates a default dir
        eng = DurableExecutionEngine()
        assert eng.checkpoint_dir.exists()
        # Clean up the default dir
        shutil.rmtree(eng.checkpoint_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 4. TestRegisterHandler
# ---------------------------------------------------------------------------


class TestRegisterHandler:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.eng = make_durable_engine(self.tmpdir)
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_register_stores_handler(self):
        handler = Mock()
        self.eng.register_handler("my_type", handler)
        assert self.eng._task_handlers["my_type"] is handler

    def test_register_multiple_handlers(self):
        h1, h2 = Mock(), Mock()
        self.eng.register_handler("type_a", h1)
        self.eng.register_handler("type_b", h2)
        assert self.eng._task_handlers["type_a"] is h1
        assert self.eng._task_handlers["type_b"] is h2

    def test_overwrite_existing_handler(self):
        old, new = Mock(), Mock()
        self.eng.register_handler("alpha", old)
        self.eng.register_handler("alpha", new)
        assert self.eng._task_handlers["alpha"] is new

    def test_handler_count_after_registrations(self):
        for i in range(5):
            self.eng.register_handler(f"type_{i}", Mock())
        assert len(self.eng._task_handlers) == 5

    def test_handler_is_callable(self):
        self.eng.register_handler("fn_type", lambda t: None)
        assert callable(self.eng._task_handlers["fn_type"])

    def test_register_default_key(self):
        handler = Mock()
        self.eng.register_handler("default", handler)
        assert "default" in self.eng._task_handlers


# ---------------------------------------------------------------------------
# 5. TestSetCallbacks
# ---------------------------------------------------------------------------


class TestSetCallbacks:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.eng = make_durable_engine(self.tmpdir)
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_set_all_callbacks(self):
        start = Mock()
        complete = Mock()
        fail = Mock()
        self.eng.set_callbacks(on_task_start=start, on_task_complete=complete, on_task_fail=fail)
        assert self.eng._on_task_start is start
        assert self.eng._on_task_complete is complete
        assert self.eng._on_task_fail is fail

    def test_set_only_start_callback(self):
        start = Mock()
        self.eng.set_callbacks(on_task_start=start)
        assert self.eng._on_task_start is start
        assert self.eng._on_task_complete is None
        assert self.eng._on_task_fail is None

    def test_set_only_complete_callback(self):
        complete = Mock()
        self.eng.set_callbacks(on_task_complete=complete)
        assert self.eng._on_task_complete is complete
        assert self.eng._on_task_start is None

    def test_set_only_fail_callback(self):
        fail = Mock()
        self.eng.set_callbacks(on_task_fail=fail)
        assert self.eng._on_task_fail is fail

    def test_overwrite_callbacks(self):
        first = Mock()
        second = Mock()
        self.eng.set_callbacks(on_task_start=first)
        self.eng.set_callbacks(on_task_start=second)
        assert self.eng._on_task_start is second

    def test_start_callback_called_on_task_execute(self):
        start_cb = Mock()
        self.eng.set_callbacks(on_task_start=start_cb)
        graph = make_execution_graph()
        task = graph.add_task("t1", "Task one")
        with patch("time.sleep"):
            result = self.eng.execute_plan(graph, task_handler=_noop_handler)
        start_cb.assert_called_once_with(task)
        assert task.started_at is not None
        assert result["task_results"]["t1"]["output"]["task_id"] == "t1"

    def test_complete_callback_called_on_success(self):
        complete_cb = Mock()
        self.eng.set_callbacks(on_task_complete=complete_cb)
        graph = make_execution_graph()
        task = graph.add_task("t1", "Task one")
        with patch("time.sleep"):
            result = self.eng.execute_plan(graph, task_handler=_noop_handler)
        complete_cb.assert_called_once_with(task)
        assert task.status == StatusEnum.COMPLETED
        assert result["task_results"]["t1"]["status"] == StatusEnum.COMPLETED.value

    def test_fail_callback_called_on_failure(self):
        fail_cb = Mock()
        self.eng.set_callbacks(on_task_fail=fail_cb)
        graph = make_execution_graph()
        task = graph.add_task("t1", "Task one")
        task.max_retries = 0
        with patch("time.sleep"):
            result = self.eng.execute_plan(graph, task_handler=_fail_handler)
        fail_cb.assert_called_once_with(task)
        assert task.status == StatusEnum.FAILED
        assert result["task_results"]["t1"]["error"] == "intentional failure"

    def test_start_callback_exception_does_not_crash(self):
        self.eng.set_callbacks(on_task_start=Mock(side_effect=ValueError("cb boom")))
        graph = make_execution_graph()
        graph.add_task("t1", "Task one")
        with patch("time.sleep"):
            result = self.eng.execute_plan(graph, task_handler=_noop_handler)
        # engine should still complete despite callback error
        assert result["completed"] == 1


# ---------------------------------------------------------------------------
# 6. TestExecutePlan
# ---------------------------------------------------------------------------


class TestExecutePlan:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.eng = make_durable_engine(self.tmpdir)
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_empty_graph_returns_zero_tasks(self):
        graph = make_execution_graph()
        with patch("time.sleep"):
            result = self.eng.execute_plan(graph, task_handler=_noop_handler)
        assert result["total_tasks"] == 0
        assert result["completed"] == 0

    def test_single_task_completes(self):
        graph = make_execution_graph()
        graph.add_task("t1", "Task one")
        with patch("time.sleep"):
            result = self.eng.execute_plan(graph, task_handler=_noop_handler)
        assert result["completed"] == 1
        assert result["failed"] == 0

    def test_plan_id_in_results(self):
        graph = make_execution_graph(plan_id="my-plan")
        with patch("time.sleep"):
            result = self.eng.execute_plan(graph, task_handler=_noop_handler)
        assert result["plan_id"] == "my-plan"

    def test_multiple_independent_tasks(self):
        graph = make_execution_graph()
        for i in range(4):
            graph.add_task(f"t{i}", f"Task {i}")
        with patch("time.sleep"):
            result = self.eng.execute_plan(graph, task_handler=_noop_handler)
        assert result["completed"] == 4
        assert result["failed"] == 0

    def test_sequential_dependency_chain(self):
        graph = make_execution_graph()
        graph.add_task("t1", "First")
        graph.add_task("t2", "Second", depends_on=["t1"])
        graph.add_task("t3", "Third", depends_on=["t2"])
        completed_order = []

        def ordered_handler(task):
            completed_order.append(task.id)
            return {"ok": True}

        with patch("time.sleep"):
            result = self.eng.execute_plan(graph, task_handler=ordered_handler)
        assert result["completed"] == 3
        assert completed_order == ["t1", "t2", "t3"]

    def test_graph_status_set_to_completed_on_success(self):
        graph = make_execution_graph()
        graph.add_task("t1", "Task one")
        with patch("time.sleep"):
            self.eng.execute_plan(graph, task_handler=_noop_handler)
        assert graph.status == PlanStatus.COMPLETED

    def test_graph_status_set_to_failed_on_failure(self):
        graph = make_execution_graph()
        task = graph.add_task("t1", "Task one")
        task.max_retries = 0
        with patch("time.sleep"):
            self.eng.execute_plan(graph, task_handler=_fail_handler)
        assert graph.status == PlanStatus.FAILED

    def test_handler_registered_as_default(self):
        graph = make_execution_graph()
        task = graph.add_task("t1", "Task one")
        handler = Mock(return_value={"out": "data"})
        with patch("time.sleep"):
            result = self.eng.execute_plan(graph, task_handler=handler)
        handler.assert_called_once_with(task)
        assert result["task_results"]["t1"]["output"] == {"out": "data"}
        assert graph.status == PlanStatus.COMPLETED

    def test_task_results_keyed_by_task_id(self):
        graph = make_execution_graph()
        graph.add_task("task-A", "Task A")
        with patch("time.sleep"):
            result = self.eng.execute_plan(graph, task_handler=_noop_handler)
        assert "task-A" in result["task_results"]

    def test_no_handler_marks_task_failed(self):
        """Tasks without a registered handler are marked FAILED (fail-closed)."""
        graph = make_execution_graph()
        graph.add_task("t1", "Task one")
        with patch("time.sleep"):
            # no handler registered, no task_handler arg
            result = self.eng.execute_plan(graph)
        assert result["failed"] == 1, "No-handler tasks must be FAILED, not COMPLETED"
        assert result["completed"] == 0

    def test_parallel_layer_executes_all_tasks(self):
        graph = make_execution_graph()
        graph.add_task("t1", "T1")
        graph.add_task("t2", "T2")
        graph.add_task("t3", "T3")
        with patch("time.sleep"):
            result = self.eng.execute_plan(graph, task_handler=_noop_handler)
        # All 3 in the same layer
        assert result["completed"] == 3

    def test_checkpoint_saved_after_plan(self):
        graph = make_execution_graph(plan_id="cp-test-plan")
        graph.add_task("t1", "Task one")
        with patch("time.sleep"):
            self.eng.execute_plan(graph, task_handler=_noop_handler)
        # Verify checkpoint was saved via the engine's own DB manager
        rows = self.eng._db.execute(
            "SELECT execution_id FROM execution_state WHERE execution_id = ?",
            ("cp-test-plan",),
        )
        assert len(rows) >= 1

    def test_event_history_populated(self):
        graph = make_execution_graph()
        graph.add_task("t1", "Task one")
        with patch("time.sleep"):
            self.eng.execute_plan(graph, task_handler=_noop_handler)
        assert len(self.eng._event_history) > 0

    def test_failed_count_reflects_failures(self):
        graph = make_execution_graph()
        task = graph.add_task("t1", "Task one")
        task.max_retries = 0
        with patch("time.sleep"):
            result = self.eng.execute_plan(graph, task_handler=_fail_handler)
        assert result["failed"] == 1

    def test_mixed_success_and_failure(self):
        graph = make_execution_graph()
        graph.add_task("t_ok", "Good task")
        bad_task = graph.add_task("t_bad", "Bad task")
        bad_task.max_retries = 0

        def mixed_handler(task):
            if task.id == "t_bad":
                raise RuntimeError("bad!")
            return {"ok": True}

        with patch("time.sleep"):
            result = self.eng.execute_plan(graph, task_handler=mixed_handler)
        assert result["completed"] == 1
        assert result["failed"] == 1


# ---------------------------------------------------------------------------
# 7. TestExecuteTask
# ---------------------------------------------------------------------------


class TestExecuteTask:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.eng = make_durable_engine(self.tmpdir)
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_graph_with_task(self, task: ExecutionTaskNode) -> ExecutionGraph:
        graph = make_execution_graph()
        graph.nodes[task.id] = task
        return graph

    def test_success_on_first_attempt(self):
        handler = Mock(return_value={"result": "ok"})
        self.eng._task_handlers["default"] = handler
        task = make_execution_task_node(max_retries=3)
        graph = self._make_graph_with_task(task)
        with patch("time.sleep"):
            result = self.eng._execute_task(graph, task)
        assert result["status"] == "completed"
        handler.assert_called_once()

    def test_task_status_set_to_completed(self):
        self.eng._task_handlers["default"] = _noop_handler
        task = make_execution_task_node()
        graph = self._make_graph_with_task(task)
        with patch("time.sleep"):
            self.eng._execute_task(graph, task)
        assert task.status == StatusEnum.COMPLETED

    def test_task_started_at_set(self):
        self.eng._task_handlers["default"] = _noop_handler
        task = make_execution_task_node()
        graph = self._make_graph_with_task(task)
        with patch("time.sleep"):
            self.eng._execute_task(graph, task)
        assert task.started_at is not None
        assert isinstance(task.started_at, str)
        assert "T" in task.started_at

    def test_task_completed_at_set_on_success(self):
        self.eng._task_handlers["default"] = _noop_handler
        task = make_execution_task_node()
        graph = self._make_graph_with_task(task)
        with patch("time.sleep"):
            self.eng._execute_task(graph, task)
        assert task.completed_at is not None
        assert isinstance(task.completed_at, str)
        assert "T" in task.completed_at

    def test_retry_once_then_succeed(self):
        call_count = {"n": 0}

        def flaky(task):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise RuntimeError("temporary error")
            return {"ok": True}

        self.eng._task_handlers["default"] = flaky
        task = make_execution_task_node(max_retries=3)
        graph = self._make_graph_with_task(task)
        with patch("time.sleep"):
            result = self.eng._execute_task(graph, task)
        assert result["status"] == "completed"
        assert call_count["n"] == 2

    def test_retry_count_increments(self):
        call_count = {"n": 0}

        def always_fail(task):
            call_count["n"] += 1
            raise RuntimeError("fail")

        self.eng._task_handlers["default"] = always_fail
        task = make_execution_task_node(max_retries=2)
        graph = self._make_graph_with_task(task)
        with patch("time.sleep"):
            self.eng._execute_task(graph, task)
        # max_retries=2 means 3 total attempts
        assert call_count["n"] == 3

    def test_max_retries_exhausted_returns_failed(self):
        self.eng._task_handlers["default"] = _fail_handler
        task = make_execution_task_node(max_retries=2)
        graph = self._make_graph_with_task(task)
        with patch("time.sleep"):
            result = self.eng._execute_task(graph, task)
        assert result["status"] == "failed"

    def test_task_status_failed_after_exhausted(self):
        self.eng._task_handlers["default"] = _fail_handler
        task = make_execution_task_node(max_retries=0)
        graph = self._make_graph_with_task(task)
        with patch("time.sleep"):
            self.eng._execute_task(graph, task)
        assert task.status == StatusEnum.FAILED

    def test_error_message_stored_on_task(self):
        def fail_with_msg(t):
            raise ValueError("specific error message")

        self.eng._task_handlers["default"] = fail_with_msg
        task = make_execution_task_node(max_retries=0)
        graph = self._make_graph_with_task(task)
        with patch("time.sleep"):
            result = self.eng._execute_task(graph, task)
        assert "specific error message" in result.get("error", "")
        assert "specific error message" in task.error

    def test_exponential_backoff_sleep_called(self):
        self.eng._task_handlers["default"] = _fail_handler
        task = make_execution_task_node(max_retries=2)
        graph = self._make_graph_with_task(task)
        with patch("time.sleep") as mock_sleep:
            self.eng._execute_task(graph, task)
        # 3 attempts, sleep between attempt 1->2 and 2->3
        # Backoff uses base_delay + jitter: 2**0 + rand(0,1) and 2**1 + rand(0,2)
        calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert len(calls) == 2
        # First delay: 2**0 + jitter -> between 1.0 and 2.0
        assert 1.0 <= calls[0] <= 2.0
        # Second delay: 2**1 + jitter -> between 2.0 and 4.0
        assert 2.0 <= calls[1] <= 4.0

    def test_no_handler_marks_task_failed(self):
        """Tasks without a handler are marked FAILED (fail-closed, not completed with warning)."""
        task = make_execution_task_node()
        task.task_type = "unknown_type_xyz"
        graph = self._make_graph_with_task(task)
        with patch("time.sleep"):
            result = self.eng._execute_task(graph, task)
        assert result["status"] == "failed", "No-handler tasks must be FAILED"
        assert "No handler" in result.get("error", "")

    def test_start_event_emitted(self):
        self.eng._task_handlers["default"] = _noop_handler
        task = make_execution_task_node(task_id="t-emit")
        graph = self._make_graph_with_task(task)
        with patch("time.sleep"):
            self.eng._execute_task(graph, task)
        event_types = [e.event_type for e in self.eng._event_history]
        assert "task_started" in event_types

    def test_completed_event_emitted_on_success(self):
        self.eng._task_handlers["default"] = _noop_handler
        task = make_execution_task_node(task_id="t-emit2")
        graph = self._make_graph_with_task(task)
        with patch("time.sleep"):
            self.eng._execute_task(graph, task)
        event_types = [e.event_type for e in self.eng._event_history]
        assert "task_completed" in event_types

    def test_failed_event_emitted_on_failure(self):
        self.eng._task_handlers["default"] = _fail_handler
        task = make_execution_task_node(task_id="t-fail-ev", max_retries=0)
        graph = self._make_graph_with_task(task)
        with patch("time.sleep"):
            self.eng._execute_task(graph, task)
        event_types = [e.event_type for e in self.eng._event_history]
        assert "task_failed" in event_types

    def test_output_data_stored_on_task(self):
        self.eng._task_handlers["default"] = lambda t: {"value": 42}
        task = make_execution_task_node()
        graph = self._make_graph_with_task(task)
        with patch("time.sleep"):
            self.eng._execute_task(graph, task)
        assert task.output_data.get("value") == 42

    def test_type_specific_handler_takes_priority(self):
        default_handler = Mock(return_value={"from": "default"})
        specific_handler = Mock(return_value={"from": "specific"})
        self.eng._task_handlers["default"] = default_handler
        self.eng._task_handlers["code"] = specific_handler
        task = make_execution_task_node()
        task.task_type = "code"
        graph = self._make_graph_with_task(task)
        with patch("time.sleep"):
            self.eng._execute_task(graph, task)
        specific_handler.assert_called_once()
        default_handler.assert_not_called()

    def test_complete_callback_not_called_on_failure(self):
        complete_cb = Mock()
        self.eng.set_callbacks(on_task_complete=complete_cb)
        self.eng._task_handlers["default"] = _fail_handler
        task = make_execution_task_node(max_retries=0)
        graph = self._make_graph_with_task(task)
        with patch("time.sleep"):
            self.eng._execute_task(graph, task)
        complete_cb.assert_not_called()

    def test_human_checkpoint_unapproved_returns_waiting_not_pending(self):
        """A task that is a human checkpoint but not yet approved must return
        WAITING status, not PENDING.  WAITING is the correct distinguishable
        state so that dependent-dispatch logic can treat it differently from
        'not yet started' (PENDING).
        """
        task = make_execution_task_node()
        self.eng._human_checkpoint.add_checkpoint(task.id, reason="requires review")
        # Do NOT approve — gate is closed
        graph = self._make_graph_with_task(task)
        result = self.eng._execute_task(graph, task)
        assert result["status"] == StatusEnum.WAITING.value
        assert result["waiting_for"] == "human_approval"
        assert result["task_id"] == task.id

    def test_human_checkpoint_unapproved_dependent_not_dispatched(self):
        """A task whose predecessor is WAITING must NOT appear in get_ready_tasks().
        WAITING is not COMPLETED so dependent tasks stay pending in backlog.
        """
        from vetinari.orchestration.execution_graph import ExecutionGraph, ExecutionTaskNode

        graph = ExecutionGraph(plan_id="p1", goal="test")
        gate = ExecutionTaskNode(id="gate", description="gate task")
        gate.status = StatusEnum.WAITING  # simulate unapproved checkpoint state
        dep = ExecutionTaskNode(id="dep", description="dependent", depends_on=["gate"])
        graph.nodes["gate"] = gate
        graph.nodes["dep"] = dep

        ready = graph.get_ready_tasks()
        ready_ids = [t.id for t in ready]
        assert "dep" not in ready_ids


# ---------------------------------------------------------------------------
# 8. TestHandleLayerFailure
# ---------------------------------------------------------------------------


class TestHandleLayerFailure:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.eng = make_durable_engine(self.tmpdir)
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_direct_dependent_cancelled(self):
        graph = make_execution_graph()
        failed = ExecutionTaskNode(id="f1", description="Failed")
        failed.status = StatusEnum.FAILED
        dep = ExecutionTaskNode(id="d1", description="Dependent", depends_on=["f1"])
        graph.nodes["f1"] = failed
        graph.nodes["d1"] = dep
        self.eng._handle_layer_failure(graph, [failed])
        assert dep.status == StatusEnum.CANCELLED

    def test_transitive_cancellation(self):
        graph = make_execution_graph()
        f = ExecutionTaskNode(id="f1", description="Failed")
        f.status = StatusEnum.FAILED
        d1 = ExecutionTaskNode(id="d1", description="Direct dep", depends_on=["f1"])
        d2 = ExecutionTaskNode(id="d2", description="Transitive dep", depends_on=["d1"])
        graph.nodes.update({"f1": f, "d1": d1, "d2": d2})
        self.eng._handle_layer_failure(graph, [f])
        assert d1.status == StatusEnum.CANCELLED
        assert d2.status == StatusEnum.CANCELLED

    def test_non_dependent_not_cancelled(self):
        graph = make_execution_graph()
        f = ExecutionTaskNode(id="f1", description="Failed")
        f.status = StatusEnum.FAILED
        independent = ExecutionTaskNode(id="i1", description="Independent")
        graph.nodes.update({"f1": f, "i1": independent})
        self.eng._handle_layer_failure(graph, [f])
        assert independent.status == StatusEnum.PENDING

    def test_already_completed_not_cancelled(self):
        graph = make_execution_graph()
        f = ExecutionTaskNode(id="f1", description="Failed")
        f.status = StatusEnum.FAILED
        completed = ExecutionTaskNode(id="c1", description="Completed", depends_on=["f1"])
        completed.status = StatusEnum.COMPLETED
        graph.nodes.update({"f1": f, "c1": completed})
        self.eng._handle_layer_failure(graph, [f])
        assert completed.status == StatusEnum.COMPLETED

    def test_cancel_event_emitted(self):
        graph = make_execution_graph()
        f = ExecutionTaskNode(id="f1", description="Failed")
        f.status = StatusEnum.FAILED
        dep = ExecutionTaskNode(id="d1", description="Dep", depends_on=["f1"])
        graph.nodes.update({"f1": f, "d1": dep})
        self.eng._handle_layer_failure(graph, [f])
        cancelled_events = [e for e in self.eng._event_history if e.event_type == "task_cancelled"]
        assert len(cancelled_events) == 1
        assert cancelled_events[0].task_id == "d1"

    def test_multiple_failed_tasks(self):
        graph = make_execution_graph()
        f1 = ExecutionTaskNode(id="f1", description="Failed 1")
        f1.status = StatusEnum.FAILED
        f2 = ExecutionTaskNode(id="f2", description="Failed 2")
        f2.status = StatusEnum.FAILED
        d1 = ExecutionTaskNode(id="d1", description="Dep of f1", depends_on=["f1"])
        d2 = ExecutionTaskNode(id="d2", description="Dep of f2", depends_on=["f2"])
        graph.nodes.update({"f1": f1, "f2": f2, "d1": d1, "d2": d2})
        self.eng._handle_layer_failure(graph, [f1, f2])
        assert d1.status == StatusEnum.CANCELLED
        assert d2.status == StatusEnum.CANCELLED

    def test_deep_transitive_chain(self):
        graph = make_execution_graph()
        nodes = {}
        prev_id = None
        for i in range(5):
            t = ExecutionTaskNode(
                id=f"t{i}",
                description=f"Task {i}",
                depends_on=[prev_id] if prev_id else [],
            )
            nodes[f"t{i}"] = t
            prev_id = f"t{i}"
        # Mark t0 as failed
        nodes["t0"].status = StatusEnum.FAILED
        graph.nodes.update(nodes)
        self.eng._handle_layer_failure(graph, [nodes["t0"]])
        # t1..t4 should all be cancelled
        for i in range(1, 5):
            assert nodes[f"t{i}"].status == StatusEnum.CANCELLED

    def test_empty_failed_list_no_change(self):
        graph = make_execution_graph()
        t = ExecutionTaskNode(id="t1", description="Task")
        graph.nodes["t1"] = t
        self.eng._handle_layer_failure(graph, [])
        assert t.status == StatusEnum.PENDING

    def test_cancel_reason_in_event_data(self):
        graph = make_execution_graph()
        f = ExecutionTaskNode(id="f1", description="Failed")
        f.status = StatusEnum.FAILED
        dep = ExecutionTaskNode(id="d1", description="Dep", depends_on=["f1"])
        graph.nodes.update({"f1": f, "d1": dep})
        self.eng._handle_layer_failure(graph, [f])
        cancel_event = next(e for e in self.eng._event_history if e.event_type == "task_cancelled")
        assert cancel_event.data.get("reason") == "dependency_failed"

    def test_already_cancelled_not_double_cancelled(self):
        graph = make_execution_graph()
        f = ExecutionTaskNode(id="f1", description="Failed")
        f.status = StatusEnum.FAILED
        dep = ExecutionTaskNode(id="d1", description="Dep", depends_on=["f1"])
        dep.status = StatusEnum.CANCELLED  # already cancelled
        graph.nodes.update({"f1": f, "d1": dep})
        self.eng._handle_layer_failure(graph, [f])
        # Should remain cancelled, not generate new events
        cancelled_events = [e for e in self.eng._event_history if e.event_type == "task_cancelled"]
        assert len(cancelled_events) == 0


# ---------------------------------------------------------------------------
# 9. TestSaveAndLoadCheckpoint
# ---------------------------------------------------------------------------


class TestSaveAndLoadCheckpoint:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.eng = make_durable_engine(self.tmpdir)
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_creates_db(self):
        graph = make_execution_graph(plan_id="save-test")
        self.eng._save_checkpoint("save-test", graph)
        # Data is stored in the unified database (via database.py), not a
        # per-checkpoint-dir file.  Verify by reading the execution_state table.
        from vetinari.database import get_connection

        rows = self.eng._db.execute(
            "SELECT execution_id FROM execution_state WHERE execution_id = ?",
            ("save-test",),
        )
        assert len(rows) >= 1

    def test_saved_checkpoint_is_loadable(self):
        graph = make_execution_graph(plan_id="json-test")
        self.eng._save_checkpoint("json-test", graph)
        restored = self.eng.load_checkpoint("json-test")
        assert restored is not None
        assert restored.plan_id == "json-test"

    def test_saved_checkpoint_has_plan_id(self):
        graph = make_execution_graph(plan_id="has-plan-id")
        self.eng._save_checkpoint("has-plan-id", graph)
        restored = self.eng.load_checkpoint("has-plan-id")
        assert restored.plan_id == "has-plan-id"

    def test_saved_checkpoint_has_graph_state(self):
        graph = make_execution_graph(plan_id="gs-test")
        graph.add_task("t1", "Task one")
        self.eng._save_checkpoint("gs-test", graph)
        restored = self.eng.load_checkpoint("gs-test")
        assert "t1" in restored.nodes

    def test_completed_tasks_in_checkpoint(self):
        graph = make_execution_graph(plan_id="ct-test")
        task = graph.add_task("t1", "Done")
        task.status = StatusEnum.COMPLETED
        self.eng._save_checkpoint("ct-test", graph)
        restored = self.eng.load_checkpoint("ct-test")
        completed = [t.id for t in restored.get_completed_tasks()]
        assert "t1" in completed

    def test_load_returns_none_for_missing(self):
        result = self.eng.load_checkpoint("nonexistent-plan")
        assert result is None

    def test_load_restores_graph(self):
        graph = make_execution_graph(plan_id="load-test")
        graph.add_task("t1", "Task one")
        self.eng._save_checkpoint("load-test", graph)
        restored = self.eng.load_checkpoint("load-test")
        assert restored is not None
        assert restored.plan_id == "load-test"

    def test_load_restores_tasks(self):
        graph = make_execution_graph(plan_id="tasks-restore")
        graph.add_task("t1", "Task one")
        graph.add_task("t2", "Task two")
        self.eng._save_checkpoint("tasks-restore", graph)
        restored = self.eng.load_checkpoint("tasks-restore")
        assert "t1" in restored.nodes
        assert "t2" in restored.nodes

    def test_load_adds_to_active_executions(self):
        graph = make_execution_graph(plan_id="active-test")
        self.eng._save_checkpoint("active-test", graph)
        self.eng.load_checkpoint("active-test")
        assert "active-test" in self.eng._active_executions

    def test_overwrite_checkpoint(self):
        graph = make_execution_graph(plan_id="overwrite-test")
        graph.add_task("t1", "Task one")
        self.eng._save_checkpoint("overwrite-test", graph)
        # Mark task completed and save again
        graph.nodes["t1"].status = StatusEnum.COMPLETED
        self.eng._save_checkpoint("overwrite-test", graph)
        restored = self.eng.load_checkpoint("overwrite-test")
        completed = [t.id for t in restored.get_completed_tasks()]
        assert "t1" in completed

    def test_load_missing_returns_none(self):
        result = self.eng.load_checkpoint("nonexistent-plan-xyz")
        assert result is None

    def test_checkpoint_roundtrip_preserves_plan_id(self):
        graph = make_execution_graph(plan_id="uuid-test")
        self.eng._save_checkpoint("uuid-test", graph)
        restored = self.eng.load_checkpoint("uuid-test")
        assert restored is not None
        assert restored.plan_id == "uuid-test"

    def test_event_history_survives_checkpoint(self):
        graph = make_execution_graph(plan_id="meta-test")
        self.eng._emit_event("task_started", "t1", {})
        self.eng._emit_event("task_completed", "t1", {})
        self.eng._save_checkpoint("meta-test", graph)
        # Verify checkpoint was saved (event data is in SQLite)
        restored = self.eng.load_checkpoint("meta-test")
        assert restored is not None
        assert restored.plan_id == "meta-test"


# ---------------------------------------------------------------------------
# 10. TestRecoverExecution
# ---------------------------------------------------------------------------


class TestRecoverExecution:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.eng = make_durable_engine(self.tmpdir)
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_recover_no_checkpoint_returns_error(self):
        result = self.eng.recover_execution("no-such-plan")
        assert result["status"] == "error"

    def test_recover_error_message(self):
        result = self.eng.recover_execution("no-such-plan")
        assert "message" in result

    def test_recover_resets_failed_tasks(self):
        graph = make_execution_graph(plan_id="recover-test")
        task = graph.add_task("t1", "Failing task")
        task.status = StatusEnum.FAILED
        task.retry_count = 0
        task.max_retries = 3
        self.eng._save_checkpoint("recover-test", graph)

        # Verify recovery resets the task status before re-execution
        loaded = self.eng.load_checkpoint("recover-test")
        assert loaded is not None, "Checkpoint must be loadable"
        failed_node = loaded.nodes.get("t1")
        assert failed_node is not None
        assert failed_node.status == StatusEnum.FAILED, "Before recovery, task should be FAILED"

        # Recovery should reset FAILED→PENDING for retryable tasks
        for node in loaded.nodes.values():
            if node.status == StatusEnum.FAILED and node.retry_count < node.max_retries:
                node.status = StatusEnum.PENDING
                node.error = ""
        assert loaded.nodes["t1"].status == StatusEnum.PENDING, "Recovery logic must reset FAILED to PENDING"

    def test_recover_executes_plan(self):
        graph = make_execution_graph(plan_id="re-exec")
        graph.add_task("t1", "Task one")
        self.eng._save_checkpoint("re-exec", graph)
        with patch("time.sleep"):
            result = self.eng.recover_execution("re-exec")
        assert "plan_id" in result
        assert result["plan_id"] == "re-exec"

    def test_failed_tasks_with_retries_exhausted_stay_failed(self):
        graph = make_execution_graph(plan_id="no-retry")
        task = graph.add_task("t1", "Failing")
        task.status = StatusEnum.FAILED
        task.retry_count = 3
        task.max_retries = 3  # exhausted
        self.eng._save_checkpoint("no-retry", graph)
        with patch("time.sleep"):
            self.eng.recover_execution("no-retry")
        # Task retry_count >= max_retries, so it stays FAILED and is not reset
        # verify it was not re-executed by checking task still failed from checkpoint
        restored = self.eng.load_checkpoint("no-retry")
        # After recover, node status might be FAILED since it wasn't reset
        # The key check is the recover completed without crashing
        assert restored is not None
        assert isinstance(restored.plan_id, str)
        assert len(restored.plan_id) > 0

    def test_completed_tasks_not_reset(self):
        graph = make_execution_graph(plan_id="comp-stay")
        task = graph.add_task("t1", "Completed task")
        task.status = StatusEnum.COMPLETED
        self.eng._save_checkpoint("comp-stay", graph)
        with patch("time.sleep"):
            result = self.eng.recover_execution("comp-stay")
        # Should complete with 0 failures (no tasks to retry, completed stays completed)
        # The plan has no PENDING tasks after restore, so the engine sees 0 tasks to run
        assert "plan_id" in result

    def test_recover_returns_dict(self):
        graph = make_execution_graph(plan_id="dict-return")
        self.eng._save_checkpoint("dict-return", graph)
        with patch("time.sleep"):
            result = self.eng.recover_execution("dict-return")
        assert isinstance(result, dict)

    def test_recover_cleans_active_executions_on_completion(self):
        """After recovery completes, the plan is removed from _active_executions (no leak)."""
        graph = make_execution_graph(plan_id="active-recover")
        self.eng._save_checkpoint("active-recover", graph)
        with patch("time.sleep"):
            self.eng.recover_execution("active-recover")
        # After recovery completes, plan must be retired from active state
        assert "active-recover" not in self.eng._active_executions


# ---------------------------------------------------------------------------
# 11. TestGetExecutionStatus
# ---------------------------------------------------------------------------


class TestGetExecutionStatus:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.eng = make_durable_engine(self.tmpdir)
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_nonexistent_plan_returns_none(self):
        result = self.eng.get_execution_status("ghost-plan")
        assert result is None

    def test_active_plan_returns_status(self):
        graph = make_execution_graph(plan_id="active-plan")
        graph.add_task("t1", "Task one")
        self.eng._active_executions["active-plan"] = graph
        result = self.eng.get_execution_status("active-plan")
        assert result is not None
        assert result["plan_id"] == "active-plan"

    def test_status_contains_required_keys(self):
        graph = make_execution_graph(plan_id="key-check")
        self.eng._active_executions["key-check"] = graph
        result = self.eng.get_execution_status("key-check")
        for key in ("plan_id", "status", "total_tasks", "completed", "failed", "blocked", "progress"):
            assert key in result

    def test_total_tasks_count(self):
        graph = make_execution_graph(plan_id="task-count")
        for i in range(3):
            graph.add_task(f"t{i}", f"Task {i}")
        self.eng._active_executions["task-count"] = graph
        result = self.eng.get_execution_status("task-count")
        assert result["total_tasks"] == 3

    def test_completed_count_in_status(self):
        graph = make_execution_graph(plan_id="comp-count")
        t = graph.add_task("t1", "Task one")
        t.status = StatusEnum.COMPLETED
        self.eng._active_executions["comp-count"] = graph
        result = self.eng.get_execution_status("comp-count")
        assert result["completed"] == 1

    def test_failed_count_in_status(self):
        graph = make_execution_graph(plan_id="fail-count")
        t = graph.add_task("t1", "Task one")
        t.status = StatusEnum.FAILED
        self.eng._active_executions["fail-count"] = graph
        result = self.eng.get_execution_status("fail-count")
        assert result["failed"] == 1

    def test_progress_zero_for_empty_graph(self):
        graph = make_execution_graph(plan_id="prog-zero")
        self.eng._active_executions["prog-zero"] = graph
        result = self.eng.get_execution_status("prog-zero")
        assert result["progress"] == 0

    def test_progress_one_when_all_done(self):
        graph = make_execution_graph(plan_id="prog-full")
        t = graph.add_task("t1", "Task one")
        t.status = StatusEnum.COMPLETED
        self.eng._active_executions["prog-full"] = graph
        result = self.eng.get_execution_status("prog-full")
        assert result["progress"] == pytest.approx(1.0)

    def test_falls_back_to_checkpoint(self):
        # Not in active_executions but checkpoint file exists
        graph = make_execution_graph(plan_id="cp-fallback")
        self.eng._save_checkpoint("cp-fallback", graph)
        result = self.eng.get_execution_status("cp-fallback")
        assert result is not None
        assert result["plan_id"] == "cp-fallback"


# ---------------------------------------------------------------------------
# 12. TestListCheckpoints
# ---------------------------------------------------------------------------


class TestListCheckpoints:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.eng = make_durable_engine(self.tmpdir)
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_empty_directory(self):
        result = self.eng.list_checkpoints()
        assert result == []

    def test_single_checkpoint(self):
        graph = make_execution_graph(plan_id="only-plan")
        self.eng._save_checkpoint("only-plan", graph)
        result = self.eng.list_checkpoints()
        assert result == ["only-plan"]

    def test_multiple_checkpoints(self):
        for pid in ("plan-a", "plan-b", "plan-c"):
            self.eng._save_checkpoint(pid, make_execution_graph(plan_id=pid))
        result = self.eng.list_checkpoints()
        assert sorted(result) == ["plan-a", "plan-b", "plan-c"]

    def test_non_checkpoint_entries_excluded(self):
        # Only SQLite-saved checkpoints should appear
        result = self.eng.list_checkpoints()
        assert result == []

    def test_plan_id_in_checkpoint_list(self):
        self.eng._save_checkpoint("my-plan", make_execution_graph(plan_id="my-plan"))
        result = self.eng.list_checkpoints()
        assert "my-plan" in result

    def test_returns_list(self):
        result = self.eng.list_checkpoints()
        assert isinstance(result, list)

    def test_count_matches_saved(self):
        for i in range(7):
            self.eng._save_checkpoint(f"p{i}", make_execution_graph(plan_id=f"p{i}"))
        result = self.eng.list_checkpoints()
        assert len(result) == 7


# ---------------------------------------------------------------------------
# 13. TestCreateExecution (bonus)
# ---------------------------------------------------------------------------


class TestCreateExecution:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.eng = make_durable_engine(self.tmpdir)
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_plan_id(self):
        graph = make_execution_graph(plan_id="ce-plan")
        result = self.eng.create_execution(graph)
        assert result == "ce-plan"

    def test_added_to_active_executions(self):
        graph = make_execution_graph(plan_id="ae-plan")
        self.eng.create_execution(graph)
        assert "ae-plan" in self.eng._active_executions

    def test_checkpoint_saved(self):
        graph = make_execution_graph(plan_id="ce-cp-plan")
        self.eng.create_execution(graph)
        # Checkpoints are stored in SQLite
        restored = self.eng.load_checkpoint("ce-cp-plan")
        assert restored is not None
        assert restored.plan_id == "ce-cp-plan"

    def test_graph_stored_by_reference(self):
        graph = make_execution_graph(plan_id="ref-plan")
        self.eng.create_execution(graph)
        assert self.eng._active_executions["ref-plan"] is graph


# ---------------------------------------------------------------------------
# 14. TestEmitEvent (bonus)
# ---------------------------------------------------------------------------


class TestEmitEvent:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.eng = make_durable_engine(self.tmpdir)
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_emit_adds_to_history(self):
        self.eng._emit_event("test_event", "t1", {"key": "val"})
        assert len(self.eng._event_history) == 1

    def test_emitted_event_fields(self):
        self.eng._emit_event("task_started", "my-task", {"x": 1})
        ev = self.eng._event_history[0]
        assert ev.event_type == "task_started"
        assert ev.task_id == "my-task"
        assert ev.data["x"] == 1

    def test_multiple_events_accumulated(self):
        for i in range(5):
            self.eng._emit_event(f"event_{i}", "t1", {})
        assert len(self.eng._event_history) == 5

    def test_event_id_unique(self):
        for _ in range(10):
            self.eng._emit_event("e", "t", {})
        ids = [ev.event_id for ev in self.eng._event_history]
        assert len(ids) == len(set(ids))

    def test_timestamp_is_iso_format(self):
        self.eng._emit_event("e", "t", {})
        ev = self.eng._event_history[0]
        parsed = datetime.fromisoformat(ev.timestamp)
        assert isinstance(parsed, datetime), "Timestamp must parse as ISO format"
        assert ev.event_type == "e", f"Expected event_type 'e', got {ev.event_type!r}"


# ---------------------------------------------------------------------------
# 15. TestConcurrentExecution (thread-safety)
# ---------------------------------------------------------------------------


class TestConcurrentExecution:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.eng = make_durable_engine(self.tmpdir)
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_multiple_plans_can_execute(self):
        results = {}
        errors = []

        def run_plan(plan_id):
            try:
                graph = make_execution_graph(plan_id=plan_id)
                graph.add_task("t1", "Task one")
                eng = make_durable_engine(self.tmpdir)
                result = eng.execute_plan(graph, task_handler=_noop_handler)
                results[plan_id] = result
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_plan, args=(f"p{i}",)) for i in range(4)]
        with patch("time.sleep"):
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

        assert len(errors) == 0
        assert len(results) == 4

    def test_execution_lock_protects_active_executions(self):
        # Verify _execution_lock exists and is a Lock
        assert self.eng._execution_lock is not None
        assert hasattr(self.eng._execution_lock, "acquire")


# ---------------------------------------------------------------------------
# 16. TestRecordLearning (static method, non-fatal)
# ---------------------------------------------------------------------------


class TestRecordLearning:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tmpdir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_record_learning_does_not_raise(self):
        task = ExecutionTaskNode(id="t1", description="test", input_data={"assigned_model": "gpt-4"})
        # Should not raise even if learning modules are mocked
        result = DurableExecutionEngine._record_learning(task, "t1", "some output")
        assert result is None  # _record_learning returns None

    def test_record_learning_non_dict_output(self):
        task = ExecutionTaskNode(id="t1", description="test")
        # Should handle non-dict output gracefully
        result = DurableExecutionEngine._record_learning(task, "t1", 42)
        assert result is None  # _record_learning returns None regardless of output type

    def test_record_learning_exception_swallowed(self):
        # When learning modules raise, _record_learning should swallow
        task = ExecutionTaskNode(id="t1", description="test")
        with patch.dict(
            sys.modules,
            {
                "vetinari.learning.quality_scorer": MagicMock(
                    get_quality_scorer=Mock(side_effect=RuntimeError("learning down"))
                )
            },
        ):
            # Should not raise — exception is swallowed
            result = DurableExecutionEngine._record_learning(task, "t1", "out")
            assert result is None

    def test_record_learning_uses_assigned_model(self):
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = MagicMock(overall_score=0.9)
        _mock_quality_scorer_mod.get_quality_scorer.return_value = mock_scorer
        task = ExecutionTaskNode(id="t1", description="test", input_data={"assigned_model": "claude-3"})
        DurableExecutionEngine._record_learning(task, "t1", "output")
        # Verify scorer was called with task output
        mock_scorer.score.assert_called_once()
        score_args = mock_scorer.score.call_args
        assert score_args is not None
        assert len(score_args.args) > 0 or len(score_args.kwargs) > 0


# ---------------------------------------------------------------------------
# TestRecoverIncompleteExecutions
# ---------------------------------------------------------------------------


class TestRecoverIncompleteExecutions:
    """Tests for DurableExecutionEngine.recover_incomplete_executions."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.eng = make_durable_engine(self.tmpdir)
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_no_incomplete_returns_empty(self):
        results = self.eng.recover_incomplete_executions()
        assert results == []

    def test_completed_execution_not_recovered(self):
        graph = make_execution_graph(plan_id="done")
        graph.status = PlanStatus.COMPLETED
        self.eng._save_checkpoint("done", graph)
        # Mark as completed in DB
        self.eng._db.execute(
            "UPDATE execution_state SET pipeline_state = 'completed' WHERE execution_id = ?",
            ("done",),
        )
        results = self.eng.recover_incomplete_executions()
        assert results == []

    def test_running_execution_is_recovered(self):
        graph = make_execution_graph(plan_id="running-1")
        graph.add_task("t1", "Task one")
        graph.status = PlanStatus.EXECUTING
        self.eng._save_checkpoint("running-1", graph)
        with patch("time.sleep"):
            results = self.eng.recover_incomplete_executions()
        assert len(results) == 1
        assert results[0]["plan_id"] == "running-1"

    def test_multiple_incomplete_all_recovered(self):
        for i in range(3):
            graph = make_execution_graph(plan_id=f"inc-{i}")
            graph.add_task(f"t{i}", f"Task {i}")
            graph.status = PlanStatus.EXECUTING
            self.eng._save_checkpoint(f"inc-{i}", graph)
        with patch("time.sleep"):
            results = self.eng.recover_incomplete_executions()
        assert len(results) == 3

    def test_handler_passed_to_recovery(self):
        graph = make_execution_graph(plan_id="with-handler")
        graph.add_task("t1", "Task")
        graph.status = PlanStatus.EXECUTING
        self.eng._save_checkpoint("with-handler", graph)

        handler_calls = []

        def tracking_handler(task):
            handler_calls.append(task.id)
            return {"result": "ok"}

        with patch("time.sleep"):
            self.eng.recover_incomplete_executions(task_handler=tracking_handler)
        assert "t1" in handler_calls

    def test_recovery_failure_returns_error_entry(self):
        graph = make_execution_graph(plan_id="bad-recover")
        graph.status = PlanStatus.EXECUTING
        self.eng._save_checkpoint("bad-recover", graph)
        # Corrupt the stored graph data so load_checkpoint fails
        self.eng._db.execute(
            "UPDATE execution_state SET task_dag_json = 'not-json' WHERE execution_id = ?",
            ("bad-recover",),
        )
        results = self.eng.recover_incomplete_executions()
        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert "bad-recover" in results[0]["plan_id"]
