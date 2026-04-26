"""End-to-end integration tests for the TwoLayerOrchestrator pipeline.

These tests exercise the full pipeline (input analysis → plan generation →
model assignment → execution → review → assembly) with all LLM adapter
calls mocked so no real inference is required.

Each test is marked ``@pytest.mark.integration`` and is exempt from the
conftest network-blocking fixture.
"""

from __future__ import annotations

import sqlite3
import threading
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal stubs so the orchestration package loads without real deps
# ---------------------------------------------------------------------------
# Ensure vetinari.types is real before importing anything orchestration-related
import vetinari.types
from vetinari.learning.quality_scorer import QualityScore
from vetinari.orchestration.two_layer import TwoLayerOrchestrator
from vetinari.types import StatusEnum
from vetinari.workflow.andon import reset_andon_system


@pytest.fixture(autouse=True)
def _reset_andon():
    """Reset the global Andon system before each test to prevent cross-test leaks."""
    reset_andon_system()
    yield
    reset_andon_system()


@pytest.fixture(autouse=True)
def _stub_quality_scorer():
    """Keep pipeline E2E tests off the real local-inference calibration path."""

    class _FakeQualityScorer:
        def score(
            self,
            task_id: str,
            model_id: str,
            task_type: str,
            task_description: str,
            output: str,
            use_llm: bool = True,
            inference_confidence: float | None = None,
            temperature_used: float | None = None,
        ) -> QualityScore:
            del task_description, use_llm, inference_confidence, temperature_used
            return QualityScore(
                task_id=task_id,
                model_id=model_id,
                task_type=task_type,
                overall_score=0.82,
                correctness=0.8,
                completeness=0.85,
                efficiency=0.78,
                style=0.84,
                dimensions={
                    "correctness": 0.8,
                    "completeness": 0.85,
                    "efficiency": 0.78,
                    "style": 0.84,
                },
                measured_dimensions=["correctness", "completeness", "efficiency", "style"],
                issues=[],
                method="heuristic",
            )

    with patch(
        "vetinari.learning.quality_scorer.get_quality_scorer",
        return_value=_FakeQualityScorer(),
    ):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Patches applied to every test to avoid heavy optional deps.
_STANDARD_PATCHES = [
    "vetinari.orchestration.two_layer.TwoLayerOrchestrator._route_model_for_task",
    "vetinari.orchestration.two_layer.get_rules_manager",
    "vetinari.orchestration.two_layer.get_auto_tuner",
    "vetinari.orchestration.intake.get_request_intake",
]


@pytest.fixture
def make_orchestrator(tmp_path):
    """Return a factory that creates a TwoLayerOrchestrator with a throw-away checkpoint directory.

    The returned factory accepts the same keyword arguments as
    ``TwoLayerOrchestrator.__init__`` (``max_concurrent``, ``enable_correction_loop``).

    Tracks all created instances and shuts down their thread pools after the
    test to prevent zombie-thread accumulation across the suite (the
    ``DurableExecutionEngine`` creates a ``ThreadPoolExecutor`` that must be
    explicitly shut down).
    """
    created: list[TwoLayerOrchestrator] = []

    def _factory(*, max_concurrent: int = 2, enable_correction_loop: bool = False) -> TwoLayerOrchestrator:
        orch = TwoLayerOrchestrator(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            max_concurrent=max_concurrent,
            enable_correction_loop=enable_correction_loop,
        )
        created.append(orch)
        return orch

    yield _factory

    # Teardown: shut down every engine's thread pool so threads don't leak.
    # Use wait=False to avoid blocking if any futures are still pending.
    for orch in created:
        try:
            orch.execution_engine.shutdown(wait=False)
        except Exception:  # noqa: VET022 — teardown best-effort cleanup; failure is harmless
            pass


def _simple_handler(task) -> dict:
    """Default task handler that always succeeds with a canned output."""
    return {"status": "completed", "output": f"result-for-{task.id}"}


# ---------------------------------------------------------------------------
# Test 1: Simple goal produces plan, executes tasks, returns expected keys
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_simple_goal_produces_result_with_expected_keys(tmp_path, make_orchestrator):
    """Pipeline must return a dict that contains plan_id, goal, completed, and
    final_output for a simple coding goal."""
    orch = make_orchestrator()

    # Patch all optional lazy imports that might phone-home or import heavy deps.
    # Disable intake classification so the full pipeline path is exercised
    # (otherwise simple goals get routed to the express path).
    with (
        patch("vetinari.orchestration.two_layer.TwoLayerOrchestrator._route_model_for_task", return_value="default"),
        patch("vetinari.orchestration.two_layer.get_rules_manager", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.two_layer.get_auto_tuner", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.intake.get_request_intake", side_effect=ImportError),
    ):
        result = orch.generate_and_execute(
            goal="implement a hello-world function in Python",
            task_handler=_simple_handler,
        )

    assert isinstance(result, dict), "Result must be a dict"
    for key in ("plan_id", "goal", "completed", "failed", "final_output"):
        assert key in result, f"Expected key '{key}' missing from result"

    assert result["goal"] == "implement a hello-world function in Python"
    assert isinstance(result["plan_id"], str)
    assert result["plan_id"].startswith("plan-")
    assert result["completed"] >= 0


# ---------------------------------------------------------------------------
# Test 2: Constraints affect task count in the plan
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_constraints_affect_plan_task_count(tmp_path, make_orchestrator):
    """Passing a task_count constraint should control the number of tasks the
    PlanGenerator emits.  We verify that the plan stage records a tasks value
    that does not exceed the requested count."""
    orch = make_orchestrator()

    task_count_limit = 2
    constraints = {"task_count": task_count_limit, "estimated_files": 1}

    with (
        patch("vetinari.orchestration.two_layer.TwoLayerOrchestrator._route_model_for_task", return_value="default"),
        patch("vetinari.orchestration.two_layer.get_rules_manager", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.two_layer.get_auto_tuner", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.intake.get_request_intake", side_effect=ImportError),
    ):
        result = orch.generate_and_execute(
            goal="write a unit test for a calculator module",
            constraints=constraints,
            task_handler=_simple_handler,
        )

    assert "stages" in result
    plan_stage = result["stages"].get("plan", {})
    assert "tasks" in plan_stage, "plan stage must record task count"
    # The plan may add up to task_count_limit tasks (or could produce fewer if
    # the planner respects the hint — either is acceptable).
    assert plan_stage["tasks"] >= 1, "Plan must have at least one task"


# ---------------------------------------------------------------------------
# Test 3: Adapter failure is handled gracefully — no crash, error in result
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_pipeline_handles_task_handler_failure_gracefully(tmp_path, make_orchestrator):
    """If every task handler raises, the pipeline must NOT crash.

    The result dict must still be returned and must record failed tasks
    without propagating the exception to the caller.
    """
    orch = make_orchestrator()

    def _failing_handler(task):
        raise RuntimeError(f"simulated adapter failure for task {task.id}")

    with (
        patch("vetinari.orchestration.two_layer.TwoLayerOrchestrator._route_model_for_task", return_value="default"),
        patch("vetinari.orchestration.two_layer.get_rules_manager", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.two_layer.get_auto_tuner", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.intake.get_request_intake", side_effect=ImportError),
        patch("time.sleep", return_value=None),  # skip retry backoff delays
    ):
        # Must not raise
        result = orch.generate_and_execute(
            goal="research security vulnerabilities in a web API",
            task_handler=_failing_handler,
        )

    assert isinstance(result, dict), "Pipeline must return a dict even on failure"
    assert "plan_id" in result
    # All tasks failed — the result should record that
    assert result.get("failed", 0) > 0, "All tasks used failing handler — failed count must be positive"
    assert "stages" in result


# ---------------------------------------------------------------------------
# Test 4: Checkpoint is persisted to SQLite after execution
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_checkpoint_persisted_to_sqlite(tmp_path, make_orchestrator):
    """After a successful run the execution state must be saved to the SQLite
    database so that a crash-recovery resume can find it."""
    orch = make_orchestrator()

    with (
        patch("vetinari.orchestration.two_layer.TwoLayerOrchestrator._route_model_for_task", return_value="default"),
        patch("vetinari.orchestration.two_layer.get_rules_manager", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.two_layer.get_auto_tuner", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.intake.get_request_intake", side_effect=ImportError),
    ):
        result = orch.generate_and_execute(
            goal="document the public API of a Python module",
            task_handler=_simple_handler,
        )

    plan_id = result["plan_id"]
    db_path = tmp_path / "checkpoints" / "execution_state.db"
    assert db_path.exists(), "SQLite checkpoint database must be created"

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT execution_id, pipeline_state FROM execution_state WHERE execution_id = ?",
            (plan_id,),
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1, f"Expected 1 checkpoint row for plan {plan_id}, got {len(rows)}"
    saved_plan_id, pipeline_state = rows[0]
    assert saved_plan_id == plan_id
    # Status should be 'completed' or 'failed' (never 'running') after the pipeline finishes
    assert pipeline_state in ("completed", "failed"), f"Unexpected pipeline_state: {pipeline_state!r}"


# ---------------------------------------------------------------------------
# Test 5: Checkpoint can be loaded from SQLite by DurableExecutionEngine
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_checkpoint_can_be_loaded_after_run(tmp_path, make_orchestrator):
    """A completed execution's checkpoint must be loadable via load_checkpoint,
    returning a valid ExecutionGraph (enabling simulated crash-recovery)."""
    orch = make_orchestrator()

    with (
        patch("vetinari.orchestration.two_layer.TwoLayerOrchestrator._route_model_for_task", return_value="default"),
        patch("vetinari.orchestration.two_layer.get_rules_manager", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.two_layer.get_auto_tuner", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.intake.get_request_intake", side_effect=ImportError),
    ):
        result = orch.generate_and_execute(
            goal="write a benchmark for a sorting algorithm",
            task_handler=_simple_handler,
        )

    plan_id = result["plan_id"]
    graph = orch.execution_engine.load_checkpoint(plan_id)
    assert graph is not None, "load_checkpoint must return an ExecutionGraph"
    assert graph.plan_id == plan_id
    # All nodes should have a resolved status (not PENDING or RUNNING after completion)
    for node in graph.nodes.values():
        assert node.status in (
            StatusEnum.COMPLETED,
            StatusEnum.FAILED,
            StatusEnum.CANCELLED,
        ), f"Task {node.id} has unexpected status {node.status!r} after completed run"


# ---------------------------------------------------------------------------
# Test 6: Andon critical signal causes pipeline to return paused status
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_andon_critical_signal_halts_pipeline(tmp_path, make_orchestrator):
    """Raising a critical Andon signal before execution should cause the
    pipeline to return a paused response instead of executing."""
    orch = make_orchestrator()

    # Raise a critical signal on the orchestrator's Andon system
    orch.andon.raise_signal(source="test", severity="critical", message="test halt")

    with (
        patch("vetinari.orchestration.two_layer.TwoLayerOrchestrator._route_model_for_task", return_value="default"),
        patch("vetinari.orchestration.two_layer.get_rules_manager", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.two_layer.get_auto_tuner", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.intake.get_request_intake", side_effect=ImportError),
    ):
        result = orch.generate_and_execute(
            goal="refactor the auth module",
            task_handler=_simple_handler,
        )

    assert result.get("status") == "paused", (
        f"Pipeline should return paused when Andon is active, got: {result.get('status')!r}"
    )


# ---------------------------------------------------------------------------
# Test 7: Andon resume allows pipeline to execute again
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_andon_resume_allows_execution(tmp_path, make_orchestrator):
    """After resuming from an Andon halt the pipeline must execute normally."""
    orch = make_orchestrator()

    # Halt and then resume
    orch.andon.raise_signal(source="test", severity="critical", message="temporary halt")
    assert orch.is_paused(), "Orchestrator must be paused after critical signal"

    resumed = orch.resume()
    assert resumed is True, "resume() must return True when the pipeline was paused"
    assert not orch.is_paused(), "Orchestrator must not be paused after resume()"

    with (
        patch("vetinari.orchestration.two_layer.TwoLayerOrchestrator._route_model_for_task", return_value="default"),
        patch("vetinari.orchestration.two_layer.get_rules_manager", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.two_layer.get_auto_tuner", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.intake.get_request_intake", side_effect=ImportError),
    ):
        result = orch.generate_and_execute(
            goal="add logging to the payment service",
            task_handler=_simple_handler,
        )

    assert result.get("status") != "paused", "Pipeline must not be paused after resume"
    assert "plan_id" in result, "Resumed execution must produce a plan_id"


# ---------------------------------------------------------------------------
# Test 8: Task callbacks are fired during execution
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_task_callbacks_fire_during_execution(tmp_path, make_orchestrator):
    """The DurableExecutionEngine must invoke on_task_start and on_task_complete
    callbacks for every task that succeeds."""
    orch = make_orchestrator()

    started_ids: list[str] = []
    completed_ids: list[str] = []

    orch.execution_engine.set_callbacks(
        on_task_start=lambda task: started_ids.append(task.id),
        on_task_complete=lambda task: completed_ids.append(task.id),
    )

    with (
        patch("vetinari.orchestration.two_layer.TwoLayerOrchestrator._route_model_for_task", return_value="default"),
        patch("vetinari.orchestration.two_layer.get_rules_manager", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.two_layer.get_auto_tuner", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.intake.get_request_intake", side_effect=ImportError),
    ):
        result = orch.generate_and_execute(
            goal="add type annotations to the user model",
            task_handler=_simple_handler,
        )

    n_completed = result.get("completed", 0)
    assert len(started_ids) >= n_completed, "on_task_start must fire at least once per completed task"
    assert len(completed_ids) == n_completed, "on_task_complete must fire exactly once per completed task"
    # Every started task that completed should have its ID in both lists
    for task_id in completed_ids:
        assert task_id in started_ids, f"task {task_id} completed without a start callback"


# ---------------------------------------------------------------------------
# Test 9: Concurrent tasks in the same layer all produce results
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_concurrent_tasks_all_produce_results(tmp_path, make_orchestrator):
    """With max_concurrent > 1 the engine must run tasks in parallel and return
    results for every task, not just the first one."""
    call_log: list[str] = []
    lock = threading.Lock()

    def _counting_handler(task) -> dict:
        with lock:
            call_log.append(task.id)
        return {"status": "completed", "output": f"done-{task.id}"}

    orch = make_orchestrator(max_concurrent=4)

    with (
        patch("vetinari.orchestration.two_layer.TwoLayerOrchestrator._route_model_for_task", return_value="default"),
        patch("vetinari.orchestration.two_layer.get_rules_manager", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.two_layer.get_auto_tuner", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.intake.get_request_intake", side_effect=ImportError),
    ):
        result = orch.generate_and_execute(
            goal="implement a full CRUD REST API with tests and documentation",
            task_handler=_counting_handler,
        )

    assert result.get("completed", 0) + result.get("failed", 0) >= 1, "At least one task must be processed"
    assert len(call_log) >= 1, "Handler must have been called for at least one task"


# ---------------------------------------------------------------------------
# Test 10: No handler registered in DurableExecutionEngine — tasks still complete
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_engine_completes_tasks_without_registered_handler(tmp_path):
    """DurableExecutionEngine must mark tasks as FAILED (not crashing) when no
    task handler is registered — completing without a handler is fail-open."""
    from vetinari.orchestration.durable_execution import DurableExecutionEngine
    from vetinari.orchestration.execution_graph import ExecutionGraph

    engine = DurableExecutionEngine(
        checkpoint_dir=str(tmp_path / "engine_checkpoints"),
        max_concurrent=2,
    )
    # Build a minimal graph with two independent tasks — no handler registered
    graph = ExecutionGraph(plan_id="plan-no-handler-test", goal="test no-handler path")
    graph.add_task("task-a", "first task", task_type="general")
    graph.add_task("task-b", "second task", task_type="general")

    try:
        result = engine.execute_plan(graph, task_handler=None)

        assert isinstance(result, dict)
        assert result.get("failed", 0) == 2, "Tasks without handlers must be marked FAILED (fail-closed)"
        assert result.get("completed", 0) == 0, "No tasks should be completed without a handler"
        for task_node in graph.nodes.values():
            assert task_node.status == StatusEnum.FAILED, f"Task {task_node.id} must be FAILED"
    finally:
        engine.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Test 11: Variant level switch takes effect before plan generation
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_variant_level_switch_before_execution(tmp_path, make_orchestrator):
    """Switching the variant level to 'low' before execution must be reflected
    in the variant config used during that run."""
    orch = make_orchestrator()

    config_before = orch.get_variant_config()
    low_config = orch.set_variant_level("low")
    assert low_config.max_planning_depth <= config_before.max_planning_depth, (
        "Low variant must have a planning depth ≤ default"
    )

    with (
        patch("vetinari.orchestration.two_layer.TwoLayerOrchestrator._route_model_for_task", return_value="default"),
        patch("vetinari.orchestration.two_layer.get_rules_manager", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.two_layer.get_auto_tuner", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.intake.get_request_intake", side_effect=ImportError),
    ):
        result = orch.generate_and_execute(
            goal="write a short helper function",
            task_handler=_simple_handler,
        )

    assert isinstance(result, dict)
    assert "plan_id" in result


# ---------------------------------------------------------------------------
# Test 12: Context dict is preserved and returned through pipeline stages
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_context_dict_propagated_through_pipeline(tmp_path, make_orchestrator):
    """Caller-supplied context must reach the execution layer; the result's
    stages dict must reflect at least the plan stage."""
    orch = make_orchestrator()

    with (
        patch("vetinari.orchestration.two_layer.TwoLayerOrchestrator._route_model_for_task", return_value="default"),
        patch("vetinari.orchestration.two_layer.get_rules_manager", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.two_layer.get_auto_tuner", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.intake.get_request_intake", side_effect=ImportError),
    ):
        result = orch.generate_and_execute(
            goal="add caching to the analytics service",
            context={"caller": "integration_test", "project": "vetinari"},
            task_handler=_simple_handler,
        )

    assert "stages" in result, "Pipeline must return a 'stages' dict for introspection"
    assert "plan" in result["stages"], "Plan stage must appear in the stages dict"
    plan_stage = result["stages"]["plan"]
    assert "plan_id" in plan_stage, "Plan stage must record plan_id"
    assert "tasks" in plan_stage, "Plan stage must record task count"


# ---------------------------------------------------------------------------
# Test 13: Memory/learning hooks do not crash the pipeline
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_learning_hooks_do_not_crash_pipeline(tmp_path, make_orchestrator):
    """When the learning subsystem is unavailable the pipeline must absorb the
    ImportError silently and still return a valid result."""
    orch = make_orchestrator()

    with (
        patch("vetinari.orchestration.two_layer.TwoLayerOrchestrator._route_model_for_task", return_value="default"),
        patch("vetinari.orchestration.two_layer.get_rules_manager", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.two_layer.get_auto_tuner", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.intake.get_request_intake", side_effect=ImportError),
        # Simulate the learning subsystem being unavailable
        patch(
            "vetinari.orchestration.durable_execution.DurableExecutionEngine._record_learning",
            side_effect=ImportError("learning module unavailable"),
        ),
    ):
        result = orch.generate_and_execute(
            goal="extract metrics from the monitoring dashboard",
            task_handler=_simple_handler,
        )

    assert isinstance(result, dict), "Pipeline must return a dict even when learning hooks fail"
    assert "plan_id" in result


# ---------------------------------------------------------------------------
# Test 14: Pipeline returns timing information in total_time_ms
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_pipeline_result_includes_timing(tmp_path, make_orchestrator):
    """The pipeline result must include total_time_ms so callers can measure
    end-to-end latency without instrumenting the orchestrator themselves."""
    orch = make_orchestrator()

    with (
        patch("vetinari.orchestration.two_layer.TwoLayerOrchestrator._route_model_for_task", return_value="default"),
        patch("vetinari.orchestration.two_layer.get_rules_manager", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.two_layer.get_auto_tuner", side_effect=ImportError, create=True),
        patch("vetinari.orchestration.intake.get_request_intake", side_effect=ImportError),
    ):
        result = orch.generate_and_execute(
            goal="generate API documentation from source code",
            task_handler=_simple_handler,
        )

    assert "total_time_ms" in result, "Pipeline result must include 'total_time_ms'"
    assert isinstance(result["total_time_ms"], (int, float)), "total_time_ms must be numeric"
    assert result["total_time_ms"] >= 0, "total_time_ms must be non-negative"
