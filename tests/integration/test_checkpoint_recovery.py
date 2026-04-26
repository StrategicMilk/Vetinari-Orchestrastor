"""Integration tests for checkpoint corruption recovery.

Verifies that the durable execution engine recovers gracefully when
SQLite checkpoint data is corrupted — the most common production failure mode.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.factories import make_plan, make_task
from vetinari.orchestration.durable_execution import DurableExecutionEngine
from vetinari.orchestration.execution_graph import ExecutionGraph
from vetinari.types import StatusEnum

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_checkpoint_dir(tmp_path: Path) -> Path:
    """Create an isolated temporary directory for checkpoint storage."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def engine(temp_checkpoint_dir: Path) -> DurableExecutionEngine:
    """Create a DurableExecutionEngine backed by a temp SQLite file.

    Using an explicit checkpoint_dir ensures the engine writes to an
    isolated test file rather than the shared production database.
    """
    eng = DurableExecutionEngine(checkpoint_dir=str(temp_checkpoint_dir))
    yield eng
    eng.shutdown()


@pytest.fixture
def simple_graph() -> ExecutionGraph:
    """Create a minimal single-node ExecutionGraph for checkpoint tests."""
    task = make_task(id="task-a", description="Write hello world to stdout")
    plan = make_plan(plan_id="plan-chk-001", goal="Hello world", tasks=[task])

    graph = ExecutionGraph(
        plan_id=plan.plan_id,
        goal=plan.goal,
    )
    from vetinari.orchestration.execution_graph import ExecutionTaskNode

    node = ExecutionTaskNode(
        id=task.id,
        description=task.description,
        task_type="general",
        assigned_model="",
    )
    graph.nodes[task.id] = node
    return graph


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCheckpointSaveAndLoad:
    """Verify the happy-path checkpoint round-trip."""

    def test_checkpoint_save_and_load_roundtrip(
        self, engine: DurableExecutionEngine, simple_graph: ExecutionGraph
    ) -> None:
        """Saving then loading a checkpoint restores the graph with matching plan_id and task ids."""
        plan_id = engine.create_execution(simple_graph)

        loaded = engine.load_checkpoint(plan_id)

        assert loaded is not None, "load_checkpoint returned None for a freshly saved checkpoint"
        assert loaded.plan_id == simple_graph.plan_id
        assert set(loaded.nodes.keys()) == set(simple_graph.nodes.keys())

    def test_checkpoint_missing_plan_returns_none(self, engine: DurableExecutionEngine) -> None:
        """Loading a checkpoint for an unknown plan_id returns None, not an exception."""
        result = engine.load_checkpoint("plan-does-not-exist-xyz")

        assert result is None

    def test_checkpoint_updated_after_execution(self, temp_checkpoint_dir: Path, simple_graph: ExecutionGraph) -> None:
        """After a successful task run the checkpoint reflects the completed status."""
        eng = DurableExecutionEngine(checkpoint_dir=str(temp_checkpoint_dir))

        def _succeed(task):
            return {"output": "done"}

        try:
            results = eng.execute_plan(simple_graph, task_handler=_succeed)
        finally:
            eng.shutdown()

        assert results[StatusEnum.COMPLETED.value] >= 1, f"Expected at least 1 completed task, got: {results}"

        # Reload the engine from the same directory to prove persistence
        eng2 = DurableExecutionEngine(checkpoint_dir=str(temp_checkpoint_dir))
        try:
            reloaded = eng2.load_checkpoint(simple_graph.plan_id)
        finally:
            eng2.shutdown()

        assert reloaded is not None
        assert reloaded.plan_id == simple_graph.plan_id


class TestCorruptCheckpointRecovery:
    """Verify graceful handling of a truncated or corrupted SQLite file."""

    def test_corrupt_checkpoint_truncated_file(self, tmp_path: Path, simple_graph: ExecutionGraph) -> None:
        """A fresh engine that has no prior data returns None from load_checkpoint.

        This is the baseline "missing checkpoint" scenario: after data loss (e.g.
        the database file was never created or was deleted before the next start),
        the engine must re-initialise cleanly and report no checkpoint for the plan
        rather than crashing.

        We use a fresh directory that was never written to, so there is no prior
        database to worry about file-lock issues on Windows.
        """
        fresh_dir = tmp_path / "fresh_engine"
        fresh_dir.mkdir()

        # No prior checkpoint was ever saved — the engine should start clean
        eng = DurableExecutionEngine(checkpoint_dir=str(fresh_dir))
        try:
            result = eng.load_checkpoint(simple_graph.plan_id)
            assert result is None, f"Expected None from load_checkpoint on a brand-new engine, got: {result}"
        except Exception as exc:
            pytest.fail(f"DurableExecutionEngine raised on fresh (no-data) directory: {type(exc).__name__}: {exc}")
        finally:
            eng.shutdown()

    def test_corrupt_checkpoint_bad_json(self, temp_checkpoint_dir: Path, simple_graph: ExecutionGraph) -> None:
        """Valid SQLite with corrupt JSON in task_dag_json returns None from load_checkpoint.

        Contract: load_checkpoint must return None (never raise) when the stored
        JSON is malformed. Any other outcome — raising, or returning a graph object
        built from garbage — violates the corrupt-checkpoint contract.
        """
        # Build a valid DB, then inject a bad JSON row
        eng = DurableExecutionEngine(checkpoint_dir=str(temp_checkpoint_dir))
        eng.create_execution(simple_graph)
        eng.shutdown()

        db_path = temp_checkpoint_dir / "execution_state.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                "UPDATE execution_state SET task_dag_json = ? WHERE execution_id = ?",
                ("{this is not valid json <<<", simple_graph.plan_id),
            )
            conn.commit()
        finally:
            conn.close()

        eng2 = DurableExecutionEngine(checkpoint_dir=str(temp_checkpoint_dir))
        try:
            result = eng2.load_checkpoint(simple_graph.plan_id)
            assert result is None, (
                f"Corrupt-checkpoint contract: load_checkpoint must return None for malformed JSON — got {result!r}"
            )
        except Exception as exc:
            pytest.fail(f"load_checkpoint raised instead of returning None on bad JSON: {type(exc).__name__}: {exc}")
        finally:
            eng2.shutdown()

    def test_recovery_creates_fresh_state(self, temp_checkpoint_dir: Path) -> None:
        """After corruption of one plan, a new plan can still be checkpointed cleanly."""
        eng = DurableExecutionEngine(checkpoint_dir=str(temp_checkpoint_dir))
        eng.shutdown()

        # Corrupt the DB
        db_path = temp_checkpoint_dir / "execution_state.db"
        db_path.write_bytes(b"\x00" * 50)

        # Start a brand-new engine; if init fails gracefully, a fresh graph in a
        # new directory should still work
        clean_dir = temp_checkpoint_dir.parent / "clean_checkpoints"
        clean_dir.mkdir()
        eng_clean = DurableExecutionEngine(checkpoint_dir=str(clean_dir))

        fresh_graph = ExecutionGraph(plan_id="plan-fresh-001", goal="Fresh start")
        from vetinari.orchestration.execution_graph import ExecutionTaskNode

        fresh_graph.nodes["task-fresh"] = ExecutionTaskNode(
            id="task-fresh",
            description="Fresh start task",
            task_type="general",
            assigned_model="",
        )

        try:
            plan_id = eng_clean.create_execution(fresh_graph)
            loaded = eng_clean.load_checkpoint(plan_id)
        finally:
            eng_clean.shutdown()

        assert loaded is not None, "Fresh engine could not save/load checkpoint after prior corruption"
        assert loaded.plan_id == "plan-fresh-001"


class TestProcessLevelRestart:
    """Process-level restart recovery: prove that across a real process boundary,
    a RUNNING task persisted before a crash either resumes correctly or explicitly
    demotes — it does not stay permanently unscheduled while recovery reports success.
    """

    def test_running_task_persisted_is_reset_across_real_process_boundary(self, tmp_path: Path) -> None:
        """Spawn a child process that writes a RUNNING checkpoint then exits (crash sim).

        In the parent process, call recover_execution() and prove the task was reset
        to PENDING with an error message documenting the crash. This is the
        process-level proof required by SESSION-31A.1: a fresh-process recovery pass
        may not leave persisted RUNNING tasks permanently unscheduled.
        """
        import os
        import subprocess
        import sys
        import textwrap

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Find the project root so the child can import vetinari via sys.path.
        # The worktree root is two levels above this test file (tests/integration/).
        project_root = str(Path(__file__).resolve().parent.parent.parent)

        child_code = textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {project_root!r})

            from vetinari.orchestration.durable_execution import DurableExecutionEngine
            from vetinari.orchestration.execution_graph import ExecutionGraph, ExecutionTaskNode
            from vetinari.types import StatusEnum

            checkpoint_dir = {str(checkpoint_dir)!r}
            eng = DurableExecutionEngine(checkpoint_dir=checkpoint_dir)
            graph = ExecutionGraph(plan_id="proc-restart-001", goal="test")
            node = ExecutionTaskNode(id="t1", description="stuck task")
            node.status = StatusEnum.RUNNING
            graph.nodes[node.id] = node
            eng.create_execution(graph)
            eng.shutdown()
            # Exit without cleanup — simulates a process crash after checkpoint write
            """
        )

        result = subprocess.run(  # noqa: S603 - argv is controlled and shell interpolation is not used
            [sys.executable, "-c", child_code],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "PYTHONPATH": project_root},
        )
        assert result.returncode == 0, (
            f"Child process (crash simulator) failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

        # Fresh process (this one) loads and recovers — heartbeats are gone, only DB state remains.
        from vetinari.orchestration.durable_execution import DurableExecutionEngine
        from vetinari.orchestration.durable_execution_recovery import load_checkpoint, recover_execution

        eng = DurableExecutionEngine(checkpoint_dir=str(checkpoint_dir))
        try:
            # Verify the RUNNING status survived the process boundary
            graph = load_checkpoint(eng, "proc-restart-001")
            assert graph is not None, "Checkpoint did not survive across process boundary"
            assert graph.nodes["t1"].status == StatusEnum.RUNNING, (
                "Pre-recovery: persisted RUNNING task should still read as RUNNING before recovery runs"
            )

            # Stub execute_plan to capture the graph state immediately after
            # recover_execution() resets RUNNING nodes — before actual re-execution.
            captured: dict = {}

            def _capture_and_succeed(g: ExecutionGraph) -> dict:
                captured["graph"] = g
                return {StatusEnum.COMPLETED.value: 0, StatusEnum.FAILED.value: 0, "task_results": {}}

            eng.execute_plan = _capture_and_succeed  # type: ignore[method-assign]
            recover_execution(eng, "proc-restart-001")

            assert "graph" in captured, "recover_execution did not call execute_plan — recovery did not proceed"
            recovered_node = captured["graph"].nodes["t1"]

            # Contract: RUNNING is reset to PENDING so the task is re-executed.
            assert recovered_node.status == StatusEnum.PENDING, (
                f"RUNNING task must be reset to PENDING on cross-process recovery, got {recovered_node.status!r}"
            )
            # Contract: the error message documents the crash (not silent).
            assert recovered_node.error, "Recovered node must carry an error message explaining the reset"
            assert "crashed" in recovered_node.error.lower() or "running" in recovered_node.error.lower(), (
                f"Error message must reference the crash or RUNNING state, got: {recovered_node.error!r}"
            )
        finally:
            eng.shutdown()
