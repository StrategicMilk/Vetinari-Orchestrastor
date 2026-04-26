"""Performance benchmark tests for regression detection.

Each benchmark measures wall-clock time for a critical operation against a
threshold. All heavy external dependencies (LLM inference, network) are mocked
so runtimes reflect pure Python logic and I/O within the codebase.

Run with:
    python -m pytest tests/benchmarks/ -v --tb=short
"""

from __future__ import annotations

import sys
import time
import unittest.mock as mock
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.benchmarks.conftest import benchmark_timer
from tests.factories import make_task

# ── Module-level stubs for heavy optional deps ────────────────────────────────
# Installed before any vetinari sub-module imports so lazy imports resolve to
# these stubs rather than triggering real I/O at collection time.

_mock_qs = MagicMock()
_mock_qs.score.return_value = MagicMock(overall_score=0.85)
_mock_qs.get_history.return_value = []

sys.modules.setdefault(
    "vetinari.learning.quality_scorer",
    MagicMock(get_quality_scorer=lambda: _mock_qs),
)
sys.modules.setdefault(
    "vetinari.learning.feedback_loop",
    MagicMock(get_feedback_loop=lambda: MagicMock()),
)
sys.modules.setdefault(
    "vetinari.learning.model_selector",
    MagicMock(get_thompson_selector=lambda: MagicMock()),
)
sys.modules.setdefault(
    "vetinari.learning.workflow_learner",
    MagicMock(get_workflow_learner=lambda: MagicMock(get_recommendations=lambda g: {})),
)


# ── Benchmark 1: Plan generation latency ─────────────────────────────────────


@pytest.mark.benchmark
class TestPlanGenerationLatency:
    """Generating a plan must complete within 10 seconds for a 50-task workload."""

    def test_plan_generation_under_10s(self, tmp_path, monkeypatch):
        """PlanModeEngine.generate_plan() with mocked adapter finishes in < 10 s.

        The engine falls back to template-based candidate generation when the
        adapter returns an error response — this is the code path exercised here.
        Timing includes the full generate_plan() call including risk scoring
        and subtask creation.
        """
        monkeypatch.setenv("VETINARI_DB_PATH", str(tmp_path / "bench_plan.db"))
        from vetinari.database import reset_for_testing

        reset_for_testing()

        from vetinari.memory import get_memory_store
        from vetinari.planning.plan_mode import PlanModeEngine
        from vetinari.planning.plan_types import PlanGenerationRequest, TaskDomain

        # Mock the memory store so no real DB writes happen
        mock_memory = MagicMock()
        mock_memory.store.return_value = "mem-id-001"

        # Mock the adapter manager so no real inference happens; engine uses fallback
        mock_adapter = MagicMock()
        mock_response = MagicMock()
        mock_response.status = "error"
        mock_response.error = "no model loaded"
        mock_adapter.infer.return_value = mock_response

        with (
            patch("vetinari.planning.plan_mode.get_memory_store", return_value=mock_memory),
            patch("vetinari.planning.plan_mode.get_unified_memory_store", return_value=mock_memory),
            patch("vetinari.adapter_manager.get_adapter_manager", return_value=mock_adapter),
        ):
            engine = PlanModeEngine(memory_store=mock_memory)

            request = PlanGenerationRequest(
                goal="Build a production-grade REST API with authentication and data persistence",
                constraints="Must use Python, must have tests",
                plan_depth_cap=4,
                max_candidates=3,
                domain_hint=TaskDomain.CODING,
                dry_run=True,
            )

            with benchmark_timer("plan_generation") as t:
                plan = engine.generate_plan(request)

        assert t.elapsed_s < 10.0, (
            f"Plan generation took {t.elapsed_s:.3f}s — exceeded 10s threshold. "
            "This indicates O(n^2) or blocking I/O in the planning path."
        )
        assert plan is not None, "generate_plan returned None"
        assert len(plan.plan_id) > 0, "Generated plan has no plan_id"

        reset_for_testing()


# ── Benchmark 2: Checkpoint save + load cycle ─────────────────────────────────


@pytest.mark.benchmark
class TestCheckpointSaveLoadCycle:
    """A checkpoint save + load round-trip must complete within 1 second."""

    def test_checkpoint_round_trip_under_1s(self, tmp_path, monkeypatch):
        """DurableExecutionEngine save + load checkpoint completes in < 1 s.

        Tests the SQLite persistence path used after every task completion.
        A regression here indicates slow serialization or unintended full-table
        scans in the WAL path.
        """
        monkeypatch.setenv("VETINARI_DB_PATH", str(tmp_path / "bench_checkpoint.db"))
        from vetinari.database import reset_for_testing

        reset_for_testing()

        from vetinari.orchestration.durable_execution import DurableExecutionEngine
        from vetinari.orchestration.execution_graph import ExecutionGraph, ExecutionTaskNode
        from vetinari.types import StatusEnum

        engine = DurableExecutionEngine(
            checkpoint_dir=str(tmp_path / "cp"),
            max_concurrent=2,
            default_timeout=10.0,
        )
        plan_id = "bench-plan-001"
        graph = ExecutionGraph(plan_id=plan_id, goal="benchmark checkpoint save/load")

        # Add a realistic number of tasks to the graph
        for i in range(20):
            node = ExecutionTaskNode(
                id=f"task-{i:03d}",
                description=f"Benchmark task {i}",
                task_type="general",
            )
            node.status = StatusEnum.COMPLETED if i < 10 else StatusEnum.PENDING
            graph.nodes[node.id] = node

        with benchmark_timer("checkpoint_save_load") as t:
            engine._save_checkpoint(plan_id, graph)
            loaded = engine.load_checkpoint(plan_id)

        assert t.elapsed_s < 1.0, (
            f"Checkpoint round-trip took {t.elapsed_s:.3f}s — exceeded 1s threshold. "
            "Check for full-table scans or unbounded serialization in WAL path."
        )
        assert loaded is not None, "load_checkpoint returned None — save/load failed"
        assert loaded.plan_id == plan_id, f"Loaded plan_id {loaded.plan_id!r} does not match saved {plan_id!r}"
        assert len(loaded.nodes) == 20, f"Expected 20 nodes after round-trip, got {len(loaded.nodes)}"

        reset_for_testing()


# ── Benchmark 3: Task normalization throughput ────────────────────────────────


@pytest.mark.benchmark
class TestTaskNormalizationThroughput:
    """Normalizing 1 000 task dicts must complete within 1 second."""

    def test_normalize_1000_tasks_under_1s(self):
        """_normalize_task() on 1 000 raw task dicts completes in < 1 s.

        Task normalization runs per-task during project execution. A regression
        here scales directly with plan size, so it must stay O(1) per task.
        """
        from vetinari.web.projects_lifecycle import _normalize_task

        # Build raw task dicts that exercise all normalization branches
        raw_tasks = [
            {
                "subtask_id": f"subtask_{i:04d}",
                "description": f"Benchmark task {i}: process and transform data",
                "inputs": [f"input_{i}"],
                "outputs": [f"output_{i}"],
                "dependencies": [f"subtask_{i - 1:04d}"] if i > 0 else [],
                "domain": "coding",
            }
            for i in range(1000)
        ]
        goal_text = "Build a high-performance data pipeline"

        with benchmark_timer("task_normalization_1000") as t:
            normalized = [_normalize_task(task, idx, goal_text) for idx, task in enumerate(raw_tasks)]

        assert t.elapsed_s < 1.0, (
            f"Normalizing 1000 tasks took {t.elapsed_s:.3f}s — exceeded 1s threshold. "
            "Check for O(n) work inside the per-task normalization function."
        )
        assert len(normalized) == 1000, f"Expected 1000 normalized tasks, got {len(normalized)}"
        # Spot-check the first and last tasks for correctness
        first = normalized[0]
        assert first["id"] == "subtask_0000", f"Unexpected id: {first['id']!r}"
        assert "description" in first, "Normalized task missing 'description'"
        assert "inputs" in first, "Normalized task missing 'inputs'"
        assert "agent_type" in first, "Normalized task missing 'agent_type'"

        last = normalized[-1]
        assert last["id"] == "subtask_0999", f"Unexpected last id: {last['id']!r}"


# ── Benchmark 4: Event serialization throughput ────────────────────────────────


@pytest.mark.benchmark
class TestEventSerializationThroughput:
    """Creating and serializing 1 000 events must complete within 0.5 seconds."""

    def test_create_and_serialize_1000_events_under_0_5s(self):
        """TaskStarted / TaskCompleted event creation and JSON serialization is < 0.5 s.

        SSE pipelines create one event per task state transition. A regression
        here limits pipeline throughput.
        """
        import dataclasses
        import json
        import time as _time

        from vetinari.events import TaskCompleted, TaskStarted

        events: list[TaskStarted | TaskCompleted] = []
        ts = _time.time()

        with benchmark_timer("event_creation_1000") as t:
            for i in range(500):
                events.append(
                    TaskStarted(
                        event_type="TaskStarted",
                        timestamp=ts + i,
                        task_id=f"task-{i:04d}",
                        agent_type="worker",
                    )
                )
                events.append(
                    TaskCompleted(
                        event_type="TaskCompleted",
                        timestamp=ts + i + 0.5,
                        task_id=f"task-{i:04d}",
                        agent_type="worker",
                        success=True,
                        duration_ms=float(i * 10),
                    )
                )
            # Serialize all events to JSON (simulates SSE wire format)
            serialized = [json.dumps(dataclasses.asdict(e)) for e in events]

        assert t.elapsed_s < 0.5, (
            f"Creating + serializing 1000 events took {t.elapsed_s:.3f}s — "
            "exceeded 0.5s threshold. Check for per-event I/O or allocation."
        )
        assert len(serialized) == 1000, f"Expected 1000 serialized events, got {len(serialized)}"
        # Verify the serialized output is valid JSON with expected fields
        first_event = json.loads(serialized[0])
        assert "event_type" in first_event, "Serialized event missing 'event_type'"
        assert "task_id" in first_event, "Serialized event missing 'task_id'"
        assert "timestamp" in first_event, "Serialized event missing 'timestamp'"


class TestMemoryBenchmarks:
    """Verify peak memory stays within acceptable bounds."""

    @pytest.mark.benchmark
    def test_peak_heap_under_500mb(self) -> None:
        """Peak heap usage during a simulated workload stays under 500 MB.

        Creates 10,000 task objects and a plan to simulate a realistic
        working set. psutil tracks RSS before and after; delta must be
        under 500 MB.
        """
        import os

        import psutil

        from tests.factories import make_plan

        process = psutil.Process(os.getpid())
        rss_before = process.memory_info().rss

        # Simulate a realistic working set: many tasks + plans in memory
        tasks = [make_task(description=f"benchmark task {i}") for i in range(10_000)]
        plans = [make_plan(goal=f"plan-{i}", tasks=tasks[i * 10 : (i + 1) * 10]) for i in range(1000)]
        # Force materialization (prevent lazy evaluation)
        total_tasks = sum(len(p.tasks) for p in plans)

        rss_after = process.memory_info().rss
        delta_mb = (rss_after - rss_before) / (1024 * 1024)

        assert delta_mb < 500, f"Peak heap delta {delta_mb:.1f} MB exceeds 500 MB threshold"
        assert total_tasks == 10_000, f"Expected 10000 tasks, got {total_tasks}"
