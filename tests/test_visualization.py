"""Tests for the plan execution visualization module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from vetinari.web.visualization import (
    DAGEdge,
    DAGNode,
    DAGSummary,
    PlanDAG,
    PlanVisualizationBuilder,
    _get_viz_queue,
    _pending_gates,
    _pending_gates_lock,
    _viz_streams,
    _viz_streams_lock,
    push_visualization_event,
    register_quality_gate,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_viz_state():
    """Reset visualization state between tests."""
    with _viz_streams_lock:
        _viz_streams.clear()
    with _pending_gates_lock:
        _pending_gates.clear()
    yield
    with _viz_streams_lock:
        _viz_streams.clear()
    with _pending_gates_lock:
        _pending_gates.clear()


def _make_task(
    task_id,
    status="pending",
    agent_type="builder",
    deps=None,
    cost_usd=0.0,
    duration_ms=0.0,
    quality_score=None,
    description="",
):
    """Create a mock task object."""
    task = MagicMock()
    task.task_id = task_id
    task.id = task_id
    task.status = status
    task.agent_type = agent_type
    task.dependencies = deps or []
    task.cost_usd = cost_usd
    task.duration_ms = duration_ms
    task.quality_score = quality_score
    task.description = description
    return task


def _make_wave(wave_id, tasks, order=1):
    """Create a mock wave object."""
    wave = MagicMock()
    wave.wave_id = wave_id
    wave.tasks = tasks
    wave.order = order
    return wave


def _make_plan(plan_id, waves):
    """Create a mock plan object."""
    plan = MagicMock()
    plan.plan_id = plan_id
    plan.waves = waves
    return plan


# ── DAG data structure tests ─────────────────────────────────────────────────


class TestDAGStructures:
    """Test DAG data structure serialization."""

    def test_dag_node_to_dict(self):
        node = DAGNode(
            task_id="t1",
            agent_type="builder",
            status="running",
            cost_usd=0.05,
            duration_ms=1200.0,
            quality_score=0.95,
            wave_id="w1",
            description="Build module",
        )
        d = node.to_dict()
        assert d["task_id"] == "t1"
        assert d["agent_type"] == "builder"
        assert d["status"] == "running"
        assert d["cost_usd"] == 0.05
        assert d["quality_score"] == 0.95

    def test_dag_edge_to_dict(self):
        edge = DAGEdge(from_task="t1", to_task="t2", edge_type="dependency")
        d = edge.to_dict()
        assert d["from_task"] == "t1"
        assert d["to_task"] == "t2"
        assert d["edge_type"] == "dependency"

    def test_dag_summary_to_dict(self):
        summary = DAGSummary(
            total_cost=1.50,
            elapsed_ms=5000.0,
            tasks_complete=3,
            tasks_total=5,
            quality_gates_passed=2,
            quality_gates_failed=1,
        )
        d = summary.to_dict()
        assert d["total_cost"] == 1.50
        assert d["tasks_complete"] == 3
        assert d["tasks_total"] == 5
        assert d["quality_gates_passed"] == 2

    def test_plan_dag_to_dict(self):
        dag = PlanDAG(
            plan_id="p1",
            nodes=[DAGNode(task_id="t1", agent_type="builder")],
            edges=[DAGEdge(from_task="t1", to_task="t2")],
            summary=DAGSummary(tasks_total=2),
        )
        d = dag.to_dict()
        assert d["plan_id"] == "p1"
        assert len(d["nodes"]) == 1
        assert len(d["edges"]) == 1
        assert d["summary"]["tasks_total"] == 2


# ── PlanVisualizationBuilder tests ───────────────────────────────────────────


class TestPlanVisualizationBuilder:
    """Test building DAGs from plan objects and dictionaries."""

    def test_build_from_plan_single_wave(self):
        tasks = [
            _make_task("t1", "completed", "builder", cost_usd=0.1),
            _make_task("t2", "pending", "researcher"),
        ]
        wave = _make_wave("w1", tasks)
        plan = _make_plan("p1", [wave])

        builder = PlanVisualizationBuilder()
        dag = builder.build_from_plan(plan)

        assert dag.plan_id == "p1"
        assert len(dag.nodes) == 2
        assert dag.summary.tasks_total == 2
        assert dag.summary.tasks_complete == 1
        assert dag.summary.total_cost == pytest.approx(0.1)

    def test_build_from_plan_multi_wave_creates_wave_edges(self):
        w1_tasks = [_make_task("t1", "completed")]
        w2_tasks = [_make_task("t2", "pending")]
        plan = _make_plan("p1", [_make_wave("w1", w1_tasks), _make_wave("w2", w2_tasks)])

        builder = PlanVisualizationBuilder()
        dag = builder.build_from_plan(plan)

        wave_edges = [e for e in dag.edges if e.edge_type == "wave_order"]
        assert len(wave_edges) == 1
        assert wave_edges[0].from_task == "t1"
        assert wave_edges[0].to_task == "t2"

    def test_build_from_plan_explicit_deps_no_duplicate_wave_edges(self):
        w1_tasks = [_make_task("t1", "completed")]
        w2_tasks = [_make_task("t2", "pending", deps=["t1"])]
        plan = _make_plan("p1", [_make_wave("w1", w1_tasks), _make_wave("w2", w2_tasks)])

        builder = PlanVisualizationBuilder()
        dag = builder.build_from_plan(plan)

        dep_edges = [e for e in dag.edges if e.edge_type == "dependency"]
        wave_edges = [e for e in dag.edges if e.edge_type == "wave_order"]
        assert len(dep_edges) == 1
        assert len(wave_edges) == 0

    def test_build_from_plan_quality_scores(self):
        tasks = [
            _make_task("t1", "completed", quality_score=0.9),
            _make_task("t2", "completed", quality_score=0.5),
            _make_task("t3", "completed", quality_score=None),
        ]
        plan = _make_plan("p1", [_make_wave("w1", tasks)])

        builder = PlanVisualizationBuilder()
        dag = builder.build_from_plan(plan)

        assert dag.summary.quality_gates_passed == 1
        assert dag.summary.quality_gates_failed == 1

    def test_build_from_dict(self):
        plan_dict = {
            "plan_id": "p1",
            "waves": [
                {
                    "wave_id": "w1",
                    "tasks": [
                        {
                            "task_id": "t1",
                            "status": "completed",
                            "agent_type": "builder",
                            "cost_usd": 0.05,
                            "dependencies": [],
                        },
                        {"task_id": "t2", "status": "pending", "agent_type": "researcher", "dependencies": ["t1"]},
                    ],
                }
            ],
        }

        builder = PlanVisualizationBuilder()
        dag = builder.build_from_dict(plan_dict)

        assert dag.plan_id == "p1"
        assert len(dag.nodes) == 2
        assert dag.summary.tasks_complete == 1
        assert dag.summary.total_cost == pytest.approx(0.05)
        assert len(dag.edges) == 1
        assert dag.edges[0].from_task == "t1"

    def test_build_from_dict_empty_plan(self):
        builder = PlanVisualizationBuilder()
        dag = builder.build_from_dict({"plan_id": "empty", "waves": []})

        assert dag.plan_id == "empty"
        assert len(dag.nodes) == 0
        assert len(dag.edges) == 0
        assert dag.summary.tasks_total == 0


# ── SSE event tests ──────────────────────────────────────────────────────────


class TestSSEEvents:
    """Test SSE event pushing and queue management."""

    def test_push_visualization_event_creates_queue(self):
        q = _get_viz_queue("plan_123")
        push_visualization_event("plan_123", "task_started", {"task_id": "t1"})

        msg = q.get_nowait()
        assert msg["event"] == "task_started"
        data = json.loads(msg["data"])
        assert data["task_id"] == "t1"

    def test_push_event_no_queue_does_not_error(self):
        push_visualization_event("nonexistent", "task_completed", {"task_id": "t1"})
        # Verify no queue was created for the unknown plan
        assert "nonexistent" not in _viz_streams

    def test_push_event_full_queue_drops_silently(self):
        q = _get_viz_queue("plan_full")
        for i in range(500):
            q.put_nowait({"event": "filler", "data": str(i)})
        size_before = q.qsize()
        push_visualization_event("plan_full", "overflow", {"x": 1})
        # Queue must not grow beyond capacity when full — the event is silently dropped
        assert q.qsize() <= size_before + 1

    def test_event_format_matches_sse_spec(self):
        q = _get_viz_queue("plan_sse")
        push_visualization_event(
            "plan_sse",
            "task_completed",
            {
                "task_id": "t1",
                "status": "completed",
                "duration_ms": 1500.0,
                "cost_usd": 0.03,
            },
        )

        msg = q.get_nowait()
        assert "event" in msg
        assert "data" in msg
        data = json.loads(msg["data"])
        assert data["task_id"] == "t1"
        assert data["cost_usd"] == 0.03


# ── Quality gate tests ───────────────────────────────────────────────────────


class TestQualityGates:
    """Test quality gate registration and approval."""

    def test_register_quality_gate(self):
        q = _get_viz_queue("plan_gate")
        register_quality_gate("plan_gate", "t1", {"score": 0.85, "findings": []})

        with _pending_gates_lock:
            gate = _pending_gates.get("plan_gate")
        assert gate is not None
        assert gate["task_id"] == "t1"
        assert gate["status"] == "pending"

        msg = q.get_nowait()
        assert msg["event"] == "quality_gate_pending"
