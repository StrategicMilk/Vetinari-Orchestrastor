"""Plan execution visualization — DAG structures and real-time SSE state.

Provides DAG rendering, cost accumulation, quality gate indicators,
and SSE event infrastructure for plan execution monitoring.
Route handlers live in litestar_visualization.py.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field

from vetinari.constants import SSE_VISUALIZATION_QUEUE_SIZE
from vetinari.types import StatusEnum
from vetinari.utils.serialization import dataclass_to_dict

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class DAGNode:
    """A node in the plan execution DAG representing a single task."""

    task_id: str
    agent_type: str
    status: str = StatusEnum.PENDING.value
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    quality_score: float | None = None
    wave_id: str = ""
    description: str = ""

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"DAGNode(task_id={self.task_id!r}, agent_type={self.agent_type!r}, status={self.status!r})"

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return dataclass_to_dict(self)


@dataclass
class DAGEdge:
    """An edge in the plan execution DAG representing a dependency."""

    from_task: str
    to_task: str
    edge_type: str = "dependency"

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return dataclass_to_dict(self)


@dataclass
class DAGSummary:
    """Summary statistics for the plan execution DAG."""

    total_cost: float = 0.0
    elapsed_ms: float = 0.0
    tasks_complete: int = 0
    tasks_total: int = 0
    quality_gates_passed: int = 0
    quality_gates_failed: int = 0

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return (
            f"DAGSummary(tasks_complete={self.tasks_complete!r},"
            f" tasks_total={self.tasks_total!r}, total_cost={self.total_cost!r})"
        )

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return dataclass_to_dict(self)


@dataclass
class PlanDAG:
    """Complete DAG structure for plan visualization."""

    plan_id: str
    nodes: list[DAGNode] = field(default_factory=list)
    edges: list[DAGEdge] = field(default_factory=list)
    summary: DAGSummary = field(default_factory=DAGSummary)

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"PlanDAG(plan_id={self.plan_id!r}, nodes={len(self.nodes)!r}, edges={len(self.edges)!r})"

    def to_dict(self) -> dict:
        """Serialize to dictionary.

        Returns:
            Dictionary representation of the DAG.
        """
        return {
            "plan_id": self.plan_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "summary": self.summary.to_dict(),
        }


# ── Builder ──────────────────────────────────────────────────────────────────


class PlanVisualizationBuilder:
    """Converts plan/wave/task data into a DAG JSON structure for visualization.

    Supports both the planning module (Plan with Waves) and AgentGraph
    execution plans.
    """

    def build_from_plan(self, plan) -> PlanDAG:
        """Build a DAG from a planning.Plan object with waves and tasks.

        Args:
            plan: A vetinari.planning.planning.Plan instance.

        Returns:
            PlanDAG with nodes, edges, and summary.
        """
        dag = PlanDAG(plan_id=plan.plan_id)
        cost_total = 0.0
        tasks_complete = 0
        tasks_total = 0
        quality_gates_passed = 0
        quality_gates_failed = 0

        prev_wave_task_ids: list[str] = []

        for wave in plan.waves:
            current_wave_task_ids: list[str] = []

            for task in wave.tasks:
                tasks_total += 1
                is_complete = task.status == StatusEnum.COMPLETED.value
                if is_complete:
                    tasks_complete += 1

                # Extract cost from task metadata if available
                task_cost = getattr(task, "cost_usd", 0.0) or 0.0
                cost_total += task_cost

                quality_score = getattr(task, "quality_score", None)
                if quality_score is not None:
                    if quality_score >= 0.7:  # 70% threshold for quality gate pass
                        quality_gates_passed += 1
                    else:
                        quality_gates_failed += 1

                node = DAGNode(
                    task_id=task.task_id if hasattr(task, "task_id") else task.id,
                    agent_type=getattr(task, "agent_type", "unknown"),
                    status=task.status,
                    cost_usd=task_cost,
                    duration_ms=getattr(task, "duration_ms", 0.0) or 0.0,
                    quality_score=quality_score,
                    wave_id=wave.wave_id,
                    description=getattr(task, "description", ""),
                )
                dag.nodes.append(node)
                current_wave_task_ids.append(node.task_id)

                # Intra-task dependencies
                deps = getattr(task, "dependencies", []) or []
                for dep in deps:
                    dag.edges.append(
                        DAGEdge(
                            from_task=dep,
                            to_task=node.task_id,
                            edge_type="dependency",
                        ),
                    )

            # Inter-wave edges: previous wave tasks → current wave tasks
            if prev_wave_task_ids and current_wave_task_ids:
                for prev_id in prev_wave_task_ids:
                    for curr_id in current_wave_task_ids:
                        # Only add wave edges if no explicit dependency exists
                        has_explicit = any(e.from_task == prev_id and e.to_task == curr_id for e in dag.edges)
                        if not has_explicit:
                            dag.edges.append(
                                DAGEdge(
                                    from_task=prev_id,
                                    to_task=curr_id,
                                    edge_type="wave_order",
                                ),
                            )

            prev_wave_task_ids = current_wave_task_ids

        dag.summary = DAGSummary(
            total_cost=cost_total,
            elapsed_ms=0.0,
            tasks_complete=tasks_complete,
            tasks_total=tasks_total,
            quality_gates_passed=quality_gates_passed,
            quality_gates_failed=quality_gates_failed,
        )

        return dag

    def build_from_dict(self, plan_dict: dict) -> PlanDAG:
        """Build a DAG from a plan dictionary (e.g., from plan.to_dict()).

        Args:
            plan_dict: Dictionary representation of a plan.

        Returns:
            PlanDAG with nodes, edges, and summary.
        """
        plan_id = plan_dict.get("plan_id", "unknown")
        dag = PlanDAG(plan_id=plan_id)
        cost_total = 0.0
        tasks_complete = 0
        tasks_total = 0

        prev_wave_task_ids: list[str] = []

        for wave_data in plan_dict.get("waves", []):
            current_wave_task_ids: list[str] = []

            for task_data in wave_data.get("tasks", []):
                tasks_total += 1
                status = task_data.get("status", StatusEnum.PENDING.value)
                if status == StatusEnum.COMPLETED.value:
                    tasks_complete += 1

                task_cost = task_data.get("cost_usd", 0.0)
                cost_total += task_cost

                task_id = task_data.get("task_id", task_data.get("id", ""))
                node = DAGNode(
                    task_id=task_id,
                    agent_type=task_data.get("agent_type", "unknown"),
                    status=status,
                    cost_usd=task_cost,
                    duration_ms=task_data.get("duration_ms", 0.0),
                    quality_score=task_data.get("quality_score"),
                    wave_id=wave_data.get("wave_id", ""),
                    description=task_data.get("description", ""),
                )
                dag.nodes.append(node)
                current_wave_task_ids.append(task_id)

                for dep in task_data.get("dependencies", []):
                    dag.edges.append(
                        DAGEdge(
                            from_task=dep,
                            to_task=task_id,
                            edge_type="dependency",
                        ),
                    )

            if prev_wave_task_ids and current_wave_task_ids:
                for prev_id in prev_wave_task_ids:
                    for curr_id in current_wave_task_ids:
                        has_explicit = any(e.from_task == prev_id and e.to_task == curr_id for e in dag.edges)
                        if not has_explicit:
                            dag.edges.append(
                                DAGEdge(
                                    from_task=prev_id,
                                    to_task=curr_id,
                                    edge_type="wave_order",
                                ),
                            )

            prev_wave_task_ids = current_wave_task_ids

        dag.summary = DAGSummary(
            total_cost=cost_total,
            tasks_complete=tasks_complete,
            tasks_total=tasks_total,
        )

        return dag


# ── SSE event registry ───────────────────────────────────────────────────────

_viz_streams: dict[str, queue.Queue] = {}
_viz_streams_lock = threading.Lock()


def push_visualization_event(plan_id: str, event_type: str, data: dict) -> None:
    """Push a visualization SSE event for a plan.

    Called from agent execution code to emit real-time updates.

    Args:
        plan_id: The plan being executed.
        event_type: One of: task_started, task_completed, task_failed,
            quality_gate_result, cost_update.
        data: Structured event data as a dictionary.
    """
    with _viz_streams_lock:
        q = _viz_streams.get(plan_id)
    if q:
        try:
            q.put_nowait({"event": event_type, "data": json.dumps(data)})
        except queue.Full:
            logger.warning("Visualization SSE queue full for plan %s, dropping event", plan_id)


def _get_viz_queue(plan_id: str) -> queue.Queue:
    """Get or create an SSE queue for a plan's visualization stream.

    Args:
        plan_id: The plan identifier.

    Returns:
        The queue for SSE events.
    """
    with _viz_streams_lock:
        if plan_id not in _viz_streams:
            _viz_streams[plan_id] = queue.Queue(maxsize=SSE_VISUALIZATION_QUEUE_SIZE)
        return _viz_streams[plan_id]


def _remove_viz_queue(plan_id: str, q: queue.Queue | None = None) -> None:
    """Remove a visualization SSE queue when its stream ends."""
    with _viz_streams_lock:
        current = _viz_streams.get(plan_id)
        if q is None or current is q:
            _viz_streams.pop(plan_id, None)


# ── Gate approval state ──────────────────────────────────────────────────────

_pending_gates: dict[str, dict] = {}  # plan_id -> gate info
_pending_gates_lock = threading.Lock()


def register_quality_gate(plan_id: str, task_id: str, details: dict) -> None:
    """Register a quality gate checkpoint requiring human approval.

    Args:
        plan_id: The plan identifier.
        task_id: The task at the quality gate.
        details: Gate details (score, findings, etc.).
    """
    with _pending_gates_lock:
        _pending_gates[plan_id] = {
            "task_id": task_id,
            "details": details,
            "registered_at": time.time(),
            "status": StatusEnum.PENDING.value,
        }
    push_visualization_event(
        plan_id,
        "quality_gate_pending",
        {
            "task_id": task_id,
            "details": details,
        },
    )
