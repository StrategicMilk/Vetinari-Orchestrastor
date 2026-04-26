"""Value Stream Mapping — Lean visualization of pipeline flow.

Instruments the pipeline to produce value stream maps showing where time
is spent, where work waits, and where rework happens. The factory equivalent
of timing each station with a stopwatch.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StationMetrics:
    """Metrics for a single pipeline station (agent).

    Args:
        agent_type: The agent type.
        queue_time_ms: Average time tasks spent waiting in queue.
        processing_time_ms: Average time tasks spent being processed.
        rework_count: Number of rework events at this station.
        tasks_processed: Total tasks that passed through.
    """

    agent_type: str = ""
    queue_time_ms: float = 0.0
    processing_time_ms: float = 0.0
    rework_count: int = 0
    tasks_processed: int = 0

    def __repr__(self) -> str:
        return (
            f"StationMetrics(agent_type={self.agent_type!r}, "
            f"queue_time_ms={self.queue_time_ms!r}, processing_time_ms={self.processing_time_ms!r}, "
            f"tasks_processed={self.tasks_processed!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "agent_type": self.agent_type,
            "queue_time_ms": round(self.queue_time_ms, 2),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "rework_count": self.rework_count,
            "tasks_processed": self.tasks_processed,
        }


@dataclass
class ValueStreamReport:
    """Value stream metrics for a single execution.

    Args:
        execution_id: The execution this report covers.
        total_lead_time_ms: Total wall-clock time from first event to last.
        per_station: Metrics broken down by pipeline station.
        value_add_ratio: Processing time / total lead time (lean metric).
        waste_time_ms: Queue time + rework time.
        stations_skipped: List of stations that were skipped.
    """

    execution_id: str = ""
    total_lead_time_ms: float = 0.0
    per_station: dict[str, StationMetrics] = field(default_factory=dict)
    value_add_ratio: float = 0.0
    waste_time_ms: float = 0.0
    stations_skipped: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"ValueStreamReport(execution_id={self.execution_id!r}, "
            f"total_lead_time_ms={self.total_lead_time_ms!r}, "
            f"value_add_ratio={self.value_add_ratio!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "execution_id": self.execution_id,
            "total_lead_time_ms": round(self.total_lead_time_ms, 2),
            "per_station": {k: v.to_dict() for k, v in self.per_station.items()},
            "value_add_ratio": round(self.value_add_ratio, 4),
            "waste_time_ms": round(self.waste_time_ms, 2),
            "stations_skipped": self.stations_skipped,
        }


@dataclass
class AggregateReport:
    """Weekly aggregate value stream metrics.

    Args:
        days: Number of days covered.
        avg_lead_time_ms: Average lead time across executions.
        bottleneck_station: Station with highest queue + processing time.
        value_add_ratio: Average value-add ratio.
        avg_waste_pct: Average waste percentage.
        total_executions: Number of executions in the period.
        avg_rework_rate: Average rework events per execution.
    """

    days: int = 7
    avg_lead_time_ms: float = 0.0
    bottleneck_station: str = ""
    value_add_ratio: float = 0.0
    avg_waste_pct: float = 0.0
    total_executions: int = 0
    avg_rework_rate: float = 0.0

    def __repr__(self) -> str:
        return (
            f"AggregateReport(days={self.days!r}, avg_lead_time_ms={self.avg_lead_time_ms!r}, "
            f"bottleneck_station={self.bottleneck_station!r}, "
            f"total_executions={self.total_executions!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "days": self.days,
            "avg_lead_time_ms": round(self.avg_lead_time_ms, 2),
            "bottleneck_station": self.bottleneck_station,
            "value_add_ratio": round(self.value_add_ratio, 4),
            "avg_waste_pct": round(self.avg_waste_pct, 4),
            "total_executions": self.total_executions,
            "avg_rework_rate": round(self.avg_rework_rate, 4),
        }


class ValueStreamAnalyzer:
    """Computes value stream metrics from timing events.

    Collects TaskTimingRecord events and produces per-execution and
    aggregate value stream reports.
    """

    def __init__(self) -> None:
        self._events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()

    def record_event(
        self,
        execution_id: str,
        task_id: str,
        agent_type: str,
        timing_event: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a timing event for value stream analysis.

        Args:
            execution_id: The execution this event belongs to.
            task_id: The task identifier.
            agent_type: The agent type.
            timing_event: The timing event type (TimingEvent value).
            metadata: Additional context.
        """
        with self._lock:
            self._events[execution_id].append({
                "task_id": task_id,
                "agent_type": agent_type,
                "timing_event": timing_event,
                "timestamp": time.time(),
                "metadata": metadata or {},  # noqa: VET112 - empty fallback preserves optional request metadata contract
            })

    def compute_metrics(self, execution_id: str) -> ValueStreamReport:
        """Compute value stream metrics for a single execution.

        Args:
            execution_id: The execution to analyze.

        Returns:
            ValueStreamReport with per-station metrics.
        """
        with self._lock:
            events = list(self._events.get(execution_id, []))

        if not events:
            return ValueStreamReport(execution_id=execution_id)

        # Sort by timestamp
        events.sort(key=lambda e: e["timestamp"])

        total_lead_time_ms = (events[-1]["timestamp"] - events[0]["timestamp"]) * 1000.0

        # Group events by agent
        per_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for evt in events:
            per_agent[evt["agent_type"]].append(evt)

        stations: dict[str, StationMetrics] = {}
        total_processing = 0.0
        total_queue = 0.0
        total_rework_ms = 0.0
        total_rework = 0
        skipped: set[str] = set()

        for agent_type, agent_events in per_agent.items():
            queue_times: list[float] = []
            proc_times: list[float] = []
            rework_count = 0
            tasks_processed = 0

            # Match queued → dispatched → completed sequences.
            # Keep earliest queued/dispatched time per task to avoid overwrite
            # when the same task is re-queued or re-dispatched.
            queued_at: dict[str, float] = {}
            dispatched_at: dict[str, float] = {}
            rework_start: dict[str, float] = {}

            for evt in agent_events:
                tid = evt["task_id"]
                ts = evt["timestamp"]
                te = evt["timing_event"]

                if te == "task_queued":
                    if tid not in queued_at:  # preserve earliest queued time
                        queued_at[tid] = ts
                elif te == "task_dispatched":
                    if tid not in dispatched_at:  # preserve earliest dispatched time
                        dispatched_at[tid] = ts
                    if tid in queued_at:
                        queue_times.append((ts - queued_at[tid]) * 1000.0)
                elif te == "task_completed":
                    tasks_processed += 1
                    if tid in dispatched_at:
                        proc_times.append((ts - dispatched_at[tid]) * 1000.0)
                    # Close any open rework window
                    if tid in rework_start:
                        total_rework_ms += (ts - rework_start.pop(tid)) * 1000.0
                elif te in ("task_rejected", "task_rework"):
                    rework_count += 1
                    rework_start[tid] = ts  # open rework window for actual duration
                elif te == "task_skipped":
                    skipped.add(agent_type)

            avg_queue = sum(queue_times) / len(queue_times) if queue_times else 0.0
            avg_proc = sum(proc_times) / len(proc_times) if proc_times else 0.0

            stations[agent_type] = StationMetrics(
                agent_type=agent_type,
                queue_time_ms=avg_queue,
                processing_time_ms=avg_proc,
                rework_count=rework_count,
                tasks_processed=tasks_processed,
            )

            total_processing += sum(proc_times)
            total_queue += sum(queue_times)
            total_rework += rework_count

        value_add_ratio = total_processing / total_lead_time_ms if total_lead_time_ms > 0 else 0.0
        waste_time_ms = total_queue + total_rework_ms

        return ValueStreamReport(
            execution_id=execution_id,
            total_lead_time_ms=total_lead_time_ms,
            per_station=stations,
            value_add_ratio=min(1.0, value_add_ratio),
            waste_time_ms=waste_time_ms,
            stations_skipped=list(skipped),
        )

    def get_aggregate_report(self, days: int = 7) -> AggregateReport:
        """Produce an aggregate report across all tracked executions.

        Args:
            days: Number of days to include (from now backwards).

        Returns:
            AggregateReport with weekly aggregate metrics.
        """
        cutoff = time.time() - (days * 86400)

        with self._lock:
            execution_ids = list(self._events.keys())

        if not execution_ids:
            return AggregateReport(days=days)

        # Snapshot which execution IDs fall within the cutoff window before
        # calling compute_metrics (which also acquires _lock) to avoid deadlock.
        in_window: list[str] = []
        for eid in execution_ids:
            with self._lock:
                events = self._events.get(eid, [])
                # Use last-event timestamp so executions that started before the
                # cutoff but completed within the window are included correctly.
                if events and events[-1]["timestamp"] >= cutoff:
                    in_window.append(eid)

        reports: list[ValueStreamReport] = [self.compute_metrics(eid) for eid in in_window]

        if not reports:
            return AggregateReport(days=days)

        avg_lead = sum(r.total_lead_time_ms for r in reports) / len(reports)
        avg_var = sum(r.value_add_ratio for r in reports) / len(reports)
        total_waste = sum(r.waste_time_ms for r in reports)
        total_lead = sum(r.total_lead_time_ms for r in reports)
        avg_waste_pct = total_waste / total_lead if total_lead > 0 else 0.0
        total_rework = sum(sum(s.rework_count for s in r.per_station.values()) for r in reports)
        avg_rework_rate = total_rework / len(reports)

        # Find bottleneck: station with highest combined queue + processing
        station_totals: dict[str, float] = defaultdict(float)
        for r in reports:
            for agent, station in r.per_station.items():
                station_totals[agent] += station.queue_time_ms + station.processing_time_ms
        bottleneck = max(station_totals, key=station_totals.__getitem__) if station_totals else ""

        return AggregateReport(
            days=days,
            avg_lead_time_ms=avg_lead,
            bottleneck_station=bottleneck,
            value_add_ratio=avg_var,
            avg_waste_pct=avg_waste_pct,
            total_executions=len(reports),
            avg_rework_rate=avg_rework_rate,
        )

    def clear(self) -> None:
        """Clear all recorded events."""
        with self._lock:
            self._events.clear()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_value_stream_analyzer: ValueStreamAnalyzer | None = None
_vs_lock = threading.Lock()


def get_value_stream_analyzer() -> ValueStreamAnalyzer:
    """Return the singleton ValueStreamAnalyzer instance.

    Returns:
        The shared ValueStreamAnalyzer instance.
    """
    global _value_stream_analyzer
    if _value_stream_analyzer is None:
        with _vs_lock:
            if _value_stream_analyzer is None:
                _value_stream_analyzer = ValueStreamAnalyzer()
    return _value_stream_analyzer


def reset_value_stream_analyzer() -> None:
    """Reset the singleton for testing."""
    global _value_stream_analyzer
    with _vs_lock:
        _value_stream_analyzer = None
