"""Tests for vetinari.analytics.value_stream and events.TimingEvent."""

from __future__ import annotations

import time

import pytest

from vetinari.analytics.value_stream import (
    AggregateReport,
    StationMetrics,
    ValueStreamAnalyzer,
    ValueStreamReport,
    get_value_stream_analyzer,
    reset_value_stream_analyzer,
)
from vetinari.events import TaskTimingRecord, TimingEvent

# ---------------------------------------------------------------------------
# TimingEvent enum
# ---------------------------------------------------------------------------


class TestTimingEvent:
    def test_has_all_six_values(self):
        values = {e.value for e in TimingEvent}
        assert values == {
            "task_queued",
            "task_dispatched",
            "task_completed",
            "task_rejected",
            "task_rework",
            "task_skipped",
        }

    def test_values_accessible_by_name(self):
        assert TimingEvent.TASK_QUEUED.value == "task_queued"
        assert TimingEvent.TASK_DISPATCHED.value == "task_dispatched"
        assert TimingEvent.TASK_COMPLETED.value == "task_completed"
        assert TimingEvent.TASK_REJECTED.value == "task_rejected"
        assert TimingEvent.TASK_REWORK.value == "task_rework"
        assert TimingEvent.TASK_SKIPPED.value == "task_skipped"


# ---------------------------------------------------------------------------
# TaskTimingRecord dataclass
# ---------------------------------------------------------------------------


class TestTaskTimingRecord:
    def test_creates_with_defaults(self):
        rec = TaskTimingRecord(event_type="", timestamp=0.0)
        assert rec.event_type == "TaskTimingRecord"  # post_init sets it
        assert rec.task_id == ""
        assert rec.execution_id == ""
        assert rec.agent_type == ""
        assert rec.timing_event == ""
        assert rec.metadata == {}

    def test_creates_with_values(self):
        rec = TaskTimingRecord(
            event_type="",
            timestamp=1234.0,
            task_id="t1",
            execution_id="e1",
            agent_type="builder",
            timing_event=TimingEvent.TASK_QUEUED.value,
            metadata={"queue_depth": 3},
        )
        assert rec.event_type == "TaskTimingRecord"
        assert rec.task_id == "t1"
        assert rec.execution_id == "e1"
        assert rec.agent_type == "builder"
        assert rec.timing_event == "task_queued"
        assert rec.metadata["queue_depth"] == 3

    def test_post_init_overrides_event_type(self):
        rec = TaskTimingRecord(event_type="something_else", timestamp=0.0)
        assert rec.event_type == "TaskTimingRecord"


# ---------------------------------------------------------------------------
# ValueStreamAnalyzer — record_event
# ---------------------------------------------------------------------------


class TestValueStreamAnalyzerRecordEvent:
    def setup_method(self):
        self.analyzer = ValueStreamAnalyzer()

    def test_record_event_stores_event(self):
        self.analyzer.record_event("exec1", "task1", "builder", "task_queued")
        with self.analyzer._lock:
            events = self.analyzer._events["exec1"]
        assert len(events) == 1
        assert events[0]["task_id"] == "task1"
        assert events[0]["agent_type"] == "builder"
        assert events[0]["timing_event"] == "task_queued"

    def test_record_event_stores_metadata(self):
        self.analyzer.record_event("exec1", "task1", "builder", "task_queued", metadata={"depth": 5})
        with self.analyzer._lock:
            events = self.analyzer._events["exec1"]
        assert events[0]["metadata"]["depth"] == 5

    def test_record_event_defaults_empty_metadata(self):
        self.analyzer.record_event("exec1", "task1", "builder", "task_queued")
        with self.analyzer._lock:
            events = self.analyzer._events["exec1"]
        assert events[0]["metadata"] == {}

    def test_multiple_executions_isolated(self):
        self.analyzer.record_event("execA", "t1", "planner", "task_queued")
        self.analyzer.record_event("execB", "t2", "builder", "task_queued")
        with self.analyzer._lock:
            assert len(self.analyzer._events["execA"]) == 1
            assert len(self.analyzer._events["execB"]) == 1


# ---------------------------------------------------------------------------
# ValueStreamAnalyzer — compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def setup_method(self):
        self.analyzer = ValueStreamAnalyzer()

    def _add_full_sequence(
        self,
        execution_id: str,
        task_id: str,
        agent_type: str,
        queue_delay: float = 0.02,
        proc_delay: float = 0.02,
    ) -> None:
        """Helper: inject queued → dispatched → completed with controlled timestamps."""
        now = time.time()
        with self.analyzer._lock:
            self.analyzer._events[execution_id].append({
                "task_id": task_id,
                "agent_type": agent_type,
                "timing_event": "task_queued",
                "timestamp": now,
                "metadata": {},
            })
            self.analyzer._events[execution_id].append({
                "task_id": task_id,
                "agent_type": agent_type,
                "timing_event": "task_dispatched",
                "timestamp": now + queue_delay,
                "metadata": {},
            })
            self.analyzer._events[execution_id].append({
                "task_id": task_id,
                "agent_type": agent_type,
                "timing_event": "task_completed",
                "timestamp": now + queue_delay + proc_delay,
                "metadata": {},
            })

    def test_empty_execution_returns_empty_report(self):
        report = self.analyzer.compute_metrics("nonexistent")
        assert report.execution_id == "nonexistent"
        assert report.total_lead_time_ms == 0.0
        assert report.per_station == {}

    def test_compute_queue_and_processing_time(self):
        self._add_full_sequence("e1", "t1", "builder", queue_delay=0.1, proc_delay=0.2)
        report = self.analyzer.compute_metrics("e1")

        assert "builder" in report.per_station
        station = report.per_station["builder"]
        assert station.tasks_processed == 1
        # queue_time_ms should be ~100ms (0.1 * 1000)
        assert 80.0 <= station.queue_time_ms <= 200.0
        # processing_time_ms should be ~200ms (0.2 * 1000)
        assert 150.0 <= station.processing_time_ms <= 400.0

    def test_compute_total_lead_time(self):
        self._add_full_sequence("e1", "t1", "builder", queue_delay=0.05, proc_delay=0.05)
        report = self.analyzer.compute_metrics("e1")
        # Lead time should be at least 100ms
        assert report.total_lead_time_ms >= 90.0

    def test_rework_events_counted(self):
        now = time.time()
        with self.analyzer._lock:
            self.analyzer._events["e1"].extend([
                {
                    "task_id": "t1",
                    "agent_type": "builder",
                    "timing_event": "task_queued",
                    "timestamp": now,
                    "metadata": {},
                },
                {
                    "task_id": "t1",
                    "agent_type": "builder",
                    "timing_event": "task_rework",
                    "timestamp": now + 0.01,
                    "metadata": {},
                },
                {
                    "task_id": "t1",
                    "agent_type": "builder",
                    "timing_event": "task_rework",
                    "timestamp": now + 0.02,
                    "metadata": {},
                },
            ])
        report = self.analyzer.compute_metrics("e1")
        assert report.per_station["builder"].rework_count == 2

    def test_rejected_events_counted_as_rework(self):
        now = time.time()
        with self.analyzer._lock:
            self.analyzer._events["e1"].extend([
                {
                    "task_id": "t1",
                    "agent_type": "quality",
                    "timing_event": "task_queued",
                    "timestamp": now,
                    "metadata": {},
                },
                {
                    "task_id": "t1",
                    "agent_type": "quality",
                    "timing_event": "task_rejected",
                    "timestamp": now + 0.01,
                    "metadata": {},
                },
            ])
        report = self.analyzer.compute_metrics("e1")
        assert report.per_station["quality"].rework_count == 1

    def test_skipped_events_recorded(self):
        now = time.time()
        with self.analyzer._lock:
            self.analyzer._events["e1"].extend([
                {
                    "task_id": "t1",
                    "agent_type": "researcher",
                    "timing_event": "task_skipped",
                    "timestamp": now,
                    "metadata": {},
                },
            ])
        report = self.analyzer.compute_metrics("e1")
        assert "researcher" in report.stations_skipped

    def test_value_add_ratio_bounded(self):
        self._add_full_sequence("e1", "t1", "builder", queue_delay=0.01, proc_delay=0.05)
        report = self.analyzer.compute_metrics("e1")
        assert 0.0 <= report.value_add_ratio <= 1.0

    def test_waste_time_includes_queue_and_rework(self):
        now = time.time()
        with self.analyzer._lock:
            self.analyzer._events["e1"].extend([
                {
                    "task_id": "t1",
                    "agent_type": "builder",
                    "timing_event": "task_queued",
                    "timestamp": now,
                    "metadata": {},
                },
                {
                    "task_id": "t1",
                    "agent_type": "builder",
                    "timing_event": "task_dispatched",
                    "timestamp": now + 0.1,
                    "metadata": {},
                },
                {
                    "task_id": "t1",
                    "agent_type": "builder",
                    "timing_event": "task_rework",
                    "timestamp": now + 0.15,
                    "metadata": {},
                },
                {
                    "task_id": "t1",
                    "agent_type": "builder",
                    "timing_event": "task_completed",
                    "timestamp": now + 0.2,
                    "metadata": {},
                },
            ])
        report = self.analyzer.compute_metrics("e1")
        # waste includes queue (~100ms) + rework estimate (1000ms per rework event)
        assert report.waste_time_ms > 0.0


# ---------------------------------------------------------------------------
# ValueStreamAnalyzer — get_aggregate_report
# ---------------------------------------------------------------------------


class TestAggregateReport:
    def setup_method(self):
        self.analyzer = ValueStreamAnalyzer()

    def _inject_execution(self, execution_id: str, agent_type: str = "builder") -> None:
        """Inject a minimal valid execution."""
        now = time.time()
        with self.analyzer._lock:
            self.analyzer._events[execution_id].extend([
                {
                    "task_id": "t1",
                    "agent_type": agent_type,
                    "timing_event": "task_queued",
                    "timestamp": now,
                    "metadata": {},
                },
                {
                    "task_id": "t1",
                    "agent_type": agent_type,
                    "timing_event": "task_dispatched",
                    "timestamp": now + 0.05,
                    "metadata": {},
                },
                {
                    "task_id": "t1",
                    "agent_type": agent_type,
                    "timing_event": "task_completed",
                    "timestamp": now + 0.1,
                    "metadata": {},
                },
            ])

    def test_empty_returns_default_report(self):
        report = self.analyzer.get_aggregate_report()
        assert report.total_executions == 0
        assert report.bottleneck_station == ""

    def test_aggregate_counts_executions(self):
        self._inject_execution("e1")
        self._inject_execution("e2")
        report = self.analyzer.get_aggregate_report(days=1)
        assert report.total_executions == 2

    def test_aggregate_avg_lead_time(self):
        self._inject_execution("e1")
        self._inject_execution("e2")
        report = self.analyzer.get_aggregate_report(days=1)
        assert report.avg_lead_time_ms > 0.0

    def test_aggregate_identifies_bottleneck(self):
        # Two stations: slow_agent has much more time than fast_agent
        now = time.time()
        with self.analyzer._lock:
            self.analyzer._events["e1"].extend([
                {
                    "task_id": "t1",
                    "agent_type": "slow_agent",
                    "timing_event": "task_queued",
                    "timestamp": now,
                    "metadata": {},
                },
                {
                    "task_id": "t1",
                    "agent_type": "slow_agent",
                    "timing_event": "task_dispatched",
                    "timestamp": now + 0.5,
                    "metadata": {},
                },
                {
                    "task_id": "t1",
                    "agent_type": "slow_agent",
                    "timing_event": "task_completed",
                    "timestamp": now + 1.0,
                    "metadata": {},
                },
                {
                    "task_id": "t2",
                    "agent_type": "fast_agent",
                    "timing_event": "task_queued",
                    "timestamp": now + 1.0,
                    "metadata": {},
                },
                {
                    "task_id": "t2",
                    "agent_type": "fast_agent",
                    "timing_event": "task_dispatched",
                    "timestamp": now + 1.01,
                    "metadata": {},
                },
                {
                    "task_id": "t2",
                    "agent_type": "fast_agent",
                    "timing_event": "task_completed",
                    "timestamp": now + 1.02,
                    "metadata": {},
                },
            ])
        report = self.analyzer.get_aggregate_report(days=1)
        assert report.bottleneck_station == "slow_agent"

    def test_aggregate_respects_days_cutoff(self):
        # Inject old event by manipulating timestamps
        old_time = time.time() - (10 * 86400)  # 10 days ago
        with self.analyzer._lock:
            self.analyzer._events["old_exec"].extend([
                {
                    "task_id": "t1",
                    "agent_type": "builder",
                    "timing_event": "task_queued",
                    "timestamp": old_time,
                    "metadata": {},
                },
                {
                    "task_id": "t1",
                    "agent_type": "builder",
                    "timing_event": "task_completed",
                    "timestamp": old_time + 1.0,
                    "metadata": {},
                },
            ])
        report = self.analyzer.get_aggregate_report(days=7)
        assert report.total_executions == 0

    def test_days_parameter_passed_through(self):
        report = self.analyzer.get_aggregate_report(days=30)
        assert report.days == 30


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_station_metrics_to_dict(self):
        sm = StationMetrics(
            agent_type="builder",
            queue_time_ms=12.345,
            processing_time_ms=98.765,
            rework_count=2,
            tasks_processed=5,
        )
        d = sm.to_dict()
        assert d["agent_type"] == "builder"
        assert d["queue_time_ms"] == 12.35
        assert d["processing_time_ms"] == 98.77
        assert d["rework_count"] == 2
        assert d["tasks_processed"] == 5

    def test_value_stream_report_to_dict(self):
        report = ValueStreamReport(
            execution_id="e1",
            total_lead_time_ms=500.0,
            per_station={"builder": StationMetrics(agent_type="builder")},
            value_add_ratio=0.75,
            waste_time_ms=125.0,
            stations_skipped=["researcher"],
        )
        d = report.to_dict()
        assert d["execution_id"] == "e1"
        assert d["total_lead_time_ms"] == 500.0
        assert "builder" in d["per_station"]
        assert d["value_add_ratio"] == 0.75
        assert d["waste_time_ms"] == 125.0
        assert d["stations_skipped"] == ["researcher"]

    def test_aggregate_report_to_dict(self):
        report = AggregateReport(
            days=7,
            avg_lead_time_ms=300.0,
            bottleneck_station="builder",
            value_add_ratio=0.6,
            avg_waste_pct=0.4,
            total_executions=10,
            avg_rework_rate=0.5,
        )
        d = report.to_dict()
        assert d["days"] == 7
        assert d["avg_lead_time_ms"] == 300.0
        assert d["bottleneck_station"] == "builder"
        assert d["value_add_ratio"] == 0.6
        assert d["avg_waste_pct"] == 0.4
        assert d["total_executions"] == 10
        assert d["avg_rework_rate"] == 0.5


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def setup_method(self):
        reset_value_stream_analyzer()

    def teardown_method(self):
        reset_value_stream_analyzer()

    def test_get_value_stream_analyzer_returns_instance(self):
        analyzer = get_value_stream_analyzer()
        assert isinstance(analyzer, ValueStreamAnalyzer)

    def test_get_value_stream_analyzer_is_singleton(self):
        a1 = get_value_stream_analyzer()
        a2 = get_value_stream_analyzer()
        assert a1 is a2

    def test_reset_creates_fresh_instance(self):
        a1 = get_value_stream_analyzer()
        reset_value_stream_analyzer()
        a2 = get_value_stream_analyzer()
        assert a1 is not a2


# ---------------------------------------------------------------------------
# Import test
# ---------------------------------------------------------------------------


def test_import_value_stream_analyzer():
    from vetinari.analytics.value_stream import ValueStreamAnalyzer as VSA

    assert VSA is ValueStreamAnalyzer
