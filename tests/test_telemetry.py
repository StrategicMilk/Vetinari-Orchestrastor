"""
Unit tests for the telemetry module.

Tests metrics collection, aggregation, and export functionality.
"""

import json
import os
import tempfile

import pytest

from vetinari.telemetry import (
    AdapterMetrics,
    MemoryMetrics,
    PlanMetrics,
    get_telemetry_collector,
    reset_telemetry,
)


class TestAdapterMetrics:
    """Test adapter metrics."""

    def test_adapter_metrics_initialization(self):
        """Test AdapterMetrics initialization."""
        metrics = AdapterMetrics(provider="openai", model="gpt-4")
        assert metrics.provider == "openai"
        assert metrics.model == "gpt-4"
        assert metrics.total_requests == 0
        assert metrics.success_rate == 0.0
        assert metrics.avg_latency_ms == 0.0

    def test_adapter_metrics_success_rate(self):
        """Test success rate calculation."""
        metrics = AdapterMetrics(provider="openai", model="gpt-4")
        metrics.total_requests = 10
        metrics.successful_requests = 7
        metrics.failed_requests = 3

        assert metrics.success_rate == pytest.approx(70.0)

    def test_adapter_metrics_avg_latency(self):
        """Test average latency calculation."""
        metrics = AdapterMetrics(provider="openai", model="gpt-4")
        metrics.successful_requests = 2
        metrics.total_latency_ms = 300.0  # 150 + 150

        assert metrics.avg_latency_ms == pytest.approx(150.0)


class TestMemoryMetrics:
    """Test memory metrics."""

    def test_memory_metrics_initialization(self):
        """Test MemoryMetrics initialization."""
        metrics = MemoryMetrics(backend="oc")
        assert metrics.backend == "oc"
        assert metrics.total_writes == 0
        assert metrics.dedup_hit_rate == 0.0

    def test_dedup_hit_rate(self):
        """Test dedup hit rate calculation."""
        metrics = MemoryMetrics(backend="oc")
        metrics.dedup_hits = 75
        metrics.dedup_misses = 25

        assert metrics.dedup_hit_rate == pytest.approx(75.0)

    def test_average_latencies(self):
        """Test average latency calculations."""
        metrics = MemoryMetrics(backend="oc")
        metrics.write_latency_ms = [5.0, 6.0, 4.0]
        metrics.read_latency_ms = [2.0, 2.5, 1.5]
        metrics.search_latency_ms = [20.0, 25.0, 15.0]

        assert metrics.avg_write_latency() == pytest.approx(5.0)
        assert metrics.avg_read_latency() == pytest.approx(2.0)
        assert metrics.avg_search_latency() == pytest.approx(20.0)


class TestPlanMetrics:
    """Test plan metrics."""

    def test_plan_metrics_initialization(self):
        """Test PlanMetrics initialization."""
        metrics = PlanMetrics()
        assert metrics.total_decisions == 0
        assert metrics.approval_rate == 0.0
        assert metrics.average_risk_score == 0.0

    def test_approval_rate(self):
        """Test approval rate calculation."""
        metrics = PlanMetrics()
        metrics.total_decisions = 10
        metrics.approved_decisions = 8
        metrics.rejected_decisions = 2

        assert metrics.approval_rate == pytest.approx(80.0)

    def test_average_risk_score_update(self):
        """Test average risk score calculation."""
        metrics = PlanMetrics()
        metrics.risk_scores = [0.2, 0.5, 0.3]
        metrics.update_average_risk_score()

        assert metrics.average_risk_score == pytest.approx(0.33, abs=0.005)

    def test_average_approval_time_update(self):
        """Test average approval time calculation."""
        metrics = PlanMetrics()
        metrics.approval_times_ms = [100.0, 150.0, 200.0]
        metrics.update_average_approval_time()

        assert metrics.average_approval_time_ms == pytest.approx(150.0)


class TestTelemetryCollector:
    """Test TelemetryCollector."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        reset_telemetry()
        self.collector = get_telemetry_collector()

    def test_singleton_instance(self):
        """Test singleton pattern."""
        collector1 = get_telemetry_collector()
        collector2 = get_telemetry_collector()
        assert collector1 is collector2

    def test_record_adapter_latency(self):
        """Test recording adapter latency."""
        self.collector.record_adapter_latency("openai", "gpt-4", 150.5, success=True, tokens_used=100)

        metrics = self.collector.get_adapter_metrics("openai")
        assert "openai:gpt-4" in metrics
        assert metrics["openai:gpt-4"].total_requests == 1
        assert metrics["openai:gpt-4"].successful_requests == 1

    def test_record_memory_write(self):
        """Test recording memory write."""
        self.collector.record_memory_write("oc", 5.2)
        self.collector.record_memory_write("oc", 4.8)

        metrics = self.collector.get_memory_metrics("oc")
        assert metrics["oc"].total_writes == 2

    def test_record_plan_decision(self):
        """Test recording plan decision."""
        self.collector.record_plan_decision("approve", risk_score=0.3, auto_approved=True)

        metrics = self.collector.get_plan_metrics()
        assert metrics.total_decisions == 1
        assert metrics.approved_decisions == 1
        assert metrics.auto_approved_decisions == 1

    def test_export_json(self):
        """Test JSON export."""
        self.collector.record_adapter_latency("openai", "gpt-4", 150.5)
        self.collector.record_plan_decision("approve", risk_score=0.3)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "telemetry.json")
            success = self.collector.export_json(export_path)

            assert success
            with open(export_path) as f:
                data = json.load(f)
                assert "timestamp" in data
                assert "adapters" in data
                assert "plan_mode" in data

    def test_export_prometheus(self):
        """Test Prometheus export."""
        self.collector.record_adapter_latency("openai", "gpt-4", 150.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "metrics.txt")
            success = self.collector.export_prometheus(export_path)

            assert success
            with open(export_path) as f:
                content = f.read()
                assert "vetinari_adapter" in content

    def test_reset(self):
        """Test telemetry reset."""
        self.collector.record_adapter_latency("openai", "gpt-4", 150.5)
        self.collector.record_plan_decision("approve", risk_score=0.3)

        self.collector.reset()

        metrics = self.collector.get_plan_metrics()
        assert metrics.total_decisions == 0
        assert len(self.collector.adapter_metrics) == 0


class TestTelemetryThreadSafety:
    """Test thread safety of telemetry collector."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        reset_telemetry()
        self.collector = get_telemetry_collector()

    def test_concurrent_recording(self):
        """Test concurrent metric recording."""
        import threading

        def record_metrics(thread_id):
            for _i in range(10):
                self.collector.record_adapter_latency("openai", "gpt-4", 100.0 + thread_id, success=True)
                self.collector.record_plan_decision("approve", risk_score=0.5)

        threads = [threading.Thread(target=record_metrics, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        metrics = self.collector.get_plan_metrics()
        assert metrics.total_decisions == 50  # 5 threads * 10 decisions
