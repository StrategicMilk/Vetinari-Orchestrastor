"""
Unit tests for the telemetry module.

Tests metrics collection, aggregation, and export functionality.
"""

import unittest
import json
import tempfile
import os
from vetinari.telemetry import (
    TelemetryCollector, AdapterMetrics, MemoryMetrics, PlanMetrics,
    get_telemetry_collector, reset_telemetry
)


class TestAdapterMetrics(unittest.TestCase):
    """Test adapter metrics."""
    
    def test_adapter_metrics_initialization(self):
        """Test AdapterMetrics initialization."""
        metrics = AdapterMetrics(provider="openai", model="gpt-4")
        self.assertEqual(metrics.provider, "openai")
        self.assertEqual(metrics.model, "gpt-4")
        self.assertEqual(metrics.total_requests, 0)
        self.assertEqual(metrics.success_rate, 0.0)
        self.assertEqual(metrics.avg_latency_ms, 0.0)
    
    def test_adapter_metrics_success_rate(self):
        """Test success rate calculation."""
        metrics = AdapterMetrics(provider="openai", model="gpt-4")
        metrics.total_requests = 10
        metrics.successful_requests = 7
        metrics.failed_requests = 3
        
        self.assertAlmostEqual(metrics.success_rate, 70.0, places=1)
    
    def test_adapter_metrics_avg_latency(self):
        """Test average latency calculation."""
        metrics = AdapterMetrics(provider="openai", model="gpt-4")
        metrics.successful_requests = 2
        metrics.total_latency_ms = 300.0  # 150 + 150
        
        self.assertAlmostEqual(metrics.avg_latency_ms, 150.0, places=1)


class TestMemoryMetrics(unittest.TestCase):
    """Test memory metrics."""
    
    def test_memory_metrics_initialization(self):
        """Test MemoryMetrics initialization."""
        metrics = MemoryMetrics(backend="oc")
        self.assertEqual(metrics.backend, "oc")
        self.assertEqual(metrics.total_writes, 0)
        self.assertEqual(metrics.dedup_hit_rate, 0.0)
    
    def test_dedup_hit_rate(self):
        """Test dedup hit rate calculation."""
        metrics = MemoryMetrics(backend="oc")
        metrics.dedup_hits = 75
        metrics.dedup_misses = 25
        
        self.assertAlmostEqual(metrics.dedup_hit_rate, 75.0, places=1)
    
    def test_average_latencies(self):
        """Test average latency calculations."""
        metrics = MemoryMetrics(backend="oc")
        metrics.write_latency_ms = [5.0, 6.0, 4.0]
        metrics.read_latency_ms = [2.0, 2.5, 1.5]
        metrics.search_latency_ms = [20.0, 25.0, 15.0]
        
        self.assertAlmostEqual(metrics.avg_write_latency(), 5.0, places=1)
        self.assertAlmostEqual(metrics.avg_read_latency(), 2.0, places=1)
        self.assertAlmostEqual(metrics.avg_search_latency(), 20.0, places=1)


class TestPlanMetrics(unittest.TestCase):
    """Test plan metrics."""
    
    def test_plan_metrics_initialization(self):
        """Test PlanMetrics initialization."""
        metrics = PlanMetrics()
        self.assertEqual(metrics.total_decisions, 0)
        self.assertEqual(metrics.approval_rate, 0.0)
        self.assertEqual(metrics.average_risk_score, 0.0)
    
    def test_approval_rate(self):
        """Test approval rate calculation."""
        metrics = PlanMetrics()
        metrics.total_decisions = 10
        metrics.approved_decisions = 8
        metrics.rejected_decisions = 2
        
        self.assertAlmostEqual(metrics.approval_rate, 80.0, places=1)
    
    def test_average_risk_score_update(self):
        """Test average risk score calculation."""
        metrics = PlanMetrics()
        metrics.risk_scores = [0.2, 0.5, 0.3]
        metrics.update_average_risk_score()
        
        self.assertAlmostEqual(metrics.average_risk_score, 0.33, places=2)
    
    def test_average_approval_time_update(self):
        """Test average approval time calculation."""
        metrics = PlanMetrics()
        metrics.approval_times_ms = [100.0, 150.0, 200.0]
        metrics.update_average_approval_time()
        
        self.assertAlmostEqual(metrics.average_approval_time_ms, 150.0, places=1)


class TestTelemetryCollector(unittest.TestCase):
    """Test TelemetryCollector."""
    
    def setUp(self):
        self.collector = get_telemetry_collector()
    
    def test_singleton_instance(self):
        """Test singleton pattern."""
        collector1 = get_telemetry_collector()
        collector2 = get_telemetry_collector()
        self.assertIs(collector1, collector2)
    
    def test_record_adapter_latency(self):
        """Test recording adapter latency."""
        self.collector.record_adapter_latency("openai", "gpt-4", 150.5, success=True, tokens_used=100)
        
        metrics = self.collector.get_adapter_metrics("openai")
        self.assertIn("openai:gpt-4", metrics)
        self.assertEqual(metrics["openai:gpt-4"].total_requests, 1)
        self.assertEqual(metrics["openai:gpt-4"].successful_requests, 1)
    
    def test_record_memory_write(self):
        """Test recording memory write."""
        self.collector.record_memory_write("oc", 5.2)
        self.collector.record_memory_write("oc", 4.8)
        
        metrics = self.collector.get_memory_metrics("oc")
        self.assertEqual(metrics["oc"].total_writes, 2)
    
    def test_record_plan_decision(self):
        """Test recording plan decision."""
        self.collector.record_plan_decision("approve", risk_score=0.3, auto_approved=True)
        
        metrics = self.collector.get_plan_metrics()
        self.assertEqual(metrics.total_decisions, 1)
        self.assertEqual(metrics.approved_decisions, 1)
        self.assertEqual(metrics.auto_approved_decisions, 1)
    
    def test_export_json(self):
        """Test JSON export."""
        self.collector.record_adapter_latency("openai", "gpt-4", 150.5)
        self.collector.record_plan_decision("approve", risk_score=0.3)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "telemetry.json")
            success = self.collector.export_json(export_path)
            
            self.assertTrue(success)
            with open(export_path) as f:
                data = json.load(f)
                self.assertIn("timestamp", data)
                self.assertIn("adapters", data)
                self.assertIn("plan_mode", data)
    
    def test_export_prometheus(self):
        """Test Prometheus export."""
        self.collector.record_adapter_latency("openai", "gpt-4", 150.5)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "metrics.txt")
            success = self.collector.export_prometheus(export_path)
            
            self.assertTrue(success)
            with open(export_path) as f:
                content = f.read()
                self.assertIn("vetinari_adapter", content)
    
    def test_reset(self):
        """Test telemetry reset."""
        self.collector.record_adapter_latency("openai", "gpt-4", 150.5)
        self.collector.record_plan_decision("approve", risk_score=0.3)
        
        self.collector.reset()
        
        metrics = self.collector.get_plan_metrics()
        self.assertEqual(metrics.total_decisions, 0)
        self.assertEqual(len(self.collector.adapter_metrics), 0)


class TestTelemetryThreadSafety(unittest.TestCase):
    """Test thread safety of telemetry collector."""
    
    def setUp(self):
        self.collector = get_telemetry_collector()
    
    def test_concurrent_recording(self):
        """Test concurrent metric recording."""
        import threading
        
        def record_metrics(thread_id):
            for i in range(10):
                self.collector.record_adapter_latency(
                    "openai", "gpt-4", 100.0 + thread_id, success=True
                )
                self.collector.record_plan_decision("approve", risk_score=0.5)
        
        threads = [threading.Thread(target=record_metrics, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        metrics = self.collector.get_plan_metrics()
        self.assertEqual(metrics.total_decisions, 50)  # 5 threads * 10 decisions


if __name__ == "__main__":
    unittest.main()
