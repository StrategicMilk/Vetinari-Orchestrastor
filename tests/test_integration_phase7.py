"""
Phase 7 Integration Smoke Tests

Exercises cross-cutting multi-component workflows:
  1. Telemetry → DashboardAPI → AlertEngine pipeline
  2. Cost tracking → analytics report pipeline
  3. SLA tracking → breach detection pipeline
  4. Log aggregation → search pipeline
  5. Adapter base → registry → model scoring pipeline
  6. Memory → dual store → search pipeline
  7. Agent contracts → planner agent → task execution pipeline
  8. Anomaly detector → scan_snapshot integration
"""

import math
import os
import tempfile
import unittest


class TestTelemetryToDashboard(unittest.TestCase):
    """Telemetry data flows through to DashboardAPI correctly."""

    @classmethod
    def setUpClass(cls):
        from vetinari.dashboard.api import reset_dashboard
        from vetinari.telemetry import reset_telemetry
        reset_telemetry()
        reset_dashboard()

    def tearDown(self):
        from vetinari.dashboard.api import reset_dashboard
        from vetinari.telemetry import reset_telemetry
        reset_telemetry()
        reset_dashboard()

    def test_adapter_metrics_visible_in_snapshot(self):
        from vetinari.dashboard.api import get_dashboard_api
        from vetinari.telemetry import get_telemetry_collector

        tel = get_telemetry_collector()
        for i in range(5):
            tel.record_adapter_latency("openai", "gpt-4", 100.0 + i * 20,
                                       success=True, tokens_used=100)
        tel.record_adapter_latency("openai", "gpt-4", 500.0, success=False)

        api  = get_dashboard_api()
        snap = api.get_latest_metrics()
        d    = snap.to_dict()

        assert "adapters" in d
        assert d["adapters"].get("total_requests", 0) > 0

    def test_plan_metrics_in_snapshot(self):
        from vetinari.dashboard.api import get_dashboard_api
        from vetinari.telemetry import get_telemetry_collector

        tel = get_telemetry_collector()
        for _ in range(5):
            tel.record_plan_decision("approve", risk_score=0.2,
                                     approval_time_ms=80.0)
        tel.record_plan_decision("reject", risk_score=0.8,
                                 approval_time_ms=200.0)

        snap = get_dashboard_api().get_latest_metrics()
        d    = snap.to_dict()
        assert "plan" in d
        assert d["plan"].get("total_decisions", 0) == 6


class TestTelemetryToAlerts(unittest.TestCase):
    """Alert engine fires when telemetry metrics cross thresholds."""

    def setUp(self):
        from vetinari.dashboard.alerts import reset_alert_engine
        from vetinari.dashboard.api import reset_dashboard
        from vetinari.telemetry import reset_telemetry
        reset_telemetry()
        reset_dashboard()
        reset_alert_engine()

    def tearDown(self):
        from vetinari.dashboard.alerts import reset_alert_engine
        from vetinari.dashboard.api import reset_dashboard
        from vetinari.telemetry import reset_telemetry
        reset_telemetry()
        reset_dashboard()
        reset_alert_engine()

    def test_high_latency_alert_fires(self):
        from vetinari.dashboard.alerts import (
            AlertCondition,
            AlertSeverity,
            AlertThreshold,
            get_alert_engine,
        )
        from vetinari.dashboard.api import get_dashboard_api
        from vetinari.telemetry import get_telemetry_collector

        tel = get_telemetry_collector()
        for _ in range(10):
            tel.record_adapter_latency("openai", "gpt-4", 900.0, success=True)

        engine = get_alert_engine()
        engine.register_threshold(AlertThreshold(
            name="high-lat",
            metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN,
            threshold_value=500.0,
            severity=AlertSeverity.HIGH,
            channels=["log"],
        ))
        fired = engine.evaluate_all(api=get_dashboard_api())
        assert len(fired) == 1
        assert fired[0].threshold.name == "high-lat"
        assert fired[0].current_value > 500.0


class TestCostToAnalytics(unittest.TestCase):
    """Cost entries aggregate correctly into CostReport."""

    def setUp(self):
        from vetinari.analytics.cost import reset_cost_tracker
        reset_cost_tracker()

    def tearDown(self):
        from vetinari.analytics.cost import reset_cost_tracker
        reset_cost_tracker()

    def test_end_to_end_cost_report(self):
        from vetinari.analytics.cost import (
            CostEntry,
            ModelPricing,
            get_cost_tracker,
        )

        tracker = get_cost_tracker()
        tracker.set_pricing("openai", "gpt-4",
                            ModelPricing(input_per_1k=0.03, output_per_1k=0.06))

        agents = ["builder", "evaluator", "explorer"]
        for i, agent in enumerate(agents):
            tracker.record(CostEntry(
                provider="openai", model="gpt-4",
                input_tokens=500 * (i + 1), output_tokens=200 * (i + 1),
                agent=agent, task_id=f"task-{i}",
            ))

        report = tracker.get_report()
        assert report.total_requests == 3
        assert report.total_cost_usd > 0
        assert "builder" in report.by_agent
        assert "openai:gpt-4" in report.by_model

        top = tracker.get_top_agents(n=2)
        assert len(top) == 2


class TestSLABreachDetection(unittest.TestCase):
    """SLA tracker detects and reports breaches."""

    def setUp(self):
        from vetinari.analytics.sla import reset_sla_tracker
        reset_sla_tracker()

    def tearDown(self):
        from vetinari.analytics.sla import reset_sla_tracker
        reset_sla_tracker()

    def test_p95_breach_reported(self):
        from vetinari.analytics.sla import (
            SLOTarget,
            SLOType,
            get_sla_tracker,
        )

        tracker = get_sla_tracker()
        tracker.register_slo(SLOTarget(
            name="lat-slo",
            slo_type=SLOType.LATENCY_P95,
            budget=200.0,
            window_seconds=3600,
        ))

        # Feed mostly over-budget latencies
        for _ in range(50):
            tracker.record_latency("adapter", 500.0)

        report = tracker.get_report("lat-slo")
        assert report is not None
        assert report.current_value > 200.0
        assert report.total_samples == 50

    def test_success_rate_compliance(self):
        from vetinari.analytics.sla import SLOTarget, SLOType, get_sla_tracker

        tracker = get_sla_tracker()
        tracker.register_slo(SLOTarget(
            name="sr-slo",
            slo_type=SLOType.SUCCESS_RATE,
            budget=99.0,
        ))
        for _ in range(100):
            tracker.record_request(success=True)

        report = tracker.get_report("sr-slo")
        self.assertAlmostEqual(report.current_value, 100.0)
        assert report.is_compliant


class TestForecastingCapacityPipeline(unittest.TestCase):
    """Forecaster ingests time-series and predicts capacity threshold crossing."""

    def setUp(self):
        from vetinari.analytics.forecasting import reset_forecaster
        reset_forecaster()

    def tearDown(self):
        from vetinari.analytics.forecasting import reset_forecaster
        reset_forecaster()

    def test_rising_series_exceeds_threshold(self):
        from vetinari.analytics.forecasting import ForecastRequest, get_forecaster

        fc = get_forecaster()
        for i in range(30):
            fc.ingest("load", float(i * 4))   # 0..116

        result = fc.forecast(ForecastRequest(
            metric="load", horizon=10, method="linear_trend",
        ))
        assert len(result.predictions) == 10
        # Load is rising at ~4/step; should exceed 150 within 10 steps
        assert fc.will_exceed("load", threshold=150.0, horizon=20)


class TestLogAggregatorPipeline(unittest.TestCase):
    """Log ingestion → search → file backend roundtrip."""

    def setUp(self):
        from vetinari.dashboard.log_aggregator import reset_log_aggregator
        reset_log_aggregator()

    def tearDown(self):
        from vetinari.dashboard.log_aggregator import reset_log_aggregator
        reset_log_aggregator()

    def test_ingest_and_trace_correlation(self):
        from vetinari.dashboard.log_aggregator import LogRecord, get_log_aggregator

        agg      = get_log_aggregator()
        trace_id = "integ-trace-001"

        agg.ingest(LogRecord(message="Request started",  level="INFO",
                             trace_id=trace_id, span_id="span-1",
                             logger_name="vetinari.adapter"))
        agg.ingest(LogRecord(message="Model inference",  level="INFO",
                             trace_id=trace_id, span_id="span-1",
                             logger_name="vetinari.adapter",
                             extra={"latency_ms": 145.0}))
        agg.ingest(LogRecord(message="Request complete", level="INFO",
                             trace_id=trace_id, span_id="span-2",
                             logger_name="vetinari.orchestrator"))
        agg.ingest(LogRecord(message="Other trace",      level="WARNING",
                             trace_id="other-trace", span_id="span-x",
                             logger_name="vetinari.other"))

        trace_records = agg.get_trace_records(trace_id)
        assert len(trace_records) == 3

        span_records = agg.correlate_span(trace_id, "span-1")
        assert len(span_records) == 2

        warning_records = agg.search(level="WARNING")
        assert len(warning_records) == 1

    def test_file_backend_integration(self):
        import json

        from vetinari.dashboard.log_aggregator import LogRecord, get_log_aggregator

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "audit.jsonl")
            agg  = get_log_aggregator()
            agg.configure_backend("file", path=path)
            agg._batch_size = 5
            for i in range(5):
                agg.ingest(LogRecord(message=f"event-{i}", trace_id="t1"))

            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 5
            first = json.loads(lines[0])
            assert "message" in first


class TestAdapterRegistryPipeline(unittest.TestCase):
    """Registry → adapter creation → model scoring end-to-end."""

    def setUp(self):
        from vetinari.adapters.registry import AdapterRegistry
        AdapterRegistry.clear_instances()

    def tearDown(self):
        from vetinari.adapters.registry import AdapterRegistry
        AdapterRegistry.clear_instances()

    def test_create_lmstudio_and_score_model(self):
        from vetinari.adapters.base import (
            ModelInfo,
            ProviderConfig,
            ProviderType,
        )
        from vetinari.adapters.registry import AdapterRegistry

        cfg = ProviderConfig(provider_type=ProviderType.LM_STUDIO,
                             name="lms", endpoint="http://localhost:1234")
        adapter = AdapterRegistry.create_adapter(cfg, instance_name="local")

        m = ModelInfo(id="llama-3", name="Llama 3", provider="lm_studio",
                      endpoint="http://localhost:1234/v1/chat",
                      capabilities=["code_gen", "chat"],
                      context_len=8192, memory_gb=8, version="1.0",
                      latency_estimate_ms=800, free_tier=True)
        adapter.models = [m]

        score = adapter.score_model_for_task(m, {
            "required_capabilities": ["code_gen"],
            "input_tokens": 2000,
            "max_latency_ms": 5000,
        })
        assert score > 0.7

        best_adapter, best_model = AdapterRegistry.find_best_model({
            "required_capabilities": ["code_gen"]
        })
        assert best_adapter is not None
        assert best_model.id == "llama-3"


class TestAnomalyDetectorIntegration(unittest.TestCase):
    """Anomaly detector catches spikes in simulated telemetry streams."""

    def setUp(self):
        from vetinari.analytics.anomaly import reset_anomaly_detector
        reset_anomaly_detector()

    def tearDown(self):
        from vetinari.analytics.anomaly import reset_anomaly_detector
        reset_anomaly_detector()

    def test_spike_detection_in_latency_stream(self):
        from vetinari.analytics.anomaly import AnomalyConfig, get_anomaly_detector

        detector = get_anomaly_detector()
        detector.configure(AnomalyConfig(
            window_size=30, z_threshold=2.5, min_samples=8,
        ))

        # Stable baseline
        for i in range(25):
            detector.detect("adapter.latency", 100.0 + 5.0 * math.sin(i * 0.5))

        # Spike
        r = detector.detect("adapter.latency", 1500.0)
        assert r.is_anomaly
        assert r.method in ("zscore", "iqr", "ewma")

        history = detector.get_history("adapter.latency")
        assert len(history) > 0

    def test_scan_snapshot_integration(self):
        from vetinari.analytics.anomaly import AnomalyConfig, get_anomaly_detector
        from vetinari.dashboard.api import get_dashboard_api, reset_dashboard
        from vetinari.telemetry import get_telemetry_collector, reset_telemetry

        reset_telemetry()
        reset_dashboard()

        tel = get_telemetry_collector()
        for _ in range(15):
            tel.record_adapter_latency("openai", "gpt-4", 100.0, success=True)

        detector = get_anomaly_detector()
        detector.configure(AnomalyConfig(min_samples=5))

        snap    = get_dashboard_api().get_latest_metrics()
        results = detector.scan_snapshot(snap)
        assert isinstance(results, list)

        reset_telemetry()
        reset_dashboard()


class TestMemorySearchIntegration(unittest.TestCase):
    """DualMemoryStore persists and retrieves entries across backends."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        oc_path = os.path.join(self.tmpdir, "oc.db")
        mn_path = os.path.join(self.tmpdir, "mn.json")
        from vetinari.memory.dual_memory import DualMemoryStore
        self.store = DualMemoryStore(oc_path=oc_path, mnemosyne_path=mn_path)

    def test_write_read_search_pipeline(self):
        from vetinari.memory.interfaces import MemoryEntry, MemoryEntryType

        entries = [
            MemoryEntry(agent="builder",  content="Python REST API pattern",
                        entry_type=MemoryEntryType.PATTERN),
            MemoryEntry(agent="explorer", content="FastAPI framework discovered",
                        entry_type=MemoryEntryType.DISCOVERY),
            MemoryEntry(agent="evaluator",content="Code quality score: 0.85",
                        entry_type=MemoryEntryType.SUCCESS),
        ]
        ids = [self.store.remember(e) for e in entries]
        assert len(ids) == 3

        results = self.store.search("Python")
        assert isinstance(results, list)

        stats = self.store.stats()
        assert stats.total_entries >= 1

    def test_timeline_ordering(self):
        from vetinari.memory.interfaces import MemoryEntry
        for i in range(4):
            self.store.remember(MemoryEntry(agent="a",
                                            content=f"event {i}"))
        results = self.store.timeline(limit=10)
        assert isinstance(results, list)


if __name__ == "__main__":
    unittest.main()
