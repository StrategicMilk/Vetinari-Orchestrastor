"""
Regression Tests — Phase 5 (Advanced Analytics & Cost Optimization)

Guards the public contracts of every Phase 5 analytics component.
"""

import math
import time
import unittest

# ─── AnomalyDetector ─────────────────────────────────────────────────────────

from vetinari.analytics.anomaly import (
    AnomalyConfig, AnomalyResult,
    get_anomaly_detector, reset_anomaly_detector,
)


class TestAnomalyDetectorContract(unittest.TestCase):

    def setUp(self):
        self.det = get_anomaly_detector()
        self.det.configure(AnomalyConfig(min_samples=5, z_threshold=2.0))

    def tearDown(self):
        pass

    def test_detect_returns_result(self):
        r = self.det.detect("lat", 100.0)
        self.assertIsInstance(r, AnomalyResult)

    def test_result_to_dict_keys(self):
        r = self.det.detect("lat", 100.0)
        for k in ("metric","value","timestamp","is_anomaly","method","score","reason"):
            self.assertIn(k, r.to_dict())

    def test_normal_after_stable_baseline(self):
        for v in [100.0 + 2.0 * math.sin(i) for i in range(20)]:
            self.det.detect("lat", v)
        r = self.det.detect("lat", 101.0)
        self.assertFalse(r.is_anomaly)

    def test_spike_is_anomaly(self):
        for _ in range(20):
            self.det.detect("lat", 100.0 + 2.0)
        r = self.det.detect("lat", 9999.0)
        self.assertTrue(r.is_anomaly)

    def test_get_history_list(self):
        self.assertIsInstance(self.det.get_history(), list)

    def test_get_stats_keys(self):
        stats = self.det.get_stats()
        for k in ("tracked_metrics","total_anomalies","config"):
            self.assertIn(k, stats)

    def test_singleton(self):
        self.assertIs(get_anomaly_detector(), get_anomaly_detector())


# ─── CostTracker ─────────────────────────────────────────────────────────────

from vetinari.analytics.cost import (
    CostEntry, CostReport, ModelPricing,
    get_cost_tracker, reset_cost_tracker,
)


class TestCostTrackerContract(unittest.TestCase):

    def setUp(self):
        self.tracker = get_cost_tracker()

    def tearDown(self):
        pass

    def test_record_auto_costs(self):
        self.tracker.set_pricing("openai", "gpt-4",
                                 ModelPricing(input_per_1k=0.03, output_per_1k=0.06))
        e = self.tracker.record(CostEntry(
            provider="openai", model="gpt-4",
            input_tokens=1000, output_tokens=500))
        self.assertGreater(e.cost_usd, 0)

    def test_report_shape(self):
        self.tracker.record(CostEntry(provider="p", model="m",
                                      agent="a", task_id="t", cost_usd=0.01))
        r = self.tracker.get_report()
        self.assertIsInstance(r, CostReport)
        d = r.to_dict()
        for k in ("total_cost_usd","total_tokens","total_requests",
                  "by_agent","by_provider","by_model","by_task","entries"):
            self.assertIn(k, d)

    def test_filter_by_agent(self):
        self.tracker.record(CostEntry(provider="p", model="m",
                                      agent="x", cost_usd=0.05))
        self.tracker.record(CostEntry(provider="p", model="m",
                                      agent="y", cost_usd=0.10))
        r = self.tracker.get_report(agent="x")
        self.assertEqual(r.total_requests, 1)
        self.assertAlmostEqual(r.total_cost_usd, 0.05)

    def test_filter_since_future_empty(self):
        self.tracker.record(CostEntry(provider="p", model="m", cost_usd=0.01))
        r = self.tracker.get_report(since=time.time() + 3600)
        self.assertEqual(r.total_requests, 0)

    def test_singleton(self):
        self.assertIs(get_cost_tracker(), get_cost_tracker())


# ─── SLATracker ──────────────────────────────────────────────────────────────

from vetinari.analytics.sla import (
    SLOTarget, SLOType, SLAReport, SLABreach,
    get_sla_tracker, reset_sla_tracker,
)


class TestSLATrackerContract(unittest.TestCase):

    def setUp(self):
        self.tracker = get_sla_tracker()

    def tearDown(self):
        pass

    def test_register_and_list(self):
        self.tracker.register_slo(SLOTarget(
            name="lat", slo_type=SLOType.LATENCY_P95, budget=500.0))
        # 3 pre-registered defaults + 1 new
        self.assertEqual(len(self.tracker.list_slos()), 4)

    def test_report_shape(self):
        self.tracker.register_slo(SLOTarget(
            name="lat", slo_type=SLOType.LATENCY_P95, budget=500.0))
        for _ in range(20):
            self.tracker.record_latency("k", 200.0)
        r = self.tracker.get_report("lat")
        self.assertIsInstance(r, SLAReport)
        d = r.to_dict()
        for k in ("slo","total_samples","good_samples","compliance_pct",
                  "is_compliant","current_value","breaches"):
            self.assertIn(k, d)

    def test_unknown_slo_returns_none(self):
        self.assertIsNone(self.tracker.get_report("__missing__"))

    def test_no_samples_100pct_compliant(self):
        self.tracker.register_slo(SLOTarget(
            name="empty", slo_type=SLOType.SUCCESS_RATE, budget=99.0))
        r = self.tracker.get_report("empty")
        self.assertTrue(r.is_compliant)

    def test_success_rate_slo(self):
        self.tracker.register_slo(SLOTarget(
            name="sr", slo_type=SLOType.SUCCESS_RATE, budget=99.0))
        for _ in range(100):
            self.tracker.record_request(success=True)
        r = self.tracker.get_report("sr")
        self.assertAlmostEqual(r.current_value, 100.0)

    def test_get_all_reports(self):
        self.tracker.register_slo(SLOTarget(
            name="a", slo_type=SLOType.LATENCY_P95, budget=500.0))
        self.tracker.register_slo(SLOTarget(
            name="b", slo_type=SLOType.SUCCESS_RATE, budget=99.0))
        # 3 pre-registered defaults + 2 new
        self.assertEqual(len(self.tracker.get_all_reports()), 5)

    def test_singleton(self):
        self.assertIs(get_sla_tracker(), get_sla_tracker())


# ─── Forecaster ──────────────────────────────────────────────────────────────

from vetinari.analytics.forecasting import (
    ForecastRequest, ForecastResult,
    get_forecaster, reset_forecaster,
)


class TestForecasterContract(unittest.TestCase):

    def setUp(self):
        self.fc = get_forecaster()

    def tearDown(self):
        pass

    def test_forecast_returns_result(self):
        for i in range(20):
            self.fc.ingest("lat", float(i * 10))
        r = self.fc.forecast(ForecastRequest(metric="lat", horizon=3))
        self.assertIsInstance(r, ForecastResult)

    def test_result_shape(self):
        for i in range(20):
            self.fc.ingest("lat", float(i * 10))
        r = self.fc.forecast(ForecastRequest(metric="lat", horizon=5))
        self.assertEqual(len(r.predictions), 5)
        self.assertEqual(len(r.confidence_lo), 5)
        self.assertEqual(len(r.confidence_hi), 5)
        d = r.to_dict()
        for k in ("metric","method","horizon","predictions",
                  "confidence_lo","confidence_hi","trend_slope","rmse"):
            self.assertIn(k, d)

    def test_all_methods_work(self):
        self.fc.ingest_many("m", [float(i) for i in range(30)])
        for method in ("sma", "exp_smoothing", "linear_trend", "seasonal"):
            r = self.fc.forecast(ForecastRequest(
                metric="m", horizon=3, method=method, period=7))
            self.assertEqual(len(r.predictions), 3,
                             f"Method {method!r} returned wrong length")

    def test_empty_history_no_crash(self):
        r = self.fc.forecast(ForecastRequest(metric="__empty__", horizon=3))
        self.assertEqual(len(r.predictions), 3)

    def test_will_exceed_rising(self):
        # Series 0, 5, 10 … 95; slope=5, last=95 → exceeds 110 within 5 steps
        self.fc.ingest_many("lat", [float(i * 5) for i in range(20)])
        self.assertTrue(self.fc.will_exceed("lat", threshold=110.0, horizon=10))

    def test_unknown_method_raises(self):
        self.fc.ingest_many("lat", [1.0]*10)
        with self.assertRaises(ValueError):
            self.fc.forecast(ForecastRequest(metric="lat", method="__bad__", horizon=2))

    def test_singleton(self):
        self.assertIs(get_forecaster(), get_forecaster())

    def test_get_stats_keys(self):
        stats = self.fc.get_stats()
        for k in ("tracked_metrics", "history_sizes"):
            self.assertIn(k, stats)


# ─── Package-level import contract ───────────────────────────────────────────

class TestAnalyticsPackageImport(unittest.TestCase):

    def test_top_level_imports(self):
        from vetinari.analytics import (
            AnomalyConfig, AnomalyDetector, AnomalyResult,
            get_anomaly_detector, reset_anomaly_detector,
            CostEntry, CostReport, CostTracker, ModelPricing,
            get_cost_tracker, reset_cost_tracker,
            SLABreach, SLAReport, SLATracker, SLOTarget, SLOType,
            get_sla_tracker, reset_sla_tracker,
            ForecastRequest, ForecastResult, Forecaster,
            get_forecaster, reset_forecaster,
        )

    def test_version_attribute(self):
        import vetinari.analytics as pkg
        self.assertTrue(hasattr(pkg, "__version__"))


if __name__ == "__main__":
    unittest.main()
