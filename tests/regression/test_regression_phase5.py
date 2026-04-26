"""
Regression Tests — Phase 5 (Advanced Analytics & Cost Optimization)

Guards the public contracts of every Phase 5 analytics component.
"""

import math
import time

import pytest

# ─── AnomalyDetector ─────────────────────────────────────────────────────────
from vetinari.analytics.anomaly import (
    AnomalyConfig,
    AnomalyResult,
    get_anomaly_detector,
    reset_anomaly_detector,
)
from vetinari.exceptions import ConfigurationError


class TestAnomalyDetectorContract:
    @pytest.fixture(autouse=True)
    def _setup(self):
        reset_anomaly_detector()
        self.det = get_anomaly_detector()
        self.det.configure(AnomalyConfig(min_samples=5, z_threshold=2.0))
        yield
        reset_anomaly_detector()

    def test_detect_returns_result(self):
        r = self.det.detect("lat", 100.0)
        assert isinstance(r, AnomalyResult)

    def test_result_to_dict_keys(self):
        r = self.det.detect("lat", 100.0)
        for k in ("metric", "value", "timestamp", "is_anomaly", "method", "score", "reason"):
            assert k in r.to_dict()

    def test_normal_after_stable_baseline(self):
        for v in [100.0 + 2.0 * math.sin(i) for i in range(20)]:
            self.det.detect("lat", v)
        r = self.det.detect("lat", 101.0)
        assert not r.is_anomaly

    def test_spike_is_anomaly(self):
        for _ in range(20):
            self.det.detect("lat", 100.0 + 2.0)
        r = self.det.detect("lat", 9999.0)
        assert r.is_anomaly

    def test_get_history_list(self):
        assert isinstance(self.det.get_history(), list)

    def test_get_stats_keys(self):
        stats = self.det.get_stats()
        for k in ("tracked_metrics", "total_anomalies", "config"):
            assert k in stats

    def test_singleton(self):
        assert get_anomaly_detector() is get_anomaly_detector()


# ─── CostTracker ─────────────────────────────────────────────────────────────

from vetinari.analytics.cost import (
    CostEntry,
    CostReport,
    ModelPricing,
    get_cost_tracker,
    reset_cost_tracker,
)


class TestCostTrackerContract:
    @pytest.fixture(autouse=True)
    def _setup(self):
        reset_cost_tracker()
        self.tracker = get_cost_tracker()
        yield
        reset_cost_tracker()

    def test_record_auto_costs(self):
        self.tracker.set_pricing("openai", "gpt-4", ModelPricing(input_per_1k=0.03, output_per_1k=0.06))
        e = self.tracker.record(CostEntry(provider="openai", model="gpt-4", input_tokens=1000, output_tokens=500))
        assert e.cost_usd > 0

    def test_report_shape(self):
        self.tracker.record(CostEntry(provider="p", model="m", agent="a", task_id="t", cost_usd=0.01))
        r = self.tracker.get_report()
        assert isinstance(r, CostReport)
        d = r.to_dict()
        for k in (
            "total_cost_usd",
            "total_tokens",
            "total_requests",
            "by_agent",
            "by_provider",
            "by_model",
            "by_task",
            "entries",
        ):
            assert k in d

    def test_filter_by_agent(self):
        self.tracker.record(CostEntry(provider="p", model="m", agent="x", cost_usd=0.05))
        self.tracker.record(CostEntry(provider="p", model="m", agent="y", cost_usd=0.10))
        r = self.tracker.get_report(agent="x")
        assert r.total_requests == 1
        assert r.total_cost_usd == pytest.approx(0.05)

    def test_filter_since_future_empty(self):
        self.tracker.record(CostEntry(provider="p", model="m", cost_usd=0.01))
        r = self.tracker.get_report(since=time.time() + 3600)
        assert r.total_requests == 0

    def test_singleton(self):
        assert get_cost_tracker() is get_cost_tracker()


# ─── SLATracker ──────────────────────────────────────────────────────────────

from vetinari.analytics.sla import (
    SLAReport,
    SLOTarget,
    SLOType,
    get_sla_tracker,
    reset_sla_tracker,
)


class TestSLATrackerContract:
    @pytest.fixture(autouse=True)
    def _setup(self):
        reset_sla_tracker()
        self.tracker = get_sla_tracker()
        yield
        reset_sla_tracker()

    def test_register_and_list(self):
        self.tracker.register_slo(SLOTarget(name="lat", slo_type=SLOType.LATENCY_P95, budget=500.0))
        assert len(self.tracker.list_slos()) == 1

    def test_report_shape(self):
        self.tracker.register_slo(SLOTarget(name="lat", slo_type=SLOType.LATENCY_P95, budget=500.0))
        for _ in range(20):
            self.tracker.record_latency("k", 200.0)
        r = self.tracker.get_report("lat")
        assert isinstance(r, SLAReport)
        d = r.to_dict()
        for k in (
            "slo",
            "total_samples",
            "good_samples",
            "compliance_pct",
            "is_compliant",
            "current_value",
            "breaches",
        ):
            assert k in d

    def test_unknown_slo_returns_none(self):
        assert self.tracker.get_report("__missing__") is None

    def test_no_samples_100pct_compliant(self):
        self.tracker.register_slo(SLOTarget(name="empty", slo_type=SLOType.SUCCESS_RATE, budget=99.0))
        r = self.tracker.get_report("empty")
        assert r.is_compliant

    def test_success_rate_slo(self):
        self.tracker.register_slo(SLOTarget(name="sr", slo_type=SLOType.SUCCESS_RATE, budget=99.0))
        for _ in range(100):
            self.tracker.record_request(success=True)
        r = self.tracker.get_report("sr")
        assert r.current_value == pytest.approx(100.0)

    def test_get_all_reports(self):
        self.tracker.register_slo(SLOTarget(name="a", slo_type=SLOType.LATENCY_P95, budget=500.0))
        self.tracker.register_slo(SLOTarget(name="b", slo_type=SLOType.SUCCESS_RATE, budget=99.0))
        assert len(self.tracker.get_all_reports()) == 2

    def test_singleton(self):
        assert get_sla_tracker() is get_sla_tracker()


# ─── Forecaster ──────────────────────────────────────────────────────────────

import pytest

from vetinari.analytics.forecasting import (
    ForecastRequest,
    ForecastResult,
    get_forecaster,
    reset_forecaster,
)


class TestForecasterContract:
    @pytest.fixture(autouse=True)
    def _setup(self):
        reset_forecaster()
        self.fc = get_forecaster()
        yield
        reset_forecaster()

    def test_forecast_returns_result(self):
        for i in range(20):
            self.fc.ingest("lat", float(i * 10))
        r = self.fc.forecast(ForecastRequest(metric="lat", horizon=3))
        assert isinstance(r, ForecastResult)

    def test_result_shape(self):
        for i in range(20):
            self.fc.ingest("lat", float(i * 10))
        r = self.fc.forecast(ForecastRequest(metric="lat", horizon=5))
        assert len(r.predictions) == 5
        assert len(r.confidence_lo) == 5
        assert len(r.confidence_hi) == 5
        d = r.to_dict()
        for k in (
            "metric",
            "forecast_method_used",
            "horizon",
            "predictions",
            "confidence_lo",
            "confidence_hi",
            "trend_slope",
            "rmse",
        ):
            assert k in d

    def test_all_methods_work(self):
        self.fc.ingest_many("m", [float(i) for i in range(30)])
        for method in ("sma", "exp_smoothing", "linear_trend", "seasonal"):
            r = self.fc.forecast(ForecastRequest(metric="m", horizon=3, method=method, period=7))
            assert len(r.predictions) == 3, f"Method {method!r} returned wrong length"

    def test_empty_history_no_crash(self):
        r = self.fc.forecast(ForecastRequest(metric="__empty__", horizon=3))
        assert len(r.predictions) == 3

    def test_will_exceed_rising(self):
        # Series 0, 5, 10 … 95; slope=5, last=95 → exceeds 110 within 5 steps
        self.fc.ingest_many("lat", [float(i * 5) for i in range(20)])
        assert self.fc.will_exceed("lat", threshold=110.0, horizon=10)

    def test_unknown_method_raises(self):
        self.fc.ingest_many("lat", [1.0] * 10)
        with pytest.raises(ConfigurationError):
            self.fc.forecast(ForecastRequest(metric="lat", method="__bad__", horizon=2))

    def test_singleton(self):
        assert get_forecaster() is get_forecaster()

    def test_get_stats_keys(self):
        stats = self.fc.get_stats()
        for k in ("tracked_metrics", "history_sizes"):
            assert k in stats


# ─── Package-level import contract ───────────────────────────────────────────


class TestAnalyticsPackageImport:
    def test_top_level_imports(self):
        import types

        import vetinari.analytics  # Verify analytics package is importable at top level

        assert isinstance(vetinari.analytics, types.ModuleType)

    def test_version_attribute(self):
        import vetinari.analytics as pkg

        assert hasattr(pkg, "__version__")
