"""
Tests for analytics REST API endpoints (Phase 5, Step 12)

Tests cover:
- Cost report and top agents/models endpoints
- SLA compliance reports
- Anomaly detection history
- Forecast endpoint
- AutoTuner endpoint
- Error handling and empty data
"""

import pytest
import json

try:
    from flask import Flask
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

if HAS_FLASK:
    from vetinari.dashboard.rest_api import create_app
    from vetinari.dashboard import reset_dashboard

from vetinari.analytics.cost import get_cost_tracker, reset_cost_tracker, CostEntry
from vetinari.analytics.sla import (
    get_sla_tracker, reset_sla_tracker, SLOTarget, SLOType,
)
from vetinari.analytics.forecasting import get_forecaster, reset_forecaster
from vetinari.analytics.anomaly import get_anomaly_detector, reset_anomaly_detector


def _reset_all():
    reset_cost_tracker()
    reset_sla_tracker()
    reset_forecaster()
    reset_anomaly_detector()


@pytest.mark.skipif(not HAS_FLASK, reason="Flask not installed")
class TestCostEndpoints:
    """Tests for /api/v1/analytics/cost and /api/v1/analytics/cost/top."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _reset_all()
        reset_dashboard()
        self.app = create_app(debug=True)
        self.client = self.app.test_client()
        yield
        _reset_all()

    def test_cost_report_empty(self):
        resp = self.client.get("/api/v1/analytics/cost")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["total_requests"] == 0

    def test_cost_report_with_data(self):
        tracker = get_cost_tracker()
        tracker.record(CostEntry(
            provider="lm_studio", model="qwen-7b",
            input_tokens=100, output_tokens=50,
            agent="builder", task_id="t1",
        ))
        resp = self.client.get("/api/v1/analytics/cost")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["total_requests"] >= 1
        assert data["total_tokens"] >= 150

    def test_cost_report_filtered_by_agent(self):
        tracker = get_cost_tracker()
        tracker.record(CostEntry(
            provider="lm_studio", model="m1",
            input_tokens=100, output_tokens=50, agent="planner",
        ))
        tracker.record(CostEntry(
            provider="lm_studio", model="m1",
            input_tokens=200, output_tokens=100, agent="builder",
        ))
        resp = self.client.get("/api/v1/analytics/cost?agent=planner")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["total_requests"] == 1

    def test_cost_top_empty(self):
        resp = self.client.get("/api/v1/analytics/cost/top")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["top_agents"] == []
        assert data["top_models"] == []

    def test_cost_top_with_data(self):
        tracker = get_cost_tracker()
        tracker.record(CostEntry(
            provider="openai", model="gpt-4",
            input_tokens=1000, output_tokens=500, agent="builder",
        ))
        resp = self.client.get("/api/v1/analytics/cost/top?n=3")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert len(data["top_agents"]) >= 1
        assert len(data["top_models"]) >= 1


@pytest.mark.skipif(not HAS_FLASK, reason="Flask not installed")
class TestSLAEndpoints:
    """Tests for /api/v1/analytics/sla and /api/v1/analytics/sla/<name>."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _reset_all()
        reset_dashboard()
        self.app = create_app(debug=True)
        self.client = self.app.test_client()
        yield
        _reset_all()

    def test_sla_empty(self):
        resp = self.client.get("/api/v1/analytics/sla")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["count"] == 0
        assert data["reports"] == []

    def test_sla_with_registered_slo(self):
        tracker = get_sla_tracker()
        tracker.register_slo(SLOTarget(
            name="test-latency", slo_type=SLOType.LATENCY_P95,
            budget=2000.0, window_seconds=3600,
        ))
        tracker.record_latency("key", latency_ms=100.0, success=True)
        tracker.record_request(success=True)

        resp = self.client.get("/api/v1/analytics/sla")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["count"] >= 1

    def test_sla_single_report(self):
        tracker = get_sla_tracker()
        tracker.register_slo(SLOTarget(
            name="my-slo", slo_type=SLOType.SUCCESS_RATE,
            budget=90.0, window_seconds=3600,
        ))
        tracker.record_request(success=True)

        resp = self.client.get("/api/v1/analytics/sla/my-slo")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["slo"]["name"] == "my-slo"

    def test_sla_single_not_found(self):
        resp = self.client.get("/api/v1/analytics/sla/nonexistent")
        assert resp.status_code == 404
        data = json.loads(resp.data)
        assert "error" in data


@pytest.mark.skipif(not HAS_FLASK, reason="Flask not installed")
class TestAnomalyEndpoint:
    """Tests for /api/v1/analytics/anomalies."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _reset_all()
        reset_dashboard()
        self.app = create_app(debug=True)
        self.client = self.app.test_client()
        yield
        _reset_all()

    def test_anomalies_empty(self):
        resp = self.client.get("/api/v1/analytics/anomalies")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["count"] == 0
        assert data["anomalies"] == []

    def test_anomalies_with_data(self):
        detector = get_anomaly_detector()
        # Feed normal data then an outlier
        for v in [100, 101, 99, 100, 102, 98, 100]:
            detector.detect("test.metric", float(v))
        # Big outlier
        detector.detect("test.metric", 9999.0)

        resp = self.client.get("/api/v1/analytics/anomalies")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        # May or may not have anomalies depending on thresholds, but endpoint works
        assert "count" in data
        assert "anomalies" in data


@pytest.mark.skipif(not HAS_FLASK, reason="Flask not installed")
class TestForecastEndpoint:
    """Tests for /api/v1/analytics/forecast."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _reset_all()
        reset_dashboard()
        self.app = create_app(debug=True)
        self.client = self.app.test_client()
        yield
        _reset_all()

    def test_forecast_no_data(self):
        resp = self.client.get("/api/v1/analytics/forecast?metric=test")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "predictions" in data

    def test_forecast_with_data(self):
        fc = get_forecaster()
        for v in [100, 110, 120, 130, 140, 150]:
            fc.ingest("test.latency", float(v))

        resp = self.client.get(
            "/api/v1/analytics/forecast?metric=test.latency&horizon=3&method=linear_trend"
        )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert len(data["predictions"]) == 3
        assert data["method"] == "linear_trend"
        assert "confidence_lo" in data
        assert "confidence_hi" in data

    def test_forecast_invalid_method(self):
        fc = get_forecaster()
        for v in [1, 2, 3]:
            fc.ingest("m", float(v))

        resp = self.client.get(
            "/api/v1/analytics/forecast?metric=m&method=bogus"
        )
        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert "error" in data

    def test_forecast_default_params(self):
        resp = self.client.get("/api/v1/analytics/forecast")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["metric"] == "adapter.latency"
        assert data["horizon"] == 5


@pytest.mark.skipif(not HAS_FLASK, reason="Flask not installed")
class TestAutoTunerEndpoint:
    """Tests for /api/v1/analytics/autotuner."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _reset_all()
        reset_dashboard()
        self.app = create_app(debug=True)
        self.client = self.app.test_client()
        yield
        _reset_all()

    def test_autotuner_endpoint(self):
        resp = self.client.get("/api/v1/analytics/autotuner")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "config" in data
        assert "history" in data
        assert isinstance(data["config"], dict)
        assert isinstance(data["history"], list)
