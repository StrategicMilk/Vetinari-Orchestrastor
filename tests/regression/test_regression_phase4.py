"""
Regression Tests — Phase 4 (Dashboard & Monitoring)

These tests guard against regressions in the public contracts of every
Phase 4 component.  They must remain green as the codebase evolves.

Coverage:
  - DashboardAPI public surface (metrics, traces, stats)
  - REST API endpoint shape and status codes
  - AlertEngine threshold lifecycle + firing contract
  - LogAggregator ingestion, search, backend dispatch
  - Dashboard UI route presence
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

# ─── DashboardAPI ─────────────────────────────────────────────────────────────
from vetinari.dashboard.api import (
    MetricsSnapshot,
    TimeSeriesData,
    TraceDetail,
    get_dashboard_api,
    reset_dashboard,
)


class TestDashboardAPIContract(unittest.TestCase):
    """Public API shape must never silently change."""

    @classmethod
    def setUpClass(cls):
        reset_dashboard()
        cls.api = get_dashboard_api()

    @classmethod
    def tearDownClass(cls):
        reset_dashboard()

    # MetricsSnapshot contract
    def test_get_latest_metrics_returns_snapshot(self):
        snap = self.api.get_latest_metrics()
        assert isinstance(snap, MetricsSnapshot)

    def test_snapshot_to_dict_required_keys(self):
        d = self.api.get_latest_metrics().to_dict()
        for key in ("timestamp", "uptime_ms", "adapters", "memory", "plan"):
            assert key in d, f"Missing key '{key}' in MetricsSnapshot.to_dict()"

    # TimeSeriesData contract
    def test_get_timeseries_valid_metrics(self):
        for metric in ("latency", "success_rate", "token_usage", "memory_latency"):
            result = self.api.get_timeseries_data(metric)
            assert isinstance(result, TimeSeriesData), f"get_timeseries_data({metric!r}) returned wrong type"

    def test_get_timeseries_invalid_returns_none(self):
        assert self.api.get_timeseries_data("__nonexistent__") is None

    def test_timeseries_to_dict_required_keys(self):
        d = self.api.get_timeseries_data("latency").to_dict()
        for key in ("metric", "unit", "min", "max", "avg", "points"):
            assert key in d

    # Trace contract
    def test_search_traces_returns_list(self):
        result = self.api.search_traces()
        assert isinstance(result, list)

    def test_add_and_retrieve_trace(self):
        td = TraceDetail(
            trace_id="reg-test-001",
            start_time="2026-01-01T00:00:00+00:00",
            end_time="2026-01-01T00:00:01+00:00",
            duration_ms=1000.0,
            status="success",
            spans=[{"span_id": "s1", "operation": "root", "duration_ms": 1000.0}],
        )
        ok = self.api.add_trace(td)
        assert ok
        detail = self.api.get_trace_detail("reg-test-001")
        assert detail is not None
        assert detail.trace_id == "reg-test-001"

    def test_trace_detail_not_found_returns_none(self):
        assert self.api.get_trace_detail("__does_not_exist__") is None

    def test_get_stats_required_keys(self):
        stats = self.api.get_stats()
        for k in ("total_traces_stored", "trace_list_size", "timestamp"):
            assert k in stats

    def test_singleton_contract(self):
        assert get_dashboard_api() is get_dashboard_api()


# ─── REST API ─────────────────────────────────────────────────────────────────

from vetinari.dashboard.rest_api import create_app


class TestRESTAPIContract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        reset_dashboard()
        app = create_app()
        app.config["TESTING"] = True
        cls.client = app.test_client()

    @classmethod
    def tearDownClass(cls):
        reset_dashboard()

    def _json(self, path):
        r = self.client.get(path)
        return r, json.loads(r.data)

    def test_health_200(self):
        r, d = self._json("/api/v1/health")
        assert r.status_code == 200
        assert d["status"] == "healthy"

    def test_stats_200(self):
        r, _ = self._json("/api/v1/stats")
        assert r.status_code == 200

    def test_metrics_latest_200(self):
        r, d = self._json("/api/v1/metrics/latest")
        assert r.status_code == 200
        for k in ("timestamp", "adapters", "memory", "plan"):
            assert k in d

    def test_timeseries_valid_200(self):
        r, _ = self._json("/api/v1/metrics/timeseries?metric=latency")
        assert r.status_code == 200

    def test_timeseries_invalid_400(self):
        r, d = self._json("/api/v1/metrics/timeseries?metric=__bad__")
        assert r.status_code == 400
        assert "error" in d

    def test_traces_list_200(self):
        r, d = self._json("/api/v1/traces?limit=10")
        assert r.status_code == 200
        assert "count" in d
        assert "traces" in d

    def test_trace_detail_404(self):
        r, d = self._json("/api/v1/traces/__reg_nonexistent__")
        assert r.status_code == 404
        assert "error" in d

    def test_dashboard_ui_route_200(self):
        r = self.client.get("/dashboard")
        assert r.status_code == 200
        assert b"dashboard.js" in r.data

    def test_cors_headers_present(self):
        # P1.H2: CORS is restricted to localhost origins; no wildcard.
        # A request without Origin should NOT get Access-Control-Allow-Origin.
        r_no_origin = self.client.get("/api/v1/health")
        assert "Access-Control-Allow-Origin" not in r_no_origin.headers
        # A request from an allowed localhost origin should get the header.
        r_localhost = self.client.get(
            "/api/v1/health",
            headers={"Origin": "http://localhost:5000"},
        )
        assert "Access-Control-Allow-Origin" in r_localhost.headers
        assert r_localhost.headers["Access-Control-Allow-Origin"] == "http://localhost:5000"


# ─── AlertEngine ──────────────────────────────────────────────────────────────

from vetinari.dashboard.alerts import (
    AlertCondition,
    AlertThreshold,
    get_alert_engine,
    reset_alert_engine,
)


class TestAlertEngineContract(unittest.TestCase):

    def setUp(self):
        reset_alert_engine()
        self.engine = get_alert_engine()

    def tearDown(self):
        reset_alert_engine()

    def _mock_api(self, latency=100.0):
        snap = MagicMock()
        snap.to_dict.return_value = {
            "adapters": {"average_latency_ms": latency},
            "memory": {}, "plan": {},
        }
        api = MagicMock()
        api.get_latest_metrics.return_value = snap
        return api

    def test_register_replaces_by_name(self):
        self.engine.register_threshold(AlertThreshold(
            name="t", metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN, threshold_value=100.0))
        self.engine.register_threshold(AlertThreshold(
            name="t", metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN, threshold_value=200.0))
        ts = self.engine.list_thresholds()
        assert len(ts) == 1
        assert ts[0].threshold_value == 200.0

    def test_evaluate_fires_and_suppresses(self):
        self.engine.register_threshold(AlertThreshold(
            name="t", metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN, threshold_value=50.0))
        fired1 = self.engine.evaluate_all(api=self._mock_api(latency=200.0))
        fired2 = self.engine.evaluate_all(api=self._mock_api(latency=200.0))
        assert len(fired1) == 1
        assert len(fired2) == 0

    def test_evaluate_clears_after_condition_resolves(self):
        self.engine.register_threshold(AlertThreshold(
            name="t", metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN, threshold_value=50.0))
        self.engine.evaluate_all(api=self._mock_api(latency=200.0))
        self.engine.evaluate_all(api=self._mock_api(latency=10.0))
        assert len(self.engine.get_active_alerts()) == 0

    def test_singleton_contract(self):
        assert get_alert_engine() is get_alert_engine()

    def test_alert_record_to_dict(self):
        self.engine.register_threshold(AlertThreshold(
            name="t2", metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN, threshold_value=50.0))
        fired = self.engine.evaluate_all(api=self._mock_api(latency=200.0))
        assert len(fired) == 1
        d = fired[0].to_dict()
        for k in ("threshold", "current_value", "trigger_time"):
            assert k in d


# ─── LogAggregator ────────────────────────────────────────────────────────────

from vetinari.dashboard.log_aggregator import (
    LogRecord,
    get_log_aggregator,
    reset_log_aggregator,
)


class TestLogAggregatorContract(unittest.TestCase):

    def setUp(self):
        reset_log_aggregator()
        self.agg = get_log_aggregator()

    def tearDown(self):
        reset_log_aggregator()

    def test_ingest_and_retrieve(self):
        self.agg.ingest(LogRecord(message="reg-test", trace_id="r-001"))
        results = self.agg.search(trace_id="r-001")
        assert len(results) == 1
        assert results[0].message == "reg-test"

    def test_search_returns_newest_first(self):
        for i in range(5):
            r = LogRecord(message=f"msg-{i}", trace_id="r-002")
            r.timestamp = float(i)
            self.agg.ingest(r)
        results = self.agg.search(trace_id="r-002")
        assert results[0].timestamp >= results[-1].timestamp

    def test_file_backend_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "reg.jsonl")
            self.agg.configure_backend("file", path=path)
            self.agg.ingest(LogRecord(message="file-reg-test"))
            self.agg.flush()
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["message"] == "file-reg-test"

    def test_log_record_to_dict_shape(self):
        r = LogRecord(message="shape-test", trace_id="t", span_id="s")
        d = r.to_dict()
        for k in ("message", "level", "timestamp", "trace_id", "span_id"):
            assert k in d

    def test_singleton_contract(self):
        assert get_log_aggregator() is get_log_aggregator()


if __name__ == "__main__":
    unittest.main()
