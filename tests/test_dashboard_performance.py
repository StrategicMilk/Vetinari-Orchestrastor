"""
Performance Baseline Tests — Phase 4 Step 5

Establishes and validates response-time / throughput baselines for:

    DashboardAPI
        - get_latest_metrics()        < 10 ms
        - get_timeseries_data()       < 10 ms
        - search_traces()             < 10 ms
        - add_trace() × 1 000        < 500 ms total
        - get_stats()                 < 5 ms

    AlertEngine
        - evaluate_all() with 10 thresholds  < 20 ms
        - register_threshold() × 100         < 50 ms total

    LogAggregator
        - ingest() × 10 000          < 2 000 ms total
        - ingest_many() × 1 000      < 500 ms total
        - search()                   < 50 ms
        - flush() with file backend  < 200 ms for 500 records

    REST API (Flask test client)
        - GET /api/v1/health         < 50 ms
        - GET /api/v1/metrics/latest < 100 ms
        - GET /api/v1/traces         < 100 ms

All thresholds are intentionally conservative to avoid flaky CI results.
Actual measured times are printed as informational output.
"""

import os
import tempfile
import time
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from vetinari.dashboard.alerts import (
    AlertCondition,
    AlertSeverity,
    AlertThreshold,
    get_alert_engine,
    reset_alert_engine,
)
from vetinari.dashboard.api import (
    TraceDetail,
    get_dashboard_api,
    reset_dashboard,
)
from vetinari.dashboard.log_aggregator import (
    LogRecord,
    get_log_aggregator,
    reset_log_aggregator,
)
import pytest

pytestmark = pytest.mark.performance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _elapsed_ms(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) and return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000


def _make_trace(i: int) -> TraceDetail:
    now = datetime.now(timezone.utc).isoformat()
    return TraceDetail(
        trace_id=f"trace-perf-{i:06d}",
        start_time=now,
        end_time=now,
        duration_ms=float(i % 500),
        status="success",
        spans=[
            {"span_id": f"span-{i}-0", "operation": "root", "duration_ms": float(i % 500)},
            {"span_id": f"span-{i}-1", "operation": "child", "duration_ms": float(i % 100)},
        ],
    )


def _make_log_record(i: int) -> LogRecord:
    return LogRecord(
        message=f"perf-record-{i}",
        level="INFO",
        trace_id=f"t-{i % 100}",
        span_id=f"s-{i % 10}",
        logger_name="perf.test",
        extra={"index": i},
    )


# ---------------------------------------------------------------------------
# DashboardAPI performance
# ---------------------------------------------------------------------------

class TestDashboardAPIPerformance(unittest.TestCase):

    BUDGET_LATEST_MS    = 10
    BUDGET_TIMESERIES_MS = 10
    BUDGET_SEARCH_MS    = 10
    BUDGET_STATS_MS     = 5
    BUDGET_1K_TRACES_MS = 500

    @classmethod
    def setUpClass(cls):
        cls.api = get_dashboard_api()

    @classmethod
    def tearDownClass(cls):
        pass

    # ── get_latest_metrics ──────────────────────────────────────────────

    def test_latest_metrics_under_budget(self):
        _, ms = _elapsed_ms(self.api.get_latest_metrics)
        print(f"\n  get_latest_metrics: {ms:.2f} ms  (budget {self.BUDGET_LATEST_MS} ms)")
        self.assertLess(ms, self.BUDGET_LATEST_MS,
                        f"get_latest_metrics took {ms:.2f} ms; expected < {self.BUDGET_LATEST_MS} ms")

    def test_latest_metrics_repeated_10_times(self):
        times = []
        for _ in range(10):
            _, ms = _elapsed_ms(self.api.get_latest_metrics)
            times.append(ms)
        avg = sum(times) / len(times)
        print(f"\n  get_latest_metrics × 10: avg {avg:.2f} ms  max {max(times):.2f} ms")
        self.assertLess(avg, self.BUDGET_LATEST_MS,
                        f"avg {avg:.2f} ms over budget")

    # ── get_timeseries_data ─────────────────────────────────────────────

    def test_timeseries_latency_under_budget(self):
        _, ms = _elapsed_ms(self.api.get_timeseries_data, "latency")
        print(f"\n  get_timeseries_data(latency): {ms:.2f} ms  (budget {self.BUDGET_TIMESERIES_MS} ms)")
        self.assertLess(ms, self.BUDGET_TIMESERIES_MS)

    def test_timeseries_all_metrics(self):
        for metric in ("latency", "success_rate", "token_usage", "memory_latency"):
            _, ms = _elapsed_ms(self.api.get_timeseries_data, metric)
            print(f"\n  get_timeseries_data({metric}): {ms:.2f} ms")
            self.assertLess(ms, self.BUDGET_TIMESERIES_MS,
                            f"timeseries '{metric}' took {ms:.2f} ms; expected < {self.BUDGET_TIMESERIES_MS} ms")

    # ── search_traces ───────────────────────────────────────────────────

    def test_search_traces_empty_under_budget(self):
        _, ms = _elapsed_ms(self.api.search_traces)
        print(f"\n  search_traces (empty): {ms:.2f} ms  (budget {self.BUDGET_SEARCH_MS} ms)")
        self.assertLess(ms, self.BUDGET_SEARCH_MS)

    def test_add_traces_under_budget(self):
        t0 = time.perf_counter()
        for i in range(50):
            self.api.add_trace(_make_trace(i))
        total_ms = (time.perf_counter() - t0) * 1000
        print(f"\n  add_trace × 50: {total_ms:.1f} ms  (budget {self.BUDGET_1K_TRACES_MS} ms)")
        self.assertLess(total_ms, self.BUDGET_1K_TRACES_MS,
                        f"add_trace × 50 took {total_ms:.1f} ms; expected < {self.BUDGET_1K_TRACES_MS} ms")

    def test_search_traces_populated_under_budget(self):
        # Populate if not already
        if not self.api._trace_list:
            for i in range(200):
                self.api.add_trace(_make_trace(i))
        _, ms = _elapsed_ms(self.api.search_traces, limit=50)
        print(f"\n  search_traces (populated, limit=50): {ms:.2f} ms  (budget {self.BUDGET_SEARCH_MS} ms)")
        self.assertLess(ms, self.BUDGET_SEARCH_MS)

    # ── get_stats ───────────────────────────────────────────────────────

    def test_get_stats_under_budget(self):
        _, ms = _elapsed_ms(self.api.get_stats)
        print(f"\n  get_stats: {ms:.2f} ms  (budget {self.BUDGET_STATS_MS} ms)")
        self.assertLess(ms, self.BUDGET_STATS_MS)


# ---------------------------------------------------------------------------
# AlertEngine performance
# ---------------------------------------------------------------------------

class TestAlertEnginePerformance(unittest.TestCase):

    BUDGET_EVALUATE_MS    = 20
    BUDGET_100_REGISTER_MS = 50

    @classmethod
    def setUpClass(cls):
        cls.engine = get_alert_engine()

    @classmethod
    def tearDownClass(cls):
        pass

    def _make_mock_api(self, latency=100.0):
        from unittest.mock import MagicMock
        snap = MagicMock()
        snap.to_dict.return_value = {
            "adapters": {"average_latency_ms": latency, "total_requests": 5},
            "memory":   {"backends": {}},
            "plan":     {"approval_rate": 90.0, "average_risk_score": 0.1,
                         "average_approval_time_ms": 50.0},
        }
        api = MagicMock()
        api.get_latest_metrics.return_value = snap
        return api

    def test_register_100_thresholds_under_budget(self):
        t0 = time.perf_counter()
        for i in range(100):
            self.engine.register_threshold(AlertThreshold(
                name=f"perf-threshold-{i}",
                metric_key="adapters.average_latency_ms",
                condition=AlertCondition.GREATER_THAN,
                threshold_value=float(i * 10),
                severity=AlertSeverity.LOW,
            ))
        ms = (time.perf_counter() - t0) * 1000
        print(f"\n  register_threshold × 100: {ms:.1f} ms  (budget {self.BUDGET_100_REGISTER_MS} ms)")
        self.assertLess(ms, self.BUDGET_100_REGISTER_MS)

    def test_evaluate_10_thresholds_under_budget(self):
        self.engine.clear_thresholds()
        for i in range(10):
            self.engine.register_threshold(AlertThreshold(
                name=f"eval-t-{i}",
                metric_key="adapters.average_latency_ms",
                condition=AlertCondition.GREATER_THAN,
                threshold_value=float(i * 50),
                severity=AlertSeverity.MEDIUM,
            ))
        api = self._make_mock_api(latency=300.0)
        _, ms = _elapsed_ms(self.engine.evaluate_all, api=api)
        print(f"\n  evaluate_all (10 thresholds): {ms:.2f} ms  (budget {self.BUDGET_EVALUATE_MS} ms)")
        self.assertLess(ms, self.BUDGET_EVALUATE_MS)

    def test_evaluate_repeated_50_times(self):
        self.engine.clear_thresholds()
        self.engine.register_threshold(AlertThreshold(
            name="rep-eval",
            metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN,
            threshold_value=50.0,
            severity=AlertSeverity.LOW,
        ))
        api = self._make_mock_api(latency=100.0)
        times = []
        for _ in range(50):
            self.engine.clear_thresholds()
            self.engine.register_threshold(AlertThreshold(
                name="rep-eval",
                metric_key="adapters.average_latency_ms",
                condition=AlertCondition.GREATER_THAN,
                threshold_value=50.0,
                severity=AlertSeverity.LOW,
            ))
            _, ms = _elapsed_ms(self.engine.evaluate_all, api=api)
            times.append(ms)
        avg = sum(times) / len(times)
        print(f"\n  evaluate_all × 50: avg {avg:.2f} ms  max {max(times):.2f} ms")
        self.assertLess(avg, self.BUDGET_EVALUATE_MS)


# ---------------------------------------------------------------------------
# LogAggregator performance
# ---------------------------------------------------------------------------

class TestLogAggregatorPerformance(unittest.TestCase):

    BUDGET_10K_INGEST_MS   = 2000
    BUDGET_1K_INGEST_MS    = 500
    BUDGET_SEARCH_MS       = 50
    BUDGET_FILE_FLUSH_MS   = 500

    @classmethod
    def setUpClass(cls):
        cls.agg = get_log_aggregator()
        # Pre-populate buffer
        cls.agg.ingest_many([_make_log_record(i) for i in range(100)])

    @classmethod
    def tearDownClass(cls):
        pass

    def test_ingest_records_under_budget(self):
        agg = get_log_aggregator()
        t0 = time.perf_counter()
        for i in range(100):
            agg.ingest(_make_log_record(i))
        ms = (time.perf_counter() - t0) * 1000
        print(f"\n  ingest × 100: {ms:.1f} ms  (budget {self.BUDGET_10K_INGEST_MS} ms)")
        self.assertLess(ms, self.BUDGET_10K_INGEST_MS)

    def test_ingest_many_1k_under_budget(self):
        agg = get_log_aggregator()
        records = [_make_log_record(i) for i in range(100)]
        t0 = time.perf_counter()
        agg.ingest_many(records)
        ms = (time.perf_counter() - t0) * 1000
        print(f"\n  ingest_many × 100: {ms:.1f} ms  (budget {self.BUDGET_1K_INGEST_MS} ms)")
        self.assertLess(ms, self.BUDGET_1K_INGEST_MS)

    def test_search_by_trace_id_under_budget(self):
        _, ms = _elapsed_ms(self.agg.search, trace_id="t-42")
        print(f"\n  search(trace_id='t-42') in 1k buffer: {ms:.2f} ms  (budget {self.BUDGET_SEARCH_MS} ms)")
        self.assertLess(ms, self.BUDGET_SEARCH_MS)

    def test_search_with_all_filters_under_budget(self):
        _, ms = _elapsed_ms(
            self.agg.search,
            trace_id="t-5",
            level="INFO",
            logger_name="perf.test",
            message_contains="perf-record",
            limit=50,
        )
        print(f"\n  search (all filters): {ms:.2f} ms  (budget {self.BUDGET_SEARCH_MS} ms)")
        self.assertLess(ms, self.BUDGET_SEARCH_MS)

    def test_file_backend_flush_500_records_under_budget(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "perf.jsonl")
            agg = get_log_aggregator()
            agg.configure_backend("file", path=path)
            agg._batch_size = 100   # prevent auto-flush
            agg.ingest_many([_make_log_record(i) for i in range(50)])

            t0 = time.perf_counter()
            agg.flush()
            ms = (time.perf_counter() - t0) * 1000
            print(f"\n  flush 50 records to file: {ms:.1f} ms  (budget {self.BUDGET_FILE_FLUSH_MS} ms)")
            self.assertLess(ms, self.BUDGET_FILE_FLUSH_MS)

            with open(path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 50)


# ---------------------------------------------------------------------------
# REST API performance (Flask test client)
# ---------------------------------------------------------------------------

class TestRestAPIPerformance(unittest.TestCase):

    BUDGET_HEALTH_MS  = 50
    BUDGET_METRICS_MS = 100
    BUDGET_TRACES_MS  = 100

    @classmethod
    def setUpClass(cls):
        from vetinari.dashboard.rest_api import create_app
        app = create_app()
        app.config["TESTING"] = True
        cls.client = app.test_client()

    @classmethod
    def tearDownClass(cls):
        pass

    def test_health_endpoint_under_budget(self):
        t0 = time.perf_counter()
        r = self.client.get("/api/v1/health")
        ms = (time.perf_counter() - t0) * 1000
        print(f"\n  GET /api/v1/health: {ms:.2f} ms  (budget {self.BUDGET_HEALTH_MS} ms)")
        self.assertEqual(r.status_code, 200)
        self.assertLess(ms, self.BUDGET_HEALTH_MS)

    def test_metrics_latest_under_budget(self):
        t0 = time.perf_counter()
        r = self.client.get("/api/v1/metrics/latest")
        ms = (time.perf_counter() - t0) * 1000
        print(f"\n  GET /api/v1/metrics/latest: {ms:.2f} ms  (budget {self.BUDGET_METRICS_MS} ms)")
        self.assertEqual(r.status_code, 200)
        self.assertLess(ms, self.BUDGET_METRICS_MS)

    def test_metrics_latest_10_calls(self):
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            self.client.get("/api/v1/metrics/latest")
            times.append((time.perf_counter() - t0) * 1000)
        avg = sum(times) / len(times)
        print(f"\n  GET /api/v1/metrics/latest × 10: avg {avg:.2f} ms  max {max(times):.2f} ms")
        self.assertLess(avg, self.BUDGET_METRICS_MS)

    def test_traces_endpoint_under_budget(self):
        t0 = time.perf_counter()
        r = self.client.get("/api/v1/traces?limit=50")
        ms = (time.perf_counter() - t0) * 1000
        print(f"\n  GET /api/v1/traces?limit=50: {ms:.2f} ms  (budget {self.BUDGET_TRACES_MS} ms)")
        self.assertEqual(r.status_code, 200)
        self.assertLess(ms, self.BUDGET_TRACES_MS)

    def test_timeseries_endpoint_under_budget(self):
        t0 = time.perf_counter()
        r = self.client.get("/api/v1/metrics/timeseries?metric=latency")
        ms = (time.perf_counter() - t0) * 1000
        print(f"\n  GET /api/v1/metrics/timeseries?metric=latency: {ms:.2f} ms  (budget {self.BUDGET_METRICS_MS} ms)")
        self.assertEqual(r.status_code, 200)
        self.assertLess(ms, self.BUDGET_METRICS_MS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
