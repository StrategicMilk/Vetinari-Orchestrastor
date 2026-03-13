"""Tests for vetinari/analytics/anomaly.py (Phase 5)"""
import unittest

from vetinari.analytics.anomaly import (
    AnomalyConfig,
    AnomalyResult,
    get_anomaly_detector,
    reset_anomaly_detector,
)


def _engine():
    reset_anomaly_detector()
    e = get_anomaly_detector()
    e.configure(AnomalyConfig(window_size=20, z_threshold=2.5,
                              iqr_factor=1.5, ewma_alpha=0.3,
                              ewma_threshold=2.5, min_samples=5))
    return e


class TestAnomalyResult(unittest.TestCase):
    def test_to_dict_keys(self):
        r = AnomalyResult(metric="m", value=1.0)
        d = r.to_dict()
        for k in ("metric","value","timestamp","is_anomaly","method","score","reason"):
            assert k in d

    def test_default_not_anomaly(self):
        r = AnomalyResult(metric="m", value=5.0)
        assert not r.is_anomaly


class TestAnomalyDetectorSingleton(unittest.TestCase):
    def setUp(self):  reset_anomaly_detector()
    def tearDown(self): reset_anomaly_detector()

    def test_same_instance(self):
        assert get_anomaly_detector() is get_anomaly_detector()

    def test_reset_new_instance(self):
        a = get_anomaly_detector()
        reset_anomaly_detector()
        assert a is not get_anomaly_detector()


class TestInsufficientSamples(unittest.TestCase):
    def setUp(self):  reset_anomaly_detector()
    def tearDown(self): reset_anomaly_detector()

    def test_no_anomaly_below_min_samples(self):
        e = get_anomaly_detector()
        e.configure(AnomalyConfig(min_samples=10))
        for i in range(9):
            r = e.detect("m", float(i))
            assert not r.is_anomaly


class TestZScoreDetection(unittest.TestCase):
    def setUp(self):  reset_anomaly_detector()
    def tearDown(self): reset_anomaly_detector()

    def test_zscore_spike_detected(self):
        e = _engine()
        # Build stable baseline
        for _ in range(15):
            e.detect("lat", 100.0)
        # Inject a large spike
        r = e.detect("lat", 2000.0)
        assert r.is_anomaly
        assert r.method == "zscore"
        assert r.score > 0

    def test_normal_value_not_flagged(self):
        e = _engine()
        # Use a spread so IQR > 0 and a 2 % deviation is clearly normal
        import math
        for i in range(20):
            e.detect("lat", 100.0 + 5.0 * math.sin(i))
        r = e.detect("lat", 100.5)   # well within any fence
        assert not r.is_anomaly


class TestIQRDetection(unittest.TestCase):
    def setUp(self):  reset_anomaly_detector()
    def tearDown(self): reset_anomaly_detector()

    def test_iqr_outlier_detected(self):
        e = _engine()
        # Tight cluster — std is 0, so Z-score won't fire; IQR should
        for _ in range(16):
            e.detect("tok", 100.0)
        r = e.detect("tok", 500.0)
        # either zscore or iqr should fire
        assert r.is_anomaly

    def test_value_within_iqr_fence_not_flagged(self):
        e = _engine()
        for v in [90, 95, 100, 105, 110, 90, 95, 100, 105, 110,
                  90, 95, 100, 105, 110]:
            e.detect("tok", float(v))
        r = e.detect("tok", 102.0)
        assert not r.is_anomaly


class TestEWMADetection(unittest.TestCase):
    def setUp(self):  reset_anomaly_detector()
    def tearDown(self): reset_anomaly_detector()

    def test_ewma_deviation_detected(self):
        e = _engine()
        # Build EWMA on a stable series
        for i in range(20):
            e.detect("sr", 99.5 + (i % 2) * 0.1)
        # Sudden drop
        r = e.detect("sr", 10.0)
        assert r.is_anomaly


class TestScanSnapshot(unittest.TestCase):
    def setUp(self):  reset_anomaly_detector()
    def tearDown(self): reset_anomaly_detector()

    def test_scan_returns_list(self):
        from unittest.mock import MagicMock
        e = _engine()
        snap = MagicMock()
        snap.to_dict.return_value = {"adapters": {"average_latency_ms": 100.0}}
        result = e.scan_snapshot(snap)
        assert isinstance(result, list)


class TestHistory(unittest.TestCase):
    def setUp(self):  reset_anomaly_detector()
    def tearDown(self): reset_anomaly_detector()

    def test_history_accumulates(self):
        e = _engine()
        for _ in range(15):
            e.detect("m", 100.0)
        e.detect("m", 9999.0)   # spike
        assert len(e.get_history()) > 0

    def test_history_filter_by_metric(self):
        e = _engine()
        for _ in range(15): e.detect("a", 100.0)
        e.detect("a", 9999.0)
        for _ in range(15): e.detect("b", 50.0)
        e.detect("b", 9999.0)
        hist_a = e.get_history("a")
        for r in hist_a:
            assert r.metric == "a"

    def test_clear_history(self):
        e = _engine()
        for _ in range(15): e.detect("m", 100.0)
        e.detect("m", 9999.0)
        e.clear_history()
        assert len(e.get_history()) == 0

    def test_get_stats(self):
        e = _engine()
        stats = e.get_stats()
        for k in ("tracked_metrics", "total_anomalies", "config"):
            assert k in stats


if __name__ == "__main__":
    unittest.main()
