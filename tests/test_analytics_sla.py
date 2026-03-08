"""Tests for vetinari/analytics/sla.py (Phase 5)"""
import time
import unittest

from vetinari.analytics.sla import (
    SLABreach, SLAReport, SLATracker, SLOTarget, SLOType,
    get_sla_tracker, reset_sla_tracker,
)


def _tracker():
    return get_sla_tracker()


class TestSLOTarget(unittest.TestCase):
    def test_to_dict(self):
        s = SLOTarget(name="lat", slo_type=SLOType.LATENCY_P95, budget=500.0)
        d = s.to_dict()
        self.assertEqual(d["slo_type"], "latency_p95")
        self.assertEqual(d["budget"], 500.0)


class TestSLABreach(unittest.TestCase):
    def test_to_dict(self):
        b = SLABreach(slo_name="lat", value=600.0, budget=500.0)
        d = b.to_dict()
        self.assertEqual(d["slo_name"], "lat")


class TestSLATrackerSingleton(unittest.TestCase):
    def setUp(self):    reset_sla_tracker()
    def tearDown(self): reset_sla_tracker()

    def test_same_instance(self):
        self.assertIs(get_sla_tracker(), get_sla_tracker())

    def test_reset_new_instance(self):
        a = get_sla_tracker()
        reset_sla_tracker()
        self.assertIsNot(a, get_sla_tracker())


class TestSLOManagement(unittest.TestCase):
    def setUp(self):    reset_sla_tracker()
    def tearDown(self): reset_sla_tracker()

    def test_register_and_list(self):
        t = _tracker()
        t.register_slo(SLOTarget(name="a", slo_type=SLOType.LATENCY_P95, budget=500.0))
        # 3 pre-registered defaults + 1 new
        self.assertEqual(len(t.list_slos()), 4)

    def test_unregister(self):
        t = _tracker()
        t.register_slo(SLOTarget(name="b", slo_type=SLOType.SUCCESS_RATE, budget=99.0))
        self.assertTrue(t.unregister_slo("b"))
        # 3 pre-registered defaults remain
        self.assertEqual(len(t.list_slos()), 3)

    def test_unregister_nonexistent(self):
        t = _tracker()
        self.assertFalse(t.unregister_slo("ghost"))


class TestLatencyP95(unittest.TestCase):
    def setUp(self):    reset_sla_tracker()
    def tearDown(self): reset_sla_tracker()

    def test_report_no_samples(self):
        t = _tracker()
        t.register_slo(SLOTarget(name="lat", slo_type=SLOType.LATENCY_P95, budget=500.0))
        r = t.get_report("lat")
        self.assertEqual(r.total_samples, 0)
        self.assertTrue(r.is_compliant)

    def test_all_under_budget_compliant(self):
        t = _tracker()
        t.register_slo(SLOTarget(name="lat", slo_type=SLOType.LATENCY_P95, budget=500.0))
        for _ in range(50):
            t.record_latency("key", 200.0)
        r = t.get_report("lat")
        self.assertTrue(r.is_compliant)
        self.assertLessEqual(r.current_value, 500.0)

    def test_mostly_over_budget_not_compliant(self):
        t = _tracker()
        t.register_slo(SLOTarget(name="lat", slo_type=SLOType.LATENCY_P95, budget=100.0))
        for _ in range(50):
            t.record_latency("key", 800.0)
        r = t.get_report("lat")
        self.assertGreater(r.current_value, 100.0)

    def test_report_to_dict(self):
        t = _tracker()
        t.register_slo(SLOTarget(name="lat", slo_type=SLOType.LATENCY_P95, budget=500.0))
        t.record_latency("key", 200.0)
        d = t.get_report("lat").to_dict()
        for k in ("slo","total_samples","good_samples","compliance_pct",
                  "is_compliant","current_value","breaches"):
            self.assertIn(k, d)


class TestSuccessRate(unittest.TestCase):
    def setUp(self):    reset_sla_tracker()
    def tearDown(self): reset_sla_tracker()

    def test_all_success_compliant(self):
        t = _tracker()
        t.register_slo(SLOTarget(name="sr", slo_type=SLOType.SUCCESS_RATE, budget=99.0))
        for _ in range(100):
            t.record_request(success=True)
        r = t.get_report("sr")
        self.assertAlmostEqual(r.current_value, 100.0)
        self.assertTrue(r.is_compliant)

    def test_high_failure_rate_not_compliant(self):
        t = _tracker()
        t.register_slo(SLOTarget(name="sr", slo_type=SLOType.SUCCESS_RATE, budget=99.0))
        for _ in range(50): t.record_request(success=False)
        for _ in range(50): t.record_request(success=True)
        r = t.get_report("sr")
        self.assertAlmostEqual(r.current_value, 50.0, delta=1.0)


class TestErrorRate(unittest.TestCase):
    def setUp(self):    reset_sla_tracker()
    def tearDown(self): reset_sla_tracker()

    def test_low_error_rate_compliant(self):
        t = _tracker()
        t.register_slo(SLOTarget(name="er", slo_type=SLOType.ERROR_RATE, budget=5.0))
        for _ in range(99): t.record_request(success=True)
        t.record_request(success=False)
        r = t.get_report("er")
        self.assertLessEqual(r.current_value, 5.0)


class TestApprovalRate(unittest.TestCase):
    def setUp(self):    reset_sla_tracker()
    def tearDown(self): reset_sla_tracker()

    def test_all_approved_compliant(self):
        t = _tracker()
        t.register_slo(SLOTarget(name="ar", slo_type=SLOType.APPROVAL_RATE, budget=90.0))
        for _ in range(10):
            t.record_plan_decision(approved=True)
        r = t.get_report("ar")
        self.assertAlmostEqual(r.current_value, 100.0)

    def test_low_approval_not_compliant(self):
        t = _tracker()
        t.register_slo(SLOTarget(name="ar", slo_type=SLOType.APPROVAL_RATE, budget=90.0))
        for _ in range(5): t.record_plan_decision(approved=False)
        for _ in range(5): t.record_plan_decision(approved=True)
        r = t.get_report("ar")
        self.assertLess(r.current_value, 90.0)


class TestDirectMetric(unittest.TestCase):
    def setUp(self):    reset_sla_tracker()
    def tearDown(self): reset_sla_tracker()

    def test_record_metric_direct(self):
        t = _tracker()
        t.register_slo(SLOTarget(name="custom", slo_type=SLOType.LATENCY_P95, budget=300.0))
        for v in [100, 150, 200, 120, 130]:
            t.record_metric("custom", float(v))
        r = t.get_report("custom")
        self.assertEqual(r.total_samples, 5)


class TestGetAllReports(unittest.TestCase):
    def setUp(self):    reset_sla_tracker()
    def tearDown(self): reset_sla_tracker()

    def test_returns_all(self):
        t = _tracker()
        t.register_slo(SLOTarget(name="a", slo_type=SLOType.LATENCY_P95, budget=500.0))
        t.register_slo(SLOTarget(name="b", slo_type=SLOType.SUCCESS_RATE, budget=99.0))
        reports = t.get_all_reports()
        # 3 pre-registered defaults + 2 new
        self.assertEqual(len(reports), 5)


class TestStats(unittest.TestCase):
    def setUp(self):    reset_sla_tracker()
    def tearDown(self): reset_sla_tracker()

    def test_stats_keys(self):
        t = _tracker()
        stats = t.get_stats()
        for k in ("registered_slos", "total_breaches", "slo_names"):
            self.assertIn(k, stats)

    def test_clear(self):
        t = _tracker()
        t.register_slo(SLOTarget(name="lat", slo_type=SLOType.LATENCY_P95, budget=500.0))
        for _ in range(5): t.record_latency("k", 200.0)
        t.clear()
        r = t.get_report("lat")
        self.assertEqual(r.total_samples, 0)

    def test_get_report_unknown_returns_none(self):
        t = _tracker()
        self.assertIsNone(t.get_report("unknown"))


if __name__ == "__main__":
    unittest.main()
