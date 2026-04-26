"""Tests for vetinari/analytics/sla.py (Phase 5)"""

from __future__ import annotations

import pytest

from vetinari.analytics.sla import (
    SLABreach,
    SLOTarget,
    SLOType,
    get_sla_tracker,
    reset_sla_tracker,
)


def _tracker():
    reset_sla_tracker()
    return get_sla_tracker()


class TestSLOTarget:
    def test_to_dict(self) -> None:
        s = SLOTarget(name="lat", slo_type=SLOType.LATENCY_P95, budget=500.0)
        d = s.to_dict()
        assert d["slo_type"] == "latency_p95"
        assert d["budget"] == 500.0


class TestSLABreach:
    def test_to_dict(self) -> None:
        b = SLABreach(slo_name="lat", value=600.0, budget=500.0)
        d = b.to_dict()
        assert d["slo_name"] == "lat"

    @pytest.mark.parametrize(
        "breach",
        [
            SLABreach(slo_name="", value=600.0, budget=500.0),
            SLABreach(slo_name="lat", value=True, budget=500.0),
            SLABreach(slo_name="lat", value=float("nan"), budget=500.0),
            SLABreach(slo_name="lat", value=float("inf"), budget=500.0),
            SLABreach(slo_name="lat", value=-1.0, budget=500.0),
            SLABreach(slo_name="lat", value=600.0, budget=-1.0),
        ],
    )
    def test_record_breach_rejects_invalid_values(self, breach: SLABreach) -> None:
        tracker = _tracker()

        with pytest.raises(ValueError, match="SLA breach"):
            tracker.record_breach(breach)

        assert tracker.get_stats()["total_breaches"] == 0


class TestSLATrackerSingleton:
    def setup_method(self) -> None:
        reset_sla_tracker()

    def teardown_method(self) -> None:
        reset_sla_tracker()

    def test_same_instance(self) -> None:
        assert get_sla_tracker() is get_sla_tracker()

    def test_reset_new_instance(self) -> None:
        a = get_sla_tracker()
        reset_sla_tracker()
        assert a is not get_sla_tracker()


class TestSLOManagement:
    def setup_method(self) -> None:
        reset_sla_tracker()

    def teardown_method(self) -> None:
        reset_sla_tracker()

    def test_register_and_list(self) -> None:
        t = _tracker()
        t.register_slo(SLOTarget(name="a", slo_type=SLOType.LATENCY_P95, budget=500.0))
        assert len(t.list_slos()) == 1

    def test_unregister(self) -> None:
        t = _tracker()
        t.register_slo(SLOTarget(name="b", slo_type=SLOType.SUCCESS_RATE, budget=99.0))
        assert t.unregister_slo("b")
        assert len(t.list_slos()) == 0

    def test_unregister_nonexistent(self) -> None:
        t = _tracker()
        assert not t.unregister_slo("ghost")


class TestLatencyP95:
    def setup_method(self) -> None:
        reset_sla_tracker()

    def teardown_method(self) -> None:
        reset_sla_tracker()

    def test_report_no_samples(self) -> None:
        t = _tracker()
        t.register_slo(SLOTarget(name="lat", slo_type=SLOType.LATENCY_P95, budget=500.0))
        r = t.get_report("lat")
        assert r.total_samples == 0
        assert r.is_compliant

    def test_all_under_budget_compliant(self) -> None:
        t = _tracker()
        t.register_slo(SLOTarget(name="lat", slo_type=SLOType.LATENCY_P95, budget=500.0))
        for _ in range(50):
            t.record_latency("key", 200.0)
        r = t.get_report("lat")
        assert r.is_compliant
        assert r.current_value <= 500.0

    def test_mostly_over_budget_not_compliant(self) -> None:
        t = _tracker()
        t.register_slo(SLOTarget(name="lat", slo_type=SLOType.LATENCY_P95, budget=100.0))
        for _ in range(50):
            t.record_latency("key", 800.0)
        r = t.get_report("lat")
        assert r.current_value > 100.0

    def test_report_to_dict(self) -> None:
        t = _tracker()
        t.register_slo(SLOTarget(name="lat", slo_type=SLOType.LATENCY_P95, budget=500.0))
        t.record_latency("key", 200.0)
        d = t.get_report("lat").to_dict()
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


class TestSuccessRate:
    def setup_method(self) -> None:
        reset_sla_tracker()

    def teardown_method(self) -> None:
        reset_sla_tracker()

    def test_all_success_compliant(self) -> None:
        t = _tracker()
        t.register_slo(SLOTarget(name="sr", slo_type=SLOType.SUCCESS_RATE, budget=99.0))
        for _ in range(100):
            t.record_request(success=True)
        r = t.get_report("sr")
        assert r.current_value == pytest.approx(100.0)
        assert r.is_compliant

    def test_high_failure_rate_not_compliant(self) -> None:
        t = _tracker()
        t.register_slo(SLOTarget(name="sr", slo_type=SLOType.SUCCESS_RATE, budget=99.0))
        for _ in range(50):
            t.record_request(success=False)
        for _ in range(50):
            t.record_request(success=True)
        r = t.get_report("sr")
        assert r.current_value == pytest.approx(50.0, abs=1.0)


class TestErrorRate:
    def setup_method(self) -> None:
        reset_sla_tracker()

    def teardown_method(self) -> None:
        reset_sla_tracker()

    def test_low_error_rate_compliant(self) -> None:
        t = _tracker()
        t.register_slo(SLOTarget(name="er", slo_type=SLOType.ERROR_RATE, budget=5.0))
        for _ in range(99):
            t.record_request(success=True)
        t.record_request(success=False)
        r = t.get_report("er")
        assert r.current_value <= 5.0


class TestApprovalRate:
    def setup_method(self) -> None:
        reset_sla_tracker()

    def teardown_method(self) -> None:
        reset_sla_tracker()

    def test_all_approved_compliant(self) -> None:
        t = _tracker()
        t.register_slo(SLOTarget(name="ar", slo_type=SLOType.APPROVAL_RATE, budget=90.0))
        for _ in range(10):
            t.record_plan_decision(approved=True)
        r = t.get_report("ar")
        assert r.current_value == pytest.approx(100.0)

    def test_low_approval_not_compliant(self) -> None:
        t = _tracker()
        t.register_slo(SLOTarget(name="ar", slo_type=SLOType.APPROVAL_RATE, budget=90.0))
        for _ in range(5):
            t.record_plan_decision(approved=False)
        for _ in range(5):
            t.record_plan_decision(approved=True)
        r = t.get_report("ar")
        assert r.current_value < 90.0


class TestDirectMetric:
    def setup_method(self) -> None:
        reset_sla_tracker()

    def teardown_method(self) -> None:
        reset_sla_tracker()

    def test_record_metric_direct(self) -> None:
        t = _tracker()
        t.register_slo(SLOTarget(name="custom", slo_type=SLOType.LATENCY_P95, budget=300.0))
        for v in [100, 150, 200, 120, 130]:
            t.record_metric("custom", float(v))
        r = t.get_report("custom")
        assert r.total_samples == 5


class TestGetAllReports:
    def setup_method(self) -> None:
        reset_sla_tracker()

    def teardown_method(self) -> None:
        reset_sla_tracker()

    def test_returns_all(self) -> None:
        t = _tracker()
        t.register_slo(SLOTarget(name="a", slo_type=SLOType.LATENCY_P95, budget=500.0))
        t.register_slo(SLOTarget(name="b", slo_type=SLOType.SUCCESS_RATE, budget=99.0))
        reports = t.get_all_reports()
        assert len(reports) == 2


class TestStats:
    def setup_method(self) -> None:
        reset_sla_tracker()

    def teardown_method(self) -> None:
        reset_sla_tracker()

    def test_stats_keys(self) -> None:
        t = _tracker()
        stats = t.get_stats()
        for k in ("registered_slos", "total_breaches", "slo_names"):
            assert k in stats

    def test_clear(self) -> None:
        t = _tracker()
        t.register_slo(SLOTarget(name="lat", slo_type=SLOType.LATENCY_P95, budget=500.0))
        for _ in range(5):
            t.record_latency("k", 200.0)
        t.clear()
        r = t.get_report("lat")
        assert r.total_samples == 0

    def test_get_report_unknown_returns_none(self) -> None:
        t = _tracker()
        assert t.get_report("unknown") is None
