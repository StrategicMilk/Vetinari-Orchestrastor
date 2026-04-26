"""Tests for vetinari.learning.shadow_testing."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from vetinari.learning.shadow_testing import (
    _MAX_ERROR_RATE_INCREASE,
    _MAX_LATENCY_RATIO,
    _MIN_QUALITY_IMPROVEMENT,
    _ROLLBACK_QUALITY_MARGIN,
    _ROLLBACK_WINDOW_HOURS,
    ShadowMetrics,
    ShadowTest,
    ShadowTestRunner,
    get_shadow_test_runner,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _runner(tmp_path: Path) -> ShadowTestRunner:
    """Return a ShadowTestRunner that persists to tmp_path."""
    with patch("vetinari.learning.shadow_testing.get_user_dir", return_value=tmp_path):
        return ShadowTestRunner()


def _fill(
    runner: ShadowTestRunner,
    test_id: str,
    *,
    prod_quality: float,
    prod_latency: float,
    cand_quality: float,
    cand_latency: float,
    n: int = 10,
) -> None:
    """Record n identical paired observations for both variants."""
    for _ in range(n):
        runner.record_production(test_id, quality=prod_quality, latency_ms=prod_latency)
        runner.record_candidate(test_id, quality=cand_quality, latency_ms=cand_latency)


# ---------------------------------------------------------------------------
# ShadowMetrics
# ---------------------------------------------------------------------------


class TestShadowMetrics:
    def test_avg_quality_empty(self) -> None:
        m = ShadowMetrics()
        assert m.avg_quality == 0.0

    def test_avg_quality_with_data(self) -> None:
        m = ShadowMetrics(quality_scores=[0.6, 0.8, 1.0])
        assert m.avg_quality == pytest.approx(0.8)

    def test_avg_latency_empty(self) -> None:
        m = ShadowMetrics()
        assert m.avg_latency == 0.0

    def test_avg_latency_with_data(self) -> None:
        m = ShadowMetrics(latency_ms_values=[100.0, 200.0])
        assert m.avg_latency == pytest.approx(150.0)

    def test_error_rate_no_runs(self) -> None:
        m = ShadowMetrics()
        assert m.error_rate == 0.0

    def test_error_rate_with_errors(self) -> None:
        m = ShadowMetrics(error_count=2, total_runs=10)
        assert m.error_rate == pytest.approx(0.2)

    def test_repr(self) -> None:
        m = ShadowMetrics(total_runs=5, error_count=1)
        assert "total_runs=5" in repr(m)
        assert "error_count=1" in repr(m)


# ---------------------------------------------------------------------------
# ShadowTest repr
# ---------------------------------------------------------------------------


class TestShadowTest:
    def test_repr_shows_id_and_status(self) -> None:
        t = ShadowTest(
            test_id="shadow_abc",
            description="test",
            production_config={},
            candidate_config={},
        )
        assert "shadow_abc" in repr(t)
        assert "running" in repr(t)


# ---------------------------------------------------------------------------
# ShadowTestRunner — creation
# ---------------------------------------------------------------------------


class TestCreateTest:
    def test_returns_prefixed_id(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        tid = runner.create_test("desc", {}, {})
        assert tid.startswith("shadow_")

    def test_unique_ids(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        ids = {runner.create_test("x", {}, {}) for _ in range(20)}
        assert len(ids) == 20

    def test_test_appears_in_active_tests(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        tid = runner.create_test("active check", {}, {})
        active = runner.get_active_tests()
        assert any(t["test_id"] == tid for t in active)

    def test_persists_to_disk(self, tmp_path: Path) -> None:
        with patch("vetinari.learning.shadow_testing.get_user_dir", return_value=tmp_path):
            runner = ShadowTestRunner()
            runner.create_test("persist check", {}, {})
            state_file = tmp_path / "shadow_tests.json"
            assert state_file.exists()


# ---------------------------------------------------------------------------
# ShadowTestRunner — record methods
# ---------------------------------------------------------------------------


class TestRecording:
    def test_production_runs_incremented(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        tid = runner.create_test("rec", {}, {})
        runner.record_production(tid, quality=0.8, latency_ms=100.0)
        runner.record_production(tid, quality=0.9, latency_ms=110.0, error=True)
        with runner._lock:
            test = runner._tests[tid]
        assert test.production_metrics.total_runs == 2
        assert test.production_metrics.error_count == 1

    def test_candidate_runs_incremented(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        tid = runner.create_test("rec", {}, {})
        runner.record_candidate(tid, quality=0.85, latency_ms=120.0)
        with runner._lock:
            test = runner._tests[tid]
        assert test.candidate_metrics.total_runs == 1

    def test_record_ignores_concluded_test(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        tid = runner.create_test("concluded", {}, {})
        _fill(runner, tid, prod_quality=0.8, prod_latency=100, cand_quality=0.9, cand_latency=100)
        runner.evaluate(tid)  # promotes or rejects
        with runner._lock:
            runs_before = runner._tests[tid].production_metrics.total_runs
        runner.record_production(tid, quality=0.5, latency_ms=200)
        with runner._lock:
            runs_after = runner._tests[tid].production_metrics.total_runs
        assert runs_before == runs_after

    def test_record_unknown_test_is_silent(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        # Recording to unknown test names should silently succeed
        result1 = runner.record_production("shadow_nonexistent", quality=0.5, latency_ms=100)
        result2 = runner.record_candidate("shadow_nonexistent", quality=0.5, latency_ms=100)
        assert result1 is None
        assert result2 is None


# ---------------------------------------------------------------------------
# ShadowTestRunner — evaluate: insufficient data
# ---------------------------------------------------------------------------


class TestEvaluateInsufficientData:
    def test_returns_insufficient_data_when_empty(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        tid = runner.create_test("empty", {}, {}, min_samples=5)
        result = runner.evaluate(tid)
        assert result["decision"] == "insufficient_data"
        assert result["production_runs"] == 0
        assert result["candidate_runs"] == 0
        assert result["min_samples"] == 5

    def test_returns_insufficient_data_partial(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        tid = runner.create_test("partial", {}, {}, min_samples=5)
        for _ in range(3):
            runner.record_production(tid, quality=0.8, latency_ms=100)
            runner.record_candidate(tid, quality=0.8, latency_ms=100)
        result = runner.evaluate(tid)
        assert result["decision"] == "insufficient_data"

    def test_not_found(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        result = runner.evaluate("shadow_missing")
        assert result["decision"] == "not_found"

    def test_already_concluded(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        tid = runner.create_test("dup", {}, {})
        _fill(runner, tid, prod_quality=0.8, prod_latency=100, cand_quality=0.9, cand_latency=100)
        runner.evaluate(tid)
        result = runner.evaluate(tid)
        assert result["decision"] in {"promoted", "rejected"}
        assert "already" in result["reasoning"]


# ---------------------------------------------------------------------------
# ShadowTestRunner — evaluate: promotion
# ---------------------------------------------------------------------------


class TestEvaluatePromotion:
    def test_promotes_when_candidate_better(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        tid = runner.create_test("better cand", {}, {"new": True})
        _fill(runner, tid, prod_quality=0.7, prod_latency=100, cand_quality=0.8, cand_latency=110)
        result = runner.evaluate(tid)
        assert result["decision"] == "promote"
        assert result["quality_delta"] == pytest.approx(0.1, abs=1e-4)
        assert result["candidate_config"] == {"new": True}

    def test_promotes_when_candidate_equal(self, tmp_path: Path) -> None:
        """Equal metrics should promote since regression is within floor."""
        runner = _runner(tmp_path)
        tid = runner.create_test("equal", {}, {})
        _fill(runner, tid, prod_quality=0.8, prod_latency=100, cand_quality=0.8, cand_latency=100)
        result = runner.evaluate(tid)
        assert result["decision"] == "promote"

    def test_promotes_within_quality_floor(self, tmp_path: Path) -> None:
        """A small regression just inside the floor should still promote."""
        runner = _runner(tmp_path)
        tid = runner.create_test("tiny regression", {}, {})
        # delta = -0.01 which is >= _MIN_QUALITY_IMPROVEMENT (-0.02)
        _fill(runner, tid, prod_quality=0.8, prod_latency=100, cand_quality=0.79, cand_latency=100)
        result = runner.evaluate(tid)
        assert result["decision"] == "promote"

    def test_test_status_becomes_promoted(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        tid = runner.create_test("status", {}, {})
        _fill(runner, tid, prod_quality=0.7, prod_latency=100, cand_quality=0.8, cand_latency=100)
        runner.evaluate(tid)
        with runner._lock:
            assert runner._tests[tid].status == "promoted"
            assert runner._tests[tid].promoted_at is not None

    def test_promotion_includes_metric_fields(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        tid = runner.create_test("fields", {}, {})
        _fill(runner, tid, prod_quality=0.7, prod_latency=100, cand_quality=0.8, cand_latency=100)
        result = runner.evaluate(tid)
        assert "quality_delta" in result
        assert "latency_ratio" in result
        assert "error_rate_delta" in result


# ---------------------------------------------------------------------------
# ShadowTestRunner — evaluate: rejection
# ---------------------------------------------------------------------------


class TestEvaluateRejection:
    def test_rejects_quality_regression(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        tid = runner.create_test("quality fail", {}, {})
        # delta = -0.1 which is < -0.02 floor
        _fill(runner, tid, prod_quality=0.8, prod_latency=100, cand_quality=0.7, cand_latency=100)
        result = runner.evaluate(tid)
        assert result["decision"] == "reject"
        assert any("Quality" in r for r in result["reasons"])

    def test_rejects_latency_regression(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        tid = runner.create_test("latency fail", {}, {})
        # ratio = 200/100 = 2.0 which is > 1.3 max
        _fill(runner, tid, prod_quality=0.8, prod_latency=100, cand_quality=0.85, cand_latency=200)
        result = runner.evaluate(tid)
        assert result["decision"] == "reject"
        assert any("Latency" in r for r in result["reasons"])

    def test_rejects_error_rate_increase(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        tid = runner.create_test("error fail", {}, {}, min_samples=5)
        for _ in range(5):
            runner.record_production(tid, quality=0.8, latency_ms=100, error=False)
        for _ in range(5):
            runner.record_candidate(tid, quality=0.8, latency_ms=100, error=True)
        result = runner.evaluate(tid)
        assert result["decision"] == "reject"
        assert any("Error" in r for r in result["reasons"])

    def test_multiple_failures_listed(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        tid = runner.create_test("multi fail", {}, {})
        # Both quality and latency fail
        _fill(runner, tid, prod_quality=0.8, prod_latency=100, cand_quality=0.65, cand_latency=200)
        result = runner.evaluate(tid)
        assert result["decision"] == "reject"
        assert len(result["reasons"]) >= 2

    def test_test_status_becomes_rejected(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        tid = runner.create_test("status reject", {}, {})
        _fill(runner, tid, prod_quality=0.8, prod_latency=100, cand_quality=0.65, cand_latency=100)
        runner.evaluate(tid)
        with runner._lock:
            assert runner._tests[tid].status == "rejected"


# ---------------------------------------------------------------------------
# ShadowTestRunner — check_rollback
# ---------------------------------------------------------------------------


class TestCheckRollback:
    def test_no_rollback_for_nonexistent_test(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        assert runner.check_rollback("shadow_ghost", 0.1) is False

    def test_no_rollback_for_running_test(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        tid = runner.create_test("still running", {}, {})
        assert runner.check_rollback(tid, 0.1) is False

    def test_no_rollback_past_window(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        tid = runner.create_test("old promo", {}, {})
        _fill(runner, tid, prod_quality=0.8, prod_latency=100, cand_quality=0.85, cand_latency=100)
        runner.evaluate(tid)
        # Backdate promotion to beyond rollback window
        past = (datetime.now(timezone.utc) - timedelta(hours=_ROLLBACK_WINDOW_HOURS + 1)).isoformat()
        with runner._lock:
            runner._tests[tid].promoted_at = past
        assert runner.check_rollback(tid, 0.01) is False

    def test_rollback_triggered_on_degradation_within_window(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        tid = runner.create_test("fresh promo", {}, {})
        _fill(runner, tid, prod_quality=0.8, prod_latency=100, cand_quality=0.85, cand_latency=100)
        runner.evaluate(tid)
        # Quality falls well below baseline (0.8 - 0.05 margin = 0.75, so 0.6 triggers)
        triggered = runner.check_rollback(tid, 0.6)
        assert triggered is True
        with runner._lock:
            assert runner._tests[tid].status == "rolled_back"

    def test_no_rollback_when_quality_above_margin(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        tid = runner.create_test("stable promo", {}, {})
        _fill(runner, tid, prod_quality=0.8, prod_latency=100, cand_quality=0.85, cand_latency=100)
        runner.evaluate(tid)
        # 0.76 is within margin (floor = 0.8 - 0.05 = 0.75)
        triggered = runner.check_rollback(tid, 0.76)
        assert triggered is False
        with runner._lock:
            assert runner._tests[tid].status == "promoted"


# ---------------------------------------------------------------------------
# ShadowTestRunner — get_active_tests
# ---------------------------------------------------------------------------


class TestGetActiveTests:
    def test_only_running_tests_returned(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        t_run = runner.create_test("running", {}, {})
        t_done = runner.create_test("done", {}, {})
        _fill(runner, t_done, prod_quality=0.8, prod_latency=100, cand_quality=0.9, cand_latency=100)
        runner.evaluate(t_done)
        active_ids = {t["test_id"] for t in runner.get_active_tests()}
        assert t_run in active_ids
        assert t_done not in active_ids

    def test_result_has_expected_keys(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        runner.create_test("key check", {}, {})
        active = runner.get_active_tests()
        assert len(active) == 1
        keys = set(active[0])
        assert keys == {
            "test_id",
            "description",
            "status",
            "production_runs",
            "candidate_runs",
            "created_at",
        }


# ---------------------------------------------------------------------------
# ShadowTestRunner — state persistence
# ---------------------------------------------------------------------------


class TestStatePersistence:
    def test_state_survives_reload(self, tmp_path: Path) -> None:
        # Metrics are flushed to disk when evaluate() or create_test() is called.
        # Record observations, then evaluate (which persists) and verify reload.
        with patch("vetinari.learning.shadow_testing.get_user_dir", return_value=tmp_path):
            runner1 = ShadowTestRunner()
            tid = runner1.create_test("persist", {"k": "v"}, {"k2": "v2"}, min_samples=3)
            for _ in range(3):
                runner1.record_production(tid, quality=0.75, latency_ms=150)
                runner1.record_candidate(tid, quality=0.80, latency_ms=160)
            runner1.evaluate(tid)  # triggers _save_state with accumulated metrics

            runner2 = ShadowTestRunner()

        with runner2._lock:
            test = runner2._tests.get(tid)
        assert test is not None
        assert test.description == "persist"
        assert test.production_metrics.total_runs == 3
        assert test.candidate_metrics.total_runs == 3

    def test_corrupt_state_file_starts_empty(self, tmp_path: Path) -> None:
        state_file = tmp_path / "shadow_tests.json"
        state_file.write_text("not valid json", encoding="utf-8")
        with patch("vetinari.learning.shadow_testing.get_user_dir", return_value=tmp_path):
            runner = ShadowTestRunner()
        assert runner.get_active_tests() == []


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_same_instance_returned(self) -> None:
        r1 = get_shadow_test_runner()
        r2 = get_shadow_test_runner()
        assert r1 is r2

    def test_thread_safe_singleton(self) -> None:
        instances: list[ShadowTestRunner] = []
        errors: list[Exception] = []

        def grab() -> None:
            try:
                instances.append(get_shadow_test_runner())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=grab) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len({id(i) for i in instances}) == 1
