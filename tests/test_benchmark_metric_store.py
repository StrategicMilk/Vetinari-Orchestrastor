"""Regression tests for benchmark metric persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from vetinari.benchmarks.benchmark_types import (
    BenchmarkLayer,
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkTier,
)
from vetinari.benchmarks.runner import MetricStore


@pytest.fixture
def isolated_metric_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Use an isolated unified database for MetricStore tests."""
    from vetinari import database

    db_path = tmp_path / "metrics.db"
    monkeypatch.setenv("VETINARI_DB_PATH", str(db_path))
    database.reset_for_testing()
    yield db_path
    database.reset_for_testing()


def _report(run_id: str = "run-idempotent-001") -> BenchmarkReport:
    report = BenchmarkReport(
        run_id=run_id,
        suite_name="idempotency-suite",
        layer=BenchmarkLayer.AGENT,
        tier=BenchmarkTier.FAST,
        started_at="2026-04-21T00:00:00+00:00",
        finished_at="2026-04-21T00:00:01+00:00",
        results=[
            BenchmarkResult(
                run_id=run_id,
                case_id="case-a",
                suite_name="idempotency-suite",
                passed=True,
                score=1.0,
                latency_ms=10.0,
                tokens_consumed=100,
                timestamp="2026-04-21T00:00:00+00:00",
            ),
            BenchmarkResult(
                run_id=run_id,
                case_id="case-b",
                suite_name="idempotency-suite",
                passed=False,
                score=0.25,
                latency_ms=30.0,
                tokens_consumed=20,
                error="expected failure",
                timestamp="2026-04-21T00:00:01+00:00",
            ),
        ],
        metadata={"source": "test"},
    )
    report.compute_aggregates()
    report.pass_k = 0.5
    return report


def test_save_report_replaces_existing_case_rows(isolated_metric_db: Path) -> None:
    """Saving the same run twice must not duplicate per-case evidence rows."""
    store = MetricStore()
    report = _report()

    store.save_report(report)
    store.save_report(report)

    loaded = store.load_report(report.run_id)
    results = store.load_results(report.run_id)

    assert loaded is not None
    assert loaded["total_cases"] == 2
    assert loaded["passed_cases"] == 1
    assert loaded["pass_at_1"] == 0.5
    assert {row["case_id"] for row in results} == {"case-a", "case-b"}
    assert len(results) == 2
