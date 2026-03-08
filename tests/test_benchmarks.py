"""Tests for the multi-layer benchmark framework (Task 17)."""

import tempfile
from pathlib import Path

import pytest

from vetinari.benchmarks.runner import (
    BenchmarkCase,
    BenchmarkLayer,
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuiteAdapter,
    BenchmarkTier,
    ComparisonReport,
    MetricStore,
    get_default_runner,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyAdapter(BenchmarkSuiteAdapter):
    """Minimal adapter for testing."""
    name = "dummy"
    layer = BenchmarkLayer.AGENT
    tier = BenchmarkTier.FAST

    def __init__(self, cases=None, pass_all=True):
        self._cases = cases or [
            BenchmarkCase(
                case_id=f"dummy-{i}",
                suite_name="dummy",
                description=f"Dummy case {i}",
                input_data={"value": i},
            )
            for i in range(5)
        ]
        self._pass_all = pass_all

    def load_cases(self, limit=None):
        cases = self._cases
        if limit is not None:
            cases = cases[:limit]
        return cases

    def run_case(self, case, run_id):
        return BenchmarkResult(
            case_id=case.case_id,
            suite_name=self.name,
            run_id=run_id,
            passed=self._pass_all,
            score=0.9 if self._pass_all else 0.2,
            latency_ms=10.0,
            tokens_consumed=100,
            cost_usd=0.001,
        )

    def evaluate(self, result):
        return result.score


class FailingAdapter(DummyAdapter):
    """Adapter where all cases fail."""
    name = "failing"

    def __init__(self):
        super().__init__(pass_all=False)


# ---------------------------------------------------------------------------
# BenchmarkRunner basics
# ---------------------------------------------------------------------------

class TestBenchmarkRunner:
    def test_create_runner(self):
        runner = BenchmarkRunner()
        assert runner.list_suites() == []

    def test_register_suite(self):
        runner = BenchmarkRunner()
        runner.register_suite(DummyAdapter())
        suites = runner.list_suites()
        assert len(suites) == 1
        assert suites[0]["name"] == "dummy"
        assert suites[0]["layer"] == "AGENT"
        assert suites[0]["tier"] == "fast"

    def test_register_multiple_suites(self):
        runner = BenchmarkRunner()
        runner.register_suite(DummyAdapter())
        runner.register_suite(FailingAdapter())
        assert len(runner.list_suites()) == 2

    def test_run_unknown_suite_raises(self):
        runner = BenchmarkRunner()
        with pytest.raises(ValueError, match="Unknown suite"):
            runner.run_suite("nonexistent")

    def test_run_single_bad_format_raises(self):
        runner = BenchmarkRunner()
        with pytest.raises(ValueError, match="suite_name:case_id"):
            runner.run_single("no-colon")


# ---------------------------------------------------------------------------
# Suite execution
# ---------------------------------------------------------------------------

class TestSuiteExecution:
    def _runner(self, tmp_path):
        runner = BenchmarkRunner(db_path=tmp_path / "bench.db")
        runner.register_suite(DummyAdapter())
        runner.register_suite(FailingAdapter())
        return runner

    def test_run_suite_basic(self, tmp_path):
        runner = self._runner(tmp_path)
        report = runner.run_suite("dummy")
        assert report.suite_name == "dummy"
        assert report.total_cases == 5
        assert report.passed_cases == 5
        assert report.pass_at_1 == 1.0
        assert report.avg_score > 0

    def test_run_suite_with_limit(self, tmp_path):
        runner = self._runner(tmp_path)
        report = runner.run_suite("dummy", limit=3)
        assert report.total_cases == 3

    def test_run_suite_with_trials(self, tmp_path):
        runner = self._runner(tmp_path)
        report = runner.run_suite("dummy", trials=3)
        # 5 cases * 3 trials = 15 results
        assert len(report.results) == 15
        assert report.pass_k == 1.0  # all pass all trials

    def test_run_failing_suite(self, tmp_path):
        runner = self._runner(tmp_path)
        report = runner.run_suite("failing")
        assert report.passed_cases == 0
        assert report.pass_at_1 == 0.0

    def test_run_single_case(self, tmp_path):
        runner = self._runner(tmp_path)
        result = runner.run_single("dummy:dummy-0")
        assert result.case_id == "dummy-0"
        assert result.passed

    def test_run_single_unknown_case(self, tmp_path):
        runner = self._runner(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            runner.run_single("dummy:nonexistent")


# ---------------------------------------------------------------------------
# Metric persistence
# ---------------------------------------------------------------------------

class TestMetricStore:
    def test_save_and_load_report(self, tmp_path):
        store = MetricStore(db_path=tmp_path / "test.db")
        report = BenchmarkReport(
            run_id="test-run-1",
            suite_name="dummy",
            layer=BenchmarkLayer.AGENT,
            tier=BenchmarkTier.FAST,
            results=[],
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:01:00Z",
            total_cases=5,
            passed_cases=4,
            pass_at_1=0.8,
        )
        store.save_report(report)
        loaded = store.load_report("test-run-1")
        assert loaded is not None
        assert loaded["suite_name"] == "dummy"
        assert loaded["pass_at_1"] == 0.8

    def test_load_nonexistent(self, tmp_path):
        store = MetricStore(db_path=tmp_path / "test.db")
        assert store.load_report("nonexistent") is None

    def test_list_runs(self, tmp_path):
        store = MetricStore(db_path=tmp_path / "test.db")
        for i in range(3):
            report = BenchmarkReport(
                run_id=f"run-{i}",
                suite_name="dummy",
                layer=BenchmarkLayer.AGENT,
                tier=BenchmarkTier.FAST,
                results=[],
                started_at=f"2026-01-0{i+1}T00:00:00Z",
            )
            store.save_report(report)
        runs = store.list_runs()
        assert len(runs) == 3

    def test_list_runs_filtered(self, tmp_path):
        store = MetricStore(db_path=tmp_path / "test.db")
        for name in ["a", "b", "a"]:
            import uuid
            report = BenchmarkReport(
                run_id=f"run-{uuid.uuid4().hex[:6]}",
                suite_name=name,
                layer=BenchmarkLayer.AGENT,
                tier=BenchmarkTier.FAST,
                results=[],
                started_at="2026-01-01T00:00:00Z",
            )
            store.save_report(report)
        runs = store.list_runs(suite_name="a")
        assert len(runs) == 2


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

class TestComparison:
    def test_compare_runs(self, tmp_path):
        runner = BenchmarkRunner(db_path=tmp_path / "comp.db")
        runner.register_suite(DummyAdapter())
        r1 = runner.run_suite("dummy")
        r2 = runner.run_suite("dummy")
        comp = runner.compare_runs(r1.run_id, r2.run_id)
        assert isinstance(comp, ComparisonReport)
        assert comp.run_a == r1.run_id
        assert comp.run_b == r2.run_id

    def test_compare_nonexistent_raises(self, tmp_path):
        runner = BenchmarkRunner(db_path=tmp_path / "comp.db")
        with pytest.raises(ValueError, match="not found"):
            runner.compare_runs("nope-a", "nope-b")

    def test_get_last_comparison(self, tmp_path):
        runner = BenchmarkRunner(db_path=tmp_path / "comp.db")
        runner.register_suite(DummyAdapter())
        # Need at least 2 runs
        runner.run_suite("dummy")
        runner.run_suite("dummy")
        comp = runner.get_last_comparison("dummy")
        assert comp is not None

    def test_get_last_comparison_insufficient_runs(self, tmp_path):
        runner = BenchmarkRunner(db_path=tmp_path / "comp.db")
        runner.register_suite(DummyAdapter())
        runner.run_suite("dummy")
        assert runner.get_last_comparison("dummy") is None


# ---------------------------------------------------------------------------
# Report aggregation
# ---------------------------------------------------------------------------

class TestBenchmarkReport:
    def test_compute_aggregates(self):
        results = [
            BenchmarkResult("c1", "s", "r1", True, 0.9, latency_ms=10, tokens_consumed=50),
            BenchmarkResult("c2", "s", "r1", False, 0.3, latency_ms=20, tokens_consumed=100),
        ]
        report = BenchmarkReport(
            run_id="r1",
            suite_name="s",
            layer=BenchmarkLayer.AGENT,
            tier=BenchmarkTier.FAST,
            results=results,
            started_at="t0",
        )
        report.compute_aggregates()
        assert report.total_cases == 2
        assert report.passed_cases == 1
        assert report.pass_at_1 == 0.5
        assert report.avg_score == pytest.approx(0.6, abs=0.01)
        assert report.total_tokens == 150

    def test_summary_dict(self):
        report = BenchmarkReport(
            run_id="r1",
            suite_name="dummy",
            layer=BenchmarkLayer.AGENT,
            tier=BenchmarkTier.FAST,
            results=[],
            started_at="t0",
        )
        summary = report.summary_dict()
        assert "run_id" in summary
        assert "pass@1" in summary
        assert "pass^k" in summary


# ---------------------------------------------------------------------------
# Adapter loading
# ---------------------------------------------------------------------------

class TestAdapters:
    def test_swe_bench_loads(self):
        from vetinari.benchmarks.swe_bench import SWEBenchAdapter
        adapter = SWEBenchAdapter()
        assert adapter.name == "swe_bench"
        cases = adapter.load_cases(limit=2)
        assert len(cases) <= 2

    def test_tau_bench_loads(self):
        from vetinari.benchmarks.tau_bench import TauBenchAdapter
        adapter = TauBenchAdapter()
        assert adapter.name == "tau_bench"
        cases = adapter.load_cases(limit=2)
        assert len(cases) <= 2

    def test_toolbench_loads(self):
        from vetinari.benchmarks.toolbench import ToolBenchAdapter
        adapter = ToolBenchAdapter()
        assert adapter.name == "toolbench"
        cases = adapter.load_cases(limit=2)
        assert len(cases) <= 2

    def test_taskbench_loads(self):
        from vetinari.benchmarks.taskbench import TaskBenchAdapter
        adapter = TaskBenchAdapter()
        assert adapter.name == "taskbench"
        cases = adapter.load_cases(limit=2)
        assert len(cases) <= 2

    def test_api_bank_loads(self):
        from vetinari.benchmarks.api_bank import APIBankAdapter
        adapter = APIBankAdapter()
        assert adapter.name == "api_bank"
        cases = adapter.load_cases(limit=2)
        assert len(cases) <= 2

    def test_get_default_runner(self, tmp_path):
        runner = get_default_runner(db_path=tmp_path / "default.db")
        suites = runner.list_suites()
        assert len(suites) == 5  # all 5 adapters

    def test_adapter_evaluate(self):
        from vetinari.benchmarks.swe_bench import SWEBenchAdapter
        adapter = SWEBenchAdapter()
        cases = adapter.load_cases(limit=1)
        result = adapter.run_case(cases[0], "test-run")
        score = adapter.evaluate(result)
        assert 0.0 <= score <= 1.0
