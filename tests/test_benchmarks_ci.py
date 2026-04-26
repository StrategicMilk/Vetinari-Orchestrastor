"""Tests for the CI benchmark suite and cost benchmarking.

Covers:
- CostBenchmark.record() correctness
- CostBenchmark.to_dict() schema
- run_ci_benchmarks() return schema
- Each score entry has required fields
"""

from __future__ import annotations

import builtins
import importlib
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# CostBenchmark tests
# ---------------------------------------------------------------------------


class TestCostBenchmarkRecord:
    """Tests for CostBenchmark.record()."""

    def test_record_calculates_total_tokens(self) -> None:
        """record() should sum input and output tokens into total_tokens."""
        from vetinari.benchmarks import CostBenchmark

        bench = CostBenchmark(task_type="plan_generation")
        bench.record(input_tokens=400, output_tokens=100, cost_per_1k=0.001)

        assert bench.input_tokens == 400
        assert bench.output_tokens == 100
        assert bench.total_tokens == 500

    def test_record_calculates_cost(self) -> None:
        """record() should compute estimated_cost_usd = (total / 1000) * rate."""
        from vetinari.benchmarks import CostBenchmark

        bench = CostBenchmark(task_type="decomposition")
        bench.record(input_tokens=1000, output_tokens=0, cost_per_1k=0.002)

        assert bench.total_tokens == 1000
        assert abs(bench.estimated_cost_usd - 0.002) < 1e-9

    def test_record_zero_tokens(self) -> None:
        """record() with zero tokens should produce zero cost."""
        from vetinari.benchmarks import CostBenchmark

        bench = CostBenchmark(task_type="noop")
        bench.record(input_tokens=0, output_tokens=0)

        assert bench.total_tokens == 0
        assert bench.estimated_cost_usd == 0.0

    def test_record_updates_recorded_at(self) -> None:
        """record() should update recorded_at to a non-empty ISO timestamp."""
        from vetinari.benchmarks import CostBenchmark

        bench = CostBenchmark(task_type="latency_test")
        bench.record(input_tokens=10, output_tokens=5)

        assert bench.recorded_at  # not empty
        # recorded_at must be a valid-looking ISO string
        assert "T" in bench.recorded_at

    def test_record_negative_input_tokens_raises(self) -> None:
        """record() should raise ValueError for negative input_tokens."""
        import pytest

        from vetinari.benchmarks import CostBenchmark

        bench = CostBenchmark(task_type="bad")
        with pytest.raises(ValueError, match="input_tokens"):
            bench.record(input_tokens=-1, output_tokens=10)

    def test_record_negative_output_tokens_raises(self) -> None:
        """record() should raise ValueError for negative output_tokens."""
        import pytest

        from vetinari.benchmarks import CostBenchmark

        bench = CostBenchmark(task_type="bad")
        with pytest.raises(ValueError, match="output_tokens"):
            bench.record(input_tokens=10, output_tokens=-5)

    def test_record_negative_cost_per_1k_raises(self) -> None:
        """record() should raise ValueError for negative cost_per_1k."""
        import pytest

        from vetinari.benchmarks import CostBenchmark

        bench = CostBenchmark(task_type="bad")
        with pytest.raises(ValueError, match="cost_per_1k"):
            bench.record(input_tokens=10, output_tokens=5, cost_per_1k=-0.001)

    def test_record_idempotent_on_repeated_calls(self) -> None:
        """Calling record() a second time should overwrite all fields."""
        from vetinari.benchmarks import CostBenchmark

        bench = CostBenchmark(task_type="retry_task")
        bench.record(input_tokens=100, output_tokens=50)
        bench.record(input_tokens=200, output_tokens=10)

        assert bench.input_tokens == 200
        assert bench.output_tokens == 10
        assert bench.total_tokens == 210


class TestCostBenchmarkToDict:
    """Tests for CostBenchmark.to_dict()."""

    def test_to_dict_returns_dict(self) -> None:
        """to_dict() must return a plain dict."""
        from vetinari.benchmarks import CostBenchmark

        bench = CostBenchmark(task_type="test")
        result = bench.to_dict()

        assert isinstance(result, dict)

    def test_to_dict_has_expected_keys(self) -> None:
        """to_dict() must include all required keys."""
        from vetinari.benchmarks import CostBenchmark

        required_keys = {
            "task_type",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "estimated_cost_usd",
            "recorded_at",
        }
        bench = CostBenchmark(task_type="schema_check")
        bench.record(input_tokens=50, output_tokens=20)

        result = bench.to_dict()
        assert required_keys.issubset(result.keys())

    def test_to_dict_values_match_attributes(self) -> None:
        """to_dict() values must mirror the dataclass attributes."""
        from vetinari.benchmarks import CostBenchmark

        bench = CostBenchmark(task_type="value_check")
        bench.record(input_tokens=300, output_tokens=150, cost_per_1k=0.005)

        result = bench.to_dict()
        assert result["task_type"] == "value_check"
        assert result["input_tokens"] == 300
        assert result["output_tokens"] == 150
        assert result["total_tokens"] == 450
        assert abs(result["estimated_cost_usd"] - 0.00225) < 1e-7

    def test_to_dict_is_json_serialisable(self) -> None:
        """to_dict() output must be JSON-serialisable without errors."""
        import json

        from vetinari.benchmarks import CostBenchmark

        bench = CostBenchmark(task_type="json_test")
        bench.record(input_tokens=100, output_tokens=50)

        serialised = json.dumps(bench.to_dict())
        # Verify the serialised string is non-empty and round-trips correctly.
        assert isinstance(serialised, str) and len(serialised) > 2
        assert json.loads(serialised)["task_type"] == "json_test"


# ---------------------------------------------------------------------------
# run_ci_benchmarks() schema tests
# ---------------------------------------------------------------------------


class TestRunCiBenchmarks:
    """Tests for run_ci_benchmarks() return value schema."""

    def test_returns_dict(self) -> None:
        """run_ci_benchmarks() must return a dict."""
        from vetinari.benchmarks import run_ci_benchmarks

        result = run_ci_benchmarks()
        assert isinstance(result, dict)

    def test_has_required_top_level_keys(self) -> None:
        """Result must include timestamp, suite, scores, and overall_passed."""
        from vetinari.benchmarks import run_ci_benchmarks

        result = run_ci_benchmarks()
        required = {"timestamp", "suite", "scores", "overall_passed"}
        assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"

    def test_suite_is_ci(self) -> None:
        """suite field must equal 'ci'."""
        from vetinari.benchmarks import run_ci_benchmarks

        result = run_ci_benchmarks()
        assert result["suite"] == "ci"

    def test_timestamp_is_iso_string(self) -> None:
        """timestamp must be a non-empty ISO 8601 string."""
        from vetinari.benchmarks import run_ci_benchmarks

        result = run_ci_benchmarks()
        ts = result["timestamp"]
        assert isinstance(ts, str)
        assert "T" in ts  # ISO 8601 separator

    def test_scores_is_list(self) -> None:
        """scores must be a list."""
        from vetinari.benchmarks import run_ci_benchmarks

        result = run_ci_benchmarks()
        assert isinstance(result["scores"], list)

    def test_scores_non_empty(self) -> None:
        """scores list must have at least one entry."""
        from vetinari.benchmarks import run_ci_benchmarks

        result = run_ci_benchmarks()
        assert len(result["scores"]) >= 1

    def test_overall_passed_is_bool(self) -> None:
        """overall_passed must be a Python bool."""
        from vetinari.benchmarks import run_ci_benchmarks

        result = run_ci_benchmarks()
        assert isinstance(result["overall_passed"], bool)

    def test_overall_passed_matches_scores(self) -> None:
        """overall_passed must equal True iff all score entries passed."""
        from vetinari.benchmarks import run_ci_benchmarks

        result = run_ci_benchmarks()
        expected = all(s["passed"] for s in result["scores"])
        assert result["overall_passed"] == expected

    def test_plan_latency_import_failure_is_not_passing(self) -> None:
        from vetinari.benchmarks import ci_benchmarks

        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "vetinari.planning.plan_mode":
                raise ImportError("planner unavailable")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocked_import):
            result = ci_benchmarks._ci_plan_latency()

        assert result["passed"] is False
        assert "error" in result

    def test_token_optimizer_import_failure_is_not_passing(self) -> None:
        from vetinari.benchmarks import ci_benchmarks

        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "vetinari.token_optimizer":
                raise ImportError("optimizer unavailable")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocked_import):
            result = ci_benchmarks._ci_token_optimization()

        assert result["passed"] is False
        assert "error" in result

    def test_toolbench_failure_path_does_not_score_expected_answers(self) -> None:
        from vetinari.benchmarks.toolbench import ToolBenchAdapter

        adapter = ToolBenchAdapter()
        case = adapter.load_cases(limit=1)[0]

        with patch.object(adapter, "_run_via_agent", side_effect=RuntimeError("tool path unavailable")):
            result = adapter.run_case(case, "run-1")

        score = adapter.evaluate(result)
        assert result.output["benchmark_mode"] == "unavailable"
        assert "error" in result.output
        assert score < 0.5

    @pytest.mark.parametrize(
        ("adapter_ref", "method_name"),
        [
            ("vetinari.benchmarks.taskbench.TaskBenchAdapter", "_run_via_planner"),
            ("vetinari.benchmarks.toolbench.ToolBenchAdapter", "_run_via_agent"),
            ("vetinari.benchmarks.tau_bench.TauBenchAdapter", "_run_via_orchestrator"),
            ("vetinari.benchmarks.api_bank.APIBankAdapter", "_run_via_orchestrator"),
            ("vetinari.benchmarks.swe_bench.SWEBenchAdapter", "_generate_patch_via_orchestrator"),
        ],
    )
    def test_benchmark_adapters_fail_closed_when_live_path_is_unavailable(
        self,
        adapter_ref: str,
        method_name: str,
    ) -> None:
        module_name, class_name = adapter_ref.rsplit(".", 1)
        module = importlib.import_module(module_name)
        adapter_cls = getattr(module, class_name)
        adapter = adapter_cls()
        case = adapter.load_cases(limit=1)[0]

        with patch.object(adapter, method_name, side_effect=RuntimeError("live path unavailable")):
            result = adapter.run_case(case, "run-1")

        score = adapter.evaluate(result)
        assert result.error == "live path unavailable"
        assert result.output["benchmark_mode"] == "unavailable"
        assert score < 0.5


class TestCiBenchmarkScoreFields:
    """Tests that each score entry has the required fields."""

    def test_each_score_has_name(self) -> None:
        """Every score entry must have a 'name' key with a non-empty string."""
        from vetinari.benchmarks import run_ci_benchmarks

        result = run_ci_benchmarks()
        for entry in result["scores"]:
            assert "name" in entry, f"Missing 'name' in {entry}"
            assert isinstance(entry["name"], str)
            assert entry["name"]

    def test_each_score_has_score(self) -> None:
        """Every score entry must have a numeric 'score' key."""
        from vetinari.benchmarks import run_ci_benchmarks

        result = run_ci_benchmarks()
        for entry in result["scores"]:
            assert "score" in entry, f"Missing 'score' in {entry}"
            assert isinstance(entry["score"], (int, float))

    def test_each_score_has_threshold(self) -> None:
        """Every score entry must have a numeric 'threshold' key."""
        from vetinari.benchmarks import run_ci_benchmarks

        result = run_ci_benchmarks()
        for entry in result["scores"]:
            assert "threshold" in entry, f"Missing 'threshold' in {entry}"
            assert isinstance(entry["threshold"], (int, float))

    def test_each_score_has_passed(self) -> None:
        """Every score entry must have a boolean 'passed' key."""
        from vetinari.benchmarks import run_ci_benchmarks

        result = run_ci_benchmarks()
        for entry in result["scores"]:
            assert "passed" in entry, f"Missing 'passed' in {entry}"
            assert isinstance(entry["passed"], bool)

    def test_score_names_are_unique(self) -> None:
        """All score names must be distinct within a single run."""
        from vetinari.benchmarks import run_ci_benchmarks

        result = run_ci_benchmarks()
        names = [s["name"] for s in result["scores"]]
        assert len(names) == len(set(names)), f"Duplicate score names: {names}"

    def test_expected_benchmark_names_present(self) -> None:
        """The three mandatory CI benchmark names must be present."""
        from vetinari.benchmarks import run_ci_benchmarks

        result = run_ci_benchmarks()
        names = {s["name"] for s in result["scores"]}
        expected = {"plan_latency_ms", "decomposition_score", "token_optimization_ratio"}
        assert expected.issubset(names), f"Missing benchmarks: {expected - names}"
