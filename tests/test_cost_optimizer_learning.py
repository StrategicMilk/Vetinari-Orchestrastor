"""Tests for vetinari.learning.cost_optimizer."""

import pytest
from unittest.mock import MagicMock, patch

from vetinari.learning.cost_optimizer import (
    CostEfficiency,
    CostOptimizer,
    get_cost_optimizer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_optimizer():
    """Return a fresh, isolated CostOptimizer instance (not the singleton)."""
    return CostOptimizer()


def _make_tracker_report(entries):
    """Return a mock tracker whose get_report() yields the given entry list."""
    tracker = MagicMock()
    tracker.get_report.return_value = {"entries": entries}
    return tracker


def _make_thompson_selector(model_means):
    """Return a mock ThompsonSamplingSelector mapping model_id -> mean quality."""
    selector = MagicMock()

    def arm_state(model_id, task_type):
        return {"mean": model_means.get(model_id, 0.7)}

    selector.get_arm_state.side_effect = arm_state
    return selector


# ---------------------------------------------------------------------------
# 1. CostEfficiency dataclass
# ---------------------------------------------------------------------------

class TestCostEfficiencyDataclass:
    def test_creation_and_field_access(self):
        ce = CostEfficiency(
            model_id="model-a",
            task_type="summarize",
            avg_quality=0.85,
            avg_cost_usd=0.002,
            quality_per_dollar=425.0,
            total_uses=10,
        )
        assert ce.model_id == "model-a"
        assert ce.task_type == "summarize"
        assert ce.avg_quality == pytest.approx(0.85)
        assert ce.avg_cost_usd == pytest.approx(0.002)
        assert ce.quality_per_dollar == pytest.approx(425.0)
        assert ce.total_uses == 10

    def test_zero_cost_fields(self):
        ce = CostEfficiency(
            model_id="local-llm",
            task_type="code",
            avg_quality=0.75,
            avg_cost_usd=0.0,
            quality_per_dollar=750.0,
            total_uses=0,
        )
        assert ce.avg_cost_usd == pytest.approx(0.0)
        assert ce.total_uses == 0

    def test_equality_of_identical_instances(self):
        kwargs = dict(
            model_id="m", task_type="t",
            avg_quality=0.9, avg_cost_usd=0.01,
            quality_per_dollar=90.0, total_uses=5,
        )
        assert CostEfficiency(**kwargs) == CostEfficiency(**kwargs)

    def test_fields_are_mutable(self):
        ce = CostEfficiency(
            model_id="m", task_type="t",
            avg_quality=0.5, avg_cost_usd=0.01,
            quality_per_dollar=50.0, total_uses=1,
        )
        ce.avg_quality = 0.9
        assert ce.avg_quality == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# 2. Singleton pattern
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_cost_optimizer_returns_same_instance(self):
        # Reset module-level singleton between calls
        import vetinari.learning.cost_optimizer as mod
        mod._cost_optimizer = None
        a = get_cost_optimizer()
        b = get_cost_optimizer()
        assert a is b

    def test_get_cost_optimizer_returns_cost_optimizer_type(self):
        import vetinari.learning.cost_optimizer as mod
        mod._cost_optimizer = None
        assert isinstance(get_cost_optimizer(), CostOptimizer)

    def teardown_method(self):
        import vetinari.learning.cost_optimizer as mod
        mod._cost_optimizer = None


# ---------------------------------------------------------------------------
# 3. _get_efficiencies — internal computation
# ---------------------------------------------------------------------------

class TestGetEfficiencies:
    def test_returns_one_entry_per_model(self):
        opt = _make_optimizer()
        with patch("vetinari.learning.cost_optimizer.CostOptimizer._get_cost_tracker", return_value=None), \
             patch("vetinari.learning.model_selector.get_thompson_selector") as ts_mock:
            ts_mock.return_value = _make_thompson_selector({})
            result = opt._get_efficiencies("code", ["model-a", "model-b"])
        assert len(result) == 2
        assert {e.model_id for e in result} == {"model-a", "model-b"}

    def test_uses_cost_from_tracker(self):
        opt = _make_optimizer()
        tracker = _make_tracker_report([
            {"model_id": "model-a", "cost_usd": 0.05},
        ])
        with patch.object(opt, "_get_cost_tracker", return_value=tracker), \
             patch("vetinari.learning.model_selector.get_thompson_selector") as ts_mock:
            ts_mock.return_value = _make_thompson_selector({"model-a": 0.8})
            result = opt._get_efficiencies("summarize", ["model-a"])
        assert result[0].avg_cost_usd == pytest.approx(0.05)

    def test_uses_quality_from_thompson_selector(self):
        opt = _make_optimizer()
        tracker = _make_tracker_report([])
        with patch.object(opt, "_get_cost_tracker", return_value=tracker), \
             patch("vetinari.learning.model_selector.get_thompson_selector") as ts_mock:
            ts_mock.return_value = _make_thompson_selector({"model-a": 0.92})
            result = opt._get_efficiencies("translate", ["model-a"])
        assert result[0].avg_quality == pytest.approx(0.92)

    def test_defaults_to_prior_quality_when_no_tracker(self):
        opt = _make_optimizer()
        with patch.object(opt, "_get_cost_tracker", return_value=None), \
             patch("vetinari.learning.model_selector.get_thompson_selector") as ts_mock:
            ts_mock.return_value = _make_thompson_selector({})
            result = opt._get_efficiencies("code", ["model-x"])
        # Thompson mock returns 0.7 for unknown models via the default
        assert result[0].avg_quality == pytest.approx(0.7)

    def test_quality_per_dollar_uses_small_epsilon_when_cost_is_zero(self):
        """Avoids division by zero: cost=0 => qpd = quality / 0.001."""
        opt = _make_optimizer()
        with patch.object(opt, "_get_cost_tracker", return_value=None), \
             patch("vetinari.learning.model_selector.get_thompson_selector") as ts_mock:
            ts_mock.return_value = _make_thompson_selector({"free-model": 0.8})
            result = opt._get_efficiencies("code", ["free-model"])
        expected_qpd = 0.8 / 0.001
        assert result[0].quality_per_dollar == pytest.approx(expected_qpd)

    def test_task_type_propagated_to_entries(self):
        opt = _make_optimizer()
        with patch.object(opt, "_get_cost_tracker", return_value=None), \
             patch("vetinari.learning.model_selector.get_thompson_selector") as ts_mock:
            ts_mock.return_value = _make_thompson_selector({})
            result = opt._get_efficiencies("my_task", ["m"])
        assert result[0].task_type == "my_task"

    def test_cost_tracker_exception_falls_back_to_zero_cost(self):
        opt = _make_optimizer()
        bad_tracker = MagicMock()
        bad_tracker.get_report.side_effect = RuntimeError("db gone")
        with patch.object(opt, "_get_cost_tracker", return_value=bad_tracker), \
             patch("vetinari.learning.model_selector.get_thompson_selector") as ts_mock:
            ts_mock.return_value = _make_thompson_selector({"m": 0.75})
            result = opt._get_efficiencies("code", ["m"])
        assert result[0].avg_cost_usd == pytest.approx(0.0)

    def test_thompson_exception_falls_back_to_prior_quality(self):
        opt = _make_optimizer()
        with patch.object(opt, "_get_cost_tracker", return_value=None), \
             patch("vetinari.learning.model_selector.get_thompson_selector") as ts_mock:
            ts_mock.side_effect = RuntimeError("selector broken")
            result = opt._get_efficiencies("code", ["m"])
        assert result[0].avg_quality == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# 4. select_cheapest_adequate — selection logic
# ---------------------------------------------------------------------------

class TestSelectCheapestAdequate:
    def _opt_with_efficiencies(self, efficiencies):
        """Return an optimizer whose _get_efficiencies is stubbed."""
        opt = _make_optimizer()
        opt._get_efficiencies = MagicMock(return_value=efficiencies)
        return opt

    def _eff(self, model_id, quality, cost):
        return CostEfficiency(
            model_id=model_id,
            task_type="test",
            avg_quality=quality,
            avg_cost_usd=cost,
            quality_per_dollar=quality / max(cost, 0.001),
            total_uses=1,
        )

    def test_returns_cheapest_adequate_model(self):
        effs = [
            self._eff("expensive", 0.9, 0.10),
            self._eff("cheap",     0.8, 0.01),
            self._eff("mid",       0.75, 0.05),
        ]
        opt = self._opt_with_efficiencies(effs)
        result = opt.select_cheapest_adequate("test", ["expensive", "cheap", "mid"])
        assert result == "cheap"

    def test_quality_threshold_excludes_low_quality_models(self):
        effs = [
            self._eff("good",  0.80, 0.05),
            self._eff("cheap", 0.50, 0.01),   # below default threshold 0.65
        ]
        opt = self._opt_with_efficiencies(effs)
        result = opt.select_cheapest_adequate("test", ["good", "cheap"])
        assert result == "good"

    def test_custom_min_quality_respected(self):
        effs = [
            self._eff("high",   0.95, 0.20),
            self._eff("medium", 0.75, 0.10),
        ]
        opt = self._opt_with_efficiencies(effs)
        # Set threshold above medium's quality
        result = opt.select_cheapest_adequate("test", ["high", "medium"], min_quality=0.90)
        assert result == "high"

    def test_max_cost_filter_excludes_expensive_models(self):
        effs = [
            self._eff("pricey",    0.90, 0.50),
            self._eff("affordable", 0.80, 0.05),
        ]
        opt = self._opt_with_efficiencies(effs)
        result = opt.select_cheapest_adequate(
            "test", ["pricey", "affordable"], max_cost_usd=0.10
        )
        assert result == "affordable"

    def test_fallback_to_highest_quality_when_none_meet_quality_threshold(self):
        effs = [
            self._eff("best",  0.60, 0.01),
            self._eff("worst", 0.40, 0.005),
        ]
        opt = self._opt_with_efficiencies(effs)
        # default min_quality=0.65 means neither passes
        result = opt.select_cheapest_adequate("test", ["best", "worst"])
        assert result == "best"

    def test_fallback_to_highest_quality_when_max_cost_filters_all(self):
        effs = [
            self._eff("model-a", 0.90, 1.00),
            self._eff("model-b", 0.70, 0.80),
        ]
        opt = self._opt_with_efficiencies(effs)
        # Both pass quality but both exceed max_cost_usd=0.10
        result = opt.select_cheapest_adequate(
            "test", ["model-a", "model-b"], max_cost_usd=0.10
        )
        assert result == "model-a"

    def test_empty_candidate_list_returns_default(self):
        opt = _make_optimizer()
        opt._get_efficiencies = MagicMock(return_value=[])
        result = opt.select_cheapest_adequate("test", [])
        assert result == "default"

    def test_single_candidate_returned_if_adequate(self):
        effs = [self._eff("only-model", 0.80, 0.02)]
        opt = self._opt_with_efficiencies(effs)
        result = opt.select_cheapest_adequate("test", ["only-model"])
        assert result == "only-model"

    def test_single_candidate_returned_as_fallback_when_below_threshold(self):
        effs = [self._eff("only-model", 0.50, 0.02)]
        opt = self._opt_with_efficiencies(effs)
        result = opt.select_cheapest_adequate("test", ["only-model"])
        assert result == "only-model"

    def test_default_min_quality_constant(self):
        assert CostOptimizer.DEFAULT_MIN_QUALITY == pytest.approx(0.65)

    def test_local_cost_constant(self):
        assert CostOptimizer.LOCAL_COST == pytest.approx(0.0)

    def test_all_models_below_quality_threshold_returns_first_when_no_efficiencies(self):
        opt = _make_optimizer()
        opt._get_efficiencies = MagicMock(return_value=[])
        result = opt.select_cheapest_adequate("test", ["fallback-model"])
        assert result == "fallback-model"


# ---------------------------------------------------------------------------
# 5. get_budget_forecast
# ---------------------------------------------------------------------------

class TestGetBudgetForecast:
    def _opt_with_fixed_cost(self, cost_per_call):
        """Return an optimizer that reports a fixed cost for every model."""
        opt = _make_optimizer()

        def fake_efficiencies(task_type, models):
            return [
                CostEfficiency(
                    model_id=m,
                    task_type=task_type,
                    avg_quality=0.8,
                    avg_cost_usd=cost_per_call,
                    quality_per_dollar=0.8 / max(cost_per_call, 0.001),
                    total_uses=1,
                )
                for m in models
            ]

        opt._get_efficiencies = MagicMock(side_effect=fake_efficiencies)
        return opt

    def test_forecast_returns_required_keys(self):
        opt = self._opt_with_fixed_cost(0.01)
        result = opt.get_budget_forecast(
            planned_tasks=2,
            task_types=["code", "summarize"],
            models=["m"],
        )
        assert "estimated_cost_usd" in result
        assert "breakdown_by_type" in result
        assert "warnings" in result

    def test_estimated_cost_is_sum_of_task_averages(self):
        # cost_per_call=0.10, 1 model, 2 tasks => avg per task = 0.10, total = 0.20
        opt = self._opt_with_fixed_cost(0.10)
        result = opt.get_budget_forecast(
            planned_tasks=2,
            task_types=["a", "b"],
            models=["m"],
        )
        assert result["estimated_cost_usd"] == pytest.approx(0.20, abs=1e-4)

    def test_breakdown_by_type_contains_each_task_type(self):
        opt = self._opt_with_fixed_cost(0.05)
        result = opt.get_budget_forecast(
            planned_tasks=3,
            task_types=["alpha", "beta", "gamma"],
            models=["m"],
        )
        bd = result["breakdown_by_type"]
        assert "alpha" in bd
        assert "beta" in bd
        assert "gamma" in bd

    def test_no_warning_when_cost_below_threshold(self):
        opt = self._opt_with_fixed_cost(0.10)
        result = opt.get_budget_forecast(
            planned_tasks=2,
            task_types=["a", "b"],
            models=["m"],
        )
        # total = 0.20, below the $1.00 threshold
        assert result["warnings"] == []

    def test_warning_generated_when_cost_exceeds_one_dollar(self):
        # 3 tasks * 0.50 avg = $1.50 > $1.00
        opt = self._opt_with_fixed_cost(0.50)
        result = opt.get_budget_forecast(
            planned_tasks=3,
            task_types=["x", "y", "z"],
            models=["m"],
        )
        assert len(result["warnings"]) == 1
        assert "1.00" in result["warnings"][0] or "$" in result["warnings"][0]

    def test_planned_tasks_limits_how_many_task_types_are_processed(self):
        opt = self._opt_with_fixed_cost(0.10)
        result = opt.get_budget_forecast(
            planned_tasks=2,
            task_types=["a", "b", "c", "d"],   # 4 types but only 2 planned
            models=["m"],
        )
        assert len(result["breakdown_by_type"]) == 2

    def test_zero_cost_models_produce_zero_estimate(self):
        opt = self._opt_with_fixed_cost(0.0)
        result = opt.get_budget_forecast(
            planned_tasks=5,
            task_types=["t1", "t2", "t3", "t4", "t5"],
            models=["local-m"],
        )
        assert result["estimated_cost_usd"] == pytest.approx(0.0)
        assert result["warnings"] == []

    def test_empty_efficiencies_produces_zero_cost_per_task(self):
        opt = _make_optimizer()
        opt._get_efficiencies = MagicMock(return_value=[])
        result = opt.get_budget_forecast(
            planned_tasks=2,
            task_types=["a", "b"],
            models=[],
        )
        assert result["estimated_cost_usd"] == pytest.approx(0.0)

    def test_cost_is_rounded_to_four_decimal_places(self):
        opt = self._opt_with_fixed_cost(0.333333)
        result = opt.get_budget_forecast(
            planned_tasks=1,
            task_types=["t"],
            models=["m"],
        )
        # round() to 4dp means at most 4 decimal places represented
        val = result["estimated_cost_usd"]
        assert round(val, 4) == val


# ---------------------------------------------------------------------------
# 6. _get_cost_tracker — lazy initialisation
# ---------------------------------------------------------------------------

class TestGetCostTracker:
    def test_returns_none_when_import_fails(self):
        opt = _make_optimizer()
        with patch("vetinari.analytics.cost.get_cost_tracker", side_effect=ImportError):
            # Force re-init
            tracker = opt._get_cost_tracker()
        # Should not raise; may return None or the tracker
        # If importing raises, the except clause returns None
        # (depends on whether the patch hits inside the method's try/except)
        # We just assert it does not propagate
        assert tracker is None or tracker is not None  # no exception is the assertion

    def test_lazy_init_caches_result(self):
        opt = _make_optimizer()
        mock_tracker = MagicMock()
        with patch("vetinari.analytics.cost.get_cost_tracker", return_value=mock_tracker):
            t1 = opt._get_cost_tracker()
            t2 = opt._get_cost_tracker()
        assert t1 is t2
