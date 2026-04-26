"""Tests for vetinari/analytics/forecasting.py (Phase 5)"""

from __future__ import annotations

import pytest

from vetinari.analytics.forecasting import (
    ForecastRequest,
    _ols,
    _rmse,
    _stddev,
    get_forecaster,
    reset_forecaster,
)
from vetinari.analytics.sla import _percentile
from vetinari.exceptions import ConfigurationError


def _fc():
    reset_forecaster()
    return get_forecaster()


class TestMathHelpers:
    def test_ols_slope(self) -> None:
        # y = 2x: slope should be 2, intercept 0
        y = [0.0, 2.0, 4.0, 6.0, 8.0]
        slope, intercept = _ols(y)
        assert slope == pytest.approx(2.0, abs=1e-5)
        assert intercept == pytest.approx(0.0, abs=1e-5)

    def test_ols_flat_line(self) -> None:
        y = [5.0] * 10
        slope, intercept = _ols(y)
        assert slope == pytest.approx(0.0, abs=1e-5)
        assert intercept == pytest.approx(5.0, abs=1e-5)

    def test_rmse_zero(self) -> None:
        y = [1.0, 2.0, 3.0]
        assert _rmse(y, y) == pytest.approx(0.0)

    def test_rmse_nonzero(self) -> None:
        assert _rmse([1.0, 2.0], [3.0, 4.0]) > 0

    def test_percentile_median(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(vals, 50) == pytest.approx(3.0)

    def test_stddev_constant(self) -> None:
        assert _stddev([5.0] * 10) == pytest.approx(0.0)


class TestForecasterSingleton:
    def setup_method(self) -> None:
        reset_forecaster()

    def teardown_method(self) -> None:
        reset_forecaster()

    def test_same_instance(self) -> None:
        assert get_forecaster() is get_forecaster()

    def test_reset_new_instance(self) -> None:
        a = get_forecaster()
        reset_forecaster()
        assert a is not get_forecaster()


class TestIngestion:
    def setup_method(self) -> None:
        reset_forecaster()

    def teardown_method(self) -> None:
        reset_forecaster()

    def test_ingest_single(self) -> None:
        fc = _fc()
        fc.ingest("lat", 100.0)
        assert len(fc.get_history("lat")) == 1

    def test_ingest_many(self) -> None:
        fc = _fc()
        fc.ingest_many("lat", [1.0, 2.0, 3.0])
        assert len(fc.get_history("lat")) == 3

    def test_list_metrics(self) -> None:
        fc = _fc()
        fc.ingest("a", 1.0)
        fc.ingest("b", 2.0)
        assert "a" in fc.list_metrics()
        assert "b" in fc.list_metrics()

    def test_clear(self) -> None:
        fc = _fc()
        fc.ingest_many("lat", [1.0] * 10)
        fc.clear()
        assert fc.list_metrics() == []


class TestSMAForecast:
    def setup_method(self) -> None:
        reset_forecaster()

    def teardown_method(self) -> None:
        reset_forecaster()

    def test_sma_constant_series(self) -> None:
        fc = _fc()
        fc.ingest_many("lat", [100.0] * 20)
        r = fc.forecast(ForecastRequest(metric="lat", horizon=3, method="sma"))
        assert len(r.predictions) == 3
        for p in r.predictions:
            assert p == pytest.approx(100.0, abs=1e-2)

    def test_sma_result_fields(self) -> None:
        fc = _fc()
        fc.ingest_many("lat", [50.0] * 10)
        r = fc.forecast(ForecastRequest(metric="lat", horizon=2, method="sma"))
        assert r.forecast_method_used == "sma"
        assert r.horizon == 2
        assert len(r.confidence_lo) == 2
        assert len(r.confidence_hi) == 2


class TestExpSmoothingForecast:
    def setup_method(self) -> None:
        reset_forecaster()

    def teardown_method(self) -> None:
        reset_forecaster()

    def test_es_on_rising_series(self) -> None:
        fc = _fc()
        for i in range(20):
            fc.ingest("val", float(i * 10))
        r = fc.forecast(ForecastRequest(metric="val", horizon=3, method="exp_smoothing"))
        assert len(r.predictions) == 3
        # All predictions should be positive
        for p in r.predictions:
            assert p > 0

    def test_es_rmse_set(self) -> None:
        fc = _fc()
        fc.ingest_many("val", [100.0] * 15)
        r = fc.forecast(ForecastRequest(metric="val", horizon=1, method="exp_smoothing"))
        assert isinstance(r.rmse, float)


class TestLinearTrendForecast:
    def setup_method(self) -> None:
        reset_forecaster()

    def teardown_method(self) -> None:
        reset_forecaster()

    def test_linear_trend_slope(self) -> None:
        fc = _fc()
        # Perfect linear: y = 10*x
        fc.ingest_many("lat", [float(i * 10) for i in range(20)])
        r = fc.forecast(ForecastRequest(metric="lat", horizon=5, method="linear_trend"))
        assert r.trend_slope == pytest.approx(10.0, abs=0.1)
        assert len(r.predictions) == 5

    def test_predictions_increase_for_rising_series(self) -> None:
        fc = _fc()
        fc.ingest_many("lat", [float(i) for i in range(20)])
        r = fc.forecast(ForecastRequest(metric="lat", horizon=3, method="linear_trend"))
        assert r.predictions[-1] > r.predictions[0]

    def test_constant_series_slope_near_zero(self) -> None:
        fc = _fc()
        fc.ingest_many("lat", [50.0] * 20)
        r = fc.forecast(ForecastRequest(metric="lat", horizon=3, method="linear_trend"))
        assert r.trend_slope == pytest.approx(0.0, abs=0.01)


class TestSeasonalForecast:
    def setup_method(self) -> None:
        reset_forecaster()

    def teardown_method(self) -> None:
        reset_forecaster()

    def test_seasonal_returns_correct_horizon(self) -> None:
        fc = _fc()
        import math

        # Build a periodic signal
        data = [50.0 + 10.0 * math.sin(2 * math.pi * i / 7) for i in range(28)]
        fc.ingest_many("s", data)
        r = fc.forecast(ForecastRequest(metric="s", horizon=7, method="seasonal", period=7))
        assert len(r.predictions) == 7
        assert r.forecast_method_used == "seasonal"

    def test_seasonal_fallback_with_little_data(self) -> None:
        fc = _fc()
        fc.ingest_many("s", [1.0, 2.0, 3.0])
        r = fc.forecast(ForecastRequest(metric="s", horizon=3, method="seasonal", period=7))
        assert len(r.predictions) == 3


class TestInsufficientHistory:
    def setup_method(self) -> None:
        reset_forecaster()

    def teardown_method(self) -> None:
        reset_forecaster()

    def test_empty_history_returns_zeros(self) -> None:
        fc = _fc()
        r = fc.forecast(ForecastRequest(metric="nonexistent", horizon=3))
        assert r.predictions == [0.0, 0.0, 0.0]

    def test_one_point_repeats(self) -> None:
        fc = _fc()
        fc.ingest("lat", 42.0)
        r = fc.forecast(ForecastRequest(metric="lat", horizon=2, method="linear_trend"))
        assert r.predictions == [42.0, 42.0]


class TestUnknownMethod:
    def setup_method(self) -> None:
        reset_forecaster()

    def teardown_method(self) -> None:
        reset_forecaster()

    def test_unknown_method_raises(self) -> None:
        fc = _fc()
        fc.ingest_many("lat", [1.0] * 10)
        with pytest.raises(ConfigurationError):
            fc.forecast(ForecastRequest(metric="lat", horizon=3, method="magic"))


class TestCapacityHelpers:
    def setup_method(self) -> None:
        reset_forecaster()

    def teardown_method(self) -> None:
        reset_forecaster()

    def test_will_exceed_rising(self) -> None:
        fc = _fc()
        fc.ingest_many("lat", [float(i * 10) for i in range(20)])
        assert fc.will_exceed("lat", threshold=300.0, horizon=20)

    def test_will_not_exceed_flat(self) -> None:
        fc = _fc()
        fc.ingest_many("lat", [50.0] * 20)
        assert not fc.will_exceed("lat", threshold=1000.0, horizon=10)

    def test_steps_until_threshold(self) -> None:
        fc = _fc()
        fc.ingest_many("lat", [float(i * 10) for i in range(20)])
        steps = fc.steps_until_threshold("lat", threshold=300.0, horizon=50)
        assert steps is not None
        assert steps > 0

    def test_steps_until_threshold_never(self) -> None:
        fc = _fc()
        fc.ingest_many("lat", [50.0] * 20)
        steps = fc.steps_until_threshold("lat", threshold=1_000_000.0, horizon=10)
        assert steps is None


class TestGetStats:
    def setup_method(self) -> None:
        reset_forecaster()

    def teardown_method(self) -> None:
        reset_forecaster()

    def test_stats(self) -> None:
        fc = _fc()
        fc.ingest_many("a", [1.0] * 5)
        fc.ingest_many("b", [2.0] * 3)
        stats = fc.get_stats()
        assert stats["tracked_metrics"] == 2
        assert stats["history_sizes"]["a"] == 5
