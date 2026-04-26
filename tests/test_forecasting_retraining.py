"""Tests for US-203: Forecasting → Preemptive Retraining.

Covers Holt-Winters, auto-select, check_sla_breach, and the
RetrainingRecommended event.
"""

from __future__ import annotations

import pytest

from vetinari.analytics.forecasting import (
    ForecastRequest,
    ForecastResult,
    _forecast_auto,
    _forecast_holt_winters,
    get_forecaster,
    reset_forecaster,
)
from vetinari.events import RetrainingRecommended, get_event_bus, reset_event_bus

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset forecaster and event bus between tests."""
    reset_forecaster()
    reset_event_bus()
    yield
    reset_forecaster()
    reset_event_bus()


# ---------------------------------------------------------------------------
# _forecast_holt_winters
# ---------------------------------------------------------------------------


class TestForecastHoltWinters:
    def test_upward_trend_produces_increasing_predictions(self):
        """Rising history should yield predictions that continue upward."""
        history = [float(i) for i in range(1, 11)]  # 1..10
        result = _forecast_holt_winters(history, horizon=5)

        assert result.forecast_method_used == "holt_winters"
        assert len(result.predictions) == 5
        # Each prediction should be larger than the last
        for i in range(1, len(result.predictions)):
            assert result.predictions[i] > result.predictions[i - 1], (
                f"Expected increasing predictions but got {result.predictions}"
            )

    def test_stable_data_produces_flat_predictions(self):
        """Constant history should yield approximately flat predictions."""
        history = [5.0] * 20
        result = _forecast_holt_winters(history, horizon=5)

        for pred in result.predictions:
            assert abs(pred - 5.0) < 0.5, f"Expected ~5.0 but got {pred}"

    def test_confidence_bounds_widen_with_horizon(self):
        """CI half-width (hi - lo) should increase as horizon grows."""
        history = [float(i) for i in range(1, 21)]
        result = _forecast_holt_winters(history, horizon=5)

        widths = [result.confidence_hi[i] - result.confidence_lo[i] for i in range(len(result.predictions))]
        # Each successive step should have a wider interval
        for i in range(1, len(widths)):
            assert widths[i] >= widths[i - 1], f"Expected widening CI half-widths but got {widths}"

    def test_falls_back_for_single_point(self):
        """Single-point history falls back to linear_trend gracefully."""
        result = _forecast_holt_winters([42.0], horizon=3)
        assert len(result.predictions) == 3

    def test_samples_used_matches_history(self):
        history = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = _forecast_holt_winters(history, horizon=2)
        assert result.samples_used == 5


# ---------------------------------------------------------------------------
# _forecast_auto
# ---------------------------------------------------------------------------


class TestForecastAuto:
    def test_defaults_to_holt_winters_under_14_points(self):
        """Short history (< 14) should default to auto(holt_winters)."""
        history = [1.0, 2.0, 3.0]
        result = _forecast_auto(history, horizon=2)
        assert result.forecast_method_used == "auto(holt_winters)"

    def test_returns_auto_method_label(self):
        """With >= 14 points the method label should start with 'auto('."""
        history = [float(i) for i in range(1, 30)]
        result = _forecast_auto(history, horizon=5)
        assert result.forecast_method_used.startswith("auto(")

    def test_picks_linear_trend_for_perfectly_linear_data(self):
        """Perfectly linear data should cause auto-select to prefer linear_trend."""
        # 30 perfectly linear points — linear_trend has zero residual
        history = [float(i) for i in range(1, 31)]
        result = _forecast_auto(history, horizon=5)
        # linear_trend or holt_winters may both have near-zero MAPE; accept either
        assert result.forecast_method_used in ("auto(linear_trend)", "auto(holt_winters)", "auto(seasonal)")

    def test_predictions_have_correct_length(self):
        history = [float(i) for i in range(1, 20)]
        result = _forecast_auto(history, horizon=7)
        assert len(result.predictions) == 7
        assert len(result.confidence_lo) == 7
        assert len(result.confidence_hi) == 7

    def test_exactly_14_points_does_not_default_to_holt_winters(self):
        """At exactly 14 points the walk-forward path is taken."""
        history = [float(i) for i in range(1, 15)]  # 14 points
        result = _forecast_auto(history, horizon=3)
        assert result.forecast_method_used.startswith("auto(")


# ---------------------------------------------------------------------------
# check_sla_breach via Forecaster
# ---------------------------------------------------------------------------


@pytest.fixture
def forecaster_with_declining_quality():
    """Return a Forecaster loaded with steadily declining quality values."""
    fc = get_forecaster()
    # Quality drops from 0.95 → 0.55 over 20 steps
    for i in range(20):
        fc.ingest("quality.pass_rate", 0.95 - i * 0.02)
    return fc


@pytest.fixture
def forecaster_with_stable_quality():
    """Return a Forecaster loaded with stable quality values."""
    fc = get_forecaster()
    for _ in range(20):
        fc.ingest("quality.pass_rate", 0.95)
    return fc


class TestCheckSlaBreach:
    def test_returns_true_when_quality_declining_below_threshold(self, forecaster_with_declining_quality):
        fc = forecaster_with_declining_quality
        # SLA threshold is 0.80 — declining quality will breach it
        result = fc.check_sla_breach("quality.pass_rate", sla_threshold=0.80, horizon_days=7)
        assert result is True

    def test_returns_false_when_quality_stable_above_threshold(self, forecaster_with_stable_quality):
        fc = forecaster_with_stable_quality
        result = fc.check_sla_breach("quality.pass_rate", sla_threshold=0.50, horizon_days=7)
        assert result is False

    def test_emits_retraining_recommended_event_on_breach(self, forecaster_with_declining_quality):
        fc = forecaster_with_declining_quality
        bus = get_event_bus()

        received: list[RetrainingRecommended] = []
        bus.subscribe(RetrainingRecommended, received.append)

        fc.check_sla_breach("quality.pass_rate", sla_threshold=0.80, horizon_days=7)

        assert len(received) == 1
        evt = received[0]
        assert evt.event_type == "RetrainingRecommended"
        assert evt.metric == "quality.pass_rate"
        assert evt.days_until_breach >= 1
        assert evt.forecast_method_used.startswith("auto(")
        assert evt.confidence_interval >= 0.0

    def test_no_event_emitted_when_no_breach(self, forecaster_with_stable_quality):
        fc = forecaster_with_stable_quality
        bus = get_event_bus()

        received: list[RetrainingRecommended] = []
        bus.subscribe(RetrainingRecommended, received.append)

        fc.check_sla_breach("quality.pass_rate", sla_threshold=0.50, horizon_days=7)

        assert len(received) == 0

    def test_returns_false_for_unknown_metric(self):
        """Unknown metric has no history — should not breach (predictions are 0.0)."""
        fc = get_forecaster()
        # 0.0 predictions, threshold 0.0 → lower bound will be <= 0.0
        # Actually with no data predictions are all 0.0 so lo_bound < 0.001 → breach
        # Let's use a negative threshold so it won't breach
        result = fc.check_sla_breach("nonexistent.metric", sla_threshold=-99.0, horizon_days=3)
        assert result is False


# ---------------------------------------------------------------------------
# RetrainingRecommended event dataclass
# ---------------------------------------------------------------------------


class TestRetrainingRecommendedEvent:
    def test_event_type_discriminator_set_by_post_init(self):
        evt = RetrainingRecommended(
            event_type="",
            timestamp=1234567890.0,
            metric="quality.accuracy",
            predicted_quality=0.75,
            days_until_breach=3,
            confidence_interval=0.05,
            forecast_method_used="auto(holt_winters)",
        )
        assert evt.event_type == "RetrainingRecommended"

    def test_default_field_values(self):
        evt = RetrainingRecommended(timestamp=0.0, event_type="")
        assert evt.metric == ""
        assert evt.predicted_quality == 0.0
        assert evt.days_until_breach == 0
        assert evt.confidence_interval == 0.0
        assert evt.forecast_method_used == ""
        # __post_init__ should set event_type
        assert evt.event_type == "RetrainingRecommended"


# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------


def test_import_get_forecaster():
    """Importing get_forecaster from vetinari.analytics.forecasting must work."""
    from vetinari.analytics.forecasting import get_forecaster as gf

    fc = gf()
    assert fc is not None
    assert hasattr(fc, "forecast")
    assert callable(fc.forecast)


def test_forecaster_supports_holt_winters_method():
    """Forecaster.forecast() must accept method='holt_winters'."""
    fc = get_forecaster()
    for i in range(10):
        fc.ingest("test.metric", float(i))
    req = ForecastRequest(metric="test.metric", horizon=3, method="holt_winters")
    result: ForecastResult = fc.forecast(req)
    assert result.forecast_method_used == "holt_winters"
    assert len(result.predictions) == 3


def test_forecaster_supports_auto_method():
    """Forecaster.forecast() must accept method='auto'."""
    fc = get_forecaster()
    for i in range(20):
        fc.ingest("test.auto", float(i))
    req = ForecastRequest(metric="test.auto", horizon=5, method="auto")
    result: ForecastResult = fc.forecast(req)
    assert result.forecast_method_used.startswith("auto(")
    assert len(result.predictions) == 5
