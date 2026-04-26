"""Forecasting & Capacity Planning — vetinari.analytics.forecasting  (Phase 5).

Provides lightweight, dependency-free time-series forecasting suitable for
short-horizon capacity planning:

    SimpleMovingAverage (SMA)   — mean of the last N points.
    ExponentialSmoothing (ES)   — Holt single exponential smoothing.
    LinearTrend                 — ordinary least-squares linear extrapolation.
    SeasonalDecomposition       — additive trend + weekly seasonality.

All methods operate on plain Python lists / deques.

Usage
-----
    from vetinari.analytics.forecasting import get_forecaster, ForecastRequest

    fc = get_forecaster()
    fc.ingest("adapter.latency", 120.0)   # call repeatedly as data arrives
    # ...

    result = fc.forecast(ForecastRequest(
        metric="adapter.latency",
        horizon=5,
        method="linear_trend",
    ))
    logger.debug(result.predictions)   # list of 5 forecasted values
    logger.debug(result.trend_slope)
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from vetinari.exceptions import ConfigurationError
from vetinari.utils.serialization import dataclass_to_dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / Result
# ---------------------------------------------------------------------------


@dataclass
class ForecastRequest:
    """Parameters for a forecast call."""

    metric: str
    horizon: int = 5  # number of future steps to predict
    method: str = "linear_trend"  # sma | exp_smoothing | linear_trend | seasonal
    alpha: float = 0.3  # smoothing factor for ES
    period: int = 7  # season length for seasonal decomposition

    def __repr__(self) -> str:
        return f"ForecastRequest(metric={self.metric!r}, horizon={self.horizon!r}, method={self.method!r})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize this forecast request to a plain dictionary for JSON export.

        Returns:
            Dictionary containing the metric name, horizon, method, and tuning parameters.
        """
        return cast(dict[str, Any], dataclass_to_dict(self))


@dataclass
class ForecastResult:
    """Output of a forecast operation."""

    metric: str
    forecast_method_used: str
    horizon: int
    predictions: list[float]  # noqa: VET220 — fixed-length horizon output, not a growing buffer
    confidence_lo: list[float]  # noqa: VET220 — fixed-length horizon output, not a growing buffer
    confidence_hi: list[float]  # noqa: VET220 — fixed-length horizon output, not a growing buffer
    trend_slope: float = 0.0  # rate of change per step (linear_trend only)
    rmse: float = 0.0  # in-sample root mean squared error
    samples_used: int = 0

    def __repr__(self) -> str:
        return (
            f"ForecastResult(metric={self.metric!r}, forecast_method_used={self.forecast_method_used!r}, "
            f"horizon={self.horizon!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this forecast result to a plain dictionary for JSON export.

        Returns:
            Dictionary containing predictions, confidence bounds, trend slope, and RMSE.
        """
        return cast(dict[str, Any], dataclass_to_dict(self))


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _ols(y: list[float]) -> tuple[float, float]:
    """Return (slope, intercept) of the OLS line through enumerate(y)."""
    n = len(y)
    sx = n * (n - 1) / 2  # sum of 0..n-1
    sx2 = n * (n - 1) * (2 * n - 1) / 6
    sy = sum(y)
    sxy = sum(i * v for i, v in enumerate(y))
    denom = n * sx2 - sx * sx
    if denom == 0:
        return 0.0, sum(y) / n if y else 0.0
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def _rmse(actual: list[float], predicted: list[float]) -> float:
    if not actual:
        return 0.0
    n = min(len(actual), len(predicted))
    return math.sqrt(sum((actual[i] - predicted[i]) ** 2 for i in range(n)) / n)


def _stddev(vals: list[float]) -> float:
    """Delegate to canonical stddev with sample correction."""
    from vetinari.utils.math_helpers import stddev

    return stddev(vals, sample=True)


def _conf_bounds(preds: list[float], std: float, z: float = 1.28) -> tuple[list[float], list[float]]:
    lo = [p - z * std for p in preds]
    hi = [p + z * std for p in preds]
    return lo, hi


# ---------------------------------------------------------------------------
# Forecasting methods (pure functions)
# ---------------------------------------------------------------------------


def _forecast_sma(history: list[float], horizon: int, window: int = 10) -> ForecastResult:
    w = history[-window:] if len(history) >= window else history
    pred = sum(w) / len(w) if w else 0.0
    preds = [pred] * horizon
    std = _stddev(history)
    lo, hi = _conf_bounds(preds, std)
    fitted = [pred] * len(history)
    return ForecastResult(
        metric="",
        forecast_method_used="sma",
        horizon=horizon,
        predictions=preds,
        confidence_lo=lo,
        confidence_hi=hi,
        rmse=_rmse(history, fitted),
        samples_used=len(history),
    )


def _forecast_exp_smoothing(history: list[float], horizon: int, alpha: float = 0.3) -> ForecastResult:
    if not history:
        return ForecastResult(
            metric="",
            forecast_method_used="exp_smoothing",
            horizon=horizon,
            predictions=[0.0] * horizon,
            confidence_lo=[0.0] * horizon,
            confidence_hi=[0.0] * horizon,
        )
    level = history[0]
    fitted: list[float] = []
    for v in history:
        fitted.append(level)
        level = alpha * v + (1 - alpha) * level
    # The last `level` is the h-step ahead forecast (flat)
    preds = [level] * horizon
    std = _stddev(history)
    lo, hi = _conf_bounds(preds, std)
    return ForecastResult(
        metric="",
        forecast_method_used="exp_smoothing",
        horizon=horizon,
        predictions=preds,
        confidence_lo=lo,
        confidence_hi=hi,
        rmse=_rmse(history, fitted),
        samples_used=len(history),
    )


def _forecast_linear_trend(history: list[float], horizon: int) -> ForecastResult:
    if len(history) < 2:
        return ForecastResult(
            metric="",
            forecast_method_used="linear_trend",
            horizon=horizon,
            predictions=[history[-1] if history else 0.0] * horizon,
            confidence_lo=[0.0] * horizon,
            confidence_hi=[0.0] * horizon,
        )
    slope, intercept = _ols(history)
    n = len(history)
    preds = [intercept + slope * (n + i) for i in range(horizon)]
    fitted = [intercept + slope * i for i in range(n)]
    std = _stddev([a - f for a, f in zip(history, fitted)])
    lo, hi = _conf_bounds(preds, std)
    return ForecastResult(
        metric="",
        forecast_method_used="linear_trend",
        horizon=horizon,
        predictions=preds,
        confidence_lo=lo,
        confidence_hi=hi,
        trend_slope=slope,
        rmse=_rmse(history, fitted),
        samples_used=n,
    )


def _forecast_seasonal(history: list[float], horizon: int, period: int = 7) -> ForecastResult:
    """Additive decomposition: trend (OLS) + seasonal indices."""
    n = len(history)
    if n < period * 2:
        # Fallback to linear trend when insufficient history
        result = _forecast_linear_trend(history, horizon)
        result.forecast_method_used = "seasonal"
        return result

    # Detrend
    slope, intercept = _ols(history)
    detrended = [history[i] - (intercept + slope * i) for i in range(n)]

    # Seasonal indices (average residual per phase)
    indices = [0.0] * period
    counts = [0] * period
    for i, v in enumerate(detrended):
        p = i % period
        indices[p] += v
        counts[p] += 1
    indices = [indices[i] / counts[i] if counts[i] else 0.0 for i in range(period)]

    # Forecast: trend + seasonal index
    preds = [intercept + slope * (n + h) + indices[(n + h) % period] for h in range(horizon)]
    fitted = [intercept + slope * i + indices[i % period] for i in range(n)]
    std = _stddev([a - f for a, f in zip(history, fitted)])
    lo, hi = _conf_bounds(preds, std)
    return ForecastResult(
        metric="",
        forecast_method_used="seasonal",
        horizon=horizon,
        predictions=preds,
        confidence_lo=lo,
        confidence_hi=hi,
        trend_slope=slope,
        rmse=_rmse(history, fitted),
        samples_used=n,
    )


def _forecast_holt_winters(history: list[float], horizon: int, alpha: float = 0.3, beta: float = 0.1) -> ForecastResult:
    """Holt-Winters double exponential smoothing (level + trend).

    Captures both the current level and the trend direction, ideal for
    quality metrics with gradual drift.

    Args:
        history: Historical values.
        horizon: Number of future steps to predict.
        alpha: Level smoothing factor (0 < alpha < 1).
        beta: Trend smoothing factor (0 < beta < 1).

    Returns:
        ForecastResult with predictions and confidence bounds.
    """
    if len(history) < 2:
        return _forecast_linear_trend(history, horizon)

    # Initialize level and trend
    level = history[0]
    trend = history[1] - history[0]
    fitted: list[float] = []

    for value in history:
        fitted.append(level + trend)
        new_level = alpha * value + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        level = new_level
        trend = new_trend

    # Forecast: level + trend * steps_ahead
    preds = [level + trend * (h + 1) for h in range(horizon)]

    # Residuals for confidence intervals
    residuals = [history[i] - fitted[i] for i in range(len(history))]
    std = _stddev(residuals) if len(residuals) >= 2 else _stddev(history)

    # 80% confidence (z=1.28) and 95% confidence (z=1.96)
    lo_80 = [p - 1.28 * std * math.sqrt(h + 1) for h, p in enumerate(preds)]
    hi_80 = [p + 1.28 * std * math.sqrt(h + 1) for h, p in enumerate(preds)]
    return ForecastResult(
        metric="",
        forecast_method_used="holt_winters",
        horizon=horizon,
        predictions=preds,
        confidence_lo=lo_80,
        confidence_hi=hi_80,
        trend_slope=trend,
        rmse=_rmse(history, fitted),
        samples_used=len(history),
    )


def _forecast_auto(history: list[float], horizon: int, period: int = 7) -> ForecastResult:
    """Auto-select best forecast method using walk-forward cross-validation.

    Evaluates Holt-Winters, linear trend, and seasonal methods on recent data.
    Selects the method with lowest MAPE (Mean Absolute Percentage Error).
    Defaults to Holt-Winters when insufficient data for comparison (<14 points).

    Args:
        history: Historical values.
        horizon: Number of future steps to predict.
        period: Season length for seasonal decomposition.

    Returns:
        ForecastResult from the best-performing method.
    """
    if len(history) < 14:
        result = _forecast_holt_winters(history, horizon)
        result.forecast_method_used = "auto(holt_winters)"
        return result

    # Walk-forward cross-validation: use last 7 points as validation
    val_size = min(7, len(history) // 3)
    train = history[:-val_size]
    actual = history[-val_size:]

    methods: dict[str, Callable[[list[float], int], ForecastResult]] = {
        "holt_winters": lambda h, hz: _forecast_holt_winters(h, hz),
        "linear_trend": lambda h, hz: _forecast_linear_trend(h, hz),
    }
    if len(history) >= period * 2:
        methods["seasonal"] = lambda h, hz: _forecast_seasonal(h, hz, period)

    best_method = "holt_winters"
    best_mape = float("inf")

    for name, fn in methods.items():
        result = fn(train, val_size)
        # Compute MAPE
        mape = 0.0
        valid_count = 0
        for a, p in zip(actual, result.predictions):
            if abs(a) > 1e-10:
                mape += abs((a - p) / a)
                valid_count += 1
        if valid_count > 0:
            mape /= valid_count
        else:
            mape = float("inf")

        if mape < best_mape:
            best_mape = mape
            best_method = name

    # Re-run best method on full history
    result = methods[best_method](history, horizon)
    result.forecast_method_used = f"auto({best_method})"
    return result


_METHODS = {
    "sma": lambda h, req: _forecast_sma(h, req.horizon),
    "exp_smoothing": lambda h, req: _forecast_exp_smoothing(h, req.horizon, req.alpha),
    "linear_trend": lambda h, req: _forecast_linear_trend(h, req.horizon),
    "seasonal": lambda h, req: _forecast_seasonal(h, req.horizon, req.period),
    "holt_winters": lambda h, req: _forecast_holt_winters(h, req.horizon),
    "auto": lambda h, req: _forecast_auto(h, req.horizon, req.period),
}


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------


class Forecaster:
    """Manages time-series history and produces forecasts.

    Singleton — use ``get_forecaster()``.
    """

    _instance: Forecaster | None = None
    _class_lock = threading.Lock()
    MAX_HISTORY = 1_000

    def __new__(cls) -> Forecaster:
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._setup()
        return cls._instance

    def _setup(self) -> None:
        self._lock = threading.RLock()
        self._history: dict[str, deque[float]] = {}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    # How often (in observations) to run the automatic trend check
    _ALERT_CHECK_INTERVAL: int = 100
    # Minimum absolute trend slope to trigger an alert
    _ALERT_SLOPE_THRESHOLD: float = 0.05

    def ingest(self, metric: str, value: float) -> None:
        """Append a new observation for a metric.

        Every ``_ALERT_CHECK_INTERVAL`` observations, runs a quick linear
        forecast and emits an alert via AlertEngine if the trend slope
        exceeds the threshold (rising latency, rising cost, etc.).

        Args:
            metric: The metric.
            value: The value.
        """
        with self._lock:
            q = self._history.setdefault(metric, deque(maxlen=self.MAX_HISTORY))
            q.append(value)

        # Periodic trend check
        if len(q) > 0 and len(q) % self._ALERT_CHECK_INTERVAL == 0:
            self._check_trend_and_alert(metric)

    def ingest_many(self, metric: str, values: list[float]) -> None:
        """Ingest many.

        Args:
            metric: The metric.
            values: The values.
        """
        for v in values:
            self.ingest(metric, v)

    def _check_trend_and_alert(self, metric: str) -> None:
        """Run a quick linear forecast and emit alert if trend is rising.

        Called automatically every ``_ALERT_CHECK_INTERVAL`` observations.
        Runs in the caller's thread — kept lightweight to avoid blocking.
        """
        try:
            result = self.forecast(
                ForecastRequest(
                    metric=metric,
                    horizon=5,
                    method="linear_trend",
                )
            )
            if result.trend_slope is not None and abs(result.trend_slope) > self._ALERT_SLOPE_THRESHOLD:
                direction = "rising" if result.trend_slope > 0 else "falling"
                logger.warning(
                    "Forecast alert: %s is %s (slope=%.4f) — predicted next 5: %s",
                    metric,
                    direction,
                    result.trend_slope,
                    [round(p, 2) for p in result.predictions[:3]],
                )
                # Emit via EventBus for downstream consumers (dashboard, alerting)
                try:
                    from vetinari.events import Event, get_event_bus

                    get_event_bus().publish(Event(event_type="forecast.trend_alert", timestamp=time.time()))
                except Exception:
                    logger.warning("EventBus publish failed for trend alert on %s — alert not delivered", metric)
        except Exception:
            logger.warning("Trend check failed for %s — trend alerts may be missed", metric)

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        """Produce a forecast for *request.metric* using *request.method*.

        Returns a ForecastResult with ``horizon`` predictions.
        If insufficient history exists (< 2 points) the last known value
        is repeated for all steps.

        Returns:
            ForecastResult containing the requested number of predicted values,
            80% confidence bounds, trend slope (linear/Holt-Winters only), and
            in-sample RMSE.

        Raises:
            ValueError: If ``request.method`` is not a recognised method name.
        """
        with self._lock:
            history = list(self._history.get(request.metric, deque()))

        # Validate method name early (before any fallback paths)
        method_fn = _METHODS.get(request.method)
        if method_fn is None:
            raise ConfigurationError(f"Unknown forecasting method '{request.method}'. Valid: {sorted(_METHODS)}")

        if len(history) < 2:
            if len(history) == 1:
                # Single data point — repeat it
                preds = [history[0]] * request.horizon
            else:
                preds = [0.0] * request.horizon
            return ForecastResult(
                metric=request.metric,
                forecast_method_used=request.method,
                horizon=request.horizon,
                predictions=preds,
                confidence_lo=preds,
                confidence_hi=preds,
                samples_used=len(history),
            )

        # Simple moving average fallback for sparse data (2-4 points)
        if len(history) < 5:
            window = len(history)
            avg = sum(history[-window:]) / window
            trend = (history[-1] - history[0]) / max(window - 1, 1)
            preds = [avg + trend * (i + 1) for i in range(request.horizon)]
            # Simple confidence bands (widen with horizon)
            spread = max(abs(max(history) - min(history)), abs(avg) * 0.1)
            lo = [p - spread * (1 + 0.1 * i) for i, p in enumerate(preds)]
            hi = [p + spread * (1 + 0.1 * i) for i, p in enumerate(preds)]
            return ForecastResult(
                metric=request.metric,
                forecast_method_used=request.method,
                horizon=request.horizon,
                predictions=preds,
                confidence_lo=lo,
                confidence_hi=hi,
                samples_used=len(history),
            )

        result = method_fn(history, request)
        result.metric = request.metric
        return result

    # ------------------------------------------------------------------
    # Capacity planning helpers
    # ------------------------------------------------------------------

    def will_exceed(self, metric: str, threshold: float, horizon: int = 10, method: str = "linear_trend") -> bool:
        """Return True if the forecasted trajectory is predicted to exceed.

        *threshold* within *horizon* steps.

        Args:
            metric: The metric.
            threshold: The threshold.
            horizon: The horizon.
            method: The method.

        Returns:
            True if successful, False otherwise.
        """
        req = ForecastRequest(metric=metric, horizon=horizon, method=method)
        result = self.forecast(req)
        return any(p > threshold for p in result.predictions)

    def steps_until_threshold(
        self,
        metric: str,
        threshold: float,
        horizon: int = 50,
        method: str = "linear_trend",
    ) -> int | None:
        """Return the number of steps until the forecast first exceeds *threshold*,.

        or None if it does not within *horizon*.

        Args:
            metric: The metric.
            threshold: The threshold.
            horizon: The horizon.
            method: The method.

        Returns:
            The computed value.
        """
        req = ForecastRequest(metric=metric, horizon=horizon, method=method)
        result = self.forecast(req)
        for i, p in enumerate(result.predictions):
            if p > threshold:
                return i + 1
        return None

    def check_sla_breach(
        self,
        metric: str,
        sla_threshold: float,
        horizon_days: int = 7,
    ) -> bool:
        """Check if forecast predicts SLA breach within horizon.

        Emits RetrainingRecommended event when the 80% confidence interval
        lower bound crosses the SLA threshold.

        Args:
            metric: The quality metric to check.
            sla_threshold: Minimum acceptable quality level.
            horizon_days: How far ahead to forecast (days).

        Returns:
            True if breach predicted, False otherwise.
        """
        req = ForecastRequest(metric=metric, horizon=horizon_days, method="auto")
        result = self.forecast(req)

        # Check if 80% CI lower bound crosses SLA threshold
        for day_idx, lo_bound in enumerate(result.confidence_lo):
            if lo_bound < sla_threshold:
                logger.warning(
                    "SLA breach predicted for %s in %d days (predicted_lo=%.3f < threshold=%.3f, method=%s)",
                    metric,
                    day_idx + 1,
                    lo_bound,
                    sla_threshold,
                    result.forecast_method_used,
                )
                # Emit event
                try:
                    from vetinari.events import RetrainingRecommended, get_event_bus

                    ci_width = result.confidence_hi[day_idx] - lo_bound
                    event = RetrainingRecommended(
                        event_type="",
                        timestamp=time.time(),
                        metric=metric,
                        predicted_quality=result.predictions[day_idx],
                        days_until_breach=day_idx + 1,
                        confidence_interval=ci_width,
                        forecast_method_used=result.forecast_method_used,
                    )
                    get_event_bus().publish(event)
                except Exception:  # Broad: event emission is best-effort; never blocks forecasting
                    logger.exception("Failed to emit RetrainingRecommended event")
                return True

        return False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_history(self, metric: str) -> list[float]:
        """Get history.

        Returns:
            List of results.
        """
        with self._lock:
            return list(self._history.get(metric, []))

    def list_metrics(self) -> list[str]:
        """List metrics.

        Returns:
            Names of all metrics for which at least one observation has been
            ingested, in insertion order.
        """
        with self._lock:
            return list(self._history.keys())

    def get_stats(self) -> dict[str, Any]:
        """Summarise the current forecaster state.

        Returns:
            Dictionary with ``tracked_metrics`` (number of distinct metric
            names ingested) and ``history_sizes`` (mapping of metric name to
            the number of stored observations for that metric).
        """
        with self._lock:
            return {
                "tracked_metrics": len(self._history),
                "history_sizes": {k: len(v) for k, v in self._history.items()},
            }

    def clear(self) -> None:
        """Clear for the current context."""
        with self._lock:
            self._history.clear()


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------


def get_forecaster() -> Forecaster:
    """Return the singleton Forecaster instance, creating it if necessary.

    Returns:
        The shared Forecaster singleton used for all time-series forecasting.
    """
    return Forecaster()


def reset_forecaster() -> None:
    """Reset forecaster."""
    with Forecaster._class_lock:
        Forecaster._instance = None
