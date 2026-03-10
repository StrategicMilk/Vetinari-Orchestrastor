"""
Forecasting & Capacity Planning — vetinari.analytics.forecasting  (Phase 5)

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
    print(result.predictions)   # list of 5 forecasted values
    print(result.trend_slope)
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / Result
# ---------------------------------------------------------------------------

@dataclass
class ForecastRequest:
    """Parameters for a forecast call."""
    metric:     str
    horizon:    int   = 5       # number of future steps to predict
    method:     str   = "linear_trend"   # sma | exp_smoothing | linear_trend | seasonal
    alpha:      float = 0.3     # smoothing factor for ES
    period:     int   = 7       # season length for seasonal decomposition

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric":  self.metric,
            "horizon": self.horizon,
            "method":  self.method,
            "alpha":   self.alpha,
            "period":  self.period,
        }


@dataclass
class ForecastResult:
    """Output of a forecast operation."""
    metric:       str
    method:       str
    horizon:      int
    predictions:  List[float]
    confidence_lo: List[float]   # lower 80% confidence bound
    confidence_hi: List[float]   # upper 80% confidence bound
    trend_slope:  float = 0.0    # rate of change per step (linear_trend only)
    rmse:         float = 0.0    # in-sample root mean squared error
    samples_used: int   = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric":        self.metric,
            "method":        self.method,
            "horizon":       self.horizon,
            "predictions":   self.predictions,
            "confidence_lo": self.confidence_lo,
            "confidence_hi": self.confidence_hi,
            "trend_slope":   self.trend_slope,
            "rmse":          self.rmse,
            "samples_used":  self.samples_used,
        }


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _ols(y: List[float]) -> Tuple[float, float]:
    """Return (slope, intercept) of the OLS line through enumerate(y)."""
    n = len(y)
    sx = n * (n - 1) / 2           # sum of 0..n-1
    sx2 = n * (n - 1) * (2 * n - 1) / 6
    sy  = sum(y)
    sxy = sum(i * v for i, v in enumerate(y))
    denom = n * sx2 - sx * sx
    if denom == 0:
        return 0.0, sum(y) / n if y else 0.0
    slope     = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def _rmse(actual: List[float], predicted: List[float]) -> float:
    if not actual:
        return 0.0
    n = min(len(actual), len(predicted))
    return math.sqrt(sum((actual[i] - predicted[i]) ** 2 for i in range(n)) / n)


def _stddev(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _conf_bounds(preds: List[float], std: float, z: float = 1.28
                 ) -> Tuple[List[float], List[float]]:
    lo = [p - z * std for p in preds]
    hi = [p + z * std for p in preds]
    return lo, hi


# ---------------------------------------------------------------------------
# Forecasting methods (pure functions)
# ---------------------------------------------------------------------------

def _forecast_sma(history: List[float], horizon: int, window: int = 10
                  ) -> ForecastResult:
    w = history[-window:] if len(history) >= window else history
    pred = sum(w) / len(w) if w else 0.0
    preds = [pred] * horizon
    std = _stddev(history)
    lo, hi = _conf_bounds(preds, std)
    fitted = [pred] * len(history)
    return ForecastResult(
        metric="", method="sma", horizon=horizon,
        predictions=preds, confidence_lo=lo, confidence_hi=hi,
        rmse=_rmse(history, fitted), samples_used=len(history),
    )


def _forecast_exp_smoothing(history: List[float], horizon: int,
                             alpha: float = 0.3) -> ForecastResult:
    if not history:
        return ForecastResult(
            metric="", method="exp_smoothing", horizon=horizon,
            predictions=[0.0] * horizon, confidence_lo=[0.0] * horizon,
            confidence_hi=[0.0] * horizon,
        )
    level = history[0]
    fitted: List[float] = []
    for v in history:
        fitted.append(level)
        level = alpha * v + (1 - alpha) * level
    # The last `level` is the h-step ahead forecast (flat)
    preds = [level] * horizon
    std = _stddev(history)
    lo, hi = _conf_bounds(preds, std)
    return ForecastResult(
        metric="", method="exp_smoothing", horizon=horizon,
        predictions=preds, confidence_lo=lo, confidence_hi=hi,
        rmse=_rmse(history, fitted), samples_used=len(history),
    )


def _forecast_linear_trend(history: List[float], horizon: int) -> ForecastResult:
    if len(history) < 2:
        return ForecastResult(
            metric="", method="linear_trend", horizon=horizon,
            predictions=[history[-1] if history else 0.0] * horizon,
            confidence_lo=[0.0] * horizon, confidence_hi=[0.0] * horizon,
        )
    slope, intercept = _ols(history)
    n = len(history)
    preds = [intercept + slope * (n + i) for i in range(horizon)]
    fitted = [intercept + slope * i for i in range(n)]
    std = _stddev([a - f for a, f in zip(history, fitted)])
    lo, hi = _conf_bounds(preds, std)
    return ForecastResult(
        metric="", method="linear_trend", horizon=horizon,
        predictions=preds, confidence_lo=lo, confidence_hi=hi,
        trend_slope=slope, rmse=_rmse(history, fitted),
        samples_used=n,
    )


def _forecast_seasonal(history: List[float], horizon: int,
                        period: int = 7) -> ForecastResult:
    """Additive decomposition: trend (OLS) + seasonal indices."""
    n = len(history)
    if n < period * 2:
        # Fallback to linear trend when insufficient history
        result = _forecast_linear_trend(history, horizon)
        result.method = "seasonal"
        return result

    # Detrend
    slope, intercept = _ols(history)
    detrended = [history[i] - (intercept + slope * i) for i in range(n)]

    # Seasonal indices (average residual per phase)
    indices = [0.0] * period
    counts  = [0]   * period
    for i, v in enumerate(detrended):
        p = i % period
        indices[p] += v
        counts[p]  += 1
    indices = [indices[i] / counts[i] if counts[i] else 0.0 for i in range(period)]

    # Forecast: trend + seasonal index
    preds = [
        intercept + slope * (n + h) + indices[(n + h) % period]
        for h in range(horizon)
    ]
    fitted = [intercept + slope * i + indices[i % period] for i in range(n)]
    std = _stddev([a - f for a, f in zip(history, fitted)])
    lo, hi = _conf_bounds(preds, std)
    return ForecastResult(
        metric="", method="seasonal", horizon=horizon,
        predictions=preds, confidence_lo=lo, confidence_hi=hi,
        trend_slope=slope, rmse=_rmse(history, fitted),
        samples_used=n,
    )


_METHODS = {
    "sma":           lambda h, req: _forecast_sma(h, req.horizon),
    "exp_smoothing": lambda h, req: _forecast_exp_smoothing(h, req.horizon, req.alpha),
    "linear_trend":  lambda h, req: _forecast_linear_trend(h, req.horizon),
    "seasonal":      lambda h, req: _forecast_seasonal(h, req.horizon, req.period),
}


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------

class Forecaster:
    """
    Manages time-series history and produces forecasts.
    Singleton — use ``get_forecaster()``.
    """

    _instance:   Optional["Forecaster"] = None
    _class_lock  = threading.Lock()
    MAX_HISTORY  = 1_000

    def __new__(cls) -> "Forecaster":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._setup()
        return cls._instance

    def _setup(self) -> None:
        self._lock    = threading.RLock()
        self._history: Dict[str, Deque[float]] = {}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, metric: str, value: float) -> None:
        """Append a new observation for a metric."""
        with self._lock:
            q = self._history.setdefault(metric, deque(maxlen=self.MAX_HISTORY))
            q.append(value)

    def ingest_many(self, metric: str, values: List[float]) -> None:
        for v in values:
            self.ingest(metric, v)

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        """
        Produce a forecast for *request.metric* using *request.method*.

        Returns a ForecastResult with ``horizon`` predictions.
        If insufficient history exists (< 2 points) the last known value
        is repeated for all steps.
        """
        with self._lock:
            history = list(self._history.get(request.metric, deque()))

        # Validate method name early (before any fallback paths)
        method_fn = _METHODS.get(request.method)
        if method_fn is None:
            raise ValueError(
                f"Unknown forecasting method '{request.method}'. "
                f"Valid: {sorted(_METHODS)}"
            )

        if len(history) < 2:
            if len(history) == 1:
                # Single data point — repeat it
                preds = [history[0]] * request.horizon
            else:
                preds = [0.0] * request.horizon
            return ForecastResult(
                metric=request.metric, method=request.method,
                horizon=request.horizon,
                predictions=preds,
                confidence_lo=preds, confidence_hi=preds,
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
                metric=request.metric, method=request.method,
                horizon=request.horizon,
                predictions=preds,
                confidence_lo=lo, confidence_hi=hi,
                samples_used=len(history),
            )

        result = method_fn(history, request)
        result.metric = request.metric
        return result

    # ------------------------------------------------------------------
    # Capacity planning helpers
    # ------------------------------------------------------------------

    def will_exceed(self, metric: str, threshold: float,
                    horizon: int = 10, method: str = "linear_trend") -> bool:
        """
        Return True if the forecasted trajectory is predicted to exceed
        *threshold* within *horizon* steps.
        """
        req = ForecastRequest(metric=metric, horizon=horizon, method=method)
        result = self.forecast(req)
        return any(p > threshold for p in result.predictions)

    def steps_until_threshold(self, metric: str, threshold: float,
                               horizon: int = 50,
                               method: str = "linear_trend") -> Optional[int]:
        """
        Return the number of steps until the forecast first exceeds *threshold*,
        or None if it does not within *horizon*.
        """
        req = ForecastRequest(metric=metric, horizon=horizon, method=method)
        result = self.forecast(req)
        for i, p in enumerate(result.predictions):
            if p > threshold:
                return i + 1
        return None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_history(self, metric: str) -> List[float]:
        with self._lock:
            return list(self._history.get(metric, []))

    def list_metrics(self) -> List[str]:
        with self._lock:
            return list(self._history.keys())

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "tracked_metrics": len(self._history),
                "history_sizes":   {k: len(v) for k, v in self._history.items()},
            }

    def clear(self) -> None:
        with self._lock:
            self._history.clear()


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

def get_forecaster() -> Forecaster:
    return Forecaster()


def reset_forecaster() -> None:
    with Forecaster._class_lock:
        Forecaster._instance = None
