"""
Anomaly Detection — vetinari.analytics.anomaly  (Phase 5)

Provides statistical anomaly detection for Vetinari telemetry streams using
three complementary algorithms, all implemented with pure Python / stdlib so
no additional dependencies are needed:

    Z-Score detector        — flags values that deviate more than N standard
                              deviations from a rolling mean.

    IQR detector            — flags values outside the inter-quartile fence
                              (Q1 - k*IQR, Q3 + k*IQR).

    EWMA detector           — exponentially-weighted moving-average; flags
                              values whose deviation from the EWMA exceeds a
                              multiple of the EWMA standard deviation.

Usage
-----
    from vetinari.analytics.anomaly import get_anomaly_detector, AnomalyConfig

    detector = get_anomaly_detector()
    detector.configure(AnomalyConfig(z_threshold=3.0, ewma_alpha=0.3))

    result = detector.detect("adapter.latency", 850.0)
    if result.is_anomaly:
        print(result.reason, result.score)

    # Bulk ingest from MetricsSnapshot
    from vetinari.dashboard.api import get_dashboard_api
    anomalies = detector.scan_snapshot(get_dashboard_api().get_latest_metrics())
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AnomalyConfig:
    """Tuning parameters for all detectors."""
    window_size:    int   = 50     # rolling window length
    z_threshold:    float = 3.0    # standard deviations for Z-score alert
    iqr_factor:     float = 1.5    # fence multiplier for IQR alert
    ewma_alpha:     float = 0.2    # smoothing factor (0 < alpha < 1)
    ewma_threshold: float = 3.0    # sigma multiples for EWMA alert
    min_samples:    int   = 5      # minimum samples before flagging


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class AnomalyResult:
    """Detection outcome for a single observation."""
    metric:     str
    value:      float
    timestamp:  float = field(default_factory=time.time)
    is_anomaly: bool  = False
    method:     str   = ""      # "zscore" | "iqr" | "ewma" | ""
    score:      float = 0.0     # magnitude (e.g. number of sigma)
    reason:     str   = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric":     self.metric,
            "value":      self.value,
            "timestamp":  self.timestamp,
            "is_anomaly": self.is_anomaly,
            "method":     self.method,
            "score":      self.score,
            "reason":     self.reason,
        }


# ---------------------------------------------------------------------------
# Per-metric state
# ---------------------------------------------------------------------------

@dataclass
class _MetricState:
    window:       Deque[float] = field(default_factory=lambda: deque())
    ewma_mean:    Optional[float] = None
    ewma_var:     float = 0.0

    def push(self, value: float, window_size: int) -> None:
        self.window.append(value)
        if len(self.window) > window_size:
            self.window.popleft()


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals)


def _stddev(vals: List[float], mu: Optional[float] = None) -> float:
    if len(vals) < 2:
        return 0.0
    m = mu if mu is not None else _mean(vals)
    variance = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(variance)


def _percentile(sorted_vals: List[float], p: float) -> float:
    """Linear interpolation percentile on a sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    idx = p / 100 * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])


class AnomalyDetector:
    """
    Thread-safe anomaly detector. Singleton — use ``get_anomaly_detector()``.
    """

    _instance:    Optional["AnomalyDetector"] = None
    _class_lock   = threading.Lock()

    def __new__(cls) -> "AnomalyDetector":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._setup()
        return cls._instance

    def _setup(self) -> None:
        self._lock   = threading.RLock()
        self._config = AnomalyConfig()
        self._states: Dict[str, _MetricState] = {}
        self._history: List[AnomalyResult] = []

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(self, config: AnomalyConfig) -> None:
        with self._lock:
            self._config = config

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, metric: str, value: float) -> AnomalyResult:
        """
        Ingest a single observation and return an AnomalyResult.
        The value is always appended to the rolling window.
        """
        with self._lock:
            cfg = self._config
            state = self._states.setdefault(metric, _MetricState())
            state.push(value, cfg.window_size)

            # Update EWMA
            if state.ewma_mean is None:
                state.ewma_mean = value
                state.ewma_var  = 0.0
            else:
                diff = value - state.ewma_mean
                state.ewma_mean += cfg.ewma_alpha * diff
                state.ewma_var  = (1 - cfg.ewma_alpha) * (
                    state.ewma_var + cfg.ewma_alpha * diff * diff
                )

            n = len(state.window)
            if n < cfg.min_samples:
                return AnomalyResult(metric=metric, value=value)

            vals = list(state.window)

            # 1. Z-score
            mu  = _mean(vals)
            std = _stddev(vals, mu)
            if std > 0:
                z = abs(value - mu) / std
                if z > cfg.z_threshold:
                    result = AnomalyResult(
                        metric=metric, value=value,
                        is_anomaly=True, method="zscore",
                        score=z,
                        reason=f"z={z:.2f} > threshold {cfg.z_threshold}",
                    )
                    self._history.append(result)
                    return result

            # 2. IQR
            sorted_vals = sorted(vals)
            q1 = _percentile(sorted_vals, 25)
            q3 = _percentile(sorted_vals, 75)
            iqr = q3 - q1
            if iqr > 0:
                lo = q1 - cfg.iqr_factor * iqr
                hi = q3 + cfg.iqr_factor * iqr
                if value < lo or value > hi:
                    dist = max(abs(value - lo), abs(value - hi)) / iqr
                    result = AnomalyResult(
                        metric=metric, value=value,
                        is_anomaly=True, method="iqr",
                        score=dist,
                        reason=f"value {value:.3g} outside IQR fence [{lo:.3g}, {hi:.3g}]",
                    )
                    self._history.append(result)
                    return result

            # 3. EWMA
            ewma_std = math.sqrt(state.ewma_var) if state.ewma_var > 0 else 0.0
            if ewma_std > 0:
                dev = abs(value - state.ewma_mean) / ewma_std
                if dev > cfg.ewma_threshold:
                    result = AnomalyResult(
                        metric=metric, value=value,
                        is_anomaly=True, method="ewma",
                        score=dev,
                        reason=f"EWMA dev={dev:.2f} > threshold {cfg.ewma_threshold}",
                    )
                    self._history.append(result)
                    return result

            return AnomalyResult(metric=metric, value=value)

    def scan_snapshot(self, snapshot: Any) -> List[AnomalyResult]:
        """
        Feed all numeric leaf values from a MetricsSnapshot into the detector.
        Returns only the anomalies found.
        """
        snap_dict = snapshot.to_dict() if hasattr(snapshot, "to_dict") else {}
        anomalies: List[AnomalyResult] = []

        def _walk(node: Any, path: str) -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    _walk(v, f"{path}.{k}" if path else k)
            elif isinstance(node, (int, float)) and not isinstance(node, bool):
                r = self.detect(path, float(node))
                if r.is_anomaly:
                    anomalies.append(r)

        _walk(snap_dict, "")
        return anomalies

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_history(self, metric: Optional[str] = None) -> List[AnomalyResult]:
        with self._lock:
            if metric:
                return [r for r in self._history if r.metric == metric]
            return list(self._history)

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()

    def clear_state(self) -> None:
        with self._lock:
            self._states.clear()
            self._history.clear()

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "tracked_metrics": len(self._states),
                "total_anomalies": len(self._history),
                "config": {
                    "window_size":    self._config.window_size,
                    "z_threshold":    self._config.z_threshold,
                    "iqr_factor":     self._config.iqr_factor,
                    "ewma_alpha":     self._config.ewma_alpha,
                    "ewma_threshold": self._config.ewma_threshold,
                    "min_samples":    self._config.min_samples,
                },
            }


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

def get_anomaly_detector() -> AnomalyDetector:
    return AnomalyDetector()


def reset_anomaly_detector() -> None:
    with AnomalyDetector._class_lock:
        AnomalyDetector._instance = None
