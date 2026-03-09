"""
SLA Tracking — vetinari.analytics.sla  (Phase 5)

Tracks Service Level Objectives (SLOs) and computes SLA compliance metrics.

Concepts
--------
    SLOTarget   — a named objective: e.g. "p95 latency < 500 ms".
    SLAWindow   — a rolling or fixed time window over which compliance is
                  measured (e.g. "last 24 hours").
    SLAReport   — computed compliance: requests in/out of budget,
                  compliance %, breach list.

Built-in SLO types
------------------
    latency_p50  / latency_p95 / latency_p99   — percentile latency budget
    success_rate                                 — minimum success %
    error_rate                                   — maximum error %
    throughput                                   — minimum requests/s
    approval_rate                                — minimum plan approval %

Usage
-----
    from vetinari.analytics.sla import get_sla_tracker, SLOTarget, SLOType

    tracker = get_sla_tracker()
    tracker.register_slo(SLOTarget(
        name="api-latency-p95",
        slo_type=SLOType.LATENCY_P95,
        budget=500.0,          # ms
        window_seconds=3600,   # evaluate over last hour
    ))

    # Feed an observation
    tracker.record_latency("openai:gpt-4", latency_ms=320.0)

    report = tracker.get_report("api-latency-p95")
    print(report.compliance_pct, report.is_compliant)
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SLOType(Enum):
    LATENCY_P50    = "latency_p50"
    LATENCY_P95    = "latency_p95"
    LATENCY_P99    = "latency_p99"
    SUCCESS_RATE   = "success_rate"    # budget = minimum % (e.g. 99.0)
    ERROR_RATE     = "error_rate"      # budget = maximum % (e.g. 1.0)
    THROUGHPUT     = "throughput"      # budget = minimum req/s
    APPROVAL_RATE  = "approval_rate"   # budget = minimum % (e.g. 90.0)


# ---------------------------------------------------------------------------
# SLO target definition
# ---------------------------------------------------------------------------

@dataclass
class SLOTarget:
    """A single Service Level Objective."""
    name:           str
    slo_type:       SLOType
    budget:         float               # threshold value (semantics depend on type)
    window_seconds: float = 3600.0      # rolling window for compliance evaluation
    description:    str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name":           self.name,
            "slo_type":       self.slo_type.value,
            "budget":         self.budget,
            "window_seconds": self.window_seconds,
            "description":    self.description,
        }


# ---------------------------------------------------------------------------
# Breach record
# ---------------------------------------------------------------------------

@dataclass
class SLABreach:
    """A single moment when an SLO was violated."""
    slo_name:  str
    value:     float
    budget:    float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slo_name":  self.slo_name,
            "value":     self.value,
            "budget":    self.budget,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# SLA report
# ---------------------------------------------------------------------------

@dataclass
class SLAReport:
    """Computed compliance for a single SLO."""
    slo:            SLOTarget
    window_start:   float
    window_end:     float
    total_samples:  int
    good_samples:   int
    compliance_pct: float
    is_compliant:   bool
    current_value:  float     # latest computed metric value
    breaches:       List[SLABreach] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slo":            self.slo.to_dict(),
            "window_start":   self.window_start,
            "window_end":     self.window_end,
            "total_samples":  self.total_samples,
            "good_samples":   self.good_samples,
            "compliance_pct": self.compliance_pct,
            "is_compliant":   self.is_compliant,
            "current_value":  self.current_value,
            "breaches":       [b.to_dict() for b in self.breaches],
        }


# ---------------------------------------------------------------------------
# Internal observation
# ---------------------------------------------------------------------------

@dataclass
class _Obs:
    value:     float
    timestamp: float
    success:   bool = True   # for success/error rate SLOs


def _percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    idx = p / 100 * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class SLATracker:
    """
    Thread-safe SLA / SLO tracker.  Singleton — use ``get_sla_tracker()``.
    """

    _instance:   Optional["SLATracker"] = None
    _class_lock  = threading.Lock()

    def __new__(cls) -> "SLATracker":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._setup()
        return cls._instance

    def _setup(self) -> None:
        self._lock    = threading.RLock()
        self._slos:   Dict[str, SLOTarget] = {}
        # keyed by SLO name → deque of _Obs within a rolling window
        self._obs:    Dict[str, Deque[_Obs]] = {}
        self._breaches: List[SLABreach] = []
        # Per-model latency observations for model-level compliance queries
        self._model_obs: Dict[str, Deque[_Obs]] = {}

    # ------------------------------------------------------------------
    # SLO management
    # ------------------------------------------------------------------

    def register_slo(self, slo: SLOTarget) -> None:
        with self._lock:
            self._slos[slo.name] = slo
            self._obs.setdefault(slo.name, deque())
            logger.debug("Registered SLO: %s (%s budget=%s)", slo.name, slo.slo_type.value, slo.budget)

    def unregister_slo(self, name: str) -> bool:
        with self._lock:
            existed = name in self._slos
            self._slos.pop(name, None)
            self._obs.pop(name, None)
            return existed

    def list_slos(self) -> List[SLOTarget]:
        with self._lock:
            return list(self._slos.values())

    # ------------------------------------------------------------------
    # Recording observations
    # ------------------------------------------------------------------

    def record_latency(self, key: str, latency_ms: float, success: bool = True) -> None:
        """Feed a latency observation (ms) to all latency-type SLOs."""
        now = time.time()
        with self._lock:
            for slo in self._slos.values():
                if slo.slo_type in (SLOType.LATENCY_P50, SLOType.LATENCY_P95, SLOType.LATENCY_P99):
                    self._push(slo.name, _Obs(value=latency_ms, timestamp=now, success=success))
            # Also track per-model for model-level compliance queries
            q = self._model_obs.setdefault(key, deque())
            q.append(_Obs(value=latency_ms, timestamp=now, success=success))
            # Evict observations older than 1 hour
            cutoff = now - 3600
            while q and q[0].timestamp < cutoff:
                q.popleft()

    def record_request(self, success: bool) -> None:
        """Feed a success/failure observation to success/error-rate SLOs."""
        now = time.time()
        value = 1.0 if success else 0.0
        with self._lock:
            for slo in self._slos.values():
                if slo.slo_type in (SLOType.SUCCESS_RATE, SLOType.ERROR_RATE):
                    self._push(slo.name, _Obs(value=value, timestamp=now, success=success))

    def record_plan_decision(self, approved: bool) -> None:
        """Feed a plan-gate decision to approval-rate SLOs."""
        now = time.time()
        with self._lock:
            for slo in self._slos.values():
                if slo.slo_type == SLOType.APPROVAL_RATE:
                    self._push(slo.name, _Obs(value=1.0 if approved else 0.0,
                                              timestamp=now, success=approved))

    def record_metric(self, slo_name: str, value: float, success: bool = True) -> None:
        """Directly push a value to a named SLO's observation queue."""
        with self._lock:
            if slo_name in self._slos:
                self._push(slo_name, _Obs(value=value, timestamp=time.time(), success=success))

    def _push(self, slo_name: str, obs: _Obs) -> None:
        """Must be called under self._lock."""
        q = self._obs[slo_name]
        q.append(obs)
        # Evict observations outside the window
        slo = self._slos[slo_name]
        cutoff = obs.timestamp - slo.window_seconds
        while q and q[0].timestamp < cutoff:
            q.popleft()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_report(self, slo_name: str) -> Optional[SLAReport]:
        """Compute the compliance report for a named SLO."""
        with self._lock:
            slo = self._slos.get(slo_name)
            if slo is None:
                return None
            obs_list = list(self._obs.get(slo_name, []))

        if not obs_list:
            now = time.time()
            return SLAReport(
                slo=slo, window_start=now - slo.window_seconds,
                window_end=now, total_samples=0, good_samples=0,
                compliance_pct=100.0, is_compliant=True, current_value=0.0,
            )

        now         = time.time()
        window_start = now - slo.window_seconds
        values      = [o.value for o in obs_list]
        n           = len(obs_list)

        # Compute current metric value and good/bad split
        if slo.slo_type == SLOType.LATENCY_P50:
            current = _percentile(values, 50)
            good    = sum(1 for v in values if v <= slo.budget)
        elif slo.slo_type == SLOType.LATENCY_P95:
            current = _percentile(values, 95)
            good    = sum(1 for v in values if v <= slo.budget)
        elif slo.slo_type == SLOType.LATENCY_P99:
            current = _percentile(values, 99)
            good    = sum(1 for v in values if v <= slo.budget)
        elif slo.slo_type == SLOType.SUCCESS_RATE:
            current = (sum(o.success for o in obs_list) / n) * 100
            good    = n if current >= slo.budget else 0
        elif slo.slo_type == SLOType.ERROR_RATE:
            current = (sum(not o.success for o in obs_list) / n) * 100
            good    = n if current <= slo.budget else 0
        elif slo.slo_type == SLOType.THROUGHPUT:
            elapsed = max((obs_list[-1].timestamp - obs_list[0].timestamp), 1.0)
            current = n / elapsed
            good    = n if current >= slo.budget else 0
        elif slo.slo_type == SLOType.APPROVAL_RATE:
            current = (sum(o.success for o in obs_list) / n) * 100
            good    = n if current >= slo.budget else 0
        else:
            current = _percentile(values, 95)
            good    = n

        compliance_pct = (good / n * 100) if n > 0 else 100.0
        is_compliant   = compliance_pct >= 99.0   # 99% good samples = in-SLA

        # Collect breaches within window
        window_breaches = [
            b for b in self._breaches
            if b.slo_name == slo_name and b.timestamp >= window_start
        ]

        return SLAReport(
            slo=slo,
            window_start=window_start,
            window_end=now,
            total_samples=n,
            good_samples=good,
            compliance_pct=compliance_pct,
            is_compliant=is_compliant,
            current_value=current,
            breaches=window_breaches,
        )

    def get_all_reports(self) -> List[SLAReport]:
        """Return reports for every registered SLO."""
        with self._lock:
            names = list(self._slos.keys())
        return [r for name in names if (r := self.get_report(name)) is not None]

    def get_model_compliance(self, model_id: str, budget_ms: float = 500.0) -> Optional[float]:
        """Get latency SLA compliance % for a specific model.

        Computes the percentage of recorded latency observations that were
        under *budget_ms*.  Returns ``None`` if no observations exist for
        the model.
        """
        with self._lock:
            q = self._model_obs.get(model_id)
            if not q:
                return None
            vals = list(q)
        good = sum(1 for o in vals if o.value <= budget_ms)
        return (good / len(vals)) * 100.0 if vals else None

    def record_breach(self, breach: SLABreach) -> None:
        with self._lock:
            self._breaches.append(breach)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "registered_slos": len(self._slos),
                "total_breaches":  len(self._breaches),
                "slo_names":       list(self._slos.keys()),
            }

    def clear(self) -> None:
        with self._lock:
            self._obs.clear()
            self._breaches.clear()
            self._model_obs.clear()
            for name in self._slos:
                self._obs[name] = deque()


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

def get_sla_tracker() -> SLATracker:
    return SLATracker()


def reset_sla_tracker() -> None:
    with SLATracker._class_lock:
        SLATracker._instance = None
