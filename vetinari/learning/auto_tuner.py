"""
Auto-Tuner - Vetinari Self-Improvement Subsystem

Monitors SLA compliance and anomaly patterns, then automatically adjusts
system configuration to maintain performance targets.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TuningAction:
    """A recorded auto-tuning adjustment."""
    timestamp: str
    trigger: str       # What caused the tuning
    parameter: str     # What was changed
    old_value: Any
    new_value: Any
    rationale: str
    auto_applied: bool


class AutoTuner:
    """
    SLA-driven automatic system parameter adjustment.

    Monitors:
    - SLA compliance from the SLATracker
    - Anomaly detection from the AnomalyDetector
    - Cost trends from the CostTracker
    - Forecast thresholds from the Forecaster

    Adjusts (automatically or with approval):
    - Concurrency limits
    - Model routing weights
    - Anomaly detection thresholds
    - Retry parameters
    """

    MAX_AUTO_CONCURRENT = 8
    MIN_AUTO_CONCURRENT = 1

    def __init__(self):
        self._actions: List[TuningAction] = []
        self._current_config: Dict[str, Any] = {
            "max_concurrent": 4,
            "anomaly_threshold": 3.0,
            "retry_backoff_cap": 30,
            "min_quality_threshold": 0.65,
        }

    def run_cycle(self) -> List[TuningAction]:
        """
        Run one auto-tuning cycle.

        Checks all monitored metrics and applies necessary adjustments.

        Returns:
            List of TuningActions applied in this cycle.
        """
        applied: List[TuningAction] = []

        # Check SLA compliance
        applied.extend(self._tune_from_sla())

        # Check anomaly patterns
        applied.extend(self._tune_from_anomalies())

        # Check cost trends
        applied.extend(self._tune_from_costs())

        if applied:
            logger.info(f"[AutoTuner] Applied {len(applied)} tuning actions")
        return applied

    def _tune_from_sla(self) -> List[TuningAction]:
        """Adjust concurrency based on SLA compliance."""
        actions: List[TuningAction] = []
        try:
            from vetinari.analytics.sla import get_sla_tracker
            tracker = get_sla_tracker()
            raw_reports = tracker.get_all_reports() if hasattr(tracker, "get_all_reports") else []
            # get_all_reports() returns List[SLAReport]; convert to dict for iteration
            if isinstance(raw_reports, list):
                reports = {r.slo.name if hasattr(r, "slo") else str(i): r
                           for i, r in enumerate(raw_reports)}
            else:
                reports = raw_reports  # legacy dict path

            for slo_name, report in reports.items():
                report_dict = report.to_dict() if hasattr(report, "to_dict") else report
                if not report_dict.get("is_compliant", True):
                    # SLA breach -- reduce concurrency to slow down
                    current = self._current_config["max_concurrent"]
                    if current > self.MIN_AUTO_CONCURRENT:
                        new_val = max(self.MIN_AUTO_CONCURRENT, current - 1)
                        action = self._apply("SLA breach: " + slo_name,
                                             "max_concurrent", current, new_val,
                                             f"SLA '{slo_name}' breached -- reducing concurrency",
                                             auto=True)
                        actions.append(action)
                elif report_dict.get("compliance_pct", 100) > 99:
                    # Excellent compliance -- try increasing concurrency
                    current = self._current_config["max_concurrent"]
                    if current < self.MAX_AUTO_CONCURRENT:
                        new_val = min(self.MAX_AUTO_CONCURRENT, current + 1)
                        action = self._apply("SLA excellent: " + slo_name,
                                             "max_concurrent", current, new_val,
                                             f"SLA '{slo_name}' excellent -- increasing concurrency",
                                             auto=True)
                        actions.append(action)
        except Exception as e:
            logger.debug(f"SLA tuning failed: {e}")
        return actions

    def _tune_from_anomalies(self) -> List[TuningAction]:
        """Adjust anomaly thresholds based on false-positive rate."""
        actions: List[TuningAction] = []
        try:
            from vetinari.analytics.anomaly import get_anomaly_detector
            detector = get_anomaly_detector()
            history = getattr(detector, "_anomaly_history", [])

            # If too many recent anomalies, threshold is too sensitive
            def _ts(a):
                try:
                    return datetime.fromisoformat(a.get("detected_at", "2000-01-01")).timestamp()
                except (ValueError, TypeError):
                    return 0.0
            recent = [a for a in history if isinstance(a, dict) and
                      (datetime.now().timestamp() - _ts(a)) < 3600]

            current_thresh = self._current_config["anomaly_threshold"]
            if len(recent) > 10 and current_thresh < 5.0:
                new_thresh = round(current_thresh + 0.5, 1)
                action = self._apply(
                    "High anomaly rate", "anomaly_threshold",
                    current_thresh, new_thresh,
                    f"High false-positive rate ({len(recent)} in 1h) -- relaxing threshold",
                    auto=True
                )
                actions.append(action)
            elif len(recent) == 0 and current_thresh > 2.0:
                new_thresh = round(current_thresh - 0.5, 1)
                action = self._apply(
                    "Low anomaly rate", "anomaly_threshold",
                    current_thresh, new_thresh,
                    "No recent anomalies -- tightening threshold",
                    auto=True
                )
                actions.append(action)
        except Exception as e:
            logger.debug(f"Anomaly tuning failed: {e}")
        return actions

    def _tune_from_costs(self) -> List[TuningAction]:
        """Warn if costs are trending upward."""
        actions: List[TuningAction] = []
        try:
            from vetinari.analytics.forecasting import get_forecaster
            forecaster = get_forecaster()
            will_exceed = forecaster.will_exceed("cost_per_hour", threshold=1.0, horizon=24)
            if will_exceed:
                action = self._apply(
                    "Cost forecast", "min_quality_threshold",
                    self._current_config["min_quality_threshold"],
                    min(0.9, self._current_config["min_quality_threshold"] + 0.05),
                    "Cost forecast exceeds $1/hr -- raising quality threshold to use cheaper models",
                    auto=False  # Requires human approval
                )
                actions.append(action)
        except Exception as e:
            logger.debug(f"Cost tuning failed: {e}")
        return actions

    def _apply(self, trigger: str, parameter: str, old_val: Any, new_val: Any,
               rationale: str, auto: bool) -> TuningAction:
        """Apply a tuning action."""
        if auto:
            self._current_config[parameter] = new_val

        action = TuningAction(
            timestamp=datetime.now().isoformat(),
            trigger=trigger,
            parameter=parameter,
            old_value=old_val,
            new_value=new_val,
            rationale=rationale,
            auto_applied=auto,
        )
        self._actions.append(action)

        if auto:
            logger.info(f"[AutoTuner] {parameter}: {old_val} → {new_val} ({rationale})")
        else:
            logger.warning(f"[AutoTuner] MANUAL ACTION NEEDED: {parameter}: {old_val} → {new_val} ({rationale})")

        # Log to memory as a DECISION entry
        try:
            from vetinari.memory import MemoryEntry, MemoryEntryType, get_dual_memory_store
            store = get_dual_memory_store()
            import json
            store.remember(MemoryEntry(
                agent="auto_tuner",
                entry_type=MemoryEntryType.DECISION,
                content=json.dumps({"parameter": parameter, "old": old_val, "new": new_val,
                                    "rationale": rationale, "auto": auto}),
                summary=f"AutoTuner: {parameter} {old_val}→{new_val}",
                provenance="auto_tuner",
            ))
        except Exception:
            pass

        return action

    def get_config(self) -> Dict[str, Any]:
        """Get current tuned configuration."""
        return dict(self._current_config)

    def get_history(self) -> List[Dict[str, Any]]:
        """Get tuning action history."""
        from dataclasses import asdict
        return [asdict(a) for a in self._actions]


_auto_tuner: Optional[AutoTuner] = None


def get_auto_tuner() -> AutoTuner:
    global _auto_tuner
    if _auto_tuner is None:
        _auto_tuner = AutoTuner()
    return _auto_tuner
