"""
Alert System for Vetinari Dashboard

Provides threshold-based alerting on top of the MetricsSnapshot produced by DashboardAPI.

Supported metric keys (dot-notation):
    adapters.total_requests
    adapters.total_failed
    adapters.average_latency_ms
    adapters.total_tokens_used
    plan.total_decisions
    plan.approval_rate
    plan.average_risk_score
    plan.average_approval_time_ms

Alert channels:
    log      - emit via Python logging (default)
    email    - SMTP email (configure via VETINARI_SMTP_* env vars)
    webhook  - HTTP POST (configure via VETINARI_WEBHOOK_URL env var)

Usage:
    from vetinari.dashboard.alerts import get_alert_engine, AlertThreshold, AlertCondition, AlertSeverity

    engine = get_alert_engine()
    engine.register_threshold(AlertThreshold(
        name="high-latency",
        metric_key="adapters.average_latency_ms",
        condition=AlertCondition.GREATER_THAN,
        threshold_value=500.0,
        severity=AlertSeverity.HIGH,
        channels=["log"],
    ))
    engine.evaluate_all()
"""

import logging
import os
import smtplib
import threading
import time
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import requests

from vetinari.dashboard.api import DashboardAPI, MetricsSnapshot, get_dashboard_api

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AlertCondition(Enum):
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    EQUALS = "eq"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AlertThreshold:
    """Defines a rule to evaluate against a MetricsSnapshot value."""

    name: str
    metric_key: str          # dot-notation path into MetricsSnapshot.to_dict()
    condition: AlertCondition
    threshold_value: float
    severity: AlertSeverity = AlertSeverity.MEDIUM
    channels: List[str] = field(default_factory=lambda: ["log"])
    # duration_seconds > 0 means the condition must persist that long before firing
    duration_seconds: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metric_key": self.metric_key,
            "condition": self.condition.value,
            "threshold_value": self.threshold_value,
            "severity": self.severity.value,
            "channels": self.channels,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class AlertRecord:
    """An alert that has been triggered."""

    threshold: AlertThreshold
    current_value: float
    trigger_time: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold.to_dict(),
            "current_value": self.current_value,
            "trigger_time": self.trigger_time,
        }


# ---------------------------------------------------------------------------
# Metric resolution
# ---------------------------------------------------------------------------

def _resolve_metric(snapshot_dict: Dict[str, Any], key: str) -> Optional[float]:
    """
    Walk a dot-notation key into the snapshot dictionary and return a float.

    Returns None when the key path does not exist or the value is not numeric.
    """
    parts = key.split(".")
    node: Any = snapshot_dict
    for part in parts:
        if not isinstance(node, dict):
            return None
        node = node.get(part)
        if node is None:
            return None
    if isinstance(node, (int, float)):
        return float(node)
    return None


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------

def _dispatch_log(alert: AlertRecord) -> None:
    level = logging.ERROR if alert.threshold.severity == AlertSeverity.HIGH else logging.WARNING
    logger.log(
        level,
        "ALERT [%s] %s: %s %.4g %s %.4g",
        alert.threshold.severity.value.upper(),
        alert.threshold.name,
        alert.threshold.metric_key,
        alert.current_value,
        alert.threshold.condition.value,
        alert.threshold.threshold_value,
        extra={"alert": alert.to_dict()},
    )


def _dispatch_email(alert: AlertRecord) -> None:
    """Send alert via SMTP email. Configure with VETINARI_SMTP_* env vars."""
    smtp_host = os.environ.get("VETINARI_SMTP_HOST")
    smtp_port = int(os.environ.get("VETINARI_SMTP_PORT", "587"))
    from_addr = os.environ.get("VETINARI_ALERT_FROM")
    to_addr = os.environ.get("VETINARI_ALERT_TO")

    if not smtp_host or not from_addr or not to_addr:
        logger.info(
            "EMAIL: skipping alert '%s' — VETINARI_SMTP_HOST, VETINARI_ALERT_FROM, "
            "and VETINARI_ALERT_TO must all be set",
            alert.threshold.name,
        )
        return

    subject = (
        f"[Vetinari Alert] [{alert.threshold.severity.value.upper()}] "
        f"{alert.threshold.name}"
    )
    body = (
        f"Alert: {alert.threshold.name}\n"
        f"Severity: {alert.threshold.severity.value.upper()}\n"
        f"Metric: {alert.threshold.metric_key}\n"
        f"Current value: {alert.current_value:.4g}\n"
        f"Threshold ({alert.threshold.condition.value}): {alert.threshold.threshold_value:.4g}\n"
        f"Triggered at: {alert.trigger_time}\n"
    )

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            smtp_user = os.environ.get("VETINARI_SMTP_USER")
            smtp_pass = os.environ.get("VETINARI_SMTP_PASS")
            if smtp_user and smtp_pass:
                server.login(smtp_user, smtp_pass)
            server.sendmail(from_addr, [to_addr], msg.as_string())
        logger.info("EMAIL: sent alert '%s' to %s", alert.threshold.name, to_addr)
    except Exception as exc:
        logger.error("EMAIL: failed to send alert '%s': %s", alert.threshold.name, exc)


def _dispatch_webhook(alert: AlertRecord) -> None:
    """POST alert details as JSON. Configure with VETINARI_WEBHOOK_URL env var."""
    url = os.environ.get("VETINARI_WEBHOOK_URL")
    if not url:
        logger.info(
            "WEBHOOK: skipping alert '%s' — VETINARI_WEBHOOK_URL not set",
            alert.threshold.name,
        )
        return

    payload = {
        "name": alert.threshold.name,
        "metric": alert.threshold.metric_key,
        "value": alert.current_value,
        "threshold": alert.threshold.threshold_value,
        "severity": alert.threshold.severity.value,
        "timestamp": alert.trigger_time,
    }

    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        logger.info(
            "WEBHOOK: posted alert '%s' to %s (status %s)",
            alert.threshold.name,
            url,
            response.status_code,
        )
    except Exception as exc:
        logger.error(
            "WEBHOOK: failed to post alert '%s' to %s: %s",
            alert.threshold.name,
            url,
            exc,
        )


DISPATCHERS: Dict[str, Callable[[AlertRecord], None]] = {
    "log": _dispatch_log,
    "email": _dispatch_email,
    "webhook": _dispatch_webhook,
}


# ---------------------------------------------------------------------------
# Alert engine
# ---------------------------------------------------------------------------

class AlertEngine:
    """
    Evaluates registered AlertThreshold rules against live metrics.

    Thread-safe singleton — obtain via get_alert_engine().
    """

    _instance: Optional["AlertEngine"] = None
    _class_lock = threading.Lock()

    def __new__(cls) -> "AlertEngine":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._setup()
        return cls._instance

    # ------------------------------------------------------------------
    # Initialisation (called once by __new__)
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        self._lock = threading.RLock()
        self._thresholds: Dict[str, AlertThreshold] = {}   # keyed by name
        self._active: Dict[str, AlertRecord] = {}          # currently firing
        self._history: List[AlertRecord] = []              # all past firings
        # duration tracking: name -> (first_trigger_time,)
        self._duration_start: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Threshold management
    # ------------------------------------------------------------------

    def register_threshold(self, threshold: AlertThreshold) -> None:
        """Add (or replace) a named threshold."""
        with self._lock:
            self._thresholds[threshold.name] = threshold
            logger.debug("Registered alert threshold '%s'", threshold.name)

    def unregister_threshold(self, name: str) -> bool:
        """Remove a threshold by name. Returns True if it existed."""
        with self._lock:
            existed = name in self._thresholds
            self._thresholds.pop(name, None)
            self._active.pop(name, None)
            self._duration_start.pop(name, None)
            return existed

    def clear_thresholds(self) -> None:
        """Remove all thresholds and reset alert state."""
        with self._lock:
            self._thresholds.clear()
            self._active.clear()
            self._duration_start.clear()
            logger.debug("Cleared all alert thresholds")

    def list_thresholds(self) -> List[AlertThreshold]:
        with self._lock:
            return list(self._thresholds.values())

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_all(self, api: Optional[DashboardAPI] = None) -> List[AlertRecord]:
        """
        Evaluate all registered thresholds against the current metrics snapshot.

        Args:
            api: DashboardAPI instance to use (defaults to the global singleton).

        Returns:
            List of AlertRecord objects that fired in this evaluation cycle.
        """
        if api is None:
            api = get_dashboard_api()

        snapshot: MetricsSnapshot = api.get_latest_metrics()
        snapshot_dict = snapshot.to_dict()
        fired: List[AlertRecord] = []

        with self._lock:
            for name, threshold in list(self._thresholds.items()):
                value = _resolve_metric(snapshot_dict, threshold.metric_key)
                if value is None:
                    logger.debug(
                        "Metric key '%s' not found in snapshot; skipping threshold '%s'",
                        threshold.metric_key,
                        name,
                    )
                    continue

                triggered = self._check_condition(threshold, value)

                if triggered:
                    record = self._handle_triggered(threshold, name, value)
                    if record is not None:
                        fired.append(record)
                else:
                    # Clear duration tracking and active alert when condition no longer holds
                    self._duration_start.pop(name, None)
                    if name in self._active:
                        del self._active[name]
                        logger.info("Alert '%s' cleared (condition no longer met)", name)

        return fired

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_condition(self, threshold: AlertThreshold, value: float) -> bool:
        if threshold.condition == AlertCondition.GREATER_THAN:
            return value > threshold.threshold_value
        if threshold.condition == AlertCondition.LESS_THAN:
            return value < threshold.threshold_value
        if threshold.condition == AlertCondition.EQUALS:
            return value == threshold.threshold_value
        return False

    def _handle_triggered(
        self, threshold: AlertThreshold, name: str, value: float
    ) -> Optional[AlertRecord]:
        """
        Handle a triggered condition, respecting duration requirements.

        Returns an AlertRecord if the alert should fire now, otherwise None.
        """
        now = time.time()

        if threshold.duration_seconds > 0:
            if name not in self._duration_start:
                # Condition just became true — start tracking
                self._duration_start[name] = now
                logger.debug(
                    "Alert '%s' condition met; waiting %.1fs before firing",
                    name,
                    threshold.duration_seconds,
                )
                return None

            elapsed = now - self._duration_start[name]
            if elapsed < threshold.duration_seconds:
                logger.debug(
                    "Alert '%s' duration not yet satisfied (%.1f / %.1fs)",
                    name,
                    elapsed,
                    threshold.duration_seconds,
                )
                return None
            # Duration satisfied — fall through to fire

        # Fire only once until the condition clears (suppression)
        if name in self._active:
            return None

        record = AlertRecord(threshold=threshold, current_value=value, trigger_time=now)
        self._active[name] = record
        self._history.append(record)
        self._dispatch(record)
        return record

    def _dispatch(self, alert: AlertRecord) -> None:
        for channel in alert.threshold.channels:
            dispatcher = DISPATCHERS.get(channel)
            if dispatcher:
                try:
                    dispatcher(alert)
                except Exception as exc:  # pragma: no cover
                    logger.error("Dispatcher '%s' raised an error: %s", channel, exc)
            else:
                logger.warning("Unknown alert channel '%s'", channel)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_active_alerts(self) -> List[AlertRecord]:
        """Return currently active (firing) alerts."""
        with self._lock:
            return list(self._active.values())

    def get_history(self) -> List[AlertRecord]:
        """Return all alerts that have ever fired in this session."""
        with self._lock:
            return list(self._history)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "registered_thresholds": len(self._thresholds),
                "active_alerts": len(self._active),
                "total_fired": len(self._history),
            }


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

def get_alert_engine() -> AlertEngine:
    """Return the global AlertEngine singleton."""
    return AlertEngine()


def reset_alert_engine() -> None:
    """Destroy the singleton (for tests / clean shutdown)."""
    with AlertEngine._class_lock:
        AlertEngine._instance = None
    logger.debug("AlertEngine singleton reset")
