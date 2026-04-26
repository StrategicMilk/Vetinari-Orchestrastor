"""Alert System for Vetinari Dashboard.

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

from __future__ import annotations

import json
import logging
import os
import smtplib
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any

from vetinari.constants import ALERT_SEND_TIMEOUT
from vetinari.dashboard.api import DashboardAPI, MetricsSnapshot, get_dashboard_api
from vetinari.database import get_connection
from vetinari.http import create_session
from vetinari.utils.serialization import dataclass_to_dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AlertSeverity(Enum):
    """Alert severity."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AlertCondition(Enum):
    """Alert condition."""

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
    metric_key: str  # dot-notation path into MetricsSnapshot.to_dict()
    condition: AlertCondition
    threshold_value: float
    severity: AlertSeverity = AlertSeverity.MEDIUM
    channels: list[str] = field(default_factory=lambda: ["log"])
    # duration_seconds > 0 means the condition must persist that long before firing
    duration_seconds: int = 0

    def __repr__(self) -> str:
        return f"AlertThreshold(name={self.name!r}, metric_key={self.metric_key!r}, severity={self.severity!r})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize this alert rule's name, metric key, condition, severity, channels, and duration."""
        return dataclass_to_dict(self)


@dataclass(frozen=True)
class AlertRecord:
    """An alert that has been triggered."""

    threshold: AlertThreshold
    current_value: float
    trigger_time: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return f"AlertRecord(threshold={self.threshold.name!r}, current_value={self.current_value!r})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize this fired alert including its threshold definition, observed value, and trigger time."""
        return dataclass_to_dict(self)


# ---------------------------------------------------------------------------
# Metric resolution
# ---------------------------------------------------------------------------


def _resolve_metric(snapshot_dict: dict[str, Any], key: str) -> float | None:
    """Walk a dot-notation key into the snapshot dictionary and return a float.

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

    subject = f"[Vetinari Alert] [{alert.threshold.severity.value.upper()}] {alert.threshold.name}"
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
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            smtp_user = os.environ.get("VETINARI_SMTP_USER")
            smtp_pass = os.environ.get("VETINARI_SMTP_PASS")
            if smtp_user and smtp_pass:
                server.login(smtp_user, smtp_pass)
            server.sendmail(from_addr, [to_addr], msg.as_string())
        logger.info("EMAIL: sent alert '%s' to %s", alert.threshold.name, to_addr)
    except Exception:
        logger.exception("EMAIL: failed to send alert '%s'", alert.threshold.name)


_WEBHOOK_MAX_ATTEMPTS = 3
_WEBHOOK_BACKOFF_BASE = 1.0  # seconds; doubles on each retry


def _dispatch_webhook(alert: AlertRecord) -> None:
    """POST alert details as JSON with up to 3 attempts and exponential backoff.

    Retries on any exception (network error, non-2xx response, timeout).
    Backoff delays: 1 s after attempt 1, 2 s after attempt 2.
    Configure with VETINARI_WEBHOOK_URL env var.

    Args:
        alert: The fired AlertRecord to dispatch.
    """
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

    last_exc: Exception | None = None
    for attempt in range(1, _WEBHOOK_MAX_ATTEMPTS + 1):
        try:
            with create_session() as session:
                response = session.post(url, json=payload, timeout=ALERT_SEND_TIMEOUT)
            response.raise_for_status()
            logger.info(
                "WEBHOOK: posted alert '%s' to %s (status %s, attempt %d/%d)",
                alert.threshold.name,
                url,
                response.status_code,
                attempt,
                _WEBHOOK_MAX_ATTEMPTS,
            )
            return
        except Exception as exc:
            last_exc = exc
            if attempt < _WEBHOOK_MAX_ATTEMPTS:
                delay = _WEBHOOK_BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "WEBHOOK: attempt %d/%d failed for alert '%s': %s — retrying in %.1fs",
                    attempt,
                    _WEBHOOK_MAX_ATTEMPTS,
                    alert.threshold.name,
                    exc,
                    delay,
                )
                time.sleep(delay)

    logger.error(
        "WEBHOOK: all %d attempts failed for alert '%s' to %s: %s",
        _WEBHOOK_MAX_ATTEMPTS,
        alert.threshold.name,
        url,
        last_exc,
    )


def _dispatch_dashboard(alert: AlertRecord) -> None:
    """Deliver an alert to the dashboard via the NotificationManager.

    Uses HIGH priority so the notification appears prominently in the dashboard
    notification panel and is delivered via the desktop channel if configured.
    Failures are swallowed so a missing NotificationManager does not interrupt
    alert processing.

    Args:
        alert: The fired AlertRecord to dispatch.
    """
    try:
        from vetinari.notifications.manager import get_notification_manager
        from vetinari.types import NotificationPriority

        mgr = get_notification_manager()
        severity_label = alert.threshold.severity.value.upper()
        title = f"[{severity_label}] {alert.threshold.name}"
        body = (
            f"Metric '{alert.threshold.metric_key}' is {alert.current_value:.4g} "
            f"({alert.threshold.condition.value} {alert.threshold.threshold_value:.4g})"
        )
        priority = (
            NotificationPriority.CRITICAL
            if alert.threshold.severity == AlertSeverity.HIGH
            else NotificationPriority.HIGH
        )
        mgr.notify(
            title=title,
            body=body,
            priority=priority,
            action_type="alert",
            metadata=alert.to_dict(),
        )
        logger.debug(
            "DASHBOARD: dispatched alert '%s' via NotificationManager",
            alert.threshold.name,
        )
    except Exception as exc:
        logger.warning(
            "DASHBOARD: could not dispatch alert '%s' to NotificationManager — dashboard notification skipped: %s",
            alert.threshold.name,
            exc,
        )


DISPATCHERS: dict[str, Callable[[AlertRecord], None]] = {
    "log": _dispatch_log,
    "email": _dispatch_email,
    "webhook": _dispatch_webhook,
    "dashboard": _dispatch_dashboard,
}


# ---------------------------------------------------------------------------
# Alert engine
# ---------------------------------------------------------------------------


class AlertEngine:
    """Evaluates registered AlertThreshold rules against live metrics.

    Thread-safe singleton — obtain via get_alert_engine().
    """

    _instance: AlertEngine | None = None
    _class_lock = threading.Lock()

    def __new__(cls) -> AlertEngine:
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
        self._thresholds: dict[str, AlertThreshold] = {}  # keyed by name
        self._active: dict[str, AlertRecord] = {}  # currently firing
        self._history: deque[AlertRecord] = deque(maxlen=500)  # all past firings
        # duration tracking: name -> (first_trigger_time,)
        self._duration_start: dict[str, float] = {}
        # _custom_db_path is set when configure_persistence() is called with a path;
        # when None, all DB operations use the unified get_connection() (ADR-0072).
        self._custom_db_path: Path | None = None

    def configure_persistence(self, db_path: Path | str | None = None) -> None:
        """Enable SQLite-backed alert history persistence.

        When ``db_path`` is provided, that specific file is used (useful for
        test isolation). When omitted, the unified database via
        ``get_connection()`` is used (ADR-0072).

        Creates the alert_history table if it does not exist and loads any
        previously persisted alerts into the in-memory history.

        Args:
            db_path: Path to a specific SQLite database file, or None to use
                the unified database.
        """
        self._custom_db_path = Path(db_path) if db_path is not None else None
        if self._custom_db_path is not None:
            self._custom_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._load_history()

    def _init_db(self) -> None:
        """Create the alert_history table if it does not exist."""
        try:
            if self._custom_db_path is not None:
                import contextlib
                import sqlite3 as _sqlite3

                with contextlib.closing(_sqlite3.connect(str(self._custom_db_path))) as conn:
                    conn.execute(
                        """CREATE TABLE IF NOT EXISTS alert_history (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            threshold_name TEXT NOT NULL,
                            metric_key TEXT NOT NULL,
                            condition TEXT NOT NULL,
                            threshold_value REAL NOT NULL,
                            severity TEXT NOT NULL,
                            channels TEXT NOT NULL,
                            current_value REAL NOT NULL,
                            trigger_time REAL NOT NULL
                        )"""
                    )
                    conn.commit()
            else:
                conn = get_connection()
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS alert_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        threshold_name TEXT NOT NULL,
                        metric_key TEXT NOT NULL,
                        condition TEXT NOT NULL,
                        threshold_value REAL NOT NULL,
                        severity TEXT NOT NULL,
                        channels TEXT NOT NULL,
                        current_value REAL NOT NULL,
                        trigger_time REAL NOT NULL
                    )"""
                )
                conn.commit()
        except Exception:
            logger.exception("Failed to initialize alert_history table")

    def _persist_alert(self, record: AlertRecord) -> None:
        """Write a single alert record to SQLite."""
        _INSERT_SQL = (
            "INSERT INTO alert_history"
            " (threshold_name, metric_key, condition, threshold_value,"
            "  severity, channels, current_value, trigger_time)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )
        _params = (
            record.threshold.name,
            record.threshold.metric_key,
            record.threshold.condition.value,
            record.threshold.threshold_value,
            record.threshold.severity.value,
            json.dumps(record.threshold.channels),
            record.current_value,
            record.trigger_time,
        )
        try:
            if self._custom_db_path is not None:
                import contextlib
                import sqlite3 as _sqlite3

                with contextlib.closing(_sqlite3.connect(str(self._custom_db_path))) as conn:
                    conn.execute(_INSERT_SQL, _params)
                    conn.commit()
            else:
                conn = get_connection()
                conn.execute(_INSERT_SQL, _params)
                conn.commit()
        except Exception:
            logger.exception("Failed to persist alert '%s' to SQLite", record.threshold.name)

    def _load_history(self) -> None:
        """Load persisted alert history from SQLite into memory."""
        _SELECT_SQL = (
            "SELECT threshold_name, metric_key, condition,"
            " threshold_value, severity, channels,"
            " current_value, trigger_time"
            " FROM alert_history ORDER BY trigger_time"
        )
        try:
            if self._custom_db_path is not None:
                import contextlib
                import sqlite3 as _sqlite3

                with contextlib.closing(_sqlite3.connect(str(self._custom_db_path))) as conn:
                    rows = conn.execute(_SELECT_SQL).fetchall()
            else:
                rows = get_connection().execute(_SELECT_SQL).fetchall()
            with self._lock:
                for row in rows:
                    threshold = AlertThreshold(
                        name=row[0],
                        metric_key=row[1],
                        condition=AlertCondition(row[2]),
                        threshold_value=row[3],
                        severity=AlertSeverity(row[4]),
                        channels=json.loads(row[5]),
                    )
                    record = AlertRecord(
                        threshold=threshold,
                        current_value=row[6],
                        trigger_time=row[7],
                    )
                    self._history.append(record)
            logger.debug("Loaded %d persisted alerts from SQLite", len(rows))
        except Exception:
            logger.exception("Failed to load alert history from SQLite")

    # ------------------------------------------------------------------
    # Threshold management
    # ------------------------------------------------------------------

    def register_threshold(self, threshold: AlertThreshold) -> None:
        """Add (or replace) a named threshold."""
        with self._lock:
            self._thresholds[threshold.name] = threshold
            logger.debug("Registered alert threshold '%s'", threshold.name)

    def unregister_threshold(self, name: str) -> bool:
        """Remove a threshold by name. Returns True if it existed.

        Returns:
            True if successful, False otherwise.
        """
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

    def list_thresholds(self) -> list[AlertThreshold]:
        """List thresholds.

        Returns:
            List of results.
        """
        with self._lock:
            return list(self._thresholds.values())

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_all(self, api: DashboardAPI | None = None) -> list[AlertRecord]:
        """Evaluate all registered thresholds against the current metrics snapshot.

        Args:
            api: DashboardAPI instance to use (defaults to the global singleton).

        Returns:
            List of AlertRecord objects that fired in this evaluation cycle.
        """
        if api is None:
            api = get_dashboard_api()

        snapshot: MetricsSnapshot = api.get_latest_metrics()
        snapshot_dict = snapshot.to_dict()
        fired: list[AlertRecord] = []

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

    def _handle_triggered(self, threshold: AlertThreshold, name: str, value: float) -> AlertRecord | None:
        """Handle a triggered condition, respecting duration requirements.

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
        self._persist_alert(record)
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

    def get_active_alerts(self) -> list[AlertRecord]:
        """Return currently active (firing) alerts.

        Returns:
            List of results.
        """
        with self._lock:
            return list(self._active.values())

    def get_history(self) -> list[AlertRecord]:
        """Return all alerts that have ever fired in this session.

        Returns:
            List of results.
        """
        with self._lock:
            return list(self._history)

    def evaluate_anomaly(self, event: Any) -> None:
        """Handle an AnomalyDetected event from the EventBus.

        Registers the anomaly as an active alert and logs it for dashboard
        visibility. This bridges the anomaly detection subsystem with the
        alerting engine.

        Args:
            event: An ``AnomalyDetected`` event from the EventBus.
        """
        agent_type = getattr(event, "agent_type", "unknown")
        anomaly_type = getattr(event, "anomaly_type", "unknown")
        score = getattr(event, "score", 0.0)
        detectors = getattr(event, "triggered_detectors", [])
        timestamp = getattr(event, "timestamp", time.time())

        logger.warning(
            "[AlertEngine] Anomaly detected — agent=%s, type=%s, score=%.3f, detectors=%s",
            agent_type,
            anomaly_type,
            score,
            detectors,
        )

        # Construct a proper AlertThreshold + AlertRecord for the anomaly
        _severity = AlertSeverity.MEDIUM if score < 0.8 else AlertSeverity.HIGH
        _threshold_name = f"anomaly:{agent_type}:{anomaly_type}"
        threshold = AlertThreshold(
            name=_threshold_name,
            metric_key=f"anomaly.{anomaly_type}",
            condition=AlertCondition.GREATER_THAN,
            threshold_value=0.0,
            severity=_severity,
            channels=["log", "dashboard"],
        )
        record = AlertRecord(
            threshold=threshold,
            current_value=score,
            trigger_time=timestamp,
        )
        with self._lock:
            self._active[_threshold_name] = record
            self._history.append(record)
        self._persist_alert(record)

    def get_stats(self) -> dict[str, Any]:
        """Summarise current alert engine state.

        Returns:
            Dictionary with ``registered_thresholds`` (number of configured
            alert rules), ``active_alerts`` (rules currently in a fired state),
            and ``total_fired`` (cumulative count of all firings this session).
        """
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
