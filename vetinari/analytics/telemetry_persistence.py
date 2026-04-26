"""Telemetry persistence - snapshots telemetry to SQLite and fires threshold alerts.

Runs a background thread that wakes every ``interval_s`` seconds, calls
``TelemetryCollector.get_summary()``, serialises the result to JSON, and
inserts a row into the ``telemetry_snapshots`` table.  Old snapshots outside
the retention window are pruned on each cycle.

If the computed error rate exceeds ``error_rate_threshold`` or the p95 model
latency exceeds ``p95_latency_threshold_ms``, a WARNING is logged and an event
is published on the ``EventBus``.

This is step 3 of the observability pipeline:
  Collect (TelemetryCollector) -> Persist (this module) -> Expose (dashboard_metrics_api).
"""

from __future__ import annotations

import builtins
import contextlib
import json
import logging
import threading
import time
import weakref
from typing import Any

logger = logging.getLogger(__name__)

# Default configuration constants
_DEFAULT_INTERVAL_S: int = 60  # How often to flush a snapshot to SQLite
_DEFAULT_RETENTION_DAYS: int = 30  # Snapshots older than this are deleted
_DEFAULT_ERROR_RATE_THRESHOLD: float = 10.0  # Percent - alert when exceeded
_DEFAULT_P95_LATENCY_THRESHOLD_MS: float = 5000.0  # ms - alert when exceeded

# Schema - added to the unified database on first start()
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS telemetry_snapshots (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL    NOT NULL,
    data      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tel_snapshots_ts ON telemetry_snapshots(timestamp);
"""

_ACTIVE_INSTANCES_ATTR = "_vetinari_telemetry_persistence_instances"


def _active_instances() -> weakref.WeakSet[Any]:
    """Return the process-global registry of started persistence workers."""
    instances = getattr(builtins, _ACTIVE_INSTANCES_ATTR, None)
    if instances is None:
        instances = weakref.WeakSet()
        setattr(builtins, _ACTIVE_INSTANCES_ATTR, instances)
    return instances


class TelemetryPersistence:
    """Periodically persists telemetry snapshots and enforces alert thresholds.

    Starts a daemon background thread on ``start()``.  The thread wakes every
    ``interval_s`` seconds, snapshots ``TelemetryCollector.get_summary()`` to
    the ``telemetry_snapshots`` SQLite table, and checks alert thresholds.

    Snapshots older than ``retention_days`` are deleted on each cycle to bound
    disk growth.

    Args:
        interval_s: Seconds between persist cycles (default 60).
        retention_days: How many days of snapshots to keep (default 30).
        error_rate_threshold: Error-rate percentage that triggers a WARNING
            and an ``AlertTriggered`` event (default 10.0).
        p95_latency_threshold_ms: p95 model latency in milliseconds that
            triggers a WARNING and event (default 5000.0).
    """

    def __init__(
        self,
        interval_s: int = _DEFAULT_INTERVAL_S,
        retention_days: int = _DEFAULT_RETENTION_DAYS,
        error_rate_threshold: float = _DEFAULT_ERROR_RATE_THRESHOLD,
        p95_latency_threshold_ms: float = _DEFAULT_P95_LATENCY_THRESHOLD_MS,
    ) -> None:
        self._interval_s = interval_s
        self._retention_days = retention_days
        self._error_rate_threshold = error_rate_threshold
        self._p95_latency_threshold_ms = p95_latency_threshold_ms

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Lock guards _thread start/stop so two callers cannot race on start()
        self._lifecycle_lock = threading.Lock()
        self._last_request_total: int | None = None
        self._last_request_sample_ts: float | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background persist thread.

        Idempotent - does nothing if the thread is already running.  Creates
        the ``telemetry_snapshots`` table if it does not yet exist.
        """
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            schema_ok = self._ensure_schema()
            self._thread = threading.Thread(
                target=self._run,
                name="telemetry-persistence",
                daemon=True,  # dies with the process - no blocking shutdown needed
            )
            self._thread.start()
            _active_instances().add(self)
            if schema_ok:
                logger.info(
                    "TelemetryPersistence started (interval=%ds, retention=%dd)",
                    self._interval_s,
                    self._retention_days,
                )
            else:
                logger.warning(
                    "TelemetryPersistence started but schema bootstrap failed - "
                    "snapshots will not be persisted until the database is available "
                    "(interval=%ds, retention=%dd)",
                    self._interval_s,
                    self._retention_days,
                )

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it to exit.

        Blocks for up to ``interval_s + 2`` seconds.  Safe to call multiple
        times or before ``start()``.
        """
        self._stop_event.set()
        with self._lifecycle_lock:
            if self._thread is not None:
                self._thread.join(timeout=self._interval_s + 2)
                self._thread = None
        with contextlib.suppress(Exception):
            _active_instances().discard(self)
        logger.info("TelemetryPersistence stopped")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> bool:
        """Apply the telemetry_snapshots DDL to the unified database.

        Safe to call multiple times - uses ``CREATE TABLE IF NOT EXISTS``.

        Returns:
            True if the schema was applied successfully, False on failure.
        """
        try:
            from vetinari.database import get_connection

            conn = get_connection()
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
            return True
        except Exception as exc:
            logger.warning(
                "Could not create telemetry_snapshots table - snapshots will not be persisted: %s",
                exc,
            )
            return False

    def _run(self) -> None:
        """Main loop: sleep, persist, check alerts, prune."""
        while not self._stop_event.wait(timeout=self._interval_s):
            try:
                self._persist_snapshot()
            except Exception as exc:
                logger.warning(
                    "Telemetry persist cycle failed - snapshot skipped, will retry next cycle: %s",
                    exc,
                )

    def _persist_snapshot(self) -> None:
        """Take a snapshot from TelemetryCollector and write it to SQLite."""
        from vetinari.telemetry import get_telemetry_collector

        # Database tests frequently swap the backing SQLite file mid-process.
        # Re-applying the telemetry DDL on the current thread keeps this write
        # path resilient when the thread-local connection points at a fresh DB.
        self._ensure_schema()

        collector = get_telemetry_collector()
        summary = collector.get_summary()

        # Enrich summary with adapter details for alert threshold checks
        adapter_metrics = collector.get_adapter_metrics()
        enriched = dict(summary)
        enriched["adapter_details"] = {
            key: {
                "total_requests": m.total_requests,
                "failed_requests": m.failed_requests,
                "success_rate": m.success_rate,
                "avg_latency_ms": m.avg_latency_ms,
                "min_latency_ms": m.min_latency_ms if m.min_latency_ms != float("inf") else 0.0,
                "max_latency_ms": m.max_latency_ms,
                "total_tokens_used": m.total_tokens_used,
            }
            for key, m in adapter_metrics.items()
        }

        now = time.time()
        data_json = json.dumps(enriched, default=str)

        try:
            from vetinari.database import get_connection

            conn = get_connection()
            conn.execute(
                "INSERT INTO telemetry_snapshots (timestamp, data) VALUES (?, ?)",
                (now, data_json),
            )
            conn.commit()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Telemetry snapshot persisted at %.3f", now)
        except Exception as exc:
            logger.warning(
                "Failed to write telemetry snapshot to database - metrics not persisted: %s",
                exc,
            )

        self._check_alert_thresholds(enriched)
        self._prune_old_snapshots(now)
        self._feed_periodic_metrics(enriched, now=now)

    def _check_alert_thresholds(self, summary: dict[str, Any]) -> None:
        """Emit WARNING logs and events when error-rate or p95 latency breach thresholds.

        Args:
            summary: The enriched telemetry summary dict from the current cycle.
        """
        # -- Error rate check across all adapters --
        adapter_details: dict[str, Any] = summary.get("adapter_details", {})
        total_requests = 0
        total_failed = 0
        for _key, detail in adapter_details.items():
            if isinstance(detail, dict):
                total_requests += int(detail.get("total_requests", 0))
                total_failed += int(detail.get("failed_requests", 0))

        if total_requests == 0 and adapter_details:
            logger.warning(
                "Telemetry alert: zero requests observed across %d adapters - telemetry or inference may be stalled",
                len(adapter_details),
            )
            self._emit_alert_event(
                "zero_traffic_blackout",
                "No requests observed across telemetry adapters",
                {"adapter_count": len(adapter_details)},
            )
        elif total_requests > 0:
            error_rate = (total_failed / total_requests) * 100.0
            if error_rate > self._error_rate_threshold:
                logger.warning(
                    "Telemetry alert: error rate %.1f%% exceeds threshold %.1f%% "
                    "(%d failed / %d total requests) - investigate adapter failures",
                    error_rate,
                    self._error_rate_threshold,
                    total_failed,
                    total_requests,
                )
                self._emit_alert_event(
                    "high_error_rate",
                    f"Error rate {error_rate:.1f}% exceeds threshold {self._error_rate_threshold:.1f}%",
                    {"error_rate": error_rate, "threshold": self._error_rate_threshold},
                )

        # -- p95 latency check using MetricsCollector histogram --
        try:
            from vetinari.metrics import get_metrics

            stats = get_metrics().get_histogram_stats("vetinari.model.latency")
            if stats is not None:
                p95 = stats.get("p95", 0.0)
                if p95 > self._p95_latency_threshold_ms:
                    logger.warning(
                        "Telemetry alert: p95 model latency %.1fms exceeds threshold %.1fms "
                        "- model may be overloaded or undersized",
                        p95,
                        self._p95_latency_threshold_ms,
                    )
                    self._emit_alert_event(
                        "high_p95_latency",
                        f"p95 latency {p95:.1f}ms exceeds threshold {self._p95_latency_threshold_ms:.1f}ms",
                        {"p95_latency_ms": p95, "threshold_ms": self._p95_latency_threshold_ms},
                    )
        except Exception as exc:
            logger.warning("p95 latency alert check skipped: %s - high-latency alerts may be missed", exc)

    def _emit_alert_event(self, alert_type: str, message: str, metadata: dict[str, Any]) -> None:
        """Publish a telemetry alert event on the EventBus.

        Failures are swallowed - alerting must never crash the persist cycle.

        Args:
            alert_type: Short identifier for the alert (e.g. ``"high_error_rate"``).
            message: Human-readable description of what breached.
            metadata: Numeric context (rates, thresholds) for the alert.
        """
        try:
            from vetinari.events import TelemetryAlertEvent, get_event_bus

            get_event_bus().publish(
                TelemetryAlertEvent(
                    event_type="",  # overwritten by __post_init__
                    timestamp=time.time(),
                    alert_type=alert_type,
                    message=message,
                    metadata=metadata,
                )
            )
        except Exception as exc:
            logger.warning(
                "Could not publish telemetry alert event for %s: %s - alert not delivered to subscribers",
                alert_type,
                exc,
            )

    def _feed_periodic_metrics(self, snapshot_data: dict[str, Any], now: float | None = None) -> None:
        """Forward snapshot metrics to the analytics forecaster via the wiring module.

        Called after every successful persist cycle so the capacity forecaster
        receives throughput and latency signals without a separate scheduler.

        Args:
            snapshot_data: The enriched telemetry summary dict from this cycle.
            now: Snapshot timestamp used to compute interval request rates.
        """
        try:
            from vetinari.analytics.wiring import record_periodic_metrics

            # Derive avg_latency_ms from adapter_details (weighted mean across adapters)
            adapter_details: dict[str, Any] = snapshot_data.get("adapter_details", {})
            total_requests = 0
            total_latency_sum = 0.0
            for detail in adapter_details.values():
                if isinstance(detail, dict):
                    reqs = int(detail.get("total_requests", 0))
                    avg_lat = float(detail.get("avg_latency_ms", 0.0))
                    total_requests += reqs
                    total_latency_sum += avg_lat * reqs

            avg_latency_ms = (total_latency_sum / total_requests) if total_requests > 0 else 0.0
            current_total = int(snapshot_data.get("session_requests") or total_requests)
            sample_ts = time.time() if now is None else now
            request_rate = 0.0
            if self._last_request_total is not None and self._last_request_sample_ts is not None:
                elapsed = max(sample_ts - self._last_request_sample_ts, 0.001)
                request_delta = current_total - self._last_request_total
                if request_delta < 0:
                    request_delta = current_total
                request_rate = max(float(request_delta), 0.0) / elapsed
            self._last_request_total = current_total
            self._last_request_sample_ts = sample_ts

            queue_depth = 0
            for key in ("queue_depth", "pending_requests", "backlog"):
                if key in snapshot_data:
                    queue_depth = max(int(snapshot_data.get(key) or 0), 0)
                    break

            record_periodic_metrics(
                request_rate=request_rate,
                avg_latency_ms=avg_latency_ms,
                queue_depth=queue_depth,
            )
        except Exception as exc:
            logger.warning(
                "Periodic metrics feed to forecaster skipped: %s - forecaster predictions may be stale",
                exc,
            )

    def _prune_old_snapshots(self, now: float) -> None:
        """Delete snapshots older than ``retention_days`` from the database.

        Args:
            now: Current Unix timestamp used to compute the retention cutoff.
        """
        cutoff = now - (self._retention_days * 86400)
        self._ensure_schema()
        try:
            from vetinari.database import get_connection

            conn = get_connection()
            cursor = conn.execute(
                "DELETE FROM telemetry_snapshots WHERE timestamp < ?",
                (cutoff,),
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(
                    "Pruned %d telemetry snapshots older than %d days",
                    cursor.rowcount,
                    self._retention_days,
                )
        except Exception as exc:
            logger.warning(
                "Failed to prune old telemetry snapshots - storage may grow unbounded: %s",
                exc,
            )

    # ------------------------------------------------------------------
    # Read path (used by dashboard API)
    # ------------------------------------------------------------------

    def get_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the most recent telemetry snapshots from the database.

        Args:
            limit: Maximum number of rows to return (default 100).

        Returns:
            List of dicts with ``id``, ``timestamp`` (float), and ``data``
            (the parsed JSON summary).  Ordered newest-first.
        """
        self._ensure_schema()
        try:
            from vetinari.database import get_connection

            conn = get_connection()
            rows = conn.execute(
                "SELECT id, timestamp, data FROM telemetry_snapshots ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        except Exception as exc:
            logger.warning(
                "Failed to read telemetry history from database - returning empty list: %s",
                exc,
            )
            return []

        results: list[dict[str, Any]] = []
        for row in rows:
            try:
                results.append(
                    {
                        "id": row[0],
                        "timestamp": row[1],
                        "data": json.loads(row[2]),
                    }
                )
            except Exception as exc:
                logger.warning(
                    "Skipping malformed telemetry snapshot row id=%s - JSON parse failed: %s",
                    row[0],
                    exc,
                )
        return results


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_persistence: TelemetryPersistence | None = None
_persistence_lock = threading.Lock()


def get_telemetry_persistence(
    interval_s: int = _DEFAULT_INTERVAL_S,
    retention_days: int = _DEFAULT_RETENTION_DAYS,
    error_rate_threshold: float = _DEFAULT_ERROR_RATE_THRESHOLD,
    p95_latency_threshold_ms: float = _DEFAULT_P95_LATENCY_THRESHOLD_MS,
) -> TelemetryPersistence:
    """Return the process-wide TelemetryPersistence singleton.

    Creates the instance on first call using the supplied parameters.
    Subsequent calls ignore the parameters and return the existing instance.

    Args:
        interval_s: Seconds between persist cycles (used only on first call).
        retention_days: Days of snapshot retention (used only on first call).
        error_rate_threshold: Error-rate alert threshold in percent.
        p95_latency_threshold_ms: p95 latency alert threshold in milliseconds.

    Returns:
        The shared ``TelemetryPersistence`` instance.
    """
    global _persistence
    if _persistence is None:
        with _persistence_lock:
            if _persistence is None:
                _persistence = TelemetryPersistence(
                    interval_s=interval_s,
                    retention_days=retention_days,
                    error_rate_threshold=error_rate_threshold,
                    p95_latency_threshold_ms=p95_latency_threshold_ms,
                )
    return _persistence


def reset_telemetry_persistence() -> None:
    """Destroy the singleton for test isolation.

    Stops the background thread if running before clearing the reference.
    """
    global _persistence
    with _persistence_lock:
        for instance in list(_active_instances()):
            with contextlib.suppress(Exception):
                instance.stop()
        if _persistence is not None:
            _persistence.stop()
            _persistence = None
