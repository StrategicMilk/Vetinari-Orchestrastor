"""Tests for vetinari.analytics.telemetry_persistence module."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from vetinari.analytics.telemetry_persistence import (
    TelemetryPersistence,
    get_telemetry_persistence,
    reset_telemetry_persistence,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure the singleton is reset between tests."""
    reset_telemetry_persistence()
    yield
    reset_telemetry_persistence()


@pytest.fixture
def in_memory_db(tmp_path, monkeypatch):
    """Redirect database operations to a temporary SQLite file."""
    db_path = tmp_path / "test_telemetry.db"
    monkeypatch.setenv("VETINARI_DB_PATH", str(db_path))

    # Reset the database module's schema-init flag so it re-runs
    import vetinari.database as dbmod

    dbmod.reset_for_testing()
    yield db_path
    dbmod.reset_for_testing()


@pytest.fixture
def mock_telemetry_summary():
    """Return a minimal TelemetryCollector.get_summary() dict."""
    return {
        "total_tokens_used": 1000,
        "total_cost_usd": 0.0,
        "by_model": {"gpt-4": {"tokens": 1000, "requests": 5}},
        "by_provider": {"openai": {"tokens": 1000, "requests": 5}},
        "session_requests": 5,
    }


@pytest.fixture
def mock_adapter_metrics():
    """Return a minimal adapter_metrics dict with one entry."""
    m = MagicMock()
    m.total_requests = 10
    m.failed_requests = 0
    m.success_rate = 100.0
    m.avg_latency_ms = 120.0
    m.min_latency_ms = 80.0
    m.max_latency_ms = 200.0
    return {"openai:gpt-4": m}


# ---------------------------------------------------------------------------
# TelemetryPersistence — lifecycle
# ---------------------------------------------------------------------------


class TestTelemetryPersistenceLifecycle:
    """Tests for start/stop thread management."""

    def test_start_creates_daemon_thread(self, in_memory_db):
        """start() creates a daemon background thread."""
        p = TelemetryPersistence(interval_s=9999)
        p.start()
        assert p._thread is not None
        assert p._thread.is_alive()
        assert p._thread.daemon is True
        p.stop()

    def test_start_is_idempotent(self, in_memory_db):
        """Calling start() twice does not create a second thread."""
        p = TelemetryPersistence(interval_s=9999)
        p.start()
        thread_id = id(p._thread)
        p.start()
        assert id(p._thread) == thread_id
        p.stop()

    def test_stop_terminates_thread(self, in_memory_db):
        """stop() causes the background thread to exit."""
        p = TelemetryPersistence(interval_s=9999)
        p.start()
        thread = p._thread
        p.stop()
        assert thread is not None
        thread.join(timeout=3)
        assert not thread.is_alive()

    def test_stop_before_start_is_safe(self):
        """stop() before start() does not raise and leaves _thread as None."""
        p = TelemetryPersistence(interval_s=9999)
        p.stop()
        assert p._thread is None


# ---------------------------------------------------------------------------
# TelemetryPersistence — schema creation
# ---------------------------------------------------------------------------


class TestTelemetryPersistenceSchema:
    """Tests that the schema is created correctly."""

    def test_ensure_schema_creates_table(self, in_memory_db):
        """_ensure_schema() creates the telemetry_snapshots table."""
        p = TelemetryPersistence(interval_s=9999)
        p._ensure_schema()

        # Check the table exists via direct connection
        conn = sqlite3.connect(str(in_memory_db))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='telemetry_snapshots'")
        assert cursor.fetchone() is not None, "telemetry_snapshots table not created"
        conn.close()


# ---------------------------------------------------------------------------
# TelemetryPersistence — _persist_snapshot
# ---------------------------------------------------------------------------


class TestPersistSnapshot:
    """Tests that _persist_snapshot writes the expected row."""

    def test_persist_snapshot_writes_row(self, in_memory_db, mock_telemetry_summary, mock_adapter_metrics):
        """_persist_snapshot() inserts one row into telemetry_snapshots."""
        p = TelemetryPersistence(interval_s=9999)
        p._ensure_schema()

        with (
            patch("vetinari.telemetry.get_telemetry_collector", create=True) as mock_tc,
        ):
            collector = MagicMock()
            collector.get_summary.return_value = mock_telemetry_summary
            collector.get_adapter_metrics.return_value = mock_adapter_metrics
            mock_tc.return_value = collector

            p._persist_snapshot()

        conn = sqlite3.connect(str(in_memory_db))
        rows = conn.execute("SELECT id, timestamp, data FROM telemetry_snapshots").fetchall()
        conn.close()

        assert len(rows) == 1
        row = rows[0]
        assert row[1] > 0  # timestamp is a non-zero float
        data = json.loads(row[2])
        assert data["total_tokens_used"] == 1000
        assert "adapter_details" in data

    def test_persist_snapshot_reapplies_schema_after_db_reset(
        self,
        tmp_path,
        monkeypatch,
        mock_telemetry_summary,
        mock_adapter_metrics,
    ):
        """Writes succeed after the backing DB is swapped mid-process."""
        import vetinari.database as dbmod

        first_db = tmp_path / "first.db"
        second_db = tmp_path / "second.db"
        p = TelemetryPersistence(interval_s=9999)

        monkeypatch.setenv("VETINARI_DB_PATH", str(first_db))
        dbmod.reset_for_testing()
        p._ensure_schema()

        monkeypatch.setenv("VETINARI_DB_PATH", str(second_db))
        dbmod.reset_for_testing()
        dbmod.get_connection()

        with patch(
            "vetinari.telemetry.get_telemetry_collector",
            create=True,
        ) as mock_tc:
            collector = MagicMock()
            collector.get_summary.return_value = mock_telemetry_summary
            collector.get_adapter_metrics.return_value = mock_adapter_metrics
            mock_tc.return_value = collector

            p._persist_snapshot()

        conn = sqlite3.connect(str(second_db))
        rows = conn.execute("SELECT id, timestamp, data FROM telemetry_snapshots").fetchall()
        conn.close()

        assert len(rows) == 1

    def test_persist_snapshot_includes_adapter_details(
        self, in_memory_db, mock_telemetry_summary, mock_adapter_metrics
    ):
        """Enriched data includes adapter_details with error and latency info."""
        p = TelemetryPersistence(interval_s=9999)
        p._ensure_schema()

        with patch("vetinari.telemetry.get_telemetry_collector", create=True) as mock_tc:
            collector = MagicMock()
            collector.get_summary.return_value = mock_telemetry_summary
            collector.get_adapter_metrics.return_value = mock_adapter_metrics
            mock_tc.return_value = collector

            p._persist_snapshot()

        conn = sqlite3.connect(str(in_memory_db))
        row = conn.execute("SELECT data FROM telemetry_snapshots").fetchone()
        conn.close()

        data = json.loads(row[0])
        detail = data["adapter_details"]["openai:gpt-4"]
        assert detail["total_requests"] == 10
        assert detail["avg_latency_ms"] == 120.0


# ---------------------------------------------------------------------------
# TelemetryPersistence — alerting
# ---------------------------------------------------------------------------


class TestAlertThresholds:
    """Tests for threshold-based alerting in _check_alert_thresholds."""

    def test_high_error_rate_logs_warning(self, caplog):
        """Error rate above threshold emits a WARNING log."""
        import logging

        p = TelemetryPersistence(error_rate_threshold=5.0)
        summary = {
            "adapter_details": {
                "prov:model": {
                    "total_requests": 100,
                    "failed_requests": 20,  # 20% error rate
                }
            }
        }
        with (
            caplog.at_level(logging.WARNING, logger="vetinari.analytics.telemetry_persistence"),
            patch.object(p, "_emit_alert_event") as mock_emit,
        ):
            p._check_alert_thresholds(summary)

        assert any("error rate" in record.message.lower() for record in caplog.records)
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args[0]
        assert call_args[0] == "high_error_rate"

    def test_low_error_rate_no_warning(self, caplog):
        """Error rate below threshold does not emit a warning."""
        import logging

        p = TelemetryPersistence(error_rate_threshold=15.0)
        summary = {
            "adapter_details": {
                "prov:model": {
                    "total_requests": 100,
                    "failed_requests": 5,  # 5% error rate
                }
            }
        }
        with (
            caplog.at_level(logging.WARNING, logger="vetinari.analytics.telemetry_persistence"),
            patch.object(p, "_emit_alert_event") as mock_emit,
        ):
            p._check_alert_thresholds(summary)

        assert mock_emit.call_count == 0

    def test_high_p95_latency_logs_warning(self, caplog):
        """p95 latency above threshold emits a WARNING log."""
        import logging

        p = TelemetryPersistence(p95_latency_threshold_ms=1000.0)
        mock_stats = {"count": 50, "avg": 800.0, "p95": 2000.0}

        with (
            caplog.at_level(logging.WARNING, logger="vetinari.analytics.telemetry_persistence"),
            patch("vetinari.metrics.get_metrics") as mock_get_metrics,
            patch.object(p, "_emit_alert_event") as mock_emit,
        ):
            mock_mc = MagicMock()
            mock_mc.get_histogram_stats.return_value = mock_stats
            mock_get_metrics.return_value = mock_mc

            p._check_alert_thresholds({"adapter_details": {}})

        assert any("p95" in record.message.lower() for record in caplog.records)
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args[0]
        assert call_args[0] == "high_p95_latency"

    def test_zero_requests_emit_blackout_alert(self):
        """No requests across known adapters is treated as a telemetry blackout."""
        p = TelemetryPersistence(error_rate_threshold=5.0)
        summary = {"adapter_details": {"prov:model": {"total_requests": 0, "failed_requests": 0}}}

        with patch.object(p, "_emit_alert_event") as mock_emit:
            p._check_alert_thresholds(summary)

        mock_emit.assert_called_once()
        assert mock_emit.call_args[0][0] == "zero_traffic_blackout"

    def test_emit_alert_event_publishes_message_and_metadata(self):
        """_emit_alert_event() publishes a TelemetryAlertEvent with the given message and metadata.

        Regression test for defect #27: the published event must carry the full
        human-readable message string and the numeric metadata dict so subscribers
        can act on the alert without re-querying the telemetry store.
        """
        from vetinari.events import TelemetryAlertEvent, get_event_bus

        p = TelemetryPersistence()
        published_events: list = []

        real_bus = get_event_bus()
        with patch.object(real_bus, "publish", side_effect=published_events.append):
            p._emit_alert_event(
                alert_type="high_error_rate",
                message="Error rate 25.0% exceeds threshold 10.0%",
                metadata={"error_rate": 25.0, "threshold": 10.0, "total_requests": 100},
            )

        assert len(published_events) == 1
        event = published_events[0]
        assert isinstance(event, TelemetryAlertEvent)
        assert event.alert_type == "high_error_rate"
        assert "25.0%" in event.message, "message must contain the numeric rate"
        assert event.metadata["error_rate"] == pytest.approx(25.0)
        assert event.metadata["threshold"] == pytest.approx(10.0)
        assert event.metadata["total_requests"] == 100
        assert event.event_type == "TELEMETRY_ALERT"


class TestPeriodicMetricsFeed:
    """Tests for telemetry-to-analytics capacity metric handoff."""

    def test_uses_interval_request_rate_and_snapshot_queue_depth(self):
        p = TelemetryPersistence()
        first = {
            "session_requests": 10,
            "queue_depth": 4,
            "adapter_details": {"prov:model": {"total_requests": 10, "avg_latency_ms": 100.0}},
        }
        second = {
            "session_requests": 16,
            "queue_depth": 7,
            "adapter_details": {"prov:model": {"total_requests": 16, "avg_latency_ms": 120.0}},
        }

        with patch("vetinari.analytics.wiring.record_periodic_metrics") as mock_record:
            p._feed_periodic_metrics(first, now=100.0)
            p._feed_periodic_metrics(second, now=103.0)

        assert mock_record.call_args_list[0].kwargs["request_rate"] == pytest.approx(0.0)
        assert mock_record.call_args_list[1].kwargs["request_rate"] == pytest.approx(2.0)
        assert mock_record.call_args_list[1].kwargs["queue_depth"] == 7


# ---------------------------------------------------------------------------
# TelemetryPersistence — pruning
# ---------------------------------------------------------------------------


class TestSnapshotPruning:
    """Tests for _prune_old_snapshots."""

    def test_prune_removes_old_rows(self, in_memory_db):
        """_prune_old_snapshots() deletes rows whose timestamp is before the cutoff."""
        p = TelemetryPersistence(retention_days=1)
        p._ensure_schema()

        conn = sqlite3.connect(str(in_memory_db))
        old_ts = time.time() - (2 * 86400)  # 2 days ago
        new_ts = time.time() - 60  # 1 minute ago
        conn.execute(
            "INSERT INTO telemetry_snapshots (timestamp, data) VALUES (?, ?)",
            (old_ts, '{"test": "old"}'),
        )
        conn.execute(
            "INSERT INTO telemetry_snapshots (timestamp, data) VALUES (?, ?)",
            (new_ts, '{"test": "new"}'),
        )
        conn.commit()
        conn.close()

        p._prune_old_snapshots(time.time())

        conn = sqlite3.connect(str(in_memory_db))
        rows = conn.execute("SELECT timestamp FROM telemetry_snapshots").fetchall()
        conn.close()

        assert len(rows) == 1
        assert abs(rows[0][0] - new_ts) < 1.0


# ---------------------------------------------------------------------------
# TelemetryPersistence — get_history
# ---------------------------------------------------------------------------


class TestGetHistory:
    """Tests for TelemetryPersistence.get_history()."""

    def test_get_history_returns_snapshots_newest_first(self, in_memory_db):
        """get_history() returns rows ordered newest-first."""
        p = TelemetryPersistence(interval_s=9999)
        p._ensure_schema()

        conn = sqlite3.connect(str(in_memory_db))
        now = time.time()
        for i in range(3):
            conn.execute(
                "INSERT INTO telemetry_snapshots (timestamp, data) VALUES (?, ?)",
                (now + i, json.dumps({"index": i})),
            )
        conn.commit()
        conn.close()

        history = p.get_history(limit=10)
        assert len(history) == 3
        # Newest-first: highest timestamp first
        assert history[0]["timestamp"] >= history[1]["timestamp"]
        assert history[1]["timestamp"] >= history[2]["timestamp"]

    def test_get_history_respects_limit(self, in_memory_db):
        """get_history() returns at most ``limit`` rows."""
        p = TelemetryPersistence(interval_s=9999)
        p._ensure_schema()

        conn = sqlite3.connect(str(in_memory_db))
        now = time.time()
        for i in range(10):
            conn.execute(
                "INSERT INTO telemetry_snapshots (timestamp, data) VALUES (?, ?)",
                (now + i, '{"n": ' + str(i) + "}"),
            )
        conn.commit()
        conn.close()

        history = p.get_history(limit=3)
        assert len(history) == 3

    def test_get_history_returns_empty_list_on_db_error(self):
        """get_history() returns [] when the database is unavailable."""
        p = TelemetryPersistence(interval_s=9999)
        with patch(
            "vetinari.database.get_connection",
            side_effect=Exception("DB unavailable"),
        ):
            result = p.get_history()
        assert result == []

    def test_get_history_parses_json_data(self, in_memory_db):
        """data field in each snapshot is a parsed dict, not a raw string."""
        p = TelemetryPersistence(interval_s=9999)
        p._ensure_schema()

        conn = sqlite3.connect(str(in_memory_db))
        conn.execute(
            "INSERT INTO telemetry_snapshots (timestamp, data) VALUES (?, ?)",
            (time.time(), '{"total_tokens_used": 42}'),
        )
        conn.commit()
        conn.close()

        history = p.get_history()
        assert isinstance(history[0]["data"], dict)
        assert history[0]["data"]["total_tokens_used"] == 42


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    """Tests for get_telemetry_persistence() singleton behaviour."""

    def test_singleton_returns_same_instance(self):
        """get_telemetry_persistence() returns the same object on every call."""
        p1 = get_telemetry_persistence()
        p2 = get_telemetry_persistence()
        assert p1 is p2

    def test_reset_clears_singleton(self):
        """reset_telemetry_persistence() clears the singleton so a new one is created."""
        p1 = get_telemetry_persistence()
        reset_telemetry_persistence()
        p2 = get_telemetry_persistence()
        assert p1 is not p2

    def test_singleton_thread_safe(self):
        """Concurrent calls to get_telemetry_persistence() return the same instance."""
        results: list[TelemetryPersistence] = []
        lock = threading.Lock()

        def _get():
            instance = get_telemetry_persistence()
            with lock:
                results.append(instance)

        threads = [threading.Thread(target=_get) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len({id(r) for r in results}) == 1
