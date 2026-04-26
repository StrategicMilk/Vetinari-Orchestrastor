"""Tests for TelemetryCollector.restore_from_snapshot().

Covers the restore path end-to-end using a real in-memory SQLite database
via the VETINARI_DB_PATH environment variable override.  No mocking of DB
internals — tests exercise the real query and parse path.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from vetinari.telemetry import TelemetryCollector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot_data(
    adapters: dict[str, dict] | None = None,
    by_model: dict[str, dict] | None = None,
) -> str:
    """Build a minimal JSON snapshot string matching the persistence format.

    Args:
        adapters: Mapping of "provider:model" key to adapter_details dict.
        by_model: Mapping of model name to {"tokens": int, "requests": int}.

    Returns:
        JSON string ready to insert into telemetry_snapshots.data.
    """
    data: dict = {
        "total_tokens_used": 0,
        "total_cost_usd": 0.0,
        "by_model": by_model or {},
        "by_provider": {},
        "session_requests": 0,
        "adapter_details": adapters or {},
    }
    return json.dumps(data)


def _insert_snapshot(conn: sqlite3.Connection, data_json: str, ts: float | None = None) -> None:
    """Insert one row into telemetry_snapshots.

    Args:
        conn: Open SQLite connection.
        data_json: Serialised snapshot JSON.
        ts: Timestamp (defaults to now).
    """
    conn.execute(
        "INSERT INTO telemetry_snapshots (timestamp, data) VALUES (?, ?)",
        (ts or time.time(), data_json),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path: Path):
    """Provide a temp SQLite DB with the telemetry_snapshots table.

    Yields the db Path so the caller can insert rows, then sets
    VETINARI_DB_PATH for the duration of the test.
    """
    db_path = tmp_path / "vetinari.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS telemetry_snapshots (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL    NOT NULL,
            data      TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tel_snapshots_ts ON telemetry_snapshots(timestamp);
        """
    )
    conn.commit()

    # Reset the thread-local connection cache so get_connection() opens our DB
    from vetinari import database as _db_mod

    old_env = os.environ.get("VETINARI_DB_PATH")
    os.environ["VETINARI_DB_PATH"] = str(db_path)
    # Clear any cached thread-local connection so it reconnects to our file
    if hasattr(_db_mod._thread_local, "connection"):
        try:
            _db_mod._thread_local.connection.close()
        except Exception:  # noqa: VET022 — best-effort DB cleanup in test fixture
            pass
        del _db_mod._thread_local.connection

    yield conn, db_path

    # Restore env and thread-local state
    if old_env is None:
        os.environ.pop("VETINARI_DB_PATH", None)
    else:
        os.environ["VETINARI_DB_PATH"] = old_env
    if hasattr(_db_mod._thread_local, "connection"):
        try:
            _db_mod._thread_local.connection.close()
        except Exception:  # noqa: VET022 — best-effort DB cleanup in test fixture
            pass
        del _db_mod._thread_local.connection

    conn.close()


# ---------------------------------------------------------------------------
# Tests: no snapshot / empty DB
# ---------------------------------------------------------------------------


class TestRestoreNoSnapshot:
    """restore_from_snapshot() graceful degradation when no data exists."""

    def test_empty_table_leaves_counters_at_zero(self, tmp_db):
        """Restore from an empty table must leave all adapter counters at zero."""
        _conn, _path = tmp_db
        collector = TelemetryCollector()
        collector.restore_from_snapshot()

        assert collector.adapter_metrics == {}
        summary = collector.get_summary()
        assert summary["total_tokens_used"] == 0
        assert summary["session_requests"] == 0

    def test_no_db_table_logs_info_and_continues(self, tmp_path: Path):
        """If the table does not exist yet, restore_from_snapshot() must not raise."""
        db_path = tmp_path / "fresh.db"
        # Create a DB file with no tables at all
        conn = sqlite3.connect(str(db_path))
        conn.close()

        from vetinari import database as _db_mod

        old_env = os.environ.get("VETINARI_DB_PATH")
        os.environ["VETINARI_DB_PATH"] = str(db_path)
        if hasattr(_db_mod._thread_local, "connection"):
            try:
                _db_mod._thread_local.connection.close()
            except Exception:  # noqa: VET022 — best-effort DB cleanup in test fixture
                pass
            del _db_mod._thread_local.connection

        try:
            collector = TelemetryCollector()
            # Must not raise even though the table is missing
            collector.restore_from_snapshot()
            assert collector.adapter_metrics == {}
        finally:
            if old_env is None:
                os.environ.pop("VETINARI_DB_PATH", None)
            else:
                os.environ["VETINARI_DB_PATH"] = old_env
            if hasattr(_db_mod._thread_local, "connection"):
                try:
                    _db_mod._thread_local.connection.close()
                except Exception:  # noqa: VET022 — best-effort DB cleanup in test fixture
                    pass
                del _db_mod._thread_local.connection


# ---------------------------------------------------------------------------
# Tests: snapshot present — happy path
# ---------------------------------------------------------------------------


class TestRestoreHappyPath:
    """restore_from_snapshot() seeds counters from a valid snapshot row."""

    def test_adapter_request_counts_restored(self, tmp_db):
        """total_requests and failed_requests must match the snapshot values."""
        conn, _path = tmp_db
        _insert_snapshot(
            conn,
            _make_snapshot_data(
                adapters={
                    "llama_cpp:mistral-7b": {
                        "total_requests": 42,
                        "failed_requests": 3,
                        "avg_latency_ms": 200.0,
                        "min_latency_ms": 150.0,
                        "max_latency_ms": 350.0,
                    }
                }
            ),
        )

        collector = TelemetryCollector()
        collector.restore_from_snapshot()

        key = "llama_cpp:mistral-7b"
        assert key in collector.adapter_metrics
        m = collector.adapter_metrics[key]
        assert m.total_requests == 42
        assert m.failed_requests == 3
        assert m.successful_requests == 39  # 42 - 3

    def test_get_summary_reflects_restored_values(self, tmp_db):
        """get_summary() must include the restored request counts."""
        conn, _path = tmp_db
        _insert_snapshot(
            conn,
            _make_snapshot_data(
                adapters={
                    "litellm:gpt-4": {
                        "total_requests": 100,
                        "failed_requests": 5,
                        "avg_latency_ms": 500.0,
                        "min_latency_ms": 300.0,
                        "max_latency_ms": 800.0,
                    }
                }
            ),
        )

        collector = TelemetryCollector()
        collector.restore_from_snapshot()

        summary = collector.get_summary()
        assert summary["session_requests"] == 100

    def test_tokens_restored_from_adapter_details(self, tmp_db):
        """Token counts are restored from adapter_details.total_tokens_used, not by_model.

        The implementation reads tokens directly from the per-adapter entry to
        avoid the double-count defect that arose when multiple providers shared
        a model name (the old by_model proportional-split path).
        """
        conn, _path = tmp_db
        _insert_snapshot(
            conn,
            _make_snapshot_data(
                adapters={
                    "llama_cpp:phi-3": {
                        "total_requests": 10,
                        "failed_requests": 0,
                        "avg_latency_ms": 100.0,
                        "min_latency_ms": 80.0,
                        "max_latency_ms": 120.0,
                        "total_tokens_used": 5000,
                    }
                },
                by_model={"phi-3": {"tokens": 5000, "requests": 10}},
            ),
        )

        collector = TelemetryCollector()
        collector.restore_from_snapshot()

        m = collector.adapter_metrics["llama_cpp:phi-3"]
        assert m.total_tokens_used == 5000
        summary = collector.get_summary()
        assert summary["total_tokens_used"] == 5000

    def test_latency_fields_restored(self, tmp_db):
        """min/max/total_latency_ms must be seeded from snapshot values."""
        conn, _path = tmp_db
        _insert_snapshot(
            conn,
            _make_snapshot_data(
                adapters={
                    "llama_cpp:codellama": {
                        "total_requests": 4,
                        "failed_requests": 0,
                        "avg_latency_ms": 250.0,
                        "min_latency_ms": 200.0,
                        "max_latency_ms": 300.0,
                    }
                }
            ),
        )

        collector = TelemetryCollector()
        collector.restore_from_snapshot()

        m = collector.adapter_metrics["llama_cpp:codellama"]
        # total_latency_ms = avg * successful = 250 * 4 = 1000
        assert m.total_latency_ms == pytest.approx(1000.0)
        assert m.min_latency_ms == pytest.approx(200.0)
        assert m.max_latency_ms == pytest.approx(300.0)

    def test_multiple_adapters_all_restored(self, tmp_db):
        """All adapter entries in the snapshot must be present after restore."""
        conn, _path = tmp_db
        _insert_snapshot(
            conn,
            _make_snapshot_data(
                adapters={
                    "llama_cpp:model-a": {
                        "total_requests": 10,
                        "failed_requests": 1,
                        "avg_latency_ms": 100.0,
                        "min_latency_ms": 90.0,
                        "max_latency_ms": 110.0,
                    },
                    "litellm:model-b": {
                        "total_requests": 20,
                        "failed_requests": 0,
                        "avg_latency_ms": 200.0,
                        "min_latency_ms": 150.0,
                        "max_latency_ms": 250.0,
                    },
                }
            ),
        )

        collector = TelemetryCollector()
        collector.restore_from_snapshot()

        assert "llama_cpp:model-a" in collector.adapter_metrics
        assert "litellm:model-b" in collector.adapter_metrics

    def test_newest_snapshot_wins(self, tmp_db):
        """When multiple snapshots exist, the most recent one is used."""
        conn, _path = tmp_db
        old_ts = time.time() - 3600
        new_ts = time.time()

        _insert_snapshot(
            conn,
            _make_snapshot_data(
                adapters={
                    "llama_cpp:model-x": {
                        "total_requests": 5,
                        "failed_requests": 0,
                        "avg_latency_ms": 100.0,
                        "min_latency_ms": 100.0,
                        "max_latency_ms": 100.0,
                    }
                }
            ),
            ts=old_ts,
        )
        _insert_snapshot(
            conn,
            _make_snapshot_data(
                adapters={
                    "llama_cpp:model-x": {
                        "total_requests": 50,
                        "failed_requests": 2,
                        "avg_latency_ms": 200.0,
                        "min_latency_ms": 180.0,
                        "max_latency_ms": 220.0,
                    }
                }
            ),
            ts=new_ts,
        )

        collector = TelemetryCollector()
        collector.restore_from_snapshot()

        m = collector.adapter_metrics["llama_cpp:model-x"]
        assert m.total_requests == 50  # from the newer snapshot

    def test_restore_then_new_records_accumulate(self, tmp_db):
        """New records after restore must add on top of the restored baseline."""
        conn, _path = tmp_db
        _insert_snapshot(
            conn,
            _make_snapshot_data(
                adapters={
                    "llama_cpp:mistral-7b": {
                        "total_requests": 100,
                        "failed_requests": 5,
                        "avg_latency_ms": 200.0,
                        "min_latency_ms": 150.0,
                        "max_latency_ms": 300.0,
                    }
                }
            ),
        )

        collector = TelemetryCollector()
        collector.restore_from_snapshot()

        # Record one more call
        collector.record_adapter_latency("llama_cpp", "mistral-7b", latency_ms=250.0, success=True, tokens_used=500)

        m = collector.adapter_metrics["llama_cpp:mistral-7b"]
        assert m.total_requests == 101
        assert m.successful_requests == 96  # 95 restored + 1 new


# ---------------------------------------------------------------------------
# Tests: malformed / edge case snapshot data
# ---------------------------------------------------------------------------


class TestRestoreEdgeCases:
    """restore_from_snapshot() handles bad data without raising."""

    def test_malformed_json_logs_warning_no_raise(self, tmp_db):
        """A non-JSON data column must not raise — counters stay at zero."""
        conn, _path = tmp_db
        conn.execute(
            "INSERT INTO telemetry_snapshots (timestamp, data) VALUES (?, ?)",
            (time.time(), "not valid json {{{"),
        )
        conn.commit()

        collector = TelemetryCollector()
        collector.restore_from_snapshot()  # must not raise

        assert collector.adapter_metrics == {}

    def test_missing_adapter_details_key_skipped(self, tmp_db):
        """Snapshot with no adapter_details key must restore cleanly to zero adapters."""
        conn, _path = tmp_db
        _insert_snapshot(conn, json.dumps({"total_tokens_used": 0, "session_requests": 0}))

        collector = TelemetryCollector()
        collector.restore_from_snapshot()

        assert collector.adapter_metrics == {}

    def test_malformed_adapter_key_no_colon_skipped(self, tmp_db):
        """Adapter keys without a colon separator must be silently skipped."""
        conn, _path = tmp_db
        _insert_snapshot(
            conn,
            _make_snapshot_data(
                adapters={
                    "no-colon-here": {
                        "total_requests": 5,
                        "failed_requests": 0,
                        "avg_latency_ms": 100.0,
                        "min_latency_ms": 100.0,
                        "max_latency_ms": 100.0,
                    },
                    "llama_cpp:good-model": {
                        "total_requests": 10,
                        "failed_requests": 0,
                        "avg_latency_ms": 200.0,
                        "min_latency_ms": 200.0,
                        "max_latency_ms": 200.0,
                    },
                }
            ),
        )

        collector = TelemetryCollector()
        collector.restore_from_snapshot()

        # The malformed key is skipped; the valid one is restored
        assert "no-colon-here" not in collector.adapter_metrics
        assert "llama_cpp:good-model" in collector.adapter_metrics

    def test_zero_requests_in_snapshot(self, tmp_db):
        """An adapter entry with zero requests must restore without division errors."""
        conn, _path = tmp_db
        _insert_snapshot(
            conn,
            _make_snapshot_data(
                adapters={
                    "llama_cpp:empty-model": {
                        "total_requests": 0,
                        "failed_requests": 0,
                        "avg_latency_ms": 0.0,
                        "min_latency_ms": 0.0,
                        "max_latency_ms": 0.0,
                    }
                }
            ),
        )

        collector = TelemetryCollector()
        collector.restore_from_snapshot()

        m = collector.adapter_metrics["llama_cpp:empty-model"]
        assert m.total_requests == 0
        assert m.total_latency_ms == pytest.approx(0.0)

    def test_db_unavailable_logs_info_no_raise(self):
        """If the DB module raises on get_connection(), restore must not propagate."""
        collector = TelemetryCollector()
        with patch("vetinari.telemetry.TelemetryCollector.restore_from_snapshot") as mock_restore:
            # Verify that the method exists and is callable without error
            mock_restore.return_value = None
            collector.restore_from_snapshot()

        # Directly simulate a DB error path
        with patch("vetinari.database.get_connection", side_effect=RuntimeError("DB offline")):
            collector2 = TelemetryCollector()
            collector2.restore_from_snapshot()  # must not raise
            assert collector2.adapter_metrics == {}

    def test_shared_model_name_across_providers_no_double_count(self, tmp_db):
        """Two adapters sharing a model name must not double-count tokens after restore.

        Regression test for defect #31: the legacy by_model restore path
        distributed tokens proportionally across adapters, which caused
        incorrect totals when multiple providers (e.g. openai + azure) used
        the same model name.  The current implementation restores each adapter's
        tokens directly from adapter_details, so the total must be the exact
        per-adapter sum, not a multiple of it.
        """
        conn, _path = tmp_db
        _insert_snapshot(
            conn,
            _make_snapshot_data(
                adapters={
                    "openai:gpt-4": {
                        "total_requests": 10,
                        "failed_requests": 0,
                        "avg_latency_ms": 300.0,
                        "min_latency_ms": 200.0,
                        "max_latency_ms": 400.0,
                        "total_tokens_used": 300,
                    },
                    "azure:gpt-4": {
                        "total_requests": 20,
                        "failed_requests": 2,
                        "avg_latency_ms": 250.0,
                        "min_latency_ms": 180.0,
                        "max_latency_ms": 350.0,
                        "total_tokens_used": 500,
                    },
                }
            ),
        )

        collector = TelemetryCollector()
        collector.restore_from_snapshot()

        # Both adapter keys must exist and have distinct, correct counts
        assert "openai:gpt-4" in collector.adapter_metrics
        assert "azure:gpt-4" in collector.adapter_metrics

        openai_m = collector.adapter_metrics["openai:gpt-4"]
        azure_m = collector.adapter_metrics["azure:gpt-4"]

        assert openai_m.total_tokens_used == 300
        assert azure_m.total_tokens_used == 500

        # get_summary() by_model["gpt-4"] must be 300 + 500 = 800 — not 1600
        summary = collector.get_summary()
        gpt4_tokens = summary["by_model"]["gpt-4"]["tokens"]
        assert gpt4_tokens == 800, (
            f"Expected 800 tokens for gpt-4 across both providers, got {gpt4_tokens} "
            "(double-count regression: each adapter's tokens must be restored from "
            "adapter_details, not derived from by_model proportional split)"
        )
        assert summary["total_tokens_used"] == 800


# ---------------------------------------------------------------------------
# Tests: wiring — restore is called at startup
# ---------------------------------------------------------------------------


class TestRestoreWiring:
    """Verify that _wire_telemetry_persistence() calls restore_from_snapshot()."""

    def test_wire_calls_restore_before_persistence_start(self):
        """_wire_telemetry_persistence must call restore_from_snapshot on the collector."""
        call_order: list[str] = []

        mock_collector = TelemetryCollector.__new__(TelemetryCollector)
        mock_collector.adapter_metrics = {}
        mock_collector._lock = threading.RLock()

        def _fake_restore() -> None:
            call_order.append("restore")

        mock_collector.restore_from_snapshot = _fake_restore  # type: ignore[method-assign]

        started: list[bool] = []

        class _FakePersistence:
            def start(self) -> None:
                call_order.append("start")
                started.append(True)

        with (
            patch(
                "vetinari.telemetry.get_telemetry_collector",
                return_value=mock_collector,
            ),
            patch(
                "vetinari.analytics.telemetry_persistence.get_telemetry_persistence",
                return_value=_FakePersistence(),
            ),
        ):
            from vetinari.cli_startup import _wire_telemetry_persistence

            _wire_telemetry_persistence()

        assert call_order == ["restore", "start"], f"Expected restore before start, got: {call_order}"
