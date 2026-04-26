"""Tests for SSE event persistence helpers in vetinari.web.sse_events.

Covers _persist_sse_event, get_recent_sse_events, and cleanup_old_sse_events.
Each test uses an isolated in-memory/temporary database via VETINARI_DB_PATH so
the production database is never touched.
"""

from __future__ import annotations

import sqlite3
import time

import pytest

from vetinari.database import get_connection, reset_for_testing
from vetinari.web.sse_events import (
    _persist_sse_event,
    cleanup_old_sse_events,
    get_recent_sse_events,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Point VETINARI_DB_PATH to a per-test temp file and reset module state.

    Ensures each test starts with a fresh schema and no leftover rows.
    """
    db_file = tmp_path / "sse_test.db"
    monkeypatch.setenv("VETINARI_DB_PATH", str(db_file))
    reset_for_testing()
    # Establish connection so schema (including sse_event_log) is created.
    get_connection()
    yield db_file
    reset_for_testing()


# ---------------------------------------------------------------------------
# _persist_sse_event
# ---------------------------------------------------------------------------


class TestPersistSseEvent:
    def test_inserts_row_into_sse_event_log(self, isolated_db):
        """A persisted event must appear in the table with correct fields."""
        _persist_sse_event("proj-1", "task_started", {"task_id": "t1"})

        conn = get_connection()
        rows = conn.execute(
            "SELECT project_id, event_type, payload_json FROM sse_event_log WHERE project_id = 'proj-1'"
        ).fetchall()

        assert len(rows) == 1
        assert rows[0]["project_id"] == "proj-1"
        assert rows[0]["event_type"] == "task_started"
        assert '"task_id"' in rows[0]["payload_json"]

    def test_multiple_events_for_same_project(self, isolated_db):
        """All emitted events for a project are stored as separate rows."""
        _persist_sse_event("proj-2", "task_started", {"task_id": "t1"})
        _persist_sse_event("proj-2", "task_completed", {"task_id": "t1"})
        _persist_sse_event("proj-2", "status", {"status": "idle"})

        conn = get_connection()
        count = conn.execute("SELECT COUNT(*) FROM sse_event_log WHERE project_id = 'proj-2'").fetchone()[0]
        assert count == 3

    def test_events_for_different_projects_are_independent(self, isolated_db):
        """Events from project A must not appear in project B's query."""
        _persist_sse_event("proj-A", "status", {"status": "running"})
        _persist_sse_event("proj-B", "status", {"status": "idle"})

        conn = get_connection()
        a_rows = conn.execute("SELECT * FROM sse_event_log WHERE project_id = 'proj-A'").fetchall()
        b_rows = conn.execute("SELECT * FROM sse_event_log WHERE project_id = 'proj-B'").fetchall()

        assert len(a_rows) == 1
        assert len(b_rows) == 1

    def test_persist_handles_db_error_without_raising(self, isolated_db, monkeypatch, caplog):
        """A DB write failure is logged at WARNING and does not propagate to callers."""
        import logging

        conn = get_connection()
        # Drop the table to force a sqlite3.OperationalError on INSERT.
        conn.execute("DROP TABLE sse_event_log")
        conn.commit()

        with caplog.at_level(logging.WARNING, logger="vetinari.web.sse_events"):
            _persist_sse_event("proj-err", "error", {"error": "boom"})
        assert any("proj-err" in r.message for r in caplog.records)

    def test_payload_with_unicode_is_stored_correctly(self, isolated_db):
        """Non-ASCII payload content must round-trip through JSON without loss."""
        payload = {"message": "こんにちは — hello"}
        _persist_sse_event("proj-unicode", "thinking", payload)

        conn = get_connection()
        row = conn.execute("SELECT payload_json FROM sse_event_log WHERE project_id = 'proj-unicode'").fetchone()
        assert row is not None
        assert "こんにちは" in row["payload_json"]


# ---------------------------------------------------------------------------
# get_recent_sse_events
# ---------------------------------------------------------------------------


class TestGetRecentSseEvents:
    def test_returns_events_in_ascending_order(self, isolated_db):
        """Events must be returned oldest-first for correct replay order."""
        for i in range(3):
            _persist_sse_event("proj-order", f"type_{i}", {"seq": i})

        events = get_recent_sse_events("proj-order")
        assert len(events) == 3
        # IDs auto-increment so ascending ID == ascending insertion order.
        ids = [e["id"] for e in events]
        assert ids == sorted(ids)

    def test_returns_empty_list_for_unknown_project(self, isolated_db):
        """Querying a project with no events must return an empty list."""
        result = get_recent_sse_events("no-such-project")
        assert result == []

    def test_limit_restricts_number_of_rows(self, isolated_db):
        """The ``limit`` parameter must cap the number of returned events."""
        for i in range(10):
            _persist_sse_event("proj-limit", "status", {"n": i})

        events = get_recent_sse_events("proj-limit", limit=4)
        assert len(events) == 4

    def test_since_filters_older_events(self, isolated_db):
        """Only events with emitted_at > since must be returned."""
        conn = get_connection()
        # Insert an old event with an explicit timestamp in the past.
        conn.execute(
            "INSERT INTO sse_event_log (project_id, event_type, payload_json, emitted_at) VALUES (?, ?, ?, ?)",
            ("proj-since", "old_event", '{"old": true}', "2000-01-01 00:00:00"),
        )
        conn.commit()

        # Insert a recent event via the helper (uses datetime('now')).
        _persist_sse_event("proj-since", "new_event", {"new": True})

        events = get_recent_sse_events("proj-since", since="2000-06-01 00:00:00")
        assert len(events) == 1
        assert events[0]["event_type"] == "new_event"

    def test_since_none_returns_all_events(self, isolated_db):
        """When ``since`` is None every event for the project must be returned."""
        _persist_sse_event("proj-all", "a", {})
        _persist_sse_event("proj-all", "b", {})

        events = get_recent_sse_events("proj-all", since=None)
        assert len(events) == 2

    def test_returned_dict_has_expected_keys(self, isolated_db):
        """Each returned event dict must have id, project_id, event_type, payload, emitted_at."""
        _persist_sse_event("proj-keys", "status", {"status": "running"})

        events = get_recent_sse_events("proj-keys")
        assert len(events) == 1
        event = events[0]
        for key in ("id", "project_id", "event_type", "payload", "emitted_at"):
            assert key in event, f"Expected key '{key}' missing from returned event"
        assert event["project_id"] == "proj-keys"
        assert event["event_type"] == "status"
        assert event["payload"] == {"status": "running"}

    def test_corrupted_payload_json_does_not_raise(self, isolated_db):
        """A row with invalid JSON in payload_json must be returned with ``_raw`` fallback."""
        conn = get_connection()
        conn.execute(
            "INSERT INTO sse_event_log (project_id, event_type, payload_json) VALUES (?, ?, ?)",
            ("proj-corrupt", "broken", "not-valid-json{{{"),
        )
        conn.commit()

        events = get_recent_sse_events("proj-corrupt")
        assert len(events) == 1
        assert "_raw" in events[0]["payload"]


# ---------------------------------------------------------------------------
# cleanup_old_sse_events
# ---------------------------------------------------------------------------


class TestCleanupOldSseEvents:
    def test_deletes_events_older_than_cutoff(self, isolated_db):
        """Rows with emitted_at before the retention window must be deleted."""
        conn = get_connection()
        # Insert a row dated far in the past.
        conn.execute(
            "INSERT INTO sse_event_log (project_id, event_type, payload_json, emitted_at) VALUES (?, ?, ?, ?)",
            ("proj-cleanup", "old_event", "{}", "2000-01-01 00:00:00"),
        )
        conn.commit()

        deleted = cleanup_old_sse_events(hours=24)
        assert deleted >= 1

        remaining = conn.execute("SELECT COUNT(*) FROM sse_event_log WHERE project_id = 'proj-cleanup'").fetchone()[0]
        assert remaining == 0

    def test_keeps_recent_events(self, isolated_db):
        """Rows emitted within the retention window must not be deleted."""
        _persist_sse_event("proj-keep", "recent_event", {"x": 1})

        deleted = cleanup_old_sse_events(hours=24)

        conn = get_connection()
        remaining = conn.execute("SELECT COUNT(*) FROM sse_event_log WHERE project_id = 'proj-keep'").fetchone()[0]
        assert remaining == 1
        assert deleted == 0

    def test_returns_zero_when_nothing_to_delete(self, isolated_db):
        """cleanup_old_sse_events must return 0 when the table is empty."""
        result = cleanup_old_sse_events(hours=1)
        assert result == 0

    def test_returns_count_of_deleted_rows(self, isolated_db):
        """The return value must equal the exact number of rows removed."""
        conn = get_connection()
        for i in range(5):
            conn.execute(
                "INSERT INTO sse_event_log (project_id, event_type, payload_json, emitted_at) VALUES (?, ?, ?, ?)",
                ("proj-count", f"e{i}", "{}", "1999-12-31 00:00:00"),
            )
        conn.commit()

        deleted = cleanup_old_sse_events(hours=24)
        assert deleted == 5

    @pytest.mark.parametrize("hours", [1, 12, 48, 168])
    def test_various_retention_windows(self, isolated_db, hours):
        """cleanup_old_sse_events must accept any positive integer for hours."""
        # Should not raise regardless of the window size.
        result = cleanup_old_sse_events(hours=hours)
        assert isinstance(result, int)
        assert result >= 0


# ---------------------------------------------------------------------------
# QualityResultEvent.to_sse() includes confidence field (Session 18, Task 8)
# ---------------------------------------------------------------------------


class TestQualityResultEventConfidence:
    """Verify that QualityResultEvent carries and serialises the confidence field."""

    def test_to_sse_includes_confidence_key(self):
        """to_sse() output must contain a 'confidence' key."""
        from vetinari.web.sse_events import QualityResultEvent

        event = QualityResultEvent(
            project_id="p1",
            quality_score=0.85,
            confidence=0.9,
        )
        sse = event.to_sse()
        assert "confidence" in sse, "to_sse() is missing the 'confidence' key"

    def test_confidence_value_round_trips(self):
        """The confidence value stored on the dataclass must appear in to_sse()."""
        from vetinari.web.sse_events import QualityResultEvent

        event = QualityResultEvent(
            project_id="p2",
            quality_score=0.7,
            confidence=0.55,
        )
        sse = event.to_sse()
        assert sse["confidence"] == 0.55

    def test_default_confidence_is_zero(self):
        """When confidence is not supplied, it defaults to 0.0 in to_sse()."""
        from vetinari.web.sse_events import QualityResultEvent

        event = QualityResultEvent(project_id="p3", quality_score=0.5)
        sse = event.to_sse()
        assert sse["confidence"] == 0.0

    def test_confidence_field_exists_on_dataclass(self):
        """QualityResultEvent must have a 'confidence' attribute after construction."""
        from vetinari.web.sse_events import QualityResultEvent

        event = QualityResultEvent(project_id="p4", quality_score=0.6, confidence=1.0)
        assert hasattr(event, "confidence")
        assert event.confidence == 1.0
