"""Tests for vetinari.web.litestar_sse_replay_api  -  SSE event replay endpoint.

Verifies that the replay endpoint correctly returns persisted SSE events
from the sse_event_log table, supports sequence-based filtering, and
rejects invalid project IDs.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """Redirect all DB access to a fresh temp database for this test.

    Sets VETINARI_DB_PATH so every call to ``get_connection()`` within the
    test lands on an isolated file, preventing cross-test contamination.
    After the test the thread-local connection is closed so the next test
    starts clean.
    """
    db_path = tmp_path / "sse_replay_test.db"
    monkeypatch.setenv("VETINARI_DB_PATH", str(db_path))

    # Force creation of a fresh thread-local connection on the new path
    from vetinari.database import close_connection

    close_connection()

    yield db_path

    # Close after the test so the next test's monkeypatch takes effect cleanly
    close_connection()


@pytest.fixture
def replay_app(test_db):
    """Minimal Litestar app containing only the SSE replay handler.

    Using the full ``create_app()`` would pull in many subsystems.  This
    fixture creates a targeted app so the test only exercises replay logic.
    """
    from litestar import Litestar

    from vetinari.web.litestar_sse_replay_api import create_sse_replay_handlers

    handlers = create_sse_replay_handlers()
    assert handlers, "create_sse_replay_handlers() returned empty list  -  Litestar not installed"
    return Litestar(route_handlers=handlers, logging_config=None)


@pytest.fixture
def replay_client(replay_app):
    """Synchronous TestClient for the replay app."""
    from litestar.testing import TestClient

    with TestClient(app=replay_app) as client:
        yield client


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _push(project_id: str, event_type: str, data: dict) -> None:
    """Write one event to sse_event_log via the production push path.

    Using the real ``_push_sse_event`` exercises the same INSERT that the
    live system uses, keeping the test honest about the schema.
    """
    from vetinari.web.shared import _push_sse_event

    _push_sse_event(project_id, event_type, data)


# ---------------------------------------------------------------------------
# Tests: happy path
# ---------------------------------------------------------------------------


class TestReplayEndpointHappyPath:
    """Replay endpoint returns events in chronological order."""

    def test_replay_returns_all_events(self, replay_client) -> None:
        """Push 5 events and replay  -  all 5 returned in insertion order."""
        project_id = "proj-replay-all"
        for i in range(1, 6):
            _push(project_id, "task_started", {"task_id": f"t{i}", "index": i})

        response = replay_client.get(f"/api/v1/projects/{project_id}/events/replay")
        assert response.status_code == 200, response.text

        events = response.json()
        assert len(events) == 5

        # Verify chronological order (sequence strictly ascending)
        sequences = [e["sequence"] for e in events]
        assert sequences == sorted(sequences), "Events must be in ascending sequence order"

        # Verify shape of each event dict
        for ev in events:
            assert "id" in ev
            assert "event_type" in ev
            assert "sequence" in ev
            assert "data" in ev
            assert "emitted_at" in ev
            assert ev["event_type"] == "task_started"

        # Verify data round-trips correctly
        assert events[0]["data"]["task_id"] == "t1"
        assert events[4]["data"]["task_id"] == "t5"

    def test_replay_after_sequence_filters_events(self, replay_client) -> None:
        """Push 5 events and replay with after_sequence=3  -  only events 4 and 5 returned."""
        project_id = "proj-replay-filter"
        for i in range(1, 6):
            _push(project_id, "task_completed", {"task_id": f"t{i}"})

        # Read all events first to learn the actual sequence numbers
        all_resp = replay_client.get(f"/api/v1/projects/{project_id}/events/replay")
        assert all_resp.status_code == 200
        all_events = all_resp.json()
        assert len(all_events) == 5

        # Use the third event's sequence as the cutoff
        cutoff_seq = all_events[2]["sequence"]

        response = replay_client.get(
            f"/api/v1/projects/{project_id}/events/replay",
            params={"after_sequence": cutoff_seq},
        )
        assert response.status_code == 200, response.text

        filtered = response.json()
        assert len(filtered) == 2, f"Expected 2 events after seq {cutoff_seq}, got {len(filtered)}"

        # All returned events must have sequence strictly greater than cutoff
        for ev in filtered:
            assert ev["sequence"] > cutoff_seq, f"Event sequence {ev['sequence']} is not > cutoff {cutoff_seq}"

    def test_replay_thinking_and_decision_events_persisted(self, replay_client) -> None:
        """ThinkingEvent and DecisionEvent are persisted and retrievable via replay."""
        project_id = "proj-replay-log-types"

        from vetinari.web.sse_events import DecisionEvent, ThinkingEvent

        thinking = ThinkingEvent(agent_type="worker", message="Analysing inputs")
        decision = DecisionEvent(
            decision_type="model_select",
            summary="Chose fast model",
            details={"model_id": "phi3"},
        )

        _push(project_id, thinking.event_type, thinking.to_sse())
        _push(project_id, decision.event_type, decision.to_sse())

        response = replay_client.get(f"/api/v1/projects/{project_id}/events/replay")
        assert response.status_code == 200, response.text

        events = response.json()
        assert len(events) == 2

        thinking_ev = events[0]
        assert thinking_ev["event_type"] == "thinking"
        assert thinking_ev["data"]["agent_type"] == "worker"
        assert thinking_ev["data"]["message"] == "Analysing inputs"

        decision_ev = events[1]
        assert decision_ev["event_type"] == "decision"
        assert decision_ev["data"]["decision_type"] == "model_select"
        assert decision_ev["data"]["summary"] == "Chose fast model"
        assert decision_ev["data"]["details"]["model_id"] == "phi3"

    def test_replay_empty_project_returns_empty_list(self, replay_client) -> None:
        """Replay for a project with no events returns an empty list."""
        response = replay_client.get("/api/v1/projects/proj-no-events/events/replay")
        assert response.status_code == 200, response.text
        assert response.json() == []

    def test_replay_limit_caps_results(self, replay_client) -> None:
        """limit parameter caps the number of returned events."""
        project_id = "proj-replay-limit"
        for i in range(10):
            _push(project_id, "status", {"status": "running", "i": i})

        response = replay_client.get(
            f"/api/v1/projects/{project_id}/events/replay",
            params={"limit": 3},
        )
        assert response.status_code == 200, response.text
        events = response.json()
        assert len(events) == 3

        # Must be the first 3 in chronological order
        sequences = [e["sequence"] for e in events]
        assert sequences == sorted(sequences)


# ---------------------------------------------------------------------------
# Tests: invalid input
# ---------------------------------------------------------------------------


class TestReplayEndpointValidation:
    """Replay endpoint rejects unsafe project IDs."""

    @pytest.mark.parametrize(
        "bad_id",
        [
            "proj id with spaces",  # space not in safe charset
            "proj\x00null",  # null byte injection
            "a" * 200,  # exceeds 128-char limit
            "proj!injection",  # bang not in safe charset
            "proj@host",  # @ not in safe charset
        ],
    )
    def test_invalid_project_id_returns_400(self, replay_client, bad_id: str) -> None:
        """Unsafe project IDs (chars outside [A-Za-z0-9_-] or too long) return HTTP 400."""
        import urllib.parse

        # Percent-encode the full bad_id so it arrives as one path segment.
        # this avoids the HTTP client splitting it into multiple path segments.
        encoded = urllib.parse.quote(bad_id, safe="")
        response = replay_client.get(f"/api/v1/projects/{encoded}/events/replay")
        assert response.status_code == 400, f"Expected 400 for project_id={bad_id!r}, got {response.status_code}"
