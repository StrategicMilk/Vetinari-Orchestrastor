"""Mounted request-level governance tests for ponder, visualization, and ADR routes.

Tests go through the full Litestar HTTP stack via TestClient — not handler-direct calls.
Covers error-handling hardening added in SESSION-27E.1:
- Ponder: try/except wrapping so helper failures return structured 500 (not raw tracebacks)
- Visualization: SSE stream returns 404 for nonexistent plans; approve-gate rejects orphaned gates
- ADR: read routes return 500 on system failure; propose/accept/from-plan/link-plan validate inputs
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from litestar.testing import TestClient

# CSRF header required for all POST/PUT through the real app stack
_CSRF = {"X-Requested-With": "XMLHttpRequest"}

# Admin token injected via env so admin_guard passes without real auth
_ADMIN_TOKEN = "test-governance-token"
_ADMIN_HEADERS = {**_CSRF, "X-Admin-Token": _ADMIN_TOKEN}


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def litestar_app():
    """Full Litestar app with heavy subsystems patched out.

    Patches _register_shutdown_handlers so signal registration doesn't
    fail in the test thread, and sets VETINARI_ADMIN_TOKEN so admin_guard
    accepts the test token without network or filesystem calls.
    """
    import os

    os.environ["VETINARI_ADMIN_TOKEN"] = _ADMIN_TOKEN
    try:
        with patch("vetinari.web.litestar_app._register_shutdown_handlers"):
            from vetinari.web.litestar_app import create_app

            yield create_app(debug=True)
    finally:
        os.environ.pop("VETINARI_ADMIN_TOKEN", None)


# ---------------------------------------------------------------------------
# TestPonderGovernance
# ---------------------------------------------------------------------------


class TestPonderGovernance:
    """Ponder route handlers return structured 500 JSON when helpers raise."""

    @pytest.mark.parametrize(
        "url,patch_target",
        [
            (
                "/api/ponder/templates",
                "vetinari.models.ponder.PonderEngine",
            ),
            (
                "/api/ponder/models",
                "vetinari.models.ponder.get_available_models",
            ),
            (
                "/api/ponder/health",
                "vetinari.models.ponder.get_ponder_health",
            ),
            (
                "/api/ponder/plan/test-plan-123",
                "vetinari.models.ponder.get_ponder_results_for_plan",
            ),
        ],
    )
    def test_get_route_returns_500_json_when_helper_raises(self, litestar_app, url: str, patch_target: str) -> None:
        """GET ponder routes return structured 500 when helpers raise RuntimeError.

        The response must be JSON with status='error', not a raw traceback or
        Litestar's unhandled exception page.
        """
        with patch(patch_target, side_effect=RuntimeError("simulated failure")):
            with TestClient(app=litestar_app) as client:
                resp = client.get(url)

        assert resp.status_code == 500, f"Expected 500 for {url}, got {resp.status_code}"
        body = resp.json()
        assert body.get("status") == "error", f"Expected status='error', got: {body}"
        assert isinstance(body["error"], str) and body["error"], f"Expected non-empty error message in body: {body}"

    def test_ponder_run_plan_returns_500_json_when_helper_raises(self, litestar_app) -> None:
        """POST /api/ponder/plan/{plan_id} returns structured 500 when ponder engine raises."""
        with patch(
            "vetinari.models.ponder.ponder_project_for_plan",
            side_effect=RuntimeError("engine exploded"),
        ):
            with TestClient(app=litestar_app) as client:
                resp = client.post(
                    "/api/ponder/plan/test-plan-xyz",
                    headers=_ADMIN_HEADERS,
                )

        assert resp.status_code == 500
        body = resp.json()
        assert body.get("status") == "error"

    def test_ponder_templates_returns_data_on_success(self, litestar_app) -> None:
        """GET /api/ponder/templates returns 200 with templates list when engine works."""
        mock_engine = MagicMock()
        mock_engine.get_template_prompts.return_value = ["prompt-a", "prompt-b"]

        with patch("vetinari.models.ponder.PonderEngine", return_value=mock_engine):
            with TestClient(app=litestar_app) as client:
                resp = client.get("/api/ponder/templates")

        assert resp.status_code == 200
        body = resp.json()
        assert body["templates"] == ["prompt-a", "prompt-b"]
        assert body["total"] == 2


# ---------------------------------------------------------------------------
# TestVisualizationGovernance
# ---------------------------------------------------------------------------


class TestVisualizationGovernance:
    """Visualization route handlers enforce plan-existence checks."""

    def test_stream_returns_404_for_nonexistent_plan(self, litestar_app) -> None:
        """GET /api/plans/{plan_id}/visualization/stream returns 404 JSON when plan not found.

        The client must receive a structured error response, NOT a 200 with
        Content-Type: text/event-stream that never emits meaningful events.
        """
        mock_manager = MagicMock()
        mock_manager.get_plan.return_value = None  # plan does not exist

        with patch("vetinari.planning.get_plan_manager", return_value=mock_manager):
            with TestClient(app=litestar_app) as client:
                resp = client.get("/api/plans/nonexistent-plan/visualization/stream")

        assert resp.status_code == 404, f"Expected 404, got {resp.status_code}"
        body = resp.json()
        assert body.get("status") == "error"
        # Must NOT be an SSE stream
        content_type = resp.headers.get("content-type", "")
        assert "text/event-stream" not in content_type, "Response should not be SSE for nonexistent plan"

    def test_stream_returns_sse_for_existing_plan(self, litestar_app) -> None:
        """GET /api/plans/{plan_id}/visualization/stream opens SSE when plan exists."""
        mock_plan = MagicMock()
        mock_manager = MagicMock()
        mock_manager.get_plan.return_value = mock_plan

        # _get_viz_queue must return a real queue so the generator doesn't crash
        import queue as _queue

        real_queue = _queue.Queue()
        # Put a sentinel so the generator can exit cleanly in tests
        real_queue.put(None)

        with (
            patch("vetinari.planning.get_plan_manager", return_value=mock_manager),
            patch("vetinari.web.visualization._get_viz_queue", return_value=real_queue),
        ):
            with TestClient(app=litestar_app) as client:
                resp = client.get(
                    "/api/plans/real-plan-id/visualization/stream",
                    headers={"Accept": "text/event-stream"},
                )

        # A 200 is expected — we're not asserting full SSE body, just that
        # the plan-exists check passed and no 404 was returned.
        assert resp.status_code == 200

    def test_approve_gate_rejects_orphaned_gate(self, litestar_app) -> None:
        """POST approve-gate returns error when gate exists but plan does not.

        Seeds _pending_gates with a gate for a plan that doesn't exist in the
        plan manager, then asserts the endpoint rejects the request rather than
        mutating orphaned state.
        """
        from vetinari.web.visualization import _pending_gates, _pending_gates_lock

        orphan_plan_id = "orphan-plan-no-such-plan"
        gate_entry = {"task_id": "task-1", "status": "pending"}

        with _pending_gates_lock:
            _pending_gates[orphan_plan_id] = gate_entry

        try:
            mock_manager = MagicMock()
            mock_manager.get_plan.return_value = None  # plan does not exist

            with patch("vetinari.planning.get_plan_manager", return_value=mock_manager):
                with TestClient(app=litestar_app) as client:
                    resp = client.post(
                        f"/api/plans/{orphan_plan_id}/approve-gate",
                        json={"action": "approve"},
                        headers=_ADMIN_HEADERS,
                    )
        finally:
            with _pending_gates_lock:
                _pending_gates.pop(orphan_plan_id, None)

        # Must not succeed — approving a gate for a nonexistent plan is an error
        assert resp.status_code in (404, 500), (
            f"Expected 404 or 500 for orphaned gate, got {resp.status_code}: {resp.json()}"
        )
        body = resp.json()
        assert body.get("status") == "error"

    def test_approve_gate_returns_404_when_no_gate(self, litestar_app) -> None:
        """POST approve-gate returns 404 when no pending gate exists for the plan."""
        with TestClient(app=litestar_app) as client:
            resp = client.post(
                "/api/plans/no-gate-plan/approve-gate",
                json={"action": "approve"},
                headers=_ADMIN_HEADERS,
            )

        assert resp.status_code == 404
        body = resp.json()
        assert body.get("status") == "error"

    def test_approve_gate_returns_400_for_invalid_action(self, litestar_app) -> None:
        """POST approve-gate returns 400 when action is not 'approve' or 'reject'."""
        with TestClient(app=litestar_app) as client:
            resp = client.post(
                "/api/plans/some-plan/approve-gate",
                json={"action": "delete"},
                headers=_ADMIN_HEADERS,
            )

        assert resp.status_code == 400
        body = resp.json()
        assert body.get("status") == "error"


# ---------------------------------------------------------------------------
# TestADRGovernance
# ---------------------------------------------------------------------------


class TestADRGovernance:
    """ADR route handlers validate inputs and handle ADR system failures gracefully."""

    # -- Read routes: return 500 when ADR system raises -----------------------

    @pytest.mark.parametrize(
        "url",
        [
            "/api/adr",
            "/api/adr/statistics",
            "/api/adr/recent",
            "/api/adr/is-high-stakes",
            "/api/adr/ADR-0001",
        ],
    )
    def test_read_routes_return_500_when_adr_system_raises(self, litestar_app, url: str) -> None:
        """GET ADR read routes return structured 500 JSON when get_adr_system() raises.

        Every read endpoint must catch the failure and return a JSON error body
        rather than propagating an unhandled exception.
        """
        with patch("vetinari.adr.get_adr_system", side_effect=RuntimeError("adr system down")):
            with TestClient(app=litestar_app) as client:
                resp = client.get(url)

        assert resp.status_code == 500, f"Expected 500 for {url}, got {resp.status_code}"
        body = resp.json()
        assert body.get("status") == "error", f"Expected status='error' in body: {body}"

    # -- api_adr_propose: requires non-empty context --------------------------

    def test_propose_returns_400_when_context_missing(self, litestar_app) -> None:
        """POST /api/adr/propose with empty body returns 400."""
        with TestClient(app=litestar_app) as client:
            resp = client.post("/api/adr/propose", json={}, headers=_ADMIN_HEADERS)

        assert resp.status_code == 400
        body = resp.json()
        assert body.get("status") == "error"
        err = body.get("error", "").lower()
        assert "context" in err or "empty" in err, f"Expected error about context or empty body, got: {err}"

    def test_propose_returns_400_when_context_empty_string(self, litestar_app) -> None:
        """POST /api/adr/propose with context='' returns 400."""
        with TestClient(app=litestar_app) as client:
            resp = client.post(
                "/api/adr/propose",
                json={"context": ""},
                headers=_ADMIN_HEADERS,
            )

        assert resp.status_code == 400
        body = resp.json()
        assert body.get("status") == "error"

    def test_propose_succeeds_with_valid_context(self, litestar_app) -> None:
        """POST /api/adr/propose with valid context returns 200."""
        mock_proposal = MagicMock()
        mock_proposal.question = "How should we store sessions?"
        mock_proposal.options = ["Redis", "DB", "Memory"]
        mock_proposal.recommended = 0
        mock_proposal.rationale = "Redis is fastest."

        mock_system = MagicMock()
        mock_system.generate_proposal.return_value = mock_proposal

        with patch("vetinari.adr.get_adr_system", return_value=mock_system):
            with TestClient(app=litestar_app) as client:
                resp = client.post(
                    "/api/adr/propose",
                    json={"context": "We need to pick a session store"},
                    headers=_ADMIN_HEADERS,
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["question"] == "How should we store sessions?"
        assert body["recommended"] == 0

    # -- api_adr_propose_accept: requires question and title ------------------

    def test_propose_accept_returns_400_with_empty_body(self, litestar_app) -> None:
        """POST /api/adr/propose/accept with {} returns 400."""
        with TestClient(app=litestar_app) as client:
            resp = client.post("/api/adr/propose/accept", json={}, headers=_ADMIN_HEADERS)

        assert resp.status_code == 400
        body = resp.json()
        assert body.get("status") == "error"

    def test_propose_accept_returns_400_when_title_missing(self, litestar_app) -> None:
        """POST /api/adr/propose/accept with question but no title returns 400."""
        with TestClient(app=litestar_app) as client:
            resp = client.post(
                "/api/adr/propose/accept",
                json={"question": "Which database?", "options": ["A", "B"]},
                headers=_ADMIN_HEADERS,
            )

        assert resp.status_code == 400
        body = resp.json()
        assert body.get("status") == "error"
        assert "title" in body.get("error", "").lower()

    # -- api_adr_deprecate: replacement_id must be str or None ----------------

    def test_deprecate_returns_400_when_replacement_id_is_dict(self, litestar_app) -> None:
        """POST /api/adr/{id}/deprecate returns 400 when replacement_id is a dict."""
        mock_adr = MagicMock()
        mock_adr.to_dict.return_value = {"id": "ADR-0001"}
        mock_system = MagicMock()
        mock_system.get_adr.return_value = mock_adr

        with patch("vetinari.adr.get_adr_system", return_value=mock_system):
            with TestClient(app=litestar_app) as client:
                resp = client.post(
                    "/api/adr/ADR-0001/deprecate",
                    json={"replacement_id": {"nested": "ADR-0010"}},
                    headers=_ADMIN_HEADERS,
                )

        assert resp.status_code == 400
        body = resp.json()
        assert body.get("status") == "error"
        assert "replacement_id" in body.get("error", "").lower()

    def test_deprecate_accepts_string_replacement_id(self, litestar_app) -> None:
        """POST /api/adr/{id}/deprecate succeeds when replacement_id is a valid string."""
        mock_adr = MagicMock()
        mock_adr.to_dict.return_value = {"id": "ADR-0001", "status": "deprecated"}
        mock_system = MagicMock()
        mock_system.deprecate_adr.return_value = mock_adr

        with patch("vetinari.adr.get_adr_system", return_value=mock_system):
            with TestClient(app=litestar_app) as client:
                resp = client.post(
                    "/api/adr/ADR-0001/deprecate",
                    json={"replacement_id": "ADR-0010"},
                    headers=_ADMIN_HEADERS,
                )

        assert resp.status_code == 200

    # -- api_add_adr_from_plan: required fields must be strings ---------------

    def test_from_plan_returns_400_when_fields_are_dicts(self, litestar_app) -> None:
        """POST /api/adr/from-plan returns 400 when required fields are dicts, not strings."""
        with TestClient(app=litestar_app) as client:
            resp = client.post(
                "/api/adr/from-plan",
                json={
                    "adr_id": {"x": 1},
                    "title": {"x": 1},
                    "context": {"x": 1},
                    "decision": {"x": 1},
                },
                headers=_ADMIN_HEADERS,
            )

        assert resp.status_code == 400
        body = resp.json()
        assert body.get("status") == "error"

    def test_from_plan_returns_400_when_fields_missing(self, litestar_app) -> None:
        """POST /api/adr/from-plan returns 400 when required fields are absent."""
        with TestClient(app=litestar_app) as client:
            resp = client.post(
                "/api/adr/from-plan",
                json={},
                headers=_ADMIN_HEADERS,
            )

        assert resp.status_code == 400
        body = resp.json()
        assert body.get("status") == "error"

    # -- api_adr_link_plan: plan_id must be a non-empty string ----------------

    def test_link_plan_returns_400_when_plan_id_is_list(self, litestar_app) -> None:
        """POST /api/adr/{id}/link-plan returns 400 when plan_id is a list."""
        mock_adr = MagicMock()
        mock_system = MagicMock()
        mock_system.get_adr.return_value = mock_adr

        with patch("vetinari.adr.get_adr_system", return_value=mock_system):
            with TestClient(app=litestar_app) as client:
                resp = client.post(
                    "/api/adr/ADR-0001/link-plan",
                    json={"plan_id": ["plan-1"]},
                    headers=_ADMIN_HEADERS,
                )

        assert resp.status_code == 400
        body = resp.json()
        assert body.get("status") == "error"
        assert "plan_id" in body.get("error", "").lower()

    def test_link_plan_returns_400_when_plan_id_missing(self, litestar_app) -> None:
        """POST /api/adr/{id}/link-plan returns 400 when plan_id is absent."""
        mock_adr = MagicMock()
        mock_system = MagicMock()
        mock_system.get_adr.return_value = mock_adr

        with patch("vetinari.adr.get_adr_system", return_value=mock_system):
            with TestClient(app=litestar_app) as client:
                resp = client.post(
                    "/api/adr/ADR-0001/link-plan",
                    json={},
                    headers=_ADMIN_HEADERS,
                )

        assert resp.status_code == 400
        body = resp.json()
        assert body.get("status") == "error"
