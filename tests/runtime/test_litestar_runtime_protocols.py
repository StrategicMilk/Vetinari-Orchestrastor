"""Runtime journey tests — Litestar HTTP stack end-to-end (SESSION-30 / SESSION-30A).

Covers US-004 (health + project + coding routes), US-005 (input validation
and A2A protocol), US-006 (approvals, autonomy, decisions, CSRF enforcement),
and US-30A.4 (mounted request-level A2A tests: cards, error semantics).

Every request goes through the real Litestar ``TestClient`` — no handler
function is called directly.  Heavy subsystems (project manager, inference,
training) are mocked at the module level so no real I/O occurs.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from litestar.testing import TestClient

from tests.factories import TEST_MODEL_ID, TEST_PROJECT_ID

# -- Constants ----------------------------------------------------------------

# All mutation requests must carry this header to pass CSRF
_CSRF = {"X-Requested-With": "XMLHttpRequest"}

# Routes that accept POST/PUT/DELETE but are CSRF-exempt (machine-to-machine)
_A2A_POST = "/api/v1/a2a"

# A minimal valid A2A JSON-RPC request
_VALID_A2A_BODY = {"jsonrpc": "2.0", "method": "a2a.getAgentCard", "id": 1}

# Admin token env + header for bypassing admin_guard in tests
_ADMIN_TOKEN = "test-runtime-token"  # Token value used in tests to satisfy admin_guard
_ADMIN_HEADERS = {"X-Admin-Token": _ADMIN_TOKEN}


# -- App fixture --------------------------------------------------------------


@pytest.fixture(scope="session")
def app() -> Any:
    """Create a single Litestar app instance shared across the test session.

    Heavy subsystems are patched before ``create_app()`` is called so that
    no real file I/O, inference, or database access occurs during tests.
    The session scope avoids rebuilding the ~300-handler app per test.
    """
    # Patch the lifespan wiring so startup does not call real subsystems
    with (
        patch("vetinari.web.litestar_app._lifespan", _noop_lifespan),
        patch("vetinari.web.litestar_app._register_shutdown_handlers"),
    ):
        from vetinari.web.litestar_app import create_app

        return create_app(debug=False)


@pytest.fixture
def client(app: Any) -> Generator[TestClient, None, None]:
    """Yield a fresh ``TestClient`` per test."""
    with TestClient(app=app) as c:
        yield c


@asynccontextmanager
async def _noop_lifespan(app: Any):
    """Drop-in lifespan that skips all subsystem wiring."""
    yield


# -- TestHealthEndpoints (US-004) --------------------------------------------


class TestHealthEndpoints:
    """Verify that both health endpoints return 200 with valid JSON (US-004)."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """GET /health returns 200 and a JSON body with a status key."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, dict)
        assert "status" in body
        assert body["status"] == "ok"

    def test_api_health_returns_200(self, client: TestClient) -> None:
        """GET /api/v1/health returns 200 and contains a status field."""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, dict)
        assert body["status"] == "healthy"


# -- TestProjectLifecycle (US-004) -------------------------------------------


class TestProjectLifecycle:
    """Verify project list and detail endpoints return valid shapes (US-004)."""

    def test_project_list_returns_200(self, client: TestClient) -> None:
        """GET /api/projects returns 200 with a dict containing a projects key.

        The handler scans the filesystem — an empty or missing projects dir
        is valid and returns an empty list.
        """
        resp = client.get("/api/projects")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, dict)
        assert "projects" in body
        assert isinstance(body["projects"], list)

    def test_project_detail_returns_proper_state(self, client: TestClient) -> None:
        """GET /api/project/{id} returns 200, 404, or 503 for any project ID.

        A 404 is correct when the project doesn't exist; 503 is acceptable
        when the server is not fully wired.  500 is never acceptable.
        """
        resp = client.get(f"/api/project/{TEST_PROJECT_ID}")
        assert resp.status_code in {200, 404, 503}


# -- TestCodingRoutes (US-004) ------------------------------------------------


class TestCodingRoutes:
    """Verify coding task endpoints accept valid input and respond (US-004)."""

    def test_coding_task_success(self, client: TestClient) -> None:
        """POST /api/coding/task with a valid body returns 201.

        The admin_guard checks the connection host against a localhost allow-list.
        TestClient connects from ``testserver.local`` which is not in that list,
        so we patch ``is_admin_connection`` to return True and bypass the guard.
        The coding agent functions are patched at the source module so the handler
        never touches real I/O or a real model.
        """
        payload = {
            "description": "Write a hello-world function",
            "type": "implement",
            "language": "python",
        }
        mock_artifact = MagicMock()
        mock_artifact.to_dict.return_value = {"code": "hello"}
        mock_artifact.task_id = "task_abc"

        mock_agent = MagicMock()
        mock_agent.is_available.return_value = True
        mock_agent.run_task.return_value = mock_artifact

        mock_task = MagicMock()
        mock_task.task_id = "task_abc"

        with (
            patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}),
            patch("vetinari.coding_agent.get_coding_agent", return_value=mock_agent),
            patch("vetinari.coding_agent.make_code_agent_task", return_value=mock_task),
        ):
            resp = client.post("/api/coding/task", json=payload, headers={**_CSRF, **_ADMIN_HEADERS})

        assert resp.status_code == 201
        body = resp.json()
        assert body["success"] is True
        assert body["task_id"] == "task_abc"
        assert body["artifact"] == {"code": "hello"}

    def test_coding_multi_step_success(self, client: TestClient) -> None:
        """POST /api/coding/multi-step with a valid body returns 201.

        Same admin_guard bypass as ``test_coding_task_success``. Each subtask in
        the request produces one artifact via the mocked agent.
        """
        payload = {
            "plan_id": "p1",
            "subtasks": [
                {
                    "subtask_id": "st_1",
                    "type": "implement",
                    "description": "Step one",
                }
            ],
        }
        mock_artifact = MagicMock()
        mock_artifact.to_dict.return_value = {"code": "step"}
        mock_artifact.task_id = "st_1"

        mock_agent = MagicMock()
        mock_agent.is_available.return_value = True
        mock_agent.run_task.return_value = mock_artifact

        mock_task = MagicMock()
        mock_task.task_id = "st_1"

        with (
            patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}),
            patch("vetinari.coding_agent.get_coding_agent", return_value=mock_agent),
            patch("vetinari.coding_agent.make_code_agent_task", return_value=mock_task),
        ):
            resp = client.post("/api/coding/multi-step", json=payload, headers={**_CSRF, **_ADMIN_HEADERS})

        assert resp.status_code == 201
        body = resp.json()
        assert body["success"] is True
        assert body["plan_id"] == "p1"
        assert len(body["results"]) == 1
        assert body["results"][0]["subtask_id"] == "st_1"
        assert body["results"][0]["success"] is True
        assert body["results"][0]["artifact"] == {"code": "step"}


# -- TestA2AProtocol (US-005) ------------------------------------------------


class TestA2AProtocol:
    """Verify A2A JSON-RPC input validation returns 400 for bad requests (US-005)."""

    def test_a2a_malformed_json_returns_400(self, client: TestClient) -> None:
        """POST /api/v1/a2a with a non-dict body (list) returns 400 or 422.

        Litestar validates ``data: dict`` before calling the handler.  When the
        body is a JSON array, Litestar may return 422 (type validation error)
        before the handler's own 400 logic fires.  Both codes indicate the
        request was correctly rejected — the contract is "not 200/500".
        """
        # A2A is CSRF-exempt — no header needed
        resp = client.post(
            _A2A_POST,
            content=b"[1,2,3]",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code in {400, 422}, (
            f"POST {_A2A_POST} with list body returned {resp.status_code} — expected 400 or 422"
        )
        body = resp.json()
        # Litestar 422 uses "detail"; handler 400 uses "error" — either is valid
        assert "error" in body or "detail" in body

    def test_a2a_missing_fields_returns_400(self, client: TestClient) -> None:
        """POST /api/v1/a2a with an object missing required fields returns 400."""
        incomplete = {"jsonrpc": "2.0"}  # missing 'method' and 'id'
        resp = client.post(
            _A2A_POST,
            json=incomplete,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        error_msg = body["error"].get("message", "")
        assert "missing" in error_msg.lower()

    def test_a2a_valid_request_not_400(self, client: TestClient) -> None:
        """POST /api/v1/a2a with a complete JSON-RPC body does not return 400."""
        mock_transport = MagicMock()
        mock_transport.handle_request.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {},
        }
        with patch("vetinari.web.litestar_app._get_a2a_transport", return_value=mock_transport):
            resp = client.post(_A2A_POST, json=_VALID_A2A_BODY)

        assert resp.status_code != 400


# -- TestPreferencesValidation (US-005) --------------------------------------


class TestPreferencesValidation:
    """Verify PUT /api/v1/preferences rejects non-object bodies (US-005)."""

    def test_preferences_string_body_returns_400(self, client: TestClient) -> None:
        """PUT /api/v1/preferences with a JSON string body returns 400 or 422.

        Litestar validates ``data: dict`` before the handler runs.  A non-dict
        body may be rejected at the framework level (422) or by handler code (400).
        """
        resp = client.put(
            "/api/v1/preferences",
            content=b'"a plain string"',
            headers={**_CSRF, "Content-Type": "application/json"},
        )
        assert resp.status_code in {400, 422}

    def test_preferences_list_body_returns_400(self, client: TestClient) -> None:
        """PUT /api/v1/preferences with a JSON array body returns 400 or 422.

        Litestar validates ``data: dict`` before the handler runs.  A list body
        may be rejected at the framework level (422) or by handler code (400).
        """
        resp = client.put(
            "/api/v1/preferences",
            json=["item1", "item2"],
            headers=_CSRF,
        )
        assert resp.status_code in {400, 422}

    def test_preferences_valid_object_succeeds(self, client: TestClient) -> None:
        """PUT /api/v1/preferences with at least one recognised preference key returns 200.

        Empty objects are rejected with 422 under the hardened contract — callers
        must provide at least one preference key to update. This test verifies the
        success path by sending a body with a real recognised key.
        """
        mock_mgr = MagicMock()
        mock_mgr.get_all.return_value = {"theme": "dark"}
        # Handler calls set_many() and iterates .items() on the result.
        # set_many returns {key: True} for accepted keys, {key: False} for rejected.
        mock_mgr.set_many.return_value = {"theme": True}

        # Handler uses a local import — patch at the source module.
        with patch("vetinari.web.preferences.get_preferences_manager", return_value=mock_mgr):
            resp = client.put(
                "/api/v1/preferences",
                json={"theme": "dark"},
                headers=_CSRF,
            )

        assert resp.status_code == 200

    def test_preferences_empty_object_rejected(self, client: TestClient) -> None:
        """PUT /api/v1/preferences with an empty JSON object is rejected (422).

        Empty-body PUTs are malformed input for this endpoint — the caller must
        supply at least one preference key. This regression test locks in the
        hardened contract from SESSION-32 so future edits cannot quietly revert
        to accepting empty objects as no-op success.
        """
        resp = client.put(
            "/api/v1/preferences",
            json={},
            headers=_CSRF,
        )
        assert resp.status_code == 422, f"Empty preferences body must be rejected with 422, got {resp.status_code}"


# -- TestSettingsValidation (US-005) -----------------------------------------


class TestSettingsValidation:
    """Verify PUT /api/v1/settings rejects invalid input (US-005)."""

    def test_settings_string_body_returns_400(self, client: TestClient) -> None:
        """PUT /api/v1/settings with a JSON string body returns 400 or 422.

        Litestar validates ``data: dict`` before the handler runs.  A non-dict
        body may be rejected at the framework level (422) or by handler code (400).
        """
        resp = client.put(
            "/api/v1/settings",
            content=b'"just a string"',
            headers={**_CSRF, "Content-Type": "application/json"},
        )
        assert resp.status_code in {400, 422}

    def test_settings_empty_body_returns_400(self, client: TestClient) -> None:
        """PUT /api/v1/settings with an empty JSON object returns 400.

        An empty dict ``{}`` passes Litestar's type validation (it is still a
        dict) but the handler explicitly rejects it with 400 because no settings
        were provided.
        """
        resp = client.put(
            "/api/v1/settings",
            json={},
            headers=_CSRF,
        )
        assert resp.status_code == 400

    def test_settings_list_body_returns_400(self, client: TestClient) -> None:
        """PUT /api/v1/settings with a JSON array body returns 400 or 422.

        Litestar validates ``data: dict`` before the handler runs.  A list body
        may be rejected at the framework level (422) or by handler code (400).
        """
        resp = client.put(
            "/api/v1/settings",
            json=["a", "b"],
            headers=_CSRF,
        )
        assert resp.status_code in {400, 422}


# -- TestMalformedInputBoundary (US-005) -------------------------------------


# Routes that must reject non-object JSON with 400/422.
# Each entry is (method, path, payload) where payload is the bad body.
_MALFORMED_CASES = [
    ("post", "/api/coding/task", ["not", "an", "object"]),
    ("post", "/api/coding/multi-step", "a string body"),
    ("put", "/api/v1/preferences", [1, 2, 3]),
    ("put", "/api/v1/settings", "string"),
]


class TestMalformedInputBoundary:
    """Parametrized: all mutation routes reject non-object JSON (US-005)."""

    @pytest.fixture(autouse=True)
    def _bypass_admin_guard(self) -> Generator[None, None, None]:
        """Allow TestClient through admin_guard for coding routes.

        Coding routes carry ``guards=[admin_guard]`` which rejects requests from
        ``testserver.local`` (not in the localhost allow-list).  The guard fires
        *before* Litestar type validation, so without this bypass the coding
        entries in ``_MALFORMED_CASES`` would return 500 instead of 400/422.

        Patching only ``is_admin_connection`` is safe for non-guarded routes
        (preferences, settings) because those routes never consult the guard.
        """
        with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
            yield

    @pytest.mark.parametrize("method,path,bad_body", _MALFORMED_CASES)
    def test_non_object_body_rejected(
        self,
        client: TestClient,
        method: str,
        path: str,
        bad_body: Any,
    ) -> None:
        """Non-object JSON bodies must produce 400 or 422, never 200/500."""
        call = getattr(client, method)
        resp = call(path, json=bad_body, headers={**_CSRF, **_ADMIN_HEADERS})
        assert resp.status_code in {400, 422}, (
            f"{method.upper()} {path} with {type(bad_body).__name__} body "
            f"returned {resp.status_code} — expected 400 or 422"
        )


# -- TestApprovalsControl (US-006) -------------------------------------------


class TestApprovalsControl:
    """Verify approvals endpoints return bounded responses (US-006)."""

    def test_approvals_pending_returns_200(self, client: TestClient) -> None:
        """GET /api/v1/approvals/pending returns 200 with a list-shaped body."""
        mock_queue = MagicMock()
        mock_queue.get_pending.return_value = []

        # Handler uses local import — patch at the source module.
        # admin_guard requires VETINARI_ADMIN_TOKEN env var + X-Admin-Token header.
        with (
            patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}),
            patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue),
        ):
            resp = client.get("/api/v1/approvals/pending", headers=_ADMIN_HEADERS)

        assert resp.status_code == 200
        body = resp.json()
        # Shape: list (the handler returns a list directly)
        assert isinstance(body, list)

    def test_approvals_post_requires_csrf(self, client: TestClient) -> None:
        """POST /api/v1/approvals/{id}/approve without X-Requested-With → 403."""
        resp = client.post(
            "/api/v1/approvals/some-action-id/approve",
            json={},
            # Deliberately omit X-Requested-With
        )
        assert resp.status_code == 403


# -- TestAutonomyControl (US-006) --------------------------------------------


class TestAutonomyControl:
    """Verify autonomy mode endpoints are accessible and CSRF-protected (US-006)."""

    def test_autonomy_status_get_returns_200(self, client: TestClient) -> None:
        """GET /api/v1/autonomy/status returns 200 or 503 (graceful degradation).

        The handler reads a local policies file — no governor call needed.
        """
        resp = client.get("/api/v1/autonomy/status")
        assert resp.status_code in {200, 503}

    def test_autonomy_promote_requires_csrf(self, client: TestClient) -> None:
        """POST /api/v1/autonomy/promote/{type} without X-Requested-With → 403."""
        resp = client.post(
            "/api/v1/autonomy/promote/file_write",
            json={},
            # Deliberately omit CSRF header
        )
        assert resp.status_code == 403


# -- TestDecisionsControl (US-006) -------------------------------------------


class TestDecisionsControl:
    """Verify decisions log endpoint returns a valid response (US-006)."""

    def test_decisions_log_returns_200_or_503(self, client: TestClient) -> None:
        """GET /api/v1/decisions/log returns 200 or 503 (graceful degradation)."""
        mock_queue = MagicMock()
        mock_queue.get_decision_log.return_value = []

        # Handler uses local import — patch at the source module.
        # admin_guard requires VETINARI_ADMIN_TOKEN env var + X-Admin-Token header.
        with (
            patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}),
            patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue),
        ):
            resp = client.get("/api/v1/decisions/log", headers=_ADMIN_HEADERS)

        assert resp.status_code in {200, 503}
        if resp.status_code == 200:
            body = resp.json()
            assert isinstance(body, dict)
            assert body["entries"] == []


# -- TestCSRFEnforcement (US-006) --------------------------------------------


# Mutation endpoints that MUST enforce CSRF (requires X-Requested-With).
# Format: (method, path, json_body)
_CSRF_PROTECTED = [
    ("post", "/api/v1/approvals/abc/approve", {}),
    ("post", "/api/v1/approvals/abc/reject", {}),
    ("put", "/api/v1/preferences", {"theme": "dark"}),
    ("put", "/api/v1/settings", {"log_level": "INFO"}),
    ("post", "/api/v1/autonomy/promote/file_write", {}),
    ("post", "/api/v1/milestones/approve", {"action": "approve"}),
]


class TestCSRFEnforcement:
    """CSRF header enforcement: mutation requests without header → 403 (US-006)."""

    @pytest.mark.parametrize("method,path,body", _CSRF_PROTECTED)
    def test_mutation_without_csrf_header_returns_403(
        self,
        client: TestClient,
        method: str,
        path: str,
        body: dict,
    ) -> None:
        """Mutation request without X-Requested-With must return 403."""
        call = getattr(client, method)
        resp = call(path, json=body)  # No CSRF header
        assert resp.status_code == 403, (
            f"{method.upper()} {path} without CSRF header returned {resp.status_code} — expected 403"
        )

    def test_a2a_post_exempt_from_csrf(self, client: TestClient) -> None:
        """POST /api/v1/a2a works without X-Requested-With (CSRF-exempt)."""
        mock_transport = MagicMock()
        mock_transport.handle_request.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {},
        }
        with patch("vetinari.web.litestar_app._get_a2a_transport", return_value=mock_transport):
            resp = client.post(_A2A_POST, json=_VALID_A2A_BODY)
            # No CSRF header — should NOT be 403
            assert resp.status_code != 403

    def test_get_requests_exempt_from_csrf(self, client: TestClient) -> None:
        """GET requests are never blocked by CSRF middleware."""
        resp = client.get("/health")
        assert resp.status_code == 200


# -- TestA2ACards (US-30A.1) -------------------------------------------------


class TestA2ACards:
    """Verify /api/v1/a2a/cards returns live agent cards pointing at real endpoints (US-30A.1).

    The cards endpoint calls ``get_all_cards()`` directly — no transport mock
    is required.  These tests prove the Litestar route is registered, the three
    cards are returned, and every card URL is the shared ``/api/v1/a2a`` mount
    point rather than a legacy per-agent path.
    """

    def test_cards_endpoint_returns_200_with_three_agents(self, client: TestClient) -> None:
        """GET /api/v1/a2a/cards returns 200 with a list of exactly three agent cards."""
        resp = client.get("/api/v1/a2a/cards")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) == 3, f"Expected 3 agent cards, got {len(body)}: {[c.get('name') for c in body]}"

    def test_all_card_urls_point_at_mounted_endpoint(self, client: TestClient) -> None:
        """Every card advertises the shared /api/v1/a2a endpoint, not per-agent legacy paths.

        All three cards must use the unified mount point.  Legacy per-agent URL
        segments (``/a2a/foreman``, ``/a2a/worker``, ``/a2a/inspector``) must
        not appear — those routes do not exist.
        """
        resp = client.get("/api/v1/a2a/cards")
        body = resp.json()
        for card in body:
            card_name = card.get("name", "<unnamed>")
            card_url = card.get("url", "")
            assert "/api/v1/a2a" in card_url, f"Card {card_name!r} has URL {card_url!r} — expected '/api/v1/a2a'"
            # Legacy per-agent paths must not appear
            assert "/a2a/foreman" not in card_url, f"Card {card_name!r} uses legacy foreman path: {card_url!r}"
            assert "/a2a/worker" not in card_url, f"Card {card_name!r} uses legacy worker path: {card_url!r}"
            assert "/a2a/inspector" not in card_url, f"Card {card_name!r} uses legacy inspector path: {card_url!r}"

    def test_worker_card_skill_count_matches_live_modes(self, client: TestClient) -> None:
        """Worker card skill count equals WorkerAgent.MODES length — card never drifts from implementation.

        Each skill's ``name`` field must match a live mode key.  This proves
        the card is computed from the live WorkerAgent, not hard-coded.
        """
        from vetinari.agents.consolidated.worker_agent import WorkerAgent

        resp = client.get("/api/v1/a2a/cards")
        body = resp.json()
        worker = next((c for c in body if c.get("name") == "Vetinari Worker"), None)
        assert worker is not None, f"No 'Vetinari Worker' card in response: {[c.get('name') for c in body]}"

        live_mode_count = len(WorkerAgent.MODES)
        assert len(worker["skills"]) == live_mode_count, (
            f"Worker card has {len(worker['skills'])} skills but WorkerAgent.MODES has {live_mode_count} entries"
        )

        # Every skill name must correspond to a live mode key
        live_mode_names = set(WorkerAgent.MODES.keys())
        card_skill_names = {s["name"] for s in worker["skills"]}
        assert card_skill_names == live_mode_names, (
            f"Skill name mismatch — in card but not MODES: {card_skill_names - live_mode_names}, "
            f"in MODES but not card: {live_mode_names - card_skill_names}"
        )

    def test_worker_card_description_reflects_live_counts(self, client: TestClient) -> None:
        """Worker card description embeds the live mode count and mode-group count.

        The description is generated from ``WorkerAgent.MODES`` and
        ``MODE_GROUPS`` at call time so it can never fall out of sync.
        """
        from vetinari.agents.consolidated.worker_agent import MODE_GROUPS, WorkerAgent

        resp = client.get("/api/v1/a2a/cards")
        body = resp.json()
        worker = next((c for c in body if c.get("name") == "Vetinari Worker"), None)
        assert worker is not None

        desc = worker.get("description", "")
        live_mode_count = len(WorkerAgent.MODES)
        live_group_count = len(set(MODE_GROUPS.values()))

        assert f"{live_mode_count} modes" in desc, (
            f"Worker description {desc!r} does not contain '{live_mode_count} modes'"
        )
        assert f"{live_group_count} mode groups" in desc, (
            f"Worker description {desc!r} does not contain '{live_group_count} mode groups'"
        )


# -- TestA2AErrorSemantics (US-30A.4) ----------------------------------------


class TestA2AErrorSemantics:
    """Verify A2A JSON-RPC error codes for task-not-found and guardrail-blocked (US-30A.4).

    JSON-RPC errors must use dedicated application-level error codes, not the
    generic -32603 internal error code.  These tests prove the transport's
    error code routing is wired end-to-end through the Litestar stack.
    """

    def test_task_status_missing_id_returns_dedicated_error_code(self, client: TestClient) -> None:
        """a2a.taskStatus on an unknown task returns -32001, not -32603 (generic internal error).

        The transport raises ``_TaskNotFoundError`` internally, which is caught
        and mapped to ``_ERR_TASK_NOT_FOUND = -32001``.  The error must be
        at the top-level JSON-RPC envelope (``"error"`` key, no ``"result"``).
        """
        # Use a real transport (not mocked) so we exercise the real error-code path.
        # The transport starts with an empty in-memory task store, so any taskId
        # that was never submitted will trigger _TaskNotFoundError.
        from vetinari.a2a.executor import VetinariA2AExecutor
        from vetinari.a2a.transport import A2ATransport

        # Fresh transport with no tasks — all taskId lookups will miss
        fresh_transport = A2ATransport(executor=VetinariA2AExecutor(recover_on_init=False))

        req = {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "a2a.taskStatus",
            "params": {"taskId": "no-such-task-id-exists"},
        }
        with patch("vetinari.web.litestar_app._get_a2a_transport", return_value=fresh_transport):
            resp = client.post(_A2A_POST, json=req)

        # JSON-RPC errors are still delivered over HTTP 200
        assert resp.status_code == 200, f"Expected HTTP 200 for JSON-RPC error, got {resp.status_code}"
        body = resp.json()

        # Error must be at top level, not nested inside result
        assert "error" in body, f"Expected top-level 'error' envelope in response, got: {body}"
        assert "result" not in body, f"Unexpected 'result' key alongside 'error': {body}"

        error_code = body["error"]["code"]
        assert error_code == -32001, (
            f"Expected TASK_NOT_FOUND error code -32001, got {error_code}. Full error: {body['error']}"
        )
        assert "not found" in body["error"]["message"].lower(), (
            f"Expected 'not found' in error message, got: {body['error']['message']!r}"
        )

    def test_task_send_blocked_input_skips_execution_and_returns_top_level_error(self, client: TestClient) -> None:
        """Blocked input fails BEFORE executor.execute and returns top-level JSON-RPC -32002 error.

        The transport runs the input guardrail BEFORE calling
        ``executor.execute()``.  When the guardrail blocks the input, the
        executor must never be invoked, and the response must carry the
        dedicated ``_ERR_GUARDRAIL_BLOCKED = -32002`` code at the top level.
        """
        from vetinari.a2a.transport import A2ATransport

        # Fake executor — its execute() must NOT be called
        fake_executor = MagicMock()
        fake_executor.execute.return_value = MagicMock(
            to_dict=lambda: {"taskId": "x", "status": "completed", "outputData": "MUST NOT APPEAR", "error": ""}
        )

        # Guardrail that blocks input unconditionally
        deny_result = MagicMock()
        deny_result.allowed = False
        deny_result.reason = "test-policy-block"

        allow_result = MagicMock()
        allow_result.allowed = True

        fake_guardrails = MagicMock()
        fake_guardrails.check_input.return_value = deny_result
        fake_guardrails.check_output.return_value = allow_result

        transport = A2ATransport(executor=fake_executor)

        req = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "a2a.taskSend",
            "params": {"taskType": "build", "inputData": {"text": "sensitive content"}},
        }

        with (
            patch("vetinari.web.litestar_app._get_a2a_transport", return_value=transport),
            patch("vetinari.safety.guardrails.get_guardrails", return_value=fake_guardrails),
        ):
            resp = client.post(_A2A_POST, json=req)

        body = resp.json()

        # Top-level error envelope — NOT nested inside result
        assert "error" in body, f"Expected top-level 'error' in response, got: {body}"
        assert "result" not in body, f"Unexpected 'result' key alongside 'error': {body}"

        error_code = body["error"]["code"]
        assert error_code == -32002, (
            f"Expected GUARDRAIL_BLOCKED error code -32002, got {error_code}. Full error: {body['error']}"
        )

        # Executor must NOT have been called — input was rejected before execution
        assert not fake_executor.execute.called, (
            "executor.execute() was called despite guardrail blocking the input — "
            "guardrail check must run BEFORE executor.execute()"
        )
