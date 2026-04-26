"""Tests for vetinari.web.litestar_app — Litestar ASGI application."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from litestar import Litestar
from litestar.testing import TestClient

from vetinari.web.litestar_app import (
    _A2A_METHODS_RETURNING_201,
    _JSONRPC_REQUIRED_FIELDS,
    _create_a2a_handlers,
    _create_health_handler,
    _get_a2a_transport,
    create_app,
    get_app,
)
from vetinari.web.litestar_skills_api import create_skills_api_handlers as _create_skills_handlers

_A2A_ADMIN_TOKEN = "test-a2a-admin"
_A2A_ADMIN_HEADERS = {"X-Admin-Token": _A2A_ADMIN_TOKEN}


class TestLitestarImport:
    """Test that the Litestar app module is importable."""

    def test_module_import(self):
        """litestar_app module should be importable."""
        from vetinari.web import litestar_app

        assert hasattr(litestar_app, "create_app")
        assert hasattr(litestar_app, "get_app")

    def test_create_app_without_litestar_raises(self):
        """create_app should raise RuntimeError when Litestar not installed."""
        with patch("vetinari.web.litestar_app._LITESTAR_AVAILABLE", False):
            from vetinari.web.litestar_app import create_app

            with pytest.raises(RuntimeError, match="Litestar is not installed"):
                create_app()


class TestLitestarApp:
    """Tests for the Litestar application factory."""

    def test_create_app_returns_litestar_instance(self):
        """create_app should return a Litestar instance."""
        app = create_app()
        assert isinstance(app, Litestar)

    def test_health_endpoint_registered(self):
        """Health check endpoint should be registered."""
        app = create_app()
        route_paths = [r.path for r in app.routes]
        assert "/health" in route_paths

    def test_skills_api_routes_registered(self):
        """Skills API routes should be registered."""
        app = create_app()
        route_paths = [r.path for r in app.routes]
        assert "/api/v1/skills/catalog" in route_paths

    def test_get_app_singleton(self):
        """get_app should return the same instance on repeated calls."""
        import vetinari.web.litestar_app as mod

        old_app = mod._app
        mod._app = None
        try:
            app1 = mod.get_app()
            app2 = mod.get_app()
            assert app1 is app2
        finally:
            mod._app = old_app


class TestLitestarHandlers:
    """Test route handler creation."""

    def test_health_handler_creation(self):
        """Health handler should be created when Litestar is available."""
        handler = _create_health_handler()
        assert handler is not None
        assert callable(handler)

    def test_skills_handlers_creation(self):
        """Skills handlers should be created when Litestar is available."""
        handlers = _create_skills_handlers()
        # create_skills_api_handlers covers catalog, capabilities, tags, agent, summaries,
        # summary, trust elevation, output validation, proposal, and validation detail routes
        assert len(handlers) == 10

    def test_health_handler_none_without_litestar(self):
        """Health handler should be None when Litestar is not available."""
        with patch("vetinari.web.litestar_app._LITESTAR_AVAILABLE", False):
            from vetinari.web.litestar_app import _create_health_handler

            assert _create_health_handler() is None

    def test_skills_handlers_empty_without_litestar(self):
        """Skills handlers should be empty list without Litestar."""
        with patch("vetinari.web.litestar_skills_api._LITESTAR_AVAILABLE", False):
            from vetinari.web.litestar_skills_api import create_skills_api_handlers

            assert create_skills_api_handlers() == []


# ── A2A HTTP semantics tests ─────────────────────────────────────────


class TestA2ATransportSingleton:
    """Verify that _get_a2a_transport() returns a stable singleton."""

    def test_singleton_returns_same_instance(self) -> None:
        """Two consecutive calls must return the exact same object."""
        import vetinari.web.litestar_app as mod

        old = mod._a2a_transport
        mod._a2a_transport = None  # reset to force re-creation
        try:
            t1 = _get_a2a_transport()
            t2 = _get_a2a_transport()
            assert t1 is t2, "Singleton returned different objects on consecutive calls"
        finally:
            mod._a2a_transport = old

    def test_singleton_is_a2a_transport_instance(self) -> None:
        """The singleton must be an A2ATransport instance."""
        import vetinari.web.litestar_app as mod
        from vetinari.a2a.transport import A2ATransport

        old = mod._a2a_transport
        mod._a2a_transport = None
        try:
            transport = _get_a2a_transport()
            assert isinstance(transport, A2ATransport)
        finally:
            mod._a2a_transport = old


class TestA2AStatusCodeConstants:
    """Sanity-check the module-level HTTP-routing constants."""

    def test_task_send_in_201_set(self) -> None:
        """a2a.taskSend must be in the 201 method set."""
        assert "a2a.taskSend" in _A2A_METHODS_RETURNING_201

    def test_get_agent_card_not_in_201_set(self) -> None:
        """a2a.getAgentCard must NOT be in the 201 method set (reads return 200)."""
        assert "a2a.getAgentCard" not in _A2A_METHODS_RETURNING_201

    def test_task_status_not_in_201_set(self) -> None:
        """a2a.taskStatus must NOT be in the 201 method set (status queries return 200)."""
        assert "a2a.taskStatus" not in _A2A_METHODS_RETURNING_201

    def test_required_fields_complete(self) -> None:
        """The required-fields set must include jsonrpc, method, and id."""
        assert frozenset({"jsonrpc", "method", "id"}) == _JSONRPC_REQUIRED_FIELDS


class TestA2AHandlerHTTPStatus:
    """Verify HTTP status codes returned by the A2A POST handlers via the real Litestar stack."""

    @pytest.fixture
    def a2a_client(self):
        """TestClient for a minimal Litestar app that includes only the A2A handlers."""
        import vetinari.web.litestar_app as mod

        # Use a fresh transport for each test to prevent cross-test state leakage
        old = mod._a2a_transport
        mod._a2a_transport = None
        try:
            with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _A2A_ADMIN_TOKEN}):
                handlers = _create_a2a_handlers()
                app = Litestar(route_handlers=handlers)
                with TestClient(app=app) as client:
                    yield client
        finally:
            mod._a2a_transport = old

    def test_malformed_body_not_a_dict_returns_4xx(self, a2a_client) -> None:
        """A non-dict body (e.g. a list) to /api/v1/a2a must return 400 or 422.

        Litestar may reject type-mismatched bodies at the framework level
        (returning 422) before the handler runs, or our validation code may
        return 400.  Either is an acceptable client-error response.
        """
        resp = a2a_client.post("/api/v1/a2a", json=["not", "a", "dict"], headers=_A2A_ADMIN_HEADERS)
        assert resp.status_code in (400, 422), f"Expected 400 or 422 for non-dict body, got {resp.status_code}"

    def test_missing_jsonrpc_field_returns_400(self, a2a_client) -> None:
        """A dict body missing 'jsonrpc' must return 400."""
        resp = a2a_client.post(
            "/api/v1/a2a",
            json={"method": "a2a.getAgentCard", "id": 1, "params": {}},
            headers=_A2A_ADMIN_HEADERS,
        )
        assert resp.status_code == 400

    def test_missing_method_field_returns_400(self, a2a_client) -> None:
        """A dict body missing 'method' must return 400."""
        resp = a2a_client.post(
            "/api/v1/a2a",
            json={"jsonrpc": "2.0", "id": 1, "params": {}},
            headers=_A2A_ADMIN_HEADERS,
        )
        assert resp.status_code == 400

    def test_missing_id_field_returns_400(self, a2a_client) -> None:
        """A dict body missing 'id' must return 400."""
        resp = a2a_client.post(
            "/api/v1/a2a",
            json={"jsonrpc": "2.0", "method": "a2a.getAgentCard", "params": {}},
            headers=_A2A_ADMIN_HEADERS,
        )
        assert resp.status_code == 400

    def test_get_agent_card_returns_200(self, a2a_client) -> None:
        """a2a.getAgentCard must return HTTP 200, not 201."""
        resp = a2a_client.post(
            "/api/v1/a2a",
            json={"jsonrpc": "2.0", "method": "a2a.getAgentCard", "id": 1, "params": {}},
            headers=_A2A_ADMIN_HEADERS,
        )
        assert resp.status_code == 200, f"a2a.getAgentCard returned {resp.status_code}, expected 200"

    def test_task_status_returns_200(self, a2a_client) -> None:
        """a2a.taskStatus must return HTTP 200 (even when the task is not found — 404 logic is in JSON-RPC layer)."""
        resp = a2a_client.post(
            "/api/v1/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "a2a.taskStatus",
                "id": 2,
                "params": {"taskId": "nonexistent-task-id"},
            },
            headers=_A2A_ADMIN_HEADERS,
        )
        assert resp.status_code == 200, f"a2a.taskStatus returned {resp.status_code}, expected 200"

    def test_task_send_returns_201(self, a2a_client) -> None:
        """a2a.taskSend must return HTTP 201 Created."""
        with patch("vetinari.a2a.executor.get_two_layer_orchestrator", return_value=None):
            resp = a2a_client.post(
                "/api/v1/a2a",
                json={
                    "jsonrpc": "2.0",
                    "method": "a2a.taskSend",
                    "id": 3,
                    "params": {"taskType": "build", "inputData": {"goal": "hello world"}},
                },
                headers=_A2A_ADMIN_HEADERS,
            )
        assert resp.status_code == 201, f"a2a.taskSend returned {resp.status_code}, expected 201"

    def test_400_response_body_is_valid_jsonrpc_error(self, a2a_client) -> None:
        """400 responses must carry a valid JSON-RPC error envelope in the body."""
        resp = a2a_client.post("/api/v1/a2a", json={"only_garbage": True}, headers=_A2A_ADMIN_HEADERS)
        assert resp.status_code == 400
        body = resp.json()
        assert body["jsonrpc"] == "2.0"
        assert "error" in body
        assert body["error"]["code"] == -32600


class TestA2ASharedTransportCrossRequest:
    """Verify that taskStatus succeeds across separate handler calls (shared transport)."""

    def test_task_status_visible_after_task_send(self) -> None:
        """A task submitted via taskSend must be queryable via taskStatus using the same transport."""
        import vetinari.web.litestar_app as mod

        old = mod._a2a_transport
        old_admin_token = os.environ.get("VETINARI_ADMIN_TOKEN")
        os.environ["VETINARI_ADMIN_TOKEN"] = _A2A_ADMIN_TOKEN
        mod._a2a_transport = None
        try:
            handlers = _create_a2a_handlers()
            app = Litestar(route_handlers=handlers)
            task_id = "cross-request-task-001"
            with TestClient(app=app) as client:
                with patch("vetinari.a2a.executor.get_two_layer_orchestrator", return_value=None):
                    send_resp = client.post(
                        "/api/v1/a2a",
                        json={
                            "jsonrpc": "2.0",
                            "method": "a2a.taskSend",
                        "id": 1,
                        "params": {"taskType": "plan", "taskId": task_id, "inputData": {}},
                    },
                    headers=_A2A_ADMIN_HEADERS,
                )
                assert send_resp.status_code == 201

                # Now query on a logically separate call — must find the task in the shared store
                status_resp = client.post(
                    "/api/v1/a2a",
                    json={
                        "jsonrpc": "2.0",
                        "method": "a2a.taskStatus",
                        "id": 2,
                        "params": {"taskId": task_id},
                    },
                    headers=_A2A_ADMIN_HEADERS,
                )
                assert status_resp.status_code == 200
                body = status_resp.json()
                assert "result" in body, f"Expected result in body, got: {body}"
                assert body["result"]["taskId"] == task_id
        finally:
            if old_admin_token is None:
                os.environ.pop("VETINARI_ADMIN_TOKEN", None)
            else:
                os.environ["VETINARI_ADMIN_TOKEN"] = old_admin_token
            mod._a2a_transport = old


class TestA2ARawEndpointHTTPStatus:
    """Verify HTTP status codes for the /api/v1/a2a/raw endpoint."""

    @pytest.fixture
    def raw_client(self):
        """TestClient for a minimal Litestar app containing the A2A raw handler."""
        import vetinari.web.litestar_app as mod

        old = mod._a2a_transport
        mod._a2a_transport = None
        try:
            with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _A2A_ADMIN_TOKEN}):
                handlers = _create_a2a_handlers()
                app = Litestar(route_handlers=handlers)
                with TestClient(app=app) as client:
                    yield client
        finally:
            mod._a2a_transport = old

    def test_invalid_json_string_returns_400(self, raw_client) -> None:
        """A body that is not valid JSON must return 400 from /api/v1/a2a/raw."""
        resp = raw_client.post(
            "/api/v1/a2a/raw",
            content=b"not valid json at all {{{{",
            headers=_A2A_ADMIN_HEADERS,
        )
        assert resp.status_code == 400

    def test_missing_required_fields_returns_400(self, raw_client) -> None:
        """A JSON string missing required JSON-RPC fields must return 400."""
        import json

        resp = raw_client.post(
            "/api/v1/a2a/raw",
            content=json.dumps({"only": "garbage"}).encode(),
            headers=_A2A_ADMIN_HEADERS,
        )
        assert resp.status_code == 400

    def test_task_send_via_raw_returns_201(self, raw_client) -> None:
        """a2a.taskSend via the raw endpoint must return 201."""
        import json

        payload = json.dumps({
            "jsonrpc": "2.0",
            "method": "a2a.taskSend",
            "id": 1,
            "params": {"taskType": "build", "inputData": {"goal": "test"}},
        })
        with patch("vetinari.a2a.executor.get_two_layer_orchestrator", return_value=None):
            resp = raw_client.post("/api/v1/a2a/raw", content=payload.encode(), headers=_A2A_ADMIN_HEADERS)
        assert resp.status_code == 201

    def test_get_agent_card_via_raw_returns_200(self, raw_client) -> None:
        """a2a.getAgentCard via the raw endpoint must return 200."""
        import json

        payload = json.dumps({
            "jsonrpc": "2.0",
            "method": "a2a.getAgentCard",
            "id": 2,
            "params": {},
        })
        resp = raw_client.post("/api/v1/a2a/raw", content=payload.encode(), headers=_A2A_ADMIN_HEADERS)
        assert resp.status_code == 200


class TestA2AAdminGuardEnforcement:
    """Prove admin_guard is mounted on all A2A routes.

    Monkeypatches VETINARI_ADMIN_TOKEN to force token-path checking in the
    guard, then hits each A2A endpoint without credentials and asserts 401.

    Without the env var set, Litestar's TestClient presents as 127.0.0.1 and
    the guard allows it via the localhost IP fallback.  Setting the token env
    var switches the guard to HMAC comparison — requests with no header then
    correctly receive 401.
    """

    A2A_ROUTES = [
        ("GET", "/api/v1/a2a/cards"),
        ("POST", "/api/v1/a2a"),
        ("POST", "/api/v1/a2a/raw"),
    ]

    @pytest.fixture
    def unauthenticated_a2a_app(self, monkeypatch):
        """Litestar app with real guard and token auth enabled via env var.

        Args:
            monkeypatch: pytest monkeypatch fixture.

        Returns:
            Litestar app instance with A2A handlers and token auth enforced.
        """
        monkeypatch.setenv("VETINARI_ADMIN_TOKEN", "test-admin-secret-33g")
        handlers = _create_a2a_handlers()
        return Litestar(route_handlers=handlers)

    @pytest.mark.parametrize("method,path", A2A_ROUTES)
    def test_a2a_route_requires_auth(self, unauthenticated_a2a_app, method, path) -> None:
        """Each A2A route returns 401 when no admin token header is provided.

        Args:
            unauthenticated_a2a_app: App fixture with token auth enforced.
            method: HTTP method string ("GET" or "POST").
            path: Route path to test.
        """
        with TestClient(app=unauthenticated_a2a_app) as client:
            if method == "GET":
                response = client.get(path)
            else:
                response = client.post(path, json={})
        assert response.status_code == 401, (
            f"Expected 401 for {method} {path} without auth, got {response.status_code}"
        )
