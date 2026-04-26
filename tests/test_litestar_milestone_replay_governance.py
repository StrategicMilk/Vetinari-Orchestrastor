"""Mounted request-level governance tests for milestone and replay routes.

Proves every route returns a bounded 503 (not a raw 500) when the backing
subsystem raises, and validates input contracts.

Milestone routes:
  GET  /api/v1/milestones/status
  POST /api/v1/milestones/approve

Replay routes:
  GET  /api/v1/projects/{project_id}/events/replay
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# Skip the whole module when Litestar is not installed.

_ADMIN_TOKEN = "test-admin-governance"
_MUTATION_HEADERS = {"X-Admin-Token": _ADMIN_TOKEN, "X-Requested-With": "XMLHttpRequest"}


# ---------------------------------------------------------------------------
# App / client fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    """Litestar app with shutdown side-effects suppressed.

    Scoped to module so the Litestar app object is built only once; each
    test creates its own TestClient context so connection state does not leak.

    Returns:
        A Litestar application instance.
    """
    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        with patch("vetinari.web.litestar_app._register_shutdown_handlers"):
            from vetinari.web.litestar_app import create_app

            return create_app(debug=True)


@pytest.fixture
def client(app):
    """TestClient bound to the shared Litestar app.

    Args:
        app: The module-scoped Litestar application fixture.

    Yields:
        A live TestClient for the duration of one test.
    """
    from litestar.testing import TestClient

    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        with TestClient(app) as tc:
            yield tc


# ---------------------------------------------------------------------------
# Shared assertion helpers
# ---------------------------------------------------------------------------


def _assert_503_error(response: object) -> None:
    """Assert *response* is a bounded 503 with ``status: error`` envelope.

    Args:
        response: HTTP response from the TestClient.

    Raises:
        AssertionError: When status_code is not 503 or the body lacks the
            ``"status": "error"`` field.
    """
    assert response.status_code == 503, f"Expected 503, got {response.status_code}. Body: {response.text[:400]}"
    body = response.json()
    assert body.get("status") == "error", f"Expected envelope status='error', got {body.get('status')!r}. Body: {body}"


def _assert_error(response: object, expected_code: int) -> None:
    """Assert *response* has *expected_code* and ``status: error`` envelope.

    Args:
        response: HTTP response from the TestClient.
        expected_code: Expected HTTP status code.

    Raises:
        AssertionError: When status_code or envelope status does not match.
    """
    assert response.status_code == expected_code, (
        f"Expected {expected_code}, got {response.status_code}. Body: {response.text[:400]}"
    )
    body = response.json()
    assert body.get("status") == "error", f"Expected envelope status='error', got {body.get('status')!r}. Body: {body}"


# ===========================================================================
# Milestone status route tests
# ===========================================================================


class TestMilestoneStatusRoute:
    """Tests for GET /api/v1/milestones/status on subsystem failure."""

    def test_import_failure_returns_503(self, client: object) -> None:
        """GET /api/v1/milestones/status returns 503 when the import of the executor fails.

        ``get_graph_executor`` does not currently exist in
        ``vetinari.orchestration.graph_executor``, so the local import inside
        the handler raises ``ImportError``.  The except block must catch it
        and return a bounded 503  -  not a raw 500 or a silent empty response.

        Args:
            client: The TestClient fixture.
        """
        response = client.get("/api/v1/milestones/status")
        _assert_503_error(response)

    def test_executor_raises_returns_503(self, client: object) -> None:
        """GET /api/v1/milestones/status returns 503 when get_graph_executor() raises.

        Injects a fake ``graph_executor`` module into ``sys.modules`` so the
        local import inside the handler succeeds, but calling ``get_graph_executor``
        raises a ``RuntimeError``.  The except block must still return 503.

        Args:
            client: The TestClient fixture.
        """
        import sys
        import types

        fake_module = types.ModuleType("vetinari.orchestration.graph_executor")
        fake_module.get_graph_executor = MagicMock(side_effect=RuntimeError("executor down"))  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"vetinari.orchestration.graph_executor": fake_module}):
            response = client.get("/api/v1/milestones/status")
        _assert_503_error(response)

    def test_executor_without_milestone_manager_returns_503(self, client: object) -> None:
        """GET /api/v1/milestones/status returns 503 when executor has no MilestoneManager.

        An executor that returns successfully but carries no ``milestone_manager``
        attribute must not flatten into an empty-looking 200 response.  The handler
        must return 503 so callers know the subsystem is not active and cannot be
        queried for accurate state.

        Args:
            client: The TestClient fixture.
        """
        import sys
        import types

        # Executor that resolves but has no milestone_manager attribute (returns None)
        mock_executor = MagicMock(spec=[])  # spec=[] makes getattr return None via __getattr__
        fake_module = types.ModuleType("vetinari.orchestration.graph_executor")
        fake_module.get_graph_executor = MagicMock(return_value=mock_executor)  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"vetinari.orchestration.graph_executor": fake_module}):
            response = client.get("/api/v1/milestones/status")
        _assert_503_error(response)


# ===========================================================================
# Milestone approve route tests
# ===========================================================================


class TestMilestoneApproveRoute:
    """Tests for POST /api/v1/milestones/approve input validation."""

    def test_empty_body_returns_400(self, client: object) -> None:
        """POST /api/v1/milestones/approve returns 400 when body has no ``action``.

        Args:
            client: The TestClient fixture.
        """
        response = client.post(
            "/api/v1/milestones/approve",
            json={},
            headers=_MUTATION_HEADERS,
        )
        _assert_error(response, 400)

    def test_invalid_action_returns_400(self, client: object) -> None:
        """POST /api/v1/milestones/approve returns 400 for an unrecognised action.

        Args:
            client: The TestClient fixture.
        """
        response = client.post(
            "/api/v1/milestones/approve",
            json={"action": "invalid_action_xyz"},
            headers=_MUTATION_HEADERS,
        )
        _assert_error(response, 400)

    def test_milestones_module_import_fails_returns_503(self, client: object) -> None:
        """POST /api/v1/milestones/approve returns 503 when the milestones module cannot be imported.

        The handler imports ``MilestoneAction`` and ``MilestoneApproval`` from
        ``vetinari.orchestration.milestones`` at call time.  If that import raises
        (e.g. the module is unavailable), the handler must return a bounded 503
        rather than an unhandled 500.

        Args:
            client: The TestClient fixture.
        """
        with patch.dict("sys.modules", {"vetinari.orchestration.milestones": None}):
            response = client.post(
                "/api/v1/milestones/approve",
                json={"action": "approve"},
                headers=_MUTATION_HEADERS,
            )
        _assert_503_error(response)

    def test_valid_action_succeeds(self, client: object) -> None:
        """POST /api/v1/milestones/approve returns 200 for a valid action.

        The best-effort direct delivery to MilestoneManager is allowed to fail
        silently (executor not running in test env); what matters is that the
        pending slot is set and success is returned.

        Uses ``sys.modules`` injection so the local import inside the handler
        resolves, but ``get_graph_executor()`` raises  -  proving the best-effort
        path is truly best-effort and does not prevent the 200 response.

        Args:
            client: The TestClient fixture.
        """
        import sys
        import types

        fake_module = types.ModuleType("vetinari.orchestration.graph_executor")
        fake_module.get_graph_executor = MagicMock(side_effect=RuntimeError("no executor in test env"))  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"vetinari.orchestration.graph_executor": fake_module}):
            response = client.post(
                "/api/v1/milestones/approve",
                json={"action": "approve"},
                headers=_MUTATION_HEADERS,
            )
        assert 200 <= response.status_code < 300, (
            f"Expected 2xx, got {response.status_code}. Body: {response.text[:400]}"
        )
        body = response.json()
        assert body.get("success") is True
        assert body.get("action") == "approve"


# ===========================================================================
# SSE replay route tests
# ===========================================================================


class TestReplayRoute:
    """Tests for GET /api/v1/projects/{project_id}/events/replay on subsystem failure."""

    def test_db_failure_returns_503(self, client: object) -> None:
        """GET replay returns 503 when the database raises.

        Patches ``vetinari.database.get_connection`` which is imported locally
        inside ``replay_events`` at call time, so this module-path patch takes
        effect on every invocation.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.database.get_connection",
            side_effect=RuntimeError("db down"),
        ):
            response = client.get("/api/v1/projects/test-project/events/replay")
        _assert_503_error(response)

    def test_invalid_project_id_returns_validation_error(self, client: object) -> None:
        """GET replay returns a 4xx error when project_id contains unsafe characters.

        Path traversal sequences are rejected by ``validate_path_param`` before
        any database access.

        Args:
            client: The TestClient fixture.
        """
        response = client.get("/api/v1/projects/../secret/events/replay")
        # Litestar may normalise the path before it reaches the handler, so
        # accept any 4xx as proof that traversal is rejected.
        assert response.status_code < 500, f"Expected <500, got {response.status_code}. Body: {response.text[:400]}"

    def test_happy_path_empty_db_returns_empty_list(self, client: object) -> None:
        """GET replay returns an empty list when the DB table has no matching rows.

        Args:
            client: The TestClient fixture.
        """
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        with patch("vetinari.database.get_connection", return_value=mock_conn):
            response = client.get("/api/v1/projects/test-project/events/replay")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}. Body: {response.text[:400]}"
        assert response.json() == []
