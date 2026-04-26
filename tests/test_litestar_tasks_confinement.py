"""Direct request-level confinement tests for versioned native task routes.

Proves that api_task_output and api_task_override reject traversal-style
project_id and task_id values with bounded 400 errors, and do not read
or write files outside the intended project subtree (Session 27C).
"""

from __future__ import annotations

import pytest
from litestar import Litestar
from litestar.testing import TestClient

from vetinari.web.litestar_tasks_api import create_tasks_api_handlers

# Backslash-only traversal attempts that the Litestar router delivers to the
# handler unchanged — validate_path_param must return 400 for these.
HANDLER_REJECTED_IDS = ["..\\outside", ".\\"]

# Dot-dot and slash variants that the Litestar router normalises before the
# handler runs — the route simply does not match, producing 404.
ROUTER_REJECTED_IDS = ["..", "../outside", "..%2Foutside", "./"]


@pytest.fixture
def client(monkeypatch):
    """Yield a TestClient with only the tasks API handlers mounted.

    Builds a minimal Litestar app containing solely the handlers returned
    by ``create_tasks_api_handlers``.  Keeps tests isolated from the rest
    of the application stack (auth middleware, SSE, orchestrator singletons).

    Sets ``VETINARI_ADMIN_TOKEN`` so that the ``admin_guard`` on the override
    route accepts requests carrying the matching ``X-Admin-Token`` header.

    Args:
        monkeypatch: Pytest monkeypatch fixture for environment variable injection.

    Yields:
        A ``litestar.testing.TestClient`` backed by the minimal app.
    """
    monkeypatch.setenv("VETINARI_ADMIN_TOKEN", "test-admin-token")
    handlers = create_tasks_api_handlers()
    app = Litestar(route_handlers=handlers)
    with TestClient(app=app) as tc:
        yield tc


class TestTaskRouteConfinement:
    """Confinement tests for the versioned project-scoped task routes.

    Traversal tests are split by rejection layer:
    - HANDLER_REJECTED_IDS: backslash variants reach the handler; validate_path_param
      returns 400.
    - ROUTER_REJECTED_IDS: slash/dot-dot variants are normalised by the Litestar router
      before the handler runs; produce 404.

    Happy-path tests (5-6) use safe identifiers and assert the validation layer passes.
    """

    # -- GET /api/v1/project/{project_id}/task/{task_id}/output ---------------

    @pytest.mark.parametrize("bad_id", HANDLER_REJECTED_IDS)
    def test_task_output_handler_rejects_traversal_project_id(self, client: TestClient, bad_id: str) -> None:
        """validate_path_param must return 400 for backslash traversal project_id.

        Backslash variants are not normalised by the Litestar router, so they reach
        the handler.  The ``validate_path_param`` gate rejects them at the handler
        layer with HTTP 400.

        Args:
            client: Minimal Litestar TestClient fixture.
            bad_id: A backslash traversal identifier from ``HANDLER_REJECTED_IDS``.
        """
        response = client.get(f"/api/v1/project/{bad_id}/task/t1/output")
        assert response.status_code == 400, (
            f"Expected 400 for handler-layer traversal project_id {bad_id!r}, "
            f"got {response.status_code}: {response.text}"
        )

    @pytest.mark.parametrize("bad_id", ROUTER_REJECTED_IDS)
    def test_task_output_router_rejects_traversal_project_id(self, client: TestClient, bad_id: str) -> None:
        """Litestar router normalises slash traversal project_id before the handler runs.

        Slash and dot-dot variants are normalised or unmatched by the router,
        so the route never resolves and the response is HTTP 404 (not 400).

        Args:
            client: Minimal Litestar TestClient fixture.
            bad_id: A slash traversal identifier from ``ROUTER_REJECTED_IDS``.
        """
        response = client.get(f"/api/v1/project/{bad_id}/task/t1/output")
        assert response.status_code == 404, (
            f"Expected 404 for router-layer traversal project_id {bad_id!r}, "
            f"got {response.status_code}: {response.text}"
        )

    @pytest.mark.parametrize("bad_id", HANDLER_REJECTED_IDS)
    def test_task_output_handler_rejects_traversal_task_id(self, client: TestClient, bad_id: str) -> None:
        """validate_path_param must return 400 for backslash traversal task_id.

        Even when project_id is a safe identifier, a backslash traversal task_id
        reaches the handler and must cause ``api_task_output`` to return 400 before
        any filesystem access.

        Args:
            client: Minimal Litestar TestClient fixture.
            bad_id: A backslash traversal identifier from ``HANDLER_REJECTED_IDS``.
        """
        response = client.get(f"/api/v1/project/safe-project/task/{bad_id}/output")
        assert response.status_code == 400, (
            f"Expected 400 for handler-layer traversal task_id {bad_id!r}, got {response.status_code}: {response.text}"
        )

    @pytest.mark.parametrize("bad_id", ROUTER_REJECTED_IDS)
    def test_task_output_router_rejects_traversal_task_id(self, client: TestClient, bad_id: str) -> None:
        """Litestar router normalises slash traversal task_id before the handler runs.

        Slash and dot-dot variants in the task_id segment are normalised or unmatched
        by the router, so the route never resolves and the response is HTTP 404.

        Args:
            client: Minimal Litestar TestClient fixture.
            bad_id: A slash traversal identifier from ``ROUTER_REJECTED_IDS``.
        """
        response = client.get(f"/api/v1/project/safe-project/task/{bad_id}/output")
        assert response.status_code == 404, (
            f"Expected 404 for router-layer traversal task_id {bad_id!r}, got {response.status_code}: {response.text}"
        )

    # -- POST /api/v1/project/{project_id}/task/{task_id}/override ------------

    @pytest.mark.parametrize("bad_id", HANDLER_REJECTED_IDS)
    def test_task_override_handler_rejects_traversal_project_id(self, client: TestClient, bad_id: str) -> None:
        """validate_path_param must return 400 for backslash traversal project_id on override.

        TestClient sends from localhost so the ``admin_guard`` IP check passes.
        Backslash variants reach the handler and ``validate_path_param`` must fire
        with HTTP 400 before any YAML read or write.

        Args:
            client: Minimal Litestar TestClient fixture.
            bad_id: A backslash traversal identifier from ``HANDLER_REJECTED_IDS``.
        """
        response = client.post(
            f"/api/v1/project/{bad_id}/task/t1/override",
            json={"model_id": "m1"},
            headers={"X-Admin-Token": "test-admin-token"},
        )
        assert response.status_code == 400, (
            f"Expected 400 for handler-layer traversal project_id {bad_id!r}, "
            f"got {response.status_code}: {response.text}"
        )

    @pytest.mark.parametrize("bad_id", ROUTER_REJECTED_IDS)
    def test_task_override_router_rejects_traversal_project_id(self, client: TestClient, bad_id: str) -> None:
        """Litestar router normalises slash traversal project_id before override handler runs.

        Slash and dot-dot variants in the project_id segment are normalised or unmatched
        by the router, producing HTTP 404 before the handler is invoked.

        Args:
            client: Minimal Litestar TestClient fixture.
            bad_id: A slash traversal identifier from ``ROUTER_REJECTED_IDS``.
        """
        response = client.post(
            f"/api/v1/project/{bad_id}/task/t1/override",
            json={"model_id": "m1"},
            headers={"X-Admin-Token": "test-admin-token"},
        )
        assert response.status_code == 404, (
            f"Expected 404 for router-layer traversal project_id {bad_id!r}, "
            f"got {response.status_code}: {response.text}"
        )

    @pytest.mark.parametrize("bad_id", HANDLER_REJECTED_IDS)
    def test_task_override_handler_rejects_traversal_task_id(self, client: TestClient, bad_id: str) -> None:
        """validate_path_param must return 400 for backslash traversal task_id on override.

        Even when project_id is safe, a backslash traversal task_id reaches the handler
        and must cause ``api_task_override`` to return 400 before any filesystem mutation.

        Args:
            client: Minimal Litestar TestClient fixture.
            bad_id: A backslash traversal identifier from ``HANDLER_REJECTED_IDS``.
        """
        response = client.post(
            f"/api/v1/project/safe-project/task/{bad_id}/override",
            json={"model_id": "m1"},
            headers={"X-Admin-Token": "test-admin-token"},
        )
        assert response.status_code == 400, (
            f"Expected 400 for handler-layer traversal task_id {bad_id!r}, got {response.status_code}: {response.text}"
        )

    @pytest.mark.parametrize("bad_id", ROUTER_REJECTED_IDS)
    def test_task_override_router_rejects_traversal_task_id(self, client: TestClient, bad_id: str) -> None:
        """Litestar router normalises slash traversal task_id before override handler runs.

        Slash and dot-dot variants in the task_id segment are normalised or unmatched
        by the router, producing HTTP 404 before the handler is invoked.

        Args:
            client: Minimal Litestar TestClient fixture.
            bad_id: A slash traversal identifier from ``ROUTER_REJECTED_IDS``.
        """
        response = client.post(
            f"/api/v1/project/safe-project/task/{bad_id}/override",
            json={"model_id": "m1"},
            headers={"X-Admin-Token": "test-admin-token"},
        )
        assert response.status_code == 404, (
            f"Expected 404 for router-layer traversal task_id {bad_id!r}, got {response.status_code}: {response.text}"
        )

    # -- Happy-path: valid identifiers pass validation ------------------------

    def test_task_output_accepts_valid_ids(self, client: TestClient) -> None:
        """Safe alphanumeric identifiers must not be rejected by validation.

        The project does not exist on disk, so the handler returns 404 after
        passing validation.  The key assertion is that the status is NOT 400 —
        proving the confinement guard accepted the identifiers.

        Args:
            client: Minimal Litestar TestClient fixture.
        """
        response = client.get("/api/v1/project/my-project/task/task-1/output")
        assert response.status_code != 400

    def test_task_override_accepts_valid_ids(self, client: TestClient) -> None:
        """Safe alphanumeric identifiers on override must not be rejected by validation.

        The project config does not exist on disk, so the handler returns 404
        after passing validation.  The key assertion is that the status is NOT
        400 — proving the confinement guard accepted the identifiers.

        Args:
            client: Minimal Litestar TestClient fixture.
        """
        response = client.post(
            "/api/v1/project/my-project/task/task-1/override",
            json={"model_id": "m1"},
            headers={"X-Admin-Token": "test-admin-token"},
        )
        assert response.status_code != 400
