"""Direct request-level confinement tests for versioned native goal-verification route.

Proves that api_verify_goal never serves a 2xx response for traversal-style
project_id values, and does not load parent-root project config through
path-like identifiers (Session 27C).

Confinement operates at two layers:

1. Litestar router — URL-encoded or slash-containing segments (``../outside``,
   ``..%2Foutside``, ``./``, ``..``) are resolved or normalised by the HTTP
   router before the handler is invoked.  The router returns 404 for these
   identifiers because no route matches the resolved URL.

2. Application layer (``validate_path_param``) — backslash-containing segments
   (``..\\outside``, ``.\\``) are delivered to the handler as-is by the router
   and are rejected there with 400 "Invalid project ID".

Both outcomes (400 and 404) are blocking responses.  Neither allows the handler
to load a project YAML at a traversal path.  The tests assert the security
contract: traversal IDs MUST NOT produce a 2xx response.
"""

from __future__ import annotations

import pytest
from litestar import Litestar
from litestar.testing import TestClient

from vetinari.web.litestar_subtasks_api import create_subtasks_api_handlers

# Bad identifiers that must be rejected at one of the two confinement layers.
# Slash/dot IDs are resolved by the Litestar router (404).
# Backslash IDs reach the handler and are rejected by validate_path_param (400).
TRAVERSAL_IDS = ["..", "..\\outside", "../outside", "..%2Foutside", ".\\", "./"]

# Subset that the router delivers to the handler unchanged — these must produce 400
# with "Invalid project ID" from validate_path_param.
HANDLER_REJECTED_IDS = ["..\\outside", ".\\"]

# Subset that are resolved/normalised by the router before the handler is called —
# these produce 404 because no route matches the resolved URL.
ROUTER_REJECTED_IDS = ["..", "../outside", "..%2Foutside", "./"]


@pytest.fixture
def client():
    """Yield a TestClient with only the subtasks API handlers mounted."""
    handlers = create_subtasks_api_handlers()
    app = Litestar(route_handlers=handlers)
    with TestClient(app=app) as tc:
        yield tc


class TestVerifyGoalRouteConfinement:
    """Confinement tests ensuring project_id path traversal is blocked at the boundary.

    Covers the ``POST /api/v1/project/{project_id}/verify-goal`` route only.
    All traversal identifiers must produce a non-2xx response.  The project
    YAML fallback path must never be reached for any traversal identifier.
    """

    @pytest.mark.parametrize("bad_id", TRAVERSAL_IDS)
    def test_verify_goal_rejects_traversal_project_id(self, client: TestClient, bad_id: str) -> None:
        """POST with a traversal project_id and a valid body is never served 2xx.

        The blocking response is either 400 (application-layer rejection by
        ``validate_path_param``) or 404 (router-layer rejection when the
        traversal segment resolves to a non-existent route).  Either is
        acceptable — neither allows the project YAML to be loaded.

        Args:
            client: Litestar TestClient with subtasks handlers mounted.
            bad_id: A traversal-style project_id value.
        """
        response = client.post(
            f"/api/v1/project/{bad_id}/verify-goal",
            json={"goal": "test goal", "final_output": "output"},
        )
        assert response.status_code in {400, 404}, (
            f"Traversal project_id {bad_id!r} must be blocked (400 or 404), got {response.status_code}: {response.text}"
        )

    @pytest.mark.parametrize("bad_id", TRAVERSAL_IDS)
    def test_verify_goal_rejects_traversal_project_id_omitted_goal(self, client: TestClient, bad_id: str) -> None:
        """POST with traversal project_id and no goal body is still blocked.

        Validates that the confinement check (router or application layer) fires
        before the project.yaml fallback path is reached.  If the YAML fallback
        ran first for a traversal ID, the response would be 200 with goal data
        from an arbitrary path — the definitive test is that no 2xx is returned.

        Args:
            client: Litestar TestClient with subtasks handlers mounted.
            bad_id: A traversal-style project_id value.
        """
        response = client.post(
            f"/api/v1/project/{bad_id}/verify-goal",
            json={"final_output": "output"},
        )
        assert response.status_code in {400, 404}, (
            f"Traversal project_id {bad_id!r} (no goal) must be blocked, got {response.status_code}: {response.text}"
        )

    @pytest.mark.parametrize("bad_id", HANDLER_REJECTED_IDS)
    def test_verify_goal_application_layer_returns_invalid_message(self, client: TestClient, bad_id: str) -> None:
        """Backslash traversal IDs that reach the handler return 400 "Invalid project ID".

        These identifiers are not normalised by the Litestar router and arrive
        at ``api_verify_goal`` intact.  ``validate_path_param`` rejects them
        and ``litestar_error_response`` builds the 400 envelope.

        Args:
            client: Litestar TestClient with subtasks handlers mounted.
            bad_id: A backslash-containing traversal identifier.
        """
        response = client.post(
            f"/api/v1/project/{bad_id}/verify-goal",
            json={"goal": "test goal", "final_output": "output"},
        )
        assert response.status_code == 400, (
            f"Expected 400 for handler-layer traversal ID {bad_id!r}, got {response.status_code}: {response.text}"
        )
        assert "Invalid" in response.text, (
            f"Response body should contain 'Invalid' for ID {bad_id!r}, got: {response.text}"
        )

    @pytest.mark.parametrize("bad_id", ROUTER_REJECTED_IDS)
    def test_verify_goal_router_layer_returns_404(self, client: TestClient, bad_id: str) -> None:
        """Slash/dot traversal IDs are normalised by the router and return 404.

        Litestar resolves ``../``, ``./``, and percent-encoded slashes in the
        URL path before route matching.  No route matches the resolved URL so
        the response is 404 — the handler is never invoked and no YAML is read.

        Args:
            client: Litestar TestClient with subtasks handlers mounted.
            bad_id: A slash-or-dot traversal identifier.
        """
        response = client.post(
            f"/api/v1/project/{bad_id}/verify-goal",
            json={"goal": "test goal", "final_output": "output"},
        )
        assert response.status_code == 404, (
            f"Expected 404 for router-normalised traversal ID {bad_id!r}, got {response.status_code}: {response.text}"
        )

    def test_verify_goal_accepts_valid_project_id(self, client: TestClient) -> None:
        """POST with a safe project_id passes path validation.

        The handler may return any non-400-with-"Invalid-project-ID" status
        depending on whether the goal verifier or its inference backend is
        available.  The critical assertion is that the request is NOT rejected
        by the confinement check.

        Args:
            client: Litestar TestClient with subtasks handlers mounted.
        """
        response = client.post(
            "/api/v1/project/my-project/verify-goal",
            json={"goal": "test goal", "final_output": "output"},
        )
        # Confinement must pass — downstream errors (verifier, inference) are acceptable.
        assert response.status_code != 400 or "Invalid project ID" not in response.text, (
            f"Valid project ID 'my-project' should not be rejected by confinement. "
            f"Got {response.status_code}: {response.text}"
        )
