"""Auth-enforcement and input-validation tests for litestar_plans_api.py.

Proves that every mutating route (POST/PUT/DELETE) on the plans API:
  - Rejects requests without X-Admin-Token with HTTP 401 or 403.
  - Accepts requests with the correct token (handler may return 400/404/422
    from missing data, but the guard must not fire).
  - Rejects traversal-style plan_id values with HTTP 400.

Read-only GETs (GET /api/v1/plans, GET /api/v1/plans/{id},
GET /api/v1/plans/{id}/status) are intentionally unguarded and are not
tested here.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from unittest.mock import patch

import pytest
from litestar import Litestar
from litestar.testing import TestClient

from vetinari.web.litestar_plans_api import create_plans_api_handlers

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Admin bypass token  -  set as env var so admin_guard passes.
_ADMIN_TOKEN = "plans-test-admin-token"

# Auth header sent with admin-level requests.
_ADMIN_HEADERS = {"X-Admin-Token": _ADMIN_TOKEN}

# A safe plan_id that passes validate_path_param.
_VALID_PLAN = "valid-plan-id"

# Traversal-style plan_id values that must be rejected with 400.
# URL-encoded so they survive router path normalisation and reach the handler.
# %2E%2E       -> ".."
# %2E%2E%5C    -> "..\\"
# my%2Eplan    -> "my.plan"  (dot outside [A-Za-z0-9_-])
_BAD_PLAN_IDS = [
    "%2E%2E",  # decodes to ".."
    "%2E%2E%5Coutside",  # decodes to "..\outside"
    "my%2Eplan",  # decodes to "my.plan"
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """TestClient for a minimal Litestar app with only the plans handlers.

    The VETINARI_ADMIN_TOKEN env var is patched so admin_guard considers the
    test token valid.  The fixture is module-scoped to avoid repeated app
    construction overhead.
    """
    import vetinari.web.litestar_plans_api  # ensure full dotted-name is in sys.modules

    handlers = create_plans_api_handlers()
    if not handlers:
        pytest.skip("Litestar not installed  -  skipping plans API tests")

    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        app = Litestar(route_handlers=handlers, debug=True)
        with TestClient(app=app) as c:
            yield c


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _assert_traversal_rejected(resp: object) -> None:
    """Assert HTTP 400 with an 'Invalid' message for traversal identifiers.

    Args:
        resp: An httpx Response object from TestClient.
    """
    assert resp.status_code == 400, (
        f"Expected HTTP 400 for traversal plan_id, got {resp.status_code}. Body: {resp.text!r}"
    )
    body = resp.json()
    error_text = body.get("error", "") or body.get("detail", "") or resp.text
    assert "Invalid" in error_text, f"Expected 'Invalid' in error message, got: {error_text!r}"


# ---------------------------------------------------------------------------
# Auth enforcement  -  POST /api/v1/plan (PlanModeEngine route)
# ---------------------------------------------------------------------------


class TestCreatePlanAuth:
    """Admin guard enforcement on POST /api/v1/plan."""

    def test_rejects_unauthenticated(self, client: TestClient) -> None:
        """POST /api/v1/plan without token must return 401 or 403."""
        resp = client.post("/api/v1/plan", json={"goal": "test"})
        assert resp.status_code in {401, 403}, (
            f"Expected 401/403 for unauthenticated POST /api/v1/plan, "
            f"got {resp.status_code}. Body: {resp.text!r}"
        )

    def test_passes_with_auth_token(self, client: TestClient) -> None:
        """POST /api/v1/plan with admin token must not return 401/403.

        No real planning engine is wired, so 400/422/500 from missing models
        is acceptable  -  the guard must not fire.
        """
        resp = client.post("/api/v1/plan", json={"goal": "test goal"}, headers=_ADMIN_HEADERS)
        assert resp.status_code not in {401, 403}, (
            f"Admin token was rejected for POST /api/v1/plan  -  guard misconfigured. "
            f"Status: {resp.status_code}. Body: {resp.text!r}"
        )


# ---------------------------------------------------------------------------
# Auth enforcement  -  POST /api/v1/plans (plan manager create)
# ---------------------------------------------------------------------------


class TestPlanCreateAuth:
    """Admin guard enforcement on POST /api/v1/plans."""

    def test_rejects_unauthenticated(self, client: TestClient) -> None:
        """POST /api/v1/plans without token must return 401 or 403."""
        resp = client.post("/api/v1/plans", json={"title": "t", "prompt": "p"})
        assert resp.status_code in {401, 403}, (
            f"Expected 401/403 for unauthenticated POST /api/v1/plans, "
            f"got {resp.status_code}. Body: {resp.text!r}"
        )

    def test_passes_with_auth_token(self, client: TestClient) -> None:
        """POST /api/v1/plans with admin token must not return 401/403."""
        resp = client.post(
            "/api/v1/plans",
            json={"title": "Test Plan", "prompt": "desc"},
            headers=_ADMIN_HEADERS,
        )
        assert resp.status_code not in {401, 403}, (
            f"Admin token was rejected for POST /api/v1/plans  -  guard misconfigured. "
            f"Status: {resp.status_code}. Body: {resp.text!r}"
        )


# ---------------------------------------------------------------------------
# Auth enforcement  -  PUT /api/v1/plans/{plan_id}
# ---------------------------------------------------------------------------


class TestPlanUpdateAuth:
    """Admin guard enforcement on PUT /api/v1/plans/{plan_id}."""

    def test_rejects_unauthenticated(self, client: TestClient) -> None:
        """PUT /api/v1/plans/{id} without token must return 401 or 403."""
        resp = client.put(f"/api/v1/plans/{_VALID_PLAN}", json={"title": "new"})
        assert resp.status_code in {401, 403}, (
            f"Expected 401/403 for unauthenticated PUT /api/v1/plans/{{id}}, "
            f"got {resp.status_code}. Body: {resp.text!r}"
        )

    def test_passes_with_auth_token(self, client: TestClient) -> None:
        """PUT /api/v1/plans/{id} with admin token must not return 401/403."""
        resp = client.put(
            f"/api/v1/plans/{_VALID_PLAN}",
            json={"title": "updated"},
            headers=_ADMIN_HEADERS,
        )
        assert resp.status_code not in {401, 403}, (
            f"Admin token was rejected for PUT /api/v1/plans/{{id}}  -  guard misconfigured. "
            f"Status: {resp.status_code}. Body: {resp.text!r}"
        )

    @pytest.mark.parametrize("bad_id", _BAD_PLAN_IDS)
    def test_rejects_traversal_plan_id(self, client: TestClient, bad_id: str) -> None:
        """PUT /api/v1/plans/<traversal> must return 400 Invalid."""
        resp = client.put(f"/api/v1/plans/{bad_id}", json={"title": "x"}, headers=_ADMIN_HEADERS)
        _assert_traversal_rejected(resp)


# ---------------------------------------------------------------------------
# Auth enforcement  -  DELETE /api/v1/plans/{plan_id}
# ---------------------------------------------------------------------------


class TestPlanDeleteAuth:
    """Admin guard enforcement on DELETE /api/v1/plans/{plan_id}."""

    def test_rejects_unauthenticated(self, client: TestClient) -> None:
        """DELETE /api/v1/plans/{id} without token must return 401 or 403."""
        resp = client.delete(f"/api/v1/plans/{_VALID_PLAN}")
        assert resp.status_code in {401, 403}, (
            f"Expected 401/403 for unauthenticated DELETE /api/v1/plans/{{id}}, "
            f"got {resp.status_code}. Body: {resp.text!r}"
        )

    def test_passes_with_auth_token(self, client: TestClient) -> None:
        """DELETE /api/v1/plans/{id} with admin token must not return 401/403."""
        resp = client.delete(f"/api/v1/plans/{_VALID_PLAN}", headers=_ADMIN_HEADERS)
        assert resp.status_code not in {401, 403}, (
            f"Admin token was rejected for DELETE /api/v1/plans/{{id}}  -  guard misconfigured. "
            f"Status: {resp.status_code}. Body: {resp.text!r}"
        )

    @pytest.mark.parametrize("bad_id", _BAD_PLAN_IDS)
    def test_rejects_traversal_plan_id(self, client: TestClient, bad_id: str) -> None:
        """DELETE /api/v1/plans/<traversal> must return 400 Invalid."""
        resp = client.delete(f"/api/v1/plans/{bad_id}", headers=_ADMIN_HEADERS)
        _assert_traversal_rejected(resp)


# ---------------------------------------------------------------------------
# Auth enforcement  -  lifecycle POST endpoints
# ---------------------------------------------------------------------------


class TestPlanLifecycleAuth:
    """Admin guard enforcement on start/pause/resume/cancel lifecycle routes."""

    @pytest.mark.parametrize("action", ["start", "pause", "resume", "cancel"])
    def test_rejects_unauthenticated(self, client: TestClient, action: str) -> None:
        """POST /api/v1/plans/{id}/{action} without token must return 401 or 403."""
        resp = client.post(f"/api/v1/plans/{_VALID_PLAN}/{action}")
        assert resp.status_code in {401, 403}, (
            f"Expected 401/403 for unauthenticated POST /api/v1/plans/{{id}}/{action}, "
            f"got {resp.status_code}. Body: {resp.text!r}"
        )

    @pytest.mark.parametrize("action", ["start", "pause", "resume", "cancel"])
    def test_passes_with_auth_token(self, client: TestClient, action: str) -> None:
        """POST /api/v1/plans/{id}/{action} with admin token must not return 401/403."""
        resp = client.post(
            f"/api/v1/plans/{_VALID_PLAN}/{action}",
            headers=_ADMIN_HEADERS,
        )
        assert resp.status_code not in {401, 403}, (
            f"Admin token was rejected for POST /api/v1/plans/{{id}}/{action}  -  guard misconfigured. "
            f"Status: {resp.status_code}. Body: {resp.text!r}"
        )
