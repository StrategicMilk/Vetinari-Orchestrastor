"""Direct request-level confinement tests for project route traversal rejection.

Proves that every owned route in litestar_projects_api.py rejects traversal-style
project_id and task_id values ('..', '../outside', '..\\\\outside') with HTTP 400
instead of operating on parent-root files.

Session 27B ownership artifact  -  see docs/audit/session-27b-route-owners.md.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from unittest.mock import patch

import pytest
from litestar import Litestar
from litestar.testing import TestClient

from vetinari.web.litestar_projects_api import create_projects_api_handlers

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Admin bypass: sets VETINARI_ADMIN_TOKEN so admin_guard passes the token check
# instead of falling through to IP-based localhost validation.
_ADMIN_TOKEN = "confinement-test-token"

# Headers sent with every request: admin auth only.
# No CSRF header needed because we build a minimal Litestar app without
# the CSRFMiddleware (which is only added in create_app()).
_ADMIN_HEADERS = {"X-Admin-Token": _ADMIN_TOKEN}

# Traversal-style project_id and task_id values that should be rejected.
#
# These are URL-encoded so they survive Litestar's path normalization and
# reach the handler as decoded parameter values containing dangerous characters.
# Literal ".." in a URL path is normalised away by the HTTP router (404), so
# it never reaches validate_path_param.  The URL-encoded equivalents do reach
# the handler and exercise the real validation logic.
#
# %2E%2E         -> ".."           -  dot-dot (no slash, not normalised away)
# %2E%2E%5C      -> "..\\"         -  dot-dot backslash
# my%2Eproject   -> "my.project"   -  dot in name (outside alphanumeric set)
_BAD_PROJECT_IDS = [
    "%2E%2E",  # decodes to ".."
    "%2E%2E%5Coutside",  # decodes to "..\outside"
    "my%2Eproject",  # decodes to "my.project"  -  dot outside [A-Za-z0-9_-]
]
_BAD_TASK_IDS = [
    "%2E%2E",  # decodes to ".."
    "%2E%2E%5Coutside",  # decodes to "..\outside"
    "my%2Etask",  # decodes to "my.task"  -  dot outside [A-Za-z0-9_-]
]

# A safe identifier that passes validate_path_param.
_VALID_PROJECT = "valid-project"
_VALID_TASK = "valid-task"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """TestClient for a minimal Litestar app with only the project handlers.

    Admin auth is satisfied by setting VETINARI_ADMIN_TOKEN before the app
    is created so the guard is already configured when handlers are built.
    The environment variable remains set for the lifetime of the module-scoped
    fixture.

    The explicit ``import vetinari.web.litestar_projects_api`` call ensures
    the module is registered in ``sys.modules`` under its full dotted name
    before Litestar resolves handler type-hint forward references.  Without
    this, Litestar raises ``KeyError: 'vetinari.web.litestar_projects_api'``
    when the module is only partially imported via a from-import.
    """
    import vetinari.web.litestar_projects_api

    handlers = create_projects_api_handlers()
    if not handlers:
        pytest.skip("Litestar not installed  -  skipping confinement tests")

    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        app = Litestar(route_handlers=handlers, debug=True)
        with TestClient(app=app) as c:
            yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_traversal_rejected(resp: object) -> None:
    """Assert that a response indicates 400 with an 'Invalid' error message.

    Args:
        resp: An httpx Response object from TestClient.
    """
    assert resp.status_code == 400, (
        f"Expected HTTP 400 for traversal-style identifier, got {resp.status_code}. Body: {resp.text!r}"
    )
    body = resp.json()
    # The error envelope uses {"status": "error", "error": "<message>", ...}
    error_text = body.get("error", "") or body.get("detail", "") or resp.text
    assert "Invalid" in error_text, f"Expected 'Invalid' in error message, got: {error_text!r}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProjectRouteConfinement:
    """Traversal-style identifier rejection for mounted project routes.

    Each test sends a request carrying a traversal-style project_id or task_id
    and asserts HTTP 400 with an 'Invalid' message.  Tests use
    @pytest.mark.parametrize so every bad-identifier variant is covered.
    """

    @pytest.mark.parametrize("bad_id", _BAD_PROJECT_IDS)
    def test_rename_rejects_traversal_project_id(self, client: TestClient, bad_id: str) -> None:
        """POST /api/project/<traversal>/rename must return 400 Invalid."""
        resp = client.post(
            f"/api/project/{bad_id}/rename",
            json={"name": "test"},
            headers=_ADMIN_HEADERS,
        )
        _assert_traversal_rejected(resp)

    @pytest.mark.parametrize("bad_id", _BAD_PROJECT_IDS)
    def test_archive_rejects_traversal_project_id(self, client: TestClient, bad_id: str) -> None:
        """POST /api/project/<traversal>/archive must return 400 Invalid."""
        resp = client.post(
            f"/api/project/{bad_id}/archive",
            json={"archive": True},
            headers=_ADMIN_HEADERS,
        )
        _assert_traversal_rejected(resp)

    @pytest.mark.parametrize("bad_id", _BAD_PROJECT_IDS)
    def test_delete_project_rejects_traversal_project_id(self, client: TestClient, bad_id: str) -> None:
        """DELETE /api/project/<traversal> must return 400 Invalid."""
        resp = client.delete(
            f"/api/project/{bad_id}",
            headers=_ADMIN_HEADERS,
        )
        _assert_traversal_rejected(resp)

    @pytest.mark.parametrize("bad_id", _BAD_PROJECT_IDS)
    def test_message_rejects_traversal_project_id(self, client: TestClient, bad_id: str) -> None:
        """POST /api/project/<traversal>/message must return 400 Invalid."""
        resp = client.post(
            f"/api/project/{bad_id}/message",
            json={"message": "test"},
            headers=_ADMIN_HEADERS,
        )
        _assert_traversal_rejected(resp)

    @pytest.mark.parametrize("bad_id", _BAD_PROJECT_IDS)
    def test_verify_goal_rejects_traversal_project_id(self, client: TestClient, bad_id: str) -> None:
        """POST /api/project/<traversal>/verify-goal must return 400 Invalid."""
        resp = client.post(
            f"/api/project/{bad_id}/verify-goal",
            json={"goal": "test"},
            headers=_ADMIN_HEADERS,
        )
        _assert_traversal_rejected(resp)

    @pytest.mark.parametrize("bad_id", _BAD_PROJECT_IDS)
    def test_review_rejects_traversal_project_id(self, client: TestClient, bad_id: str) -> None:
        """GET /api/project/<traversal>/review must return 400 Invalid."""
        resp = client.get(
            f"/api/project/{bad_id}/review",
            headers=_ADMIN_HEADERS,
        )
        _assert_traversal_rejected(resp)

    @pytest.mark.parametrize("bad_id", _BAD_PROJECT_IDS)
    def test_approve_rejects_traversal_project_id(self, client: TestClient, bad_id: str) -> None:
        """POST /api/project/<traversal>/approve must return 400 Invalid."""
        resp = client.post(
            f"/api/project/{bad_id}/approve",
            json={},
            headers=_ADMIN_HEADERS,
        )
        _assert_traversal_rejected(resp)

    @pytest.mark.parametrize("bad_id", _BAD_PROJECT_IDS)
    def test_assemble_rejects_traversal_project_id(self, client: TestClient, bad_id: str) -> None:
        """POST /api/project/<traversal>/assemble must return 400 Invalid."""
        resp = client.post(
            f"/api/project/{bad_id}/assemble",
            json={},
            headers=_ADMIN_HEADERS,
        )
        _assert_traversal_rejected(resp)

    @pytest.mark.parametrize("bad_id", _BAD_PROJECT_IDS)
    def test_download_rejects_traversal_project_id(self, client: TestClient, bad_id: str) -> None:
        """GET /api/project/<traversal>/download must return 400 Invalid."""
        resp = client.get(
            f"/api/project/{bad_id}/download",
            headers=_ADMIN_HEADERS,
        )
        _assert_traversal_rejected(resp)

    @pytest.mark.parametrize("bad_id", _BAD_PROJECT_IDS)
    def test_model_search_rejects_traversal_project_id(self, client: TestClient, bad_id: str) -> None:
        """POST /api/project/<traversal>/model-search must return 400 Invalid."""
        resp = client.post(
            f"/api/project/{bad_id}/model-search",
            json={"task_description": "test"},
            headers=_ADMIN_HEADERS,
        )
        _assert_traversal_rejected(resp)

    # -- Routes with both project_id and task_id --------------------------------

    @pytest.mark.parametrize("bad_id", _BAD_PROJECT_IDS)
    def test_delete_task_rejects_traversal_project_id(self, client: TestClient, bad_id: str) -> None:
        """DELETE /api/project/<traversal>/task/<valid> must return 400 Invalid."""
        resp = client.delete(
            f"/api/project/{bad_id}/task/{_VALID_TASK}",
            headers=_ADMIN_HEADERS,
        )
        _assert_traversal_rejected(resp)

    @pytest.mark.parametrize("bad_id", _BAD_TASK_IDS)
    def test_delete_task_rejects_traversal_task_id(self, client: TestClient, bad_id: str) -> None:
        """DELETE /api/project/<valid>/task/<traversal> must return 400 Invalid."""
        resp = client.delete(
            f"/api/project/{_VALID_PROJECT}/task/{bad_id}",
            headers=_ADMIN_HEADERS,
        )
        _assert_traversal_rejected(resp)

    @pytest.mark.parametrize("bad_id", _BAD_PROJECT_IDS)
    def test_task_override_rejects_traversal_project_id(self, client: TestClient, bad_id: str) -> None:
        """POST /api/project/<traversal>/task/<valid>/override must return 400 Invalid."""
        resp = client.post(
            f"/api/project/{bad_id}/task/{_VALID_TASK}/override",
            json={"model_id": "test"},
            headers=_ADMIN_HEADERS,
        )
        _assert_traversal_rejected(resp)

    @pytest.mark.parametrize("bad_id", _BAD_TASK_IDS)
    def test_task_override_rejects_traversal_task_id(self, client: TestClient, bad_id: str) -> None:
        """POST /api/project/<valid>/task/<traversal>/override must return 400 Invalid."""
        resp = client.post(
            f"/api/project/{_VALID_PROJECT}/task/{bad_id}/override",
            json={"model_id": "test"},
            headers=_ADMIN_HEADERS,
        )
        _assert_traversal_rejected(resp)

    @pytest.mark.parametrize("bad_id", _BAD_PROJECT_IDS)
    def test_task_output_rejects_traversal_project_id(self, client: TestClient, bad_id: str) -> None:
        """GET /api/project/<traversal>/task/<valid>/output must return 400 Invalid."""
        resp = client.get(
            f"/api/project/{bad_id}/task/{_VALID_TASK}/output",
            headers=_ADMIN_HEADERS,
        )
        _assert_traversal_rejected(resp)

    @pytest.mark.parametrize("bad_id", _BAD_TASK_IDS)
    def test_task_output_rejects_traversal_task_id(self, client: TestClient, bad_id: str) -> None:
        """GET /api/project/<valid>/task/<traversal>/output must return 400 Invalid."""
        resp = client.get(
            f"/api/project/{_VALID_PROJECT}/task/{bad_id}/output",
            headers=_ADMIN_HEADERS,
        )
        _assert_traversal_rejected(resp)
