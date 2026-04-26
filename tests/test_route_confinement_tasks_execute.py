"""Direct request-level confinement tests for task output, add/update task, and execute routes.

Session 27A ownership artifact  -  identifier confinement for the unversioned task
output route (GET /api/v1/output/{task_id}) and three project-scoped task
management routes (POST/PUT /api/project/{project_id}/task,
POST /api/project/{project_id}/execute).

Each test class builds a minimal Litestar app containing only the handler group
under test and sends a request carrying a traversal-style identifier in the
path parameter position, asserting HTTP 400 rather than a filesystem operation.

URL-encoded traversal payloads are used because Litestar's path normalizer
collapses literal ``..`` sequences before the handler receives them (resulting in
404 rather than a call to validate_path_param).  The URL-encoded equivalents
(%2E%2E, %2E%2E%5C...) survive normalisation and reach the handler with the
decoded traversal value, exercising the real validation guard.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from unittest.mock import patch

import pytest
from litestar import Litestar
from litestar.testing import TestClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Admin bypass: sets VETINARI_ADMIN_TOKEN so admin_guard passes the token check.
_ADMIN_TOKEN = "confinement-test-token"
_ADMIN_HEADERS = {"X-Admin-Token": _ADMIN_TOKEN}

# Safe identifiers that pass validate_path_param without rejection.
_VALID_PROJECT = "valid-project"
_VALID_TASK = "valid-task"

# URL-encoded traversal task_id values that survive Litestar's path normalisation
# and reach the handler with a decoded value that validate_path_param rejects.
#
# %2E%2E           -> ".."           -  dot-dot, no slash, not normalised away
# %2E%2E%5Coutside -> "..\outside"   -  dot-dot backslash (Windows-style)
# my%2Etask        -> "my.task"      -  dot outside [A-Za-z0-9_-] safe set
_BAD_TASK_IDS = [
    "%2E%2E",
    "%2E%2E%5Coutside",
    "my%2Etask",
]

# URL-encoded traversal project_id values  -  same approach as task IDs.
_BAD_PROJECT_IDS = [
    "%2E%2E",
    "%2E%2E%5Coutside",
    "my%2Eproject",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tasks_client() -> Generator[TestClient, None, None]:
    """TestClient for a minimal app with only the tasks API handlers.

    Builds from ``create_tasks_api_handlers()`` so the unversioned output route
    (GET /api/v1/output/{task_id}) is present without loading the full projects
    handler set.  Admin auth is satisfied by the VETINARI_ADMIN_TOKEN env var.

    The explicit ``import vetinari.web.litestar_tasks_api`` call ensures the
    module is registered in sys.modules so Litestar can resolve forward
    references from ``from __future__ import annotations``.
    """
    import vetinari.web.litestar_tasks_api
    from vetinari.web.litestar_tasks_api import create_tasks_api_handlers

    handlers = create_tasks_api_handlers()
    if not handlers:
        pytest.skip("Litestar not installed  -  skipping confinement tests")

    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        app = Litestar(route_handlers=handlers, debug=True)
        with TestClient(app=app) as c:
            yield c


@pytest.fixture(scope="module")
def projects_client() -> Generator[TestClient, None, None]:
    """TestClient for a minimal app with only the projects API handlers.

    Builds from ``create_projects_api_handlers()`` so the add-task, update-task,
    and execute routes are present.  Admin auth is satisfied by the
    VETINARI_ADMIN_TOKEN env var.

    The explicit ``import vetinari.web.litestar_projects_api`` call ensures the
    module is in sys.modules before Litestar resolves forward references.
    """
    import vetinari.web.litestar_projects_api
    from vetinari.web.litestar_projects_api import create_projects_api_handlers

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
    """Assert that a traversal-style identifier was rejected with HTTP 400.

    Args:
        resp: An httpx Response object from TestClient.

    The validate_path_param guard returns a 400 error envelope of the form
    ``{"status": "error", "error": "<message>"}`` with ``"Invalid"`` in the
    error message.
    """
    assert resp.status_code == 400, (
        f"Expected HTTP 400 for traversal-style identifier, got {resp.status_code}. Body: {resp.text!r}"
    )
    body = resp.json()
    error_text = body.get("error", "") or body.get("detail", "") or resp.text
    assert "Invalid" in error_text, f"Expected 'Invalid' in error message, got: {error_text!r}"


# ---------------------------------------------------------------------------
# Tests  -  GET /api/v1/output/{task_id}
# ---------------------------------------------------------------------------


class TestTaskOutputRouteConfinement:
    """Traversal-style task_id rejection for GET /api/v1/output/{task_id}.

    This route lives in litestar_tasks_api (unversioned output retrieval) and
    was hardened in Session 27A with validate_path_param.  Each test sends a
    traversal-style task_id and asserts HTTP 400 with an 'Invalid' message.
    """

    @pytest.mark.parametrize("bad_id", _BAD_TASK_IDS)
    def test_output_rejects_traversal_task_id(self, tasks_client: TestClient, bad_id: str) -> None:
        """GET /api/v1/output/<traversal> must return 400 Invalid."""
        resp = tasks_client.get(f"/api/v1/output/{bad_id}")
        _assert_traversal_rejected(resp)


# ---------------------------------------------------------------------------
# Tests  -  POST /api/project/{project_id}/task (add task)
# ---------------------------------------------------------------------------


class TestAddTaskRouteConfinement:
    """Traversal-style project_id rejection for POST /api/project/{id}/task.

    The handler rejects the request before any filesystem access when
    project_id fails validate_path_param.
    """

    @pytest.mark.parametrize("bad_id", _BAD_PROJECT_IDS)
    def test_add_task_rejects_traversal_project_id(self, projects_client: TestClient, bad_id: str) -> None:
        """POST /api/project/<traversal>/task must return 400 Invalid."""
        resp = projects_client.post(
            f"/api/project/{bad_id}/task",
            json={"description": "test task"},
            headers=_ADMIN_HEADERS,
        )
        _assert_traversal_rejected(resp)


# ---------------------------------------------------------------------------
# Tests  -  PUT /api/project/{project_id}/task/{task_id} (update task)
# ---------------------------------------------------------------------------


class TestUpdateTaskRouteConfinement:
    """Traversal-style identifier rejection for PUT /api/project/{id}/task/{id}.

    The handler validates both project_id and task_id before accessing the
    filesystem.  Tests cover each parameter independently so a regression in
    either guard is detectable without the other hiding the failure.
    """

    @pytest.mark.parametrize("bad_id", _BAD_PROJECT_IDS)
    def test_update_task_rejects_traversal_project_id(self, projects_client: TestClient, bad_id: str) -> None:
        """PUT /api/project/<traversal>/task/<valid> must return 400 Invalid."""
        resp = projects_client.put(
            f"/api/project/{bad_id}/task/{_VALID_TASK}",
            json={"status": "completed"},
            headers=_ADMIN_HEADERS,
        )
        _assert_traversal_rejected(resp)

    @pytest.mark.parametrize("bad_id", _BAD_TASK_IDS)
    def test_update_task_rejects_traversal_task_id(self, projects_client: TestClient, bad_id: str) -> None:
        """PUT /api/project/<valid>/task/<traversal> must return 400 Invalid."""
        resp = projects_client.put(
            f"/api/project/{_VALID_PROJECT}/task/{bad_id}",
            json={"status": "completed"},
            headers=_ADMIN_HEADERS,
        )
        _assert_traversal_rejected(resp)


# ---------------------------------------------------------------------------
# Tests  -  POST /api/project/{project_id}/execute
# ---------------------------------------------------------------------------


class TestExecuteRouteConfinement:
    """Traversal-style project_id rejection for POST /api/project/{id}/execute.

    The handler rejects the request before any filesystem access when
    project_id fails validate_path_param.
    """

    @pytest.mark.parametrize("bad_id", _BAD_PROJECT_IDS)
    def test_execute_rejects_traversal_project_id(self, projects_client: TestClient, bad_id: str) -> None:
        """POST /api/project/<traversal>/execute must return 400 Invalid."""
        resp = projects_client.post(
            f"/api/project/{bad_id}/execute",
            json={"task_id": "t1"},
            headers=_ADMIN_HEADERS,
        )
        _assert_traversal_rejected(resp)
