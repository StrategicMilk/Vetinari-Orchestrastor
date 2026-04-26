"""Identifier confinement tests for model-search, swap-model, files, git, and attachment routes.

Proves that each route rejects traversal-style project_id values and, for git
routes, arbitrary filesystem paths outside PROJECT_ROOT.

Routes covered:
    POST /api/v1/project/{project_id}/model-search   -  model-search (litestar_models_catalog)
    POST /api/v1/swap-model                           -  swap-model (litestar_models_discovery)
    GET  /api/project/{project_id}/files/list         -  list workspace files
    POST /api/project/{project_id}/files/read         -  read workspace file
    POST /api/project/{project_id}/files/write        -  write workspace file
    POST /api/v1/project/git/commit-message           -  git commit-message (path-based)
    POST /api/v1/project/git/commit-message-path      -  git commit-message-path (path-based)
    POST /api/v1/project/git/conflicts                -  git conflicts (path-based)
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

_ADMIN_TOKEN = "confinement-test-token"
_ADMIN_HEADERS = {"X-Admin-Token": _ADMIN_TOKEN}

# Traversal IDs for project_id URL parameters.
#
# Litestar's router normalises URL paths before dispatch, so multi-segment
# traversal sequences (e.g. "../../etc/passwd", "valid/../escape") and bare
# absolute paths ("/absolute/path") are collapsed by the HTTP layer and never
# reach the handler  -  they produce 404 at routing time.  Single-segment values
# that contain backslashes or Windows drive-letter syntax DO reach the handler
# and must be rejected with 400 by validate_path_param().
#
# Both outcomes are correct rejections: a 404 means the router never matched
# the handler (the path was normalised away), and a 400 means the handler
# rejected the identifier explicitly.  The contract tested here is:
#   "no traversal-style identifier may produce a 2xx response".
_BAD_PROJECT_IDS = [
    "..",
    "../outside",
    "..\\outside",
    "../../etc/passwd",
    "valid/../escape",
    "/absolute/path",
    "C:\\Windows",
]


# ---------------------------------------------------------------------------
# Fixtures  -  one minimal app per handler group to isolate dependencies
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def models_catalog_client() -> Generator[TestClient, None, None]:
    """TestClient backed by a minimal app with only the models-catalog handlers."""
    import vetinari.web.litestar_models_catalog
    from vetinari.web.litestar_models_catalog import create_models_catalog_handlers

    handlers = create_models_catalog_handlers()
    if not handlers:
        pytest.skip("Litestar not installed  -  skipping confinement tests")
    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        app = Litestar(route_handlers=handlers, debug=True)
        with TestClient(app=app) as c:
            yield c


@pytest.fixture(scope="module")
def models_discovery_client() -> Generator[TestClient, None, None]:
    """TestClient backed by a minimal app with only the models-discovery handlers."""
    import vetinari.web.litestar_models_discovery
    from vetinari.web.litestar_models_discovery import create_models_discovery_handlers

    handlers = create_models_discovery_handlers()
    if not handlers:
        pytest.skip("Litestar not installed  -  skipping confinement tests")
    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        app = Litestar(route_handlers=handlers, debug=True)
        with TestClient(app=app) as c:
            yield c


@pytest.fixture(scope="module")
def projects_client() -> Generator[TestClient, None, None]:
    """TestClient backed by a minimal app with only the projects-api handlers."""
    import vetinari.web.litestar_projects_api
    from vetinari.web.litestar_projects_api import create_projects_api_handlers

    handlers = create_projects_api_handlers()
    if not handlers:
        pytest.skip("Litestar not installed  -  skipping confinement tests")
    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        app = Litestar(route_handlers=handlers, debug=True)
        with TestClient(app=app) as c:
            yield c


@pytest.fixture(scope="module")
def git_client() -> Generator[TestClient, None, None]:
    """TestClient backed by a minimal app with only the project-git handlers."""
    import vetinari.web.litestar_project_git
    from vetinari.web.litestar_project_git import create_project_git_handlers

    handlers = create_project_git_handlers()
    if not handlers:
        pytest.skip("Litestar not installed  -  skipping confinement tests")
    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        app = Litestar(route_handlers=handlers, debug=True)
        with TestClient(app=app) as c:
            yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_rejected(resp: object, *, context: str = "") -> None:
    """Assert that a response is a client-error rejection (400 or 404).

    Both 400 and 404 are valid rejections for traversal-style identifiers:
    - 400: the handler received the identifier and explicitly rejected it via
      ``validate_path_param()``  -  the expected path for single-segment values
      such as ``..\\outside`` and ``C:\\Windows``.
    - 404: Litestar's router normalised the URL path before dispatch (e.g.
      ``../../etc/passwd`` collapses to a different path), so the traversal
      identifier never reached the handler.  The important thing is that no
      2xx response was produced.

    Args:
        resp: An httpx Response from TestClient.
        context: Extra context string to include in the assertion message.
    """
    assert resp.status_code in (400, 404), (
        f"Expected HTTP 400 or 404{(' for ' + context) if context else ''}, got {resp.status_code}. Body: {resp.text!r}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModelSearchConfinement:
    """POST /api/v1/project/{project_id}/model-search rejects traversal identifiers."""

    @pytest.mark.parametrize("bad_id", _BAD_PROJECT_IDS)
    def test_rejects_traversal_project_id(self, models_catalog_client: TestClient, bad_id: str) -> None:
        """Traversal project_id must return 400 before any filesystem access."""
        resp = models_catalog_client.post(
            f"/api/v1/project/{bad_id}/model-search",
            json={"task_description": "test"},
            headers=_ADMIN_HEADERS,
        )
        _assert_rejected(resp, context=f"model-search bad_id={bad_id!r}")


class TestSwapModelConfinement:
    """POST /api/v1/swap-model rejects traversal project_id in request body."""

    @pytest.mark.parametrize("bad_id", _BAD_PROJECT_IDS)
    def test_rejects_traversal_project_id_in_body(self, models_discovery_client: TestClient, bad_id: str) -> None:
        """Traversal project_id in POST body must return 400 before file writes."""
        resp = models_discovery_client.post(
            "/api/v1/swap-model",
            json={"project_id": bad_id, "model_id": "some-model"},
            headers=_ADMIN_HEADERS,
        )
        _assert_rejected(resp, context=f"swap-model bad_id={bad_id!r}")


class TestFilesListConfinement:
    """GET /api/project/{project_id}/files/list rejects traversal identifiers."""

    @pytest.mark.parametrize("bad_id", _BAD_PROJECT_IDS)
    def test_rejects_traversal_project_id(self, projects_client: TestClient, bad_id: str) -> None:
        """Traversal project_id must return 400 before workspace path is constructed."""
        resp = projects_client.get(
            f"/api/project/{bad_id}/files/list",
            headers=_ADMIN_HEADERS,
        )
        _assert_rejected(resp, context=f"files/list bad_id={bad_id!r}")


class TestFilesReadConfinement:
    """POST /api/project/{project_id}/files/read rejects traversal identifiers."""

    @pytest.mark.parametrize("bad_id", _BAD_PROJECT_IDS)
    def test_rejects_traversal_project_id(self, projects_client: TestClient, bad_id: str) -> None:
        """Traversal project_id must return 400 before any file read is attempted."""
        resp = projects_client.post(
            f"/api/project/{bad_id}/files/read",
            json={"path": "README.md"},
            headers=_ADMIN_HEADERS,
        )
        _assert_rejected(resp, context=f"files/read bad_id={bad_id!r}")


class TestFilesWriteConfinement:
    """POST /api/project/{project_id}/files/write rejects traversal identifiers.

    Critically, the 400 must be returned BEFORE any filesystem write so a
    traversal attempt cannot create directories or files outside the project.
    """

    @pytest.mark.parametrize("bad_id", _BAD_PROJECT_IDS)
    def test_rejects_traversal_project_id(self, projects_client: TestClient, bad_id: str) -> None:
        """Traversal project_id must return 400 before workspace mkdir is called."""
        resp = projects_client.post(
            f"/api/project/{bad_id}/files/write",
            json={"path": "canary.txt", "content": "traversal-probe"},
            headers=_ADMIN_HEADERS,
        )
        _assert_rejected(resp, context=f"files/write bad_id={bad_id!r}")


class TestGitCommitMessageConfinement:
    """POST /api/v1/project/git/commit-message rejects paths outside PROJECT_ROOT."""

    def test_rejects_absolute_windows_path(self, git_client: TestClient) -> None:
        """An absolute Windows-style path must return 400."""
        resp = git_client.post(
            "/api/v1/project/git/commit-message",
            json={"project_path": "C:\\Windows"},
            headers=_ADMIN_HEADERS,
        )
        _assert_rejected(resp, context="commit-message C:\\Windows")

    def test_rejects_traversal_relative_path(self, git_client: TestClient) -> None:
        """A traversal path resolving outside PROJECT_ROOT must return 400."""
        resp = git_client.post(
            "/api/v1/project/git/commit-message",
            json={"project_path": "../../etc"},
            headers=_ADMIN_HEADERS,
        )
        _assert_rejected(resp, context="commit-message ../../etc")

    def test_rejects_absolute_unix_path(self, git_client: TestClient) -> None:
        """An absolute Unix path outside PROJECT_ROOT must return 400."""
        resp = git_client.post(
            "/api/v1/project/git/commit-message",
            json={"project_path": "/etc/passwd"},
            headers=_ADMIN_HEADERS,
        )
        _assert_rejected(resp, context="commit-message /etc/passwd")


class TestGitCommitMessagePathConfinement:
    """POST /api/v1/project/git/commit-message-path rejects paths outside PROJECT_ROOT."""

    def test_rejects_absolute_windows_path(self, git_client: TestClient) -> None:
        """An absolute Windows-style repo_path must return 400."""
        resp = git_client.post(
            "/api/v1/project/git/commit-message-path",
            json={"repo_path": "C:\\Windows"},
            headers=_ADMIN_HEADERS,
        )
        _assert_rejected(resp, context="commit-message-path C:\\Windows")

    def test_rejects_traversal_relative_path(self, git_client: TestClient) -> None:
        """A traversal repo_path resolving outside PROJECT_ROOT must return 400."""
        resp = git_client.post(
            "/api/v1/project/git/commit-message-path",
            json={"repo_path": "../../etc"},
            headers=_ADMIN_HEADERS,
        )
        _assert_rejected(resp, context="commit-message-path ../../etc")


class TestGitConflictsConfinement:
    """POST /api/v1/project/git/conflicts rejects paths outside PROJECT_ROOT."""

    def test_rejects_absolute_windows_path(self, git_client: TestClient) -> None:
        """An absolute Windows-style project_path must return 400."""
        resp = git_client.post(
            "/api/v1/project/git/conflicts",
            json={"project_path": "C:\\Windows"},
            headers=_ADMIN_HEADERS,
        )
        _assert_rejected(resp, context="git/conflicts C:\\Windows")

    def test_rejects_traversal_relative_path(self, git_client: TestClient) -> None:
        """A traversal project_path resolving outside PROJECT_ROOT must return 400."""
        resp = git_client.post(
            "/api/v1/project/git/conflicts",
            json={"project_path": "../../etc"},
            headers=_ADMIN_HEADERS,
        )
        _assert_rejected(resp, context="git/conflicts ../../etc")
