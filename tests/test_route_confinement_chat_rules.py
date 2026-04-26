"""Tests proving identifier confinement for rules, events, chat export, feedback, and retry routes.

Each test class builds a minimal Litestar app containing only the handler group
under test, following the pattern established in test_project_route_confinement.py.

URL-path traversal tests use curated subsets of TRAVERSAL_IDS that survive httpx's
URL normalization and still reach the route handler with the traversal value intact.
Multi-segment traversals like ``../../etc/passwd`` are collapsed by httpx before the
request is sent, and the resulting path either hits no route (404) or resolves to a
clean identifier  -  both are safe rejections, but asserting 400 on them would be
incorrect since the handler's validate_path_param guard is never invoked.

POST-body tests use a broader set since body content is not URL-normalized.

Security invariant: no traversal-style value ever reaches a filesystem operation.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from unittest.mock import patch

import pytest
from litestar import Litestar
from litestar.testing import TestClient

from tests.factories import TRAVERSAL_IDS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ADMIN_TOKEN = "confinement-test-token"
_ADMIN_HEADERS = {"X-Admin-Token": _ADMIN_TOKEN}
_VALID_PROJECT = "valid-project"
_VALID_TASK = "valid-task"

# IDs that survive httpx normalization for ALL path param positions and reach
# the handler with the traversal value intact.
#
# Excluded:
#   - ".." / "../outside": collapsed to a shorter path (strip the last segment)
#   - "../../etc/passwd": multi-segment collapse; produces a no-route 404
#   - "valid/../escape": collapses to a clean identifier ("escape")  -  clean IDs
#     are NOT rejected (that's correct behavior), so asserting 400 would be wrong
#   - "/absolute/path": creates a double-slash mid-path for routes that have a
#     suffix after {project_id} (e.g., /events, /task_id), causing a 404 from
#     the router rather than a 400 from the handler
#   - "normal\x00null": httpx raises InvalidURL before sending
_PATH_TRAVERSAL_IDS = [
    "..\\outside",  # backslash variant  -  not collapsed, reaches handler
    "C:\\Windows",  # Windows path  -  not collapsed, reaches handler
]

# All non-null-byte traversal IDs for POST-body tests where no URL normalization
# applies and all values reach the handler for validation.
_BODY_TRAVERSAL_IDS = [t for t in TRAVERSAL_IDS if "\x00" not in t]


# ---------------------------------------------------------------------------
# Fixtures  -  one minimal app per handler group
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rules_client() -> Generator[TestClient, None, None]:
    """TestClient for the rules route handlers only.

    Explicit module import ensures the module is in sys.modules so Litestar
    can resolve forward references from ``from __future__ import annotations``.
    """
    import vetinari.web.litestar_rules_routes
    from vetinari.web.litestar_rules_routes import create_rules_routes_handlers

    handlers = create_rules_routes_handlers()
    if not handlers:
        pytest.skip("Litestar not installed  -  skipping confinement tests")

    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        app = Litestar(route_handlers=handlers, debug=True)
        with TestClient(app=app) as c:
            yield c


@pytest.fixture(scope="module")
def execution_client() -> Generator[TestClient, None, None]:
    """TestClient for the projects execution handlers only."""
    import vetinari.web.litestar_projects_execution
    from vetinari.web.litestar_projects_execution import create_projects_execution_handlers

    handlers = create_projects_execution_handlers()
    if not handlers:
        pytest.skip("Litestar not installed  -  skipping confinement tests")

    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        app = Litestar(route_handlers=handlers, debug=True)
        with TestClient(app=app) as c:
            yield c


@pytest.fixture(scope="module")
def chat_client() -> Generator[TestClient, None, None]:
    """TestClient for the chat API handlers only."""
    import vetinari.web.litestar_chat_api
    from vetinari.web.litestar_chat_api import create_chat_api_handlers

    handlers = create_chat_api_handlers()
    if not handlers:
        pytest.skip("Litestar not installed  -  skipping confinement tests")

    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        app = Litestar(route_handlers=handlers, debug=True)
        with TestClient(app=app) as c:
            yield c


# ---------------------------------------------------------------------------
# Rules route confinement
# ---------------------------------------------------------------------------


class TestRulesGetConfinement:
    """GET /api/v1/rules/project/{project_id} rejects traversal identifiers."""

    @pytest.mark.parametrize("bad_id", _PATH_TRAVERSAL_IDS)
    def test_rejects_traversal_project_id(self, rules_client, bad_id):
        resp = rules_client.get(f"/api/v1/rules/project/{bad_id}")
        assert resp.status_code == 400


class TestRulesPostConfinement:
    """POST /api/v1/rules/project/{project_id} rejects traversal identifiers."""

    @pytest.mark.parametrize("bad_id", _PATH_TRAVERSAL_IDS)
    def test_rejects_traversal_project_id(self, rules_client, bad_id):
        resp = rules_client.post(
            f"/api/v1/rules/project/{bad_id}",
            json={"rules": []},
            headers=_ADMIN_HEADERS,
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Events route confinement
# ---------------------------------------------------------------------------


class TestEventsConfinement:
    """GET /api/v1/projects/{project_id}/events rejects traversal identifiers."""

    @pytest.mark.parametrize("bad_id", _PATH_TRAVERSAL_IDS)
    def test_rejects_traversal_project_id(self, execution_client, bad_id):
        resp = execution_client.get(f"/api/v1/projects/{bad_id}/events")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Chat export confinement
# ---------------------------------------------------------------------------


class TestChatExportConfinement:
    """GET /api/v1/chat/export/{project_id} rejects traversal identifiers."""

    @pytest.mark.parametrize("bad_id", _PATH_TRAVERSAL_IDS)
    def test_rejects_traversal_project_id(self, chat_client, bad_id):
        resp = chat_client.get(f"/api/v1/chat/export/{bad_id}")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Chat feedback confinement  -  body-based, full set applies
# ---------------------------------------------------------------------------


class TestChatFeedbackConfinement:
    """POST /api/v1/chat/feedback rejects traversal project_id in POST body.

    Body content is not URL-normalized so all non-null-byte TRAVERSAL_IDS
    reach the handler and the validate_path_param guard is exercised on each.
    """

    @pytest.mark.parametrize("bad_id", _BODY_TRAVERSAL_IDS)
    def test_rejects_traversal_project_id_in_body(self, chat_client, bad_id):
        resp = chat_client.post(
            "/api/v1/chat/feedback",
            json={
                "project_id": bad_id,
                "task_id": "t1",
                "rating": "up",
                "comment": "test",
            },
            headers=_ADMIN_HEADERS,
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Chat retry confinement
# ---------------------------------------------------------------------------


class TestChatRetryConfinement:
    """POST /api/v1/chat/retry/{project_id}/{task_id} rejects traversal identifiers."""

    @pytest.mark.parametrize("bad_id", _PATH_TRAVERSAL_IDS)
    def test_rejects_traversal_project_id(self, chat_client, bad_id):
        resp = chat_client.post(
            f"/api/v1/chat/retry/{bad_id}/{_VALID_TASK}",
            json={},
            headers=_ADMIN_HEADERS,
        )
        assert resp.status_code == 400

    @pytest.mark.parametrize("bad_id", _PATH_TRAVERSAL_IDS)
    def test_rejects_traversal_task_id(self, chat_client, bad_id):
        resp = chat_client.post(
            f"/api/v1/chat/retry/{_VALID_PROJECT}/{bad_id}",
            json={},
            headers=_ADMIN_HEADERS,
        )
        assert resp.status_code == 400
