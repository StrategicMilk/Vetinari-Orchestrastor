"""Tests proving CLI and library surfaces inherit path confinement from persistence layers.

Session 27A.3  -  identical confinement boundary as lower-level persistence managers:
- ``PromptVersionManager.save_version`` rejects traversal agent_type (Session 27A.2)
- ``PlanManager.update_plan`` / ``delete_plan`` reject traversal plan_id (Session 27A.2)
- ``SubtaskTree.create_subtask`` rejects traversal plan_id (Session 27A.2)

This file proves that the CLI entry points and HTTP route handlers above those managers
propagate the same rejection rather than accidentally bypassing it.

Confinement operates at two layers:

1. Litestar router  -  URL slash/dot segments are resolved by the HTTP router (404).
2. Application layer (``validate_path_param``)  -  backslash-containing segments that reach
   the handler are rejected with HTTP 400 "Invalid ...".

For CLI commands, the persistence layer raises ``ValueError`` which ``cmd_prompt``
now catches and converts to ``return 1`` with an error message.
"""

from __future__ import annotations

import os
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Constants shared across all classes
# ---------------------------------------------------------------------------

# Identifiers that reach the handler and must be rejected by validate_path_param (400).
# URL-encoded to survive Litestar path normalisation without being treated as
# literal path separators.
_BAD_PLAN_IDS = [
    "%2E%2E",  # decodes to ".."
    "%2E%2E%5Coutside",  # decodes to "..\outside"
    "my%2Eplan",  # decodes to "my.plan"  -  dot outside [A-Za-z0-9_-]
]

# Agent names that the persistence layer rejects via ValueError.
# Only slash/backslash sequences actually escape the versions directory:
# "."  -> lowercased to "._build.json", still inside the dir (no rejection)
# ".." -> lowercased to ".._build.json", still inside the dir (no rejection)
# "../escape" -> resolves outside the dir -> ValueError
# "..\escape" -> resolves outside the dir -> ValueError
_TRAVERSAL_AGENT_NAMES = ["../escape", "..\\escape"]

# An identifier that passes validate_path_param.
_VALID_PLAN_ID = "valid-plan"

# Admin token for route-level auth bypass.
_ADMIN_TOKEN = "confinement-test-token"
_ADMIN_HEADERS = {"X-Admin-Token": _ADMIN_TOKEN}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_traversal_rejected(resp: object) -> None:
    """Assert that a response carries HTTP 400 with an 'Invalid' message.

    Args:
        resp: An httpx Response object from TestClient.
    """
    assert resp.status_code == 400, (
        f"Expected HTTP 400 for traversal-style identifier, got {resp.status_code}. Body: {resp.text!r}"
    )
    body = resp.json()
    error_text = body.get("error", "") or body.get("detail", "") or resp.text
    assert "Invalid" in error_text, f"Expected 'Invalid' in error message, got: {error_text!r}"


# ---------------------------------------------------------------------------
# CLI  -  prompt history
# ---------------------------------------------------------------------------


class TestPromptHistoryCLIConfinement:
    """``vetinari prompt history`` rejects traversal agent names.

    The persistence layer (``PromptVersionManager._get_version_file``) raises
    ``ValueError`` when the constructed path would escape the versions directory.
    ``cmd_prompt`` must catch that exception and return exit code 1 instead of
    propagating an unhandled traceback.
    """

    def _make_args(self, agent: str, mode: str = "build") -> object:
        """Build a minimal args namespace for cmd_prompt.

        Args:
            agent: Raw agent name as supplied by the user on the CLI.
            mode: Agent mode string.

        Returns:
            SimpleNamespace mirroring the shape of argparse output.
        """
        ns = types.SimpleNamespace()
        ns.action = "history"
        ns.agent = agent
        ns.mode = mode
        ns.version = None
        return ns

    @pytest.mark.parametrize("bad_agent", _TRAVERSAL_AGENT_NAMES)
    def test_history_rejects_traversal_agent_name(self, bad_agent: str, tmp_path: object) -> None:
        """``cmd_prompt history`` with a traversal agent name returns exit code 1.

        The persistence layer raises ``ValueError`` before any file is accessed.
        ``cmd_prompt`` must catch it and surface exit code 1, not a raw traceback.

        Args:
            bad_agent: A traversal-style agent name.
            tmp_path: Pytest temporary directory injected as the versions store.
        """
        from vetinari.cli_commands import cmd_prompt
        from vetinari.prompts.version_manager import PromptVersionManager

        mgr = PromptVersionManager(versions_dir=tmp_path)
        args = self._make_args(bad_agent)

        # cmd_prompt does ``from vetinari.prompts import get_version_manager``
        # at call time, so we patch the name on the source module.
        with patch("vetinari.prompts.get_version_manager", return_value=mgr):
            rc = cmd_prompt(args)

        assert rc == 1, f"cmd_prompt history with traversal agent {bad_agent!r} must return exit code 1, got {rc}"

    def test_history_accepts_valid_agent_name(self, tmp_path: object) -> None:
        """``cmd_prompt history`` with a safe agent name does not raise.

        A safe agent name passes the persistence-layer path check.  The history
        will be empty for a fresh temp directory, so the command prints a
        "no versions" message and returns 0.

        Args:
            tmp_path: Pytest temporary directory for an isolated version store.
        """
        from vetinari.cli_commands import cmd_prompt
        from vetinari.prompts.version_manager import PromptVersionManager

        mgr = PromptVersionManager(versions_dir=tmp_path)
        args = self._make_args("WORKER")

        with patch("vetinari.prompts.get_version_manager", return_value=mgr):
            rc = cmd_prompt(args)

        assert rc == 0, f"cmd_prompt history with valid agent 'WORKER' must return 0, got {rc}"


# ---------------------------------------------------------------------------
# CLI  -  prompt rollback
# ---------------------------------------------------------------------------


class TestPromptRollbackCLIConfinement:
    """``vetinari prompt rollback`` rejects traversal agent names.

    The persistence layer raises ``ValueError`` before reading any file.
    ``cmd_prompt`` must catch it and return exit code 1.
    """

    def _make_args(self, agent: str, version: str = "1.0.0", mode: str = "build") -> object:
        """Build a minimal args namespace for cmd_prompt rollback.

        Args:
            agent: Raw agent name as supplied by the user on the CLI.
            version: Version string to roll back to.
            mode: Agent mode string.

        Returns:
            SimpleNamespace mirroring the shape of argparse output.
        """
        ns = types.SimpleNamespace()
        ns.action = "rollback"
        ns.agent = agent
        ns.mode = mode
        ns.version = version
        return ns

    @pytest.mark.parametrize("bad_agent", _TRAVERSAL_AGENT_NAMES)
    def test_rollback_rejects_traversal_agent_name(self, bad_agent: str, tmp_path: object) -> None:
        """``cmd_prompt rollback`` with a traversal agent name returns exit code 1.

        ``PromptVersionManager.rollback`` calls ``_load_history`` which calls
        ``_get_version_file`` and raises ``ValueError`` for unsafe names.
        ``cmd_prompt`` must catch it.

        Args:
            bad_agent: A traversal-style agent name.
            tmp_path: Pytest temporary directory injected as the versions store.
        """
        from vetinari.cli_commands import cmd_prompt
        from vetinari.prompts.version_manager import PromptVersionManager

        mgr = PromptVersionManager(versions_dir=tmp_path)
        args = self._make_args(bad_agent)

        with patch("vetinari.prompts.get_version_manager", return_value=mgr):
            rc = cmd_prompt(args)

        assert rc == 1, f"cmd_prompt rollback with traversal agent {bad_agent!r} must return exit code 1, got {rc}"

    def test_rollback_missing_version_flag_returns_1(self, tmp_path: object) -> None:
        """``cmd_prompt rollback`` without --version returns 1 before touching the manager.

        Args:
            tmp_path: Pytest temporary directory (unused, but keeps fixture symmetry).
        """
        from vetinari.cli_commands import cmd_prompt
        from vetinari.prompts.version_manager import PromptVersionManager

        mgr = PromptVersionManager(versions_dir=tmp_path)
        ns = types.SimpleNamespace(action="rollback", agent="WORKER", mode="build", version=None)

        with patch("vetinari.prompts.get_version_manager", return_value=mgr):
            rc = cmd_prompt(ns)

        assert rc == 1


# ---------------------------------------------------------------------------
# HTTP route  -  PUT /api/v1/plans/{plan_id}
# ---------------------------------------------------------------------------


from litestar import Litestar
from litestar.testing import TestClient


@pytest.fixture(scope="module")
def plans_client() -> object:
    """Minimal Litestar app with only the plans API handlers.

    Admin auth is satisfied via VETINARI_ADMIN_TOKEN environment variable so
    the guard accepts the ``X-Admin-Token`` header.
    """
    import vetinari.web.litestar_plans_api
    from vetinari.web.litestar_plans_api import create_plans_api_handlers

    handlers = create_plans_api_handlers()
    if not handlers:
        pytest.skip("Litestar not installed  -  skipping confinement tests")

    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        app = Litestar(route_handlers=handlers, debug=True)
        with TestClient(app=app) as c:
            yield c


@pytest.fixture(scope="module")
def subtasks_client() -> object:
    """Minimal Litestar app with only the subtasks API handlers."""
    import vetinari.web.litestar_subtasks_api
    from vetinari.web.litestar_subtasks_api import create_subtasks_api_handlers

    handlers = create_subtasks_api_handlers()
    if not handlers:
        pytest.skip("Litestar not installed  -  skipping confinement tests")

    app = Litestar(route_handlers=handlers, debug=True)
    with TestClient(app=app) as c:
        yield c


class TestPlanUpdateRouteConfinement:
    """PUT /api/v1/plans/{plan_id} rejects traversal-style plan IDs with HTTP 400."""

    @pytest.mark.parametrize("bad_id", _BAD_PLAN_IDS)
    def test_update_rejects_traversal_plan_id(self, plans_client: TestClient, bad_id: str) -> None:
        """PUT /api/v1/plans/<traversal> returns 400 Invalid plan_id.

        ``validate_path_param`` fires before ``PlanManager.update_plan`` is
        called, so the rejection is pure application-layer input validation.

        Args:
            plans_client: TestClient with plans handlers mounted.
            bad_id: A traversal-style plan_id value (URL-encoded).
        """
        resp = plans_client.put(
            f"/api/v1/plans/{bad_id}",
            json={"title": "pwned"},
            headers=_ADMIN_HEADERS,
        )
        _assert_traversal_rejected(resp)

    def test_update_accepts_valid_plan_id(self, plans_client: TestClient) -> None:
        """PUT /api/v1/plans/<valid> is not blocked by the confinement check.

        The plan does not exist in the in-memory store, so the handler returns
        404  -  but the critical assertion is that it is NOT rejected with
        400 "Invalid plan_id".

        Args:
            plans_client: TestClient with plans handlers mounted.
        """
        resp = plans_client.put(
            f"/api/v1/plans/{_VALID_PLAN_ID}",
            json={"title": "updated"},
            headers=_ADMIN_HEADERS,
        )
        assert resp.status_code != 400 or "Invalid" not in resp.text, (
            f"Valid plan_id '{_VALID_PLAN_ID}' must not be rejected by confinement. Got {resp.status_code}: {resp.text}"
        )


class TestPlanDeleteRouteConfinement:
    """DELETE /api/v1/plans/{plan_id} rejects traversal-style plan IDs with HTTP 400."""

    @pytest.mark.parametrize("bad_id", _BAD_PLAN_IDS)
    def test_delete_rejects_traversal_plan_id(self, plans_client: TestClient, bad_id: str) -> None:
        """DELETE /api/v1/plans/<traversal> returns 400 Invalid plan_id.

        ``validate_path_param`` fires before ``PlanManager.delete_plan`` is
        called, preventing the persistence layer from ever seeing the identifier.

        Args:
            plans_client: TestClient with plans handlers mounted.
            bad_id: A traversal-style plan_id value (URL-encoded).
        """
        resp = plans_client.delete(
            f"/api/v1/plans/{bad_id}",
            headers=_ADMIN_HEADERS,
        )
        _assert_traversal_rejected(resp)

    def test_delete_accepts_valid_plan_id(self, plans_client: TestClient) -> None:
        """DELETE /api/v1/plans/<valid> is not blocked by the confinement check.

        The plan does not exist, so the handler returns 404  -  but the critical
        assertion is that it is NOT rejected with 400 "Invalid plan_id".

        Args:
            plans_client: TestClient with plans handlers mounted.
        """
        resp = plans_client.delete(
            f"/api/v1/plans/{_VALID_PLAN_ID}",
            headers=_ADMIN_HEADERS,
        )
        assert resp.status_code != 400 or "Invalid" not in resp.text, (
            f"Valid plan_id '{_VALID_PLAN_ID}' must not be rejected by confinement. Got {resp.status_code}: {resp.text}"
        )


# ---------------------------------------------------------------------------
# HTTP route  -  POST /api/v1/subtasks/{plan_id}
# ---------------------------------------------------------------------------


class TestSubtaskCreateRouteConfinement:
    """POST /api/v1/subtasks/{plan_id} rejects traversal-style plan IDs with HTTP 400."""

    @pytest.mark.parametrize("bad_id", _BAD_PLAN_IDS)
    def test_create_subtask_rejects_traversal_plan_id(self, subtasks_client: TestClient, bad_id: str) -> None:
        """POST /api/v1/subtasks/<traversal> returns 400 Invalid plan_id.

        ``validate_path_param`` fires before ``SubtaskTree.create_subtask`` is
        called, so the confinement check is at the application layer.

        Args:
            subtasks_client: TestClient with subtasks handlers mounted.
            bad_id: A traversal-style plan_id value (URL-encoded).
        """
        resp = subtasks_client.post(
            f"/api/v1/subtasks/{bad_id}",
            json={"description": "pwned task"},
        )
        _assert_traversal_rejected(resp)

    def test_create_subtask_accepts_valid_plan_id(self, subtasks_client: TestClient) -> None:
        """POST /api/v1/subtasks/<valid> is not blocked by the confinement check.

        The subtask creation may fail for other reasons (e.g. storage), but
        the critical assertion is that the request passes path validation.

        Args:
            subtasks_client: TestClient with subtasks handlers mounted.
        """
        resp = subtasks_client.post(
            f"/api/v1/subtasks/{_VALID_PLAN_ID}",
            json={"description": "safe task"},
        )
        assert resp.status_code != 400 or "Invalid" not in resp.text, (
            f"Valid plan_id '{_VALID_PLAN_ID}' must not be rejected by confinement. Got {resp.status_code}: {resp.text}"
        )
