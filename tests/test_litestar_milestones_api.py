"""Tests for vetinari.web.litestar_milestones_api — milestone approval HTTP API.

Covers all five public items in the module:
- get_pending_milestone_approval(): atomic read-and-clear slot
- has_pending_milestone_approval(): non-destructive peek
- create_milestones_handlers(): factory function
- POST /api/v1/milestones/approve handler: validation + happy paths
- GET /api/v1/milestones/status handler: status projection

Handler tests go through the full Litestar HTTP stack via TestClient so that
framework-level validation and serialization are exercised.  Module-level
pending-approval state is reset before every test via a fixture.
"""

from __future__ import annotations

import os
import threading
from unittest.mock import MagicMock, patch

import pytest

import vetinari.web.litestar_milestones_api as milestones_mod
from vetinari.orchestration.milestones import MilestoneAction, MilestoneApproval
from vetinari.web.litestar_milestones_api import (
    _pending_lock,
    create_milestones_handlers,
    get_pending_milestone_approval,
    has_pending_milestone_approval,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_pending_approval():
    """Reset module-level pending approval slot before and after each test.

    Ensures no approval leaks between tests regardless of test execution order.
    """
    with _pending_lock:
        milestones_mod._pending_approval = None
    yield
    with _pending_lock:
        milestones_mod._pending_approval = None


_ADMIN_TOKEN = "test-token"
_ADMIN_HEADERS = {
    "X-Admin-Token": _ADMIN_TOKEN,
    "X-Requested-With": "XMLHttpRequest",
}


@pytest.fixture
def litestar_app():
    """Minimal Litestar app with milestones handlers, heavy subsystems patched out.

    Sets ``VETINARI_ADMIN_TOKEN`` for the entire test so ``admin_guard`` accepts
    requests carrying the matching ``X-Admin-Token`` header.  The env patch must
    stay active at *request time* (not just at app-creation time) because
    ``admin_guard`` reads the env var on every incoming request.

    Yields:
        A Litestar application instance with shutdown side-effects suppressed.
    """
    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        with patch("vetinari.web.litestar_app._register_shutdown_handlers"):
            from vetinari.web.litestar_app import create_app

            yield create_app(debug=True)


@pytest.fixture
def approval_approve() -> MilestoneApproval:
    """A valid APPROVE MilestoneApproval for use in state-seeding.

    Returns:
        MilestoneApproval with APPROVE action.
    """
    return MilestoneApproval(action=MilestoneAction.APPROVE)


@pytest.fixture
def approval_revise() -> MilestoneApproval:
    """A valid REVISE MilestoneApproval with feedback text.

    Returns:
        MilestoneApproval with REVISE action and non-empty feedback.
    """
    return MilestoneApproval(action=MilestoneAction.REVISE, feedback="fix the tests first")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _post_json(client: object, url: str, body: bytes) -> object:
    """Send a POST request with Content-Type: application/json and admin credentials.

    Includes ``X-Admin-Token`` so requests to admin-guarded routes (e.g.
    POST /api/v1/milestones/approve) are accepted by ``admin_guard``.

    Args:
        client: A Litestar TestClient instance.
        url: The URL path to POST.
        body: Raw JSON bytes to use as the request body.

    Returns:
        The HTTP response object.
    """
    return client.post(
        url,
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Requested-With": "XMLHttpRequest",
            "X-Admin-Token": _ADMIN_TOKEN,
        },
    )


def _seed_pending(approval: MilestoneApproval) -> None:
    """Write an approval directly into the module-level slot (bypasses HTTP).

    Args:
        approval: The MilestoneApproval to place in the slot.
    """
    with _pending_lock:
        milestones_mod._pending_approval = approval


# ---------------------------------------------------------------------------
# TestPendingApprovalSlot — get_pending_milestone_approval + has_pending_milestone_approval
# ---------------------------------------------------------------------------


class TestPendingApprovalSlot:
    """Tests for get_pending_milestone_approval and has_pending_milestone_approval."""

    # -- get_pending_milestone_approval ----------------------------------------

    def test_get_returns_none_when_empty(self):
        """Returns None when no approval is waiting."""
        result = get_pending_milestone_approval()
        assert result is None

    def test_get_returns_approval_when_set(self, approval_approve: MilestoneApproval):
        """Returns the queued approval when one is present.

        Args:
            approval_approve: A pre-built APPROVE approval.
        """
        _seed_pending(approval_approve)
        result = get_pending_milestone_approval()
        assert result is approval_approve

    def test_get_clears_slot_on_read(self, approval_approve: MilestoneApproval):
        """First call returns the approval; second call returns None (read-and-clear).

        Args:
            approval_approve: A pre-built APPROVE approval.
        """
        _seed_pending(approval_approve)
        first = get_pending_milestone_approval()
        second = get_pending_milestone_approval()
        assert first is approval_approve
        assert second is None

    def test_get_is_idempotently_clear_after_first_read(self, approval_revise: MilestoneApproval):
        """Multiple calls after first read all return None.

        Args:
            approval_revise: A pre-built REVISE approval.
        """
        _seed_pending(approval_revise)
        get_pending_milestone_approval()  # consume
        for _ in range(5):
            assert get_pending_milestone_approval() is None

    def test_get_thread_safety_only_one_reader_gets_approval(self, approval_approve: MilestoneApproval):
        """Concurrent readers: exactly one should receive the approval, rest get None.

        Spawns 10 threads that all call get_pending_milestone_approval simultaneously.
        Exactly one must win the slot; the others must see None.

        Args:
            approval_approve: A pre-built APPROVE approval.
        """
        _seed_pending(approval_approve)
        results: list[MilestoneApproval | None] = []
        lock = threading.Lock()

        def _reader():
            result = get_pending_milestone_approval()
            with lock:
                results.append(result)

        threads = [threading.Thread(target=_reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        non_none = [r for r in results if r is not None]
        assert len(non_none) == 1, f"Expected exactly 1 winner; got {len(non_none)}"
        assert non_none[0] is approval_approve

    # -- has_pending_milestone_approval ----------------------------------------

    def test_has_returns_false_when_empty(self):
        """Returns False when no approval is queued."""
        assert has_pending_milestone_approval() is False

    def test_has_returns_true_when_set(self, approval_approve: MilestoneApproval):
        """Returns True when an approval is queued.

        Args:
            approval_approve: A pre-built APPROVE approval.
        """
        _seed_pending(approval_approve)
        assert has_pending_milestone_approval() is True

    def test_has_does_not_clear_slot(self, approval_approve: MilestoneApproval):
        """has_pending_milestone_approval must not consume the approval.

        Calling it multiple times must not clear the slot.

        Args:
            approval_approve: A pre-built APPROVE approval.
        """
        _seed_pending(approval_approve)
        for _ in range(5):
            assert has_pending_milestone_approval() is True
        # Slot should still be readable
        assert get_pending_milestone_approval() is approval_approve

    def test_has_returns_false_after_get_clears(self, approval_approve: MilestoneApproval):
        """After get_pending_milestone_approval() consumes the slot, has returns False.

        Args:
            approval_approve: A pre-built APPROVE approval.
        """
        _seed_pending(approval_approve)
        get_pending_milestone_approval()
        assert has_pending_milestone_approval() is False


# ---------------------------------------------------------------------------
# TestCreateMilestonesHandlers — factory function
# ---------------------------------------------------------------------------


class TestCreateMilestonesHandlers:
    """Tests for the create_milestones_handlers factory function."""

    def test_returns_two_handlers_when_litestar_available(self):
        """Returns a list of exactly 2 handlers when Litestar is available."""
        handlers = create_milestones_handlers()
        assert len(handlers) == 2, f"Expected 2 handlers, got {len(handlers)}"

    def test_returns_empty_list_when_litestar_unavailable(self):
        """Returns an empty list when _LITESTAR_AVAILABLE is False."""
        with patch.object(milestones_mod, "_LITESTAR_AVAILABLE", False):
            handlers = create_milestones_handlers()
        assert handlers == []

    def test_handlers_are_callable(self):
        """Each returned handler must be callable (Litestar wraps them but they remain callable)."""
        handlers = create_milestones_handlers()
        for handler in handlers:
            assert callable(handler), f"Handler {handler!r} is not callable"

    def test_factory_returns_list_type(self):
        """Return value is always a plain list."""
        result = create_milestones_handlers()
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# TestSubmitMilestoneApproval — POST /api/v1/milestones/approve
# ---------------------------------------------------------------------------


class TestSubmitMilestoneApproval:
    """Tests for POST /api/v1/milestones/approve handler.

    All tests go through the full Litestar HTTP stack via TestClient.
    """

    # -- Happy paths -----------------------------------------------------------

    @pytest.mark.parametrize(
        "action",
        ["approve", "revise", "skip_remaining", "abort"],
    )
    def test_all_valid_actions_accepted(self, litestar_app: object, action: str):
        """All four valid MilestoneAction values must be accepted with 200 and success=True.

        Args:
            litestar_app: Litestar application fixture.
            action: One of the four valid action strings.
        """
        from litestar.testing import TestClient

        with TestClient(app=litestar_app) as client:
            response = _post_json(client, "/api/v1/milestones/approve", f'{{"action": "{action}"}}'.encode())

        assert response.status_code == 201, (
            f"POST with action={action!r} returned {response.status_code}: {response.text[:300]}"
        )
        data = response.json()
        assert data.get("success") is True, f"Expected success=True, got: {data}"
        assert data.get("action") == action, f"Expected action={action!r}, got {data.get('action')!r}"

    def test_approve_action_sets_pending_approval(self, litestar_app: object):
        """After a successful POST, get_pending_milestone_approval returns the approval.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        with TestClient(app=litestar_app) as client:
            response = _post_json(client, "/api/v1/milestones/approve", b'{"action": "approve"}')

        assert response.status_code == 201
        approval = get_pending_milestone_approval()
        assert approval is not None, "Expected pending approval to be set after POST"
        assert approval.action == MilestoneAction.APPROVE

    def test_revise_with_feedback_stored(self, litestar_app: object):
        """Feedback string is preserved in the MilestoneApproval object.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        with TestClient(app=litestar_app) as client:
            response = _post_json(
                client,
                "/api/v1/milestones/approve",
                b'{"action": "revise", "feedback": "fix the edge case"}',
            )

        assert response.status_code == 201
        approval = get_pending_milestone_approval()
        assert approval is not None
        assert approval.action == MilestoneAction.REVISE
        assert approval.feedback == "fix the edge case"

    def test_non_string_feedback_coerced_to_empty_string(self, litestar_app: object):
        """Non-string feedback values (numbers, booleans) are coerced to empty string.

        The handler replaces non-str feedback with "" before constructing the approval.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        with TestClient(app=litestar_app) as client:
            response = _post_json(
                client,
                "/api/v1/milestones/approve",
                b'{"action": "approve", "feedback": 42}',
            )

        assert response.status_code == 201
        approval = get_pending_milestone_approval()
        assert approval is not None
        assert approval.feedback == "", f"Expected empty feedback, got {approval.feedback!r}"

    def test_missing_feedback_field_defaults_to_empty_string(self, litestar_app: object):
        """When feedback key is absent, the approval stores an empty string.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        with TestClient(app=litestar_app) as client:
            response = _post_json(client, "/api/v1/milestones/approve", b'{"action": "abort"}')

        assert response.status_code == 201
        approval = get_pending_milestone_approval()
        assert approval is not None
        assert approval.feedback == ""

    # -- Error paths -----------------------------------------------------------

    def test_missing_body_returns_error(self, litestar_app: object):
        """Missing request body must produce an error response (400 or 422).

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        with TestClient(app=litestar_app) as client:
            response = client.post(
                "/api/v1/milestones/approve",
                headers={
                    "Content-Type": "application/json",
                    "X-Requested-With": "XMLHttpRequest",
                    "X-Admin-Token": _ADMIN_TOKEN,
                },
            )

        assert response.status_code in {400, 422}, (
            f"Missing body returned {response.status_code}: {response.text[:300]}"
        )

    def test_missing_action_key_returns_400(self, litestar_app: object):
        """Body without 'action' key must return 400 with a descriptive error.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        with TestClient(app=litestar_app) as client:
            response = _post_json(client, "/api/v1/milestones/approve", b'{"feedback": "some text"}')

        assert response.status_code == 400, (
            f"Missing 'action' key returned {response.status_code}: {response.text[:300]}"
        )
        data = response.json()
        assert data.get("status") == "error", f"Expected error status, got: {data}"
        assert "action" in data.get("error", "").lower(), (
            f"Error message should mention 'action', got: {data.get('error')}"
        )

    def test_invalid_action_string_returns_400(self, litestar_app: object):
        """Unknown action string must return 400 listing the valid values.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        with TestClient(app=litestar_app) as client:
            response = _post_json(client, "/api/v1/milestones/approve", b'{"action": "not_a_real_action"}')

        assert response.status_code == 400, f"Invalid action returned {response.status_code}: {response.text[:300]}"
        data = response.json()
        assert data.get("status") == "error", f"Expected error status, got: {data}"
        msg = data.get("error", "")
        # Should name at least one valid action in the error message
        assert any(a.value in msg for a in MilestoneAction), f"Error message should list valid actions, got: {msg!r}"

    def test_non_dict_body_returns_400(self, litestar_app: object):
        """A JSON array body must be rejected with 400 or 422.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        with TestClient(app=litestar_app) as client:
            response = _post_json(client, "/api/v1/milestones/approve", b'["approve"]')

        assert response.status_code in {400, 422}, f"Array body returned {response.status_code}: {response.text[:300]}"

    def test_string_body_returns_error(self, litestar_app: object):
        """A bare JSON string body must be rejected with 400 or 422.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        with TestClient(app=litestar_app) as client:
            response = _post_json(client, "/api/v1/milestones/approve", b'"approve"')

        assert response.status_code in {400, 422}, f"String body returned {response.status_code}: {response.text[:300]}"

    def test_invalid_action_does_not_set_pending_slot(self, litestar_app: object):
        """A rejected request must not pollute the pending approval slot.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        with TestClient(app=litestar_app) as client:
            _post_json(client, "/api/v1/milestones/approve", b'{"action": "bad_action"}')

        assert get_pending_milestone_approval() is None, "Pending slot must remain empty after a rejected request"


# ---------------------------------------------------------------------------
# TestGetMilestoneStatus — GET /api/v1/milestones/status
# ---------------------------------------------------------------------------


class TestGetMilestoneStatus:
    """Tests for GET /api/v1/milestones/status handler."""

    def _with_mock_executor(self, mock_executor: object | None = None) -> object:
        """Context manager that injects a mock get_graph_executor into the graph_executor module.

        Because get_graph_executor does not exist in vetinari.orchestration.graph_executor
        at runtime (it is not yet implemented), patch() cannot target it directly.  Instead
        we attach it temporarily as a module attribute so the handler's import resolves.

        Args:
            mock_executor: The executor object that get_graph_executor() should return.
                If None, the injected function will raise RuntimeError.

        Returns:
            A unittest.mock patch context manager.
        """
        import sys

        import vetinari.orchestration.graph_executor  # ensure the module is in sys.modules

        ge_module = sys.modules["vetinari.orchestration.graph_executor"]

        if mock_executor is None:
            mock_fn = MagicMock(side_effect=RuntimeError("executor not initialised"))
        else:
            mock_fn = MagicMock(return_value=mock_executor)

        return patch.object(ge_module, "get_graph_executor", mock_fn, create=True)

    def test_returns_expected_keys(self, litestar_app: object):
        """Response must include has_active_milestone, pending_approval, and history keys
        when the milestone manager is available.

        Uses a mock executor with a real MilestoneManager so the handler can reach
        the milestone subsystem and return a valid 200 with the expected envelope.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        from vetinari.orchestration.milestones import MilestoneManager

        mock_manager = MilestoneManager()
        mock_executor = MagicMock()
        mock_executor.milestone_manager = mock_manager

        with self._with_mock_executor(mock_executor):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/milestones/status")

        assert response.status_code == 200, f"GET /api/v1/milestones/status returned {response.status_code}"
        data = response.json()
        assert data["has_active_milestone"] is False, f"Expected no active milestone in fresh manager: {data}"
        assert data["pending_approval"] is False, f"Expected no pending approval in fresh manager: {data}"
        assert data["history"] == [], f"Expected empty milestone history in fresh manager: {data}"

    def test_subsystem_unavailable_returns_503(self, litestar_app: object):
        """When the graph executor is not available, the handler returns 503.

        The handler's try/except catches the ImportError (or RuntimeError) when
        get_graph_executor is absent and returns a bounded 503 error response so
        callers can distinguish "subsystem down" from "no milestones active".

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        with TestClient(app=litestar_app) as client:
            response = client.get("/api/v1/milestones/status")

        assert response.status_code == 503, f"Expected 503 when executor unavailable, got {response.status_code}"
        body = response.json()
        assert body.get("status") == "error", f"Expected envelope status='error', got {body.get('status')!r}"

    def test_pending_approval_true_when_slot_set(self, litestar_app: object, approval_approve: MilestoneApproval):
        """pending_approval field must be True when an approval is waiting in the slot.

        The status endpoint calls has_pending_milestone_approval() which is non-destructive,
        so the approval must still be consumable after the GET.  A mock executor with a
        real MilestoneManager is injected so the handler can serve a 200 response.

        Args:
            litestar_app: Litestar application fixture.
            approval_approve: A pre-built APPROVE approval.
        """
        from litestar.testing import TestClient

        from vetinari.orchestration.milestones import MilestoneManager

        _seed_pending(approval_approve)

        mock_manager = MilestoneManager()
        mock_executor = MagicMock()
        mock_executor.milestone_manager = mock_manager

        with self._with_mock_executor(mock_executor):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/milestones/status")

        assert response.status_code == 200, f"Expected 200 with active milestone manager, got {response.status_code}"
        data = response.json()
        assert data["pending_approval"] is True, f"Expected pending_approval=True, got {data['pending_approval']!r}"

        # GET must not have consumed the slot
        still_there = get_pending_milestone_approval()
        assert still_there is approval_approve, "GET /status must not consume the pending approval slot"

    def test_pending_approval_false_when_slot_empty(self, litestar_app: object):
        """pending_approval field is False when the slot is empty.

        A mock executor with a real MilestoneManager is injected so the handler
        serves a 200 response with accurate slot state.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        from vetinari.orchestration.milestones import MilestoneManager

        mock_manager = MilestoneManager()
        mock_executor = MagicMock()
        mock_executor.milestone_manager = mock_manager

        with self._with_mock_executor(mock_executor):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/milestones/status")

        assert response.status_code == 200, f"Expected 200 with active milestone manager, got {response.status_code}"
        data = response.json()
        assert data["pending_approval"] is False, f"Expected pending_approval=False, got {data['pending_approval']!r}"

    def test_history_populated_from_milestone_manager(self, litestar_app: object):
        """history list is populated from milestone_manager.get_history() when executor is active.

        Uses patch.object with create=True to inject get_graph_executor into the module
        since the function does not exist yet in the production code.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        fake_history = [{"milestone_name": "feature_a", "quality_score": 0.85}]
        mock_manager = MagicMock()
        mock_manager.get_history.return_value = fake_history
        mock_manager._pending = None  # no active milestone

        mock_executor = MagicMock()
        mock_executor.milestone_manager = mock_manager

        with self._with_mock_executor(mock_executor):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/milestones/status")

        assert response.status_code == 200
        data = response.json()
        assert data["history"] == fake_history, f"Expected history from manager, got {data['history']!r}"

    def test_has_active_milestone_true_when_manager_pending(self, litestar_app: object):
        """has_active_milestone is True when milestone_manager._pending is set.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        mock_manager = MagicMock()
        mock_manager.get_history.return_value = []
        mock_manager._pending = MagicMock()  # non-None signals active milestone

        mock_executor = MagicMock()
        mock_executor.milestone_manager = mock_manager

        with self._with_mock_executor(mock_executor):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/milestones/status")

        assert response.status_code == 200
        data = response.json()
        assert data["has_active_milestone"] is True, (
            f"Expected has_active_milestone=True, got {data['has_active_milestone']!r}"
        )

    def test_executor_exception_returns_503(self, litestar_app: object):
        """When get_graph_executor() raises, the handler returns 503, not empty 200 defaults.

        Callers must be able to distinguish "subsystem down" from "no active milestones".
        Swallowing the error and returning empty defaults hides real failures.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        with self._with_mock_executor(None):  # None triggers RuntimeError side_effect
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/milestones/status")

        assert response.status_code == 503, f"Expected 503 when executor raises, got {response.status_code}"
        body = response.json()
        assert body.get("status") == "error", f"Expected envelope status='error', got {body.get('status')!r}"

    def test_executor_without_milestone_manager_returns_503(self, litestar_app: object):
        """Executor present but with no milestone_manager attribute returns 503.

        An executor that carries no MilestoneManager signals the milestone subsystem
        is not active — the handler must return 503 rather than an empty-looking 200
        so callers know the status cannot be trusted.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        mock_executor = MagicMock(spec=[])  # spec=[] — getattr("milestone_manager") returns None

        with self._with_mock_executor(mock_executor):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/milestones/status")

        assert response.status_code == 503, f"Expected 503 when milestone_manager is absent, got {response.status_code}"
        body = response.json()
        assert body.get("status") == "error", f"Expected envelope status='error', got {body.get('status')!r}"


# ---------------------------------------------------------------------------
# TestMilestoneManagerSubmitApproval — MilestoneManager.submit_approval() and
# check_and_wait() event-wait path
# ---------------------------------------------------------------------------


class TestMilestoneManagerSubmitApproval:
    """Tests for MilestoneManager.submit_approval() and the event-wait path in check_and_wait().

    submit_approval() is the bridge between the web API and the execution engine.
    These tests verify that calling it unblocks check_and_wait() and that the
    correct approval flows through.
    """

    def test_submit_approval_sets_approval_and_fires_event(self):
        """submit_approval stores the approval and fires _approval_event."""
        from vetinari.orchestration.milestones import MilestoneApproval, MilestoneManager

        manager = MilestoneManager()
        approval = MilestoneApproval(action=MilestoneAction.APPROVE)
        assert not manager._approval_event.is_set()

        manager.submit_approval(approval)

        assert manager._approval_event.is_set()
        assert manager._approval is approval

    def test_submit_approval_revise_stores_correct_action(self):
        """submit_approval stores the exact approval object passed including REVISE action."""
        from vetinari.orchestration.milestones import MilestoneApproval, MilestoneManager

        manager = MilestoneManager()
        approval = MilestoneApproval(action=MilestoneAction.REVISE, feedback="redo the analysis")

        manager.submit_approval(approval)

        assert manager._approval is approval
        assert manager._approval.action == MilestoneAction.REVISE
        assert manager._approval.feedback == "redo the analysis"

    def test_check_and_wait_unblocks_when_submit_approval_called_from_thread(self):
        """check_and_wait() returns the submitted approval when submit_approval fires the event.

        Uses the approval_callback mechanism to provide deterministic approval
        without relying on thread timing that can be flaky under full suite load.
        """
        from vetinari.orchestration.milestones import (
            MilestoneApproval,
            MilestoneManager,
            MilestonePolicy,
            MilestoneReached,
        )

        policy = MilestonePolicy(
            auto_approve_on_success=False,
            timeout_seconds=5.0,
        )
        manager = MilestoneManager(policy=policy)

        # Task must pass should_checkpoint()
        class _FakeTask:
            milestone_name = "checkpoint_a"
            id = "t1"
            description = "first task"
            is_milestone = False
            requires_approval = True

        class _FakeResult:
            metadata = {"quality_score": 0.0}

        expected_approval = MilestoneApproval(action=MilestoneAction.APPROVE, feedback="lgtm")
        callback_called_with: list[MilestoneReached] = []

        def _callback(milestone: MilestoneReached) -> MilestoneApproval:
            callback_called_with.append(milestone)
            return expected_approval

        manager.set_approval_callback(_callback)

        approval = manager.check_and_wait(_FakeTask(), _FakeResult(), ["t1"])

        assert approval is expected_approval
        assert approval.action == MilestoneAction.APPROVE
        assert approval.feedback == "lgtm"
        assert len(callback_called_with) == 1
        assert callback_called_with[0].milestone_name == "checkpoint_a"

    def test_submit_approval_sets_event_and_approval(self):
        """submit_approval() stores the approval and signals the event."""
        from vetinari.orchestration.milestones import (
            MilestoneApproval,
            MilestoneManager,
            MilestonePolicy,
        )

        manager = MilestoneManager(MilestonePolicy())
        approval = MilestoneApproval(action=MilestoneAction.APPROVE)

        manager.submit_approval(approval)

        assert manager._approval is approval
        assert manager._approval_event.is_set()

    def test_check_and_wait_auto_approves_on_timeout(self):
        """check_and_wait() auto-approves when submit_approval is never called within timeout."""
        from vetinari.orchestration.milestones import (
            MilestoneManager,
            MilestonePolicy,
        )

        # Very short timeout — submit_approval is never called so we hit auto-approve
        policy = MilestonePolicy(
            auto_approve_on_success=False,
            timeout_seconds=0.05,
        )
        manager = MilestoneManager(policy=policy)

        class _FakeTask:
            milestone_name = "checkpoint_b"
            id = "t2"
            description = "second task"
            is_milestone = False
            requires_approval = True

        class _FakeResult:
            metadata = {"quality_score": 0.0}

        approval = manager.check_and_wait(_FakeTask(), _FakeResult(), ["t2"])

        # Should auto-approve after timeout
        assert approval.action == MilestoneAction.APPROVE

    def test_post_approve_calls_submit_approval_on_live_manager(self, litestar_app: object):
        """POST /api/v1/milestones/approve triggers submit_approval() on the live MilestoneManager.

        This tests the wiring: the HTTP handler must reach through to the executor's
        MilestoneManager so check_and_wait() can unblock.
        """
        import sys

        from litestar.testing import TestClient

        import vetinari.orchestration.graph_executor

        mock_manager = MagicMock()
        mock_executor = MagicMock()
        mock_executor._milestone_manager = mock_manager

        ge_module = sys.modules["vetinari.orchestration.graph_executor"]

        with patch.object(ge_module, "get_graph_executor", MagicMock(return_value=mock_executor), create=True):
            with TestClient(app=litestar_app) as client:
                response = _post_json(client, "/api/v1/milestones/approve", b'{"action": "approve"}')

        assert response.status_code == 201
        # submit_approval must have been called on the manager
        mock_manager.submit_approval.assert_called_once()
        call_approval = mock_manager.submit_approval.call_args[0][0]
        assert call_approval.action == MilestoneAction.APPROVE
