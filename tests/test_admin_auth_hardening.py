"""Tests for SESSION-28 admin auth hardening — approval, autonomy, and milestone routes.

Verifies that every mutating route hardened in SESSION-28 enforces the admin_guard.
Tests go through the full Litestar HTTP stack via TestClient so that guard wiring,
middleware, and serialization are all exercised — handler.fn() calls are explicitly
avoided per the handler-direct-route-tests anti-pattern rule.

Covered routes (all added guards=[admin_guard] in SESSION-28):
    POST   /api/v1/approvals/{action_id}/approve
    POST   /api/v1/approvals/{action_id}/reject
    POST   /api/v1/autonomy/promote/{action_type}
    POST   /api/v1/autonomy/veto/{action_type}         (approvals_api)
    DELETE /api/v1/autonomy/veto/{action_type}         (approvals_api)
    PUT    /api/v1/autonomy/policies/{action_type}     (litestar_autonomy_api)
    POST   /api/v1/autonomy/promotions/{action_type}/veto  (litestar_autonomy_api)
    POST   /api/v1/undo/{action_id}                    (litestar_autonomy_api)
    POST   /api/v1/milestones/approve                  (litestar_milestones_api)

Test strategy: set VETINARI_ADMIN_TOKEN to a known value so the guard always
performs token comparison (bypassing the localhost-IP fallback that would make
TestClient look like an authorised caller regardless of headers).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from litestar.testing import TestClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Stable test token — used only inside this test module, never in production.
_TEST_ADMIN_TOKEN = "test-admin-token-session28-hardening"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def litestar_app():
    """Minimal Litestar app with shutdown side-effects suppressed.

    Returns:
        A Litestar application instance ready for TestClient.
    """
    with patch("vetinari.web.litestar_app._register_shutdown_handlers"):
        from vetinari.web.litestar_app import create_app

        app = create_app(debug=True)
    return app


@contextmanager
def _admin_token_env(token: str = _TEST_ADMIN_TOKEN):
    """Temporarily set VETINARI_ADMIN_TOKEN so guards perform token comparison.

    Without an env token the guard falls back to IP-based localhost check,
    which would make TestClient requests look authorised and defeat the test.

    Args:
        token: The admin token value to inject into the environment.
    """
    original = os.environ.get("VETINARI_ADMIN_TOKEN")
    os.environ["VETINARI_ADMIN_TOKEN"] = token
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("VETINARI_ADMIN_TOKEN", None)
        else:
            os.environ["VETINARI_ADMIN_TOKEN"] = original


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auth_headers(token: str = _TEST_ADMIN_TOKEN) -> dict[str, str]:
    """Build request headers containing a valid admin token.

    Args:
        token: Admin token to include in X-Admin-Token.

    Returns:
        Dict of HTTP headers including X-Admin-Token.
    """
    return {"X-Admin-Token": token, "Content-Type": "application/json", "X-Requested-With": "XMLHttpRequest"}


def _no_auth_headers() -> dict[str, str]:
    """Build request headers that pass CSRF but have no admin credentials.

    Includes ``X-Requested-With`` to pass CSRF middleware, so the request
    reaches the admin_guard where the 401 rejection should occur.

    Returns:
        Dict of HTTP headers with CSRF header but no admin token.
    """
    return {"Content-Type": "application/json", "X-Requested-With": "XMLHttpRequest"}


# ---------------------------------------------------------------------------
# TestApprovalRoutesRequireAdmin
# ---------------------------------------------------------------------------


class TestApprovalRoutesRequireAdmin:
    """All POST/DELETE routes in approvals_api must reject unauthenticated requests."""

    def test_approve_action_returns_401_without_token(self, litestar_app: object) -> None:
        """POST /api/v1/approvals/{id}/approve returns 401 without admin token.

        The admin_guard must fire before any business logic executes.

        Args:
            litestar_app: Litestar application fixture.
        """

        with _admin_token_env():
            with TestClient(app=litestar_app) as client:
                response = client.post(
                    "/api/v1/approvals/act_abc123/approve",
                    headers=_no_auth_headers(),
                )

        assert response.status_code == 401, (
            f"Expected 401 from unauthenticated POST /approve, got {response.status_code}: {response.text[:300]}"
        )

    def test_approve_action_succeeds_with_valid_token(self, litestar_app: object) -> None:
        """POST /api/v1/approvals/{id}/approve passes guard and returns 200 with valid token.

        Uses a mock approval queue so the handler can complete without real storage.

        Args:
            litestar_app: Litestar application fixture.
        """

        mock_queue = MagicMock()
        mock_queue.approve.return_value = True

        with _admin_token_env():
            with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
                with TestClient(app=litestar_app) as client:
                    response = client.post(
                        "/api/v1/approvals/act_abc123/approve",
                        headers=_auth_headers(),
                    )

        assert response.status_code == 201, (
            f"Expected 201 with valid token, got {response.status_code}: {response.text[:300]}"
        )
        data = response.json()
        assert data.get("success") is True

    def test_reject_action_returns_401_without_token(self, litestar_app: object) -> None:
        """POST /api/v1/approvals/{id}/reject returns 401 without admin token.

        Args:
            litestar_app: Litestar application fixture.
        """

        with _admin_token_env():
            with TestClient(app=litestar_app) as client:
                response = client.post(
                    "/api/v1/approvals/act_abc123/reject",
                    headers=_no_auth_headers(),
                    content=b'{"reason": "too risky"}',
                )

        assert response.status_code == 401, (
            f"Expected 401 from unauthenticated POST /reject, got {response.status_code}: {response.text[:300]}"
        )

    def test_reject_action_succeeds_with_valid_token(self, litestar_app: object) -> None:
        """POST /api/v1/approvals/{id}/reject passes guard with valid token.

        Args:
            litestar_app: Litestar application fixture.
        """

        mock_queue = MagicMock()
        mock_queue.reject.return_value = True

        with _admin_token_env():
            with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
                with TestClient(app=litestar_app) as client:
                    response = client.post(
                        "/api/v1/approvals/act_abc123/reject",
                        headers=_auth_headers(),
                        content=b'{"reason": "too risky"}',
                    )

        assert response.status_code == 201, (
            f"Expected 201 with valid token, got {response.status_code}: {response.text[:300]}"
        )

    def test_apply_promotion_returns_401_without_token(self, litestar_app: object) -> None:
        """POST /api/v1/autonomy/promote/{type} returns 401 without admin token.

        Args:
            litestar_app: Litestar application fixture.
        """

        with _admin_token_env():
            with TestClient(app=litestar_app) as client:
                response = client.post(
                    "/api/v1/autonomy/promote/param_tuning",
                    headers=_no_auth_headers(),
                )

        assert response.status_code == 401, (
            f"Expected 401 from unauthenticated POST /promote, got {response.status_code}: {response.text[:300]}"
        )

    def test_veto_promotion_returns_401_without_token(self, litestar_app: object) -> None:
        """POST /api/v1/autonomy/veto/{type} (approvals_api) returns 401 without token.

        Args:
            litestar_app: Litestar application fixture.
        """

        with _admin_token_env():
            with TestClient(app=litestar_app) as client:
                response = client.post(
                    "/api/v1/autonomy/veto/param_tuning",
                    headers=_no_auth_headers(),
                )

        assert response.status_code == 401, (
            f"Expected 401 from unauthenticated POST /veto, got {response.status_code}: {response.text[:300]}"
        )

    def test_clear_veto_returns_401_without_token(self, litestar_app: object) -> None:
        """DELETE /api/v1/autonomy/veto/{type} returns 401 without admin token.

        Args:
            litestar_app: Litestar application fixture.
        """

        with _admin_token_env():
            with TestClient(app=litestar_app) as client:
                response = client.delete(
                    "/api/v1/autonomy/veto/param_tuning",
                    headers=_no_auth_headers(),
                )

        assert response.status_code == 401, (
            f"Expected 401 from unauthenticated DELETE /veto, got {response.status_code}: {response.text[:300]}"
        )

    def test_wrong_token_returns_401(self, litestar_app: object) -> None:
        """Wrong admin token is rejected with 401 (constant-time comparison still rejects).

        Confirms the guard distinguishes a wrong token from no token at all.

        Args:
            litestar_app: Litestar application fixture.
        """

        with _admin_token_env():
            with TestClient(app=litestar_app) as client:
                response = client.post(
                    "/api/v1/approvals/act_abc123/approve",
                    headers={
                        "X-Admin-Token": "wrong-token",
                        "Content-Type": "application/json",
                        "X-Requested-With": "XMLHttpRequest",
                    },
                )

        assert response.status_code == 401, (
            f"Expected 401 with wrong token, got {response.status_code}: {response.text[:300]}"
        )


# ---------------------------------------------------------------------------
# TestAutonomyPolicyRouteRequiresAdmin
# ---------------------------------------------------------------------------


class TestAutonomyPolicyRouteRequiresAdmin:
    """Routes in litestar_autonomy_api that were hardened in SESSION-28."""

    def test_update_policy_returns_401_without_token(self, litestar_app: object) -> None:
        """PUT /api/v1/autonomy/policies/{type} returns 401 without admin token.

        Args:
            litestar_app: Litestar application fixture.
        """

        with _admin_token_env():
            with TestClient(app=litestar_app) as client:
                response = client.put(
                    "/api/v1/autonomy/policies/param_tuning",
                    headers=_no_auth_headers(),
                    content=b'{"level": "suggest"}',
                )

        assert response.status_code == 401, (
            f"Expected 401 from unauthenticated PUT /policies, got {response.status_code}: {response.text[:300]}"
        )

    def test_veto_promotion_autonomy_returns_401_without_token(self, litestar_app: object) -> None:
        """POST /api/v1/autonomy/promotions/{type}/veto returns 401 without admin token.

        Args:
            litestar_app: Litestar application fixture.
        """

        with _admin_token_env():
            with TestClient(app=litestar_app) as client:
                response = client.post(
                    "/api/v1/autonomy/promotions/param_tuning/veto",
                    headers=_no_auth_headers(),
                )

        assert response.status_code == 401, (
            f"Expected 401 from unauthenticated POST /promotions/veto, got {response.status_code}: {response.text[:300]}"
        )

    def test_undo_action_returns_401_without_token(self, litestar_app: object) -> None:
        """POST /api/v1/undo/{action_id} returns 401 without admin token.

        Args:
            litestar_app: Litestar application fixture.
        """

        with _admin_token_env():
            with TestClient(app=litestar_app) as client:
                response = client.post(
                    "/api/v1/undo/act_abc123",
                    headers=_no_auth_headers(),
                )

        assert response.status_code == 401, (
            f"Expected 401 from unauthenticated POST /undo, got {response.status_code}: {response.text[:300]}"
        )

    def test_update_policy_succeeds_with_valid_token(self, litestar_app: object) -> None:
        """PUT /api/v1/autonomy/policies/{type} passes guard with a valid admin token.

        Uses a mock load/save so the handler does not touch the real config file.

        Args:
            litestar_app: Litestar application fixture.
        """

        import vetinari.web.litestar_autonomy_api as autonomy_mod
        from vetinari.web.litestar_autonomy_api import AutonomyPolicy

        existing_policy = AutonomyPolicy(action_type="param_tuning", level="suggest", updated_at="2026-01-01T00:00:00Z")

        with _admin_token_env():
            with (
                patch.object(autonomy_mod, "_load_policies", return_value=[existing_policy]),
                patch.object(autonomy_mod, "_save_policies_unlocked"),
                patch.object(autonomy_mod, "_policies_cache", None),
            ):
                with TestClient(app=litestar_app) as client:
                    response = client.put(
                        "/api/v1/autonomy/policies/param_tuning",
                        headers=_auth_headers(),
                        content=b'{"level": "propose"}',
                    )

        # Guard passed — any non-401 indicates the guard allowed the request through.
        assert response.status_code != 401, f"Valid token was incorrectly rejected (401): {response.text[:300]}"


# ---------------------------------------------------------------------------
# TestMilestoneApproveRequiresAdmin
# ---------------------------------------------------------------------------


class TestMilestoneApproveRequiresAdmin:
    """POST /api/v1/milestones/approve must enforce admin_guard after SESSION-28."""

    def test_milestone_approve_returns_401_without_token(self, litestar_app: object) -> None:
        """POST /api/v1/milestones/approve returns 401 when no admin token is supplied.

        Args:
            litestar_app: Litestar application fixture.
        """

        with _admin_token_env():
            with TestClient(app=litestar_app) as client:
                response = client.post(
                    "/api/v1/milestones/approve",
                    headers=_no_auth_headers(),
                    content=b'{"action": "approve"}',
                )

        assert response.status_code == 401, (
            f"Expected 401 from unauthenticated POST /milestones/approve, "
            f"got {response.status_code}: {response.text[:300]}"
        )

    def test_milestone_approve_returns_401_with_wrong_token(self, litestar_app: object) -> None:
        """POST /api/v1/milestones/approve returns 401 when an incorrect token is supplied.

        Confirms the guard distinguishes a wrong token from a valid one.

        Args:
            litestar_app: Litestar application fixture.
        """

        with _admin_token_env():
            with TestClient(app=litestar_app) as client:
                response = client.post(
                    "/api/v1/milestones/approve",
                    headers={
                        "X-Admin-Token": "completely-wrong-token",
                        "Content-Type": "application/json",
                        "X-Requested-With": "XMLHttpRequest",
                    },
                    content=b'{"action": "approve"}',
                )

        assert response.status_code == 401, (
            f"Expected 401 with wrong token, got {response.status_code}: {response.text[:300]}"
        )

    def test_milestone_approve_succeeds_with_valid_token(self, litestar_app: object) -> None:
        """POST /api/v1/milestones/approve passes guard and returns 201 with valid admin token.

        Confirms that adding guards=[admin_guard] did not break the happy-path response.

        Args:
            litestar_app: Litestar application fixture.
        """

        import vetinari.web.litestar_milestones_api as milestones_mod

        with _admin_token_env():
            # Reset the pending slot so the approval does not leak between tests.

            with milestones_mod._pending_lock:
                milestones_mod._pending_approval = None

            with TestClient(app=litestar_app) as client:
                response = client.post(
                    "/api/v1/milestones/approve",
                    headers=_auth_headers(),
                    content=b'{"action": "approve"}',
                )

        # Clean up pending slot after test.
        with milestones_mod._pending_lock:
            milestones_mod._pending_approval = None

        assert response.status_code == 201, (
            f"Expected 201 with valid token, got {response.status_code}: {response.text[:300]}"
        )
        data = response.json()
        assert data.get("success") is True, f"Expected success=True, got: {data}"
        assert data.get("action") == "approve", f"Expected action='approve', got: {data.get('action')!r}"

    def test_milestone_approve_csrf_only_no_longer_sufficient(self, litestar_app: object) -> None:
        """X-Requested-With header alone must not pass the admin guard.

        Before SESSION-28 the route was guarded only by the CSRF middleware.
        After SESSION-28 an admin token is also required.  This test confirms
        that CSRF-only headers are rejected.

        Args:
            litestar_app: Litestar application fixture.
        """

        with _admin_token_env():
            with TestClient(app=litestar_app) as client:
                response = client.post(
                    "/api/v1/milestones/approve",
                    headers={
                        "X-Requested-With": "XMLHttpRequest",
                        "Content-Type": "application/json",
                    },
                    content=b'{"action": "approve"}',
                )

        assert response.status_code == 401, (
            f"CSRF-only header must not pass admin_guard — got {response.status_code}: {response.text[:300]}"
        )


# ---------------------------------------------------------------------------
# TestAdminGuardProperties
# ---------------------------------------------------------------------------


class TestAdminGuardProperties:
    """Core security properties of the admin_guard itself (ADR-0066 contract)."""

    def test_guard_raises_not_authorized_exception(self) -> None:
        """admin_guard raises NotAuthorizedException when token does not match.

        This verifies the guard signals 401 (not 403 or 500) per Litestar protocol.
        """
        from litestar.exceptions import NotAuthorizedException

        from vetinari.web.litestar_guards import admin_guard

        mock_connection = MagicMock()
        mock_connection.headers = {}  # no token headers
        mock_connection.client.host = "1.2.3.4"  # not localhost

        with _admin_token_env():
            with pytest.raises(NotAuthorizedException):
                admin_guard(mock_connection, MagicMock())

    def test_guard_passes_when_token_matches(self) -> None:
        """admin_guard returns None (no exception) when the correct token is supplied.

        Returns:
            None — guard passes by returning without raising.
        """
        from vetinari.web.litestar_guards import admin_guard

        mock_connection = MagicMock()
        mock_connection.headers = {"X-Admin-Token": _TEST_ADMIN_TOKEN}

        with _admin_token_env():
            result = admin_guard(mock_connection, MagicMock())

        # Guards return None on success; any exception would indicate a failure.
        assert result is None

    def test_guard_fails_closed_on_comparison_exception(self) -> None:
        """admin_guard returns False (not True) when token comparison raises an exception.

        Verifies the anti-pattern rule: fail CLOSED, never fail open.
        """
        from vetinari.web.litestar_guards import is_admin_connection

        mock_connection = MagicMock()
        # Make .encode() raise to trigger the except branch in is_admin_connection
        bad_provided = MagicMock()
        bad_provided.__bool__ = lambda self: True
        bad_provided.encode = MagicMock(side_effect=ValueError("encoding failed"))

        with _admin_token_env():
            # Patch the header value so provided.encode() raises
            mock_connection.headers = MagicMock()
            mock_connection.headers.get = MagicMock(return_value=bad_provided)

            result = is_admin_connection(mock_connection)

        assert result is False, f"Expected fail-closed (False) on exception, got {result!r}"

    def test_bearer_token_accepted(self) -> None:
        """admin_guard accepts Authorization: Bearer <token> as an alternative to X-Admin-Token.

        Args:
            Implicit — uses module-level _TEST_ADMIN_TOKEN.
        """
        from vetinari.web.litestar_guards import is_admin_connection

        mock_connection = MagicMock()
        mock_connection.headers = {
            "X-Admin-Token": "",
            "Authorization": f"Bearer {_TEST_ADMIN_TOKEN}",
        }

        with _admin_token_env():
            result = is_admin_connection(mock_connection)

        assert result is True, f"Expected Bearer token to be accepted, got {result!r}"


class TestCaseInsensitiveBearer:
    """RFC 7235 §5.1.2 — auth-scheme is case-insensitive.

    These tests go through the real mounted Litestar HTTP stack via TestClient so
    that middleware, guard wiring, and header normalisation are all exercised.
    A handler-direct ``.fn(...)`` call would bypass the httpx layer that an actual
    client uses, so we use a real route mounted by ``create_app``.
    """

    @pytest.fixture
    def client(self, litestar_app):
        """Yield a Litestar TestClient against the real mounted app.

        Returns:
            TestClient bound to a Litestar app produced by create_app().
        """

        with TestClient(app=litestar_app) as tc:
            yield tc

    @pytest.mark.parametrize(
        "scheme",
        ["Bearer", "bearer", "BEARER", "BeArEr", "bEaReR"],
    )
    def test_bearer_scheme_case_insensitive_over_http(self, client, scheme: str) -> None:
        """Requests using any case of 'Bearer' with a valid token succeed at the HTTP layer.

        The admin-guarded /api/v1/autonomy/policies route is invoked via GET (a read
        the guard still evaluates) with each case variant of the Bearer scheme.
        A case-sensitive implementation would return 401 for lowercase/mixed-case
        scheme names; the contract here is that all variants resolve to the same
        authenticated state.
        """
        with _admin_token_env():
            response = client.get(
                "/api/v1/analytics/cost",
                headers={"Authorization": f"{scheme} {_TEST_ADMIN_TOKEN}"},
            )
        # Not 401 — any case of Bearer must resolve to an authenticated caller.
        # 200 = route replied with policies; 403/404 = separate route-level concern.
        # The contract under test is *not 401*, which is what a case-sensitive
        # parser would return for everything except exact "Bearer".
        assert response.status_code != 401, (
            f"Authorization: {scheme} <valid-token> must be accepted (RFC 7235 §5.1.2), "
            f"got {response.status_code}. Body: {response.text[:200]}"
        )

    def test_bearer_scheme_invalid_token_still_rejected_case_insensitive(self, client) -> None:
        """Invalid token under any-case Bearer is still rejected with 401.

        Case-insensitivity applies only to the scheme name, not the credential itself.
        This confirms we have not weakened the token check.
        """
        with _admin_token_env():
            response = client.get(
                "/api/v1/analytics/cost",
                headers={"Authorization": "bearer not-the-right-token"},
            )
        assert response.status_code == 401, (
            f"Invalid token must 401 even with lowercase bearer scheme, got {response.status_code}"
        )

    def test_bearer_scheme_malformed_header_rejected(self, client) -> None:
        """Authorization header without scheme/credential separator returns 401.

        Tests a malformed header like ``Authorization: Bearer`` (no space + token)
        or a missing credential. The parser must not panic and must not treat the
        request as authenticated.
        """
        with _admin_token_env():
            for bad in ("Bearer", "Bearer ", "bearer", "bearer  ", "Bearerabc"):
                response = client.get(
                    "/api/v1/analytics/cost",
                    headers={"Authorization": bad},
                )
                assert response.status_code == 401, (
                    f"Malformed Authorization header {bad!r} must 401, got {response.status_code}"
                )
