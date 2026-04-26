"""Probe tests for SESSION-33G task 33G.2 — control-surface semantic correctness.

Each test class covers one of the five defects identified in the SESSION-33G
audit.  All tests go through the full Litestar HTTP app stack via TestClient
so that route registration, guards, middleware, and serialization are exercised
— handler.fn() calls are explicitly avoided per the handler-direct-route-tests
anti-pattern rule.

Covered defects:
    Defect 1 — GET /api/v1/system/vram returns 503 (not 200) when VRAM manager
                is unavailable.
    Defect 2 — POST /api/admin/credentials/{source_type} rejects boolean
                ``rotation_days`` with 422 instead of silently coercing ``true``
                to ``1`` (Python: ``isinstance(True, int)`` is True).
    Defect 3 — POST /api/v1/system/vram/phase rejects unknown phase strings with
                422; only "loading", "inference", "idle" are accepted.
    Defect 4 — POST /api/project/{project_id}/pause returns 409 Conflict when the
                project exists but is not currently running.
    Defect 5 — GET /api/v1/projects/{project_id}/events returns 404 for a
                project_id that does not exist on disk.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from litestar.testing import TestClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEST_ADMIN_TOKEN = "test-session-33g-control-surface-semantics"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def litestar_app():
    """Minimal Litestar app with subsystem wiring and shutdown side-effects suppressed.

    Both patches must stay active through TestClient.__enter__ (where the lifespan
    hook runs _wire_subsystems via run_in_executor). Using yield keeps the context
    managers alive for the duration of each test.

    Yields:
        A Litestar application instance ready for TestClient.
    """
    with patch("vetinari.web.litestar_app._register_shutdown_handlers"):
        with patch("vetinari.cli_startup._wire_subsystems"):
            from vetinari.web.litestar_app import create_app

            yield create_app(debug=True)


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


def _auth_headers(token: str = _TEST_ADMIN_TOKEN) -> dict[str, str]:
    """Build request headers containing a valid admin token.

    Args:
        token: Admin token to include in X-Admin-Token.

    Returns:
        Dict of HTTP headers including X-Admin-Token and Content-Type.
    """
    return {
        "X-Admin-Token": token,
        "Content-Type": "application/json",
        "X-Requested-With": "XMLHttpRequest",
    }


# ---------------------------------------------------------------------------
# Defect 1 — VRAM unavailable must return 503, not 200
# ---------------------------------------------------------------------------


class TestVramUnavailableReturns503:
    """GET /api/v1/system/vram must return 503 when the VRAM manager fails."""

    def test_vram_endpoint_returns_503_when_manager_raises(self, litestar_app: object) -> None:
        """VRAM manager throwing any exception produces HTTP 503, not 200.

        Confirms that the false-green defect is fixed: before the fix the
        handler returned 200 with an error field; now it raises
        ServiceUnavailableException so Litestar serialises a 503.

        Args:
            litestar_app: Litestar application fixture.
        """
        with patch(
            "vetinari.models.vram_manager.get_vram_manager",
            side_effect=RuntimeError("VRAM hardware not available"),
        ):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/system/vram")

        assert response.status_code == 503, (
            f"Expected 503 when VRAM manager is unavailable, got {response.status_code}: {response.text[:300]}"
        )

    def test_vram_endpoint_returns_200_when_manager_succeeds(self, litestar_app: object) -> None:
        """VRAM manager available produces HTTP 200 with expected fields.

        Args:
            litestar_app: Litestar application fixture.
        """
        mock_mgr = MagicMock()
        mock_mgr.is_thermal_throttled.return_value = False
        mock_mgr.get_phase_recommendation.return_value = "inference"
        mock_mgr.get_status.return_value = {"total_mb": 8192, "available_mb": 4096}

        with patch(
            "vetinari.models.vram_manager.get_vram_manager",
            return_value=mock_mgr,
        ):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/system/vram")

        assert response.status_code == 200, (
            f"Expected 200 when VRAM manager is available, got {response.status_code}: {response.text[:300]}"
        )
        body = response.json()
        assert "thermal_throttled" in body, f"Missing 'thermal_throttled' in response: {body}"
        assert body["thermal_throttled"] is False


# ---------------------------------------------------------------------------
# Defect 2 — rotation_days=true (boolean) must be rejected with 422
# ---------------------------------------------------------------------------


class TestCredentialRotationDaysBooleanRejected:
    """POST /api/admin/credentials/{source_type} must reject JSON boolean rotation_days."""

    def test_boolean_true_rotation_days_returns_422(self, litestar_app: object) -> None:
        """rotation_days=true (JSON boolean) must return 422, not silently coerce to 1.

        Python's ``isinstance(True, int)`` returns True, so without the explicit
        bool check the handler would accept the boolean and store 1 day.

        Args:
            litestar_app: Litestar application fixture.
        """
        payload = {"token": "secret-value", "rotation_days": True}

        with _admin_token_env():
            with TestClient(app=litestar_app) as client:
                response = client.post(
                    "/api/admin/credentials/github",
                    headers=_auth_headers(),
                    json=payload,
                )

        assert response.status_code == 422, (
            f"Expected 422 for boolean rotation_days=true, got {response.status_code}: {response.text[:300]}"
        )

    def test_boolean_false_rotation_days_returns_422(self, litestar_app: object) -> None:
        """rotation_days=false (JSON boolean) must also return 422.

        Args:
            litestar_app: Litestar application fixture.
        """
        payload = {"token": "secret-value", "rotation_days": False}

        with _admin_token_env():
            with TestClient(app=litestar_app) as client:
                response = client.post(
                    "/api/admin/credentials/github",
                    headers=_auth_headers(),
                    json=payload,
                )

        assert response.status_code == 422, (
            f"Expected 422 for boolean rotation_days=false, got {response.status_code}: {response.text[:300]}"
        )

    def test_integer_rotation_days_is_accepted(self, litestar_app: object) -> None:
        """rotation_days=30 (integer) must be accepted and stored successfully.

        Args:
            litestar_app: Litestar application fixture.
        """
        mock_mgr = MagicMock()
        payload = {"token": "secret-value", "rotation_days": 30}

        with _admin_token_env():
            with patch(
                "vetinari.credentials.get_credential_manager",
                return_value=mock_mgr,
            ):
                with TestClient(app=litestar_app) as client:
                    response = client.post(
                        "/api/admin/credentials/github",
                        headers=_auth_headers(),
                        json=payload,
                    )

        assert response.status_code == 201, (
            f"Expected 201 for valid integer rotation_days, got {response.status_code}: {response.text[:300]}"
        )
        body = response.json()
        assert body.get("status") == "ok"
        assert body.get("message") == "Credential set for github"


# ---------------------------------------------------------------------------
# Defect 3 — phase must be validated against allowlist {"loading","inference","idle"}
# ---------------------------------------------------------------------------


class TestVramPhaseAllowlist:
    """POST /api/v1/system/vram/phase must reject unknown phase strings with 422."""

    @pytest.mark.parametrize("invalid_phase", ["foobar", "LOADING", "running", "active", "  "])
    def test_invalid_phase_returns_422(self, litestar_app: object, invalid_phase: str) -> None:
        """Phase values outside the allowlist must return 422 Unprocessable Entity.

        Args:
            litestar_app: Litestar application fixture.
            invalid_phase: A phase string that is not in {"loading","inference","idle"}.
        """
        with _admin_token_env():
            with TestClient(app=litestar_app) as client:
                response = client.post(
                    "/api/v1/system/vram/phase",
                    headers=_auth_headers(),
                    json={"phase": invalid_phase},
                )

        assert response.status_code == 422, (
            f"Expected 422 for invalid phase {invalid_phase!r}, got {response.status_code}: {response.text[:300]}"
        )

    @pytest.mark.parametrize("valid_phase", ["loading", "inference", "idle"])
    def test_valid_phase_is_accepted(self, litestar_app: object, valid_phase: str) -> None:
        """Phase values in the allowlist must be accepted and return 200.

        Args:
            litestar_app: Litestar application fixture.
            valid_phase: A phase string in {"loading","inference","idle"}.
        """
        mock_mgr = MagicMock()
        mock_mgr.get_phase_recommendation.return_value = valid_phase

        with _admin_token_env():
            with patch(
                "vetinari.models.vram_manager.get_vram_manager",
                return_value=mock_mgr,
            ):
                with TestClient(app=litestar_app) as client:
                    response = client.post(
                        "/api/v1/system/vram/phase",
                        headers=_auth_headers(),
                        json={"phase": valid_phase},
                    )

        assert response.status_code == 200, (
            f"Expected 200 for valid phase {valid_phase!r}, got {response.status_code}: {response.text[:300]}"
        )
        body = response.json()
        assert body.get("phase") == valid_phase, (
            f"Response phase mismatch: expected {valid_phase!r}, got {body.get('phase')!r}"
        )

    def test_missing_phase_returns_400(self, litestar_app: object) -> None:
        """Empty or missing phase field must return 400 Bad Request.

        Args:
            litestar_app: Litestar application fixture.
        """
        with _admin_token_env():
            with TestClient(app=litestar_app) as client:
                response = client.post(
                    "/api/v1/system/vram/phase",
                    headers=_auth_headers(),
                    json={},
                )

        assert response.status_code == 400, (
            f"Expected 400 for missing phase, got {response.status_code}: {response.text[:300]}"
        )


# ---------------------------------------------------------------------------
# Defect 4 — pause on non-running project must return 409 Conflict
# ---------------------------------------------------------------------------


class TestPauseIdleProjectReturns409:
    """POST /api/project/{project_id}/pause must return 409 when project is not running."""

    def test_pause_idle_project_returns_409(self, litestar_app: object, tmp_path: Path) -> None:
        """Pausing a project that exists on disk but is not in _cancel_flags returns 409.

        The handler checks _is_project_actually_running() which reads _cancel_flags.
        When no flag is registered the project is idle and cannot be paused.

        Args:
            litestar_app: Litestar application fixture.
            tmp_path: Pytest temporary directory used as PROJECT_ROOT.
        """
        project_id = "proj_idle_test_abc"
        # Create the project directory so we pass the 404 check
        project_dir = tmp_path / "projects" / project_id
        project_dir.mkdir(parents=True)

        # Patch PROJECT_ROOT so the handler finds the project dir
        # Also patch _is_project_actually_running to return False (no active flag)
        with _admin_token_env():
            with patch("vetinari.web.shared.PROJECT_ROOT", tmp_path):
                with patch(
                    "vetinari.web.shared._is_project_actually_running",
                    return_value=False,
                ):
                    with TestClient(app=litestar_app) as client:
                        response = client.post(
                            f"/api/project/{project_id}/pause",
                            headers=_auth_headers(),
                        )

        assert response.status_code == 409, (
            f"Expected 409 when pausing idle project, got {response.status_code}: {response.text[:300]}"
        )

    def test_pause_nonexistent_project_returns_404(self, litestar_app: object, tmp_path: Path) -> None:
        """Pausing a project that does not exist on disk returns 404 Not Found.

        Args:
            litestar_app: Litestar application fixture.
            tmp_path: Pytest temporary directory used as PROJECT_ROOT.
        """
        # Do NOT create the project directory — it should not exist
        with _admin_token_env():
            with patch("vetinari.web.shared.PROJECT_ROOT", tmp_path):
                with TestClient(app=litestar_app) as client:
                    response = client.post(
                        "/api/project/proj_does_not_exist/pause",
                        headers=_auth_headers(),
                    )

        assert response.status_code == 404, (
            f"Expected 404 for nonexistent project, got {response.status_code}: {response.text[:300]}"
        )


# ---------------------------------------------------------------------------
# Defect 5 — events for nonexistent project must return 404
# ---------------------------------------------------------------------------


class TestEventsNonexistentProjectReturns404:
    """GET /api/v1/projects/{project_id}/events must return 404 for unknown projects."""

    def test_events_nonexistent_project_returns_404(self, litestar_app: object, tmp_path: Path) -> None:
        """Events endpoint returns 404 when the project directory does not exist.

        Before the fix the endpoint called get_recent_sse_events() regardless of
        whether the project existed and returned 200 with an empty list — masking
        the nonexistent-project error.

        Args:
            litestar_app: Litestar application fixture.
            tmp_path: Pytest temporary directory used as PROJECT_ROOT.
        """
        # The handler does a late import: `from vetinari.web.shared import PROJECT_ROOT`
        # so we patch the source attribute which the late import reads each call.
        with patch("vetinari.web.shared.PROJECT_ROOT", tmp_path):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/projects/proj_ghost_xyz/events")

        assert response.status_code == 404, (
            f"Expected 404 for nonexistent project events, got {response.status_code}: {response.text[:300]}"
        )

    def test_events_existing_project_returns_200(self, litestar_app: object, tmp_path: Path) -> None:
        """Events endpoint returns 200 with empty list when project exists but has no events.

        Args:
            litestar_app: Litestar application fixture.
            tmp_path: Pytest temporary directory used as PROJECT_ROOT.
        """
        project_id = "proj_real_events_test"
        project_dir = tmp_path / "projects" / project_id
        project_dir.mkdir(parents=True)

        with patch("vetinari.web.shared.PROJECT_ROOT", tmp_path):
            with patch(
                "vetinari.web.sse_events.get_recent_sse_events",
                return_value=[],
            ):
                with TestClient(app=litestar_app) as client:
                    response = client.get(f"/api/v1/projects/{project_id}/events")

        assert response.status_code == 200, (
            f"Expected 200 for existing project, got {response.status_code}: {response.text[:300]}"
        )
        body = response.json()
        assert body.get("count") == 0, f"Expected count=0 for empty events, got: {body}"
        assert body.get("events") == [], f"Expected empty events list, got: {body}"
