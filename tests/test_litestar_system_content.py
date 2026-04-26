"""Tests for vetinari.web.litestar_system_content — preferences and settings PUT endpoints.

Covers the malformed-input semantics contract:
- PUT /api/v1/preferences must reject non-object JSON (string, list, number, null) with 400/422
- PUT /api/v1/settings must reject non-object JSON with 400/422
- Empty objects are rejected for both preferences and settings (no keys)
- Valid dict with data is accepted by both handlers

These tests go through the full Litestar HTTP stack via TestClient so that
framework-level validation (content negotiation, body deserialization) is
exercised alongside the handler's explicit isinstance guard.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def litestar_app():
    """Minimal Litestar app with system-content handlers, heavy subsystems patched out.

    Returns:
        A Litestar application instance with shutdown side-effects suppressed.
    """
    with patch("vetinari.web.litestar_app._register_shutdown_handlers"):
        from vetinari.web.litestar_app import create_app

        app = create_app(debug=True)
    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _put_json(client: object, url: str, body: bytes) -> object:
    """Send a PUT request with an explicit Content-Type: application/json header.

    Args:
        client: A Litestar TestClient instance.
        url: The URL path to PUT.
        body: Raw JSON bytes to use as the request body.

    Returns:
        The HTTP response object.
    """
    return client.put(
        url, content=body, headers={"Content-Type": "application/json", "X-Requested-With": "XMLHttpRequest"}
    )


# ---------------------------------------------------------------------------
# PUT /api/v1/preferences
# ---------------------------------------------------------------------------


class TestPreferencesPutValidation:
    """PUT /api/v1/preferences — malformed-input rejection and happy path."""

    @pytest.mark.parametrize(
        "body,label",
        [
            (b'"hello"', "string"),
            (b"[1, 2, 3]", "list"),
            (b"42", "number"),
            (b"null", "null"),
        ],
    )
    def test_non_object_body_rejected(self, litestar_app: object, body: bytes, label: str) -> None:
        """Non-object JSON (string, list, number, null) must be rejected with 400 or 422.

        Litestar's signature model validation rejects non-dict payloads before
        the handler body runs. The handler's explicit isinstance guard provides
        a second line of defence in case the framework coerces the value.

        Args:
            litestar_app: Litestar application fixture.
            body: Raw JSON bytes to send.
            label: Human-readable name for the body type, used in failure messages.
        """
        from litestar.testing import TestClient

        mock_mgr = MagicMock()
        mock_mgr.get_all.return_value = {}
        mock_mgr.set_many.return_value = {}

        with patch("vetinari.web.preferences.get_preferences_manager", return_value=mock_mgr):
            with TestClient(app=litestar_app) as client:
                response = _put_json(client, "/api/v1/preferences", body)

        assert response.status_code in {400, 422}, (
            f"PUT /api/v1/preferences with {label} body returned {response.status_code}, "
            f"expected 400 or 422. Body: {response.text[:300]}"
        )

    def test_empty_object_rejected(self, litestar_app: object) -> None:
        """Empty object {} is rejected because it contains no preference update.

        A no-op PUT is ambiguous and bypasses the endpoint's whitelist semantics;
        callers must provide at least one explicit preference key.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        mock_mgr = MagicMock()
        mock_mgr.set_many.return_value = {}
        mock_mgr.get_all.return_value = {}

        with patch("vetinari.web.preferences.get_preferences_manager", return_value=mock_mgr):
            with TestClient(app=litestar_app) as client:
                response = _put_json(client, "/api/v1/preferences", b"{}")

        assert response.status_code == 422, (
            f"PUT /api/v1/preferences with {{}} returned {response.status_code}, expected 422: "
            f"{response.text[:300]}"
        )

    def test_valid_dict_with_data_succeeds(self, litestar_app: object) -> None:
        """A dict with preference keys must be accepted, applied, and return 200 with preferences map.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        mock_mgr = MagicMock()
        mock_mgr.set_many.return_value = {"theme": True}
        mock_mgr.get_all.return_value = {"theme": "dark"}

        with patch("vetinari.web.preferences.get_preferences_manager", return_value=mock_mgr):
            with TestClient(app=litestar_app) as client:
                response = _put_json(client, "/api/v1/preferences", b'{"theme": "dark"}')

        assert response.status_code == 200, (
            f"PUT /api/v1/preferences with valid dict returned {response.status_code}: {response.text[:300]}"
        )
        data = response.json()
        assert data["preferences"] == {"theme": "dark"}, f"Expected updated preferences payload in response: {data}"


# ---------------------------------------------------------------------------
# PUT /api/v1/settings
# ---------------------------------------------------------------------------


class TestSettingsPutValidation:
    """PUT /api/v1/settings — malformed-input rejection and happy path."""

    @pytest.mark.parametrize(
        "body,label",
        [
            (b'"hello"', "string"),
            (b"[1, 2, 3]", "list"),
            (b"42", "number"),
            (b"null", "null"),
        ],
    )
    def test_non_object_body_rejected(self, litestar_app: object, body: bytes, label: str) -> None:
        """Non-object JSON (string, list, number, null) must be rejected with 400 or 422.

        The handler checks isinstance(data, dict) immediately after entry.
        Litestar's signature validation provides a first layer; this guard
        is the fallback for any framework-level coercion edge-cases.

        Args:
            litestar_app: Litestar application fixture.
            body: Raw JSON bytes to send.
            label: Human-readable name for the body type, used in failure messages.
        """
        from litestar.testing import TestClient

        with TestClient(app=litestar_app) as client:
            response = _put_json(client, "/api/v1/settings", body)

        assert response.status_code in {400, 422}, (
            f"PUT /api/v1/settings with {label} body returned {response.status_code}, "
            f"expected 400 or 422. Body: {response.text[:300]}"
        )

    def test_empty_object_rejected_with_400(self, litestar_app: object) -> None:
        """Empty object {} has no settings to apply — must return 400.

        The settings handler checks ``if not data`` after the isinstance
        guard. An empty dict passes the dict check but fails the non-empty
        check, resulting in a 400 with a descriptive message.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        with TestClient(app=litestar_app) as client:
            response = _put_json(client, "/api/v1/settings", b"{}")

        assert response.status_code == 400, (
            f"PUT /api/v1/settings with {{}} returned {response.status_code}: {response.text[:300]}"
        )
        data = response.json()
        assert data.get("status") == "error", f"Expected error status in response: {data}"

    def test_valid_dict_passes_validation_layer(self, litestar_app: object) -> None:
        """A dict body must not be rejected with 400 or 422 at the validation layer.

        The settings handler writes to ``~/.vetinari/config.yaml``; the write may
        succeed or fail depending on the environment. The key contract is that a
        syntactically valid JSON object is never refused as malformed input.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        mock_settings = MagicMock()
        mock_settings.local_gpu_layers = 0
        mock_settings.local_context_length = 2048
        mock_settings.local_batch_size = 512
        mock_settings.local_flash_attn = False
        mock_settings.local_n_threads = 4

        # Patch the settings singleton calls so the handler doesn't need a real
        # settings file. File I/O is allowed to run against the real filesystem
        # (creating ~/.vetinari/ if absent); if it fails we still get a 500, not 400.
        with (
            patch("vetinari.config.settings.get_settings", return_value=mock_settings),
            patch("vetinari.config.settings.reset_settings"),
        ):
            with TestClient(app=litestar_app) as client:
                response = _put_json(
                    client,
                    "/api/v1/settings",
                    b'{"log_level": "DEBUG"}',
                )

        # A valid dict body must NOT produce a 400 or 422. The handler may
        # return 200 (success) or 500 (filesystem/settings error) but must
        # never indicate that the input was malformed.
        assert response.status_code not in {400, 422}, (
            f"PUT /api/v1/settings with valid dict returned {response.status_code} (rejection code), "
            f"indicating malformed-input validation is wrongly triggering. Body: {response.text[:300]}"
        )
