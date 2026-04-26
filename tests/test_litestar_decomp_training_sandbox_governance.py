"""Mounted request-level governance tests for decomposition, training, and sandbox routes.

Proves that every read and write handler in the decomposition, training, and
sandbox route modules returns bounded HTTP error responses on subsystem failure
rather than leaking raw 500s or masking errors with false-flag 200 ok.

All tests go through the full Litestar TestClient HTTP stack  -  no handler
function is called directly.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ADMIN_TOKEN = "test-gov-admin-token"
_ADMIN_HEADERS = {"X-Admin-Token": _ADMIN_TOKEN}
_CSRF = {"X-Requested-With": "XMLHttpRequest"}


@pytest.fixture(scope="module")
def app():
    """Litestar application with shutdown side-effects suppressed.

    The VETINARI_ADMIN_TOKEN env var must be live at request time (not just at
    app-creation time) because admin_guard reads it on every call.  We use a
    yield fixture so the patch.dict context stays open for the entire module.

    Yields:
        A Litestar application instance ready for test use.
    """
    with (
        patch("vetinari.web.litestar_app._register_shutdown_handlers"),
        patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}),
    ):
        from vetinari.web.litestar_app import create_app

        yield create_app(debug=True)


@pytest.fixture
def client(app):
    """TestClient wrapping the Litestar app.

    Args:
        app: The Litestar application fixture.

    Yields:
        A configured Litestar TestClient.
    """
    from litestar.testing import TestClient

    with TestClient(app) as tc:
        yield tc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_error_response(response: object, expected_code: int) -> None:
    """Assert the response matches the expected error code with a JSON error body.

    Args:
        response: The HTTP response from TestClient.
        expected_code: The expected HTTP status code (e.g. 400, 503).

    Raises:
        AssertionError: When status code or body envelope does not match.
    """
    assert response.status_code == expected_code, (
        f"Expected {expected_code}, got {response.status_code}: {response.text[:300]}"
    )
    data = response.json()
    assert data.get("status") == "error", f"Expected envelope status='error', got: {data}"


def _assert_503(response: object) -> None:
    """Shortcut: assert the response is a well-formed 503 error envelope.

    Args:
        response: The HTTP response from TestClient.
    """
    _assert_error_response(response, 503)


def _assert_400(response: object) -> None:
    """Shortcut: assert the response is a well-formed 400 error envelope.

    Args:
        response: The HTTP response from TestClient.
    """
    _assert_error_response(response, 400)


# ---------------------------------------------------------------------------
# TestDecompositionRoutes  -  GET /api/v1/decomposition/*
# ---------------------------------------------------------------------------


class TestDecompositionRoutes:
    """Tests for the five decomposition read routes."""

    def test_templates_engine_failure_returns_503(self, client: object) -> None:
        """GET /api/v1/decomposition/templates when engine.get_templates raises returns 503.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.planning.decomposition.decomposition_engine.get_templates",
            side_effect=RuntimeError("decomposition engine down"),
        ):
            response = client.get("/api/v1/decomposition/templates")
        _assert_503(response)

    def test_dod_dor_engine_failure_returns_503(self, client: object) -> None:
        """GET /api/v1/decomposition/dod-dor when engine.get_dod_criteria raises returns 503.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.planning.decomposition.decomposition_engine.get_dod_criteria",
            side_effect=RuntimeError("decomposition engine down"),
        ):
            response = client.get("/api/v1/decomposition/dod-dor")
        _assert_503(response)

    def test_history_engine_failure_returns_503(self, client: object) -> None:
        """GET /api/v1/decomposition/history when engine.get_decomposition_history raises returns 503.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.planning.decomposition.decomposition_engine.get_decomposition_history",
            side_effect=RuntimeError("decomposition engine down"),
        ):
            response = client.get("/api/v1/decomposition/history")
        _assert_503(response)

    def test_seed_config_engine_failure_returns_503(self, client: object) -> None:
        """GET /api/v1/decomposition/seed-config when engine attribute access raises returns 503.

        The seed-config handler reads SEED_MIX as a plain attribute.  We use
        PropertyMock on the MagicMock's type so that descriptor protocol fires
        and raises RuntimeError when SEED_MIX is accessed.

        Args:
            client: The TestClient fixture.
        """
        broken_engine = MagicMock()
        type(broken_engine).SEED_MIX = PropertyMock(side_effect=RuntimeError("engine unavailable"))
        with patch(
            "vetinari.planning.decomposition.decomposition_engine",
            broken_engine,
        ):
            response = client.get("/api/v1/decomposition/seed-config")
        _assert_503(response)

    def test_knobs_failure_returns_503(self, client: object) -> None:
        """When the decomposition agent constants are unavailable, /knobs returns 503.

        Args:
            client: The TestClient fixture.
        """
        with patch.dict(
            "sys.modules",
            {"vetinari.agents.decomposition_agent": None},
        ):
            response = client.get("/api/v1/decomposition/knobs")
        _assert_503(response)


# ---------------------------------------------------------------------------
# TestTrainingRoutes  -  /api/training/* (litestar_training_routes.py)
# ---------------------------------------------------------------------------


class TestTrainingRoutes:
    """Tests for the training routes defined in litestar_training_routes.py."""

    def test_export_invalid_format_returns_400(self, client: object) -> None:
        """POST /api/training/export with unknown format returns 400.

        Args:
            client: The TestClient fixture.
        """
        response = client.post(
            "/api/training/export",
            json={"format": "invalid_format"},
            headers={**_ADMIN_HEADERS, **_CSRF},
        )
        _assert_400(response)

    def test_export_backend_failure_returns_503(self, client: object) -> None:
        """POST /api/training/export when collector raises returns 503.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.learning.training_data.get_training_collector",
            side_effect=RuntimeError("collector down"),
        ):
            response = client.post(
                "/api/training/export",
                json={"format": "sft"},
                headers={**_ADMIN_HEADERS, **_CSRF},
            )
        _assert_503(response)

    def test_training_start_invalid_tier_returns_400(self, client: object) -> None:
        """POST /api/training/start with unknown tier returns 400.

        Args:
            client: The TestClient fixture.
        """
        response = client.post(
            "/api/training/start",
            json={"tier": "not_a_real_tier"},
            headers={**_ADMIN_HEADERS, **_CSRF},
        )
        _assert_400(response)

    @pytest.mark.parametrize(
        "tier",
        ["general", "coding", "research", "review", "individual"],
    )
    def test_training_start_valid_tiers_do_not_400(self, client: object, tier: str) -> None:
        """POST /api/training/start with a valid tier does not return 400.

        A 400 here means tier validation rejected a valid value  -  that is a
        regression. The route may still return 503 if prerequisites are unmet,
        but it must not reject valid tier values.

        Args:
            client: The TestClient fixture.
            tier: A valid tier string to test.
        """
        mock_pipeline = MagicMock()
        mock_pipeline.check_requirements.return_value = {"ready_for_training": False, "missing_libraries": ["unsloth"]}

        with patch("vetinari.training.pipeline.TrainingPipeline", return_value=mock_pipeline):
            response = client.post(
                "/api/training/start",
                json={"tier": tier},
                headers={**_ADMIN_HEADERS, **_CSRF},
            )
        # Must not be a 400 (validation rejection)  -  503 is acceptable
        assert response.status_code != 400, f"Valid tier '{tier}' was rejected with 400: {response.text[:200]}"


# ---------------------------------------------------------------------------
# TestTrainingApiPart2  -  /api/v1/training/* (litestar_training_api_part2.py)
# ---------------------------------------------------------------------------


class TestTrainingApiPart2:
    """Tests for the training lifecycle and data routes in litestar_training_api_part2.py."""

    def test_pause_scheduler_failure_returns_500(self, client: object) -> None:
        """POST /api/v1/training/pause when scheduler raises returns 500.

        The route must NOT return {"status": "ok"}  -  that is a false-flag
        success. It must surface the failure as a 500 error.

        Patches the TrainingScheduler class method directly because
        ``_get_scheduler`` is captured as a closure in the factory at
        module-load time  -  patching the module-level name after load has no
        effect on the closure variable.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.training.idle_scheduler.TrainingScheduler.pause_for_user_request",
            side_effect=RuntimeError("scheduler error"),
        ):
            response = client.post(
                "/api/v1/training/pause",
                headers={**_ADMIN_HEADERS, **_CSRF},
            )
        assert response.status_code == 500, (
            f"Expected 500 on scheduler failure, got {response.status_code}: {response.text[:200]}"
        )
        data = response.json()
        assert data.get("status") == "error", f"Expected error status, got: {data}"

    def test_resume_scheduler_failure_returns_500(self, client: object) -> None:
        """POST /api/v1/training/resume when scheduler raises returns 500.

        Patches the TrainingScheduler class method directly  -  same reason as
        ``test_pause_scheduler_failure_returns_500``.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.training.idle_scheduler.TrainingScheduler.resume_after_user_request",
            side_effect=RuntimeError("scheduler error"),
        ):
            response = client.post(
                "/api/v1/training/resume",
                headers={**_ADMIN_HEADERS, **_CSRF},
            )
        assert response.status_code == 500, (
            f"Expected 500 on scheduler failure, got {response.status_code}: {response.text[:200]}"
        )
        data = response.json()
        assert data.get("status") == "error", f"Expected error status, got: {data}"

    def test_stop_scheduler_failure_returns_500(self, client: object) -> None:
        """POST /api/v1/training/stop when scheduler raises returns 500.

        ``stop`` delegates to ``pause_for_user_request()``.  Patches the class
        method directly for the same closure-capture reason as pause.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.training.idle_scheduler.TrainingScheduler.pause_for_user_request",
            side_effect=RuntimeError("scheduler error"),
        ):
            response = client.post(
                "/api/v1/training/stop",
                headers={**_ADMIN_HEADERS, **_CSRF},
            )
        assert response.status_code == 500, (
            f"Expected 500 on scheduler failure, got {response.status_code}: {response.text[:200]}"
        )
        data = response.json()
        assert data.get("status") == "error", f"Expected error status, got: {data}"

    def test_pause_success_returns_ok(self, client: object) -> None:
        """POST /api/v1/training/pause when scheduler succeeds returns ok with paused=True.

        Litestar ``@post`` handlers default to 201 Created when returning a
        plain dict, so we assert any 2xx status rather than exactly 200.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.training.idle_scheduler.TrainingScheduler.pause_for_user_request",
            return_value=None,
        ):
            response = client.post(
                "/api/v1/training/pause",
                headers={**_ADMIN_HEADERS, **_CSRF},
            )
        assert response.status_code < 300, f"Expected 2xx success, got {response.status_code}: {response.text[:200]}"
        data = response.json()
        assert data.get("status") == "ok"
        assert data.get("paused") is True

    def test_data_seed_backend_failure_returns_503(self, client: object) -> None:
        """POST /api/v1/training/data/seed when seeder raises returns 503.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.training.data_seeder.get_training_data_seeder",
            side_effect=RuntimeError("seeder down"),
        ):
            response = client.post(
                "/api/v1/training/data/seed",
                headers={**_ADMIN_HEADERS, **_CSRF},
            )
        _assert_503(response)

    def test_seed_stream_backend_failure_returns_503(self, client: object) -> None:
        """GET /api/v1/training/data/seed/stream when seeder unavailable returns 503.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.training.data_seeder.get_training_data_seeder",
            side_effect=RuntimeError("seeder down"),
        ):
            response = client.get("/api/v1/training/data/seed/stream")
        _assert_503(response)


# ---------------------------------------------------------------------------
# TestTrainingExperimentsApi  -  /api/v1/training/progress/stream
# ---------------------------------------------------------------------------


class TestTrainingExperimentsApi:
    """Tests for the training experiments SSE route."""

    def test_progress_stream_scheduler_failure_returns_503(self, client: object) -> None:
        """GET /api/v1/training/progress/stream when scheduler probe raises returns 503.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.web.litestar_training_api._is_scheduler_training",
            side_effect=RuntimeError("scheduler broken"),
        ):
            response = client.get("/api/v1/training/progress/stream")
        _assert_503(response)


# ---------------------------------------------------------------------------
# TestSandboxRoutes  -  /api/sandbox/*
# ---------------------------------------------------------------------------


class TestSandboxExecute:
    """Tests for POST /api/sandbox/execute."""

    def test_missing_code_returns_400(self, client: object) -> None:
        """POST /api/sandbox/execute without code field returns 400.

        Args:
            client: The TestClient fixture.
        """
        response = client.post(
            "/api/sandbox/execute",
            json={},
            headers={**_ADMIN_HEADERS, **_CSRF},
        )
        _assert_400(response)

    def test_non_string_code_returns_400(self, client: object) -> None:
        """POST /api/sandbox/execute with non-string code returns 400.

        Args:
            client: The TestClient fixture.
        """
        response = client.post(
            "/api/sandbox/execute",
            json={"code": 12345},
            headers={**_ADMIN_HEADERS, **_CSRF},
        )
        _assert_400(response)

    def test_non_numeric_timeout_returns_400(self, client: object) -> None:
        """POST /api/sandbox/execute with non-numeric timeout returns 400.

        Args:
            client: The TestClient fixture.
        """
        response = client.post(
            "/api/sandbox/execute",
            json={"code": "print('hi')", "timeout": "fast"},
            headers={**_ADMIN_HEADERS, **_CSRF},
        )
        _assert_400(response)

    def test_non_dict_context_returns_400(self, client: object) -> None:
        """POST /api/sandbox/execute with non-dict context returns 400.

        Args:
            client: The TestClient fixture.
        """
        response = client.post(
            "/api/sandbox/execute",
            json={"code": "print('hi')", "context": ["not", "a", "dict"]},
            headers={**_ADMIN_HEADERS, **_CSRF},
        )
        _assert_400(response)

    def test_backend_failure_returns_503(self, client: object) -> None:
        """POST /api/sandbox/execute when sandbox_manager.execute raises returns 503.

        The handler imports ``sandbox_manager`` from the module inside the try
        block, then calls ``.execute()`` on the singleton object.  Patching the
        singleton's specific method (not the module attribute itself) is
        required because ``side_effect`` on the attribute makes *calling* the
        attribute raise, not calling a method on it.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.sandbox_manager.sandbox_manager.execute",
            side_effect=RuntimeError("sandbox manager down"),
        ):
            response = client.post(
                "/api/sandbox/execute",
                json={"code": "print('hi')"},
                headers={**_ADMIN_HEADERS, **_CSRF},
            )
        _assert_503(response)


class TestSandboxStatus:
    """Tests for GET /api/sandbox/status."""

    def test_backend_failure_returns_503(self, client: object) -> None:
        """GET /api/sandbox/status when sandbox_manager.get_status raises returns 503.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.sandbox_manager.sandbox_manager.get_status",
            side_effect=RuntimeError("sandbox manager down"),
        ):
            response = client.get(
                "/api/sandbox/status",
                headers=_ADMIN_HEADERS,
            )
        _assert_503(response)


class TestSandboxAudit:
    """Tests for GET /api/sandbox/audit."""

    def test_backend_failure_returns_503(self, client: object) -> None:
        """GET /api/sandbox/audit when sandbox_manager.get_audit_log raises returns 503.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.sandbox_manager.sandbox_manager.get_audit_log",
            side_effect=RuntimeError("sandbox manager down"),
        ):
            response = client.get(
                "/api/sandbox/audit",
                headers=_ADMIN_HEADERS,
            )
        _assert_503(response)


class TestSandboxPlugins:
    """Tests for GET /api/sandbox/plugins."""

    def test_backend_failure_returns_503(self, client: object) -> None:
        """GET /api/sandbox/plugins when ExternalPluginSandbox raises returns 503.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.sandbox_policy.ExternalPluginSandbox",
            side_effect=RuntimeError("plugin sandbox down"),
        ):
            response = client.get(
                "/api/sandbox/plugins",
                headers=_ADMIN_HEADERS,
            )
        _assert_503(response)


class TestSandboxPluginHook:
    """Tests for POST /api/sandbox/plugins/hook."""

    def test_missing_plugin_name_returns_400(self, client: object) -> None:
        """POST /api/sandbox/plugins/hook without plugin_name returns 400.

        Args:
            client: The TestClient fixture.
        """
        response = client.post(
            "/api/sandbox/plugins/hook",
            json={"hook_name": "on_load"},
            headers={**_ADMIN_HEADERS, **_CSRF},
        )
        _assert_400(response)

    def test_missing_hook_name_returns_400(self, client: object) -> None:
        """POST /api/sandbox/plugins/hook without hook_name returns 400.

        Args:
            client: The TestClient fixture.
        """
        response = client.post(
            "/api/sandbox/plugins/hook",
            json={"plugin_name": "my_plugin"},
            headers={**_ADMIN_HEADERS, **_CSRF},
        )
        _assert_400(response)

    def test_non_string_plugin_name_returns_400(self, client: object) -> None:
        """POST /api/sandbox/plugins/hook with non-string plugin_name returns 400.

        Args:
            client: The TestClient fixture.
        """
        response = client.post(
            "/api/sandbox/plugins/hook",
            json={"plugin_name": 42, "hook_name": "on_load"},
            headers={**_ADMIN_HEADERS, **_CSRF},
        )
        _assert_400(response)

    def test_non_string_hook_name_returns_400(self, client: object) -> None:
        """POST /api/sandbox/plugins/hook with non-string hook_name returns 400.

        Args:
            client: The TestClient fixture.
        """
        response = client.post(
            "/api/sandbox/plugins/hook",
            json={"plugin_name": "my_plugin", "hook_name": 99},
            headers={**_ADMIN_HEADERS, **_CSRF},
        )
        _assert_400(response)

    def test_backend_failure_returns_503(self, client: object) -> None:
        """POST /api/sandbox/plugins/hook when ExternalPluginSandbox raises returns 503.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.sandbox_policy.ExternalPluginSandbox",
            side_effect=RuntimeError("plugin sandbox down"),
        ):
            response = client.post(
                "/api/sandbox/plugins/hook",
                json={"plugin_name": "my_plugin", "hook_name": "on_load"},
                headers={**_ADMIN_HEADERS, **_CSRF},
            )
        _assert_503(response)
