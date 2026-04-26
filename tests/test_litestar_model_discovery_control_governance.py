"""Mounted request-level governance tests for model-discovery control routes.

Proves model-discovery POST routes reject empty/no-op bodies, handle cache and
filesystem failures with bounded error responses, and do not return ranked
success for empty scoring requests.
"""

from __future__ import annotations

import os
import pathlib
from unittest.mock import patch

import pytest

# Admin token used for all admin-guarded POST requests.
_ADMIN_TOKEN = "test-model-discovery-governance"

# POST mutations require X-Requested-With for CSRF and an admin token for admin_guard.
_MUTATION_HEADERS = {
    "X-Admin-Token": _ADMIN_TOKEN,
    "X-Requested-With": "XMLHttpRequest",
}


# -- Fixtures --


@pytest.fixture
def app():
    """Litestar app built per test, with shutdown side-effects suppressed.

    Sets ``VETINARI_ADMIN_TOKEN`` so the admin_guard accepts requests that carry
    the matching ``X-Admin-Token`` header.

    This is function-scoped (not module-scoped) so that test patches on
    _get_models_cached and _infer_recommended_tasks take effect before the
    handlers are created. Handlers capture their imports at creation time,
    so the app must be rebuilt for each test that needs custom mocks.

    Returns:
        A Litestar application instance ready for test use.
    """
    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        with patch("vetinari.web.litestar_app._register_shutdown_handlers"):
            from vetinari.web.litestar_app import create_app

            return create_app(debug=True)


@pytest.fixture
def client(app):
    """TestClient bound to the Litestar app, with admin token active.

    Args:
        app: The Litestar application fixture.

    Yields:
        A configured Litestar TestClient.
    """
    from litestar.testing import TestClient

    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        with TestClient(app) as tc:
            yield tc


# -- POST /api/v1/models/refresh --


class TestApiModelsRefresh:
    """Governance tests for POST /api/v1/models/refresh."""

    def test_cache_failure_returns_503(self, client):
        """Discovery service RuntimeError must produce a bounded 503, not a raw 500."""
        with patch(
            "vetinari.web.litestar_models_discovery._get_models_cached",
            side_effect=RuntimeError("discovery backend down"),
        ):
            response = client.post(
                "/api/v1/models/refresh",
                headers=_MUTATION_HEADERS,
            )

        assert response.status_code == 503

    def test_valid_refresh_succeeds(self, client):
        """Successful refresh returns models list with cached=False and correct count."""
        mock_models = [{"name": "test-model", "capabilities": [], "memory_gb": 4}]
        with patch(
            "vetinari.web.litestar_models_discovery._get_models_cached",
            return_value=mock_models,
        ):
            response = client.post(
                "/api/v1/models/refresh",
                headers=_MUTATION_HEADERS,
            )

        assert response.status_code == 201
        body = response.json()
        assert body["cached"] is False
        assert body["count"] == 1
        assert len(body["models"]) == 1
        assert body["models"][0]["name"] == "test-model"


# -- POST /api/v1/score-models --


class TestApiScoreModels:
    """Governance tests for POST /api/v1/score-models."""

    def test_empty_body_returns_400(self, client):
        """Empty body must be rejected  -  indistinguishable from a real task otherwise."""
        response = client.post(
            "/api/v1/score-models",
            json={},
            headers=_MUTATION_HEADERS,
        )
        assert response.status_code == 400

    def test_empty_task_description_returns_400(self, client):
        """Empty string for task_description must be rejected with 400."""
        response = client.post(
            "/api/v1/score-models",
            json={"task_description": ""},
            headers=_MUTATION_HEADERS,
        )
        assert response.status_code == 400

    def test_cache_failure_returns_503(self, client):
        """Discovery failure during scoring must return bounded 503, not raw 500."""
        with patch(
            "vetinari.web.litestar_models_discovery._get_models_cached",
            side_effect=RuntimeError("discovery backend down"),
        ):
            response = client.post(
                "/api/v1/score-models",
                json={"task_description": "build a web api"},
                headers=_MUTATION_HEADERS,
            )

        assert response.status_code == 503

    def test_valid_scoring_succeeds(self, client):
        """Valid task description scores available models and returns ranked list."""
        mock_models = [
            {"name": "coder-model", "capabilities": ["code_gen"], "memory_gb": 4},
            {"name": "chat-model", "capabilities": ["chat"], "memory_gb": 2},
        ]
        with patch("vetinari.web.litestar_models_discovery._get_models_cached", return_value=mock_models):
            with patch(
                "vetinari.web.litestar_models_discovery._infer_recommended_tasks",
                return_value=["code generation"],
            ):
                response = client.post(
                    "/api/v1/score-models",
                    json={"task_description": "build a web api"},
                    headers=_MUTATION_HEADERS,
                )

        assert response.status_code == 201
        body = response.json()
        assert "models" in body
        scored = body["models"]
        assert len(scored) == 2
        # coder-model ranks first  -  "api" and "web" match code_gen capability
        assert scored[0]["name"] == "coder-model"
        assert "score" in scored[0]
        assert "matches" in scored[0]


# -- POST /api/v1/model-config --


class TestApiUpdateModelConfig:
    """Governance tests for POST /api/v1/model-config."""

    def test_empty_body_returns_400(self, client):
        """Empty body must be rejected  -  silent no-op is indistinguishable from update."""
        response = client.post(
            "/api/v1/model-config",
            json={},
            headers=_MUTATION_HEADERS,
        )
        assert response.status_code == 400

    def test_unrecognized_keys_returns_400(self, client):
        """Body with only unknown keys must be rejected rather than silently ignored."""
        response = client.post(
            "/api/v1/model-config",
            json={"foo": "bar"},
            headers=_MUTATION_HEADERS,
        )
        assert response.status_code == 400

    def test_valid_update_succeeds(self, client):
        """Recognised key updates config and returns status='updated' with new values."""
        import vetinari.web.shared as shared

        original = shared.current_config.memory_budget_gb
        try:
            response = client.post(
                "/api/v1/model-config",
                json={"memory_budget_gb": 32},
                headers=_MUTATION_HEADERS,
            )
            assert response.status_code == 201
            body = response.json()
            assert body["status"] == "updated"
            assert body["memory_budget_gb"] == 32
        finally:
            # Restore original value so other tests are not affected.
            shared.current_config.memory_budget_gb = original


# -- POST /api/v1/swap-model --


class TestApiSwapModel:
    """Governance tests for POST /api/v1/swap-model."""

    def test_missing_model_id_returns_400(self, client):
        """Requests with no model_id must be rejected with 400."""
        response = client.post(
            "/api/v1/swap-model",
            json={},
            headers=_MUTATION_HEADERS,
        )
        assert response.status_code == 400

    def test_project_yaml_failure_returns_503(self, client):
        """OSError during project.yaml write must produce bounded 503, not raw 500.

        ``PROJECT_ROOT`` is captured at factory-build time as a bound Path value
        in the handler's closure and cannot be redirected via module-level patching
        after the app is built.  Instead we create the project directory under the
        real PROJECT_ROOT and patch ``pathlib.Path.open`` to raise on write mode.
        """
        from vetinari.web.shared import PROJECT_ROOT

        project_id = "test-swap-503-project"
        project_dir = PROJECT_ROOT / "projects" / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Simulate an OSError on the yaml write by patching Path.open to
            # raise only when called in write mode.
            real_open = pathlib.Path.open

            def _failing_open(self, mode="r", **kwargs):
                if "w" in mode:
                    raise OSError("disk full")
                return real_open(self, mode, **kwargs)

            with patch.object(pathlib.Path, "open", _failing_open):
                response = client.post(
                    "/api/v1/swap-model",
                    json={"model_id": "new-model", "project_id": project_id},
                    headers=_MUTATION_HEADERS,
                )
        finally:
            # Clean up the temporary project directory created under PROJECT_ROOT.
            import shutil

            shutil.rmtree(project_dir, ignore_errors=True)

        assert response.status_code == 503

    def test_valid_global_swap_succeeds(self, client):
        """Global swap (no project_id) sets active_model_id and returns status='swapped'."""
        import vetinari.web.shared as shared

        original = getattr(shared.current_config, "active_model_id", None)
        try:
            response = client.post(
                "/api/v1/swap-model",
                json={"model_id": "test-model"},
                headers=_MUTATION_HEADERS,
            )
            assert response.status_code == 201
            body = response.json()
            assert body["status"] == "swapped"
            assert body["active_model_id"] == "test-model"
        finally:
            shared.current_config.active_model_id = original
