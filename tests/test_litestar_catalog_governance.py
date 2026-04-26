"""Governance tests for catalog, template, and model-discovery read routes.

Proves that all catalog and discovery GET routes return bounded 503 on
subsystem failure -- not raw 500, not a degraded 200 that hides the error
from the caller.

Special assertions:
- ``test_model_get_relay_unavailable_returns_503``: proves relay failure gives
  503, while a relay-up "not found" still gives 404.
- ``test_models_popular_unavailable_returns_503_not_degraded_200``: proves the
  popular endpoint no longer silently returns ``{"models":[]}`` on failure.
- ``test_model_files_unavailable_returns_503_not_degraded_200``: same for
  the files endpoint.

All tests go through the full Litestar HTTP stack via TestClient so that
framework-level serialization and exception handlers are exercised.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Skip the whole module when Litestar is not installed.


# ---------------------------------------------------------------------------
# App / client fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    """Litestar app with shutdown side-effects suppressed.

    Scoped to module so the Litestar app object is only built once; each test
    creates its own TestClient context so connection state does not leak.

    Returns:
        A Litestar application instance.
    """
    with patch("vetinari.web.litestar_app._register_shutdown_handlers"):
        from vetinari.web.litestar_app import create_app

        return create_app(debug=True)


@pytest.fixture
def client(app):
    """TestClient bound to the shared Litestar app.

    Yields:
        A live TestClient for the duration of one test.
    """
    from litestar.testing import TestClient

    with TestClient(app) as tc:
        yield tc


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _assert_503_error(response: object) -> None:
    """Assert that *response* is a bounded 503 with ``status: error`` envelope.

    Args:
        response: HTTP response from the TestClient.
    """
    assert response.status_code == 503, f"Expected 503, got {response.status_code}. Body: {response.text[:400]}"
    body = response.json()
    assert body.get("status") == "error", f"Expected envelope status='error', got {body.get('status')!r}. Body: {body}"


# ---------------------------------------------------------------------------
# Model catalog routes
# ---------------------------------------------------------------------------


class TestModelCatalogList:
    """GET /api/v1/model-catalog  -  503 when model_relay unavailable."""

    def test_relay_unavailable_returns_503(self, client: object) -> None:
        """Model catalog returns 503 when model_relay module cannot be loaded.

        The handler uses a local import ``from vetinari.models.model_relay import
        model_relay`` on each request.  Injecting ``None`` into sys.modules for that
        path causes an ``ImportError`` at call time, which the try/except gate must
        convert to a bounded 503 rather than a raw 500 or a degraded 200.
        """
        import sys

        orig = sys.modules.get("vetinari.models.model_relay")
        sys.modules["vetinari.models.model_relay"] = None  # type: ignore[assignment]
        try:
            resp = client.get("/api/v1/model-catalog")
        finally:
            if orig is None:
                sys.modules.pop("vetinari.models.model_relay", None)
            else:
                sys.modules["vetinari.models.model_relay"] = orig
        # The handler wraps the relay call in try/except  -  must return 503
        _assert_503_error(resp)

    def test_relay_unavailable_via_import_error(self, client: object) -> None:
        """Model catalog returns 503 when model_relay raises on access.

        Uses ``patch.dict`` to temporarily replace the relay module with a mock
        whose ``model_relay`` attribute raises ``RuntimeError``, simulating a relay
        that is present but broken (e.g. database connection lost at startup).
        """
        from unittest.mock import MagicMock

        broken_relay = MagicMock()
        broken_relay.get_all_models.side_effect = RuntimeError("relay init failed")
        mock_module = MagicMock()
        mock_module.model_relay = broken_relay
        with patch.dict("sys.modules", {"vetinari.models.model_relay": mock_module}):
            resp = client.get("/api/v1/model-catalog")
        _assert_503_error(resp)


class TestModelGet:
    """GET /api/v1/models/{model_id}  -  503 on relay failure, 404 on model not found."""

    def test_relay_unavailable_returns_503(self, client: object) -> None:
        """Single model lookup returns 503 when relay module cannot be loaded."""
        import sys

        orig = sys.modules.get("vetinari.models.model_relay")
        sys.modules["vetinari.models.model_relay"] = None  # type: ignore[assignment]
        try:
            resp = client.get("/api/v1/models/some-model-id")
        finally:
            if orig is None:
                sys.modules.pop("vetinari.models.model_relay", None)
            else:
                sys.modules["vetinari.models.model_relay"] = orig
        _assert_503_error(resp)

    def test_model_not_found_returns_404(self, client: object) -> None:
        """Single model lookup returns 404 when relay is up but model absent."""
        mock_relay = MagicMock()
        mock_relay.get_model.return_value = None  # model not found
        with patch.dict(
            "sys.modules",
            {
                "vetinari.models.model_relay": MagicMock(model_relay=mock_relay),
            },
        ):
            resp = client.get("/api/v1/models/nonexistent-model")
        assert resp.status_code == 404, f"Expected 404, got {resp.status_code}. Body: {resp.text[:400]}"

    def test_relay_raises_returns_503(self, client: object) -> None:
        """Single model lookup returns 503 when relay.get_model raises."""
        mock_relay = MagicMock()
        mock_relay.get_model.side_effect = RuntimeError("db connection lost")
        with patch.dict(
            "sys.modules",
            {
                "vetinari.models.model_relay": MagicMock(model_relay=mock_relay),
            },
        ):
            resp = client.get("/api/v1/models/any-model-id")
        _assert_503_error(resp)


class TestModelPolicyGet:
    """GET /api/v1/models/policy  -  503 when model_relay unavailable."""

    def test_relay_unavailable_returns_503(self, client: object) -> None:
        """Routing policy GET returns 503 when relay module cannot be loaded."""
        import sys

        orig = sys.modules.get("vetinari.models.model_relay")
        sys.modules["vetinari.models.model_relay"] = None  # type: ignore[assignment]
        try:
            resp = client.get("/api/v1/models/policy")
        finally:
            if orig is None:
                sys.modules.pop("vetinari.models.model_relay", None)
            else:
                sys.modules["vetinari.models.model_relay"] = orig
        _assert_503_error(resp)

    def test_relay_raises_returns_503(self, client: object) -> None:
        """Routing policy GET returns 503 when relay.get_policy raises."""
        mock_relay = MagicMock()
        mock_relay.get_policy.side_effect = RuntimeError("relay not initialised")
        with patch.dict(
            "sys.modules",
            {
                "vetinari.models.model_relay": MagicMock(model_relay=mock_relay),
            },
        ):
            resp = client.get("/api/v1/models/policy")
        _assert_503_error(resp)


class TestModelsPopular:
    """GET /api/v1/models/popular  -  503 on failure, not degraded 200."""

    def test_relay_unavailable_returns_503_not_degraded_200(self, client: object) -> None:
        """Popular models returns 503 when relay unavailable  -  not empty 200.

        Guards against the 'unavailable-dependency pass-through' anti-pattern
        where an unavailable subsystem returns ``{"models":[]}`` with status 200,
        hiding the infrastructure failure from the caller.
        """
        import sys

        orig = sys.modules.get("vetinari.models.model_relay")
        sys.modules["vetinari.models.model_relay"] = None  # type: ignore[assignment]
        try:
            resp = client.get("/api/v1/models/popular")
        finally:
            if orig is None:
                sys.modules.pop("vetinari.models.model_relay", None)
            else:
                sys.modules["vetinari.models.model_relay"] = orig
        # MUST be 503, not 200 with empty models list
        _assert_503_error(resp)
        body = resp.json()
        assert "models" not in body or body.get("status") == "error", (
            "Response must not silently return an empty models list on relay failure"
        )

    def test_relay_raises_returns_503(self, client: object) -> None:
        """Popular models returns 503 when relay.get_all_models raises."""
        mock_relay = MagicMock()
        mock_relay.get_all_models.side_effect = RuntimeError("catalog corrupt")
        with patch.dict(
            "sys.modules",
            {
                "vetinari.models.model_relay": MagicMock(model_relay=mock_relay),
            },
        ):
            resp = client.get("/api/v1/models/popular")
        _assert_503_error(resp)


class TestModelFiles:
    """GET /api/v1/models/files  -  503 on failure, not degraded 200."""

    def test_missing_repo_id_returns_400(self, client: object) -> None:
        """Model files returns 400 when repo_id query param is absent."""
        resp = client.get("/api/v1/models/files")
        assert resp.status_code == 400, f"Expected 400, got {resp.status_code}. Body: {resp.text[:400]}"

    def test_discovery_unavailable_returns_503_not_degraded_200(self, client: object) -> None:
        """Model files returns 503 on discovery failure  -  not empty files 200.

        Guards against the 'unavailable-dependency pass-through' anti-pattern
        where ``{"files":[], "repo_id":"..."}`` with status 200 hides the failure.
        """
        import sys

        orig = sys.modules.get("vetinari.model_discovery")
        sys.modules["vetinari.model_discovery"] = None  # type: ignore[assignment]
        try:
            resp = client.get("/api/v1/models/files?repo_id=test/repo")
        finally:
            if orig is None:
                sys.modules.pop("vetinari.model_discovery", None)
            else:
                sys.modules["vetinari.model_discovery"] = orig
        # MUST be 503, not 200 with empty files list
        _assert_503_error(resp)
        body = resp.json()
        assert "files" not in body or body.get("status") == "error", (
            "Response must not silently return an empty files list on discovery failure"
        )

    def test_get_repo_files_raises_returns_503(self, client: object) -> None:
        """Model files returns 503 when ModelDiscovery.get_repo_files raises."""
        mock_discovery = MagicMock()
        mock_discovery.get_repo_files.side_effect = ConnectionError("HuggingFace unreachable")
        mock_module = MagicMock()
        mock_module.ModelDiscovery.return_value = mock_discovery
        with patch.dict("sys.modules", {"vetinari.model_discovery": mock_module}):
            resp = client.get("/api/v1/models/files?repo_id=author/model")
        _assert_503_error(resp)


# ---------------------------------------------------------------------------
# Template routes (in litestar_subtasks_api.py)
# ---------------------------------------------------------------------------


class TestTemplateVersions:
    """GET /api/v1/templates/versions  -  503 when template loader unavailable."""

    def test_loader_unavailable_returns_503(self, client: object) -> None:
        """Template versions returns 503 when template_loader cannot be loaded."""
        import sys

        orig = sys.modules.get("vetinari.template_loader")
        sys.modules["vetinari.template_loader"] = None  # type: ignore[assignment]
        try:
            resp = client.get("/api/v1/templates/versions")
        finally:
            if orig is None:
                sys.modules.pop("vetinari.template_loader", None)
            else:
                sys.modules["vetinari.template_loader"] = orig
        _assert_503_error(resp)

    def test_loader_raises_returns_503(self, client: object) -> None:
        """Template versions returns 503 when list_versions raises."""
        mock_loader = MagicMock()
        mock_loader.list_versions.side_effect = OSError("templates dir missing")
        mock_module = MagicMock()
        mock_module.template_loader = mock_loader
        with patch.dict("sys.modules", {"vetinari.template_loader": mock_module}):
            resp = client.get("/api/v1/templates/versions")
        _assert_503_error(resp)


class TestTemplates:
    """GET /api/v1/templates  -  503 when template loader unavailable."""

    def test_loader_unavailable_returns_503(self, client: object) -> None:
        """Templates list returns 503 when template_loader cannot be loaded."""
        import sys

        orig = sys.modules.get("vetinari.template_loader")
        sys.modules["vetinari.template_loader"] = None  # type: ignore[assignment]
        try:
            resp = client.get("/api/v1/templates")
        finally:
            if orig is None:
                sys.modules.pop("vetinari.template_loader", None)
            else:
                sys.modules["vetinari.template_loader"] = orig
        _assert_503_error(resp)

    def test_loader_raises_returns_503(self, client: object) -> None:
        """Templates list returns 503 when load_templates raises.

        The ``/api/v1/templates`` handler calls ``template_loader.load_templates()``.
        Injecting a mock whose ``load_templates`` raises ``OSError`` proves the
        try/except gate converts the failure to a bounded 503.
        """
        mock_loader = MagicMock()
        mock_loader.load_templates.side_effect = OSError("templates dir missing")
        mock_module = MagicMock()
        mock_module.template_loader = mock_loader
        with patch.dict("sys.modules", {"vetinari.template_loader": mock_module}):
            resp = client.get("/api/v1/templates")
        _assert_503_error(resp)


# ---------------------------------------------------------------------------
# Cost-analysis route (in litestar_cost_analysis_api.py)
# ---------------------------------------------------------------------------


class TestCostAnalysis:
    """GET /api/v1/cost-analysis  -  503 when checkpoint store unavailable."""

    # The cost-analysis endpoint is admin-guarded.  Pass the token so the guard
    # does not reject the request before the handler runs.
    _ADMIN_TOKEN: str = "test-admin-token-catalog-governance"
    _ADMIN_HEADERS: dict[str, str] = {"X-Admin-Token": _ADMIN_TOKEN}

    def test_checkpoint_store_unavailable_returns_503(self, client: object) -> None:
        """Cost analysis returns 503 when the checkpoint store cannot be reached.

        The handler imports ``get_checkpoint_store`` from
        ``vetinari.observability.checkpoints``.  Injecting ``None`` for that
        module path causes an ``ImportError`` at call time, which the handler's
        try/except gate must convert to a bounded 503 with an error envelope.
        """
        import os
        import sys

        orig = sys.modules.get("vetinari.observability.checkpoints")
        sys.modules["vetinari.observability.checkpoints"] = None  # type: ignore[assignment]
        try:
            with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": self._ADMIN_TOKEN}):
                resp = client.get("/api/v1/cost-analysis", headers=self._ADMIN_HEADERS)
        finally:
            if orig is None:
                sys.modules.pop("vetinari.observability.checkpoints", None)
            else:
                sys.modules["vetinari.observability.checkpoints"] = orig
        _assert_503_error(resp)

    def test_cost_analysis_failure_not_raw_500(self, client: object) -> None:
        """Cost analysis wraps internal errors in a 503  -  never exposes raw 500.

        Makes the checkpoint store unavailable by injecting ``None`` into
        ``sys.modules`` for ``vetinari.observability.checkpoints``.  The handler's
        try/except gate must produce a bounded 503 response with an error envelope
        rather than letting Litestar bubble a raw 500.
        """
        import os
        import sys

        orig = sys.modules.get("vetinari.observability.checkpoints")
        sys.modules["vetinari.observability.checkpoints"] = None  # type: ignore[assignment]
        try:
            with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": self._ADMIN_TOKEN}):
                resp = client.get("/api/v1/cost-analysis", headers=self._ADMIN_HEADERS)
        finally:
            if orig is None:
                sys.modules.pop("vetinari.observability.checkpoints", None)
            else:
                sys.modules["vetinari.observability.checkpoints"] = orig
        _assert_503_error(resp)
