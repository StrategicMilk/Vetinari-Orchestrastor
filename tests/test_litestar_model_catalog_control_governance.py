"""Mounted request-level governance tests for model-catalog control routes.

Proves model-catalog POST/PUT routes reject empty bodies, handle helper failures
with bounded error responses, and do not flatten missing resources or failed
background operations into success payloads.

Routes covered:
  POST /api/v1/models/select                        -  task_type required; relay failure -> 503
  PUT  /api/v1/models/policy                        -  body required; relay failure -> 503
  POST /api/v1/models/reload                        -  relay failure -> 503
  POST /api/v1/models/search                        -  discovery failure -> 503
  POST /api/v1/models/download                      -  download failure -> 502 (not success)
  POST /api/v1/project/{id}/model-search            -  search failure -> 503; degraded flag
  POST /api/v1/project/{id}/refresh-models          -  nonexistent -> 404; invalid id -> 400
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ADMIN_TOKEN = "test-catalog-control-governance"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    """Litestar app built once per module with shutdown side-effects suppressed.

    Sets the admin token env var so admin_guard accepts requests carrying the
    matching ``X-Admin-Token`` header.

    Returns:
        A Litestar application instance ready for test use.
    """
    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        with patch("vetinari.web.litestar_app._register_shutdown_handlers"):
            from vetinari.web.litestar_app import create_app

            return create_app(debug=True)


@pytest.fixture
def client(app):
    """TestClient bound to the shared Litestar app with the admin token active.

    The env var must be alive for the entire TestClient lifetime, not just
    during app creation, because admin_guard reads it at request time.

    Args:
        app: The Litestar application fixture.

    Yields:
        A configured Litestar TestClient ready for each test.
    """
    from litestar.testing import TestClient

    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        with TestClient(app) as tc:
            yield tc


# Request headers that satisfy both admin_guard (token) and any CSRF guards.
_ADMIN_HEADERS = {
    "X-Admin-Token": _ADMIN_TOKEN,
    "X-Requested-With": "XMLHttpRequest",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_error(response: object, code: int) -> dict:
    """Assert the response has the expected error status code and envelope.

    Args:
        response: The HTTP response from TestClient.
        code: The expected HTTP status code.

    Returns:
        The parsed JSON body for further assertion.

    Raises:
        AssertionError: When the status code or envelope does not match.
    """
    assert response.status_code == code, f"Expected {code}, got {response.status_code}: {response.text[:400]}"
    data = response.json()
    assert data.get("status") == "error", f"Expected envelope status='error' for {code} response, got: {data}"
    return data


# ---------------------------------------------------------------------------
# TestModelSelect  -  POST /api/v1/models/select
# ---------------------------------------------------------------------------


class TestModelSelect:
    """Tests for the POST /api/v1/models/select route."""

    def test_empty_body_returns_400(self, client: object) -> None:
        """Sending {} body returns 400 before the relay is called.

        The route must reject requests missing task_type rather than forwarding
        None to pick_model_for_task.

        Args:
            client: The TestClient fixture.
        """
        response = client.post(
            "/api/v1/models/select",
            json={},
            headers=_ADMIN_HEADERS,
        )
        _assert_error(response, 400)

    def test_missing_task_type_returns_400(self, client: object) -> None:
        """Sending task_type='' (empty string) returns 400.

        Args:
            client: The TestClient fixture.
        """
        response = client.post(
            "/api/v1/models/select",
            json={"task_type": ""},
            headers=_ADMIN_HEADERS,
        )
        _assert_error(response, 400)

    def test_relay_failure_returns_503(self, client: object) -> None:
        """When pick_model_for_task raises, the route returns a bounded 503.

        Args:
            client: The TestClient fixture.
        """
        mock_relay = MagicMock()
        mock_relay.pick_model_for_task.side_effect = RuntimeError("relay down")

        with patch("vetinari.models.model_relay.model_relay", mock_relay):
            response = client.post(
                "/api/v1/models/select",
                json={"task_type": "general"},
                headers=_ADMIN_HEADERS,
            )
        _assert_error(response, 503)

    def test_valid_request_succeeds(self, client: object) -> None:
        """When relay returns a selection, the route returns a non-error response.

        Args:
            client: The TestClient fixture.
        """
        mock_selection = MagicMock()
        mock_selection.to_dict.return_value = {"model_id": "test-model", "score": 0.9}

        mock_relay = MagicMock()
        mock_relay.pick_model_for_task.return_value = mock_selection

        with patch("vetinari.models.model_relay.model_relay", mock_relay):
            response = client.post(
                "/api/v1/models/select",
                json={"task_type": "general"},
                headers=_ADMIN_HEADERS,
            )

        assert response.status_code == 201, (
            f"Expected 201 for valid selection, got {response.status_code}: {response.text[:300]}"
        )
        data = response.json()
        assert data.get("model_id") == "test-model", f"Expected model_id in response, got: {data}"


# ---------------------------------------------------------------------------
# TestModelPolicyUpdate  -  PUT /api/v1/models/policy
# ---------------------------------------------------------------------------


class TestModelPolicyUpdate:
    """Tests for the PUT /api/v1/models/policy route."""

    def test_empty_body_returns_400(self, client: object) -> None:
        """Sending {} body returns 400 because the policy body is required.

        An empty dict must not silently create a default RoutingPolicy and
        report "updated".

        Args:
            client: The TestClient fixture.
        """
        response = client.put(
            "/api/v1/models/policy",
            json={},
            headers=_ADMIN_HEADERS,
        )
        _assert_error(response, 400)

    def test_relay_failure_returns_503(self, client: object) -> None:
        """When RoutingPolicy.from_dict raises, the route returns 503.

        Patches ``from_dict`` on the RoutingPolicy class so the exception
        fires at the exact call site the handler uses.

        Args:
            client: The TestClient fixture.
        """
        mock_policy_cls = MagicMock()
        mock_policy_cls.from_dict.side_effect = RuntimeError("routing policy down")

        with patch("vetinari.models.model_relay.RoutingPolicy", mock_policy_cls):
            response = client.put(
                "/api/v1/models/policy",
                json={"local_first": True},
                headers=_ADMIN_HEADERS,
            )
        _assert_error(response, 503)

    def test_valid_request_succeeds(self, client: object) -> None:
        """When relay updates the policy, the route returns status='updated'.

        Args:
            client: The TestClient fixture.
        """
        mock_policy = MagicMock()
        # Handler calls RoutingPolicy.from_dict(data), not RoutingPolicy(data),
        # so configure from_dict (a classmethod) to return our mock policy.
        mock_policy.to_dict.return_value = {"local_first": True, "privacy_weight": 0.8}

        mock_routing_policy_cls = MagicMock()
        mock_routing_policy_cls.from_dict.return_value = mock_policy
        mock_relay = MagicMock()
        mock_relay.set_policy.return_value = None

        with (
            patch("vetinari.models.model_relay.RoutingPolicy", mock_routing_policy_cls),
            patch("vetinari.models.model_relay.model_relay", mock_relay),
        ):
            response = client.put(
                "/api/v1/models/policy",
                json={"local_first": True, "privacy_weight": 0.8},
                headers=_ADMIN_HEADERS,
            )

        assert response.status_code == 200, (
            f"Expected 200 for policy update, got {response.status_code}: {response.text[:300]}"
        )
        data = response.json()
        assert data.get("status") == "updated", f"Expected status='updated' in response, got: {data}"
        assert data["policy"] == {"local_first": True, "privacy_weight": 0.8}, (
            f"Expected serialized policy payload in response, got: {data}"
        )


# ---------------------------------------------------------------------------
# TestModelReload  -  POST /api/v1/models/reload
# ---------------------------------------------------------------------------


class TestModelReload:
    """Tests for the POST /api/v1/models/reload route."""

    def test_relay_failure_returns_503(self, client: object) -> None:
        """When reload_catalog raises, the route returns a bounded 503.

        Args:
            client: The TestClient fixture.
        """
        mock_relay = MagicMock()
        mock_relay.reload_catalog.side_effect = RuntimeError("relay down")

        with patch("vetinari.models.model_relay.model_relay", mock_relay):
            response = client.post(
                "/api/v1/models/reload",
                json={},
                headers=_ADMIN_HEADERS,
            )
        _assert_error(response, 503)

    def test_valid_request_succeeds(self, client: object) -> None:
        """When reload_catalog succeeds, the route returns status='reloaded'.

        Args:
            client: The TestClient fixture.
        """
        mock_relay = MagicMock()
        mock_relay.reload_catalog.return_value = None
        mock_relay.get_all_models.return_value = []

        with patch("vetinari.models.model_relay.model_relay", mock_relay):
            response = client.post(
                "/api/v1/models/reload",
                json={},
                headers=_ADMIN_HEADERS,
            )

        assert response.status_code == 201, (
            f"Expected 201 for catalog reload, got {response.status_code}: {response.text[:300]}"
        )
        data = response.json()
        assert data.get("status") == "reloaded", f"Expected status='reloaded' in response, got: {data}"
        assert data["models_loaded"] == 0, f"Expected zero loaded models from empty relay, got: {data}"


# ---------------------------------------------------------------------------
# TestGlobalModelSearch  -  POST /api/v1/models/search
# ---------------------------------------------------------------------------


class TestGlobalModelSearch:
    """Tests for the POST /api/v1/models/search route."""

    def test_search_failure_returns_503(self, client: object) -> None:
        """When ModelDiscovery.search raises, the route returns 503.

        The ``_get_models_cached`` call is patched to succeed so the failure
        is isolated to the search call itself.

        Args:
            client: The TestClient fixture.
        """
        mock_discovery = MagicMock()
        mock_discovery.search.side_effect = RuntimeError("discovery service down")

        with (
            patch("vetinari.web.shared.ENABLE_EXTERNAL_DISCOVERY", True),
            patch("vetinari.web.shared._get_models_cached", return_value=[]),
            patch("vetinari.model_discovery.ModelDiscovery", return_value=mock_discovery),
        ):
            response = client.post(
                "/api/v1/models/search",
                json={"query": "fast coding model"},
                headers=_ADMIN_HEADERS,
            )
        _assert_error(response, 503)

    def test_missing_query_returns_400(self, client: object) -> None:
        """Sending {} body returns 400 because query is required.

        Args:
            client: The TestClient fixture.
        """
        with patch("vetinari.web.shared.ENABLE_EXTERNAL_DISCOVERY", True):
            response = client.post(
                "/api/v1/models/search",
                json={},
                headers=_ADMIN_HEADERS,
            )
        _assert_error(response, 400)

    def test_valid_search_succeeds(self, client: object) -> None:
        """When discovery returns candidates, the route returns status='ok'.

        Args:
            client: The TestClient fixture.
        """
        mock_candidate = MagicMock()
        mock_candidate.to_dict.return_value = {"model_id": "found-model", "score": 0.7}

        mock_discovery = MagicMock()
        mock_discovery.search.return_value = [mock_candidate]

        with (
            patch("vetinari.web.shared.ENABLE_EXTERNAL_DISCOVERY", True),
            patch("vetinari.web.shared._get_models_cached", return_value=[]),
            patch("vetinari.model_discovery.ModelDiscovery", return_value=mock_discovery),
        ):
            response = client.post(
                "/api/v1/models/search",
                json={"query": "fast coding model"},
                headers=_ADMIN_HEADERS,
            )

        assert response.status_code == 201, (
            f"Expected 201 for search, got {response.status_code}: {response.text[:300]}"
        )
        data = response.json()
        assert data.get("status") == "ok", f"Expected status='ok' in response, got: {data}"
        assert data["candidates"] == [{"model_id": "found-model", "score": 0.7}], (
            f"Expected serialized discovery candidates in response, got: {data}"
        )
        assert data["count"] == 1, f"Expected count=1 in response, got: {data}"


# ---------------------------------------------------------------------------
# TestModelDownload  -  POST /api/v1/models/download
# ---------------------------------------------------------------------------


class TestModelDownload:
    """Tests for the POST /api/v1/models/download route."""

    def test_download_failure_returns_502(self, client: object) -> None:
        """When start_download raises, the route returns 502 not a success payload.

        The caller must be able to distinguish a failed start from a successful
        one  -  returning ``status='acknowledged'`` with a 201 would mask the failure.

        Args:
            client: The TestClient fixture.
        """
        mock_discovery = MagicMock()
        mock_discovery.start_download.side_effect = RuntimeError("download backend unavailable")

        with patch("vetinari.model_discovery.ModelDiscovery", return_value=mock_discovery):
            response = client.post(
                "/api/v1/models/download",
                json={"repo_id": "author/model", "filename": "model.gguf"},
                headers=_ADMIN_HEADERS,
            )
        _assert_error(response, 502)

    def test_missing_repo_id_returns_400(self, client: object) -> None:
        """Missing repo_id returns 400.

        Args:
            client: The TestClient fixture.
        """
        response = client.post(
            "/api/v1/models/download",
            json={"filename": "model.gguf"},
            headers=_ADMIN_HEADERS,
        )
        _assert_error(response, 400)

    def test_valid_request_succeeds(self, client: object) -> None:
        """When start_download succeeds, the route returns status='started'.

        Args:
            client: The TestClient fixture.
        """
        mock_discovery = MagicMock()
        mock_discovery.start_download.return_value = None

        with patch("vetinari.model_discovery.ModelDiscovery", return_value=mock_discovery):
            response = client.post(
                "/api/v1/models/download",
                json={"repo_id": "author/model", "filename": "model.gguf"},
                headers=_ADMIN_HEADERS,
            )

        assert response.status_code == 201, (
            f"Expected 201 for download start, got {response.status_code}: {response.text[:300]}"
        )
        data = response.json()
        assert data.get("status") == "started", f"Expected status='started', got: {data}"
        assert data.get("repo_id") == "author/model", f"Expected repo_id='author/model', got: {data}"

    def test_native_snapshot_request_does_not_require_filename(self, client: object) -> None:
        """Native vLLM/NIM snapshot downloads are repo-level, not single filename downloads."""
        mock_discovery = MagicMock()
        mock_discovery.start_download.return_value = {
            "status": "started",
            "download_id": "dl-native",
            "repo_id": "author/model",
            "backend": "vllm",
            "format": "safetensors",
            "artifact_type": "snapshot",
            "path": "C:/models/native/vllm/safetensors/author--model/abc",
            "manifest_path": "C:/models/native/vllm/safetensors/author--model/abc/.vetinari-download.json",
            "file_count": 3,
        }

        with patch("vetinari.model_discovery.ModelDiscovery", return_value=mock_discovery):
            response = client.post(
                "/api/v1/models/download",
                json={"repo_id": "author/model", "backend": "vllm", "format": "safetensors"},
                headers=_ADMIN_HEADERS,
            )

        assert response.status_code == 201, response.text[:300]
        data = response.json()
        assert data["artifact_type"] == "snapshot"
        assert data["backend"] == "vllm"
        mock_discovery.start_download.assert_called_once_with(
            repo_id="author/model",
            filename=None,
            revision=None,
            backend="vllm",
            model_format="safetensors",
        )

    def test_repo_only_download_defaults_to_native_snapshot(self, client: object) -> None:
        """Repo-level downloads default to native vLLM SafeTensors snapshots."""
        mock_discovery = MagicMock()
        mock_discovery.start_download.return_value = {
            "status": "started",
            "download_id": "dl-default-native",
            "repo_id": "author/model",
            "backend": "vllm",
            "format": "safetensors",
            "artifact_type": "snapshot",
        }

        with patch("vetinari.model_discovery.ModelDiscovery", return_value=mock_discovery):
            response = client.post(
                "/api/v1/models/download",
                json={"repo_id": "author/model"},
                headers=_ADMIN_HEADERS,
            )

        assert response.status_code == 201, response.text[:300]
        data = response.json()
        assert data["backend"] == "vllm"
        assert data["format"] == "safetensors"
        mock_discovery.start_download.assert_called_once_with(
            repo_id="author/model",
            filename=None,
            revision=None,
            backend="vllm",
            model_format=None,
        )

    def test_download_status_returns_tracked_state(self, client: object) -> None:
        """Download status route exposes ModelDiscovery tracked state."""
        mock_discovery = MagicMock()
        mock_discovery.get_download_status.return_value = {
            "download_id": "dl-native",
            "status": "running",
            "backend": "vllm",
        }

        with patch("vetinari.model_discovery.ModelDiscovery", return_value=mock_discovery):
            response = client.get("/api/v1/models/download/dl-native", headers=_ADMIN_HEADERS)

        assert response.status_code == 200, response.text[:300]
        assert response.json()["status"] == "running"

    def test_download_cancel_calls_discovery_cancel(self, client: object) -> None:
        """Download cancel route wires through to ModelDiscovery cancellation."""
        mock_discovery = MagicMock()
        mock_discovery.cancel_download.return_value = True

        with patch("vetinari.model_discovery.ModelDiscovery", return_value=mock_discovery):
            response = client.post("/api/v1/models/download/dl-native/cancel", headers=_ADMIN_HEADERS)

        assert response.status_code == 200, response.text[:300]
        assert response.json()["status"] == "canceling"
        mock_discovery.cancel_download.assert_called_once_with("dl-native")


# ---------------------------------------------------------------------------
# TestProjectModelSearch  -  POST /api/v1/project/{project_id}/model-search
# ---------------------------------------------------------------------------


class TestProjectModelSearch:
    """Tests for POST /api/v1/project/{project_id}/model-search."""

    def test_search_failure_returns_503(self, client: object) -> None:
        """When ModelDiscovery.search raises, the route returns 503.

        A temporary project directory is created so the path-validation checks
        pass; the failure is isolated to the search call itself.

        Args:
            client: The TestClient fixture.
        """
        mock_discovery = MagicMock()
        mock_discovery.search.side_effect = RuntimeError("discovery down")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            project_dir = tmp_path / "projects" / "test-project"
            project_dir.mkdir(parents=True)

            with (
                patch("vetinari.web.shared.ENABLE_EXTERNAL_DISCOVERY", True),
                patch("vetinari.web.shared._project_external_model_enabled", return_value=True),
                patch("vetinari.web.shared.current_config", MagicMock()),
                patch("vetinari.models.model_pool.ModelPool", side_effect=RuntimeError("pool down")),
                patch("vetinari.model_discovery.ModelDiscovery", return_value=mock_discovery),
                patch("vetinari.web.shared.PROJECT_ROOT", tmp_path),
            ):
                response = client.post(
                    "/api/v1/project/test-project/model-search",
                    json={"task_description": "fast model"},
                    headers=_ADMIN_HEADERS,
                )
        _assert_error(response, 503)

    def test_local_models_degraded_flagged(self, client: object) -> None:
        """When ModelPool fails, response includes ``_local_models_degraded: true``.

        Callers must be able to tell that the local model list was empty due to
        a failure rather than because no models are installed.

        Args:
            client: The TestClient fixture.
        """
        mock_candidate = MagicMock()
        mock_candidate.to_dict.return_value = {"model_id": "ext-model"}
        mock_discovery = MagicMock()
        mock_discovery.search.return_value = [mock_candidate]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            project_dir = tmp_path / "projects" / "test-project"
            project_dir.mkdir(parents=True)

            with (
                patch("vetinari.web.shared.ENABLE_EXTERNAL_DISCOVERY", True),
                patch("vetinari.web.shared._project_external_model_enabled", return_value=True),
                patch("vetinari.web.shared.current_config", MagicMock()),
                patch(
                    "vetinari.models.model_pool.ModelPool",
                    side_effect=RuntimeError("pool unavailable"),
                ),
                patch("vetinari.model_discovery.ModelDiscovery", return_value=mock_discovery),
                patch("vetinari.web.shared.PROJECT_ROOT", tmp_path),
            ):
                response = client.post(
                    "/api/v1/project/test-project/model-search",
                    json={"task_description": "fast model"},
                    headers=_ADMIN_HEADERS,
                )

        assert response.status_code == 201, (
            f"Expected 201 when search succeeds, got {response.status_code}: {response.text[:400]}"
        )
        data = response.json()
        assert data.get("_local_models_degraded") is True, (
            f"Expected _local_models_degraded=True when ModelPool failed, got: {data}"
        )
        assert data.get("status") == "ok", f"Expected status='ok', got: {data}"

    def test_valid_search_succeeds(self, client: object) -> None:
        """When all helpers succeed, the route returns candidates with status='ok'.

        Args:
            client: The TestClient fixture.
        """
        mock_candidate = MagicMock()
        mock_candidate.to_dict.return_value = {"model_id": "ext-model", "score": 0.8}
        mock_discovery = MagicMock()
        mock_discovery.search.return_value = [mock_candidate]

        mock_pool = MagicMock()
        mock_pool.list_models.return_value = []

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            project_dir = tmp_path / "projects" / "test-project"
            project_dir.mkdir(parents=True)

            with (
                patch("vetinari.web.shared.ENABLE_EXTERNAL_DISCOVERY", True),
                patch("vetinari.web.shared._project_external_model_enabled", return_value=True),
                patch("vetinari.web.shared.current_config", MagicMock()),
                patch("vetinari.models.model_pool.ModelPool", return_value=mock_pool),
                patch("vetinari.model_discovery.ModelDiscovery", return_value=mock_discovery),
                patch("vetinari.web.shared.PROJECT_ROOT", tmp_path),
            ):
                response = client.post(
                    "/api/v1/project/test-project/model-search",
                    json={"task_description": "fast model"},
                    headers=_ADMIN_HEADERS,
                )

        assert response.status_code == 201, (
            f"Expected 201 for search, got {response.status_code}: {response.text[:400]}"
        )
        data = response.json()
        assert data.get("status") == "ok", f"Expected status='ok', got: {data}"
        assert data["candidates"] == [{"model_id": "ext-model", "score": 0.8}], (
            f"Expected external candidate payload in degraded response, got: {data}"
        )
        assert data.get("_local_models_degraded") is False, (
            f"Expected _local_models_degraded=False when pool succeeds, got: {data}"
        )


# ---------------------------------------------------------------------------
# TestProjectRefreshModels  -  POST /api/v1/project/{project_id}/refresh-models
# ---------------------------------------------------------------------------


class TestProjectRefreshModels:
    """Tests for POST /api/v1/project/{project_id}/refresh-models."""

    def test_nonexistent_project_returns_404(self, client: object) -> None:
        """A valid project_id that does not exist on disk returns 404.

        Args:
            client: The TestClient fixture.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Do NOT create the project directory  -  it should not exist.
            with patch("vetinari.web.shared.PROJECT_ROOT", tmp_path):
                response = client.post(
                    "/api/v1/project/nonexistent-proj/refresh-models",
                    json={},
                    headers=_ADMIN_HEADERS,
                )
        _assert_error(response, 404)

    def test_invalid_project_id_returns_400(self, client: object) -> None:
        """A project_id with unsafe characters (path traversal) returns 400.

        The route must validate the path param before touching the filesystem
        to prevent directory traversal attacks.

        Args:
            client: The TestClient fixture.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with patch("vetinari.web.shared.PROJECT_ROOT", tmp_path):
                response = client.post(
                    "/api/v1/project/..%2Fevil/refresh-models",
                    json={},
                    headers=_ADMIN_HEADERS,
                )
        # Litestar may reject the URL-encoded traversal at routing level (404/400).
        assert response.status_code in (400, 404, 422), (
            f"Expected 4xx for unsafe project_id, got {response.status_code}: {response.text[:300]}"
        )

    def test_valid_project_succeeds(self, client: object) -> None:
        """When the project exists, the route returns status='ok'.

        Args:
            client: The TestClient fixture.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            project_dir = tmp_path / "projects" / "my-project"
            project_dir.mkdir(parents=True)

            with patch("vetinari.web.shared.PROJECT_ROOT", tmp_path):
                response = client.post(
                    "/api/v1/project/my-project/refresh-models",
                    json={},
                    headers=_ADMIN_HEADERS,
                )

        assert response.status_code == 201, (
            f"Expected 201 for valid project refresh, got {response.status_code}: {response.text[:300]}"
        )
        data = response.json()
        assert data.get("status") == "ok", f"Expected status='ok', got: {data}"
        assert data["message"] == "Model cache refreshed (live search enabled)", (
            f"Expected refresh confirmation message in response, got: {data}"
        )
