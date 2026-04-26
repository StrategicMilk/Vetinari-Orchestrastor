"""Mounted request-level governance tests for model management routes.

Proves every model-management route returns bounded error responses on
subsystem failure (not raw 500) and validates input rejection on malformed
requests.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

_ADMIN_TOKEN = "test-admin-model-mgmt-governance"
# POST (mutation) requests require both the admin token and CSRF header.
_MUTATION_HEADERS = {"X-Admin-Token": _ADMIN_TOKEN, "X-Requested-With": "XMLHttpRequest"}
# GET requests only need the admin token.
_GET_HEADERS = {"X-Admin-Token": _ADMIN_TOKEN}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    """Litestar application with shutdown side-effects suppressed.

    Returns:
        A Litestar application instance ready for test use.
    """
    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        with patch("vetinari.web.litestar_app._register_shutdown_handlers"):
            from vetinari.web.litestar_app import create_app

            return create_app(debug=True)


@pytest.fixture
def client(app):
    """TestClient wrapping the Litestar app.

    Args:
        app: The Litestar application fixture.

    Yields:
        A configured Litestar TestClient.
    """
    from litestar.testing import TestClient

    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        with TestClient(app) as tc:
            yield tc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_503_error(response: object) -> None:
    """Assert the response is a well-formed 503 error envelope.

    Args:
        response: The HTTP response from TestClient.

    Raises:
        AssertionError: When status_code is not 503 or the body lacks the
            ``"status": "error"`` envelope field.
    """
    assert response.status_code == 503, f"Expected 503, got {response.status_code}: {response.text[:300]}"
    data = response.json()
    assert data.get("status") == "error", f"Expected envelope status='error', got: {data}"


def _assert_400_error(response: object) -> None:
    """Assert the response is a well-formed 400 error envelope.

    Args:
        response: The HTTP response from TestClient.

    Raises:
        AssertionError: When status_code is not 400 or the body lacks the
            ``"status": "error"`` envelope field.
    """
    assert response.status_code == 400, f"Expected 400, got {response.status_code}: {response.text[:300]}"
    data = response.json()
    assert data.get("status") == "error", f"Expected envelope status='error', got: {data}"


# ---------------------------------------------------------------------------
# TestAssignTasks  -  POST /api/v1/models/assign-tasks
# ---------------------------------------------------------------------------


class TestAssignTasks:
    """POST /api/v1/models/assign-tasks governance."""

    def test_model_pool_failure_returns_503(self, client: object) -> None:
        """When ModelPool constructor raises, the route returns a bounded 503.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.models.model_pool.ModelPool",
            side_effect=RuntimeError("model pool unavailable"),
        ):
            response = client.post(
                "/api/v1/models/assign-tasks",
                json={"tasks": [{"id": "t1"}]},
                headers=_MUTATION_HEADERS,
            )
        _assert_503_error(response)

    def test_missing_tasks_returns_400(self, client: object) -> None:
        """When the request body has no 'tasks' key, the route returns 400.

        Args:
            client: The TestClient fixture.
        """
        response = client.post(
            "/api/v1/models/assign-tasks",
            json={},
            headers=_MUTATION_HEADERS,
        )
        _assert_400_error(response)

    def test_non_list_tasks_returns_400(self, client: object) -> None:
        """When 'tasks' is not a list, the route returns 400.

        Args:
            client: The TestClient fixture.
        """
        response = client.post(
            "/api/v1/models/assign-tasks",
            json={"tasks": "not-a-list"},
            headers=_MUTATION_HEADERS,
        )
        _assert_400_error(response)

    def test_happy_path_returns_200(self, client: object) -> None:
        """When ModelPool succeeds, the route returns 200 with the tasks list.

        A valid tasks list is assigned and returned under the success envelope.

        Args:
            client: The TestClient fixture.
        """
        mock_pool_instance = MagicMock()
        mock_pool_instance.assign_tasks_to_models.return_value = None
        mock_pool_cls = MagicMock(return_value=mock_pool_instance)

        with patch("vetinari.models.model_pool.ModelPool", mock_pool_cls):
            response = client.post(
                "/api/v1/models/assign-tasks",
                json={"tasks": [{"id": "t1"}]},
                headers=_MUTATION_HEADERS,
            )

        assert response.status_code == 201, (
            f"Expected 201, got {response.status_code}: {response.text[:300]}"
        )
        body = response.json()
        data = body.get("data", body)
        assert data["tasks"] == [{"id": "t1"}], f"Expected task payload to round-trip unchanged, got: {data}"


# ---------------------------------------------------------------------------
# TestAllAvailableModels  -  GET /api/v1/models/all-available
# ---------------------------------------------------------------------------


class TestAllAvailableModels:
    """GET /api/v1/models/all-available governance."""

    def test_model_pool_failure_returns_503(self, client: object) -> None:
        """When ModelPool constructor raises, the route returns a bounded 503.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.models.model_pool.ModelPool",
            side_effect=RuntimeError("model pool unavailable"),
        ):
            response = client.get("/api/v1/models/all-available", headers=_GET_HEADERS)
        _assert_503_error(response)

    def test_happy_path_returns_200(self, client: object) -> None:
        """When ModelPool succeeds, the route returns 200 with the models list.

        An empty model list from the pool is returned under the success envelope.

        Args:
            client: The TestClient fixture.
        """
        mock_pool_instance = MagicMock()
        mock_pool_instance.get_all_available_models.return_value = []
        mock_pool_cls = MagicMock(return_value=mock_pool_instance)

        with patch("vetinari.models.model_pool.ModelPool", mock_pool_cls):
            response = client.get("/api/v1/models/all-available", headers=_GET_HEADERS)

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:300]}"
        body = response.json()
        data = body.get("data", body)
        assert data["models"] == [], f"Expected empty available-model list in response data, got: {data}"


# ---------------------------------------------------------------------------
# TestAllDraftPairStats  -  GET /api/v1/models/draft-pairs/stats
# ---------------------------------------------------------------------------


class TestAllDraftPairStats:
    """GET /api/v1/models/draft-pairs/stats governance."""

    def test_resolver_failure_returns_503(self, client: object) -> None:
        """When get_draft_pair_resolver raises, the route returns a bounded 503.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.models.draft_pair_resolver.get_draft_pair_resolver",
            side_effect=RuntimeError("resolver unavailable"),
        ):
            response = client.get("/api/v1/models/draft-pairs/stats", headers=_GET_HEADERS)
        _assert_503_error(response)

    def test_happy_path_returns_200(self, client: object) -> None:
        """When the resolver succeeds, the route returns 200 with pairs stats.

        An empty dict from get_all_stats is returned under the success envelope.

        Args:
            client: The TestClient fixture.
        """
        mock_resolver = MagicMock()
        mock_resolver.get_all_stats.return_value = {}

        with patch(
            "vetinari.models.draft_pair_resolver.get_draft_pair_resolver",
            return_value=mock_resolver,
        ):
            response = client.get("/api/v1/models/draft-pairs/stats", headers=_GET_HEADERS)

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:300]}"
        body = response.json()
        data = body.get("data", body)
        assert data["pairs"] == {}, f"Expected empty draft-pair stats in response data, got: {data}"


# ---------------------------------------------------------------------------
# TestPairStats  -  GET /api/v1/models/draft-pairs/{main}/{draft}/stats
# ---------------------------------------------------------------------------


class TestPairStats:
    """GET /api/v1/models/draft-pairs/{main}/{draft}/stats governance."""

    def test_resolver_failure_returns_503(self, client: object) -> None:
        """When get_draft_pair_resolver raises, the route returns a bounded 503.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.models.draft_pair_resolver.get_draft_pair_resolver",
            side_effect=RuntimeError("resolver unavailable"),
        ):
            response = client.get(
                "/api/v1/models/draft-pairs/main-model/draft-model/stats",
                headers=_GET_HEADERS,
            )
        _assert_503_error(response)

    def test_happy_path_returns_200(self, client: object) -> None:
        """When the resolver succeeds, the route returns 200 with pair stats.

        The resolver's get_pair_stats result is returned under the success envelope.

        Args:
            client: The TestClient fixture.
        """
        mock_resolver = MagicMock()
        mock_resolver.get_pair_stats.return_value = {
            "acceptance_rate": 0.9,
            "total": 10,
            "is_disabled": False,
        }

        with patch(
            "vetinari.models.draft_pair_resolver.get_draft_pair_resolver",
            return_value=mock_resolver,
        ):
            response = client.get(
                "/api/v1/models/draft-pairs/main-model/draft-model/stats",
                headers=_GET_HEADERS,
            )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:300]}"
        body = response.json()
        data = body.get("data", body)
        assert data["acceptance_rate"] == 0.9, f"Expected acceptance_rate=0.9 in response data, got: {data}"
        assert data["total"] == 10, f"Expected total=10 in response data, got: {data}"


# ---------------------------------------------------------------------------
# TestVramThermalStatus  -  GET /api/v1/vram/thermal-status
# ---------------------------------------------------------------------------


class TestVramThermalStatus:
    """GET /api/v1/vram/thermal-status governance."""

    def test_vram_manager_failure_returns_503(self, client: object) -> None:
        """When get_vram_manager raises, the route returns a bounded 503.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.models.vram_manager.get_vram_manager",
            side_effect=RuntimeError("vram manager unavailable"),
        ):
            response = client.get("/api/v1/vram/thermal-status", headers=_GET_HEADERS)
        _assert_503_error(response)

    def test_happy_path_returns_200(self, client: object) -> None:
        """When the VRAM manager succeeds, the route returns 200 with thermal status.

        The throttle flag and temperature reading are returned under the success envelope.

        Args:
            client: The TestClient fixture.
        """
        mock_mgr = MagicMock()
        mock_mgr.is_thermal_throttled.return_value = False
        mock_mgr.get_gpu_temperature.return_value = 45

        with patch(
            "vetinari.models.vram_manager.get_vram_manager",
            return_value=mock_mgr,
        ):
            response = client.get("/api/v1/vram/thermal-status", headers=_GET_HEADERS)

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:300]}"
        body = response.json()
        data = body.get("data", body)
        assert "throttled" in data, f"Expected 'throttled' in response data, got: {data}"
        assert data["throttled"] is False, f"Expected throttled=False, got: {data['throttled']}"


# ---------------------------------------------------------------------------
# TestVramPhaseRecommendation  -  GET /api/v1/vram/phase
# ---------------------------------------------------------------------------


class TestVramPhaseRecommendation:
    """GET /api/v1/vram/phase governance."""

    def test_vram_manager_failure_returns_503(self, client: object) -> None:
        """When get_vram_manager raises, the route returns a bounded 503.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.models.vram_manager.get_vram_manager",
            side_effect=RuntimeError("vram manager unavailable"),
        ):
            response = client.get("/api/v1/vram/phase", headers=_GET_HEADERS)
        _assert_503_error(response)

    def test_happy_path_returns_200(self, client: object) -> None:
        """When the VRAM manager succeeds, the route returns 200 with phase recommendation.

        The phase and load/unload lists from get_phase_recommendation are returned
        under the success envelope.

        Args:
            client: The TestClient fixture.
        """
        mock_mgr = MagicMock()
        mock_mgr.get_phase_recommendation.return_value = {
            "phase": "execution",
            "load": [],
            "unload": [],
        }

        with patch(
            "vetinari.models.vram_manager.get_vram_manager",
            return_value=mock_mgr,
        ):
            response = client.get("/api/v1/vram/phase", headers=_GET_HEADERS)

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:300]}"
        body = response.json()
        data = body.get("data", body)
        assert data["phase"] == "execution", f"Expected execution phase recommendation in response data, got: {data}"


# ---------------------------------------------------------------------------
# TestVramSetPhase  -  POST /api/v1/vram/phase
# ---------------------------------------------------------------------------


class TestVramSetPhase:
    """POST /api/v1/vram/phase governance."""

    def test_vram_manager_failure_returns_503(self, client: object) -> None:
        """When the vram_manager module is unimportable, the route returns 503.

        The handler imports ExecutionPhase before calling get_vram_manager,
        so we make the entire module unavailable to trigger the 503 path.

        Args:
            client: The TestClient fixture.
        """
        with patch.dict("sys.modules", {"vetinari.models.vram_manager": None}):
            response = client.post(
                "/api/v1/vram/phase",
                json={"phase": "planning"},
                headers=_MUTATION_HEADERS,
            )
        _assert_503_error(response)

    def test_missing_phase_returns_400(self, client: object) -> None:
        """When 'phase' is absent from the request body, the route returns 400.

        This is a pre-subsystem validation, so 400 is returned regardless of
        subsystem availability.

        Args:
            client: The TestClient fixture.
        """
        response = client.post(
            "/api/v1/vram/phase",
            json={},
            headers=_MUTATION_HEADERS,
        )
        _assert_400_error(response)

    def test_happy_path_returns_200(self, client: object) -> None:
        """When phase is valid and the VRAM manager succeeds, the route returns 200.

        A recognised phase value ("planning") is accepted and confirmed in the
        success envelope.

        Args:
            client: The TestClient fixture.
        """
        mock_mgr = MagicMock()
        mock_mgr.set_phase.return_value = None

        with patch(
            "vetinari.models.vram_manager.get_vram_manager",
            return_value=mock_mgr,
        ):
            response = client.post(
                "/api/v1/vram/phase",
                json={"phase": "planning"},
                headers=_MUTATION_HEADERS,
            )

        assert response.status_code == 201, (
            f"Expected 201, got {response.status_code}: {response.text[:300]}"
        )
        body = response.json()
        data = body.get("data", body)
        assert data.get("phase") == "planning", f"Expected phase='planning' in response data, got: {data}"


# ---------------------------------------------------------------------------
# TestBuildCascadeFromRouter  -  POST /api/v1/models/cascade-router/build
# ---------------------------------------------------------------------------


class TestBuildCascadeFromRouter:
    """POST /api/v1/models/cascade-router/build governance."""

    def test_router_failure_returns_503(self, client: object) -> None:
        """When get_model_router raises, the route returns a bounded 503.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.models.dynamic_model_router.get_model_router",
            side_effect=RuntimeError("router unavailable"),
        ):
            response = client.post(
                "/api/v1/models/cascade-router/build",
                json={"task_type": "general"},
                headers=_MUTATION_HEADERS,
            )
        _assert_503_error(response)

    def test_import_error_returns_503(self, client: object) -> None:
        """When the cascade_router module is unimportable, the route returns 503.

        Args:
            client: The TestClient fixture.
        """
        with patch.dict("sys.modules", {"vetinari.cascade_router": None}):
            response = client.post(
                "/api/v1/models/cascade-router/build",
                json={"task_type": "general"},
                headers=_MUTATION_HEADERS,
            )
        _assert_503_error(response)

    def test_happy_path_returns_200(self, client: object) -> None:
        """When the cascade router builds successfully, the route returns 200.

        A mock cascade with empty tiers is built and the tiers list is returned
        under the success envelope.

        Args:
            client: The TestClient fixture.
        """
        mock_cascade = MagicMock()
        mock_cascade._tiers = []
        mock_dynamic_router = MagicMock()

        with (
            patch(
                "vetinari.cascade_router.build_cascade_from_router",
                return_value=mock_cascade,
            ),
            patch(
                "vetinari.models.dynamic_model_router.get_model_router",
                return_value=mock_dynamic_router,
            ),
        ):
            response = client.post(
                "/api/v1/models/cascade-router/build",
                json={"task_type": "general"},
                headers=_MUTATION_HEADERS,
            )

        assert response.status_code == 201, (
            f"Expected 201, got {response.status_code}: {response.text[:300]}"
        )
        body = response.json()
        data = body.get("data", body)
        assert data["tiers"] == [], f"Expected empty cascade tiers in response data, got: {data}"
        assert data["task_type"] == "general", f"Expected task_type='general' in response data, got: {data}"


# ---------------------------------------------------------------------------
# TestCascadeStats  -  GET /api/v1/models/cascade/stats
# ---------------------------------------------------------------------------


class TestCascadeStats:
    """GET /api/v1/models/cascade/stats governance."""

    def test_adapter_manager_failure_returns_503(self, client: object) -> None:
        """When get_adapter_manager raises, the route returns a bounded 503.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.adapter_manager.get_adapter_manager",
            side_effect=RuntimeError("adapter manager unavailable"),
        ):
            response = client.get("/api/v1/models/cascade/stats", headers=_GET_HEADERS)
        _assert_503_error(response)

    def test_happy_path_returns_200(self, client: object) -> None:
        """When the adapter manager succeeds, the route returns 200 with stats.

        An empty stats dict is returned under the success envelope.

        Args:
            client: The TestClient fixture.
        """
        mock_adapter_mgr = MagicMock()
        mock_adapter_mgr.get_cascade_stats.return_value = {}

        with patch(
            "vetinari.adapter_manager.get_adapter_manager",
            return_value=mock_adapter_mgr,
        ):
            response = client.get("/api/v1/models/cascade/stats", headers=_GET_HEADERS)

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:300]}"
        body = response.json()
        data = body.get("data", body)
        assert data["stats"] == {}, f"Expected empty cascade stats in response data, got: {data}"


# ---------------------------------------------------------------------------
# TestCascadeDisable  -  POST /api/v1/models/cascade/disable
# ---------------------------------------------------------------------------


class TestCascadeDisable:
    """POST /api/v1/models/cascade/disable governance."""

    def test_adapter_manager_failure_returns_503(self, client: object) -> None:
        """When get_adapter_manager raises, the route returns a bounded 503.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.adapter_manager.get_adapter_manager",
            side_effect=RuntimeError("adapter manager unavailable"),
        ):
            response = client.post(
                "/api/v1/models/cascade/disable",
                headers=_MUTATION_HEADERS,
            )
        _assert_503_error(response)

    def test_happy_path_returns_200(self, client: object) -> None:
        """When the adapter manager succeeds, the route returns 200 confirming disable.

        The success envelope contains the ``disabled`` confirmation field.

        Args:
            client: The TestClient fixture.
        """
        mock_adapter_mgr = MagicMock()
        mock_adapter_mgr.disable_cascade_routing.return_value = None

        with patch(
            "vetinari.adapter_manager.get_adapter_manager",
            return_value=mock_adapter_mgr,
        ):
            response = client.post(
                "/api/v1/models/cascade/disable",
                headers=_MUTATION_HEADERS,
            )

        assert response.status_code == 201, (
            f"Expected 201, got {response.status_code}: {response.text[:300]}"
        )
        body = response.json()
        data = body.get("data", body)
        assert data.get("disabled") is True, f"Expected disabled=True in response data, got: {data}"


# ---------------------------------------------------------------------------
# TestRemoveModel  -  POST /api/v1/models/{model_id}/delete
# ---------------------------------------------------------------------------


class TestRemoveModel:
    """POST /api/v1/models/{model_id}/delete governance."""

    def test_relay_failure_returns_503(self, client: object) -> None:
        """When get_model_relay raises, the route returns a bounded 503.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.models.model_relay.get_model_relay",
            side_effect=RuntimeError("model relay unavailable"),
        ):
            response = client.post(
                "/api/v1/models/some-model-id/delete",
                headers=_MUTATION_HEADERS,
            )
        _assert_503_error(response)

    def test_happy_path_returns_200(self, client: object) -> None:
        """When the relay succeeds and the model exists, the route returns 200.

        The relay is pre-populated with the target model so the existence check
        passes and remove_model is called successfully.

        Args:
            client: The TestClient fixture.
        """
        mock_relay = MagicMock()
        mock_relay.models = {"test-model": True}
        mock_relay.remove_model.return_value = None

        with patch(
            "vetinari.models.model_relay.get_model_relay",
            return_value=mock_relay,
        ):
            response = client.post(
                "/api/v1/models/test-model/delete",
                headers=_MUTATION_HEADERS,
            )

        assert response.status_code == 201, (
            f"Expected 201, got {response.status_code}: {response.text[:300]}"
        )
        body = response.json()
        data = body.get("data", body)
        assert data.get("removed") == "test-model", f"Expected removed='test-model' in response data, got: {data}"
