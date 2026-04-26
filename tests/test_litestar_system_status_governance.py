"""Mounted request-level governance tests for system-status and system-hardware routes.

Exercises degraded-success and error-path contracts for 7 routes across two
source files.  All tests go through the full Litestar HTTP stack via TestClient
so framework-level wiring (routing, serialization, middleware) is exercised
alongside handler logic.

Routes under test:
  GET  /api/v1/status                — path redaction (litestar_system_status)
  GET  /api/v1/token-stats           — degraded collector signalling
  GET  /api/v1/search                — warnings on source failures
  GET  /api/v1/batch/queue-stats     — 503 on processor failure
  GET  /api/v1/upgrade-check         — 503 on orchestrator failure
  GET  /api/v1/system/gpu            — graceful 200 on unexpected error
  GET  /api/v1/system/vram           — 503 on manager failure
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def litestar_app():
    """Minimal Litestar app with all handlers registered, shutdown suppressed.

    Returns:
        A Litestar application instance safe to use in tests.
    """
    with patch("vetinari.web.litestar_app._register_shutdown_handlers"):
        from vetinari.web.litestar_app import create_app

        app = create_app(debug=True)
    return app


# ---------------------------------------------------------------------------
# GET /api/v1/status
# ---------------------------------------------------------------------------


class TestStatusRoute:
    """GET /api/v1/status — path leak prevention."""

    def test_status_does_not_leak_absolute_paths(self, litestar_app: object) -> None:
        """models_dir and config_path must be redacted to '<configured>', not real paths.

        Absolute filesystem paths in operator-facing status output expose
        server layout to any caller with network access to the status endpoint.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        mock_cfg = MagicMock()
        mock_cfg.api_token = ""
        mock_cfg.models_dir = "/home/user/models"
        mock_cfg.config_path = "/etc/vetinari/config.yaml"
        mock_cfg.default_models = []
        mock_cfg.fallback_models = []
        mock_cfg.uncensored_fallback_models = []
        mock_cfg.memory_budget_gb = 8
        mock_cfg.local_gpu_layers = 0
        mock_cfg.active_model_id = None

        with patch("vetinari.web.litestar_system_status.create_system_status_handlers") as _:
            pass  # imported below via the app

        with patch("vetinari.web.shared.current_config", mock_cfg):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/status")

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:300]}"
        body = response.json()
        # Navigate into the response envelope: success_response wraps in {"data": ...}
        data = body.get("data", body)
        assert data.get("models_dir") == "<configured>", f"models_dir leaked real path: {data.get('models_dir')!r}"
        assert data.get("config_path") == "<configured>", f"config_path leaked real path: {data.get('config_path')!r}"


# ---------------------------------------------------------------------------
# GET /api/v1/token-stats
# ---------------------------------------------------------------------------


class TestTokenStatsRoute:
    """GET /api/v1/token-stats — collector degradation signalling."""

    def test_both_collectors_unavailable_signals_degraded(self, litestar_app: object) -> None:
        """When both telemetry and cost collectors raise, response must flag degraded=True.

        All-zero stats are indistinguishable from 'no usage recorded' unless
        the degradation flag is present.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        with (
            patch("vetinari.telemetry.get_telemetry_collector", side_effect=RuntimeError("telemetry down")),
            patch("vetinari.analytics.cost.get_cost_tracker", side_effect=RuntimeError("cost down")),
        ):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/token-stats")

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:300]}"
        body = response.json()
        assert body.get("degraded") is True, f"Expected 'degraded': True when both collectors fail, got: {body}"
        unavailable = body.get("unavailable_sources", [])
        assert len(unavailable) > 0, f"Expected non-empty 'unavailable_sources' list, got: {unavailable}"

    def test_one_collector_available_not_fully_degraded(self, litestar_app: object) -> None:
        """When only cost tracker fails, degraded must be False and sources list partial.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        mock_tel = MagicMock()
        mock_tel.get_summary.return_value = {
            "total_tokens_used": 42,
            "session_requests": 3,
            "by_provider": {},
        }

        with (
            patch("vetinari.telemetry.get_telemetry_collector", return_value=mock_tel),
            patch("vetinari.analytics.cost.get_cost_tracker", side_effect=RuntimeError("cost down")),
        ):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/token-stats")

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:300]}"
        body = response.json()
        # Only cost_tracker failed — not fully degraded
        assert body.get("degraded") is False, f"Expected 'degraded': False when only one collector fails, got: {body}"
        unavailable = body.get("unavailable_sources", [])
        assert "cost_tracker" in unavailable, f"Expected 'cost_tracker' in unavailable_sources, got: {unavailable}"
        assert "telemetry" not in unavailable, (
            f"'telemetry' should not be in unavailable_sources when it succeeded, got: {unavailable}"
        )


# ---------------------------------------------------------------------------
# GET /api/v1/search
# ---------------------------------------------------------------------------


class TestSearchRoute:
    """GET /api/v1/search — warnings list on source failures."""

    def test_memory_failure_signals_warning(self, litestar_app: object, tmp_path: object) -> None:
        """When memory store raises, response must include 'memory_search_unavailable' warning.

        The route should still return 200 with whatever project results it
        could gather — the warning is additive, not a replacement for results.

        Args:
            litestar_app: Litestar application fixture.
            tmp_path: Pytest temporary directory.
        """
        from litestar.testing import TestClient

        # Create a minimal project directory so the project scan path runs
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()
        proj = projects_dir / "test-proj"
        proj.mkdir()
        config_file = proj / "project.yaml"
        config_file.write_text(
            "project_name: test-proj\ndescription: a searchable project\nstatus: active\n",
            encoding="utf-8",
        )

        with (
            patch("vetinari.web.shared.PROJECT_ROOT", tmp_path),
            patch(
                "vetinari.memory.unified.get_unified_memory_store",
                side_effect=RuntimeError("memory store down"),
            ),
        ):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/search?q=searchable")

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:300]}"
        body = response.json()
        warnings = body.get("warnings", [])
        assert "memory_search_unavailable" in warnings, (
            f"Expected 'memory_search_unavailable' in warnings, got: {warnings}"
        )


# ---------------------------------------------------------------------------
# GET /api/v1/batch/queue-stats
# ---------------------------------------------------------------------------


class TestBatchQueueStatsRoute:
    """GET /api/v1/batch/queue-stats — 503 when batch processor unavailable."""

    def test_processor_unavailable_returns_503(self, litestar_app: object) -> None:
        """When get_batch_processor raises, the route must return 503, not 500.

        A 503 indicates service unavailability (transient) rather than an
        unhandled server error (permanent coding defect).

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        with patch(
            "vetinari.adapters.batch_processor.get_batch_processor",
            side_effect=RuntimeError("batch processor down"),
        ):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/batch/queue-stats")

        assert response.status_code == 503, (
            f"Expected 503 when batch processor unavailable, got {response.status_code}: {response.text[:300]}"
        )


# ---------------------------------------------------------------------------
# GET /api/v1/upgrade-check
# ---------------------------------------------------------------------------


class TestUpgradeCheckRoute:
    """GET /api/v1/upgrade-check — 503 when orchestrator unavailable."""

    def test_orchestrator_unavailable_returns_503(self, litestar_app: object) -> None:
        """When get_orchestrator raises, the route must return 503.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        with patch(
            "vetinari.web.shared.get_orchestrator",
            side_effect=RuntimeError("orchestrator down"),
        ):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/upgrade-check")

        assert response.status_code == 503, (
            f"Expected 503 when orchestrator unavailable, got {response.status_code}: {response.text[:300]}"
        )


# ---------------------------------------------------------------------------
# GET /api/v1/system/gpu
# ---------------------------------------------------------------------------


class TestGpuRoute:
    """GET /api/v1/system/gpu — graceful 200 on unexpected errors."""

    def test_unexpected_gpu_error_returns_graceful_200(self, litestar_app: object) -> None:
        """Unexpected exception from _gpu_info_locked must yield 200 with gpu_available=False.

        This matches the pynvml-not-installed branch contract so callers always
        get a consistent shape regardless of which failure mode occurred.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        mock_pynvml = MagicMock()

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(
                "vetinari.web.litestar_system_hardware._gpu_info_locked",
                side_effect=RuntimeError("unexpected GPU failure"),
            ),
        ):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/system/gpu")

        assert response.status_code == 200, (
            f"Expected 200 for graceful GPU error, got {response.status_code}: {response.text[:300]}"
        )
        body = response.json()
        assert body.get("gpu_available") is False, f"Expected gpu_available=False on unexpected error, got: {body}"
        assert body["error"] == "unexpected GPU failure", f"Expected error string to surface the underlying failure, got: {body}"


# ---------------------------------------------------------------------------
# GET /api/v1/system/vram
# ---------------------------------------------------------------------------


class TestVramRoute:
    """GET /api/v1/system/vram — 503 when VRAM manager unavailable."""

    def test_manager_unavailable_returns_503(self, litestar_app: object) -> None:
        """When get_vram_manager raises, the route must return 503 not 200 with error key.

        A 200 response with an 'error' key is a contradictory contract —
        callers cannot distinguish success from failure by status code alone.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        with patch(
            "vetinari.models.vram_manager.get_vram_manager",
            side_effect=RuntimeError("vram manager down"),
        ):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/system/vram")

        assert response.status_code == 503, (
            f"Expected 503 when VRAM manager unavailable, got {response.status_code}: {response.text[:300]}"
        )
        # Ensure it's not a 200-with-error-key anti-pattern
        assert response.status_code != 200, "Must not return 200 with error key — use 503 for service unavailability"
