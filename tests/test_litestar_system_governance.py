"""Mounted governance tests for system/health read routes.

Proves that system GET routes return bounded 503 on subsystem failure, and
that the composite health endpoint reports ``degraded`` (not ``healthy``) when
most subsystems are broken.

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
# GET /api/v1/health  -  degraded contract
# ---------------------------------------------------------------------------


class TestApiHealthDegraded:
    """GET /api/v1/health  -  when majority of subsystems fail, status must be 'degraded'."""

    def test_all_subsystems_failing_returns_degraded(self, client: object) -> None:
        """When all subsystem checks raise, overall status must be 'degraded' not 'healthy'.

        The health endpoint wraps each subsystem check individually  -  it never
        returns 503 itself. But it must report the overall state truthfully.
        """
        with (
            patch(
                "vetinari.resilience.get_circuit_breaker_registry",
                side_effect=RuntimeError("circuit breaker unavailable"),
            ),
            patch(
                "vetinari.workflow.spc.get_spc_monitor",
                side_effect=RuntimeError("spc unavailable"),
            ),
            patch(
                "vetinari.workflow.andon.get_andon_system",
                side_effect=RuntimeError("andon unavailable"),
            ),
            patch(
                "vetinari.learning.model_selector.get_thompson_selector",
                side_effect=RuntimeError("thompson unavailable"),
            ),
            patch(
                "vetinari.memory.unified.get_unified_memory_store",
                side_effect=RuntimeError("memory unavailable"),
            ),
            patch(
                "vetinari.analytics.anomaly.get_anomaly_detector",
                side_effect=RuntimeError("anomaly detector unavailable"),
            ),
            patch(
                "vetinari.analytics.forecasting.get_forecaster",
                side_effect=RuntimeError("forecaster unavailable"),
            ),
        ):
            response = client.get("/api/v1/health")

        assert response.status_code == 200, (
            f"Health endpoint must always return 200 (it reports degraded in body). Got {response.status_code}"
        )
        body = response.json()
        assert body.get("status") == "degraded", (
            f"Expected status='degraded' when all subsystems fail, got {body.get('status')!r}. Body: {body}"
        )

    def test_health_response_contains_checks_key(self, client: object) -> None:
        """Health response must always include a 'checks' dict regardless of subsystem state."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert "checks" in body, f"Health response missing 'checks' key. Body: {body}"
        assert isinstance(body["checks"], dict), f"'checks' must be a dict, got {type(body['checks']).__name__}"

    def test_health_response_contains_timestamp(self, client: object) -> None:
        """Health response must include an ISO-format timestamp."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert "timestamp" in body, f"Health response missing 'timestamp' key. Body: {body}"
        # Timestamp must be a non-empty string (ISO format)
        assert isinstance(body["timestamp"], str) and body["timestamp"], (
            f"'timestamp' must be a non-empty string. Got: {body['timestamp']!r}"
        )

    def test_readiness_fails_when_composite_health_degraded(self, client: object) -> None:
        """GET /ready must fail HTTP readiness when release health is degraded."""
        with (
            patch(
                "vetinari.resilience.get_circuit_breaker_registry",
                side_effect=RuntimeError("circuit breaker unavailable"),
            ),
            patch(
                "vetinari.workflow.spc.get_spc_monitor",
                side_effect=RuntimeError("spc unavailable"),
            ),
            patch(
                "vetinari.workflow.andon.get_andon_system",
                side_effect=RuntimeError("andon unavailable"),
            ),
            patch(
                "vetinari.learning.model_selector.get_thompson_selector",
                side_effect=RuntimeError("thompson unavailable"),
            ),
            patch(
                "vetinari.memory.unified.get_unified_memory_store",
                side_effect=RuntimeError("memory unavailable"),
            ),
            patch(
                "vetinari.analytics.anomaly.get_anomaly_detector",
                side_effect=RuntimeError("anomaly detector unavailable"),
            ),
            patch(
                "vetinari.analytics.forecasting.get_forecaster",
                side_effect=RuntimeError("forecaster unavailable"),
            ),
        ):
            response = client.get("/ready")

        assert response.status_code == 503
        body = response.json()
        assert body.get("status") == "degraded"
        assert body.get("readiness") == "degraded"

    def test_health_redacts_credential_presence(self, client: object) -> None:
        """Public health must not expose cloud credential presence oracles."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        checks = response.json()["checks"]
        privacy = checks["privacy"]
        assert "cloud_apis_configured" not in privacy
        assert privacy["credential_presence"] == "redacted"
        dispatch = checks["dispatch"]
        assert dispatch["capacity"] == "redacted"
        assert "queue_depth" not in dispatch
        assert "wip_utilization" not in dispatch


# ---------------------------------------------------------------------------
# GET /api/v1/system/resources  -  503 on psutil failure
# ---------------------------------------------------------------------------


class TestApiSystemResources:
    """GET /api/v1/system/resources  -  psutil failure must return 503."""

    def test_psutil_failure_returns_503(self, client: object) -> None:
        """Patch psutil to raise; endpoint must return 503 not 500."""
        import sys

        # Patch psutil inside the handler's import scope
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.side_effect = RuntimeError("psutil failure")
        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            response = client.get("/api/v1/system/resources")
        _assert_503_error(response)

    def test_psutil_not_installed_returns_503(self, client: object) -> None:
        """When psutil is not importable, endpoint must return 503."""
        import sys

        # Remove psutil from sys.modules to simulate it not being installed
        original = sys.modules.pop("psutil", None)
        try:
            with patch.dict(sys.modules, {"psutil": None}):  # type: ignore[dict-item]
                response = client.get("/api/v1/system/resources")
            _assert_503_error(response)
        finally:
            if original is not None:
                sys.modules["psutil"] = original

    def test_system_resources_redacts_disk_path(self, client: object) -> None:
        """Public resource diagnostics must not reveal host filesystem paths."""
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = MagicMock(used=1024 * 1024, total=2 * 1024 * 1024, percent=50.0)
        mock_psutil.cpu_percent.return_value = 7.0
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.disk_usage.return_value = MagicMock(
            used=1024**3,
            total=2 * 1024**3,
            percent=50.0,
        )

        import sys

        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            response = client.get("/api/v1/system/resources")

        assert response.status_code == 200
        disk = response.json()["disk"]
        assert disk["path"] == "redacted"
        assert disk["path_redacted"] is True


# ---------------------------------------------------------------------------
# GET /api/logs/stream  -  503 on SSE backend failure
# ---------------------------------------------------------------------------


class TestApiLogsStream:
    """GET /api/logs/stream  -  handler-level verification of SSE backend failure.

    SSE streaming routes deadlock Litestar's TestClient on Windows, so this
    test verifies the handler's error guard directly rather than through the
    full HTTP stack.
    """

    def test_backend_failure_returns_503(self) -> None:
        """When get_sse_backend() raises, stream_logs must return a 503 Response."""
        from litestar import Response

        import vetinari.web.litestar_log_stream as log_mod

        with patch.object(log_mod, "get_sse_backend", side_effect=RuntimeError("backend down")):
            handlers = log_mod.create_log_stream_handlers()
            # stream_logs is the first handler returned
            stream_fn = handlers[0].fn
            result = stream_fn()

        assert isinstance(result, Response), f"Expected Response, got {type(result)}"
        assert result.status_code == 503


# ---------------------------------------------------------------------------
# GET /api/logs/recent  -  503 on SSE backend failure
# ---------------------------------------------------------------------------


class TestApiLogsRecent:
    """GET /api/logs/recent  -  handler-level verification of backend failure.

    Uses direct handler invocation because the module-level get_sse_backend
    import binding resists TestClient-scoped patching.
    """

    def test_backend_failure_returns_503(self) -> None:
        """When get_sse_backend() raises, recent_logs must return a 503 Response."""
        from litestar import Response

        import vetinari.web.litestar_log_stream as log_mod

        with patch.object(log_mod, "get_sse_backend", side_effect=RuntimeError("backend down")):
            handlers = log_mod.create_log_stream_handlers()
            # recent_logs is the second handler returned
            recent_fn = handlers[1].fn
            result = recent_fn()

        assert isinstance(result, Response), f"Expected Response, got {type(result)}"
        assert result.status_code == 503

    def test_get_recent_raises_returns_503(self) -> None:
        """When backend.get_recent() raises, recent_logs must return 503."""
        from litestar import Response

        import vetinari.web.litestar_log_stream as log_mod

        mock_backend = MagicMock()
        mock_backend.get_recent.side_effect = RuntimeError("recent logs broken")
        with patch.object(log_mod, "get_sse_backend", return_value=mock_backend):
            handlers = log_mod.create_log_stream_handlers()
            recent_fn = handlers[1].fn
            result = recent_fn()

        assert isinstance(result, Response), f"Expected Response, got {type(result)}"
        assert result.status_code == 503


# ---------------------------------------------------------------------------
# GET /api/v1/metrics/latest  -  503 on subsystem failure
# ---------------------------------------------------------------------------


class TestApiMetricsLatest:
    """GET /api/v1/metrics/latest  -  subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_dashboard_api to raise; endpoint must return 503."""
        with patch(
            "vetinari.dashboard.api.get_dashboard_api",
            side_effect=RuntimeError("metrics subsystem down"),
        ):
            response = client.get("/api/v1/metrics/latest")
        _assert_503_error(response)

    def test_get_latest_metrics_raises_returns_503(self, client: object) -> None:
        """When get_latest_metrics() raises after import, return 503."""
        mock_api = MagicMock()
        mock_api.get_latest_metrics.side_effect = RuntimeError("metrics explosion")
        with patch("vetinari.dashboard.api.get_dashboard_api", return_value=mock_api):
            response = client.get("/api/v1/metrics/latest")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# GET /api/v1/metrics/timeseries  -  503 on subsystem failure
# ---------------------------------------------------------------------------


class TestApiMetricsTimeseries:
    """GET /api/v1/metrics/timeseries  -  subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_dashboard_api to raise; timeseries endpoint must return 503."""
        with patch(
            "vetinari.dashboard.api.get_dashboard_api",
            side_effect=RuntimeError("timeseries subsystem down"),
        ):
            response = client.get("/api/v1/metrics/timeseries?metric=latency")
        _assert_503_error(response)

    def test_get_timeseries_data_raises_returns_503(self, client: object) -> None:
        """When get_timeseries_data() raises after import, return 503."""
        mock_api = MagicMock()
        mock_api.get_timeseries_data.side_effect = RuntimeError("timeseries broken")
        with patch("vetinari.dashboard.api.get_dashboard_api", return_value=mock_api):
            response = client.get("/api/v1/metrics/timeseries?metric=latency")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# GET /api/v1/traces  -  503 on subsystem failure
# ---------------------------------------------------------------------------


class TestApiTraces:
    """GET /api/v1/traces  -  subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_dashboard_api to raise; traces endpoint must return 503."""
        with patch(
            "vetinari.dashboard.api.get_dashboard_api",
            side_effect=RuntimeError("traces subsystem down"),
        ):
            response = client.get("/api/v1/traces")
        _assert_503_error(response)

    def test_search_traces_raises_returns_503(self, client: object) -> None:
        """When search_traces() raises after import, return 503."""
        mock_api = MagicMock()
        mock_api.search_traces.side_effect = RuntimeError("search broken")
        with patch("vetinari.dashboard.api.get_dashboard_api", return_value=mock_api):
            response = client.get("/api/v1/traces")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# GET /api/v1/traces/{trace_id}  -  503 on subsystem failure
# ---------------------------------------------------------------------------


class TestApiTraceDetail:
    """GET /api/v1/traces/{trace_id}  -  subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_dashboard_api to raise; trace detail endpoint must return 503."""
        with patch(
            "vetinari.dashboard.api.get_dashboard_api",
            side_effect=RuntimeError("trace detail subsystem down"),
        ):
            response = client.get("/api/v1/traces/trace-abc-123")
        _assert_503_error(response)

    def test_get_trace_detail_raises_returns_503(self, client: object) -> None:
        """When get_trace_detail() raises after import, return 503."""
        mock_api = MagicMock()
        mock_api.get_trace_detail.side_effect = RuntimeError("trace lookup broken")
        with patch("vetinari.dashboard.api.get_dashboard_api", return_value=mock_api):
            response = client.get("/api/v1/traces/trace-abc-456")
        _assert_503_error(response)

    def test_missing_trace_returns_404(self, client: object) -> None:
        """When trace is not found, endpoint must return 404 not 503."""
        mock_api = MagicMock()
        mock_api.get_trace_detail.return_value = None
        with patch("vetinari.dashboard.api.get_dashboard_api", return_value=mock_api):
            response = client.get("/api/v1/traces/nonexistent-trace")
        assert response.status_code == 404, (
            f"Missing trace must return 404, got {response.status_code}. Body: {response.text[:400]}"
        )


# ---------------------------------------------------------------------------
# GET /api/v1/analytics/cost/top  -  503 on subsystem failure
# ---------------------------------------------------------------------------


class TestApiAnalyticsCostTop:
    """GET /api/v1/analytics/cost/top  -  subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_cost_tracker to raise; endpoint must return 503."""
        with patch(
            "vetinari.analytics.cost.get_cost_tracker",
            side_effect=RuntimeError("cost tracker unavailable"),
        ):
            response = client.get("/api/v1/analytics/cost/top")
        _assert_503_error(response)

    def test_get_top_agents_raises_returns_503(self, client: object) -> None:
        """When get_top_agents() raises after import, return 503."""
        mock_tracker = MagicMock()
        mock_tracker.get_top_agents.side_effect = RuntimeError("top agents broken")
        with patch("vetinari.analytics.cost.get_cost_tracker", return_value=mock_tracker):
            response = client.get("/api/v1/analytics/cost/top")
        _assert_503_error(response)
