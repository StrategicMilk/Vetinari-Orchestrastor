"""Mounted governance tests for dashboard read routes.

Proves dashboard GET routes return bounded 503 on subsystem failure, not raw
500 or a green-gauge/idle/empty 200 that hides the error from the caller.

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
# GET /api/v1/dashboard
# ---------------------------------------------------------------------------


class TestApiDashboard:
    """GET /api/v1/dashboard  -  subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_agent_dashboard to raise; endpoint must return 503 not 500."""
        with patch(
            "vetinari.dashboard.agent_dashboard.get_agent_dashboard",
            side_effect=RuntimeError("db is gone"),
        ):
            response = client.get("/api/v1/dashboard")
        _assert_503_error(response)

    def test_get_dashboard_data_raises_returns_503(self, client: object) -> None:
        """When get_dashboard_data() raises after successful import, return 503."""
        mock_dashboard = MagicMock()
        mock_dashboard.get_dashboard_data.side_effect = RuntimeError("boom")
        with patch(
            "vetinari.dashboard.agent_dashboard.get_agent_dashboard",
            return_value=mock_dashboard,
        ):
            response = client.get("/api/v1/dashboard")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# GET /api/v1/dashboard/health
# ---------------------------------------------------------------------------


class TestApiDashboardHealth:
    """GET /api/v1/dashboard/health  -  subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_agent_dashboard to raise; endpoint must return 503."""
        with patch(
            "vetinari.dashboard.agent_dashboard.get_agent_dashboard",
            side_effect=RuntimeError("health check error"),
        ):
            response = client.get("/api/v1/dashboard/health")
        _assert_503_error(response)

    def test_get_system_health_raises_returns_503(self, client: object) -> None:
        """When get_system_health() raises after import, return 503."""
        mock_dashboard = MagicMock()
        mock_dashboard.get_system_health.side_effect = RuntimeError("health exploded")
        with patch(
            "vetinari.dashboard.agent_dashboard.get_agent_dashboard",
            return_value=mock_dashboard,
        ):
            response = client.get("/api/v1/dashboard/health")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# GET /api/v1/dashboard/agents/{agent_type}
# ---------------------------------------------------------------------------


class TestApiDashboardAgent:
    """GET /api/v1/dashboard/agents/{agent_type}  -  subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_agent_dashboard to raise; endpoint must return 503."""
        with patch(
            "vetinari.dashboard.agent_dashboard.get_agent_dashboard",
            side_effect=RuntimeError("metrics unavailable"),
        ):
            response = client.get("/api/v1/dashboard/agents/worker")
        _assert_503_error(response)

    def test_get_agent_metrics_raises_returns_503(self, client: object) -> None:
        """When get_agent_metrics() raises, return 503."""
        mock_dashboard = MagicMock()
        mock_dashboard.get_agent_metrics.side_effect = RuntimeError("metrics db gone")
        with patch(
            "vetinari.dashboard.agent_dashboard.get_agent_dashboard",
            return_value=mock_dashboard,
        ):
            response = client.get("/api/v1/dashboard/agents/foreman")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# GET /api/v1/dashboard/quality/drift
# ---------------------------------------------------------------------------


class TestApiQualityDrift:
    """GET /api/v1/dashboard/quality/drift  -  subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_quality_drift_stats to raise; endpoint must return 503."""
        with patch(
            "vetinari.analytics.wiring.get_quality_drift_stats",
            side_effect=RuntimeError("drift db error"),
        ):
            response = client.get("/api/v1/dashboard/quality/drift")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# GET /api/v1/dashboard/quality/drift-stats (alias)
# ---------------------------------------------------------------------------


class TestApiQualityDriftStats:
    """GET /api/v1/dashboard/quality/drift-stats  -  subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_quality_drift_stats to raise; alias endpoint must return 503."""
        with patch(
            "vetinari.analytics.wiring.get_quality_drift_stats",
            side_effect=RuntimeError("drift stats error"),
        ):
            response = client.get("/api/v1/dashboard/quality/drift-stats")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# GET /api/v1/benchmarks/results/{run_id}
# ---------------------------------------------------------------------------


class TestApiBenchmarkResults:
    """GET /api/v1/benchmarks/results/{run_id}  -  subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch MetricStore to raise on instantiation; endpoint must return 503."""
        with patch(
            "vetinari.benchmarks.runner.MetricStore",
            side_effect=RuntimeError("store broken"),
        ):
            response = client.get("/api/v1/benchmarks/results/run-001")
        _assert_503_error(response)

    def test_load_results_raises_returns_503(self, client: object) -> None:
        """When load_results() raises, return 503."""
        mock_store = MagicMock()
        mock_store.load_results.side_effect = RuntimeError("load failed")
        with patch("vetinari.benchmarks.runner.MetricStore", return_value=mock_store):
            response = client.get("/api/v1/benchmarks/results/run-002")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# GET /api/v1/benchmarks/suites
# ---------------------------------------------------------------------------


class TestApiBenchmarkSuites:
    """GET /api/v1/benchmarks/suites  -  subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_default_runner to raise; endpoint must return 503."""
        with patch(
            "vetinari.benchmarks.runner.get_default_runner",
            side_effect=RuntimeError("runner init failed"),
        ):
            response = client.get("/api/v1/benchmarks/suites")
        _assert_503_error(response)

    def test_list_suites_raises_returns_503(self, client: object) -> None:
        """When list_suites() raises, return 503."""
        mock_runner = MagicMock()
        mock_runner.list_suites.side_effect = RuntimeError("list failed")
        with patch("vetinari.benchmarks.runner.get_default_runner", return_value=mock_runner):
            response = client.get("/api/v1/benchmarks/suites")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# GET /api/v1/benchmarks/suites/{suite_name}/comparison
# ---------------------------------------------------------------------------


class TestApiBenchmarkComparison:
    """GET /api/v1/benchmarks/suites/{suite_name}/comparison  -  subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_default_runner to raise; endpoint must return 503."""
        with patch(
            "vetinari.benchmarks.runner.get_default_runner",
            side_effect=RuntimeError("runner broken"),
        ):
            response = client.get("/api/v1/benchmarks/suites/my-suite/comparison")
        _assert_503_error(response)

    def test_get_last_comparison_raises_returns_503(self, client: object) -> None:
        """When get_last_comparison() raises, return 503."""
        mock_runner = MagicMock()
        mock_runner.get_last_comparison.side_effect = RuntimeError("comparison exploded")
        with patch("vetinari.benchmarks.runner.get_default_runner", return_value=mock_runner):
            response = client.get("/api/v1/benchmarks/suites/my-suite/comparison")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# GET /api/v1/pipeline/status  -  MUST NOT return green 200 on DB failure
# ---------------------------------------------------------------------------


class TestApiPipelineStatus:
    """GET /api/v1/pipeline/status  -  DB failure must return 503, not green 200."""

    def test_db_failure_returns_503_not_ok(self, client: object) -> None:
        """Patch get_connection to raise; endpoint must return 503 not {status:ok}.

        Proves the route no longer flattens DB failure into a green idle 200.
        """
        with patch(
            "vetinari.database.get_connection",
            side_effect=RuntimeError("db connection failed"),
        ):
            response = client.get("/api/v1/pipeline/status")
        _assert_503_error(response)

    def test_db_failure_does_not_return_status_ok(self, client: object) -> None:
        """Response body must not contain status='ok' when DB is unavailable."""
        with patch(
            "vetinari.database.get_connection",
            side_effect=RuntimeError("db gone"),
        ):
            response = client.get("/api/v1/pipeline/status")
        assert response.status_code != 200, (
            "Pipeline status must not return 200 when DB is unavailable  -  this hides failures behind a green response"
        )
        if response.status_code == 200:
            body = response.json()
            assert body.get("status") != "ok", (
                f"Pipeline status returned status=ok with all-idle stages despite DB failure: {body}"
            )


# ---------------------------------------------------------------------------
# GET /api/v1/dashboard/model-health  -  MUST NOT return green gauges on failure
# ---------------------------------------------------------------------------


class TestApiModelHealth:
    """GET /api/v1/dashboard/model-health  -  drift failure must return 503, not green gauges."""

    def test_drift_failure_returns_503_not_green(self, client: object) -> None:
        """Patch get_drift_ensemble to raise; endpoint must return 503 not green gauges.

        Proves the route no longer hides drift-ensemble failure behind green
        traffic-light gauge readings.
        """
        with patch(
            "vetinari.analytics.quality_drift.get_drift_ensemble",
            side_effect=RuntimeError("drift ensemble unavailable"),
        ):
            with patch(
                "vetinari.analytics.wiring.get_quality_drift_stats",
                return_value={"count": 0, "stddev": 0.0},
            ):
                response = client.get("/api/v1/dashboard/model-health")
        _assert_503_error(response)

    def test_drift_failure_does_not_return_green_level(self, client: object) -> None:
        """Response must not contain green gauge levels when drift ensemble fails."""
        with patch(
            "vetinari.analytics.quality_drift.get_drift_ensemble",
            side_effect=RuntimeError("drift broken"),
        ):
            with patch(
                "vetinari.analytics.wiring.get_quality_drift_stats",
                return_value={"count": 0, "stddev": 0.0},
            ):
                response = client.get("/api/v1/dashboard/model-health")
        # Should not be a 200  -  if it somehow is, the gauges must not all be green
        if response.status_code == 200:
            body = response.json()
            input_level = (body.get("input_drift") or {}).get("level")
            assert input_level != "green", (
                "Model health returned green input_drift gauge despite drift ensemble failure  -  "
                "this hides errors as healthy readings"
            )

    def test_quality_drift_stats_failure_returns_503(self, client: object) -> None:
        """Patch get_quality_drift_stats to raise; endpoint must return 503."""
        with patch(
            "vetinari.analytics.wiring.get_quality_drift_stats",
            side_effect=RuntimeError("stats unavailable"),
        ):
            response = client.get("/api/v1/dashboard/model-health")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# GET /api/v1/dashboard/welcome-back  -  MUST NOT return empty 200 on DB failure
# ---------------------------------------------------------------------------


class TestApiWelcomeBack:
    """GET /api/v1/dashboard/welcome-back  -  DB failure must return 503, not empty 200."""

    def test_db_failure_returns_503_not_ok(self, client: object) -> None:
        """Patch get_connection to raise; endpoint must return 503 not {status:ok}.

        Proves the route no longer hides DB failure behind an empty-summary 200.
        """
        with patch(
            "vetinari.database.get_connection",
            side_effect=RuntimeError("db is unavailable"),
        ):
            response = client.get("/api/v1/dashboard/welcome-back")
        _assert_503_error(response)

    def test_db_failure_does_not_return_status_ok(self, client: object) -> None:
        """Response body must not contain status='ok' when DB is unavailable."""
        with patch(
            "vetinari.database.get_connection",
            side_effect=RuntimeError("db gone"),
        ):
            response = client.get("/api/v1/dashboard/welcome-back")
        assert response.status_code != 200, (
            "Welcome-back must not return 200 when DB is unavailable  -  "
            "this hides failures behind an empty-summary response"
        )
        if response.status_code == 200:
            body = response.json()
            assert body.get("status") != "ok", (
                f"Welcome-back returned status=ok with empty data despite DB failure: {body}"
            )
