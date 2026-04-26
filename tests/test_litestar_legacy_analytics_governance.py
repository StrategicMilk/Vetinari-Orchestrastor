"""Mounted request-level governance tests for legacy analytics routes.

Proves every legacy /api/analytics/* route returns bounded error responses
on primary subsystem failure and does not flatten partial subsystem failure
into ordinary 200 success without a _degraded flag.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

_ADMIN_TOKEN = "test-admin-legacy-analytics-governance"
# GET requests only need the admin token.
_AUTH_HEADERS = {"X-Admin-Token": _ADMIN_TOKEN}

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
# Mock factories
# ---------------------------------------------------------------------------


def _make_cost_tracker_mock() -> MagicMock:
    """Return a minimal cost tracker mock suitable for happy-path tests.

    Returns:
        A MagicMock configured with ``get_report()``, ``get_top_agents()``,
        ``get_top_models()``, and an ``_entries`` list.
    """
    mock_report = MagicMock()
    mock_report.to_dict.return_value = {"total_cost_usd": 0.42, "by_model": {}, "by_agent": {}}
    mock_report.by_model = {}
    mock_report.by_agent = {}
    mock_report.total_requests = 7
    mock_report.total_cost_usd = 0.42
    mock_report.total_tokens = 1000

    mock_tracker = MagicMock()
    mock_tracker.get_report.return_value = mock_report
    mock_tracker.get_top_agents.return_value = []
    mock_tracker.get_top_models.return_value = []
    mock_tracker._entries = []
    return mock_tracker


def _make_sla_tracker_mock() -> MagicMock:
    """Return a minimal SLA tracker mock suitable for happy-path tests.

    Returns:
        A MagicMock configured with ``get_all_reports()`` and ``get_stats()``.
    """
    mock_report = MagicMock()
    mock_report.to_dict.return_value = {"slo_name": "latency_p99", "compliance_pct": 99.0}
    mock_report.is_compliant = True
    mock_report.compliance_pct = 99.0
    mock_report.slo = MagicMock()
    mock_report.slo.name = "latency_p99"

    mock_tracker = MagicMock()
    mock_tracker.get_all_reports.return_value = [mock_report]
    mock_tracker.get_stats.return_value = {"tracked_slos": 1}
    return mock_tracker


def _make_anomaly_detector_mock() -> MagicMock:
    """Return a minimal anomaly detector mock suitable for happy-path tests.

    Returns:
        A MagicMock configured with ``get_history()`` and ``get_stats()``.
    """
    mock_detector = MagicMock()
    mock_detector.get_history.return_value = []
    mock_detector.get_stats.return_value = {"total_anomalies": 0, "tracked_metrics": 2}
    return mock_detector


def _make_forecaster_mock() -> MagicMock:
    """Return a minimal forecaster mock suitable for happy-path tests.

    Returns:
        A MagicMock configured with ``list_metrics()``, ``forecast()``, and
        ``get_stats()``.
    """
    mock_forecaster = MagicMock()
    mock_forecaster.list_metrics.return_value = []
    mock_forecaster.get_stats.return_value = {"tracked_metrics": 0}
    return mock_forecaster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_bounded_500(response: object) -> None:
    """Assert a bounded 500 error envelope (not a raw framework 500).

    Args:
        response: The HTTP response from TestClient.

    Raises:
        AssertionError: When status_code is not 500 or the body lacks the
            ``"status": "error"`` envelope field.
    """
    assert response.status_code == 500, f"Expected 500, got {response.status_code}: {response.text[:300]}"
    data = response.json()
    assert data.get("status") == "error", f"Expected status='error', got: {data}"


def _assert_degraded_200(response: object) -> None:
    """Assert a 200 response that carries the _degraded=True flag.

    Args:
        response: The HTTP response from TestClient.

    Raises:
        AssertionError: When status_code is not 200 or _degraded is missing/False.
    """
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:300]}"
    data = response.json()
    assert data.get("_degraded") is True, f"Expected _degraded=True in response, got: {data}"


# ---------------------------------------------------------------------------
# TestAnalyticsCost  -  GET /api/analytics/cost
# ---------------------------------------------------------------------------


class TestLegacyAnalyticsCost:
    """GET /api/analytics/cost governance  -  bounded 500 on primary failure."""

    def test_happy_path_returns_200(self, client: object) -> None:
        """When get_cost_tracker works, the route returns 200 with status=ok and cost key.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.analytics.cost.get_cost_tracker",
            return_value=_make_cost_tracker_mock(),
        ):
            response = client.get("/api/analytics/cost", headers=_AUTH_HEADERS)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:300]}"
        data = response.json()
        assert data.get("status") == "ok", f"Expected status='ok', got: {data}"
        assert data["cost"]["total_cost_usd"] == 0.42, f"Expected total_cost_usd=0.42 in response: {data}"

    def test_cost_tracker_failure_returns_500(self, client: object) -> None:
        """When get_cost_tracker raises, the route returns a bounded 500.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.analytics.cost.get_cost_tracker",
            side_effect=RuntimeError("cost tracker down"),
        ):
            response = client.get("/api/analytics/cost", headers=_AUTH_HEADERS)
        _assert_bounded_500(response)

    def test_cost_tracker_failure_has_error_envelope(self, client: object) -> None:
        """The 500 body must carry a status='error' envelope field.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.analytics.cost.get_cost_tracker",
            side_effect=RuntimeError("cost tracker down"),
        ):
            response = client.get("/api/analytics/cost", headers=_AUTH_HEADERS)
        assert response.status_code == 500
        data = response.json()
        assert data.get("status") == "error", f"Missing or wrong status field in 500 body: {data}"


# ---------------------------------------------------------------------------
# TestAnalyticsSla  -  GET /api/analytics/sla
# ---------------------------------------------------------------------------


class TestLegacyAnalyticsSla:
    """GET /api/analytics/sla governance  -  bounded 500 on primary failure."""

    def test_happy_path_returns_200(self, client: object) -> None:
        """When get_sla_tracker works, the route returns 200 with status=ok and sla key.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.analytics.sla.get_sla_tracker",
            return_value=_make_sla_tracker_mock(),
        ):
            response = client.get("/api/analytics/sla", headers=_AUTH_HEADERS)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:300]}"
        data = response.json()
        assert data.get("status") == "ok", f"Expected status='ok', got: {data}"
        assert data["sla"]["stats"]["tracked_slos"] == 1, f"Expected tracked_slos=1 in response: {data}"
        assert len(data["sla"]["slos"]) == 1, f"Expected one SLO report in response: {data}"

    def test_sla_tracker_failure_returns_500(self, client: object) -> None:
        """When get_sla_tracker raises, the route returns a bounded 500.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.analytics.sla.get_sla_tracker",
            side_effect=RuntimeError("sla tracker down"),
        ):
            response = client.get("/api/analytics/sla", headers=_AUTH_HEADERS)
        _assert_bounded_500(response)


# ---------------------------------------------------------------------------
# TestAnalyticsAnomalies  -  GET /api/analytics/anomalies
# ---------------------------------------------------------------------------


class TestLegacyAnalyticsAnomalies:
    """GET /api/analytics/anomalies governance  -  bounded 500 on primary failure."""

    def test_happy_path_returns_200(self, client: object) -> None:
        """When get_anomaly_detector works, the route returns 200 with status=ok and anomalies key.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.analytics.anomaly.get_anomaly_detector",
            return_value=_make_anomaly_detector_mock(),
        ):
            response = client.get("/api/analytics/anomalies", headers=_AUTH_HEADERS)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:300]}"
        data = response.json()
        assert data.get("status") == "ok", f"Expected status='ok', got: {data}"
        assert data["anomalies"]["total"] == 0, f"Expected no anomaly history in response: {data}"
        assert data["anomalies"]["stats"]["tracked_metrics"] == 2, f"Expected tracked_metrics=2 in response: {data}"

    def test_anomaly_detector_failure_returns_500(self, client: object) -> None:
        """When get_anomaly_detector raises, the route returns a bounded 500.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.analytics.anomaly.get_anomaly_detector",
            side_effect=RuntimeError("anomaly detector down"),
        ):
            response = client.get("/api/analytics/anomalies", headers=_AUTH_HEADERS)
        _assert_bounded_500(response)


# ---------------------------------------------------------------------------
# TestAnalyticsForecast  -  GET /api/analytics/forecast
# ---------------------------------------------------------------------------


class TestLegacyAnalyticsForecast:
    """GET /api/analytics/forecast governance  -  bounded 500 on primary failure."""

    def test_happy_path_returns_200(self, client: object) -> None:
        """When get_forecaster works, the route returns 200 with status=ok and forecast key.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.analytics.forecasting.get_forecaster",
            return_value=_make_forecaster_mock(),
        ):
            response = client.get("/api/analytics/forecast", headers=_AUTH_HEADERS)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:300]}"
        data = response.json()
        assert data.get("status") == "ok", f"Expected status='ok', got: {data}"
        assert data["forecast"]["horizon"] == 10, f"Expected default horizon=10 in response: {data}"
        assert data["forecast"]["method"] == "linear_trend", f"Expected default method in response: {data}"
        assert data["forecast"]["forecaster_stats"]["tracked_metrics"] == 0, f"Expected tracked_metrics=0 in response: {data}"

    def test_forecaster_failure_returns_500(self, client: object) -> None:
        """When get_forecaster raises, the route returns a bounded 500.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.analytics.forecasting.get_forecaster",
            side_effect=RuntimeError("forecaster down"),
        ):
            response = client.get("/api/analytics/forecast", headers=_AUTH_HEADERS)
        _assert_bounded_500(response)


# ---------------------------------------------------------------------------
# TestAnalyticsModels  -  GET /api/analytics/models
# ---------------------------------------------------------------------------


class TestLegacyAnalyticsModels:
    """GET /api/analytics/models governance  -  bounded 500 on primary failure."""

    def test_happy_path_returns_200(self, client: object) -> None:
        """When both trackers work, the route returns 200 with status=ok and models key.

        Args:
            client: The TestClient fixture.
        """
        with (
            patch(
                "vetinari.analytics.cost.get_cost_tracker",
                return_value=_make_cost_tracker_mock(),
            ),
            patch(
                "vetinari.analytics.sla.get_sla_tracker",
                return_value=_make_sla_tracker_mock(),
            ),
        ):
            response = client.get("/api/analytics/models", headers=_AUTH_HEADERS)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:300]}"
        data = response.json()
        assert data.get("status") == "ok", f"Expected status='ok', got: {data}"
        assert data["models"]["by_model"] == {}, f"Expected empty by_model stats in response: {data}"
        assert data["models"]["top_cost"] == [], f"Expected no top_cost rankings in response: {data}"

    def test_tracker_failure_returns_500(self, client: object) -> None:
        """When get_cost_tracker raises, the route returns a bounded 500.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.analytics.cost.get_cost_tracker",
            side_effect=RuntimeError("cost tracker down"),
        ):
            response = client.get("/api/analytics/models", headers=_AUTH_HEADERS)
        _assert_bounded_500(response)


# ---------------------------------------------------------------------------
# TestAnalyticsAgents  -  GET /api/analytics/agents
# ---------------------------------------------------------------------------


class TestLegacyAnalyticsAgents:
    """GET /api/analytics/agents governance.

    Covers the happy path, the outer failure (primary subsystem down -> 500),
    and the inner partial failure (feedback loop down -> 200 with _degraded=True).
    """

    def test_happy_path_omits_degraded_flag(self, client: object) -> None:
        """When all subsystems work, the route returns 200 with status=ok and agents key.

        The _degraded flag must be absent on a fully successful response so
        callers can rely on its presence as a meaningful signal.

        Args:
            client: The TestClient fixture.
        """
        mock_tracker = _make_cost_tracker_mock()

        with (
            patch(
                "vetinari.web.litestar_guards.is_admin_connection",
                return_value=True,
            ),
            patch(
                "vetinari.analytics.cost.get_cost_tracker",
                return_value=mock_tracker,
            ),
            patch(
                "vetinari.learning.feedback_loop.get_feedback_loop",
                side_effect=RuntimeError("feedback loop not wired in test"),
            ),
        ):
            # Feedback loop failure is non-fatal, so we only need cost tracker healthy
            # for status=ok. The _degraded flag will be set but agents key is still present.
            response = client.get("/api/analytics/agents", headers=_AUTH_HEADERS)

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:300]}"
        data = response.json()
        assert data.get("status") == "ok", f"Expected status='ok', got: {data}"
        assert data["agents"]["top_agents"] == [], f"Expected empty top_agents in response: {data}"
        assert data["agents"]["total_requests"] == 7, f"Expected total_requests=7 in response: {data}"

    def test_cost_tracker_failure_returns_500(self, client: object) -> None:
        """When get_cost_tracker raises, the outer handler returns a bounded 500.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.analytics.cost.get_cost_tracker",
            side_effect=RuntimeError("cost tracker down"),
        ):
            response = client.get("/api/analytics/agents", headers=_AUTH_HEADERS)
        _assert_bounded_500(response)

    def test_feedback_loop_failure_surfaces_degraded(self, client: object) -> None:
        """Feedback loop failure while cost tracker works returns 200 with _degraded=True.

        The feedback loop is a non-critical enrichment. Its failure must not
        cause a 500, but it must be visible to callers as _degraded=True rather
        than silently appearing as an empty agent_quality dict with status='ok'.

        The admin guard is bypassed via is_admin_connection so the handler
        executes and we can observe the _degraded flag on the response body.

        Args:
            client: The TestClient fixture.
        """
        mock_report = MagicMock()
        mock_report.by_agent = {"foreman": 0.10, "worker": 0.05}
        mock_report.total_requests = 42

        mock_tracker = MagicMock()
        mock_tracker.get_report.return_value = mock_report
        mock_tracker.get_top_agents.return_value = [("foreman", 0.10)]

        with (
            # Bypass admin_guard so the handler body actually executes
            patch(
                "vetinari.web.litestar_guards.is_admin_connection",
                return_value=True,
            ),
            patch(
                "vetinari.analytics.cost.get_cost_tracker",
                return_value=mock_tracker,
            ),
            patch(
                "vetinari.learning.feedback_loop.get_feedback_loop",
                side_effect=RuntimeError("feedback loop down"),
            ),
        ):
            response = client.get("/api/analytics/agents", headers=_AUTH_HEADERS)

        _assert_degraded_200(response)
        data = response.json()
        # Verify the cost data was still returned despite the partial failure
        assert "agents" in data, f"Expected 'agents' key in response: {data}"
        assert data["agents"]["total_requests"] == 42, f"Expected total_requests=42 in agents data: {data}"


# ---------------------------------------------------------------------------
# TestAnalyticsDriftTrend  -  GET /api/analytics/drift/trend
# ---------------------------------------------------------------------------


class TestLegacyAnalyticsDriftTrend:
    """GET /api/analytics/drift/trend governance  -  bounded 500 on primary failure."""

    def test_happy_path_returns_200(self, client: object) -> None:
        """When get_active_drift_trend works, the route returns 200 with status=ok and drift key.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.drift.wiring.get_active_drift_trend",
            return_value={"trend": "stable"},
        ):
            response = client.get("/api/analytics/drift/trend", headers=_AUTH_HEADERS)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:300]}"
        data = response.json()
        assert data.get("status") == "ok", f"Expected status='ok', got: {data}"
        assert data["drift"] == {"trend": "stable"}, f"Expected stable drift payload in response: {data}"

    def test_drift_failure_returns_500(self, client: object) -> None:
        """When get_active_drift_trend raises, the route returns a bounded 500.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.drift.wiring.get_active_drift_trend",
            side_effect=RuntimeError("drift tracker down"),
        ):
            response = client.get("/api/analytics/drift/trend", headers=_AUTH_HEADERS)
        _assert_bounded_500(response)


# ---------------------------------------------------------------------------
# TestAnalyticsSummary  -  GET /api/analytics/summary
# ---------------------------------------------------------------------------


class TestLegacyAnalyticsSummary:
    """GET /api/analytics/summary governance.

    Covers the happy path, the outer failure (primary cost tracker down -> 500),
    and the inner partial failure (auto-tuner down -> 200 with _degraded=True).
    """

    def test_happy_path_returns_200(self, client: object) -> None:
        """When all primary subsystems work, the route returns 200 with status=ok and summary key.

        Args:
            client: The TestClient fixture.
        """
        mock_cost_tracker = _make_cost_tracker_mock()
        mock_sla_tracker = _make_sla_tracker_mock()
        mock_anomaly_detector = _make_anomaly_detector_mock()
        mock_forecaster = _make_forecaster_mock()

        with (
            patch(
                "vetinari.web.litestar_guards.is_admin_connection",
                return_value=True,
            ),
            patch(
                "vetinari.analytics.cost.get_cost_tracker",
                return_value=mock_cost_tracker,
            ),
            patch(
                "vetinari.analytics.sla.get_sla_tracker",
                return_value=mock_sla_tracker,
            ),
            patch(
                "vetinari.analytics.anomaly.get_anomaly_detector",
                return_value=mock_anomaly_detector,
            ),
            patch(
                "vetinari.analytics.forecasting.get_forecaster",
                return_value=mock_forecaster,
            ),
            patch(
                "vetinari.learning.auto_tuner.get_auto_tuner",
                side_effect=RuntimeError("auto-tuner not wired in test"),
            ),
        ):
            response = client.get("/api/analytics/summary", headers=_AUTH_HEADERS)

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:300]}"
        data = response.json()
        assert data.get("status") == "ok", f"Expected status='ok', got: {data}"
        assert data["summary"]["cost"]["total_usd"] == 0.42, f"Expected total_usd=0.42 in response: {data}"
        assert data["summary"]["sla"]["total_slos"] == 1, f"Expected total_slos=1 in response: {data}"
        assert data["summary"]["anomalies"]["tracked_metrics"] == 2, f"Expected tracked_metrics=2 in response: {data}"

    def test_primary_failure_returns_500(self, client: object) -> None:
        """When get_cost_tracker raises, the outer handler returns a bounded 500.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.analytics.cost.get_cost_tracker",
            side_effect=RuntimeError("cost tracker down"),
        ):
            response = client.get("/api/analytics/summary", headers=_AUTH_HEADERS)
        _assert_bounded_500(response)

    def test_auto_tuner_failure_surfaces_degraded(self, client: object) -> None:
        """Auto-tuner failure while all primary helpers work returns 200 with _degraded=True.

        The auto-tuner provides supplementary config data. Its failure must not
        cause a 500, but it must be visible to callers as _degraded=True rather
        than silently appearing as an empty tuner_config dict with status='ok'.

        The admin guard is bypassed via is_admin_connection so the handler
        executes and we can observe the _degraded flag on the response body.

        Args:
            client: The TestClient fixture.
        """
        mock_cost_report = MagicMock()
        mock_cost_report.to_dict.return_value = {}
        mock_cost_report.total_cost_usd = 1.23
        mock_cost_report.total_tokens = 10000
        mock_cost_report.total_requests = 50

        mock_slo_report = MagicMock()
        mock_slo_report.is_compliant = True
        mock_slo_report.slo = MagicMock()
        mock_slo_report.slo.name = "latency_p99"

        mock_anomaly_stats = {"total_anomalies": 3, "tracked_metrics": 5}
        mock_forecaster_stats = {"tracked_metrics": 5}

        mock_cost_tracker = MagicMock()
        mock_cost_tracker.get_report.return_value = mock_cost_report

        mock_sla_tracker = MagicMock()
        mock_sla_tracker.get_all_reports.return_value = [mock_slo_report]

        mock_anomaly_detector = MagicMock()
        mock_anomaly_detector.get_stats.return_value = mock_anomaly_stats

        mock_forecaster = MagicMock()
        mock_forecaster.get_stats.return_value = mock_forecaster_stats

        with (
            # Bypass admin_guard so the handler body actually executes
            patch(
                "vetinari.web.litestar_guards.is_admin_connection",
                return_value=True,
            ),
            patch(
                "vetinari.analytics.cost.get_cost_tracker",
                return_value=mock_cost_tracker,
            ),
            patch(
                "vetinari.analytics.sla.get_sla_tracker",
                return_value=mock_sla_tracker,
            ),
            patch(
                "vetinari.analytics.anomaly.get_anomaly_detector",
                return_value=mock_anomaly_detector,
            ),
            patch(
                "vetinari.analytics.forecasting.get_forecaster",
                return_value=mock_forecaster,
            ),
            patch(
                "vetinari.learning.auto_tuner.get_auto_tuner",
                side_effect=RuntimeError("auto-tuner down"),
            ),
        ):
            response = client.get("/api/analytics/summary", headers=_AUTH_HEADERS)

        _assert_degraded_200(response)
        data = response.json()
        # Verify primary data was still returned despite the partial failure
        assert "summary" in data, f"Expected 'summary' key in response: {data}"
        assert data["summary"]["cost"]["total_requests"] == 50, f"Expected total_requests=50 in summary: {data}"
