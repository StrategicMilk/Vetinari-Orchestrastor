"""Mounted governance tests for analytics read routes.

Verifies that every analytics route returns 503 (not a raw 500 or a silent
200 with an ``"unavailable"`` sentinel) when its backing subsystem raises an
exception.  Also tests the partial-degraded behaviour for the overview and
alerts endpoints.

All requests go through the real Litestar TestClient stack so router
registration, middleware, guards, and serialisation are all exercised.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from litestar.testing import TestClient

# ---------------------------------------------------------------------------
# Admin auth constants
# ---------------------------------------------------------------------------

# A token value set in the env so admin_guard passes in tests.
_ADMIN_TOKEN = "test-analytics-governance-token"
_ADMIN_HEADERS = {"X-Admin-Token": _ADMIN_TOKEN}


# ---------------------------------------------------------------------------
# App / client fixtures
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _noop_lifespan(app: Any):
    """Drop-in lifespan that skips all subsystem wiring."""
    yield


@pytest.fixture(scope="module")
def app():
    """Create a Litestar app with startup/shutdown wiring suppressed.

    Uses module scope so the ~300-handler app is only built once for this
    test module.  The VETINARI_ADMIN_TOKEN env var is set before creation so
    admin_guard uses token-based auth, which the test client can satisfy via
    the X-Admin-Token header.
    """
    with (
        patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}),
        patch("vetinari.web.litestar_app._lifespan", _noop_lifespan),
        patch("vetinari.web.litestar_app._register_shutdown_handlers"),
    ):
        from vetinari.web.litestar_app import create_app

        return create_app(debug=True)


@pytest.fixture
def client(app):
    """Yield a fresh TestClient per test."""
    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        with TestClient(app=app) as tc:
            yield tc


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _raises(exc: type[Exception] = RuntimeError, msg: str = "subsystem down") -> MagicMock:
    """Return a MagicMock that raises ``exc`` when called."""
    m = MagicMock(side_effect=exc(msg))
    return m


# ---------------------------------------------------------------------------
# Problem A: 6 routes with no error handling → must return 503 on failure
# ---------------------------------------------------------------------------


class TestCostRouteGovernance:
    """GET /api/v1/analytics/cost — must return 503 when cost tracker fails."""

    def test_cost_data_returns_503_when_tracker_raises(self, client: TestClient) -> None:
        """Subsystem error must produce 503, not an unhandled 500."""
        with patch(
            "vetinari.analytics.cost.get_cost_tracker",
            side_effect=RuntimeError("cost db down"),
        ):
            resp = client.get("/api/v1/analytics/cost", headers=_ADMIN_HEADERS)
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "error"


class TestSlaRouteGovernance:
    """GET /api/v1/analytics/sla — must return 503 when SLA tracker fails."""

    def test_sla_data_returns_503_when_tracker_raises(self, client: TestClient) -> None:
        with patch(
            "vetinari.analytics.sla.get_sla_tracker",
            side_effect=RuntimeError("sla db down"),
        ):
            resp = client.get("/api/v1/analytics/sla", headers=_ADMIN_HEADERS)
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "error"


class TestAnomalyRouteGovernance:
    """GET /api/v1/analytics/anomalies — must return 503 when detector fails."""

    def test_anomaly_data_returns_503_when_detector_raises(self, client: TestClient) -> None:
        with patch(
            "vetinari.analytics.anomaly.get_anomaly_detector",
            side_effect=RuntimeError("anomaly service down"),
        ):
            resp = client.get("/api/v1/analytics/anomalies", headers=_ADMIN_HEADERS)
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "error"


class TestForecastRouteGovernance:
    """GET /api/v1/analytics/forecasts — must return 503 when forecaster fails."""

    def test_forecast_data_returns_503_when_forecaster_raises(self, client: TestClient) -> None:
        with patch(
            "vetinari.analytics.forecasting.get_forecaster",
            side_effect=RuntimeError("forecaster down"),
        ):
            resp = client.get("/api/v1/analytics/forecasts", headers=_ADMIN_HEADERS)
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "error"


class TestModelSlaCompliancePathRouteGovernance:
    """GET /api/v1/analytics/sla/model/{model_id}/compliance — must return 503."""

    def test_model_compliance_returns_503_when_tracker_raises(self, client: TestClient) -> None:
        with patch(
            "vetinari.analytics.sla.get_sla_tracker",
            side_effect=RuntimeError("sla tracker down"),
        ):
            resp = client.get(
                "/api/v1/analytics/sla/model/test-model/compliance",
                headers=_ADMIN_HEADERS,
            )
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "error"

    def test_model_compliance_returns_404_when_no_observations(self, client: TestClient) -> None:
        """Existing 404 path (no observations) must still work after hardening."""
        mock_tracker = MagicMock()
        mock_tracker.get_model_compliance.return_value = None
        with patch(
            "vetinari.analytics.sla.get_sla_tracker",
            return_value=mock_tracker,
        ):
            resp = client.get(
                "/api/v1/analytics/sla/model/unknown-model/compliance",
                headers=_ADMIN_HEADERS,
            )
        assert resp.status_code == 404


class TestModelSlaComplianceQueryRouteGovernance:
    """GET /api/v1/analytics/sla/model-compliance — must return 503."""

    def test_model_compliance_query_returns_503_when_tracker_raises(self, client: TestClient) -> None:
        with patch(
            "vetinari.analytics.sla.get_sla_tracker",
            side_effect=RuntimeError("sla tracker down"),
        ):
            resp = client.get(
                "/api/v1/analytics/sla/model-compliance?model_id=m1",
                headers=_ADMIN_HEADERS,
            )
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "error"

    def test_model_compliance_query_returns_400_when_model_id_missing(self, client: TestClient) -> None:
        """Existing 400 path (missing model_id) must still work after hardening."""
        resp = client.get(
            "/api/v1/analytics/sla/model-compliance",
            headers=_ADMIN_HEADERS,
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Problem B: 5 routes that silently flatten failures into 200
# ---------------------------------------------------------------------------


class TestAnalyticsOverviewGovernance:
    """GET /api/v1/analytics/overview — partial and total failure behaviour."""

    def test_overview_returns_503_when_all_subsystems_fail(self, client: TestClient) -> None:
        """All 3 sections failing must produce 503, not a 200 with sentinels."""
        with (
            patch(
                "vetinari.analytics.cost.get_cost_tracker",
                side_effect=RuntimeError("cost down"),
            ),
            patch(
                "vetinari.analytics.sla.get_sla_tracker",
                side_effect=RuntimeError("sla down"),
            ),
            patch(
                "vetinari.analytics.anomaly.get_anomaly_detector",
                side_effect=RuntimeError("anomaly down"),
            ),
        ):
            resp = client.get("/api/v1/analytics/overview", headers=_ADMIN_HEADERS)
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "error"

    def test_overview_returns_degraded_flag_when_one_subsystem_fails(self, client: TestClient) -> None:
        """One section failing must return 200 with ``_degraded: True``."""
        mock_report = MagicMock()
        mock_report.to_dict.return_value = {"total_cost": 1.0}
        mock_tracker = MagicMock()
        mock_tracker.get_report.return_value = mock_report
        mock_tracker.get_all_reports.return_value = []

        mock_detector = MagicMock()
        mock_detector.get_stats.return_value = {"total_anomalies": 0}

        with (
            patch("vetinari.analytics.cost.get_cost_tracker", return_value=mock_tracker),
            patch("vetinari.analytics.sla.get_sla_tracker", return_value=mock_tracker),
            patch(
                "vetinari.analytics.anomaly.get_anomaly_detector",
                side_effect=RuntimeError("anomaly down"),
            ),
        ):
            resp = client.get("/api/v1/analytics/overview", headers=_ADMIN_HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("_degraded") is True
        assert body["cost"] == {"total_cost": 1.0}
        assert body["sla"] == {"report_count": 0}
        assert body["anomalies"] == {"status": "unavailable"}

    def test_overview_returns_200_without_degraded_flag_when_all_succeed(self, client: TestClient) -> None:
        """All sections succeeding must return 200 without ``_degraded``."""
        mock_report = MagicMock()
        mock_report.to_dict.return_value = {"total_cost": 0.0}
        mock_cost_tracker = MagicMock()
        mock_cost_tracker.get_report.return_value = mock_report

        mock_sla_tracker = MagicMock()
        mock_sla_tracker.get_all_reports.return_value = []

        mock_detector = MagicMock()
        mock_detector.get_stats.return_value = {"total_anomalies": 0}

        with (
            patch("vetinari.analytics.cost.get_cost_tracker", return_value=mock_cost_tracker),
            patch("vetinari.analytics.sla.get_sla_tracker", return_value=mock_sla_tracker),
            patch("vetinari.analytics.anomaly.get_anomaly_detector", return_value=mock_detector),
        ):
            resp = client.get("/api/v1/analytics/overview", headers=_ADMIN_HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert "_degraded" not in body


class TestAdapterStatsGovernance:
    """GET /api/v1/analytics/adapters — must return 503 not 200 sentinel.

    The vetinari.adapters module is an optional subsystem that may not be
    installed. When the import raises, the handler must return 503 instead
    of the old silent {"status": "unavailable"} 200 sentinel.
    """

    def test_adapter_stats_returns_503_when_subsystem_not_installed(self, client: TestClient) -> None:
        """ImportError on vetinari.adapters must produce 503, not 200 sentinel."""
        # The vetinari.adapters module may not exist in the test environment.
        # Simulate the ImportError by temporarily removing it from sys.modules
        # (if present) and replacing with a stub that raises ImportError.
        import sys

        original = sys.modules.get("vetinari.adapters")
        # Force an ImportError by making the module lookup fail
        sys.modules["vetinari.adapters"] = None  # type: ignore[assignment]
        try:
            resp = client.get("/api/v1/analytics/adapters", headers=_ADMIN_HEADERS)
        finally:
            if original is None:
                sys.modules.pop("vetinari.adapters", None)
            else:
                sys.modules["vetinari.adapters"] = original
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "error"
        # Must NOT be the old silent sentinel format
        assert body.get("reason") is None


class TestMemoryStatsGovernance:
    """GET /api/v1/analytics/memory — must return 503 not 200 sentinel.

    The vetinari.memory module is an optional subsystem. ImportError or
    RuntimeError from get_memory_manager() must produce 503.
    """

    def test_memory_stats_returns_503_when_subsystem_not_installed(self, client: TestClient) -> None:
        """ImportError on vetinari.memory must produce 503, not 200 sentinel."""
        import sys

        original = sys.modules.get("vetinari.memory")
        sys.modules["vetinari.memory"] = None  # type: ignore[assignment]
        try:
            resp = client.get("/api/v1/analytics/memory", headers=_ADMIN_HEADERS)
        finally:
            if original is None:
                sys.modules.pop("vetinari.memory", None)
            else:
                sys.modules["vetinari.memory"] = original
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "error"
        assert body.get("reason") is None


class TestPlanStatsGovernance:
    """GET /api/v1/analytics/plan — must return 503 not 200 sentinel.

    The vetinari.analytics.plan_analytics module is an optional subsystem.
    ImportError or RuntimeError must produce 503.
    """

    def test_plan_stats_returns_503_when_subsystem_not_installed(self, client: TestClient) -> None:
        """ImportError on plan_analytics must produce 503, not 200 sentinel."""
        import sys

        original = sys.modules.get("vetinari.analytics.plan_analytics")
        sys.modules["vetinari.analytics.plan_analytics"] = None  # type: ignore[assignment]
        try:
            resp = client.get("/api/v1/analytics/plan", headers=_ADMIN_HEADERS)
        finally:
            if original is None:
                sys.modules.pop("vetinari.analytics.plan_analytics", None)
            else:
                sys.modules["vetinari.analytics.plan_analytics"] = original
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "error"
        assert body.get("reason") is None


class TestActiveAlertsGovernance:
    """GET /api/v1/analytics/alerts — partial and total failure behaviour."""

    def test_alerts_returns_503_when_both_subsystems_fail(self, client: TestClient) -> None:
        """All alert sources failing must produce 503."""
        with (
            patch(
                "vetinari.analytics.anomaly.get_anomaly_detector",
                side_effect=RuntimeError("anomaly down"),
            ),
            patch(
                "vetinari.analytics.sla.get_sla_tracker",
                side_effect=RuntimeError("sla down"),
            ),
            patch(
                "vetinari.dashboard.alerts.get_alert_engine",
                side_effect=RuntimeError("alert_engine down"),
            ),
        ):
            resp = client.get("/api/v1/analytics/alerts", headers=_ADMIN_HEADERS)
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "error"

    def test_alerts_returns_degraded_flag_when_one_source_fails(self, client: TestClient) -> None:
        """One source failing must return 200 with ``_degraded: True``."""
        mock_detector = MagicMock()
        mock_detector.get_recent_anomalies.return_value = [{"metric": "latency", "detail": "spike"}]

        with (
            patch("vetinari.analytics.anomaly.get_anomaly_detector", return_value=mock_detector),
            patch(
                "vetinari.analytics.sla.get_sla_tracker",
                side_effect=RuntimeError("sla down"),
            ),
        ):
            resp = client.get("/api/v1/analytics/alerts", headers=_ADMIN_HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("_degraded") is True
        # Anomaly alerts from the working source must still be present
        assert len(body["alerts"]) >= 1
        assert body["alerts"][0]["source"] == "anomaly"

    def test_alerts_returns_200_without_degraded_when_all_succeed(self, client: TestClient) -> None:
        """All sources working must return 200 without ``_degraded``."""
        mock_detector = MagicMock()
        mock_detector.get_recent_anomalies.return_value = []

        mock_sla = MagicMock()
        mock_sla.get_recent_breaches.return_value = []

        with (
            patch("vetinari.analytics.anomaly.get_anomaly_detector", return_value=mock_detector),
            patch("vetinari.analytics.sla.get_sla_tracker", return_value=mock_sla),
        ):
            resp = client.get("/api/v1/analytics/alerts", headers=_ADMIN_HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert "_degraded" not in body
        assert body["alerts"] == []
