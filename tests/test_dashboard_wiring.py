"""Tests for S30 — Wire Dashboard to Real System State.

Covers:
- AgentDashboard reads live circuit breaker state
- AgentDashboard reads real cost data via CostTracker.get_report()
- SystemHealth reads live circuit breaker and 24h cost summary
- AlertEngine persists alert history to SQLite
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vetinari.dashboard.agent_dashboard import AgentDashboard
from vetinari.dashboard.alerts import (
    AlertCondition,
    AlertEngine,
    AlertRecord,
    AlertSeverity,
    AlertThreshold,
    reset_alert_engine,
)
from vetinari.types import AgentType

# ── AgentDashboard cost wiring ───────────────────────────────────────────────


class TestAgentDashboardCostWiring:
    """AgentDashboard pulls cost data from CostTracker.get_report(agent=...)."""

    def test_cost_data_populates_metrics(self) -> None:
        """Metrics reflect real cost tracker report values."""
        mock_report = MagicMock()
        mock_report.total_requests = 10
        mock_report.total_cost_usd = 0.05
        mock_report.total_tokens = 2000

        mock_tracker = MagicMock()
        mock_tracker.get_report.return_value = mock_report

        with (
            patch(
                "vetinari.resilience.get_circuit_breaker_registry",
                side_effect=RuntimeError("skip"),
            ),
            patch(
                "vetinari.analytics.cost.get_cost_tracker",
                return_value=mock_tracker,
            ),
            patch(
                "vetinari.telemetry.get_telemetry_collector",
                side_effect=RuntimeError("skip"),
            ),
        ):
            dashboard = AgentDashboard()
            metrics = dashboard.get_agent_metrics(AgentType.FOREMAN.value)

        mock_tracker.get_report.assert_called_once_with(agent=AgentType.FOREMAN.value)
        assert metrics.total_cost == 0.05
        assert metrics.total_tokens == 2000
        assert metrics.total_executions == 10

    def test_cost_data_zero_requests_no_overwrite(self) -> None:
        """When cost report has zero requests, metrics stay at defaults."""
        mock_report = MagicMock()
        mock_report.total_requests = 0

        mock_tracker = MagicMock()
        mock_tracker.get_report.return_value = mock_report

        with (
            patch(
                "vetinari.resilience.get_circuit_breaker_registry",
                side_effect=RuntimeError("skip"),
            ),
            patch(
                "vetinari.analytics.cost.get_cost_tracker",
                return_value=mock_tracker,
            ),
            patch(
                "vetinari.telemetry.get_telemetry_collector",
                side_effect=RuntimeError("skip"),
            ),
        ):
            dashboard = AgentDashboard()
            metrics = dashboard.get_agent_metrics(AgentType.WORKER.value)

        assert metrics.total_cost == 0.0
        assert metrics.total_tokens == 0
        assert metrics.total_executions == 0


# ── AgentDashboard circuit breaker wiring ────────────────────────────────────


class TestAgentDashboardCircuitBreakerWiring:
    """AgentDashboard reads live circuit breaker state."""

    def test_circuit_breaker_state_reflected(self) -> None:
        """Metrics reflect real circuit breaker state value."""
        mock_cb = MagicMock()
        mock_cb.state.value = "open"

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_cb

        mock_report = MagicMock()
        mock_report.total_requests = 0
        mock_tracker = MagicMock()
        mock_tracker.get_report.return_value = mock_report

        with (
            patch(
                "vetinari.resilience.get_circuit_breaker_registry",
                return_value=mock_registry,
            ),
            patch(
                "vetinari.analytics.cost.get_cost_tracker",
                return_value=mock_tracker,
            ),
            patch(
                "vetinari.telemetry.get_telemetry_collector",
                side_effect=RuntimeError("skip"),
            ),
        ):
            dashboard = AgentDashboard()
            metrics = dashboard.get_agent_metrics(AgentType.INSPECTOR.value)

        assert metrics.circuit_breaker_state == "open"
        mock_registry.get.assert_called_with(AgentType.INSPECTOR.value)


# ── SystemHealth wiring ──────────────────────────────────────────────────────


class TestSystemHealthWiring:
    """SystemHealth pulls live circuit breaker and cost data."""

    def test_system_health_detects_open_breakers(self) -> None:
        """Open circuit breakers are reflected in health status."""
        mock_cb_closed = MagicMock()
        mock_cb_closed.state.value = "closed"

        mock_cb_open = MagicMock()
        mock_cb_open.state.value = "open"

        mock_registry = MagicMock()

        def get_cb(agent_type: str) -> MagicMock:
            if agent_type == AgentType.WORKER.value:
                return mock_cb_open
            return mock_cb_closed

        mock_registry.get.side_effect = get_cb

        mock_report = MagicMock()
        mock_report.total_cost_usd = 0.10
        mock_report.total_tokens = 5000
        mock_tracker = MagicMock()
        mock_tracker.get_report.return_value = mock_report

        with (
            patch(
                "vetinari.resilience.get_circuit_breaker_registry",
                return_value=mock_registry,
            ),
            patch(
                "vetinari.analytics.cost.get_cost_tracker",
                return_value=mock_tracker,
            ),
        ):
            dashboard = AgentDashboard()
            health = dashboard.get_system_health()

        assert health.status == "degraded"
        assert AgentType.WORKER.value in health.open_circuit_breakers
        assert health.healthy_agents == 2
        assert health.total_cost_24h == 0.10
        assert health.total_tokens_24h == 5000


# ── Alert persistence ────────────────────────────────────────────────────────


class TestAlertPersistence:
    """AlertEngine persists alert history to SQLite."""

    @pytest.fixture(autouse=True)
    def _reset_engine(self) -> None:
        """Reset the AlertEngine singleton between tests."""
        reset_alert_engine()
        yield  # type: ignore[misc]
        reset_alert_engine()

    def test_persist_and_reload_alerts(self, tmp_path: Path) -> None:
        """Alerts persisted to SQLite survive engine restart."""
        db_file = tmp_path / "alerts.db"

        # Create engine, configure persistence, fire an alert
        engine = AlertEngine()
        engine.configure_persistence(db_file)
        engine.register_threshold(
            AlertThreshold(
                name="test-high-latency",
                metric_key="adapters.average_latency_ms",
                condition=AlertCondition.GREATER_THAN,
                threshold_value=100.0,
                severity=AlertSeverity.HIGH,
                channels=["log"],
            )
        )

        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {
            "adapters": {"average_latency_ms": 200.0},
        }
        mock_api = MagicMock()
        mock_api.get_latest_metrics.return_value = mock_snapshot

        fired = engine.evaluate_all(api=mock_api)
        assert len(fired) == 1
        assert fired[0].threshold.name == "test-high-latency"

        original_history_len = len(engine.get_history())
        assert original_history_len == 1

        # Reset singleton and reload from SQLite
        reset_alert_engine()
        engine2 = AlertEngine()
        assert len(engine2.get_history()) == 0  # empty before configure
        engine2.configure_persistence(db_file)
        assert len(engine2.get_history()) == 1

        restored = engine2.get_history()[0]
        assert restored.threshold.name == "test-high-latency"
        assert restored.current_value == 200.0
        assert restored.threshold.severity == AlertSeverity.HIGH

    def test_alerts_without_persistence_still_work(self) -> None:
        """AlertEngine works normally when SQLite persistence is not configured."""
        engine = AlertEngine()
        engine.register_threshold(
            AlertThreshold(
                name="no-persist",
                metric_key="adapters.total_requests",
                condition=AlertCondition.GREATER_THAN,
                threshold_value=0.0,
                channels=["log"],
            )
        )

        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {"adapters": {"total_requests": 5}}
        mock_api = MagicMock()
        mock_api.get_latest_metrics.return_value = mock_snapshot

        fired = engine.evaluate_all(api=mock_api)
        assert len(fired) == 1
        assert len(engine.get_history()) == 1
