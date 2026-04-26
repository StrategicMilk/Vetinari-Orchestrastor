"""Tests for vetinari/dashboard/agent_dashboard.py — Defect 8 regression.

Defect 8: get_agent_metrics() previously filtered adapter metrics by substring
(``agent_type in key``), but adapter keys are named after provider/model pairs
(e.g. ``"llama_cpp:mistral"``), not agent types (``"FOREMAN"``). The filter
always returned an empty dict, so execution counts and latency were always zero.

Fix: iterate ALL adapters unconditionally.

These tests assert that adapter metrics ARE aggregated when the keys don't
contain the agent_type string, which is the normal production case.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.dashboard.agent_dashboard import AgentDashboard
from vetinari.orchestration.bottleneck import BottleneckAgentMetrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter_metrics(
    total_requests: int = 10,
    successful_requests: int = 8,
    failed_requests: int = 2,
    avg_latency_ms: float = 150.0,
    total_tokens_used: int = 5000,
    last_request_time: str = "2026-04-18T12:00:00Z",
) -> MagicMock:
    """Return a MagicMock that looks like an AdapterMetrics instance."""
    am = MagicMock()
    am.total_requests = total_requests
    am.successful_requests = successful_requests
    am.failed_requests = failed_requests
    am.avg_latency_ms = avg_latency_ms
    # Production code uses weighted-average latency: total_latency_sum / total_successful.
    # Derive total_latency_ms so the weighted average resolves back to avg_latency_ms.
    am.total_latency_ms = avg_latency_ms * successful_requests
    am.total_tokens_used = total_tokens_used
    am.last_request_time = last_request_time
    return am


# ---------------------------------------------------------------------------
# Defect 8 regression: adapter metrics aggregated regardless of key name
# ---------------------------------------------------------------------------


class TestGetAgentMetricsAdapterAggregation:
    """get_agent_metrics() must aggregate adapter metrics for ALL adapters,
    not filter by agent_type substring."""

    def test_metrics_aggregated_when_key_does_not_contain_agent_type(self):
        """successful/failed counts are non-zero even when adapter key has no agent_type substring."""
        adapter_data = {
            "llama_cpp:mistral-7b": _make_adapter_metrics(
                total_requests=10,
                successful_requests=8,
                failed_requests=2,
            )
        }
        mock_tel = MagicMock()
        mock_tel.get_adapter_metrics.return_value = adapter_data

        dashboard = AgentDashboard()
        with (
            patch("vetinari.telemetry.get_telemetry_collector", return_value=mock_tel),
            patch("vetinari.resilience.get_circuit_breaker_registry", side_effect=Exception("no cb")),
            patch("vetinari.analytics.cost.get_cost_tracker", side_effect=Exception("no cost")),
        ):
            result = dashboard.get_agent_metrics("FOREMAN")

        assert result.successful == 8, (
            f"Expected 8 successful from adapter, got {result.successful}"
        )
        assert result.failed == 2, (
            f"Expected 2 failed from adapter, got {result.failed}"
        )

    def test_total_executions_populated_from_adapter_with_non_matching_key(self):
        """total_executions reflects adapter data even when key is provider:model format."""
        adapter_data = {
            "openai:gpt-4o": _make_adapter_metrics(total_requests=25),
        }
        mock_tel = MagicMock()
        mock_tel.get_adapter_metrics.return_value = adapter_data

        dashboard = AgentDashboard()
        with (
            patch("vetinari.telemetry.get_telemetry_collector", return_value=mock_tel),
            patch("vetinari.resilience.get_circuit_breaker_registry", side_effect=Exception("no cb")),
            patch("vetinari.analytics.cost.get_cost_tracker", side_effect=Exception("no cost")),
        ):
            result = dashboard.get_agent_metrics("WORKER")

        assert result.total_executions >= 25, (
            f"Expected total_executions >= 25, got {result.total_executions}"
        )

    def test_avg_latency_ms_populated_from_adapter(self):
        """avg_latency_ms reflects adapter data when key has no agent_type substring."""
        adapter_data = {
            "llama_cpp_local:phi-3": _make_adapter_metrics(
                total_requests=5,
                avg_latency_ms=300.0,
            ),
        }
        mock_tel = MagicMock()
        mock_tel.get_adapter_metrics.return_value = adapter_data

        dashboard = AgentDashboard()
        with (
            patch("vetinari.telemetry.get_telemetry_collector", return_value=mock_tel),
            patch("vetinari.resilience.get_circuit_breaker_registry", side_effect=Exception("no cb")),
            patch("vetinari.analytics.cost.get_cost_tracker", side_effect=Exception("no cost")),
        ):
            result = dashboard.get_agent_metrics("INSPECTOR")

        assert result.avg_latency_ms == pytest.approx(300.0), (
            f"Expected avg_latency_ms=300.0, got {result.avg_latency_ms}"
        )

    def test_multiple_adapters_aggregated(self):
        """successful and failed counts are summed across all adapters."""
        adapter_data = {
            "llama_cpp:mistral-7b": _make_adapter_metrics(
                total_requests=10, successful_requests=8, failed_requests=2
            ),
            "openai:gpt-4o-mini": _make_adapter_metrics(
                total_requests=5, successful_requests=5, failed_requests=0
            ),
        }
        mock_tel = MagicMock()
        mock_tel.get_adapter_metrics.return_value = adapter_data

        dashboard = AgentDashboard()
        with (
            patch("vetinari.telemetry.get_telemetry_collector", return_value=mock_tel),
            patch("vetinari.resilience.get_circuit_breaker_registry", side_effect=Exception("no cb")),
            patch("vetinari.analytics.cost.get_cost_tracker", side_effect=Exception("no cost")),
        ):
            result = dashboard.get_agent_metrics("FOREMAN")

        # Both adapters contribute — totals must be 8+5=13 successful, 2+0=2 failed
        assert result.successful == 13
        assert result.failed == 2

    def test_empty_adapter_dict_returns_zero_metrics(self):
        """When get_adapter_metrics() returns an empty dict, counts stay at zero."""
        mock_tel = MagicMock()
        mock_tel.get_adapter_metrics.return_value = {}

        dashboard = AgentDashboard()
        with (
            patch("vetinari.telemetry.get_telemetry_collector", return_value=mock_tel),
            patch("vetinari.resilience.get_circuit_breaker_registry", side_effect=Exception("no cb")),
            patch("vetinari.analytics.cost.get_cost_tracker", side_effect=Exception("no cost")),
        ):
            result = dashboard.get_agent_metrics("FOREMAN")

        assert result.successful == 0
        assert result.failed == 0
        assert result.total_executions == 0
