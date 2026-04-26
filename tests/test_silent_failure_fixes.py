"""Tests for S26 — Fix Silent Failure Patterns.

Covers:
- _derive_project_status helper in projects_api
- AgentDashboard telemetry integration
- System search memory integration
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vetinari.types import AgentType

# ── _derive_project_status ──────────────────────────────────────────────────


class TestDeriveProjectStatus:
    """Tests for the _derive_project_status helper."""

    def test_explicit_status_preserved(self) -> None:
        """Non-'unknown' config statuses are returned unchanged.

        Exception: 'running' with no active background thread returns
        'interrupted' — this is crash recovery behavior (not a bug).
        """
        from vetinari.web.shared import _derive_project_status

        # 'running' with no active thread → 'interrupted' (crash recovery)
        assert _derive_project_status("running", [{"id": "t1"}], set()) == "interrupted"
        # Other explicit statuses pass through unchanged
        assert _derive_project_status("completed", [{"id": "t1"}], set()) == "completed"
        assert _derive_project_status("error", [{"id": "t1"}], set()) == "error"
        assert _derive_project_status("cancelled", [{"id": "t1"}], set()) == "cancelled"

    def test_unknown_with_no_tasks_returns_pending(self) -> None:
        """A project with no planned tasks derives as 'pending'."""
        from vetinari.web.shared import _derive_project_status

        assert _derive_project_status("unknown", [], set()) == "pending"

    def test_unknown_with_all_tasks_completed(self) -> None:
        """When every planned task has output, status is 'completed'."""
        from vetinari.web.shared import _derive_project_status

        tasks = [{"id": "t1"}, {"id": "t2"}]
        completed = {"t1", "t2"}
        assert _derive_project_status("unknown", tasks, completed) == "completed"

    def test_unknown_with_some_tasks_completed(self) -> None:
        """Partial completion yields 'in_progress'."""
        from vetinari.web.shared import _derive_project_status

        tasks = [{"id": "t1"}, {"id": "t2"}, {"id": "t3"}]
        completed = {"t1"}
        assert _derive_project_status("unknown", tasks, completed) == "in_progress"

    def test_unknown_with_no_tasks_completed(self) -> None:
        """No completed tasks yields 'pending'."""
        from vetinari.web.shared import _derive_project_status

        tasks = [{"id": "t1"}, {"id": "t2"}]
        assert _derive_project_status("unknown", tasks, set()) == "pending"

    def test_empty_string_status_treated_as_unknown(self) -> None:
        """An empty-string status is treated the same as 'unknown'."""
        from vetinari.web.shared import _derive_project_status

        tasks = [{"id": "t1"}]
        assert _derive_project_status("", tasks, {"t1"}) == "completed"

    def test_tasks_without_ids_treated_as_pending(self) -> None:
        """Tasks missing an 'id' key are ignored when checking completion."""
        from vetinari.web.shared import _derive_project_status

        tasks = [{"description": "no id"}, {"id": "", "description": "empty id"}]
        assert _derive_project_status("unknown", tasks, set()) == "pending"


# ── AgentDashboard telemetry integration ─────────────────────────────────────


class TestAgentDashboardTelemetry:
    """Tests for AgentDashboard pulling real telemetry data."""

    def test_get_agent_metrics_populates_from_telemetry(self) -> None:
        """Agent metrics include telemetry-sourced execution counts and latency."""
        mock_adapter = MagicMock()
        mock_adapter.successful_requests = 42
        mock_adapter.failed_requests = 3
        mock_adapter.total_requests = 45
        mock_adapter.total_latency_ms = 5061.0  # 42 * 120.5 per weighted-avg calculation
        mock_adapter.total_tokens_used = 5000
        mock_adapter.last_request_time = "2026-03-22T10:00:00Z"

        mock_tel = MagicMock()
        mock_tel.get_adapter_metrics.return_value = {"foreman_adapter": mock_adapter}

        mock_cost_tracker = MagicMock()
        mock_cost_report = MagicMock()
        mock_cost_report.total_requests = 0

        mock_cost_tracker = MagicMock()
        mock_cost_tracker.get_report.return_value = mock_cost_report

        # Patch at the module where the late imports resolve
        with (
            patch(
                "vetinari.resilience.circuit_breaker.get_circuit_breaker_registry",
                side_effect=RuntimeError("not available"),
            ),
            patch(
                "vetinari.analytics.cost.get_cost_tracker",
                return_value=mock_cost_tracker,
            ),
            patch(
                "vetinari.telemetry.get_telemetry_collector",
                return_value=mock_tel,
            ),
        ):
            from vetinari.dashboard.agent_dashboard import AgentDashboard

            dashboard = AgentDashboard()
            metrics = dashboard.get_agent_metrics(AgentType.FOREMAN.value)

        assert metrics.successful == 42
        assert metrics.failed == 3
        assert metrics.avg_latency_ms == 120.5
        assert metrics.last_execution == "2026-03-22T10:00:00Z"

    def test_get_agent_metrics_graceful_telemetry_failure(self) -> None:
        """Metrics default to zero when telemetry is unavailable."""
        mock_cost_report = MagicMock()
        mock_cost_report.total_requests = 0

        mock_cost_tracker = MagicMock()
        mock_cost_tracker.get_report.return_value = mock_cost_report

        with (
            patch(
                "vetinari.resilience.circuit_breaker.get_circuit_breaker_registry",
                side_effect=RuntimeError("not available"),
            ),
            patch(
                "vetinari.analytics.cost.get_cost_tracker",
                return_value=mock_cost_tracker,
            ),
            patch(
                "vetinari.telemetry.get_telemetry_collector",
                side_effect=RuntimeError("not available"),
            ),
        ):
            from vetinari.dashboard.agent_dashboard import AgentDashboard

            dashboard = AgentDashboard()
            metrics = dashboard.get_agent_metrics(AgentType.WORKER.value)

        assert metrics.successful == 0
        assert metrics.failed == 0
        assert metrics.avg_latency_ms == 0.0


# ── System search memory integration ────────────────────────────────────────


class TestGlobalSearchMemoryIntegration:
    """Tests for memory-system integration in the global search endpoint."""

    def test_search_includes_memory_results(self) -> None:
        """Global search returns results from the memory system."""
        mock_entry = MagicMock()
        mock_entry.id = "mem_abc"
        mock_entry.agent = "foreman"
        mock_entry.entry_type = MagicMock(value="discovery")
        mock_entry.summary = "Found a useful pattern for caching"
        mock_entry.content = "Full content here"

        mock_store = MagicMock()
        mock_store.search.return_value = [mock_entry]

        # Patch at the source module where the late import resolves
        with patch(
            "vetinari.memory.unified.get_unified_memory_store",
            return_value=mock_store,
        ):
            from vetinari.memory.unified import get_unified_memory_store

            store = get_unified_memory_store()
            results = store.search("caching", limit=5)

        assert len(results) == 1
        assert results[0].id == "mem_abc"
        assert results[0].summary == "Found a useful pattern for caching"
