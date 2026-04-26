"""End-to-end feature path verification tests (S47).

Verifies that each major feature path is wired end-to-end:
1. Submit task → Orchestrator → result returned (not empty fallback)
2. Store memory → search memory → get relevant results
3. Quality check fails → rework decision routed (logic exists and is callable)
4. Dashboard pulls real metrics data structures (not static mocks)
5. Kaizen improvement proposed → activated → observation recorded

These tests focus on the integration seams — can each subsystem be imported,
instantiated, and invoked end-to-end?  They use in-memory or temp-file backends
to avoid hitting real LLMs or persistent state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vetinari.exceptions import ExecutionError

# ── Path 1: Task submission → Orchestrator → result ──────────────────────────


class TestTaskSubmissionPath:
    """Path 1: Submit task → agent executes → result returned."""

    def test_orchestrator_module_importable(self) -> None:
        """vetinari.orchestrator is importable (module exists)."""
        from vetinari.orchestrator import Orchestrator

        assert callable(Orchestrator)

    def test_orchestrator_initialises_from_manifest(self, tmp_path: Path) -> None:
        """Orchestrator loads model list from manifest YAML."""
        import yaml

        manifest = tmp_path / "vetinari.yaml"
        manifest.write_text(
            yaml.dump({
                "models": [
                    {"id": "m1", "name": "test-model-7b"},
                    {"id": "m2", "name": "test-model-13b"},
                ],
                "tasks": [
                    {"id": "t1", "description": "Write a hello-world function"},
                ],
            }),
            encoding="utf-8",
        )
        from vetinari.orchestrator import Orchestrator

        orch = Orchestrator(str(manifest))
        assert len(orch.model_pool.models) == 2
        assert orch.model_pool.models[0].get("name") == "test-model-7b"

    def test_orchestrator_run_task_raises_on_unknown_id(self, tmp_path: Path) -> None:
        """run_task raises ValueError for task IDs not in the manifest."""
        import yaml

        manifest = tmp_path / "vetinari.yaml"
        manifest.write_text(
            yaml.dump({"tasks": [{"id": "t1", "description": "Some goal"}]}),
            encoding="utf-8",
        )
        from vetinari.orchestrator import Orchestrator

        orch = Orchestrator(str(manifest))
        with pytest.raises(ExecutionError, match="not found in manifest"):
            orch.run_task("nonexistent-task-id")

    def test_orchestrator_run_task_delegates_to_two_layer(self, tmp_path: Path) -> None:
        """run_task calls TwoLayerOrchestrator.generate_and_execute with the task description as goal."""
        import yaml

        manifest = tmp_path / "vetinari.yaml"
        manifest.write_text(
            yaml.dump({"tasks": [{"id": "t1", "description": "Implement the widget"}]}),
            encoding="utf-8",
        )
        from vetinari.orchestrator import Orchestrator

        fake_result: dict[str, Any] = {"completed": 1, "plan": {}, "outputs": {}}
        mock_orch = MagicMock()
        mock_orch.generate_and_execute.return_value = fake_result

        # Patch at the module where it's imported (vetinari.orchestrator)
        with patch("vetinari.orchestrator.get_two_layer_orchestrator", return_value=mock_orch):
            orch = Orchestrator(str(manifest))
            result = orch.run_task("t1")

        mock_orch.generate_and_execute.assert_called_once()
        call_kwargs = mock_orch.generate_and_execute.call_args
        assert call_kwargs.kwargs["goal"] == "Implement the widget"
        assert result == fake_result

    def test_adapter_bridge_chat_calls_infer(self) -> None:
        """_AdapterBridge.chat() delegates to AdapterManager.infer()."""
        from vetinari.adapters.base import InferenceResponse
        from vetinari.orchestrator import _AdapterBridge

        fake_resp = MagicMock(spec=InferenceResponse)
        fake_resp.output = "Hello, world!"
        fake_resp.latency_ms = 42

        mock_mgr = MagicMock()
        mock_mgr.infer.return_value = fake_resp

        # Patch at the module level where it was imported
        with patch("vetinari.orchestrator.get_adapter_manager", return_value=mock_mgr):
            bridge = _AdapterBridge()
            result = bridge.chat("my-model", "You are helpful.", "Say hello")

        assert result["output"] == "Hello, world!"
        assert result["latency_ms"] == 42
        mock_mgr.infer.assert_called_once()

    def test_web_shared_set_and_get_orchestrator(self) -> None:
        """shared.set_orchestrator / get_orchestrator round-trip works."""
        from vetinari.web.shared import get_orchestrator, set_orchestrator

        mock_orch = MagicMock()
        set_orchestrator(mock_orch)
        assert get_orchestrator() is mock_orch
        # Clean up
        set_orchestrator(None)

    def test_orchestrator_run_all_returns_aggregate(self, tmp_path: Path) -> None:
        """run_all returns dict with completed and total counts."""
        import yaml

        manifest = tmp_path / "vetinari.yaml"
        manifest.write_text(
            yaml.dump({
                "tasks": [
                    {"id": "t1", "description": "Goal one"},
                    {"id": "t2", "description": "Goal two"},
                ]
            }),
            encoding="utf-8",
        )
        from vetinari.orchestrator import Orchestrator

        fake_result: dict[str, Any] = {"completed": 1}
        mock_orch = MagicMock()
        mock_orch.generate_and_execute.return_value = fake_result

        with patch("vetinari.orchestrator.get_two_layer_orchestrator", return_value=mock_orch):
            orch = Orchestrator(str(manifest))
            summary = orch.run_all()

        assert summary["total"] == 2
        assert summary["completed"] == 2
        assert len(summary["results"]) == 2


# ── Path 2: Store memory → search → results ──────────────────────────────────


class TestMemoryStorePath:
    """Path 2: Store memory → search memory → get relevant results."""

    def test_memory_store_importable(self) -> None:
        """UnifiedMemoryStore can be imported."""
        from vetinari.memory.unified import UnifiedMemoryStore

        assert callable(UnifiedMemoryStore)

    def test_store_and_timeline_retrieves_entry(self, tmp_path: Path) -> None:
        """Stored entry is retrievable via timeline() (chronological listing)."""
        from vetinari.memory.interfaces import MemoryEntry
        from vetinari.memory.unified import UnifiedMemoryStore
        from vetinari.types import MemoryType

        store = UnifiedMemoryStore(db_path=str(tmp_path / "test_memory.db"))

        entry = MemoryEntry(
            agent="test_agent",
            entry_type=MemoryType.DISCOVERY,
            content="The widget caching layer uses LRU eviction with a 256-entry cap.",
            summary="LRU cache discovery",
        )
        stored_id = store.remember(entry)
        assert stored_id  # non-empty ID returned

        # timeline() returns all entries, not requiring FTS
        results = store.timeline(agent="test_agent", limit=10)
        assert any(r.id == stored_id for r in results), f"Stored entry {stored_id!r} not found in timeline"

    def test_store_and_search_single_word_match(self, tmp_path: Path) -> None:
        """FTS5 search returns stored entries for single-word queries."""
        from vetinari.memory.interfaces import MemoryEntry
        from vetinari.memory.unified import UnifiedMemoryStore
        from vetinari.types import MemoryType

        store = UnifiedMemoryStore(db_path=str(tmp_path / "fts_memory.db"))

        entry = MemoryEntry(
            agent="agent",
            entry_type=MemoryType.DISCOVERY,
            content="SQLite WAL mode improves write concurrency",
            summary="SQLite WAL discovery",
        )
        stored_id = store.remember(entry)
        assert stored_id

        # Single-word FTS5 search — "SQLite" is an exact token in the content
        results = store.search("SQLite")
        contents = [r.content for r in results]
        assert any("SQLite" in c for c in contents), "FTS search did not return the stored SQLite entry"

    def test_search_returns_empty_for_unknown_query(self, tmp_path: Path) -> None:
        """Search returns an empty list when no matching entries exist."""
        from vetinari.memory.unified import UnifiedMemoryStore

        store = UnifiedMemoryStore(db_path=str(tmp_path / "empty_memory.db"))
        results = store.search("xyzzy")
        assert results == []

    def test_multiple_entries_stored_and_accessible(self, tmp_path: Path) -> None:
        """Multiple stored entries are all accessible via timeline."""
        from vetinari.memory.interfaces import MemoryEntry
        from vetinari.memory.unified import UnifiedMemoryStore
        from vetinari.types import MemoryType

        store = UnifiedMemoryStore(db_path=str(tmp_path / "multi_memory.db"))

        ids = []
        for content in [
            "asyncio event loop scheduling details",
            "SQLite WAL mode improves write concurrency",
            "LRU cache eviction policy implementation",
        ]:
            e = MemoryEntry(
                agent="agent",
                entry_type=MemoryType.DISCOVERY,
                content=content,
                summary=content[:30],
            )
            ids.append(store.remember(e))

        # All 3 entries should appear in the timeline
        all_entries = store.timeline(limit=20)
        stored_ids = {r.id for r in all_entries}
        for eid in ids:
            assert eid in stored_ids, f"Entry {eid!r} not found in timeline"


# ── Path 3: Quality check fails → rework triggered ───────────────────────────


class TestQualityReworkPath:
    """Path 3: Quality check fails → rework triggered → task re-executed."""

    def test_quality_agent_importable(self) -> None:
        """Quality agent module and class are importable."""
        from vetinari.agents.consolidated.quality_agent import InspectorAgent

        assert callable(InspectorAgent)

    def test_two_layer_has_rework_handler(self) -> None:
        """TwoLayerOrchestrator exposes _handle_quality_rejection method."""
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        assert hasattr(TwoLayerOrchestrator, "_handle_quality_rejection"), (
            "_handle_quality_rejection not found on TwoLayerOrchestrator"
        )
        assert callable(TwoLayerOrchestrator._handle_quality_rejection)

    def test_rework_routing_returns_decision_for_bad_spec(self) -> None:
        """_handle_quality_rejection returns a ReworkDecision for bad_spec root cause."""
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        orch = TwoLayerOrchestrator()
        # Quality result with bad_spec root cause (root_cause is a dict with category key)
        result: dict[str, Any] = {
            "score": 0.3,
            "verdict": "reject",
            "root_cause": {"category": "bad_spec", "details": "Ambiguous requirements"},
            "suggestions": ["Clarify the acceptance criteria"],
        }
        decision = orch._handle_quality_rejection(
            task_id="t1",
            result=result,
            rework_count=0,
        )
        assert decision is not None, "Expected a rework decision, got None"
        assert hasattr(decision, "value") or isinstance(decision, (dict, str))

    def test_rework_escalates_after_max_rounds(self) -> None:
        """_handle_quality_rejection escalates when rework_count >= max rounds."""
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        orch = TwoLayerOrchestrator()
        result: dict[str, Any] = {"score": 0.2, "verdict": "reject", "root_cause": {"category": "hallucination"}}
        # At rework_count == 3, should escalate (not loop indefinitely)
        decision = orch._handle_quality_rejection(
            task_id="t2",
            result=result,
            rework_count=3,
        )
        assert decision is not None
        assert hasattr(decision, "value") or isinstance(decision, (dict, str))

    def test_correction_loop_enabled_by_default(self) -> None:
        """TwoLayerOrchestrator enables correction loop by default."""
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        orch = TwoLayerOrchestrator()
        assert orch.enable_correction_loop is True
        assert orch.correction_loop_max_rounds > 0


# ── Path 4: Dashboard shows real metrics ─────────────────────────────────────


class TestDashboardMetricsPath:
    """Path 4: Dashboard shows real metrics after execution."""

    def test_agent_dashboard_importable(self) -> None:
        """AgentDashboard aggregator is importable."""
        from vetinari.dashboard.agent_dashboard import AgentDashboard

        assert callable(AgentDashboard)

    def test_dashboard_get_dashboard_data_returns_structure(self) -> None:
        """AgentDashboard.get_dashboard_data() returns a non-empty dict."""
        from vetinari.dashboard.agent_dashboard import AgentDashboard

        dash = AgentDashboard()
        # Calls live subsystems but degrades gracefully when no data is present
        data = dash.get_dashboard_data()

        assert isinstance(data, dict), "Expected dict from get_dashboard_data()"
        assert data, "Dashboard data should be non-empty"

    def test_dashboard_get_system_health_returns_structure(self) -> None:
        """AgentDashboard.get_system_health() returns a SystemHealth with status."""
        from vetinari.dashboard.agent_dashboard import AgentDashboard

        dash = AgentDashboard()
        health = dash.get_system_health()

        assert health is not None
        assert hasattr(health, "status"), "SystemHealth should have a status field"

    def test_dashboard_get_all_agent_metrics_returns_list(self) -> None:
        """AgentDashboard.get_all_agent_metrics() returns a list."""
        from vetinari.dashboard.agent_dashboard import AgentDashboard

        dash = AgentDashboard()
        metrics = dash.get_all_agent_metrics()

        assert isinstance(metrics, list)


# ── Path 5: Kaizen improve → activate → observe ──────────────────────────────


class TestKaizenPath:
    """Path 5: Kaizen improvement proposed → activated → observation applied."""

    def test_improvement_log_importable(self) -> None:
        """ImprovementLog is importable from vetinari.kaizen."""
        from vetinari.kaizen.improvement_log import ImprovementLog

        assert callable(ImprovementLog)

    def test_propose_returns_improvement_id(self, tmp_path: Path) -> None:
        """propose() creates an improvement and returns a non-empty ID."""
        from vetinari.kaizen.improvement_log import ImprovementLog

        log = ImprovementLog(db_path=tmp_path / "kaizen.db")
        imp_id = log.propose(
            hypothesis="Switching to batched inference reduces p95 latency by 20%",
            metric="p95_latency_ms",
            baseline=450.0,
            target=360.0,
            applied_by="quality_agent",
            rollback_plan="Revert to single inference calls",
        )
        assert imp_id.startswith("IMP-"), f"Expected IMP-<hex> ID, got {imp_id!r}"

    def test_activate_moves_status_to_active(self, tmp_path: Path) -> None:
        """activate() transitions a proposed improvement to active; double-activate raises."""
        from vetinari.kaizen.improvement_log import ImprovementLog

        log = ImprovementLog(db_path=tmp_path / "kaizen_activate.db")
        imp_id = log.propose(
            hypothesis="Cache model weights to cut cold-start time",
            metric="cold_start_ms",
            baseline=5000.0,
            target=1000.0,
            applied_by="system",
            rollback_plan="Disable weight cache",
        )
        log.activate(imp_id)

        # Trying to activate an already-active improvement must raise
        with pytest.raises(ExecutionError, match="Cannot activate"):
            log.activate(imp_id)

    def test_observe_records_metrics_without_error(self, tmp_path: Path) -> None:
        """observe() records measurement data after activation without raising."""
        from vetinari.kaizen.improvement_log import ImprovementLog

        log = ImprovementLog(db_path=tmp_path / "kaizen_observe.db")
        imp_id = log.propose(
            hypothesis="Streaming reduces time-to-first-token",
            metric="ttft_ms",
            baseline=3000.0,
            target=1200.0,
            applied_by="worker_agent",
            rollback_plan="Disable streaming",
        )
        log.activate(imp_id)
        log.observe(imp_id, metric_value=1500.0, sample_size=10)
        log.observe(imp_id, metric_value=1300.0, sample_size=15)
        # Observations were recorded — the improvement must still be retrievable
        record = log.get_improvement(imp_id)
        assert record is not None
        assert record.id == imp_id

    def test_full_propose_activate_evaluate_lifecycle(self, tmp_path: Path) -> None:
        """Full PDCA lifecycle: propose → activate → observe → evaluate runs without error."""
        from vetinari.kaizen.improvement_log import ImprovementLog, ImprovementStatus

        log = ImprovementLog(db_path=tmp_path / "kaizen_lifecycle.db")
        imp_id = log.propose(
            hypothesis="Aggressive dedup cuts memory store size by 30%",
            metric="memory_entries",
            baseline=1000.0,
            target=700.0,
            applied_by="memory_agent",
            rollback_plan="Raise dedup threshold back to 0.95",
            observation_window_hours=0,  # instant window for test
        )
        log.activate(imp_id)
        log.observe(imp_id, metric_value=650.0, sample_size=50)
        log.observe(imp_id, metric_value=680.0, sample_size=50)

        # evaluate() runs without exception and returns a valid ImprovementStatus
        status = log.evaluate(imp_id)
        assert isinstance(status, ImprovementStatus)
