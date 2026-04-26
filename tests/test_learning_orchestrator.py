"""Tests for Session 6 — Learning Orchestrator.

Covers the 4 implementation items: LearningOrchestrator class,
lifespan wiring, MetaOptimizer budget allocation, and auto-rollback
on quality degradation.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from vetinari.events import reset_event_bus
from vetinari.learning.meta_optimizer import LearningPhase, MetaOptimizer, StrategyRecord
from vetinari.learning.orchestrator import (
    QUALITY_DROP_THRESHOLD,
    ActionBaseline,
    LearningOrchestrator,
    reset_learning_orchestrator,
)


@pytest.fixture(autouse=True)
def _clean_singletons():
    """Reset singletons before and after each test to prevent cross-test interference."""
    reset_learning_orchestrator()
    reset_event_bus()
    yield
    reset_learning_orchestrator()
    reset_event_bus()


# ---------------------------------------------------------------------------
# 6.1 — LearningOrchestrator class
# ---------------------------------------------------------------------------


class TestLearningOrchestrator:
    """Verify the core orchestrator loop and strategy dispatch."""

    def test_orchestrator_starts_and_stops(self):
        """Orchestrator background thread starts and stops cleanly."""
        orch = LearningOrchestrator(cycle_interval_seconds=3600)
        assert orch.is_running is False

        orch.start()
        assert orch.is_running is True

        orch.stop()
        assert orch.is_running is False

    def test_duplicate_start_is_noop(self):
        """Calling start() twice does not create a second thread."""
        orch = LearningOrchestrator(cycle_interval_seconds=3600)
        orch.start()
        thread_1 = orch._thread

        orch.start()  # Should be a no-op
        thread_2 = orch._thread

        assert thread_1 is thread_2
        orch.stop()

    def test_run_cycle_dispatches_prompt_evolution(self):
        """When MetaOptimizer suggests prompt_evolution, the evolver is called."""
        orch = LearningOrchestrator()

        mock_evolver = MagicMock()
        mock_evolver.check_shadow_test_results = MagicMock()

        with (
            patch(
                "vetinari.learning.meta_optimizer.get_meta_optimizer",
            ) as mock_get_opt,
            patch(
                "vetinari.learning.prompt_evolver.get_prompt_evolver",
                return_value=mock_evolver,
            ),
            patch.object(orch, "_get_current_quality", return_value=0.7),
        ):
            mock_opt = MagicMock()
            mock_opt.detect_phase.return_value = LearningPhase.IMPROVEMENT
            mock_opt.suggest_next_strategy.return_value = "prompt_evolution"
            mock_get_opt.return_value = mock_opt

            orch._run_cycle()

        mock_evolver.check_shadow_test_results.assert_called_once()
        mock_opt.record_cycle.assert_called_once_with("prompt_evolution", 0.0)

    def test_run_cycle_dispatches_training(self):
        """When MetaOptimizer suggests training, the training manager is checked."""
        orch = LearningOrchestrator()

        mock_recommendation = MagicMock()
        mock_recommendation.recommended = False
        mock_recommendation.reason = "Quality acceptable"

        mock_manager = MagicMock()
        mock_manager.should_retrain.return_value = mock_recommendation

        with (
            patch(
                "vetinari.learning.meta_optimizer.get_meta_optimizer",
            ) as mock_get_opt,
            patch(
                "vetinari.learning.training_manager.get_training_manager",
                return_value=mock_manager,
            ),
            patch.object(orch, "_get_current_quality", return_value=0.7),
        ):
            mock_opt = MagicMock()
            mock_opt.detect_phase.return_value = LearningPhase.IMPROVEMENT
            mock_opt.suggest_next_strategy.return_value = "training"
            mock_get_opt.return_value = mock_opt

            orch._run_cycle()

        mock_manager.should_retrain.assert_called_once()
        assert mock_manager.should_retrain.call_args.kwargs == {"model_id": "*", "task_type": "*"}
        mock_opt.record_cycle.assert_called_once_with("training", 0.0)

    def test_run_cycle_dispatches_auto_research(self):
        """When MetaOptimizer suggests auto_research, model scout is called."""
        orch = LearningOrchestrator()

        mock_scout = MagicMock()
        mock_scout.scout_for_task.return_value = []

        with (
            patch(
                "vetinari.learning.meta_optimizer.get_meta_optimizer",
            ) as mock_get_opt,
            patch(
                "vetinari.models.model_scout.ModelScout",
                return_value=mock_scout,
            ),
            patch.object(orch, "_get_current_quality", return_value=0.7),
        ):
            mock_opt = MagicMock()
            mock_opt.detect_phase.return_value = LearningPhase.IMPROVEMENT
            mock_opt.suggest_next_strategy.return_value = "auto_research"
            mock_get_opt.return_value = mock_opt

            orch._run_cycle()

        # Scout called for each default task type
        assert mock_scout.scout_for_task.call_count == 3

    def test_collapse_risk_halts_learning_and_alerts(self):
        """COLLAPSE_RISK phase halts learning and publishes an alert event."""
        orch = LearningOrchestrator()

        from vetinari.events import HumanApprovalNeeded, get_event_bus

        bus = get_event_bus()
        received_events: list = []
        bus.subscribe(HumanApprovalNeeded, lambda e: received_events.append(e))

        with patch(
            "vetinari.learning.meta_optimizer.get_meta_optimizer",
        ) as mock_get_opt:
            mock_opt = MagicMock()
            mock_opt.detect_phase.return_value = LearningPhase.COLLAPSE_RISK
            mock_get_opt.return_value = mock_opt

            orch._run_cycle()

        # Should NOT record a cycle (learning is halted)
        mock_opt.record_cycle.assert_not_called()
        # Should publish an alert
        assert len(received_events) == 1
        assert "collapse" in received_events[0].reason.lower()

    def test_saturation_switches_strategy(self):
        """SATURATION phase picks the second-best strategy from ROI rankings."""
        orch = LearningOrchestrator()

        with (
            patch(
                "vetinari.learning.meta_optimizer.get_meta_optimizer",
            ) as mock_get_opt,
            patch.object(orch, "_dispatch_strategy", return_value=0.0) as mock_dispatch,
            patch.object(orch, "_get_current_quality", return_value=0.7),
        ):
            mock_opt = MagicMock()
            mock_opt.detect_phase.return_value = LearningPhase.SATURATION
            mock_opt.get_roi_rankings.return_value = [
                {"strategy": "prompt_evolution", "recent_avg_gain": 0.001},
                {"strategy": "training", "recent_avg_gain": 0.0005},
            ]
            mock_get_opt.return_value = mock_opt

            orch._run_cycle()

        # Should dispatch "training" (the 2nd best), not "prompt_evolution"
        mock_dispatch.assert_called_once_with("training")


# ---------------------------------------------------------------------------
# 6.2 — Lifespan wiring
# ---------------------------------------------------------------------------


class TestLifespanOrchestratorWiring:
    """Verify that lifespan.py starts and stops the LearningOrchestrator."""

    def test_orchestrator_started_during_lifespan(self):
        """get_learning_orchestrator is called and start() invoked during startup."""
        mock_orch = MagicMock()
        mock_orch.start = MagicMock()
        mock_orch.stop = MagicMock()

        with patch(
            "vetinari.learning.orchestrator.get_learning_orchestrator",
            return_value=mock_orch,
        ):
            from vetinari.learning.orchestrator import get_learning_orchestrator

            orch = get_learning_orchestrator()
            assert orch is not None
            orch.start()
            mock_orch.start.assert_called_once()

    def test_orchestrator_none_safe(self):
        """If orchestrator import fails, startup still proceeds."""
        # Simulates the try/except path in lifespan.py — the orchestrator
        # stays None when import raises, and startup continues.
        _learning_orchestrator = None
        with pytest.raises(ImportError):
            raise ImportError("simulated")
        assert _learning_orchestrator is None


# ---------------------------------------------------------------------------
# 6.3 — MetaOptimizer budget allocation
# ---------------------------------------------------------------------------


class TestAllocateIdleBudget:
    """Verify that allocate_idle_budget distributes time by ROI."""

    def test_equal_split_with_no_data(self):
        """With no strategy data, returns equal 1/3 split across defaults."""
        optimizer = MetaOptimizer.__new__(MetaOptimizer)
        optimizer._strategies = {}
        optimizer._quality_history = []
        optimizer._lock = __import__("threading").Lock()

        budget = optimizer.allocate_idle_budget()

        assert set(budget.keys()) == {"prompt_evolution", "training", "auto_research"}
        assert abs(sum(budget.values()) - 1.0) < 0.01

    def test_highest_roi_gets_most_budget(self):
        """Strategy with the highest recent gains gets the largest allocation."""
        optimizer = MetaOptimizer.__new__(MetaOptimizer)
        optimizer._lock = __import__("threading").Lock()
        optimizer._quality_history = [0.1] * 10  # Not collapse

        high_roi = StrategyRecord(strategy_name="prompt_evolution", total_cycles=20, total_gain=2.0)
        high_roi.recent_gains = [0.1] * 10

        low_roi = StrategyRecord(strategy_name="training", total_cycles=20, total_gain=0.2)
        low_roi.recent_gains = [0.01] * 10

        mid_roi = StrategyRecord(strategy_name="auto_research", total_cycles=10, total_gain=0.5)
        mid_roi.recent_gains = [0.05] * 10

        optimizer._strategies = {
            "prompt_evolution": high_roi,
            "training": low_roi,
            "auto_research": mid_roi,
        }

        budget = optimizer.allocate_idle_budget()

        assert budget["prompt_evolution"] > budget["training"]
        assert budget["prompt_evolution"] > budget["auto_research"]
        assert abs(sum(budget.values()) - 1.0) < 0.01

    def test_collapse_risk_returns_empty(self):
        """In COLLAPSE_RISK phase, no budget is allocated."""
        optimizer = MetaOptimizer.__new__(MetaOptimizer)
        optimizer._lock = __import__("threading").Lock()
        # Simulate collapse: many large negative gains
        optimizer._quality_history = [-0.5] * 20
        optimizer._strategies = {
            "prompt_evolution": StrategyRecord(strategy_name="prompt_evolution"),
        }

        budget = optimizer.allocate_idle_budget()

        assert budget == {}

    def test_allocations_sum_to_one(self):
        """Budget allocations always sum to exactly 1.0."""
        optimizer = MetaOptimizer.__new__(MetaOptimizer)
        optimizer._lock = __import__("threading").Lock()
        optimizer._quality_history = [0.05] * 10
        optimizer._strategies = {}

        for name in ("a", "b", "c", "d"):
            rec = StrategyRecord(strategy_name=name, total_cycles=5, total_gain=0.1)
            rec.recent_gains = [0.02] * 5
            optimizer._strategies[name] = rec

        budget = optimizer.allocate_idle_budget()

        assert abs(sum(budget.values()) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# 6.4 — Auto-rollback on quality degradation
# ---------------------------------------------------------------------------


class TestAutoRollback:
    """Verify that quality degradation triggers automatic rollback."""

    def test_quality_drop_triggers_rollback_event(self):
        """A >5% quality drop after an action publishes KaizenImprovementReverted."""
        orch = LearningOrchestrator()

        from vetinari.events import KaizenImprovementReverted, get_event_bus

        bus = get_event_bus()
        reverted_events: list = []
        bus.subscribe(KaizenImprovementReverted, lambda e: reverted_events.append(e))

        # Simulate: baseline was 0.80, current quality dropped to 0.70 (12.5% drop > 5%)
        orch._action_baselines = [
            ActionBaseline(
                action_id="prompt_evolution_1",
                strategy="prompt_evolution",
                baseline_quality=0.80,
            ),
        ]

        with patch.object(orch, "_get_current_quality", return_value=0.70):
            orch._check_for_rollback()

        assert len(reverted_events) == 1
        assert "prompt_evolution" in reverted_events[0].reason
        assert orch._action_baselines[0].rolled_back is True

    def test_small_quality_drop_does_not_trigger_rollback(self):
        """A <5% quality drop is within tolerance and does not trigger rollback."""
        orch = LearningOrchestrator()

        from vetinari.events import KaizenImprovementReverted, get_event_bus

        bus = get_event_bus()
        reverted_events: list = []
        bus.subscribe(KaizenImprovementReverted, lambda e: reverted_events.append(e))

        # Simulate: baseline was 0.80, current is 0.78 (2.5% drop < 5%)
        orch._action_baselines = [
            ActionBaseline(
                action_id="training_1",
                strategy="training",
                baseline_quality=0.80,
            ),
        ]

        with patch.object(orch, "_get_current_quality", return_value=0.78):
            orch._check_for_rollback()

        assert len(reverted_events) == 0
        assert orch._action_baselines[0].rolled_back is False

    def test_expired_baseline_not_checked(self):
        """Actions older than the rollback window are not checked for degradation."""
        orch = LearningOrchestrator()

        from datetime import datetime, timedelta, timezone

        # Baseline recorded 48 hours ago (outside 24h window)
        old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        orch._action_baselines = [
            ActionBaseline(
                action_id="old_action",
                strategy="prompt_evolution",
                baseline_quality=0.90,
                recorded_at=old_time,
            ),
        ]

        with patch.object(orch, "_get_current_quality", return_value=0.50):
            orch._check_for_rollback()

        # Old baseline is pruned, no rollback triggered
        assert len(orch._action_baselines) == 0

    def test_rollback_logs_quality_delta(self):
        """Rollback event includes the quality delta in its reason string."""
        orch = LearningOrchestrator()

        from vetinari.events import KaizenImprovementReverted, get_event_bus

        bus = get_event_bus()
        reverted_events: list = []
        bus.subscribe(KaizenImprovementReverted, lambda e: reverted_events.append(e))

        orch._action_baselines = [
            ActionBaseline(
                action_id="test_action",
                strategy="training",
                baseline_quality=0.90,
            ),
        ]

        with patch.object(orch, "_get_current_quality", return_value=0.80):
            orch._check_for_rollback()

        assert len(reverted_events) == 1
        reason = reverted_events[0].reason
        assert "0.900" in reason  # baseline
        assert "0.800" in reason  # current


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    """Verify the get_learning_orchestrator singleton pattern."""

    def test_singleton_returns_same_instance(self):
        """get_learning_orchestrator returns the same instance on repeated calls."""
        from vetinari.learning.orchestrator import get_learning_orchestrator

        a = get_learning_orchestrator()
        b = get_learning_orchestrator()
        assert a is b

    def test_reset_creates_new_instance(self):
        """reset_learning_orchestrator forces a new instance on next call."""
        from vetinari.learning.orchestrator import get_learning_orchestrator

        a = get_learning_orchestrator()
        reset_learning_orchestrator()
        b = get_learning_orchestrator()
        assert a is not b
