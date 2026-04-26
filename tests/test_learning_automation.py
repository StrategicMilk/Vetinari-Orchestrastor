"""Tests for Session 26 learning automation features.

Covers the TrainingScheduler singleton getter, manual training cycle
control, LearningOrchestrator dispatch wiring, and web-layer scheduler
integration verified by runtime state rather than source inspection.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared fixture — reset singletons before every test so no state leaks
# between tests in either direction.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_scheduler_singletons():
    """Isolate each test by resetting all scheduler singletons.

    Saves the original singleton values from both modules, sets them to None,
    yields, then restores the originals.  This prevents any test from
    polluting the singleton for subsequent tests regardless of test ordering.
    """
    import vetinari.training.idle_scheduler as _sched_mod
    import vetinari.web.litestar_training_api as _api_mod

    orig_sched = _sched_mod._scheduler_instance
    orig_api = _api_mod._scheduler_singleton

    _sched_mod._scheduler_instance = None
    _api_mod._scheduler_singleton = None

    yield

    _sched_mod._scheduler_instance = orig_sched
    _api_mod._scheduler_singleton = orig_api


# ---------------------------------------------------------------------------
# TestGetTrainingScheduler — 3 tests
# ---------------------------------------------------------------------------


class TestGetTrainingScheduler:
    """get_training_scheduler() returns a live TrainingScheduler singleton."""

    def test_returns_training_scheduler_instance(self) -> None:
        """get_training_scheduler() returns a TrainingScheduler object."""
        from vetinari.training.idle_scheduler import TrainingScheduler, get_training_scheduler

        scheduler = get_training_scheduler()
        assert isinstance(scheduler, TrainingScheduler), f"Expected TrainingScheduler, got {type(scheduler).__name__}"

    def test_returns_same_instance_on_repeated_calls(self) -> None:
        """Repeated calls to get_training_scheduler() return the identical object."""
        from vetinari.training.idle_scheduler import get_training_scheduler

        first = get_training_scheduler()
        second = get_training_scheduler()
        assert first is second, "get_training_scheduler() must return the same singleton on every call"

    def test_scheduler_carries_idle_detector(self) -> None:
        """The returned scheduler has an IdleDetector attached."""
        from vetinari.training.idle_scheduler import IdleDetector, get_training_scheduler

        scheduler = get_training_scheduler()
        assert hasattr(scheduler, "_idle_detector"), "TrainingScheduler must expose _idle_detector"
        assert isinstance(scheduler._idle_detector, IdleDetector), (
            f"Expected IdleDetector, got {type(scheduler._idle_detector).__name__}"
        )


# ---------------------------------------------------------------------------
# TestStartManualCycle — 6 tests
# ---------------------------------------------------------------------------


class TestStartManualCycle:
    """TrainingScheduler.start_manual_cycle() controls and records jobs correctly."""

    @pytest.fixture
    def scheduler(self):
        """Return a fresh TrainingScheduler with _execute_training_cycle patched out."""
        from vetinari.training.idle_scheduler import IdleDetector, TrainingScheduler

        detector = IdleDetector()
        sched = TrainingScheduler(idle_detector=detector)
        return sched

    def test_returns_manual_prefixed_job_id(self, scheduler) -> None:
        """start_manual_cycle() returns a job_id that starts with 'manual-'."""
        with patch.object(scheduler, "_execute_training_cycle"):
            job_id = scheduler.start_manual_cycle("Test activity")

        assert isinstance(job_id, str), "job_id must be a string"
        assert job_id.startswith("manual-"), f"job_id must start with 'manual-', got {job_id!r}"

    def test_records_job_in_history(self, scheduler) -> None:
        """The triggered job appears in scheduler._history after the call."""
        with patch.object(scheduler, "_execute_training_cycle"):
            job_id = scheduler.start_manual_cycle("History test")

        assert len(scheduler._history) == 1, f"Expected 1 history entry, got {len(scheduler._history)}"
        assert scheduler._history[0]["job_id"] == job_id, "History entry job_id must match the returned job_id"

    def test_records_custom_activity_description(self, scheduler) -> None:
        """The history entry preserves the caller-supplied activity description."""
        custom_desc = "Fine-tune reasoning paths"
        with patch.object(scheduler, "_execute_training_cycle"):
            scheduler.start_manual_cycle(custom_desc)

        assert scheduler._history[0]["activity_description"] == custom_desc, (
            f"History must record activity_description={custom_desc!r}, "
            f"got {scheduler._history[0]['activity_description']!r}"
        )

    def test_uses_non_empty_default_description(self, scheduler) -> None:
        """Omitting activity_description produces a non-empty default."""
        with patch.object(scheduler, "_execute_training_cycle"):
            scheduler.start_manual_cycle()

        assert len(scheduler._history) == 1, "Expected 1 history entry"
        desc = scheduler._history[0]["activity_description"]
        assert isinstance(desc, str) and len(desc) > 0, (
            f"Default activity_description must be a non-empty string, got {desc!r}"
        )

    def test_returns_already_running_when_job_active(self, scheduler) -> None:
        """Returns 'already_running' when _current_job is already set."""
        from datetime import datetime, timezone

        from vetinari.training.idle_scheduler import TrainingJob

        # Manually plant an active job
        active_job = TrainingJob(
            job_id="existing-job-001",
            status="running",
            activity_description="Active task",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        scheduler._current_job = active_job

        with patch.object(scheduler, "_execute_training_cycle"):
            result = scheduler.start_manual_cycle("Should be blocked")

        assert result == "already_running", f"Expected 'already_running' when a job is active, got {result!r}"

    def test_does_not_append_to_history_when_already_running(self, scheduler) -> None:
        """History is not updated when the call is rejected due to an active job."""
        from datetime import datetime, timezone

        from vetinari.training.idle_scheduler import TrainingJob

        active_job = TrainingJob(
            job_id="blocking-job-002",
            status="running",
            activity_description="Blocking task",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        scheduler._current_job = active_job

        with patch.object(scheduler, "_execute_training_cycle"):
            scheduler.start_manual_cycle("Would-be second job")

        assert len(scheduler._history) == 0, (
            "History must not be updated when the call is rejected because a job is already running"
        )


# ---------------------------------------------------------------------------
# TestOrchestratorDispatch — 4 tests
# ---------------------------------------------------------------------------


class TestOrchestratorDispatch:
    """LearningOrchestrator._run_training() dispatches correctly based on recommendations."""

    def test_dispatches_to_scheduler_and_returns_gain_when_recommended(self) -> None:
        """When should_retrain recommends, _run_training dispatches a cycle and returns 0.01."""
        from vetinari.learning.orchestrator import LearningOrchestrator
        from vetinari.learning.training_manager import RetrainingRecommendation

        orch = LearningOrchestrator()

        recommendation = RetrainingRecommendation(
            model_id="*",
            task_type="*",
            current_avg_quality=0.60,
            baseline_quality=0.80,
            degradation=0.25,
            recommended=True,
            reason="Quality degraded 25% below baseline.",
        )

        mock_manager = MagicMock()
        mock_manager.should_retrain.return_value = recommendation

        mock_scheduler = MagicMock()
        mock_scheduler.start_manual_cycle.return_value = "manual-abc123"

        with (
            patch("vetinari.learning.training_manager.get_training_manager", return_value=mock_manager),
            patch(
                "vetinari.training.idle_scheduler.get_training_scheduler",
                return_value=mock_scheduler,
            ),
        ):
            gain = orch._run_training()

        assert gain == 0.01, f"Expected gain=0.01 when retraining is dispatched, got {gain}"
        mock_scheduler.start_manual_cycle.assert_called_once()

    def test_skips_and_returns_zero_when_not_recommended(self) -> None:
        """When should_retrain does not recommend, _run_training returns 0.0."""
        from vetinari.learning.orchestrator import LearningOrchestrator
        from vetinari.learning.training_manager import RetrainingRecommendation

        orch = LearningOrchestrator()

        recommendation = RetrainingRecommendation(
            model_id="*",
            task_type="*",
            current_avg_quality=0.85,
            baseline_quality=0.80,
            degradation=0.0,
            recommended=False,
            reason="Quality acceptable.",
        )

        mock_manager = MagicMock()
        mock_manager.should_retrain.return_value = recommendation

        with patch("vetinari.learning.training_manager.get_training_manager", return_value=mock_manager):
            gain = orch._run_training()

        assert gain == 0.0, f"Expected gain=0.0 when no retraining needed, got {gain}"

    def test_returns_zero_when_scheduler_says_already_running(self) -> None:
        """When the scheduler reports 'already_running', _run_training returns 0.0."""
        from vetinari.learning.orchestrator import LearningOrchestrator
        from vetinari.learning.training_manager import RetrainingRecommendation

        orch = LearningOrchestrator()

        recommendation = RetrainingRecommendation(
            model_id="*",
            task_type="*",
            current_avg_quality=0.60,
            baseline_quality=0.80,
            degradation=0.25,
            recommended=True,
            reason="Quality degraded.",
        )

        mock_manager = MagicMock()
        mock_manager.should_retrain.return_value = recommendation

        mock_scheduler = MagicMock()
        mock_scheduler.start_manual_cycle.return_value = "already_running"

        with (
            patch("vetinari.learning.training_manager.get_training_manager", return_value=mock_manager),
            patch(
                "vetinari.training.idle_scheduler.get_training_scheduler",
                return_value=mock_scheduler,
            ),
        ):
            gain = orch._run_training()

        assert gain == 0.0, f"Expected gain=0.0 when scheduler reports 'already_running', got {gain}"

    def test_swallows_exceptions_and_returns_zero(self) -> None:
        """Exceptions inside _run_training are caught and 0.0 is returned."""
        from vetinari.learning.orchestrator import LearningOrchestrator

        orch = LearningOrchestrator()

        with patch(
            "vetinari.learning.training_manager.get_training_manager",
            side_effect=RuntimeError("Simulated training-manager failure"),
        ):
            gain = orch._run_training()

        assert gain == 0.0, f"Expected gain=0.0 when an exception occurs, got {gain}"


# ---------------------------------------------------------------------------
# TestWebLayerWiring — 5 tests
# ---------------------------------------------------------------------------


class TestWebLayerWiring:
    """Web-layer scheduler helpers share the canonical singleton and reflect live state."""

    def test_get_scheduler_returns_identical_object_as_get_training_scheduler(self) -> None:
        """_get_scheduler() and get_training_scheduler() resolve to the same object."""
        from vetinari.training.idle_scheduler import get_training_scheduler
        from vetinari.web.litestar_training_api import _get_scheduler

        canonical = get_training_scheduler()
        via_api = _get_scheduler()

        assert via_api is canonical, (
            "_get_scheduler() must return the identical singleton as get_training_scheduler(); "
            "got two different objects — the web layer is holding a private copy"
        )

    def test_get_scheduler_returns_training_scheduler_type(self) -> None:
        """_get_scheduler() returns a TrainingScheduler instance (not None, not a stub)."""
        from vetinari.training.idle_scheduler import TrainingScheduler
        from vetinari.web.litestar_training_api import _get_scheduler

        scheduler = _get_scheduler()
        assert isinstance(scheduler, TrainingScheduler), f"Expected TrainingScheduler, got {type(scheduler).__name__}"

    def test_lifespan_import_path_resolves_to_canonical_singleton(self) -> None:
        """The import path lifespan.py uses points to the same singleton as get_training_scheduler().

        lifespan.py does: from vetinari.web.litestar_training_api import _get_scheduler
        and then calls _get_scheduler().  This test imports via the same symbol and
        confirms it returns the canonical instance — no source inspection.
        """
        from vetinari.training.idle_scheduler import get_training_scheduler

        # Import the same symbol that lifespan.py imports at runtime
        from vetinari.web.litestar_training_api import _get_scheduler as lifespan_get_scheduler

        canonical = get_training_scheduler()
        via_lifespan_path = lifespan_get_scheduler()

        assert via_lifespan_path is canonical, (
            "The scheduler obtained via the lifespan import path must be the same object "
            "as get_training_scheduler() — they are currently two different instances"
        )

    def test_is_scheduler_training_returns_false_when_no_active_job(self) -> None:
        """_is_scheduler_training() returns False when _current_job is None."""
        from vetinari.training.idle_scheduler import get_training_scheduler
        from vetinari.web.litestar_training_api import _is_scheduler_training

        # Ensure scheduler exists and has no active job
        scheduler = get_training_scheduler()
        scheduler._current_job = None

        result = _is_scheduler_training()
        assert result is False, f"Expected _is_scheduler_training()=False when no job is active, got {result!r}"

    def test_is_scheduler_training_returns_true_when_job_active(self) -> None:
        """_is_scheduler_training() returns True when _current_job is set."""
        from datetime import datetime, timezone

        from vetinari.training.idle_scheduler import TrainingJob, get_training_scheduler
        from vetinari.web.litestar_training_api import _is_scheduler_training

        scheduler = get_training_scheduler()
        scheduler._current_job = TrainingJob(
            job_id="test-active-job-999",
            status="running",
            activity_description="Active wiring test job",
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        try:
            result = _is_scheduler_training()
            assert result is True, f"Expected _is_scheduler_training()=True when a job is active, got {result!r}"
        finally:
            # Clean up so the fixture restore doesn't leave a job in the scheduler
            scheduler._current_job = None
