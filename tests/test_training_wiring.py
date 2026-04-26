"""Tests for training pipeline end-to-end wiring.

Verifies that the training system components are properly connected:
TrainingManager -> TrainingPipeline, Web API -> TrainingScheduler,
Curriculum -> Pipeline, and the DAPO stage pipeline.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class TestTrainLocalWiring:
    """Verify TrainingManager.train_local() delegates to TrainingPipeline."""

    def test_train_local_calls_pipeline(self):
        """train_local() should delegate to TrainingPipeline.run() when libs are available."""
        mock_run = MagicMock()
        mock_run.return_value = MagicMock(
            success=True,
            output_model_path="/trained/model",
            training_examples=100,
            eval_score=0.85,
        )
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.check_requirements.return_value = {
            "ready_for_training": True,
            "libraries": {"trl": True, "unsloth": True},
        }
        mock_pipeline.return_value.run = mock_run

        with patch.dict(
            "sys.modules",
            {"vetinari.training.pipeline": MagicMock(TrainingPipeline=mock_pipeline)},
        ):
            from vetinari.learning.training_manager import (
                TrainingDataset,
                TrainingManager,
            )

            mgr = TrainingManager()
            # Create a minimal dataset
            dataset = TrainingDataset(
                records=[{"task": "test", "prompt": "p", "response": "r"}] * 200,
                format="sft",
                stats={"total_records": 200},
            )
            result = mgr.train_local("test-model", dataset)

        assert hasattr(result, "success"), "result must be a TrainingResult"
        assert hasattr(result, "model_path"), "result must carry model_path"
        assert hasattr(result, "duration_seconds"), "result must carry timing"
        assert result.success is True
        assert result.model_path == "/trained/model"
        assert result.metrics.get("training_examples") == 100
        assert result.metrics.get("eval_score") == pytest.approx(0.85)
        assert result.error is None
        mock_run.assert_called_once()

    def test_train_local_no_training_libs(self):
        """train_local() should return failure when training libraries are missing."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.check_requirements.return_value = {
            "ready_for_training": False,
            "libraries": {"trl": False, "unsloth": False, "peft": False},
        }

        with patch.dict(
            "sys.modules",
            {"vetinari.training.pipeline": MagicMock(TrainingPipeline=mock_pipeline)},
        ):
            from vetinari.learning.training_manager import (
                TrainingDataset,
                TrainingManager,
            )

            mgr = TrainingManager()
            dataset = TrainingDataset(
                records=[{"task": "test"}] * 200,
                format="sft",
                stats={"total_records": 200},
            )
            result = mgr.train_local("test-model", dataset)
            assert result.success is False
            assert result.model_path is None
            assert isinstance(result.error, str)
            assert "training libraries not installed" in result.error.lower()
            assert "trl, unsloth, peft" in result.error.lower()

    def test_train_local_tracks_failed_job_for_too_small_dataset(self):
        from vetinari.learning.training_manager import TrainingDataset, TrainingManager

        mgr = TrainingManager()
        dataset = TrainingDataset(
            records=[{"task": "test", "prompt": "p", "response": "r"}],
            format="sft",
            stats={"total_records": 1},
        )

        result = mgr.train_local("test-model", dataset)
        jobs = mgr.list_jobs()

        assert result.success is False
        assert len(jobs) == 1
        assert jobs[0].status == "failed"
        assert jobs[0].result is result

    def test_should_retrain_supports_model_and_task_wildcards(self):
        from vetinari.learning.training_manager import TrainingManager

        collector = MagicMock()
        collector._load_all.return_value = [
            SimpleNamespace(model_id="qwen-7b", task_type="coding", score=0.50),
            SimpleNamespace(model_id="llama-8b", task_type="research", score=0.70),
        ]
        mgr = TrainingManager()

        with patch.object(mgr, "_get_collector", return_value=collector):
            recommendation = mgr.should_retrain(model_id="*", task_type="*")

        assert recommendation.current_avg_quality == pytest.approx(0.60)
        assert recommendation.recommended is True
        assert "No records found" not in recommendation.reason


class TestCurriculumWiring:
    """Verify curriculum dispatches to training pipeline."""

    def test_run_activity_fine_tune(self):
        """run_activity() should call _run_fine_tune for FINE_TUNE_WEAK_SKILL."""
        from vetinari.training.curriculum import (
            TrainingActivityType,
            TrainingCurriculum,
        )

        curriculum = TrainingCurriculum()
        # Mock next_activity to return a fine-tune activity
        mock_activity = MagicMock()
        mock_activity.type = TrainingActivityType.FINE_TUNE_WEAK_SKILL
        mock_activity.description = "Test fine-tune"
        mock_activity.metadata = {"task_type": "coding"}

        with patch.object(curriculum, "next_activity", return_value=mock_activity):
            with patch.object(curriculum, "_run_fine_tune") as mock_ft:
                curriculum.run_activity("test", job_id="job-1")
                mock_ft.assert_called_once_with(mock_activity)

    def test_curriculum_candidates_graceful_fallback(self):
        """All curriculum candidates should return None gracefully when modules unavailable."""
        from vetinari.training.curriculum import TrainingCurriculum

        curriculum = TrainingCurriculum()
        # next_activity should work even when all upstream modules are missing
        activity = curriculum.next_activity()
        assert activity is not None
        assert activity.description  # Should return default calibration activity


class TestDAPOStageWiring:
    """Verify SimPO and DAPO stages handle missing data gracefully."""

    def test_simpo_skips_without_trl(self):
        """SimPO stage should skip gracefully when trl is not installed."""
        from vetinari.training.dapo import StageResult, TrainingStageOrchestrator

        # Mock the DataCurator to avoid network calls for model loading
        mock_curator = MagicMock()
        mock_curator.curate_dpo.side_effect = ValueError("No preference data available")

        with patch(
            "vetinari.training.pipeline.DataCurator",
            return_value=mock_curator,
        ):
            orchestrator = TrainingStageOrchestrator()
            result = orchestrator._run_simpo("test-model", Path("."), {})
            assert result.success is True
            # Either it trained or it skipped
            assert result.stage_name == "simpo"
            assert isinstance(result, StageResult)

    def test_dapo_skips_without_groups(self):
        """DAPO stage should skip when insufficient response groups."""
        from vetinari.training.dapo import TrainingStageOrchestrator

        mock_collector = MagicMock()
        mock_collector.export_ranking_dataset.return_value = []  # No groups

        with patch(
            "vetinari.learning.training_data.get_training_collector",
            return_value=mock_collector,
        ):
            orchestrator = TrainingStageOrchestrator()
            result = orchestrator._run_dapo("test-model", Path("."), {})
            assert result.success is True
            assert result.metrics.get("skipped") is True

    def test_validate_stage_missing_path(self):
        """Validation should return False for a non-existent output path."""
        from vetinari.training.dapo import TrainingStageOrchestrator

        orchestrator = TrainingStageOrchestrator()
        result = orchestrator._validate_stage_output(
            "sft",
            "/input/model",
            "/nonexistent/output/model",
        )
        assert result is False

    def test_validate_stage_same_model(self):
        """Validation should return True when input equals output (stage skipped)."""
        from vetinari.training.dapo import TrainingStageOrchestrator

        orchestrator = TrainingStageOrchestrator()
        result = orchestrator._validate_stage_output(
            "simpo",
            "/some/model",
            "/some/model",
        )
        assert result is True


class TestCountingMethods:
    """Verify TrainingDataCollector counting methods."""

    def test_count_records(self):
        """count_records() should delegate to get_stats()."""
        from vetinari.learning.training_data import TrainingDataCollector

        collector = TrainingDataCollector.__new__(TrainingDataCollector)
        collector._output_path = Path("/nonexistent")
        collector._lock = MagicMock()
        collector._queue = MagicMock()
        collector._shutdown = MagicMock()
        collector._record_count = 0

        with patch.object(collector, "get_stats", return_value={"total": 42}):
            assert collector.count_records() == 42

    def test_count_reasoning_episodes_empty(self):
        """count_reasoning_episodes() should return 0 for nonexistent file."""
        from vetinari.learning.training_data import TrainingDataCollector

        collector = TrainingDataCollector.__new__(TrainingDataCollector)
        collector._output_path = Path("/nonexistent/training.jsonl")
        collector._lock = MagicMock()

        assert collector.count_reasoning_episodes() == 0

    def test_count_execution_traces_empty(self):
        """count_execution_traces() should return 0 for nonexistent file."""
        from vetinari.learning.training_data import TrainingDataCollector

        collector = TrainingDataCollector.__new__(TrainingDataCollector)
        collector._output_path = Path("/nonexistent/training.jsonl")
        collector._lock = MagicMock()

        assert collector.count_execution_traces() == 0
