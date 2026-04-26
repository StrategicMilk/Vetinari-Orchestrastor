"""Tests for the module-level training API utility functions.

These functions are importable without a Flask request context and are consumed
by the dashboard and other internal callers.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vetinari.web.litestar_training_api import (
    get_quality_comparison,
    get_training_history,
    get_training_status,
)

# ---------------------------------------------------------------------------
# get_training_status
# ---------------------------------------------------------------------------


class TestGetTrainingStatus:
    def test_returns_dict_with_required_keys(self) -> None:
        status = get_training_status()

        assert isinstance(status, dict)
        for key in ("status", "current_job", "last_run", "records_collected", "curriculum_phase", "next_activity"):
            assert key in status, f"Missing key: {key}"

    def test_defaults_to_idle_when_subsystems_unavailable(self) -> None:
        with (
            patch("vetinari.web.litestar_training_api.get_training_status.__module__"),
        ):
            status = get_training_status()
        assert status["status"] in ("idle", "running")

    def test_records_collected_reflects_collector_stats(self) -> None:
        mock_collector = MagicMock()
        mock_collector.get_stats.return_value = {"total": 42}

        with patch(
            "vetinari.learning.training_data.get_training_collector",
            return_value=mock_collector,
        ):
            status = get_training_status()

        assert status["records_collected"] == 42

    def test_status_is_running_when_detector_not_idle(self) -> None:
        mock_detector = MagicMock()
        mock_detector.idle = False

        with (
            patch("vetinari.learning.training_data.get_training_collector", side_effect=ImportError),
            patch("vetinari.training.idle_scheduler.get_idle_detector", return_value=mock_detector),
            patch("vetinari.training.curriculum.TrainingCurriculum", side_effect=ImportError),
        ):
            status = get_training_status()

        assert status["status"] == "running"

    def test_tolerates_all_subsystem_failures(self) -> None:
        with (
            patch("vetinari.learning.training_data.get_training_collector", side_effect=RuntimeError("boom")),
            patch("vetinari.training.idle_scheduler.get_idle_detector", side_effect=RuntimeError("boom")),
            patch("vetinari.training.curriculum.TrainingCurriculum", side_effect=RuntimeError("boom")),
        ):
            status = get_training_status()

        # Should return sentinel values, not raise
        assert status["records_collected"] == 0
        assert status["curriculum_phase"] == "unknown"

    def test_next_activity_populated_when_curriculum_available(self) -> None:
        mock_curriculum = MagicMock()
        mock_curriculum.get_status.return_value = {"phase": "sft"}
        mock_act = MagicMock()
        mock_act.type.value = "train"
        mock_act.description = "SFT on code tasks"
        mock_act.priority = 1
        mock_curriculum.next_activity.return_value = mock_act

        with (
            patch("vetinari.learning.training_data.get_training_collector", side_effect=ImportError),
            patch("vetinari.training.idle_scheduler.get_idle_detector", side_effect=ImportError),
            patch("vetinari.training.curriculum.TrainingCurriculum", return_value=mock_curriculum),
        ):
            status = get_training_status()

        assert status["curriculum_phase"] == "sft"
        assert status["next_activity"] is not None
        assert status["next_activity"]["type"] == "train"
        assert status["next_activity"]["description"] == "SFT on code tasks"


# ---------------------------------------------------------------------------
# get_training_history
# ---------------------------------------------------------------------------


class TestGetTrainingHistory:
    def test_returns_list(self) -> None:
        history = get_training_history()
        assert isinstance(history, list)

    def test_returns_empty_list_when_all_subsystems_unavailable(self) -> None:
        with (
            patch("vetinari.training.quality_gate.get_training_quality_gate", side_effect=ImportError),
            patch("vetinari.learning.auto_tuner.get_auto_tuner", side_effect=ImportError),
        ):
            history = get_training_history()

        assert history == []

    def test_quality_gate_entries_tagged_with_type(self) -> None:
        mock_gate = MagicMock()
        mock_gate.get_history.return_value = [{"timestamp": "2026-01-01T00:00:00", "decision": "accept"}]

        with (
            patch("vetinari.training.quality_gate.get_training_quality_gate", return_value=mock_gate),
            patch("vetinari.learning.auto_tuner.get_auto_tuner", side_effect=ImportError),
        ):
            history = get_training_history()

        assert len(history) == 1
        assert history[0]["type"] == "quality_gate"
        assert history[0]["decision"] == "accept"

    def test_auto_tuner_entries_tagged_with_type(self) -> None:
        mock_tuner = MagicMock()
        mock_tuner.get_history.return_value = [{"timestamp": "2026-01-02T00:00:00", "action": "lr_adjust"}]

        with (
            patch("vetinari.training.quality_gate.get_training_quality_gate", side_effect=ImportError),
            patch("vetinari.learning.auto_tuner.get_auto_tuner", return_value=mock_tuner),
        ):
            history = get_training_history()

        assert len(history) == 1
        assert history[0]["type"] == "auto_tune"
        assert history[0]["action"] == "lr_adjust"

    def test_results_sorted_descending_by_timestamp(self) -> None:
        mock_gate = MagicMock()
        mock_gate.get_history.return_value = [
            {"timestamp": "2026-01-01T00:00:00"},
            {"timestamp": "2026-01-03T00:00:00"},
        ]
        mock_tuner = MagicMock()
        mock_tuner.get_history.return_value = [{"timestamp": "2026-01-02T00:00:00"}]

        with (
            patch("vetinari.training.quality_gate.get_training_quality_gate", return_value=mock_gate),
            patch("vetinari.learning.auto_tuner.get_auto_tuner", return_value=mock_tuner),
        ):
            history = get_training_history()

        timestamps = [e["timestamp"] for e in history]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_respects_limit_parameter(self) -> None:
        mock_gate = MagicMock()
        mock_gate.get_history.return_value = [{"timestamp": f"2026-01-{i:02d}T00:00:00"} for i in range(1, 11)]
        mock_tuner = MagicMock()
        mock_tuner.get_history.return_value = [{"timestamp": f"2026-02-{i:02d}T00:00:00"} for i in range(1, 11)]

        with (
            patch("vetinari.training.quality_gate.get_training_quality_gate", return_value=mock_gate),
            patch("vetinari.learning.auto_tuner.get_auto_tuner", return_value=mock_tuner),
        ):
            history = get_training_history(limit=5)

        assert len(history) == 5


# ---------------------------------------------------------------------------
# get_quality_comparison
# ---------------------------------------------------------------------------


class TestGetQualityComparison:
    def test_returns_dict_with_required_keys(self) -> None:
        result = get_quality_comparison()
        for key in ("baseline_quality", "candidate_quality", "quality_delta", "decision", "latency_ratio"):
            assert key in result, f"Missing key: {key}"

    def test_returns_no_data_sentinel_when_quality_gate_unavailable(self) -> None:
        with patch(
            "vetinari.training.quality_gate.get_training_quality_gate",
            side_effect=ImportError,
        ):
            result = get_quality_comparison()

        assert result["decision"] == "no_data"
        assert result["quality_delta"] == 0.0

    def test_returns_no_data_sentinel_when_history_empty(self) -> None:
        mock_gate = MagicMock()
        mock_gate.get_history.return_value = []

        with patch("vetinari.training.quality_gate.get_training_quality_gate", return_value=mock_gate):
            result = get_quality_comparison()

        assert result["decision"] == "no_data"

    def test_returns_values_from_latest_history_entry(self) -> None:
        mock_gate = MagicMock()
        mock_gate.get_history.return_value = [
            {
                "baseline_quality": 0.72,
                "candidate_quality": 0.85,
                "quality_delta": 0.13,
                "decision": "accept",
                "latency_ratio": 0.95,
            }
        ]

        with patch("vetinari.training.quality_gate.get_training_quality_gate", return_value=mock_gate):
            result = get_quality_comparison()

        assert result["baseline_quality"] == 0.72
        assert result["candidate_quality"] == 0.85
        assert result["quality_delta"] == 0.13
        assert result["decision"] == "accept"
        assert result["latency_ratio"] == 0.95

    def test_tolerates_subsystem_exception_gracefully(self) -> None:
        with patch(
            "vetinari.training.quality_gate.get_training_quality_gate",
            side_effect=RuntimeError("db locked"),
        ):
            result = get_quality_comparison()

        assert result["decision"] == "no_data"
        assert isinstance(result["latency_ratio"], float)
