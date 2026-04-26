"""Tests for ContextDistillationDatasetBuilder in vetinari.training.pipeline.

Uses a mock EpisodeMemory to avoid touching the SQLite database.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.factories import make_mock_episode, make_mock_episodes
from vetinari.training.pipeline import ContextDistillationDatasetBuilder, DistillationDatasetInfo
from vetinari.types import AgentType

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildDatasetFromEpisodes:
    def test_build_dataset_from_episodes(self, tmp_path: Path):
        """build_dataset writes a JSONL file and returns DistillationDatasetInfo."""
        episodes = make_mock_episodes(150)
        builder = ContextDistillationDatasetBuilder(min_quality=0.8, min_pairs=10)

        with patch.object(builder, "_query_episodes", return_value=episodes):
            output = tmp_path / "dataset.jsonl"
            info = builder.build_dataset(output)

        assert info is not None
        assert isinstance(info, DistillationDatasetInfo)
        assert info.num_examples == 150
        assert Path(info.output_path).exists()

    def test_dataset_file_has_correct_line_count(self, tmp_path: Path):
        """Each episode produces exactly one JSONL line."""
        episodes = make_mock_episodes(50)
        builder = ContextDistillationDatasetBuilder(min_quality=0.8, min_pairs=10)

        with patch.object(builder, "_query_episodes", return_value=episodes):
            output = tmp_path / "out.jsonl"
            info = builder.build_dataset(output)

        lines = Path(info.output_path).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 50


class TestQualityFiltering:
    def test_quality_filtering(self, tmp_path: Path):
        """Builder only includes episodes meeting the quality threshold (via mock)."""
        high_q = make_mock_episodes(120, quality=0.95)
        # The builder delegates filtering to EpisodeMemory; we test that the
        # avg_quality in the info reflects the actual episodes returned.
        builder = ContextDistillationDatasetBuilder(min_quality=0.9, min_pairs=10)

        with patch.object(builder, "_query_episodes", return_value=high_q):
            info = builder.build_dataset(tmp_path / "data.jsonl")

        assert info is not None
        assert abs(info.avg_quality - 0.95) < 1e-4

    def test_task_type_passed_to_query(self):
        """build_dataset passes task_type through to _query_episodes."""
        builder = ContextDistillationDatasetBuilder(min_quality=0.5, min_pairs=1)

        with patch.object(builder, "_query_episodes", return_value=make_mock_episodes(10)) as mock_q:
            with patch("builtins.open", MagicMock()):
                # We just care that the call was made with the right task_type
                pass  # open() mock prevents actual file write

        # Direct call to verify argument forwarding
        with patch.object(builder, "_query_episodes", return_value=[]) as mock_q:
            builder.build_dataset("/dev/null", task_type="coding")
        mock_q.assert_called_once_with("coding")


class TestJSONLFormatCorrect:
    def test_jsonl_format_correct(self, tmp_path: Path):
        """Every line in the output is valid JSON with required keys."""
        episodes = make_mock_episodes(5)
        builder = ContextDistillationDatasetBuilder(min_quality=0.0, min_pairs=1)

        with patch.object(builder, "_query_episodes", return_value=episodes):
            info = builder.build_dataset(tmp_path / "check.jsonl")

        assert isinstance(info, DistillationDatasetInfo)
        assert info.num_examples > 0
        lines = Path(info.output_path).read_text(encoding="utf-8").splitlines()
        for line in lines:
            record = json.loads(line)
            assert "instruction" in record
            assert "input" in record
            assert "output" in record
            assert "metadata" in record
            assert record["input"] == ""  # Alpaca-style — empty input field

    def test_jsonl_metadata_keys_present(self, tmp_path: Path):
        """Each JSONL record's metadata contains quality, agent_type, model_id."""
        ep = make_mock_episode(quality_score=0.88, agent_type="ORACLE", model_id="llama-3")
        builder = ContextDistillationDatasetBuilder(min_quality=0.0, min_pairs=1)

        with patch.object(builder, "_query_episodes", return_value=[ep]):
            info = builder.build_dataset(tmp_path / "meta.jsonl")

        record = json.loads(Path(info.output_path).read_text(encoding="utf-8"))
        assert record["metadata"]["quality"] == pytest.approx(0.88)
        assert record["metadata"]["agent_type"] == "ORACLE"
        assert record["metadata"]["model_id"] == "llama-3"


class TestInsufficientDataReturnsNone:
    def test_insufficient_data_returns_none(self, tmp_path: Path):
        """build_dataset returns None when fewer than min_pairs episodes found."""
        episodes = make_mock_episodes(5)  # Only 5, but min_pairs=100
        builder = ContextDistillationDatasetBuilder(min_quality=0.8, min_pairs=100)

        with patch.object(builder, "_query_episodes", return_value=episodes):
            result = builder.build_dataset(tmp_path / "empty.jsonl")

        assert result is None
        # File should NOT have been written
        assert not (tmp_path / "empty.jsonl").exists()

    def test_exactly_min_pairs_is_accepted(self, tmp_path: Path):
        """Exactly min_pairs episodes should proceed (not return None)."""
        episodes = make_mock_episodes(10)
        builder = ContextDistillationDatasetBuilder(min_quality=0.0, min_pairs=10)

        with patch.object(builder, "_query_episodes", return_value=episodes):
            result = builder.build_dataset(tmp_path / "exact.jsonl")

        assert result is not None
        assert result.num_examples == 10

    def test_zero_episodes_returns_none(self, tmp_path: Path):
        """Empty episode list returns None regardless of min_pairs."""
        builder = ContextDistillationDatasetBuilder(min_quality=0.8, min_pairs=1)

        with patch.object(builder, "_query_episodes", return_value=[]):
            result = builder.build_dataset(tmp_path / "none.jsonl")

        assert result is None
