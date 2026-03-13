"""Tests for vetinari.agents.decomposition_agent — decomposition interface."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vetinari.agents.decomposition_agent import (
    DEFAULT_MAX_DEPTH,
    MAX_MAX_DEPTH,
    MIN_MAX_DEPTH,
    RECURSION_KNOBS,
    SEED_MIX,
    SEED_RATE,
    DecompositionAgent,
    decomposition_agent,
)


class TestConstants:
    """Tests for module-level constants."""

    def test_recursion_knobs_keys(self):
        expected = {"breadth_first_weight", "min_subtask_words", "max_subtasks_per_level", "quality_threshold"}
        assert set(RECURSION_KNOBS.keys()) == expected

    def test_depth_bounds(self):
        assert MIN_MAX_DEPTH < DEFAULT_MAX_DEPTH < MAX_MAX_DEPTH

    def test_seed_values(self):
        assert 0 < SEED_RATE < 1
        assert 0 < SEED_MIX < 1


class TestDecompositionAgent:
    """Tests for the DecompositionAgent class."""

    def test_singleton_exists(self):
        assert isinstance(decomposition_agent, DecompositionAgent)

    def test_decompose_from_prompt_error_handling(self):
        """When decomposition_engine raises, returns error dict."""
        agent = DecompositionAgent()
        mock_engine = MagicMock()
        mock_engine.decompose_task.side_effect = RuntimeError("engine failure")
        with patch.dict("sys.modules", {"vetinari.decomposition": MagicMock(decomposition_engine=mock_engine)}):
            result = agent.decompose_from_prompt(None, "test prompt")
            assert result["status"] == "error"
            assert "engine failure" in result["error"]

    def test_decompose_from_prompt_returns_dict(self):
        """Result is always a dict with 'status' key."""
        agent = DecompositionAgent()
        result = agent.decompose_from_prompt(None, "test")
        assert isinstance(result, dict)
        assert "status" in result
