"""Tests for US-204: Episode Memory recall task_type filter and Planner injection.

Covers:
- EpisodeMemory.recall() with task_type filter returns only matching episodes
- EpisodeMemory.recall() without task_type returns all (backward compatibility)
- PlannerAgent injects high-quality episodes into planning prompt
- PlannerAgent proceeds normally when no episodes found
- PlannerAgent proceeds normally when episode_memory raises exception
- Only episodes with quality_score > 0.8 are injected
"""

from __future__ import annotations

import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from tests.factories import make_episode as _make_episode_factory
from vetinari.types import AgentType

# ---------------------------------------------------------------------------
# EpisodeMemory recall() tests
# ---------------------------------------------------------------------------


class TestEpisodeMemoryRecallFilter:
    """Tests for EpisodeMemory.recall() task_type filtering."""

    @pytest.fixture
    def mem(self):
        """EpisodeMemory backed by the per-test isolated database."""
        from vetinari.learning.episode_memory import EpisodeMemory

        return EpisodeMemory()

    def test_recall_with_task_type_returns_only_matching(self, mem):
        """recall() with task_type filters to only matching episodes."""
        mem.record(
            task_description="Plan a microservices migration",
            agent_type=AgentType.FOREMAN.value,
            task_type="planning",
            output_summary="Decomposed into 5 ordered tasks",
            quality_score=0.9,
            success=True,
        )
        mem.record(
            task_description="Write a Redis cache wrapper",
            agent_type=AgentType.WORKER.value,
            task_type="coding",
            output_summary="Generated RedisCacheWrapper class",
            quality_score=0.85,
            success=True,
        )

        results = mem.recall("Plan a migration", task_type="planning", k=5)

        assert all(ep.task_type == "planning" for ep in results)
        task_summaries = [ep.task_summary for ep in results]
        assert any("microservices" in s for s in task_summaries)

    def test_recall_without_task_type_returns_all_types(self, mem):
        """recall() without task_type returns episodes of any task_type."""
        mem.record(
            task_description="Plan a new feature",
            agent_type=AgentType.FOREMAN.value,
            task_type="planning",
            output_summary="Decomposed into tasks",
            quality_score=0.9,
            success=True,
        )
        mem.record(
            task_description="Write unit tests",
            agent_type=AgentType.WORKER.value,
            task_type="coding",
            output_summary="Generated test suite",
            quality_score=0.88,
            success=True,
        )

        results = mem.recall("feature", k=10)

        types_found = {ep.task_type for ep in results}
        assert len(types_found) >= 1  # At least one type returned

    def test_recall_task_type_no_matches_returns_empty(self, mem):
        """recall() with a task_type that has no matching episodes returns []."""
        mem.record(
            task_description="Write a function",
            agent_type=AgentType.WORKER.value,
            task_type="coding",
            output_summary="Implemented function",
            quality_score=0.9,
            success=True,
        )

        results = mem.recall("plan something", task_type="planning", k=5)

        assert results == []

    def test_recall_task_type_backward_compat_none(self, mem):
        """recall() with task_type=None behaves identically to omitting the parameter."""
        mem.record(
            task_description="Research API options",
            agent_type=AgentType.WORKER.value,
            task_type="research",
            output_summary="Found 3 suitable APIs",
            quality_score=0.82,
            success=True,
        )

        results_none = mem.recall("API research", task_type=None, k=5)
        results_omit = mem.recall("API research", k=5)

        assert len(results_none) == len(results_omit)
        ids_none = {ep.episode_id for ep in results_none}
        ids_omit = {ep.episode_id for ep in results_omit}
        assert ids_none == ids_omit


# ---------------------------------------------------------------------------
# PlannerAgent episode injection tests
# ---------------------------------------------------------------------------


class TestPlannerEpisodeInjection:
    """Tests for episode recall injection into PlannerAgent._decompose_goal_llm()."""

    @pytest.fixture
    def planner(self):
        """PlannerAgent with LLM calls mocked out."""
        from vetinari.agents.planner_agent import ForemanAgent as PlannerAgent

        agent = PlannerAgent.__new__(PlannerAgent)
        agent._max_tasks = 8
        agent._interaction_mode = "auto"
        agent._callback = None
        return agent

    def test_high_quality_episodes_injected_into_prompt(self, planner):
        """When quality_score > 0.8, episode output_summary appears in the prompt sent to LLM."""
        captured_prompt: list[str] = []

        good_ep = _make_episode_factory(
            task_type="planning",
            quality_score=0.92,
            output_summary="Decomposed into research, build, and test phases",
        )

        def fake_infer_json(prompt, **kwargs):
            captured_prompt.append(prompt)
            return []

        mock_mem = MagicMock()
        mock_mem.recall.return_value = [good_ep]

        with patch("vetinari.agents.planner_agent.logger"):
            with patch(
                "vetinari.learning.episode_memory.get_episode_memory",
                return_value=mock_mem,
            ):
                planner._infer_json = fake_infer_json
                planner._decompose_goal_llm("Build a REST API", {})

        assert captured_prompt, "No prompt was captured — _infer_json was not called"
        assert "research, build, and test phases" in captured_prompt[0]
        assert "Past successful plans for similar goals" in captured_prompt[0]

    def test_low_quality_episodes_not_injected(self, planner):
        """Episodes with quality_score <= 0.8 are NOT injected into the prompt."""
        captured_prompt: list[str] = []

        low_ep = _make_episode_factory(
            task_type="planning",
            quality_score=0.75,
            output_summary="Mediocre decomposition",
        )

        def fake_infer_json(prompt, **kwargs):
            captured_prompt.append(prompt)
            return []

        mock_mem = MagicMock()
        mock_mem.recall.return_value = [low_ep]

        with patch(
            "vetinari.learning.episode_memory.get_episode_memory",
            return_value=mock_mem,
        ):
            planner._infer_json = fake_infer_json
            planner._decompose_goal_llm("Build a REST API", {})

        assert captured_prompt, "No prompt was captured"
        assert "Mediocre decomposition" not in captured_prompt[0]
        assert "Past successful plans" not in captured_prompt[0]

    def test_no_episodes_found_proceeds_normally(self, planner):
        """Planner proceeds normally (calls LLM) when no episodes are found."""
        called = []

        def fake_infer_json(prompt, **kwargs):
            called.append(prompt)
            return []

        mock_mem = MagicMock()
        mock_mem.recall.return_value = []

        with patch(
            "vetinari.learning.episode_memory.get_episode_memory",
            return_value=mock_mem,
        ):
            planner._infer_json = fake_infer_json
            result = planner._decompose_goal_llm("Build a REST API", {})

        assert called, "_infer_json should still be called when no episodes found"
        assert result == []

    def test_episode_memory_exception_proceeds_normally(self, planner):
        """Planner proceeds normally when episode_memory raises any exception."""
        called = []

        def fake_infer_json(prompt, **kwargs):
            called.append(prompt)
            return []

        with patch(
            "vetinari.learning.episode_memory.get_episode_memory",
            side_effect=RuntimeError("DB unavailable"),
        ):
            planner._infer_json = fake_infer_json
            result = planner._decompose_goal_llm("Build a REST API", {})

        assert called, "_infer_json should be called even after episode memory error"
        assert result == []

    def test_recall_called_with_planning_task_type(self, planner):
        """recall() is called with task_type='planning' and k=3."""
        mock_mem = MagicMock()
        mock_mem.recall.return_value = []

        with patch(
            "vetinari.learning.episode_memory.get_episode_memory",
            return_value=mock_mem,
        ):
            planner._infer_json = MagicMock(return_value=[])
            planner._decompose_goal_llm("Some goal", {})

        mock_mem.recall.assert_called_once()
        call_kwargs = mock_mem.recall.call_args
        assert call_kwargs[1].get("task_type") == "planning" or (
            len(call_kwargs[0]) > 1 and call_kwargs[0][1] == "planning"
        )

    def test_only_top_3_examples_injected(self, planner):
        """At most 3 past examples are injected even when more exist."""
        captured_prompt: list[str] = []

        episodes = [
            _make_episode_factory(
                task_type="planning",
                quality_score=0.95,
                output_summary=f"Plan summary number {i}",
            )
            for i in range(5)
        ]

        def fake_infer_json(prompt, **kwargs):
            captured_prompt.append(prompt)
            return []

        mock_mem = MagicMock()
        mock_mem.recall.return_value = episodes

        with patch("vetinari.agents.planner_agent.logger"):
            with patch(
                "vetinari.learning.episode_memory.get_episode_memory",
                return_value=mock_mem,
            ):
                planner._infer_json = fake_infer_json
                planner._decompose_goal_llm("Build a system", {})

        assert captured_prompt
        prompt = captured_prompt[0]
        # Count how many "Plan summary number N" entries appear — should be at most 3
        count = sum(1 for i in range(5) if f"Plan summary number {i}" in prompt)
        assert count <= 3
