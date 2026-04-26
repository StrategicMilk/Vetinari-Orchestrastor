"""Tests for vetinari.learning.episodic_recall — planning-oriented recall API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vetinari.types import AgentType


def test_recall_for_planning_returns_empty_on_import_error():
    """recall_for_planning returns an empty list when the episode memory raises on import."""
    # Patch the local import inside the function so get_episode_memory raises
    with patch.dict("sys.modules", {"vetinari.learning.episode_memory": None}):
        from vetinari.learning import episodic_recall

        result = episodic_recall.recall_for_planning("build a REST API")

    assert isinstance(result, list)
    assert result == []


def test_recall_for_planning_success():
    """recall_for_planning formats episode dicts correctly on success."""
    fake_ep = MagicMock()
    fake_ep.task_summary = "Implement user auth"
    fake_ep.output_summary = "JWT-based auth added"
    fake_ep.quality_score = 0.92
    fake_ep.agent_type = AgentType.WORKER.value
    fake_ep.model_id = "llama3"

    fake_memory = MagicMock()
    fake_memory.recall.return_value = [fake_ep]

    with patch("vetinari.learning.episode_memory.get_episode_memory", return_value=fake_memory):
        from vetinari.learning import episodic_recall

        result = episodic_recall.recall_for_planning("add user authentication", task_type="code", k=1)

    assert len(result) == 1
    entry = result[0]
    assert entry["task_summary"] == "Implement user auth"
    assert entry["output_summary"] == "JWT-based auth added"
    assert entry["quality_score"] == 0.92
    assert entry["agent_type"] == AgentType.WORKER.value
    assert entry["model_id"] == "llama3"


def test_recall_for_planning_handles_exception_gracefully():
    """recall_for_planning returns empty list when the memory call raises."""
    with patch("vetinari.learning.episode_memory.get_episode_memory", side_effect=RuntimeError("db locked")):
        from vetinari.learning import episodic_recall

        result = episodic_recall.recall_for_planning("some goal")

    assert result == []


def test_recall_failure_patterns_success():
    """recall_failure_patterns delegates to get_failure_patterns and returns the result."""
    fake_memory = MagicMock()
    fake_memory.get_failure_patterns.return_value = ["timeout on LLM call", "empty output from Worker"]

    with patch("vetinari.learning.episode_memory.get_episode_memory", return_value=fake_memory):
        from vetinari.learning import episodic_recall

        result = episodic_recall.recall_failure_patterns(agent_type=AgentType.WORKER.value, task_type="code")

    assert result == ["timeout on LLM call", "empty output from Worker"]
    fake_memory.get_failure_patterns.assert_called_once_with(AgentType.WORKER.value, "code")


def test_recall_failure_patterns_handles_exception():
    """recall_failure_patterns returns empty list when the memory call raises."""
    with patch("vetinari.learning.episode_memory.get_episode_memory", side_effect=OSError("disk error")):
        from vetinari.learning import episodic_recall

        result = episodic_recall.recall_failure_patterns(agent_type=AgentType.FOREMAN.value, task_type="planning")

    assert result == []


def test_recall_few_shot_examples_success():
    """recall_few_shot_examples delegates to export_few_shot_examples and returns the result."""
    fake_collector = MagicMock()
    fake_collector.export_few_shot_examples.return_value = [
        {"input": "Write a function", "output": "def foo(): pass"},
    ]

    with patch("vetinari.learning.training_data.get_training_collector", return_value=fake_collector):
        from vetinari.learning import episodic_recall

        result = episodic_recall.recall_few_shot_examples(task_type="code", k=1)

    assert len(result) == 1
    assert result[0]["input"] == "Write a function"
    fake_collector.export_few_shot_examples.assert_called_once_with("code", k=1)


def test_recall_few_shot_examples_handles_exception():
    """recall_few_shot_examples returns empty list when the collector call raises."""
    with patch("vetinari.learning.training_data.get_training_collector", side_effect=ValueError("no data")):
        from vetinari.learning import episodic_recall

        result = episodic_recall.recall_few_shot_examples(task_type="general")

    assert result == []


# -- recall_similar_episodes adaptive retrieval -------------------------------


def test_recall_similar_episodes_high_confidence_skips_retrieval():
    """HIGH confidence skips retrieval — adaptive retrieval returns empty list."""
    from vetinari.learning.episodic_recall import recall_similar_episodes
    from vetinari.types import ConfidenceLevel

    # HIGH is not in the retrieval confidence set — should return [] immediately
    result = recall_similar_episodes(
        "implement a cache layer",
        confidence_level=ConfidenceLevel.HIGH,
    )
    assert result == []


def test_recall_similar_episodes_medium_confidence_skips_retrieval():
    """MEDIUM confidence also skips retrieval for efficiency."""
    from vetinari.learning.episodic_recall import recall_similar_episodes
    from vetinari.types import ConfidenceLevel

    result = recall_similar_episodes(
        "write unit tests",
        confidence_level=ConfidenceLevel.MEDIUM,
    )
    assert result == []


def test_recall_similar_episodes_low_confidence_attempts_retrieval():
    """LOW confidence triggers retrieval from episode memory."""
    from vetinari.learning.episodic_recall import recall_similar_episodes
    from vetinari.types import ConfidenceLevel

    fake_ep = MagicMock()
    fake_ep.task_summary = "Build API endpoint"
    fake_ep.output_summary = "REST endpoint built"
    fake_ep.quality_score = 0.85
    fake_ep.timestamp = 0.0
    fake_ep.model_id = "llama3"
    fake_ep.error_message = ""

    fake_memory = MagicMock()
    fake_memory.recall.return_value = [fake_ep]

    with patch("vetinari.learning.episode_memory.get_episode_memory", return_value=fake_memory):
        result = recall_similar_episodes(
            "build a REST endpoint",
            confidence_level=ConfidenceLevel.LOW,
            k=1,
        )

    # Retrieval was attempted — fake memory returned one episode
    assert isinstance(result, list)
    assert len(result) >= 1


def test_recall_similar_episodes_very_low_confidence_attempts_retrieval():
    """VERY_LOW confidence triggers retrieval — highest priority for past experience."""
    from vetinari.learning.episodic_recall import recall_similar_episodes
    from vetinari.types import ConfidenceLevel

    fake_memory = MagicMock()
    fake_memory.recall.return_value = []

    with patch("vetinari.learning.episode_memory.get_episode_memory", return_value=fake_memory):
        result = recall_similar_episodes(
            "do something very uncertain",
            confidence_level=ConfidenceLevel.VERY_LOW,
            k=3,
        )

    # Retrieval was attempted (fake memory returned empty — that's fine)
    fake_memory.recall.assert_called_once()
    assert isinstance(result, list)
