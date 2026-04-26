"""Episodic Recall — high-level API for retrieving relevant past experiences.

Wraps EpisodeMemory with planning-oriented retrieval:
- Recall by task similarity
- Recall failure patterns for avoidance
- Recall successful strategies for reuse
- Adaptive retrieval (only on low confidence) for efficiency
- Importance scoring: recency * quality_impact * novelty

Simplifies the interface for plan generators and prompt assemblers
that need past experience context.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from vetinari.types import ConfidenceLevel

logger = logging.getLogger(__name__)

# Confidence levels that trigger episodic retrieval
_RETRIEVAL_CONFIDENCE_LEVELS: frozenset[ConfidenceLevel] = frozenset({
    ConfidenceLevel.LOW,
    ConfidenceLevel.VERY_LOW,
})


def recall_for_planning(
    goal: str,
    task_type: str = "general",
    k: int = 3,
) -> list[dict[str, Any]]:
    """Recall relevant past episodes for plan generation context.

    Returns successful episodes matching the goal description, formatted
    as planning context dicts with task_summary, output_summary, and quality.

    Args:
        goal: The goal or task description to match against.
        task_type: Task type filter.
        k: Maximum episodes to return.

    Returns:
        List of episode summary dicts for injection into planning prompts.
    """
    try:
        from vetinari.learning.episode_memory import get_episode_memory

        episodes = get_episode_memory().recall(
            query=goal[:300],
            k=k,
            successful_only=True,
            task_type=task_type,
        )
        return [
            {
                "task_summary": ep.task_summary,
                "output_summary": ep.output_summary,
                "quality_score": ep.quality_score,
                "agent_type": ep.agent_type,
                "model_id": ep.model_id,
            }
            for ep in episodes
        ]
    except Exception:
        logger.warning(
            "Episodic recall for planning failed — returning empty context so planning proceeds without history"
        )
        return []


def recall_failure_patterns(
    agent_type: str,
    task_type: str,
) -> list[str]:
    """Recall common failure patterns to inject as avoidance context.

    Args:
        agent_type: Agent type to check failures for.
        task_type: Task type to check failures for.

    Returns:
        List of failure summary strings, most recent first.
    """
    try:
        from vetinari.learning.episode_memory import get_episode_memory

        return get_episode_memory().get_failure_patterns(agent_type, task_type)
    except Exception:
        logger.warning(
            "Failure pattern recall failed for agent_type=%s task_type=%s — returning empty list",
            agent_type,
            task_type,
        )
        return []


def recall_few_shot_examples(
    task_type: str,
    k: int = 3,
) -> list[dict[str, str]]:
    """Recall top-scoring examples for few-shot prompt construction.

    Args:
        task_type: Task type to retrieve examples for.
        k: Maximum examples to return.

    Returns:
        List of {"input": ..., "output": ...} dicts for few-shot injection.
    """
    try:
        from vetinari.learning.training_data import get_training_collector

        return get_training_collector().export_few_shot_examples(task_type, k=k)
    except Exception:
        logger.warning(
            "Few-shot example recall failed for task_type=%s — returning empty list so prompt proceeds without examples",
            task_type,
        )
        return []


def recall_similar_episodes(
    task_description: str,
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM,
    k: int = 3,
) -> list[dict[str, Any]]:
    """Retrieve similar past episodes with adaptive retrieval and importance scoring.

    Only retrieves when confidence is LOW or VERY_LOW — high-confidence
    tasks skip retrieval for speed. Results are ranked by importance:
    ``recency * quality_impact * novelty``.

    Args:
        task_description: The current task description to match against.
        confidence_level: Current output confidence level. HIGH/MEDIUM skip
            retrieval for efficiency.
        k: Maximum episodes to return.

    Returns:
        List of episode context dicts with task_summary, approach,
        quality_score, errors, and importance_score. Empty list if
        confidence is high enough to skip retrieval.
    """
    # Adaptive retrieval: skip for high-confidence tasks
    if confidence_level not in _RETRIEVAL_CONFIDENCE_LEVELS:
        logger.info(
            "Episodic retrieval skipped — confidence=%s is above retrieval threshold",
            confidence_level.value,
        )
        return []

    try:
        from vetinari.learning.episode_memory import get_episode_memory

        episodes = get_episode_memory().recall(
            query=task_description[:300],
            k=k * 2,  # Fetch more for importance scoring
        )

        now = time.time()
        scored = []
        for ep in episodes:
            # Importance = recency * quality_impact * novelty
            age_hours = max(1.0, (now - getattr(ep, "timestamp", now)) / 3600)
            recency = 1.0 / (1.0 + age_hours / 24.0)  # Decays over days
            quality_impact = abs(getattr(ep, "quality_score", 0.5) - 0.5) * 2  # Distance from average
            novelty = 1.0  # Default — could be refined with dedup logic
            importance = recency * quality_impact * novelty

            scored.append({
                "task_summary": getattr(ep, "task_summary", ""),
                "approach": getattr(ep, "output_summary", ""),
                "quality_score": getattr(ep, "quality_score", 0.0),
                "errors": getattr(ep, "error_message", ""),
                "model_id": getattr(ep, "model_id", ""),
                "agent_type": getattr(ep, "agent_type", ""),
                "importance_score": round(importance, 4),
            })

        # Sort by importance descending, take top k
        scored.sort(key=lambda x: x["importance_score"], reverse=True)
        return scored[:k]

    except Exception:
        logger.warning(
            "Episodic recall for similar episodes failed — returning empty context so task proceeds without history"
        )
        return []
