"""Decomposition Agent Interface.

==============================
Provides the decomposition_agent singleton and RECURSION_KNOBS used by
the Decomposition Lab API endpoints.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Recursion control knobs
RECURSION_KNOBS = {
    "breadth_first_weight": 0.6,  # Prefer breadth over depth
    "min_subtask_words": 5,  # Min words in subtask description
    "max_subtasks_per_level": 8,  # Max subtasks per decomposition level
    "quality_threshold": 0.65,  # Minimum quality score to accept a subtask
}

SEED_RATE = 0.3
SEED_MIX = 0.5
DEFAULT_MAX_DEPTH = 14
MIN_MAX_DEPTH = 12
MAX_MAX_DEPTH = 16


class DecompositionAgent:
    """Agent responsible for recursively decomposing plans via the PlannerAgent.

    Provides the `decompose_from_prompt` interface used by the Decomp Lab.
    """

    def decompose_from_prompt(self, plan: Any, prompt: str) -> dict[str, Any]:
        """Decompose a prompt in the context of an existing plan.

        Returns a dict with tasks, depth info, and metadata.

        Args:
            plan: The plan.
            prompt: The prompt.

        Returns:
            The result string.
        """
        try:
            from vetinari.decomposition import decomposition_engine

            subtasks = decomposition_engine.decompose_task(
                task_prompt=prompt,
                parent_task_id=getattr(plan, "plan_id", "root"),
                depth=0,
                max_depth=DEFAULT_MAX_DEPTH,
                plan_id=getattr(plan, "plan_id", "default"),
            )
            return {
                "status": "ok",
                "plan_id": getattr(plan, "plan_id", "unknown"),
                "prompt": prompt,
                "subtasks": subtasks,
                "subtask_count": len(subtasks),
                "knobs": RECURSION_KNOBS,
            }
        except Exception as e:
            logger.error("DecompositionAgent.decompose_from_prompt failed: %s", e)
            return {"status": "error", "error": str(e)}


# Module-level singleton
decomposition_agent = DecompositionAgent()
