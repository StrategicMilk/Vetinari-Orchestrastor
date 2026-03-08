"""
Vetinari Self-Refinement Loop
================================
Implements iterative self-correction for agent outputs.

After an agent produces an initial response, the loop:
1. Critiques the output using a second pass (optionally a different model)
2. If the critique identifies improvements, asks the agent to revise
3. Repeats up to MAX_ROUNDS times or until "no improvements needed"

Calibration for local models (5090, 32B Q4 model):
- Each round costs ~50-135 seconds of generation time
- Limit to 2 rounds for tasks where latency matters
- 3 rounds only for high-importance tasks (score below threshold)
- Trigger only when initial quality score < 0.7

Usage::

    from vetinari.learning.self_refinement import SelfRefinementLoop

    refiner = SelfRefinementLoop(adapter_manager)
    refined = refiner.refine(
        task_description="Implement a Redis cache wrapper",
        initial_output="class RedisCacheWrapper: pass",
        task_type="coding",
        model_id="qwen3-vl-32b",
        importance=0.9,   # 0.0-1.0; high importance = more rounds
    )
    print(refined.output)
    print(refined.rounds_used)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Maximum rounds per importance tier
_ROUNDS_BY_IMPORTANCE = {
    "high": 3,    # importance >= 0.8
    "medium": 2,  # importance >= 0.5
    "low": 1,     # importance < 0.5
}

# Only trigger refinement when initial quality is below this
_QUALITY_TRIGGER_THRESHOLD = float(
    os.environ.get("VETINARI_REFINE_THRESHOLD", "0.70")
)


@dataclass
class RefinementResult:
    """Result of a self-refinement cycle."""
    output: str
    rounds_used: int
    initial_quality: float
    final_quality: float
    improved: bool
    critique_summary: str = ""


class SelfRefinementLoop:
    """
    Iterative critique-and-revise loop for agent outputs.

    Calibrated for local LM Studio inference speed:
    - Importance >= 0.8 -> max 3 rounds
    - Importance 0.5-0.8 -> max 2 rounds
    - Importance < 0.5  -> max 1 round (single critique only)
    """

    def __init__(self, adapter_manager=None):
        self._adapter_manager = adapter_manager

    def refine(
        self,
        task_description: str,
        initial_output: str,
        task_type: str = "general",
        model_id: str = "default",
        importance: float = 0.5,
        initial_quality: Optional[float] = None,
    ) -> RefinementResult:
        """Run the self-refinement loop.

        Args:
            task_description: What the task asked for
            initial_output:   The agent's first response
            task_type:        coding / research / analysis / etc.
            model_id:         Model that produced the initial output
            importance:       Task importance (0.0-1.0); controls max rounds
            initial_quality:  Pre-computed quality score; skip if already good

        Returns:
            RefinementResult with the (possibly improved) output.
        """
        # Check if refinement is even needed
        if initial_quality is not None and initial_quality >= _QUALITY_TRIGGER_THRESHOLD:
            return RefinementResult(
                output=initial_output,
                rounds_used=0,
                initial_quality=initial_quality,
                final_quality=initial_quality,
                improved=False,
                critique_summary="Skipped: quality already above threshold",
            )

        if not self._adapter_manager:
            return RefinementResult(
                output=initial_output,
                rounds_used=0,
                initial_quality=initial_quality or 0.7,
                final_quality=initial_quality or 0.7,
                improved=False,
                critique_summary="Skipped: no adapter available",
            )

        max_rounds = self._get_max_rounds(importance)
        current_output = initial_output
        critique_summary = ""
        rounds_used = 0

        for round_num in range(1, max_rounds + 1):
            critique = self._critique(task_description, current_output, task_type, model_id)

            if not critique or "no improvements needed" in critique.lower():
                logger.debug("[SelfRefinement] Round %s: no improvements needed", round_num)
                break

            critique_summary = critique[:300]

            revised = self._revise(
                task_description, current_output, critique, task_type, model_id
            )
            if not revised or revised == current_output:
                logger.debug("[SelfRefinement] Round %s: revision unchanged, stopping", round_num)
                break

            current_output = revised
            rounds_used = round_num
            logger.info(
                f"[SelfRefinement] Round {round_num}/{max_rounds} completed for task '{task_description[:50]}'"
            )

        improved = rounds_used > 0 and current_output != initial_output
        final_quality = initial_quality or 0.7
        if improved:
            final_quality = min(1.0, (initial_quality or 0.65) + 0.05 * rounds_used)

        return RefinementResult(
            output=current_output,
            rounds_used=rounds_used,
            initial_quality=initial_quality or 0.65,
            final_quality=final_quality,
            improved=improved,
            critique_summary=critique_summary,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _critique(
        self,
        task_description: str,
        output: str,
        task_type: str,
        model_id: str,
    ) -> Optional[str]:
        """Ask the model to critique its own output."""
        critique_prompt = (
            f"Review this {task_type} output critically.\n\n"
            f"TASK: {task_description[:300]}\n\n"
            f"OUTPUT:\n{output[:2000]}\n\n"
            "If the output is complete and correct, respond with exactly: "
            "\"No improvements needed.\"\n"
            "Otherwise, list SPECIFIC improvements needed (max 3 bullet points). "
            "Be concrete, not generic."
        )
        try:
            return self._call_llm(
                model_id=model_id,
                system="You are a rigorous quality reviewer. Be honest and specific.",
                user=critique_prompt,
                max_tokens=300,
                temperature=0.2,
            )
        except Exception as e:
            logger.debug("[SelfRefinement] Critique failed: %s", e)
            return None

    def _revise(
        self,
        task_description: str,
        original_output: str,
        critique: str,
        task_type: str,
        model_id: str,
    ) -> Optional[str]:
        """Ask the model to produce a revised output based on the critique."""
        revise_prompt = (
            f"Revise this {task_type} output based on the critique.\n\n"
            f"TASK: {task_description[:300]}\n\n"
            f"PREVIOUS OUTPUT:\n{original_output[:2000]}\n\n"
            f"CRITIQUE:\n{critique[:500]}\n\n"
            "Provide the improved, complete output. Apply all critique points."
        )
        try:
            return self._call_llm(
                model_id=model_id,
                system="You are an expert reviser. Apply the critique precisely.",
                user=revise_prompt,
                max_tokens=2000,
                temperature=0.1,
            )
        except Exception as e:
            logger.debug("[SelfRefinement] Revise failed: %s", e)
            return None

    def _call_llm(
        self,
        model_id: str,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Optional[str]:
        """Call LM Studio directly."""
        host = os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")
        import requests as _req
        resp = _req.post(
            f"{host}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=180,  # Local models can be slow
        )
        if resp.status_code == 200:
            return (
                resp.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
        return None

    @staticmethod
    def _get_max_rounds(importance: float) -> int:
        if importance >= 0.8:
            return _ROUNDS_BY_IMPORTANCE["high"]
        if importance >= 0.5:
            return _ROUNDS_BY_IMPORTANCE["medium"]
        return _ROUNDS_BY_IMPORTANCE["low"]


# ---------------------------------------------------------------------------
# Module-level accessor
# ---------------------------------------------------------------------------

_refiner: Optional[SelfRefinementLoop] = None
_refiner_lock = __import__("threading").Lock()


def get_self_refiner(adapter_manager=None) -> SelfRefinementLoop:
    """Return the global SelfRefinementLoop singleton."""
    global _refiner
    if _refiner is None:
        with _refiner_lock:
            if _refiner is None:
                _refiner = SelfRefinementLoop(adapter_manager)
    return _refiner
