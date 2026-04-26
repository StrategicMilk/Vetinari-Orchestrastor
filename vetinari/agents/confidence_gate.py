"""Confidence-Gated Execution — routes agent output based on token logprob confidence.

After an agent generates output, this module assesses confidence from
token logprobs (free from llama-cpp-python — no reward model needed)
and routes to the appropriate next step. This is a thin wrapper around
the canonical ConfidenceComputer in vetinari.awareness.

Pipeline position: Intake → Planning → Execution → **Confidence Gate** → Quality Gate → Assembly.
"""

from __future__ import annotations

import logging

from vetinari.awareness.confidence import ConfidenceComputer, ConfidenceResult

logger = logging.getLogger(__name__)


class ConfidenceGate:
    """Thin wrapper around ConfidenceComputer for the agent execution pipeline.

    Provides the same interface the orchestrator expects while delegating
    all computation to the canonical ConfidenceComputer.
    """

    def __init__(
        self,
        threshold_high: float = -0.5,
        threshold_medium: float = -1.5,
        threshold_low: float = -3.0,
    ) -> None:
        self._computer = ConfidenceComputer(
            threshold_high=threshold_high,
            threshold_medium=threshold_medium,
            threshold_low=threshold_low,
        )
        # Expose thresholds for wiring.py introspection
        self._threshold_high = threshold_high
        self._threshold_medium = threshold_medium
        self._threshold_low = threshold_low

    def assess_confidence(self, logprobs: list[float]) -> ConfidenceResult:
        """Classify output confidence from a sequence of token logprobs.

        Args:
            logprobs: List of per-token log probabilities from the model.
                      Empty list triggers the 'I don't know' protocol.

        Returns:
            Canonical ConfidenceResult with score, level, action, and factors.
        """
        return self._computer.compute(logprobs)

    def route_by_confidence(
        self,
        logprobs: list[float],
        task_type: str = "general",
    ) -> ConfidenceResult:
        """Determine the post-generation action based on output confidence.

        Args:
            logprobs: Per-token log probabilities from the model.
            task_type: Task type for context (may influence routing).

        Returns:
            ConfidenceResult with level, action, score, and explanation.
        """
        result = self._computer.compute(logprobs, task_type=task_type)
        logger.info(
            "Confidence gate: level=%s score=%.3f action=%s task_type=%s",
            result.level.value,
            result.score,
            result.action.value,
            task_type,
        )
        return result

    def assess_semantic_entropy(
        self,
        responses: list[str],
        similarity_threshold: float = 0.7,
    ) -> ConfidenceResult:
        """Assess confidence via semantic entropy for open-ended tasks.

        Args:
            responses: Multiple generated responses for the same prompt.
            similarity_threshold: Fraction of similar responses needed for HIGH.

        Returns:
            ConfidenceResult based on response diversity.
        """
        return self._computer.compute_from_responses(responses, similarity_threshold)
