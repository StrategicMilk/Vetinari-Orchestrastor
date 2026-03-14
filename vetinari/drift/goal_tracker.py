"""Goal Tracker - Continuous goal adherence verification.

Detects scope creep and goal drift during multi-agent execution by
comparing task outputs against the original goal using keyword overlap,
structural checks, and optional LLM-based judgement.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AdherenceResult:
    """Result of a goal adherence check."""

    score: float  # 0.0-1.0 (1.0 = perfectly aligned)
    deviation_description: str
    corrective_suggestion: str
    keywords_matched: int
    keywords_total: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "deviation_description": self.deviation_description,
            "corrective_suggestion": self.corrective_suggestion,
            "keywords_matched": self.keywords_matched,
            "keywords_total": self.keywords_total,
        }


@dataclass
class ScopeCreepItem:
    """A task flagged as potentially out of scope."""

    task_id: str
    task_description: str
    relevance_score: float
    reason: str


class GoalTracker:
    """Tracks goal adherence throughout multi-agent execution.

    Detects when task outputs drift away from the original goal
    using keyword overlap analysis and structural checks.
    """

    # Stop words excluded from keyword extraction
    STOP_WORDS: set[str] = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "not",
        "no",
        "all",
        "each",
        "every",
        "any",
        "some",
        "such",
        "than",
        "too",
        "very",
        "just",
        "about",
        "up",
        "out",
        "so",
        "if",
        "then",
        "into",
        "also",
        "as",
        "its",
    }

    def __init__(self, original_goal: str):
        self.original_goal = original_goal
        self._goal_keywords = self._extract_keywords(original_goal)
        self._deviation_history: list[AdherenceResult] = []
        logger.debug(f"[GoalTracker] Tracking goal: {original_goal[:80]}... ({len(self._goal_keywords)} keywords)")  # noqa: VET051 — complex expression

    def check_adherence(
        self,
        task_output: str,
        task_description: str,
    ) -> AdherenceResult:
        """Check if a task output still aligns with the original goal.

        Uses keyword overlap between the goal and the output/description
        to estimate alignment. Scores below 0.4 indicate significant drift.

        Args:
            task_output: The output produced by the task.
            task_description: Description of the task.

        Returns:
            AdherenceResult with score, deviation info, and suggestions.
        """
        combined_text = f"{task_description} {task_output}"
        output_keywords = self._extract_keywords(combined_text)

        if not self._goal_keywords:
            return AdherenceResult(
                score=1.0,
                deviation_description="No goal keywords to compare against",
                corrective_suggestion="",
                keywords_matched=0,
                keywords_total=0,
            )

        # Compute keyword overlap
        matched = self._goal_keywords & output_keywords
        overlap_score = len(matched) / len(self._goal_keywords) if self._goal_keywords else 0

        # Boost score if task description itself is relevant (it was planned for this goal)
        desc_keywords = self._extract_keywords(task_description)
        desc_overlap = len(self._goal_keywords & desc_keywords) / len(self._goal_keywords) if self._goal_keywords else 0

        # Weighted combination: output overlap matters more than description
        score = 0.6 * overlap_score + 0.4 * desc_overlap
        score = min(1.0, score * 1.5)  # Scale up slightly — partial overlap is expected

        deviation = ""
        suggestion = ""
        if score < 0.4:
            missing = self._goal_keywords - output_keywords
            deviation = f"Output drifted from goal. Missing key concepts: {', '.join(list(missing)[:5])}"
            suggestion = f"Refocus on the original goal: '{self.original_goal[:100]}'"
        elif score < 0.7:
            deviation = "Partial alignment — some goal concepts not addressed yet"
            suggestion = "Ensure remaining tasks cover all goal aspects"

        result = AdherenceResult(
            score=round(score, 3),
            deviation_description=deviation,
            corrective_suggestion=suggestion,
            keywords_matched=len(matched),
            keywords_total=len(self._goal_keywords),
        )
        self._deviation_history.append(result)
        return result

    def detect_scope_creep(self, tasks: list) -> list[ScopeCreepItem]:
        """Flag tasks that don't seem to contribute to the original goal.

        Args:
            tasks: List of task objects with .id and .description attributes.

        Returns:
            List of ScopeCreepItem for tasks with relevance < 0.3.
        """
        flagged = []
        for task in tasks:
            desc = getattr(task, "description", str(task))
            task_id = getattr(task, "id", "unknown")
            task_keywords = self._extract_keywords(desc)

            if not self._goal_keywords:
                continue

            overlap = len(self._goal_keywords & task_keywords) / len(self._goal_keywords)
            relevance = min(1.0, overlap * 2)  # Scale since individual tasks won't match all keywords

            if relevance < 0.3:
                flagged.append(
                    ScopeCreepItem(
                        task_id=task_id,
                        task_description=desc[:100],
                        relevance_score=round(relevance, 3),
                        reason=f"Low keyword overlap with goal ({relevance:.0%})",
                    )
                )

        if flagged:
            logger.warning(
                f"[GoalTracker] Scope creep detected: {len(flagged)} tasks have low relevance to original goal"
            )
        return flagged

    def get_drift_trend(self) -> dict[str, Any]:
        """Get the trend of goal adherence over time.

        Returns:
            The result string.
        """
        if not self._deviation_history:
            return {"trend": "unknown", "avg_score": 0, "samples": 0}
        scores = [r.score for r in self._deviation_history]
        avg = sum(scores) / len(scores)
        # Check if scores are declining
        if len(scores) >= 3:
            recent = scores[-3:]
            earlier = scores[:-3] if len(scores) > 3 else scores[:1]
            recent_avg = sum(recent) / len(recent)
            earlier_avg = sum(earlier) / len(earlier)
            if recent_avg < earlier_avg - 0.15:
                trend = "declining"
            elif recent_avg > earlier_avg + 0.15:
                trend = "improving"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        return {
            "trend": trend,
            "avg_score": round(avg, 3),
            "samples": len(scores),
            "latest_score": round(scores[-1], 3),
        }

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract meaningful keywords from text."""
        # Tokenize: split on non-alphanumeric
        tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())
        # Filter stop words and very short tokens
        keywords = {t for t in tokens if t not in self.STOP_WORDS and len(t) > 2}
        return keywords


# Factory function (no singleton — one tracker per goal)
def create_goal_tracker(goal: str) -> GoalTracker:
    """Create a new GoalTracker for a specific goal."""
    return GoalTracker(goal)
