"""Dynamic Complexity Router (C11).

================================
Classifies tasks as simple/moderate/complex and routes them to appropriate
pipeline subsets.

- Simple tasks skip research, go straight to Builder/Operations
- Moderate tasks follow the standard pipeline
- Complex tasks add Oracle contrarian review stage
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Complexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class RoutingDecision:
    """Result of complexity routing analysis."""

    complexity: Complexity
    skip_stages: list[str]
    add_stages: list[str]
    recommended_agents: list[str]
    reasoning: str
    execution_strategy: str = "pipeline"  # "pipeline" or "code_mode"

    def to_dict(self) -> dict[str, Any]:
        return {
            "complexity": self.complexity.value,
            "skip_stages": self.skip_stages,
            "add_stages": self.add_stages,
            "recommended_agents": self.recommended_agents,
            "reasoning": self.reasoning,
            "execution_strategy": self.execution_strategy,
        }


# ── Complexity signals ────────────────────────────────────────────────

_COMPLEX_SIGNALS = [
    r"architect",
    r"security audit",
    r"refactor\s+(?:entire|all|full|whole)",
    r"migrat(?:e|ion)",
    r"multi[- ]?(?:service|module|component)",
    r"distributed",
    r"backwards?\s*compat",
    r"performance\s+(?:critical|sensitive)",
    r"risk\s+assess",
    r"compliance",
    r"design\s+system",
]

_SIMPLE_SIGNALS = [
    r"^(?:fix|update|change|rename|add|remove)\s+\w+$",
    r"typo",
    r"bump\s+version",
    r"update\s+(?:readme|docs|changelog)",
    r"^format\b",
    r"^lint\b",
    r"simple\s+(?:bug|fix|change)",
    r"one[- ]line",
    r"trivial",
]


def classify_complexity(
    description: str,
    task_count: int = 1,
    estimated_files: int = 0,
) -> Complexity:
    """Classify a task's complexity.

    Uses keyword analysis, task count, and estimated file count as signals.
    """
    desc_lower = description.lower()

    # Check for explicit complexity signals
    complex_score = sum(1 for p in _COMPLEX_SIGNALS if re.search(p, desc_lower))
    simple_score = sum(1 for p in _SIMPLE_SIGNALS if re.search(p, desc_lower))

    # Word count as proxy for specification complexity
    word_count = len(description.split())
    if word_count > 100:
        complex_score += 2
    elif word_count > 50:
        complex_score += 1
    elif word_count < 15:
        simple_score += 1

    # Task count and file estimates
    if task_count > 5:
        complex_score += 2
    elif task_count > 2:
        complex_score += 1

    if estimated_files > 10:
        complex_score += 1
    elif estimated_files <= 2:
        simple_score += 1

    # Decision
    if complex_score >= 3 and complex_score > simple_score:
        return Complexity.COMPLEX
    if simple_score >= 2 and simple_score > complex_score:
        return Complexity.SIMPLE
    return Complexity.MODERATE


def route_by_complexity(
    description: str,
    task_count: int = 1,
    estimated_files: int = 0,
) -> RoutingDecision:
    """Produce a routing decision based on task complexity.

    Returns which pipeline stages to skip/add and which agents to engage.
    """
    complexity = classify_complexity(description, task_count, estimated_files)

    if complexity == Complexity.SIMPLE:
        return RoutingDecision(
            complexity=complexity,
            skip_stages=["research", "contrarian_review"],
            add_stages=[],
            recommended_agents=["BUILDER", "OPERATIONS"],
            reasoning="Simple task — skip research, direct to execution",
            execution_strategy="pipeline",
        )
    elif complexity == Complexity.COMPLEX:
        return RoutingDecision(
            complexity=complexity,
            skip_stages=[],
            add_stages=["contrarian_review", "risk_assessment"],
            recommended_agents=[
                "PLANNER",
                "CONSOLIDATED_RESEARCHER",
                "CONSOLIDATED_ORACLE",
                "BUILDER",
                "QUALITY",
            ],
            reasoning="Complex task — full pipeline with Oracle contrarian review",
            execution_strategy="code_mode" if task_count > 3 else "pipeline",
        )
    else:
        return RoutingDecision(
            complexity=complexity,
            skip_stages=[],
            add_stages=[],
            recommended_agents=["PLANNER", "BUILDER", "QUALITY"],
            reasoning="Moderate task — standard pipeline",
            execution_strategy="pipeline",
        )
