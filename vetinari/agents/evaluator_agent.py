"""Legacy redirect — use QualityAgent directly."""

from __future__ import annotations

from vetinari.agents.consolidated.quality_agent import QualityAgent as EvaluatorAgent
from vetinari.agents.consolidated.quality_agent import get_quality_agent as get_evaluator_agent

__all__ = ["EvaluatorAgent", "get_evaluator_agent"]
