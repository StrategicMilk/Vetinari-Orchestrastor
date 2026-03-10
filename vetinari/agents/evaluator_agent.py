"""Legacy redirect — use QualityAgent directly."""
from vetinari.agents.consolidated.quality_agent import QualityAgent as EvaluatorAgent, get_quality_agent as get_evaluator_agent  # noqa: F401

__all__ = ["EvaluatorAgent", "get_evaluator_agent"]
