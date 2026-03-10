"""Legacy redirect — use OperationsAgent directly."""
from vetinari.agents.consolidated.operations_agent import OperationsAgent as ImprovementAgent, get_operations_agent as get_improvement_agent  # noqa: F401

__all__ = ["ImprovementAgent", "get_improvement_agent"]
