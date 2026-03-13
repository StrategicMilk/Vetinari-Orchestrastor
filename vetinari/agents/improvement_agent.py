"""Legacy redirect — use OperationsAgent directly."""

from __future__ import annotations

from vetinari.agents.consolidated.operations_agent import OperationsAgent as ImprovementAgent
from vetinari.agents.consolidated.operations_agent import get_operations_agent as get_improvement_agent

__all__ = ["ImprovementAgent", "get_improvement_agent"]
