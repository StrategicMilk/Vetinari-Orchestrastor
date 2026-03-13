"""Legacy redirect — use OperationsAgent directly."""

from __future__ import annotations

from vetinari.agents.consolidated.operations_agent import OperationsAgent as ExperimentationManagerAgent
from vetinari.agents.consolidated.operations_agent import get_operations_agent as get_experimentation_manager_agent

__all__ = ["ExperimentationManagerAgent", "get_experimentation_manager_agent"]
