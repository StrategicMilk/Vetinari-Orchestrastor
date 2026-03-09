"""Legacy redirect — use OperationsAgent directly."""
from vetinari.agents.consolidated.operations_agent import OperationsAgent as ExperimentationManagerAgent, get_operations_agent as get_experimentation_manager_agent  # noqa: F401

__all__ = ["ExperimentationManagerAgent", "get_experimentation_manager_agent"]
