"""Legacy redirect — use OperationsAgent directly."""
from vetinari.agents.consolidated.operations_agent import OperationsAgent as DocumentationAgent, get_operations_agent as get_documentation_agent  # noqa: F401

__all__ = ["DocumentationAgent", "get_documentation_agent"]
