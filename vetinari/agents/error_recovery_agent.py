"""Legacy redirect — use OperationsAgent directly."""
from vetinari.agents.consolidated.operations_agent import OperationsAgent as ErrorRecoveryAgent, get_operations_agent as get_error_recovery_agent  # noqa: F401

__all__ = ["ErrorRecoveryAgent", "get_error_recovery_agent"]
