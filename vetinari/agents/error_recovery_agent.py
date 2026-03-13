"""Legacy redirect — use OperationsAgent directly."""

from __future__ import annotations

from vetinari.agents.consolidated.operations_agent import OperationsAgent as ErrorRecoveryAgent
from vetinari.agents.consolidated.operations_agent import get_operations_agent as get_error_recovery_agent

__all__ = ["ErrorRecoveryAgent", "get_error_recovery_agent"]
