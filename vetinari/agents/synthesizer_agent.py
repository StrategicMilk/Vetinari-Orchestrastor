"""Legacy redirect — use OperationsAgent directly."""

from __future__ import annotations

from vetinari.agents.consolidated.operations_agent import OperationsAgent as SynthesizerAgent
from vetinari.agents.consolidated.operations_agent import get_operations_agent as get_synthesizer_agent

__all__ = ["SynthesizerAgent", "get_synthesizer_agent"]
