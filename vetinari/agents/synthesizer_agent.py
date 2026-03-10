"""Legacy redirect — use OperationsAgent directly."""
from vetinari.agents.consolidated.operations_agent import OperationsAgent as SynthesizerAgent, get_operations_agent as get_synthesizer_agent  # noqa: F401

__all__ = ["SynthesizerAgent", "get_synthesizer_agent"]
