"""Agent interface contracts — re-export hub for the three-agent factory pipeline.

Imports from sub-modules to stay under the 550-line file limit:
- interface_types.py: CapabilityType, Capability, AgentInterface data classes
- foreman_interface.py: FOREMAN_INTERFACE constant
- worker_interface.py: WORKER_INTERFACE constant
- inspector_interface_data.py: INSPECTOR_INTERFACE constant
"""

from __future__ import annotations

from vetinari.agents.foreman_interface import FOREMAN_INTERFACE
from vetinari.agents.inspector_interface_data import INSPECTOR_INTERFACE
from vetinari.agents.interface_types import AgentInterface, Capability, CapabilityType
from vetinari.agents.worker_interface import WORKER_INTERFACE
from vetinari.types import AgentType

__all__ = [
    "AGENT_INTERFACES",
    "FOREMAN_INTERFACE",
    "INSPECTOR_INTERFACE",
    "WORKER_INTERFACE",
    "AgentInterface",
    "Capability",
    "CapabilityType",
    "get_agent_interface",
]


# Interface registry — 3-agent model only
AGENT_INTERFACES: dict[str, AgentInterface] = {
    AgentType.FOREMAN.value: FOREMAN_INTERFACE,
    AgentType.WORKER.value: WORKER_INTERFACE,
    AgentType.INSPECTOR.value: INSPECTOR_INTERFACE,
}


def get_agent_interface(agent_type: str) -> AgentInterface | None:
    """Get interface contract for an agent type.

    Args:
        agent_type: The agent type string (e.g. "FOREMAN", "WORKER", "INSPECTOR").

    Returns:
        AgentInterface for the given type, or None if not registered.
    """
    return AGENT_INTERFACES.get(agent_type)
