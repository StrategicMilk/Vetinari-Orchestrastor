"""Agent interface type definitions — CapabilityType, Capability, AgentInterface.

These data structures define the interface contracts used by the three-agent
factory pipeline (Foreman, Worker, Inspector). Imported by foreman_interface.py,
worker_interface.py, and inspector_interface_data.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vetinari.utils.serialization import dataclass_to_dict


class CapabilityType(Enum):
    """Types of agent capabilities."""

    DISCOVERY = "discovery"  # Search and lookup
    ANALYSIS = "analysis"  # Research, evaluation, architecture
    SYNTHESIS = "synthesis"  # Combining, summarising, narrating
    GENERATION = "generation"  # Code, UI, schema, pipeline generation
    VERIFICATION = "verification"  # Review, audit, contrarian checks
    DOCUMENTATION = "documentation"  # Docs, reports, changelogs
    OPTIMIZATION = "optimization"  # Cost, performance, simplification
    TESTING = "testing"  # Test generation and validation
    GOVERNANCE = "governance"  # Security, policy enforcement


@dataclass
class Capability:
    """Definition of an agent capability."""

    name: str
    type: CapabilityType
    description: str
    input_schema: dict[str, Any]  # JSON schema for inputs
    output_schema: dict[str, Any]  # JSON schema for outputs
    version: str = "1.0.0"
    deprecated: bool = False

    def __repr__(self) -> str:
        return f"Capability(name={self.name!r}, type={self.type!r}, version={self.version!r})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return dataclass_to_dict(self)


@dataclass
class AgentInterface:
    """Formal interface contract for an agent."""

    agent_name: str
    agent_type: str
    version: str
    capabilities: list[Capability] = field(default_factory=list)
    required_context: list[str] = field(default_factory=list)
    error_codes: dict[str, str] = field(default_factory=dict)

    def get_capability(self, capability_name: str) -> Capability | None:
        """Get a capability by name.

        Args:
            capability_name: Name of the capability to look up.

        Returns:
            Matching Capability, or None if not found.
        """
        return next((c for c in self.capabilities if c.name == capability_name), None)

    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a capability.

        Args:
            capability_name: Name of the capability to check.

        Returns:
            True if the capability exists, False otherwise.
        """
        return self.get_capability(capability_name) is not None

    def __repr__(self) -> str:
        return (
            f"AgentInterface(agent_name={self.agent_name!r}, agent_type={self.agent_type!r}, version={self.version!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return dataclass_to_dict(self)
