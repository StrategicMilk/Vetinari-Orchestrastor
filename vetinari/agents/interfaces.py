"""
Agent Interface Contracts

This module defines formal interface contracts for agents, ensuring consistent
communication protocols and standardized capabilities across all agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol
from enum import Enum


class CapabilityType(Enum):
    """Types of agent capabilities."""
    DISCOVERY = "discovery"           # Explorer, Librarian
    ANALYSIS = "analysis"             # Researcher, Evaluator, Oracle
    SYNTHESIS = "synthesis"           # Synthesizer
    GENERATION = "generation"         # Builder, UI Planner
    VERIFICATION = "verification"     # Evaluator, Security Auditor
    DOCUMENTATION = "documentation"   # Documentation Agent
    OPTIMIZATION = "optimization"     # Cost Planner
    TESTING = "testing"               # Test Automation
    GOVERNANCE = "governance"         # Security Auditor


@dataclass
class Capability:
    """Definition of an agent capability."""
    name: str
    type: CapabilityType
    description: str
    input_schema: Dict[str, Any]      # JSON schema for inputs
    output_schema: Dict[str, Any]     # JSON schema for outputs
    version: str = "1.0.0"
    deprecated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "version": self.version,
            "deprecated": self.deprecated
        }


@dataclass
class AgentInterface:
    """Formal interface contract for an agent."""
    agent_name: str
    agent_type: str
    version: str
    capabilities: List[Capability] = field(default_factory=list)
    required_context: List[str] = field(default_factory=list)
    error_codes: Dict[str, str] = field(default_factory=dict)
    
    def get_capability(self, capability_name: str) -> Optional[Capability]:
        """Get a capability by name."""
        return next((c for c in self.capabilities if c.name == capability_name), None)
    
    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a capability."""
        return self.get_capability(capability_name) is not None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "version": self.version,
            "capabilities": [c.to_dict() for c in self.capabilities],
            "required_context": self.required_context,
            "error_codes": self.error_codes
        }


# ===== EXPLORER INTERFACE =====
EXPLORER_INTERFACE = AgentInterface(
    agent_name="Explorer",
    agent_type="EXPLORER",
    version="1.0.0",
    capabilities=[
        Capability(
            name="search_code_patterns",
            type=CapabilityType.DISCOVERY,
            description="Search for code patterns and implementations",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "scope": {"type": "string", "enum": ["code", "docs", "apis"]},
                    "max_results": {"type": "integer"}
                },
                "required": ["query"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "findings": {"type": "array"},
                    "references": {"type": "array"}
                }
            }
        )
    ],
    required_context=["codebase_path", "search_tools"]
)


# ===== LIBRARIAN INTERFACE =====
LIBRARIAN_INTERFACE = AgentInterface(
    agent_name="Librarian",
    agent_type="LIBRARIAN",
    version="1.0.0",
    capabilities=[
        Capability(
            name="lookup_documentation",
            type=CapabilityType.DISCOVERY,
            description="Look up authoritative documentation and APIs",
            input_schema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "string"}},
                    "depth": {"type": "string", "enum": ["summary", "detailed"]}
                },
                "required": ["topic"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "sources": {"type": "array"},
                    "fit_assessment": {"type": "string"}
                }
            }
        ),
        Capability(
            name="analyze_libraries",
            type=CapabilityType.ANALYSIS,
            description="Analyze available libraries and frameworks",
            input_schema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "criteria": {"type": "array"}
                },
                "required": ["domain"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "libraries": {"type": "array"},
                    "recommendations": {"type": "array"}
                }
            }
        )
    ],
    required_context=["api_keys", "documentation_sources"]
)


# ===== RESEARCHER INTERFACE =====
RESEARCHER_INTERFACE = AgentInterface(
    agent_name="Researcher",
    agent_type="RESEARCHER",
    version="1.0.0",
    capabilities=[
        Capability(
            name="domain_research",
            type=CapabilityType.ANALYSIS,
            description="Perform domain-specific research and analysis",
            input_schema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "questions": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["domain"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "findings": {"type": "array"},
                    "feasibility_score": {"type": "number"},
                    "recommendations": {"type": "array"}
                }
            }
        ),
        Capability(
            name="competitive_analysis",
            type=CapabilityType.ANALYSIS,
            description="Analyze competitors and market landscape",
            input_schema={
                "type": "object",
                "properties": {
                    "market": {"type": "string"},
                    "competitors": {"type": "array"}
                },
                "required": ["market"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "competitor_analysis": {"type": "array"},
                    "market_positioning": {"type": "string"}
                }
            }
        )
    ],
    required_context=["research_tools", "market_data"]
)


# ===== BUILDER INTERFACE =====
BUILDER_INTERFACE = AgentInterface(
    agent_name="Builder",
    agent_type="BUILDER",
    version="1.0.0",
    capabilities=[
        Capability(
            name="generate_scaffold",
            type=CapabilityType.GENERATION,
            description="Generate code scaffold from specification",
            input_schema={
                "type": "object",
                "properties": {
                    "spec": {"type": "string"},
                    "language": {"type": "string"},
                    "framework": {"type": "string"}
                },
                "required": ["spec"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "scaffold_code": {"type": "string"},
                    "tests": {"type": "array"},
                    "artifacts": {"type": "array"}
                }
            }
        )
    ],
    required_context=["code_generation_models", "template_library"]
)


# ===== EVALUATOR INTERFACE =====
EVALUATOR_INTERFACE = AgentInterface(
    agent_name="Evaluator",
    agent_type="EVALUATOR",
    version="1.0.0",
    capabilities=[
        Capability(
            name="evaluate_quality",
            type=CapabilityType.VERIFICATION,
            description="Evaluate code quality and design",
            input_schema={
                "type": "object",
                "properties": {
                    "artifacts": {"type": "array"},
                    "criteria": {"type": "array"}
                },
                "required": ["artifacts"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "verdict": {"type": "string"},
                    "quality_score": {"type": "number"},
                    "improvements": {"type": "array"}
                }
            }
        )
    ],
    required_context=["quality_standards", "analysis_tools"]
)


# ===== UI_PLANNER INTERFACE =====
UI_PLANNER_INTERFACE = AgentInterface(
    agent_name="UI Planner",
    agent_type="UI_PLANNER",
    version="1.0.0",
    capabilities=[
        Capability(
            name="design_ui",
            type=CapabilityType.GENERATION,
            description="Design user interface and user experience",
            input_schema={
                "type": "object",
                "properties": {
                    "requirements": {"type": "string"},
                    "framework": {"type": "string"},
                    "accessibility_level": {"type": "string"}
                },
                "required": ["requirements"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "ui_spec": {"type": "object"},
                    "components": {"type": "array"},
                    "design_tokens": {"type": "object"}
                }
            }
        )
    ],
    required_context=["design_systems", "accessibility_standards"]
)


# Interface registry
AGENT_INTERFACES: Dict[str, AgentInterface] = {
    "EXPLORER": EXPLORER_INTERFACE,
    "LIBRARIAN": LIBRARIAN_INTERFACE,
    "RESEARCHER": RESEARCHER_INTERFACE,
    "BUILDER": BUILDER_INTERFACE,
    "EVALUATOR": EVALUATOR_INTERFACE,
    "UI_PLANNER": UI_PLANNER_INTERFACE,
}


def get_agent_interface(agent_type: str) -> Optional[AgentInterface]:
    """Get interface contract for an agent type."""
    return AGENT_INTERFACES.get(agent_type)
