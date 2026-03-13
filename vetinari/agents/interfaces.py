"""Agent Interface Contracts.

This module defines formal interface contracts for agents, ensuring consistent
communication protocols and standardized capabilities across all agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CapabilityType(Enum):
    """Types of agent capabilities."""

    DISCOVERY = "discovery"  # Explorer, Librarian
    ANALYSIS = "analysis"  # Researcher, Evaluator, Oracle
    SYNTHESIS = "synthesis"  # Synthesizer
    GENERATION = "generation"  # Builder, UI Planner
    VERIFICATION = "verification"  # Evaluator, Security Auditor
    DOCUMENTATION = "documentation"  # Documentation Agent
    OPTIMIZATION = "optimization"  # Cost Planner
    TESTING = "testing"  # Test Automation
    GOVERNANCE = "governance"  # Security Auditor


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "version": self.version,
            "deprecated": self.deprecated,
        }


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
        """Get a capability by name."""
        return next((c for c in self.capabilities if c.name == capability_name), None)

    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a capability."""
        return self.get_capability(capability_name) is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "version": self.version,
            "capabilities": [c.to_dict() for c in self.capabilities],
            "required_context": self.required_context,
            "error_codes": self.error_codes,
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
                    "max_results": {"type": "integer"},
                },
                "required": ["query"],
            },
            output_schema={
                "type": "object",
                "properties": {"findings": {"type": "array"}, "references": {"type": "array"}},
            },
        )
    ],
    required_context=["codebase_path", "search_tools"],
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
                    "depth": {"type": "string", "enum": ["summary", "detailed"]},
                },
                "required": ["topic"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "sources": {"type": "array"},
                    "fit_assessment": {"type": "string"},
                },
            },
        ),
        Capability(
            name="analyze_libraries",
            type=CapabilityType.ANALYSIS,
            description="Analyze available libraries and frameworks",
            input_schema={
                "type": "object",
                "properties": {"domain": {"type": "string"}, "criteria": {"type": "array"}},
                "required": ["domain"],
            },
            output_schema={
                "type": "object",
                "properties": {"libraries": {"type": "array"}, "recommendations": {"type": "array"}},
            },
        ),
    ],
    required_context=["api_keys", "documentation_sources"],
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
                    "questions": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["domain"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "findings": {"type": "array"},
                    "feasibility_score": {"type": "number"},
                    "recommendations": {"type": "array"},
                },
            },
        ),
        Capability(
            name="competitive_analysis",
            type=CapabilityType.ANALYSIS,
            description="Analyze competitors and market landscape",
            input_schema={
                "type": "object",
                "properties": {"market": {"type": "string"}, "competitors": {"type": "array"}},
                "required": ["market"],
            },
            output_schema={
                "type": "object",
                "properties": {"competitor_analysis": {"type": "array"}, "market_positioning": {"type": "string"}},
            },
        ),
    ],
    required_context=["research_tools", "market_data"],
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
                    "framework": {"type": "string"},
                },
                "required": ["spec"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "scaffold_code": {"type": "string"},
                    "tests": {"type": "array"},
                    "artifacts": {"type": "array"},
                },
            },
        )
    ],
    required_context=["code_generation_models", "template_library"],
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
                "properties": {"artifacts": {"type": "array"}, "criteria": {"type": "array"}},
                "required": ["artifacts"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "verdict": {"type": "string"},
                    "quality_score": {"type": "number"},
                    "improvements": {"type": "array"},
                },
            },
        )
    ],
    required_context=["quality_standards", "analysis_tools"],
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
                    "accessibility_level": {"type": "string"},
                },
                "required": ["requirements"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "ui_spec": {"type": "object"},
                    "components": {"type": "array"},
                    "design_tokens": {"type": "object"},
                },
            },
        )
    ],
    required_context=["design_systems", "accessibility_standards"],
)


# ===== PLANNER INTERFACE =====
PLANNER_INTERFACE = AgentInterface(
    agent_name="Planner",
    agent_type="PLANNER",
    version="1.0.0",
    capabilities=[
        Capability(
            name="goal_decomposition",
            type=CapabilityType.ANALYSIS,
            description="Decompose a high-level goal into ordered tasks",
            input_schema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "context": {"type": "object"},
                },
                "required": ["goal"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "tasks": {"type": "array"},
                    "plan_id": {"type": "string"},
                },
            },
        ),
        Capability(
            name="task_sequencing",
            type=CapabilityType.ANALYSIS,
            description="Sequence tasks with dependency resolution",
            input_schema={
                "type": "object",
                "properties": {
                    "tasks": {"type": "array"},
                },
                "required": ["tasks"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "ordered_tasks": {"type": "array"},
                    "dependency_graph": {"type": "object"},
                },
            },
        ),
        Capability(
            name="user_clarification",
            type=CapabilityType.ANALYSIS,
            description="Ask the user clarifying questions before planning",
            input_schema={
                "type": "object",
                "properties": {
                    "ambiguity": {"type": "string"},
                },
                "required": ["ambiguity"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array"},
                },
            },
        ),
        Capability(
            name="plan_consolidation",
            type=CapabilityType.SYNTHESIS,
            description="Consolidate multiple plans or partial plans into one",
            input_schema={
                "type": "object",
                "properties": {
                    "plans": {"type": "array"},
                },
                "required": ["plans"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "consolidated_plan": {"type": "object"},
                },
            },
        ),
        Capability(
            name="context_management",
            type=CapabilityType.ANALYSIS,
            description="Manage and summarise accumulated context",
            input_schema={
                "type": "object",
                "properties": {
                    "context": {"type": "object"},
                    "max_tokens": {"type": "integer"},
                },
                "required": ["context"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "pruned_context": {"type": "object"},
                    "summary": {"type": "string"},
                },
            },
        ),
        Capability(
            name="dependency_resolution",
            type=CapabilityType.ANALYSIS,
            description="Detect and resolve circular or missing task dependencies",
            input_schema={
                "type": "object",
                "properties": {
                    "tasks": {"type": "array"},
                },
                "required": ["tasks"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "resolved_tasks": {"type": "array"},
                    "warnings": {"type": "array"},
                },
            },
        ),
    ],
    required_context=["goal", "available_agents"],
    error_codes={
        "AMBIGUOUS_GOAL": "Goal requires clarification before planning",
        "CIRCULAR_DEPENDENCY": "Task dependency graph contains a cycle",
        "NO_VIABLE_PLAN": "Could not produce a valid plan for the given goal",
    },
)


# ===== CONSOLIDATED_RESEARCHER INTERFACE =====
CONSOLIDATED_RESEARCHER_INTERFACE = AgentInterface(
    agent_name="Researcher",
    agent_type="CONSOLIDATED_RESEARCHER",
    version="1.0.0",
    capabilities=[
        Capability(
            name="code_discovery",
            type=CapabilityType.DISCOVERY,
            description="Search for code patterns, symbols, and implementations in the codebase",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "scope": {"type": "string", "enum": ["code", "docs", "apis", "all"]},
                    "max_results": {"type": "integer"},
                },
                "required": ["query"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "findings": {"type": "array"},
                    "references": {"type": "array"},
                },
            },
        ),
        Capability(
            name="domain_research",
            type=CapabilityType.ANALYSIS,
            description="Perform domain-specific research and feasibility analysis",
            input_schema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "questions": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["domain"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "findings": {"type": "array"},
                    "feasibility_score": {"type": "number"},
                    "recommendations": {"type": "array"},
                },
            },
        ),
        Capability(
            name="api_lookup",
            type=CapabilityType.DISCOVERY,
            description="Look up authoritative API and library documentation",
            input_schema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["topic"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "sources": {"type": "array"},
                    "fit_assessment": {"type": "string"},
                },
            },
        ),
        Capability(
            name="lateral_thinking",
            type=CapabilityType.ANALYSIS,
            description="Generate alternative approaches and creative solutions",
            input_schema={
                "type": "object",
                "properties": {
                    "problem": {"type": "string"},
                    "constraints": {"type": "array"},
                },
                "required": ["problem"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "alternatives": {"type": "array"},
                    "rationale": {"type": "string"},
                },
            },
        ),
        Capability(
            name="ui_design",
            type=CapabilityType.GENERATION,
            description="Design UI/UX specifications and component structures",
            input_schema={
                "type": "object",
                "properties": {
                    "requirements": {"type": "string"},
                    "framework": {"type": "string"},
                    "accessibility_level": {"type": "string"},
                },
                "required": ["requirements"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "ui_spec": {"type": "object"},
                    "components": {"type": "array"},
                    "design_tokens": {"type": "object"},
                },
            },
        ),
        Capability(
            name="database_schema_design",
            type=CapabilityType.GENERATION,
            description="Design database schemas, indexes, and data models",
            input_schema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "entities": {"type": "array"},
                    "db_type": {"type": "string"},
                },
                "required": ["domain"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "schema": {"type": "object"},
                    "migrations": {"type": "array"},
                },
            },
        ),
        Capability(
            name="devops_pipeline_design",
            type=CapabilityType.GENERATION,
            description="Design CI/CD pipelines, infrastructure, and deployment strategies",
            input_schema={
                "type": "object",
                "properties": {
                    "project_type": {"type": "string"},
                    "target_platform": {"type": "string"},
                },
                "required": ["project_type"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "pipeline_config": {"type": "object"},
                    "steps": {"type": "array"},
                },
            },
        ),
        Capability(
            name="git_workflow_analysis",
            type=CapabilityType.ANALYSIS,
            description="Analyse git history, branches, and recommend workflow improvements",
            input_schema={
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string"},
                    "focus": {"type": "string"},
                },
                "required": ["repo_path"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "analysis": {"type": "string"},
                    "recommendations": {"type": "array"},
                },
            },
        ),
    ],
    required_context=["codebase_path", "search_tools"],
    error_codes={
        "NOT_FOUND": "No results found for the given query",
        "SOURCE_UNAVAILABLE": "Required documentation source is unreachable",
    },
)


# ===== CONSOLIDATED_ORACLE INTERFACE =====
CONSOLIDATED_ORACLE_INTERFACE = AgentInterface(
    agent_name="Oracle",
    agent_type="CONSOLIDATED_ORACLE",
    version="1.0.0",
    capabilities=[
        Capability(
            name="architecture_decision",
            type=CapabilityType.ANALYSIS,
            description="Evaluate architectural options and produce a decision record",
            input_schema={
                "type": "object",
                "properties": {
                    "options": {"type": "array"},
                    "constraints": {"type": "array"},
                    "goals": {"type": "array"},
                },
                "required": ["options"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "decision": {"type": "string"},
                    "rationale": {"type": "string"},
                    "trade_offs": {"type": "array"},
                },
            },
        ),
        Capability(
            name="risk_assessment",
            type=CapabilityType.ANALYSIS,
            description="Identify and score risks for a proposed plan or design",
            input_schema={
                "type": "object",
                "properties": {
                    "proposal": {"type": "string"},
                    "context": {"type": "object"},
                },
                "required": ["proposal"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "risks": {"type": "array"},
                    "severity_scores": {"type": "object"},
                    "mitigations": {"type": "array"},
                },
            },
        ),
        Capability(
            name="ontological_analysis",
            type=CapabilityType.ANALYSIS,
            description="Deep conceptual analysis of problem domain and relationships",
            input_schema={
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "domain": {"type": "string"},
                },
                "required": ["concept"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "analysis": {"type": "string"},
                    "related_concepts": {"type": "array"},
                },
            },
        ),
        Capability(
            name="contrarian_review",
            type=CapabilityType.VERIFICATION,
            description="Challenge assumptions and surface blind spots in a proposal",
            input_schema={
                "type": "object",
                "properties": {
                    "proposal": {"type": "string"},
                    "assumptions": {"type": "array"},
                },
                "required": ["proposal"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "challenges": {"type": "array"},
                    "blind_spots": {"type": "array"},
                    "verdict": {"type": "string"},
                },
            },
        ),
    ],
    required_context=["problem_statement", "system_context"],
    error_codes={
        "INSUFFICIENT_CONTEXT": "Not enough context to make an architectural decision",
        "CONFLICTING_CONSTRAINTS": "Constraints are mutually exclusive",
    },
)


# ===== QUALITY INTERFACE =====
QUALITY_INTERFACE = AgentInterface(
    agent_name="Quality",
    agent_type="QUALITY",
    version="1.0.0",
    capabilities=[
        Capability(
            name="code_review",
            type=CapabilityType.VERIFICATION,
            description="Review code quality, logic, and maintainability",
            input_schema={
                "type": "object",
                "properties": {
                    "artifacts": {"type": "array"},
                    "criteria": {"type": "array"},
                },
                "required": ["artifacts"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "verdict": {"type": "string"},
                    "quality_score": {"type": "number"},
                    "improvements": {"type": "array"},
                },
            },
        ),
        Capability(
            name="security_audit",
            type=CapabilityType.GOVERNANCE,
            description="Audit code for security vulnerabilities and policy violations",
            input_schema={
                "type": "object",
                "properties": {
                    "artifacts": {"type": "array"},
                    "threat_model": {"type": "object"},
                },
                "required": ["artifacts"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "vulnerabilities": {"type": "array"},
                    "severity_scores": {"type": "object"},
                    "remediation": {"type": "array"},
                },
            },
        ),
        Capability(
            name="test_generation",
            type=CapabilityType.TESTING,
            description="Generate unit, integration, and e2e test suites",
            input_schema={
                "type": "object",
                "properties": {
                    "module": {"type": "string"},
                    "coverage_target": {"type": "number"},
                    "test_types": {"type": "array"},
                },
                "required": ["module"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "tests": {"type": "array"},
                    "coverage_estimate": {"type": "number"},
                },
            },
        ),
        Capability(
            name="code_simplification",
            type=CapabilityType.OPTIMIZATION,
            description="Simplify and refactor code while preserving behaviour",
            input_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "simplification_goals": {"type": "array"},
                },
                "required": ["code"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "simplified_code": {"type": "string"},
                    "changes_summary": {"type": "string"},
                },
            },
        ),
    ],
    required_context=["quality_standards", "analysis_tools"],
    error_codes={
        "NO_ARTIFACTS": "No code artifacts provided for review",
        "UNSUPPORTED_LANGUAGE": "The code language is not supported",
    },
)


# ===== OPERATIONS INTERFACE =====
OPERATIONS_INTERFACE = AgentInterface(
    agent_name="Operations",
    agent_type="OPERATIONS",
    version="1.0.0",
    capabilities=[
        Capability(
            name="documentation_generation",
            type=CapabilityType.DOCUMENTATION,
            description="Generate technical documentation, README, and API docs",
            input_schema={
                "type": "object",
                "properties": {
                    "artifacts": {"type": "array"},
                    "doc_type": {"type": "string", "enum": ["readme", "api", "guide", "changelog"]},
                },
                "required": ["artifacts"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "document": {"type": "string"},
                    "format": {"type": "string"},
                },
            },
        ),
        Capability(
            name="creative_writing",
            type=CapabilityType.SYNTHESIS,
            description="Produce creative or narrative content",
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "style": {"type": "string"},
                    "length": {"type": "integer"},
                },
                "required": ["prompt"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                },
            },
        ),
        Capability(
            name="cost_analysis",
            type=CapabilityType.OPTIMIZATION,
            description="Analyse and estimate costs for a proposed solution",
            input_schema={
                "type": "object",
                "properties": {
                    "plan": {"type": "object"},
                    "budget_constraints": {"type": "object"},
                },
                "required": ["plan"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "estimated_cost": {"type": "number"},
                    "cost_breakdown": {"type": "object"},
                    "optimisation_suggestions": {"type": "array"},
                },
            },
        ),
        Capability(
            name="experiment_management",
            type=CapabilityType.ANALYSIS,
            description="Design and track experiments and A/B tests",
            input_schema={
                "type": "object",
                "properties": {
                    "hypothesis": {"type": "string"},
                    "metrics": {"type": "array"},
                },
                "required": ["hypothesis"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "experiment_plan": {"type": "object"},
                    "success_criteria": {"type": "array"},
                },
            },
        ),
        Capability(
            name="error_recovery",
            type=CapabilityType.ANALYSIS,
            description="Diagnose failures and propose recovery strategies",
            input_schema={
                "type": "object",
                "properties": {
                    "error": {"type": "string"},
                    "context": {"type": "object"},
                },
                "required": ["error"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "root_cause": {"type": "string"},
                    "recovery_steps": {"type": "array"},
                },
            },
        ),
        Capability(
            name="synthesis",
            type=CapabilityType.SYNTHESIS,
            description="Synthesise findings from multiple agents into a coherent output",
            input_schema={
                "type": "object",
                "properties": {
                    "inputs": {"type": "array"},
                    "synthesis_goal": {"type": "string"},
                },
                "required": ["inputs"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "synthesis": {"type": "string"},
                    "key_points": {"type": "array"},
                },
            },
        ),
        Capability(
            name="improvement_suggestions",
            type=CapabilityType.OPTIMIZATION,
            description="Identify and propose improvements to code, processes, or architecture",
            input_schema={
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "metrics": {"type": "array"},
                },
                "required": ["target"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "suggestions": {"type": "array"},
                    "priority_order": {"type": "array"},
                },
            },
        ),
        Capability(
            name="monitoring",
            type=CapabilityType.ANALYSIS,
            description="Set up and interpret monitoring, alerts, and observability",
            input_schema={
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "slo_targets": {"type": "object"},
                },
                "required": ["service"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "dashboard_config": {"type": "object"},
                    "alert_rules": {"type": "array"},
                },
            },
        ),
        Capability(
            name="reporting",
            type=CapabilityType.DOCUMENTATION,
            description="Generate progress reports and executive summaries",
            input_schema={
                "type": "object",
                "properties": {
                    "data": {"type": "object"},
                    "report_type": {"type": "string"},
                    "audience": {"type": "string"},
                },
                "required": ["data"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "report": {"type": "string"},
                    "highlights": {"type": "array"},
                },
            },
        ),
    ],
    required_context=["project_context"],
    error_codes={
        "NO_INPUT": "No input data provided",
        "SYNTHESIS_CONFLICT": "Conflicting inputs cannot be synthesised without disambiguation",
    },
)


# Interface registry
AGENT_INTERFACES: dict[str, AgentInterface] = {
    "EXPLORER": EXPLORER_INTERFACE,
    "LIBRARIAN": LIBRARIAN_INTERFACE,
    "RESEARCHER": RESEARCHER_INTERFACE,
    "BUILDER": BUILDER_INTERFACE,
    "EVALUATOR": EVALUATOR_INTERFACE,
    "UI_PLANNER": UI_PLANNER_INTERFACE,
    # Consolidated agents
    "PLANNER": PLANNER_INTERFACE,
    "CONSOLIDATED_RESEARCHER": CONSOLIDATED_RESEARCHER_INTERFACE,
    "CONSOLIDATED_ORACLE": CONSOLIDATED_ORACLE_INTERFACE,
    "QUALITY": QUALITY_INTERFACE,
    "OPERATIONS": OPERATIONS_INTERFACE,
}


def get_agent_interface(agent_type: str) -> AgentInterface | None:
    """Get interface contract for an agent type."""
    return AGENT_INTERFACES.get(agent_type)
