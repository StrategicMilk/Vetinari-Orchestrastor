"""
Vetinari Agent Contracts

This module defines the canonical data contracts for Vetinari's hierarchical
multi-agent orchestration system. All agents and the Planner use these contracts.

Version: v0.1.0
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from vetinari.types import AgentType, TaskStatus, ExecutionMode  # canonical source


@dataclass
class AgentSpec:
    """Specification for an agent type."""
    agent_type: AgentType
    name: str
    description: str
    default_model: str
    thinking_variant: str = "medium"
    enabled: bool = True
    system_prompt: str = ""
    version: str = "1.0.0"
    # --- Extended fields (P5.5a) ---
    deprecated: bool = False
    replaced_by: str = ""
    jurisdiction: List[str] = field(default_factory=list)
    modes: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    can_delegate_to: List[str] = field(default_factory=list)
    max_delegation_depth: int = 3
    quality_gate_score: float = 0.7
    max_tokens: int = 4096
    timeout_seconds: int = 300

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_type": self.agent_type.value,
            "name": self.name,
            "description": self.description,
            "default_model": self.default_model,
            "thinking_variant": self.thinking_variant,
            "enabled": self.enabled,
            "system_prompt": self.system_prompt,
            "version": self.version,
            "deprecated": self.deprecated,
            "replaced_by": self.replaced_by,
            "jurisdiction": self.jurisdiction,
            "modes": self.modes,
            "capabilities": self.capabilities,
            "can_delegate_to": self.can_delegate_to,
            "max_delegation_depth": self.max_delegation_depth,
            "quality_gate_score": self.quality_gate_score,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentSpec:
        return cls(
            agent_type=AgentType(data["agent_type"]),
            name=data["name"],
            description=data["description"],
            default_model=data["default_model"],
            thinking_variant=data.get("thinking_variant", "medium"),
            enabled=data.get("enabled", True),
            system_prompt=data.get("system_prompt", ""),
            version=data.get("version", "1.0.0"),
            deprecated=data.get("deprecated", False),
            replaced_by=data.get("replaced_by", ""),
            jurisdiction=data.get("jurisdiction", []),
            modes=data.get("modes", []),
            capabilities=data.get("capabilities", []),
            can_delegate_to=data.get("can_delegate_to", []),
            max_delegation_depth=data.get("max_delegation_depth", 3),
            quality_gate_score=data.get("quality_gate_score", 0.7),
            max_tokens=data.get("max_tokens", 4096),
            timeout_seconds=data.get("timeout_seconds", 300),
        )


@dataclass
class Task:
    """A task in the plan."""
    id: str
    description: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: AgentType = AgentType.PLANNER
    model_override: str = ""
    depth: int = 0
    parent_id: str = ""
    status: TaskStatus = TaskStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "dependencies": self.dependencies,
            "assigned_agent": self.assigned_agent.value,
            "model_override": self.model_override,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "status": self.status.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Task:
        return cls(
            id=data["id"],
            description=data["description"],
            inputs=data.get("inputs", []),
            outputs=data.get("outputs", []),
            dependencies=data.get("dependencies", []),
            assigned_agent=AgentType(data.get("assigned_agent", "PLANNER")),
            model_override=data.get("model_override", ""),
            depth=data.get("depth", 0),
            parent_id=data.get("parent_id", ""),
            status=TaskStatus(data.get("status", "pending"))
        )


@dataclass
class AgentTask:
    """A task assigned to a specific agent for execution."""
    task_id: str
    agent_type: AgentType
    description: str
    prompt: str
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = ""
    started_at: str = ""
    completed_at: str = ""
    dependencies: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_type": self.agent_type.value,
            "description": self.description,
            "prompt": self.prompt,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "dependencies": self.dependencies,
            "context": self.context
        }

    @classmethod
    def from_task(cls, task: Task, prompt: str) -> AgentTask:
        """Create an AgentTask from a Task."""
        return cls(
            task_id=task.id,
            agent_type=task.assigned_agent,
            description=task.description,
            prompt=prompt,
            dependencies=task.dependencies
        )


@dataclass
class Plan:
    """A complete plan generated by the Planner."""
    plan_id: str
    version: str = "v0.1.0"
    goal: str = ""
    phase: int = 0
    tasks: List[Task] = field(default_factory=list)
    model_scores: List[Dict] = field(default_factory=list)
    notes: str = ""
    warnings: List[str] = field(default_factory=list)
    needs_context: bool = False
    follow_up_question: str = ""
    final_delivery_path: str = ""
    final_delivery_summary: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "version": self.version,
            "goal": self.goal,
            "phase": self.phase,
            "tasks": [t.to_dict() for t in self.tasks],
            "model_scores": self.model_scores,
            "notes": self.notes,
            "warnings": self.warnings,
            "needs_context": self.needs_context,
            "follow_up_question": self.follow_up_question,
            "final_delivery_path": self.final_delivery_path,
            "final_delivery_summary": self.final_delivery_summary,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Plan:
        return cls(
            plan_id=data["plan_id"],
            version=data.get("version", "v0.1.0"),
            goal=data.get("goal", ""),
            phase=data.get("phase", 0),
            tasks=[Task.from_dict(t) for t in data.get("tasks", [])],
            model_scores=data.get("model_scores", []),
            notes=data.get("notes", ""),
            warnings=data.get("warnings", []),
            needs_context=data.get("needs_context", False),
            follow_up_question=data.get("follow_up_question", ""),
            final_delivery_path=data.get("final_delivery_path", ""),
            final_delivery_summary=data.get("final_delivery_summary", ""),
            created_at=data.get("created_at", datetime.now().isoformat())
        )

    @classmethod
    def create_new(cls, goal: str, phase: int = 0) -> Plan:
        """Create a new plan with a unique ID."""
        return cls(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            goal=goal,
            phase=phase
        )


@dataclass
class AgentResult:
    """Result from an agent's execution."""
    success: bool
    output: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    provenance: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "metadata": self.metadata,
            "errors": self.errors,
            "provenance": self.provenance
        }


@dataclass
class VerificationResult:
    """Result from verification of an agent's output."""
    passed: bool
    issues: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "score": self.score
        }


# ── Active agents (v0.4.0 — 6 consolidated) ─────────────────────────
# The 6 active agents that form the production pipeline.
ACTIVE_AGENT_TYPES = {
    AgentType.PLANNER,
    AgentType.BUILDER,
    AgentType.CONSOLIDATED_RESEARCHER,
    AgentType.CONSOLIDATED_ORACLE,
    AgentType.QUALITY,
    AgentType.OPERATIONS,
}

# Maps legacy AgentType values to their consolidated replacement.
AGENT_TYPE_MAPPING: Dict[AgentType, AgentType] = {
    AgentType.EXPLORER: AgentType.CONSOLIDATED_RESEARCHER,
    AgentType.LIBRARIAN: AgentType.CONSOLIDATED_RESEARCHER,
    AgentType.RESEARCHER: AgentType.CONSOLIDATED_RESEARCHER,
    AgentType.UI_PLANNER: AgentType.CONSOLIDATED_RESEARCHER,
    AgentType.DATA_ENGINEER: AgentType.CONSOLIDATED_RESEARCHER,
    AgentType.DEVOPS: AgentType.CONSOLIDATED_RESEARCHER,
    AgentType.VERSION_CONTROL: AgentType.CONSOLIDATED_RESEARCHER,
    AgentType.ARCHITECT: AgentType.CONSOLIDATED_RESEARCHER,
    AgentType.ORACLE: AgentType.CONSOLIDATED_ORACLE,
    AgentType.PONDER: AgentType.CONSOLIDATED_ORACLE,
    AgentType.EVALUATOR: AgentType.QUALITY,
    AgentType.SECURITY_AUDITOR: AgentType.QUALITY,
    AgentType.TEST_AUTOMATION: AgentType.QUALITY,
    AgentType.SYNTHESIZER: AgentType.OPERATIONS,
    AgentType.DOCUMENTATION_AGENT: AgentType.OPERATIONS,
    AgentType.COST_PLANNER: AgentType.OPERATIONS,
    AgentType.EXPERIMENTATION_MANAGER: AgentType.OPERATIONS,
    AgentType.IMPROVEMENT: AgentType.OPERATIONS,
    AgentType.ERROR_RECOVERY: AgentType.OPERATIONS,
    AgentType.USER_INTERACTION: AgentType.PLANNER,
    AgentType.CONTEXT_MANAGER: AgentType.PLANNER,
    AgentType.ORCHESTRATOR: AgentType.PLANNER,
    AgentType.IMAGE_GENERATOR: AgentType.BUILDER,
}


def resolve_agent_type(agent_type: AgentType) -> AgentType:
    """Resolve a legacy agent type to its consolidated equivalent.

    Returns the input unchanged if it is already a consolidated type.
    """
    return AGENT_TYPE_MAPPING.get(agent_type, agent_type)


# Registry of all available agents
AGENT_REGISTRY: Dict[AgentType, AgentSpec] = {
    # ── 6 active consolidated agents ──
    AgentType.PLANNER: AgentSpec(
        agent_type=AgentType.PLANNER,
        name="Planner",
        description="Planning, goal decomposition, user interaction, context management",
        default_model="qwen2.5-72b",
        thinking_variant="xhigh",
        modes=["plan", "clarify", "consolidate", "summarise", "prune", "extract"],
        jurisdiction=[
            "vetinari/agents/planner.py",
            "vetinari/agents/contracts.py",
            "vetinari/core/",
        ],
        capabilities=[
            "goal_decomposition",
            "task_sequencing",
            "context_management",
            "user_clarification",
            "plan_consolidation",
            "dependency_resolution",
        ],
        can_delegate_to=[
            "CONSOLIDATED_RESEARCHER",
            "CONSOLIDATED_ORACLE",
            "BUILDER",
            "QUALITY",
            "OPERATIONS",
        ],
        max_delegation_depth=5,
        quality_gate_score=0.8,
        max_tokens=8192,
        timeout_seconds=600,
    ),
    AgentType.BUILDER: AgentSpec(
        agent_type=AgentType.BUILDER,
        name="Builder",
        description="Code scaffolding, boilerplate, test scaffolding, image generation",
        default_model="qwen2.5-coder-7b",
        thinking_variant="medium",
        modes=["scaffold", "boilerplate", "test_scaffold", "image_generation"],
        jurisdiction=[
            "vetinari/agents/builder.py",
            "vetinari/templates/",
        ],
        capabilities=[
            "code_scaffolding",
            "image_generation",
        ],
        can_delegate_to=["QUALITY"],
        max_delegation_depth=2,
        quality_gate_score=0.7,
        max_tokens=4096,
        timeout_seconds=300,
    ),
    AgentType.CONSOLIDATED_RESEARCHER: AgentSpec(
        agent_type=AgentType.CONSOLIDATED_RESEARCHER,
        name="Researcher",
        description="Code discovery, domain research, API lookup, lateral thinking, "
                    "UI/UX design, database schemas, DevOps pipelines, git workflow",
        default_model="qwen2.5-72b",
        thinking_variant="high",
        modes=[
            "code_discovery",
            "domain_research",
            "api_lookup",
            "lateral_thinking",
            "ui_design",
            "database",
            "devops",
            "git_workflow",
        ],
        jurisdiction=[
            "vetinari/agents/researcher.py",
            "vetinari/agents/explorer.py",
            "vetinari/agents/librarian.py",
            "vetinari/research/",
        ],
        capabilities=[
            "code_pattern_search",
            "domain_analysis",
            "api_documentation_lookup",
            "lateral_thinking",
            "ui_ux_design",
            "database_schema_design",
            "devops_pipeline_design",
            "git_workflow_analysis",
        ],
        can_delegate_to=["CONSOLIDATED_ORACLE", "BUILDER"],
        max_delegation_depth=3,
        quality_gate_score=0.75,
        max_tokens=8192,
        timeout_seconds=480,
    ),
    AgentType.CONSOLIDATED_ORACLE: AgentSpec(
        agent_type=AgentType.CONSOLIDATED_ORACLE,
        name="Oracle",
        description="Architecture decisions, risk assessment, ontological analysis, contrarian review",
        default_model="qwen3-30b-a3b",
        thinking_variant="xhigh",
        modes=[
            "architecture_decision",
            "risk_assessment",
            "ontological_analysis",
            "contrarian_review",
        ],
        jurisdiction=[
            "vetinari/agents/oracle.py",
            "vetinari/architecture/",
        ],
        capabilities=[
            "architecture_decision_support",
            "risk_and_tradeoff_analysis",
            "ontological_analysis",
            "contrarian_review",
        ],
        can_delegate_to=["CONSOLIDATED_RESEARCHER"],
        max_delegation_depth=2,
        quality_gate_score=0.85,
        max_tokens=8192,
        timeout_seconds=480,
    ),
    AgentType.QUALITY: AgentSpec(
        agent_type=AgentType.QUALITY,
        name="Quality",
        description="Code review, security audit, test generation, simplification",
        default_model="qwen2.5-coder-7b",
        thinking_variant="high",
        modes=["code_review", "security_audit", "test_generation", "simplification"],
        jurisdiction=[
            "vetinari/agents/quality.py",
            "vetinari/agents/evaluator.py",
            "vetinari/agents/security_auditor.py",
            "tests/",
        ],
        capabilities=[
            "code_review",
            "security_audit",
            "test_generation",
            "code_simplification",
        ],
        can_delegate_to=["CONSOLIDATED_RESEARCHER"],
        max_delegation_depth=2,
        quality_gate_score=0.8,
        max_tokens=4096,
        timeout_seconds=300,
    ),
    AgentType.OPERATIONS: AgentSpec(
        agent_type=AgentType.OPERATIONS,
        name="Operations",
        description="Documentation, creative writing, cost analysis, experiments, "
                    "error recovery, synthesis, improvement, monitoring",
        default_model="qwen2.5-72b",
        thinking_variant="medium",
        modes=[
            "documentation",
            "creative_writing",
            "cost_analysis",
            "experiment",
            "error_recovery",
            "synthesis",
            "improvement",
            "monitoring",
            "reporting",
        ],
        jurisdiction=[
            "vetinari/agents/operations.py",
            "vetinari/agents/synthesizer.py",
            "vetinari/agents/documentation_agent.py",
            "vetinari/agents/cost_planner.py",
            "vetinari/agents/error_recovery.py",
            "docs/",
        ],
        capabilities=[
            "documentation_generation",
            "creative_writing",
            "cost_analysis",
            "experiment_management",
            "error_recovery",
            "synthesis",
            "improvement_suggestions",
            "monitoring",
            "reporting",
        ],
        can_delegate_to=["CONSOLIDATED_RESEARCHER", "QUALITY"],
        max_delegation_depth=3,
        quality_gate_score=0.7,
        max_tokens=8192,
        timeout_seconds=480,
    ),
    # ── Legacy entries (kept for registry lookups, resolve via AGENT_TYPE_MAPPING) ──
    AgentType.EXPLORER: AgentSpec(
        agent_type=AgentType.EXPLORER, name="Explorer",
        description="[Legacy] -> CONSOLIDATED_RESEARCHER.code_discovery",
        default_model="qwen2.5-coder-7b", enabled=False,
        deprecated=True, replaced_by="CONSOLIDATED_RESEARCHER",
    ),
    AgentType.LIBRARIAN: AgentSpec(
        agent_type=AgentType.LIBRARIAN, name="Librarian",
        description="[Legacy] -> CONSOLIDATED_RESEARCHER.api_lookup",
        default_model="qwen2.5-72b", enabled=False,
        deprecated=True, replaced_by="CONSOLIDATED_RESEARCHER",
    ),
    AgentType.ORACLE: AgentSpec(
        agent_type=AgentType.ORACLE, name="Oracle (legacy)",
        description="[Legacy] -> CONSOLIDATED_ORACLE",
        default_model="qwen3-30b-a3b", enabled=False,
        deprecated=True, replaced_by="CONSOLIDATED_ORACLE",
    ),
    AgentType.RESEARCHER: AgentSpec(
        agent_type=AgentType.RESEARCHER, name="Researcher (legacy)",
        description="[Legacy] -> CONSOLIDATED_RESEARCHER.domain_research",
        default_model="qwen2.5-72b", enabled=False,
        deprecated=True, replaced_by="CONSOLIDATED_RESEARCHER",
    ),
    AgentType.EVALUATOR: AgentSpec(
        agent_type=AgentType.EVALUATOR, name="Evaluator",
        description="[Legacy] -> QUALITY.code_review",
        default_model="qwen2.5-coder-7b", enabled=False,
        deprecated=True, replaced_by="QUALITY",
    ),
    AgentType.SYNTHESIZER: AgentSpec(
        agent_type=AgentType.SYNTHESIZER, name="Synthesizer",
        description="[Legacy] -> OPERATIONS.synthesis",
        default_model="qwen2.5-72b", enabled=False,
        deprecated=True, replaced_by="OPERATIONS",
    ),
    AgentType.UI_PLANNER: AgentSpec(
        agent_type=AgentType.UI_PLANNER, name="UI Planner",
        description="[Legacy] -> CONSOLIDATED_RESEARCHER.ui_design",
        default_model="qwen2.5-vl-32b", enabled=False,
        deprecated=True, replaced_by="CONSOLIDATED_RESEARCHER",
    ),
    AgentType.SECURITY_AUDITOR: AgentSpec(
        agent_type=AgentType.SECURITY_AUDITOR, name="Security Auditor",
        description="[Legacy] -> QUALITY.security_audit",
        default_model="qwen2.5-coder-7b", enabled=False,
        deprecated=True, replaced_by="QUALITY",
    ),
    AgentType.DATA_ENGINEER: AgentSpec(
        agent_type=AgentType.DATA_ENGINEER, name="Data Engineer",
        description="[Legacy] -> CONSOLIDATED_RESEARCHER.database",
        default_model="qwen2.5-72b", enabled=False,
        deprecated=True, replaced_by="CONSOLIDATED_RESEARCHER",
    ),
    AgentType.DOCUMENTATION_AGENT: AgentSpec(
        agent_type=AgentType.DOCUMENTATION_AGENT, name="Documentation Agent",
        description="[Legacy] -> OPERATIONS.documentation",
        default_model="qwen2.5-72b", enabled=False,
        deprecated=True, replaced_by="OPERATIONS",
    ),
    AgentType.COST_PLANNER: AgentSpec(
        agent_type=AgentType.COST_PLANNER, name="Cost Planner",
        description="[Legacy] -> OPERATIONS.cost_analysis",
        default_model="qwen2.5-coder-7b", enabled=False,
        deprecated=True, replaced_by="OPERATIONS",
    ),
    AgentType.TEST_AUTOMATION: AgentSpec(
        agent_type=AgentType.TEST_AUTOMATION, name="Test Automation",
        description="[Legacy] -> QUALITY.test_generation",
        default_model="qwen2.5-coder-7b", enabled=False,
        deprecated=True, replaced_by="QUALITY",
    ),
    AgentType.EXPERIMENTATION_MANAGER: AgentSpec(
        agent_type=AgentType.EXPERIMENTATION_MANAGER, name="Experimentation Manager",
        description="[Legacy] -> OPERATIONS.experiment",
        default_model="qwen2.5-72b", enabled=False,
        deprecated=True, replaced_by="OPERATIONS",
    ),
    AgentType.IMPROVEMENT: AgentSpec(
        agent_type=AgentType.IMPROVEMENT, name="Improvement Agent",
        description="[Legacy] -> OPERATIONS.improvement",
        default_model="qwen2.5-72b", enabled=False,
        deprecated=True, replaced_by="OPERATIONS",
    ),
    AgentType.USER_INTERACTION: AgentSpec(
        agent_type=AgentType.USER_INTERACTION, name="User Interaction Agent",
        description="[Legacy] -> PLANNER.clarify",
        default_model="qwen2.5-72b", enabled=False,
        deprecated=True, replaced_by="PLANNER",
    ),
    AgentType.DEVOPS: AgentSpec(
        agent_type=AgentType.DEVOPS, name="DevOps Agent",
        description="[Legacy] -> CONSOLIDATED_RESEARCHER.devops",
        default_model="qwen2.5-coder-7b", enabled=False,
        deprecated=True, replaced_by="CONSOLIDATED_RESEARCHER",
    ),
    AgentType.VERSION_CONTROL: AgentSpec(
        agent_type=AgentType.VERSION_CONTROL, name="Version Control Agent",
        description="[Legacy] -> CONSOLIDATED_RESEARCHER.git_workflow",
        default_model="qwen2.5-coder-7b", enabled=False,
        deprecated=True, replaced_by="CONSOLIDATED_RESEARCHER",
    ),
    AgentType.ERROR_RECOVERY: AgentSpec(
        agent_type=AgentType.ERROR_RECOVERY, name="Error Recovery Agent",
        description="[Legacy] -> OPERATIONS.error_recovery",
        default_model="qwen2.5-72b", enabled=False,
        deprecated=True, replaced_by="OPERATIONS",
    ),
    AgentType.CONTEXT_MANAGER: AgentSpec(
        agent_type=AgentType.CONTEXT_MANAGER, name="Context Manager Agent",
        description="[Legacy] -> PLANNER.consolidate",
        default_model="qwen2.5-72b", enabled=False,
        deprecated=True, replaced_by="PLANNER",
    ),
    AgentType.IMAGE_GENERATOR: AgentSpec(
        agent_type=AgentType.IMAGE_GENERATOR, name="Image Generator",
        description="[Legacy] -> BUILDER.image_generation",
        default_model="qwen2.5-72b", enabled=False,
        deprecated=True, replaced_by="BUILDER",
    ),
    AgentType.PONDER: AgentSpec(
        agent_type=AgentType.PONDER, name="Ponder",
        description="[Legacy] -> CONSOLIDATED_ORACLE.ontological_analysis",
        default_model="qwen2.5-72b", enabled=False,
        deprecated=True, replaced_by="CONSOLIDATED_ORACLE",
    ),
    AgentType.ORCHESTRATOR: AgentSpec(
        agent_type=AgentType.ORCHESTRATOR, name="Orchestrator",
        description="[Legacy] -> PLANNER (clarify, consolidate, summarise, prune, extract)",
        default_model="qwen2.5-72b", enabled=False,
        deprecated=True, replaced_by="PLANNER",
    ),
    AgentType.ARCHITECT: AgentSpec(
        agent_type=AgentType.ARCHITECT, name="Architect",
        description="[Legacy] -> CONSOLIDATED_RESEARCHER (ui_design, database, devops, git_workflow)",
        default_model="qwen2.5-72b", enabled=False,
        deprecated=True, replaced_by="CONSOLIDATED_RESEARCHER",
    ),
}


def get_agent_spec(agent_type: AgentType) -> AgentSpec:
    """Get the specification for an agent type."""
    return AGENT_REGISTRY.get(agent_type)


def get_all_agent_specs() -> List[AgentSpec]:
    """Get all agent specifications."""
    return list(AGENT_REGISTRY.values())


def get_enabled_agents() -> List[AgentSpec]:
    """Get all enabled agent specifications."""
    return [spec for spec in AGENT_REGISTRY.values() if spec.enabled]
