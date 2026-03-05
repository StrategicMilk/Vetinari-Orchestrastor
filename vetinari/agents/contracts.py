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
from enum import Enum
from typing import Any, Dict, List, Optional

__all__ = [
    "AgentType",
    "TaskStatus",
    "ExecutionMode",
    "AgentSpec",
    "Task",
    "AgentTask",
    "Plan",
    "AgentResult",
    "VerificationResult",
    "get_agent_spec",
    "get_all_agent_specs",
    "get_enabled_agents",
]


class AgentType(Enum):
    """Enumeration of all Vetinari agents."""
    PLANNER = "PLANNER"
    EXPLORER = "EXPLORER"
    LIBRARIAN = "LIBRARIAN"
    ORACLE = "ORACLE"
    RESEARCHER = "RESEARCHER"
    EVALUATOR = "EVALUATOR"
    SYNTHESIZER = "SYNTHESIZER"
    BUILDER = "BUILDER"
    UI_PLANNER = "UI_PLANNER"
    SECURITY_AUDITOR = "SECURITY_AUDITOR"
    DATA_ENGINEER = "DATA_ENGINEER"
    DOCUMENTATION_AGENT = "DOCUMENTATION_AGENT"
    COST_PLANNER = "COST_PLANNER"
    TEST_AUTOMATION = "TEST_AUTOMATION"
    EXPERIMENTATION_MANAGER = "EXPERIMENTATION_MANAGER"
    IMPROVEMENT = "IMPROVEMENT"
    USER_INTERACTION = "USER_INTERACTION"
    DEVOPS = "DEVOPS"
    VERSION_CONTROL = "VERSION_CONTROL"
    ERROR_RECOVERY = "ERROR_RECOVERY"
    CONTEXT_MANAGER = "CONTEXT_MANAGER"
    IMAGE_GENERATOR = "IMAGE_GENERATOR"


class TaskStatus(Enum):
    """Status of a task in the orchestration."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"
    BLOCKED = "blocked"


class ExecutionMode(Enum):
    """Execution mode for agents."""
    PLANNING = "planning"
    EXECUTION = "execution"


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_type": self.agent_type.value,
            "name": self.name,
            "description": self.description,
            "default_model": self.default_model,
            "thinking_variant": self.thinking_variant,
            "enabled": self.enabled,
            "system_prompt": self.system_prompt,
            "version": self.version
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
            version=data.get("version", "1.0.0")
        )


@dataclass
class Task:
    """A task in the plan.

    This is the single canonical Task type used across planning, orchestration,
    and agent execution.  Fields added for planning_engine / planning
    compatibility all carry defaults so existing call-sites are unaffected.
    """
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

    # --- Fields from planning.py -------------------------------------------
    prompt: str = ""
    wave_id: str = ""
    priority: int = 5
    estimated_effort: float = 1.0
    retry_count: int = 0
    result: Any = None
    error: str = ""

    # --- Fields from planning_engine.py ------------------------------------
    assigned_model_id: str = ""
    children: List[str] = field(default_factory=list)
    owner_id: str = ""

    # --- Additional planning.py scheduling / decomposition fields ----------
    planned_start: str = ""
    planned_end: str = ""
    actual_start: str = ""
    actual_end: str = ""
    max_depth: int = 14
    max_depth_override: int = 0
    subtasks: List["Task"] = field(default_factory=list)
    decomposition_seed: str = ""
    dod_level: str = "Standard"
    dor_level: str = "Standard"

    # Backward-compat alias used by planning.py (``task.task_id``)
    @property
    def task_id(self) -> str:
        """Alias kept for backward-compatibility with planning.py callers."""
        return self.id

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
            "status": self.status.value,
            "prompt": self.prompt,
            "wave_id": self.wave_id,
            "priority": self.priority,
            "estimated_effort": self.estimated_effort,
            "retry_count": self.retry_count,
            "result": self.result,
            "error": self.error,
            "assigned_model_id": self.assigned_model_id,
            "children": self.children,
            "owner_id": self.owner_id,
            "planned_start": self.planned_start,
            "planned_end": self.planned_end,
            "actual_start": self.actual_start,
            "actual_end": self.actual_end,
            "max_depth": self.max_depth,
            "max_depth_override": self.max_depth_override,
            "subtasks": [t.to_dict() for t in self.subtasks],
            "decomposition_seed": self.decomposition_seed,
            "dod_level": self.dod_level,
            "dor_level": self.dor_level,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Task:
        subtasks = [Task.from_dict(t) for t in data.get("subtasks", [])]
        # Accept both enum-style ("PLANNER") and lower-case string ("explorer")
        raw_agent = data.get("assigned_agent", data.get("agent_type", "PLANNER"))
        try:
            agent = AgentType(raw_agent)
        except ValueError:
            agent = AgentType(raw_agent.upper()) if raw_agent else AgentType.PLANNER
        # Accept both enum-style and lower-case status strings
        raw_status = data.get("status", "pending")
        try:
            status = TaskStatus(raw_status)
        except ValueError:
            status = TaskStatus.PENDING
        return cls(
            id=data.get("id", data.get("task_id", "")),
            description=data.get("description", ""),
            inputs=data.get("inputs", []),
            outputs=data.get("outputs", []),
            dependencies=data.get("dependencies", []),
            assigned_agent=agent,
            model_override=data.get("model_override", ""),
            depth=data.get("depth", 0),
            parent_id=data.get("parent_id", ""),
            status=status,
            prompt=data.get("prompt", ""),
            wave_id=data.get("wave_id", ""),
            priority=data.get("priority", 5),
            estimated_effort=data.get("estimated_effort", 1.0),
            retry_count=data.get("retry_count", 0),
            result=data.get("result"),
            error=data.get("error", ""),
            assigned_model_id=data.get("assigned_model_id", ""),
            children=data.get("children", []),
            owner_id=data.get("owner_id", ""),
            planned_start=data.get("planned_start", ""),
            planned_end=data.get("planned_end", ""),
            actual_start=data.get("actual_start", ""),
            actual_end=data.get("actual_end", ""),
            max_depth=data.get("max_depth", 14),
            max_depth_override=data.get("max_depth_override", 0),
            subtasks=subtasks,
            decomposition_seed=data.get("decomposition_seed", ""),
            dod_level=data.get("dod_level", "Standard"),
            dor_level=data.get("dor_level", "Standard"),
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


# Registry of all available agents
AGENT_REGISTRY: Dict[AgentType, AgentSpec] = {
    AgentType.PLANNER: AgentSpec(
        agent_type=AgentType.PLANNER,
        name="Planner",
        description="Central orchestration and dynamic plan generation from goals",
        default_model="qwen2.5-72b",
        thinking_variant="xhigh"
    ),
    AgentType.EXPLORER: AgentSpec(
        agent_type=AgentType.EXPLORER,
        name="Explorer",
        description="Fast code/document/class discovery and pattern extraction",
        default_model="qwen2.5-coder-7b",
        thinking_variant="high"
    ),
    AgentType.LIBRARIAN: AgentSpec(
        agent_type=AgentType.LIBRARIAN,
        name="Librarian",
        description="Literature/library research, API/docs lookup",
        default_model="qwen2.5-72b",
        thinking_variant="medium"
    ),
    AgentType.ORACLE: AgentSpec(
        agent_type=AgentType.ORACLE,
        name="Oracle",
        description="Architectural decisions, risk assessment, debugging strategies",
        default_model="qwen3-30b-a3b",
        thinking_variant="xhigh"
    ),
    AgentType.RESEARCHER: AgentSpec(
        agent_type=AgentType.RESEARCHER,
        name="Researcher",
        description="Domain research, feasibility analysis, competitive analysis",
        default_model="qwen2.5-72b",
        thinking_variant="medium"
    ),
    AgentType.EVALUATOR: AgentSpec(
        agent_type=AgentType.EVALUATOR,
        name="Evaluator",
        description="Code quality, security checks, testability evaluation",
        default_model="qwen2.5-coder-7b",
        thinking_variant="high"
    ),
    AgentType.SYNTHESIZER: AgentSpec(
        agent_type=AgentType.SYNTHESIZER,
        name="Synthesizer",
        description="Multi-source synthesis, artifact fusion",
        default_model="qwen2.5-72b",
        thinking_variant="high"
    ),
    AgentType.BUILDER: AgentSpec(
        agent_type=AgentType.BUILDER,
        name="Builder",
        description="Code scaffolding, boilerplate, test scaffolding",
        default_model="qwen2.5-coder-7b",
        thinking_variant="medium"
    ),
    AgentType.UI_PLANNER: AgentSpec(
        agent_type=AgentType.UI_PLANNER,
        name="UI Planner",
        description="UI/UX design, front-end patterns, scaffolding",
        default_model="qwen2.5-vl-32b",
        thinking_variant="high"
    ),
    AgentType.SECURITY_AUDITOR: AgentSpec(
        agent_type=AgentType.SECURITY_AUDITOR,
        name="Security Auditor",
        description="Safety, policy compliance, vulnerability checks",
        default_model="qwen2.5-coder-7b",
        thinking_variant="high"
    ),
    AgentType.DATA_ENGINEER: AgentSpec(
        agent_type=AgentType.DATA_ENGINEER,
        name="Data Engineer",
        description="Data pipelines, schemas, migrations, ETL",
        default_model="qwen2.5-72b",
        thinking_variant="medium"
    ),
    AgentType.DOCUMENTATION_AGENT: AgentSpec(
        agent_type=AgentType.DOCUMENTATION_AGENT,
        name="Documentation Agent",
        description="Auto-generated docs, API docs, user guides",
        default_model="qwen2.5-72b",
        thinking_variant="medium"
    ),
    AgentType.COST_PLANNER: AgentSpec(
        agent_type=AgentType.COST_PLANNER,
        name="Cost Planner",
        description="Cost accounting, model selection, efficiency optimization",
        default_model="qwen2.5-coder-7b",
        thinking_variant="medium"
    ),
    AgentType.TEST_AUTOMATION: AgentSpec(
        agent_type=AgentType.TEST_AUTOMATION,
        name="Test Automation",
        description="Test generation, coverage improvement, validation",
        default_model="qwen2.5-coder-7b",
        thinking_variant="medium"
    ),
    AgentType.EXPERIMENTATION_MANAGER: AgentSpec(
        agent_type=AgentType.EXPERIMENTATION_MANAGER,
        name="Experimentation Manager",
        description="Experiment tracking, versioning, reproducibility",
        default_model="qwen2.5-72b",
        thinking_variant="medium"
    ),
    AgentType.IMPROVEMENT: AgentSpec(
        agent_type=AgentType.IMPROVEMENT,
        name="Improvement Agent",
        description="Meta-analyst reviewing system performance, recommending optimizations",
        default_model="qwen2.5-72b",
        thinking_variant="high"
    ),
    AgentType.USER_INTERACTION: AgentSpec(
        agent_type=AgentType.USER_INTERACTION,
        name="User Interaction Agent",
        description="Ambiguity detection, clarifying question generation, context gathering",
        default_model="qwen2.5-72b",
        thinking_variant="medium"
    ),
    AgentType.DEVOPS: AgentSpec(
        agent_type=AgentType.DEVOPS,
        name="DevOps Agent",
        description="CI/CD pipeline design, containerisation, IaC, deployment, monitoring",
        default_model="qwen2.5-coder-7b",
        thinking_variant="medium"
    ),
    AgentType.VERSION_CONTROL: AgentSpec(
        agent_type=AgentType.VERSION_CONTROL,
        name="Version Control Agent",
        description="Git operations, branch strategy, PR creation, code review coordination",
        default_model="qwen2.5-coder-7b",
        thinking_variant="medium"
    ),
    AgentType.ERROR_RECOVERY: AgentSpec(
        agent_type=AgentType.ERROR_RECOVERY,
        name="Error Recovery Agent",
        description="Failure analysis, retry strategies, circuit breaking, fallback planning",
        default_model="qwen2.5-72b",
        thinking_variant="high"
    ),
    AgentType.CONTEXT_MANAGER: AgentSpec(
        agent_type=AgentType.CONTEXT_MANAGER,
        name="Context Manager Agent",
        description="Long-term context management, memory consolidation, session summarisation",
        default_model="qwen2.5-72b",
        thinking_variant="medium"
    ),
    AgentType.IMAGE_GENERATOR: AgentSpec(
        agent_type=AgentType.IMAGE_GENERATOR,
        name="Image Generator",
        description="Logo, icon, UI mockup, diagram, and asset generation via Stable Diffusion or SVG",
        default_model="qwen2.5-72b",
        thinking_variant="medium"
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
