from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from vetinari.types import PlanStatus, SubtaskStatus  # canonical source


class TaskDomain(str, Enum):
    CODING = "coding"
    DATA_PROCESSING = "data_processing"
    INFRA = "infra"
    DOCS = "docs"
    AI_EXPERIMENTS = "ai_experiments"
    RESEARCH = "research"
    GENERAL = "general"


class PlanRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DefinitionOfDone:
    """Definition of Done for a task or subtask."""

    criteria: list[str] = field(default_factory=list)
    verification_method: str = ""
    auto_verify: bool = False


@dataclass
class DefinitionOfReady:
    """Definition of Ready for a task or subtask."""

    prerequisites: list[str] = field(default_factory=list)
    blockers_removed: bool = True
    dependencies_met: bool = True


@dataclass
class TaskRationale:
    """Rationale for task/model decisions."""

    reason: str
    capability_match: float | None = None
    context_fit: float | None = None
    cost_estimate: float | None = None
    policy_notes: str | None = None
    alternatives_considered: list[str] = field(default_factory=list)


@dataclass
class Subtask:
    """A discrete unit of work within a plan."""

    subtask_id: str = field(default_factory=lambda: f"subtask_{uuid.uuid4().hex[:8]}")
    plan_id: str = ""
    parent_subtask_id: str | None = None
    description: str = ""
    domain: TaskDomain = TaskDomain.GENERAL
    depth: int = 0
    status: SubtaskStatus = SubtaskStatus.PENDING
    assigned_model_id: str | None = None
    assigned_agent: str | None = None
    rationale: TaskRationale | None = None
    definition_of_done: DefinitionOfDone = field(default_factory=DefinitionOfDone)
    definition_of_ready: DefinitionOfReady = field(default_factory=DefinitionOfReady)
    expected_output: str = ""
    actual_output: str | None = None
    time_estimate_seconds: float = 0.0
    actual_duration_seconds: float = 0.0
    cost_estimate: float = 0.0
    actual_cost: float = 0.0
    dependencies: list[str] = field(default_factory=list)
    child_subtask_ids: list[str] = field(default_factory=list)
    subtask_explanation_json: str = ""  # JSON string of SubtaskExplanation
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        if self.rationale:
            data["rationale"] = asdict(self.rationale)
        if self.definition_of_done:
            data["definition_of_done"] = asdict(self.definition_of_done)
        if self.definition_of_ready:
            data["definition_of_ready"] = asdict(self.definition_of_ready)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Subtask:
        """Create Subtask from dictionary."""
        if "rationale" in data and isinstance(data["rationale"], dict):
            data["rationale"] = TaskRationale(**data["rationale"])
        if "definition_of_done" in data and isinstance(data["definition_of_done"], dict):
            data["definition_of_done"] = DefinitionOfDone(**data["definition_of_done"])
        if "definition_of_ready" in data and isinstance(data["definition_of_ready"], dict):
            data["definition_of_ready"] = DefinitionOfReady(**data["definition_of_ready"])

        for field_name in ["status", "domain"]:
            if field_name in data and isinstance(data[field_name], str):
                if field_name == "status":
                    data[field_name] = SubtaskStatus(data[field_name])
                elif field_name == "domain":
                    data[field_name] = TaskDomain(data[field_name])

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PlanCandidate:
    """A candidate plan option."""

    plan_id: str = field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    plan_version: int = 1
    summary: str = ""
    description: str = ""
    justification: str = ""
    risk_score: float = 0.0
    risk_level: PlanRiskLevel = PlanRiskLevel.LOW
    estimated_duration_seconds: float = 0.0
    estimated_cost: float = 0.0
    subtask_count: int = 0
    max_depth: int = 0
    domains: list[TaskDomain] = field(default_factory=list)
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        if self.risk_level:
            data["risk_level"] = self.risk_level.value
        if self.domains:
            data["domains"] = [d.value for d in self.domains]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanCandidate:
        """Create PlanCandidate from dictionary."""
        if "risk_level" in data and isinstance(data["risk_level"], str):
            data["risk_level"] = PlanRiskLevel(data["risk_level"])
        if "domains" in data and isinstance(data["domains"], list):
            data["domains"] = [TaskDomain(d) if isinstance(d, str) else d for d in data["domains"]]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Plan:
    """A complete plan with tasks and metadata."""

    plan_id: str = field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    plan_version: int = 1
    goal: str = ""
    constraints: str = ""
    status: PlanStatus = PlanStatus.DRAFT
    plan_candidates: list[PlanCandidate] = field(default_factory=list)
    chosen_plan_id: str | None = None
    plan_justification: str = ""
    risk_score: float = 0.0
    risk_level: PlanRiskLevel = PlanRiskLevel.LOW
    subtasks: list[Subtask] = field(default_factory=list)
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    definition_of_done: DefinitionOfDone = field(default_factory=DefinitionOfDone)
    dry_run: bool = False
    auto_approved: bool = False
    approved_by: str | None = None
    approved_at: str | None = None
    plan_explanation_json: str = ""  # JSON string of PlanExplanation
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        if self.status:
            data["status"] = self.status.value
        if self.risk_level:
            data["risk_level"] = self.risk_level.value
        if self.plan_candidates:
            data["plan_candidates"] = [c.to_dict() for c in self.plan_candidates]
        if self.subtasks:
            data["subtasks"] = [s.to_dict() for s in self.subtasks]
        if self.definition_of_done:
            data["definition_of_done"] = asdict(self.definition_of_done)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Plan:
        """Create Plan from dictionary."""
        if "status" in data and isinstance(data["status"], str):
            data["status"] = PlanStatus(data["status"])
        if "risk_level" in data and isinstance(data["risk_level"], str):
            data["risk_level"] = PlanRiskLevel(data["risk_level"])
        if "plan_candidates" in data and isinstance(data["plan_candidates"], list):
            data["plan_candidates"] = [PlanCandidate.from_dict(c) for c in data["plan_candidates"]]
        if "subtasks" in data and isinstance(data["subtasks"], list):
            data["subtasks"] = [Subtask.from_dict(s) for s in data["subtasks"]]
        if "definition_of_done" in data and isinstance(data["definition_of_done"], dict):
            data["definition_of_done"] = DefinitionOfDone(**data["definition_of_done"])

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def get_subtask(self, subtask_id: str) -> Subtask | None:
        """Get a subtask by ID."""
        for subtask in self.subtasks:
            if subtask.subtask_id == subtask_id:
                return subtask
        return None

    def get_root_subtasks(self) -> list[Subtask]:
        """Get all root-level subtasks (depth=0)."""
        return [s for s in self.subtasks if s.depth == 0]

    def get_subtasks_by_depth(self, depth: int) -> list[Subtask]:
        """Get all subtasks at a specific depth."""
        return [s for s in self.subtasks if s.depth == depth]

    def get_subtasks_by_domain(self, domain: TaskDomain) -> list[Subtask]:
        """Get all subtasks in a specific domain."""
        return [s for s in self.subtasks if s.domain == domain]

    def calculate_risk_score(self) -> float:
        """Calculate overall risk score based on subtasks."""
        if not self.subtasks:
            return 0.0

        risk_factors = []

        risk_factors.append(min(len(self.subtasks) / 50.0, 1.0))

        max_depth = max((s.depth for s in self.subtasks), default=0)
        risk_factors.append(min(max_depth / 16.0, 1.0))

        total_cost = sum(s.cost_estimate for s in self.subtasks)
        risk_factors.append(min(total_cost / 100.0, 1.0))

        complex_deps = sum(1 for deps in self.dependencies.values() if len(deps) > 3)
        risk_factors.append(min(complex_deps / 10.0, 1.0))

        self.risk_score = sum(risk_factors) / len(risk_factors)

        if self.risk_score >= 0.75:
            self.risk_level = PlanRiskLevel.CRITICAL
        elif self.risk_score >= 0.5:
            self.risk_level = PlanRiskLevel.HIGH
        elif self.risk_score >= 0.25:
            self.risk_level = PlanRiskLevel.MEDIUM
        else:
            self.risk_level = PlanRiskLevel.LOW

        return self.risk_score


@dataclass
class PlanGenerationRequest:
    """Request to generate a plan."""

    goal: str
    constraints: str = ""
    plan_depth_cap: int = 16
    max_candidates: int = 3
    domain_hint: TaskDomain | None = None
    dry_run: bool = False
    risk_threshold: float = 0.25


@dataclass
class PlanApprovalRequest:
    """Request to approve a plan.

    JSON Schema:
    {
        "plan_id": "string (required)",
        "approved": "boolean (required)",
        "approver": "string (required)",
        "reason": "string (optional)",
        "audit_id": "string (optional, auto-generated if not provided)",
        "risk_score": "float (optional)",
        "timestamp": "string (optional, auto-generated)",
        "approval_schema_version": "int (default: 1)"
    }
    """

    plan_id: str
    approved: bool
    approver: str = "system"
    reason: str = ""
    audit_id: str | None = None
    risk_score: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    approval_schema_version: int = 1


@dataclass
class PlanHistoryResponse:
    """Response containing plan history."""

    plan_id: str
    goal: str
    status: PlanStatus
    risk_score: float
    risk_level: PlanRiskLevel
    subtask_count: int
    created_at: str
    updated_at: str
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.status:
            data["status"] = self.status.value
        if self.risk_level:
            data["risk_level"] = self.risk_level.value
        return data
