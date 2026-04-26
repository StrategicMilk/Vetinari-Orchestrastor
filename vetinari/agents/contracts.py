"""Vetinari Agent Contracts.

This module defines the canonical data contracts for Vetinari's hierarchical
multi-agent orchestration system. All agents and the Planner use these contracts.

Version: v0.1.0
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from vetinari.constants import (
    AGENT_QUALITY_GATE_STRICT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_QUALITY_THRESHOLD,
    QUALITY_GATE_HIGH,
    SANDBOX_TIMEOUT,
)
from vetinari.exceptions import InsufficientEvidenceError
from vetinari.types import (  # canonical source
    AgentType,
    ArtifactKind,
    EvidenceBasis,
    InferenceStatus,
    StatusEnum,
)
from vetinari.utils.serialization import dataclass_to_dict

# -- Evidence and outcome-signal contracts (SESSION-05 / SHARD-01) ----------


@dataclass(frozen=True, slots=True)
class Provenance:
    """Origin metadata attached to every OutcomeSignal.

    Records where and how a signal was produced so consumers can trace the
    claim back to its source without replaying the full execution history.

    Attributes:
        source: Human-readable name of the producing component or agent.
        timestamp_utc: ISO-8601 UTC timestamp of when the signal was created.
        model_id: Identifier of the LLM that produced the judgment, or empty
            string when the basis is not LLM_JUDGMENT.
        tool_name: Name of the tool that produced the evidence, or empty
            string when the basis is not TOOL_EVIDENCE.
        tool_version: Version string of the tool, or empty string when not
            applicable.
        attested_by: Identity of the human who attested, or empty string when
            the basis is not HUMAN_ATTESTED.
    """

    source: str
    timestamp_utc: str
    model_id: str = ""
    tool_name: str = ""
    tool_version: str = ""
    attested_by: str = ""

    def __repr__(self) -> str:
        return (
            f"Provenance(source={self.source!r}, timestamp_utc={self.timestamp_utc!r},"
            f" tool_name={self.tool_name!r}, model_id={self.model_id!r})"
        )


@dataclass(frozen=True, slots=True)
class ToolEvidence:
    """A single deterministic tool result backing an OutcomeSignal.

    Captures the essential fields needed to reproduce and verify a tool
    invocation: what was run, what it returned, and whether it succeeded.

    Attributes:
        tool_name: Name of the tool that was invoked (e.g. "pytest", "ruff").
        command: The full command string that was executed.
        exit_code: Process exit code; 0 means success for most tools.
        stdout_snippet: First 2 KB of stdout, for human inspection.
        stdout_hash: SHA-256 hex digest of the full stdout, for integrity.
        passed: True when the tool reported a passing result.
    """

    tool_name: str
    command: str
    exit_code: int
    stdout_snippet: str = ""
    stdout_hash: str = ""
    passed: bool = False

    def __repr__(self) -> str:
        return f"ToolEvidence(tool_name={self.tool_name!r}, exit_code={self.exit_code!r}, passed={self.passed!r})"


@dataclass(frozen=True, slots=True)
class LLMJudgment:
    """A model-generated judgment backing an OutcomeSignal.

    Captures the essential fields needed to understand and audit a
    model-produced assessment: which model, what it said, and how confident.

    Attributes:
        model_id: Identifier of the model that produced the judgment.
        summary: Short (<=500 chars) plain-English summary of the judgment.
        score: Model's self-reported confidence in [0.0, 1.0].
        reasoning: Full chain-of-thought or explanation, may be long.
    """

    model_id: str
    summary: str
    score: float = 0.0
    reasoning: str = ""

    def __repr__(self) -> str:
        return f"LLMJudgment(model_id={self.model_id!r}, score={self.score!r}, summary={self.summary[:60]!r})"


@dataclass(frozen=True, slots=True)
class AttestedArtifact:
    """A concrete artifact that a human attested to as part of claim substantiation.

    Bare "a user said yes" is NOT an AttestedArtifact — it has no kind to
    attach to.  An AttestedArtifact requires a specific ArtifactKind and a
    kind-specific payload dict that can be independently verified.

    The ``payload`` dict keys depend on ``kind``:
    - COMMAND_INVOCATION: {"command": str, "stdout_hash": str, "exit_code": int}
    - COMMIT_SHA:         {"repo": str, "sha": str, "signed": bool}
    - SIGNED_REVIEW:      {"reviewer_id": str, "signature": str, "reviewed_at": str}
    - ADR_REFERENCE:      {"adr_id": str, "status": str}
    - EXTERNAL_RECEIPT:   {"issuer": str, "receipt_id": str, "url": str}

    Attributes:
        kind: Classification of what the artifact is.
        attested_by: Identity of the human providing the attestation.
        attested_at_utc: ISO-8601 UTC timestamp of the attestation.
        payload: Kind-specific structured data dict (see above).
    """

    kind: ArtifactKind
    attested_by: str
    attested_at_utc: str
    payload: dict[str, Any]

    def __repr__(self) -> str:
        return (
            f"AttestedArtifact(kind={self.kind.value!r},"
            f" attested_by={self.attested_by!r},"
            f" attested_at_utc={self.attested_at_utc!r})"
        )


@dataclass(frozen=True, slots=True)
class OutcomeSignal:
    """Evidence-backed verdict on whether an agent output meets its quality contract.

    This is the fail-closed replacement for bare boolean pass/fail returns.
    Every OutcomeSignal carries its evidence basis so consumers can decide
    whether the evidence is sufficient for the use case (promotion audit,
    release proof, human-approval gate).

    Default constructor produces ``passed=False``, ``score=0.0``,
    ``basis=EvidenceBasis.UNSUPPORTED`` — the fail-closed sentinel (Rule 2).

    The ``use_case`` field selects the enforcement regime:
    - ``None`` (default): full enforcement — HUMAN_ATTESTED signals on factual
      claim paths must carry non-empty ``attested_artifacts``.
    - ``"INTENT_CONFIRMATION"``: relaxed — bare human attestation is acceptable
      for destructive-op consent and override appeals.

    Attributes:
        passed: Whether the output clears the quality contract.
        score: Continuous quality score in [0.0, 1.0].
        basis: What kind of evidence backs this signal.
        tool_evidence: Zero or more deterministic tool results.
        llm_judgment: Optional model-generated judgment.
        attested_artifacts: Zero or more human-attested concrete artifacts.
        provenance: Origin metadata (source, timestamp, model/tool/human id).
        issues: Identified problems with the output.
        suggestions: Recommended improvements.
        use_case: Optional use-case label; set to ``"INTENT_CONFIRMATION"``
            to permit bare human attestation for consent / override paths.

    Raises:
        InsufficientEvidenceError: At construction time, if ``basis`` is
            ``HUMAN_ATTESTED``, ``use_case`` is not ``"INTENT_CONFIRMATION"``,
            and ``attested_artifacts`` is empty.  This enforces the narrowed
            HUMAN_ATTESTED semantics: bare attestation cannot close a factual
            claim.
    """

    passed: bool = False
    score: float = 0.0
    basis: EvidenceBasis = EvidenceBasis.UNSUPPORTED
    tool_evidence: tuple[ToolEvidence, ...] = field(default_factory=tuple)
    llm_judgment: LLMJudgment | None = None
    attested_artifacts: tuple[AttestedArtifact, ...] = field(default_factory=tuple)
    provenance: Provenance | None = None
    issues: tuple[str, ...] = field(default_factory=tuple)
    suggestions: tuple[str, ...] = field(default_factory=tuple)
    use_case: Literal["INTENT_CONFIRMATION"] | None = None

    def __post_init__(self) -> None:
        """Enforce the narrowed HUMAN_ATTESTED invariant at construction time.

        Raises:
            InsufficientEvidenceError: When basis is HUMAN_ATTESTED, use_case
                is not ``"INTENT_CONFIRMATION"``, and attested_artifacts is empty.
                Bare human attestation ("a user said yes") is NOT sufficient
                to close a high-accuracy factual claim.  Pass
                ``use_case="INTENT_CONFIRMATION"`` for consent / override appeal
                paths where no artifact is required.
        """
        if (
            self.basis is EvidenceBasis.HUMAN_ATTESTED
            and self.use_case != "INTENT_CONFIRMATION"
            and not self.attested_artifacts
        ):
            raise InsufficientEvidenceError(
                "HUMAN_ATTESTED OutcomeSignal requires at least one AttestedArtifact "
                "on non-intent-confirmation paths. Provide an AttestedArtifact (command "
                "invocation, commit SHA, signed review, ADR reference, or external "
                "receipt) or set use_case='INTENT_CONFIRMATION' for consent / override "
                "appeal paths.",
                basis=self.basis.value,
                use_case=self.use_case,
            )

    def __repr__(self) -> str:
        return (
            f"OutcomeSignal(passed={self.passed!r}, score={self.score!r},"
            f" basis={self.basis.value!r},"
            f" tool_evidence={len(self.tool_evidence)},"
            f" attested_artifacts={len(self.attested_artifacts)})"
        )


@dataclass(frozen=True)
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
    jurisdiction: list[str] = field(default_factory=list)
    modes: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    can_delegate_to: list[str] = field(default_factory=list)
    max_delegation_depth: int = 3
    quality_gate_score: float = DEFAULT_QUALITY_THRESHOLD
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout_seconds: int = SANDBOX_TIMEOUT
    # --- Budget enforcement fields (ADR-0075) ---
    token_budget: int = 32_000  # Maximum tokens per invocation
    iteration_cap: int = 10  # Maximum retry/iteration loops
    cost_budget_usd: float = 1.0  # Maximum cost in USD per invocation
    delegation_budget: int = 5  # Maximum recursive delegation depth
    org_level: int = 0  # Organizational hierarchy level (0=top)
    parent_agent_id: str = ""  # ID of the spawning agent, if any
    scope_id: str = ""  # Scope identifier for budget grouping
    # --- Agent instance tracking fields (plan item 7.1) ---
    agent_instance_id: str = ""  # UUID assigned at registration time; empty until registered
    children_ids: list[str] = field(default_factory=list)  # Instance IDs of spawned child agents

    def __repr__(self) -> str:
        return (
            f"AgentSpec(agent_type={self.agent_type.value!r}, name={self.name!r}, "
            f"model={self.default_model!r}, enabled={self.enabled!r}, version={self.version!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the agent specification to a plain dictionary.

        Enum fields are converted to their string values for JSON compatibility.

        Returns:
            Dictionary representation of this AgentSpec with all fields.
        """
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentSpec:
        """Reconstruct an AgentSpec from a dictionary representation.

        Missing keys fall back to dataclass defaults, allowing forward
        compatibility with older serialized specs.

        Args:
            data: Dictionary containing agent specification fields.
                Must include 'agent_type', 'name', 'description', and
                'default_model'.

        Returns:
            A new AgentSpec instance populated from the dictionary.
        """
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
            quality_gate_score=data.get("quality_gate_score", DEFAULT_QUALITY_THRESHOLD),
            max_tokens=data.get("max_tokens", DEFAULT_MAX_TOKENS),
            timeout_seconds=data.get("timeout_seconds", SANDBOX_TIMEOUT),
            token_budget=data.get("token_budget", 32_000),
            iteration_cap=data.get("iteration_cap", 10),
            cost_budget_usd=data.get("cost_budget_usd", 1.0),
            delegation_budget=data.get("delegation_budget", 5),
            org_level=data.get("org_level", 0),
            parent_agent_id=data.get("parent_agent_id", ""),
            scope_id=data.get("scope_id", ""),
            agent_instance_id=data.get("agent_instance_id", ""),
            children_ids=data.get("children_ids", []),
        )


@dataclass
class Task:
    """A task in the plan."""

    id: str
    description: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    assigned_agent: AgentType = AgentType.FOREMAN
    model_override: str = ""
    depth: int = 0
    parent_id: str = ""
    status: StatusEnum = StatusEnum.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Task(id={self.id!r}, status={self.status.value!r}, "
            f"agent={self.assigned_agent.value!r}, depth={self.depth!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the task to a plain dictionary.

        Enum fields (assigned_agent, status) are converted to their string
        values for JSON-safe output.

        Returns:
            Dictionary representation of this Task with all fields.
        """
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """Reconstruct a Task from a dictionary representation.

        Handles enum coercion for assigned_agent and status fields, with
        sensible defaults for any missing optional keys.

        Args:
            data: Dictionary containing task fields. Must include 'id'
                and 'description' at minimum.

        Returns:
            A new Task instance populated from the dictionary.
        """
        return cls(
            id=data["id"],
            description=data["description"],
            inputs=data.get("inputs", []),
            outputs=data.get("outputs", []),
            dependencies=data.get("dependencies", []),
            assigned_agent=AgentType(data.get("assigned_agent", AgentType.FOREMAN.value)),
            model_override=data.get("model_override", ""),
            depth=data.get("depth", 0),
            parent_id=data.get("parent_id", ""),
            status=StatusEnum(data.get("status", StatusEnum.PENDING.value)),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AgentTask:
    """A task assigned to a specific agent for execution."""

    task_id: str
    agent_type: AgentType
    description: str
    prompt: str
    mode: str = ""  # Execution mode hint (e.g. "research", "build", "review")
    status: StatusEnum = StatusEnum.PENDING
    result: Any = None
    error: str = ""
    started_at: str = ""
    completed_at: str = ""
    dependencies: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    context_budget_tokens: int = 4096  # Max tokens for task context block

    def __repr__(self) -> str:
        return (
            f"AgentTask(task_id={self.task_id!r}, agent_type={self.agent_type.value!r}, "
            f"status={self.status.value!r}, error={self.error!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the agent task to a plain dictionary.

        Enum fields (agent_type, status) are converted to their string
        values for JSON-safe output.

        Returns:
            Dictionary representation of this AgentTask with all fields.
        """
        return dataclass_to_dict(self)

    @classmethod
    def from_task(cls, task: Task, prompt: str) -> AgentTask:
        """Create an AgentTask from a Task."""
        return cls(
            task_id=task.id,
            agent_type=task.assigned_agent,
            description=task.description,
            prompt=prompt,
            dependencies=task.dependencies,
        )


@dataclass
class ExecutionPlan:
    """An execution-ready plan produced by the Planner for TwoLayerOrchestrator.

    This is the agent-contracts representation of a plan — it carries tasks
    as ``Task`` objects ready for agent dispatch, plus metadata for tracking
    execution progress (phase, results, completion timestamp).

    For the planning-domain Plan type (richer, includes Subtasks, risk levels,
    and definition-of-done), use ``vetinari.planning.plan_types.Plan``.
    """

    plan_id: str
    version: str = "v0.1.0"
    goal: str = ""
    phase: int = 0
    tasks: list[Task] = field(default_factory=list)
    model_scores: list[dict] = field(default_factory=list)
    notes: str = ""
    warnings: list[str] = field(default_factory=list)
    needs_context: bool = False
    follow_up_question: str = ""
    final_delivery_path: str = ""
    final_delivery_summary: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # Populated after execution with per-task results (task_id -> summary dict)
    results: dict[str, Any] = field(default_factory=dict)
    completed_at: str = ""
    original_plan: Any = None  # Reference to the planning-domain Plan if available

    def __repr__(self) -> str:
        return (
            f"ExecutionPlan(plan_id={self.plan_id!r}, phase={self.phase!r}, "
            f"tasks={len(self.tasks)}, needs_context={self.needs_context!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the plan to a plain dictionary.

        Nested Task objects are recursively serialized; enum and datetime
        fields are converted to JSON-safe values.

        Returns:
            Dictionary representation of this ExecutionPlan, including serialized tasks.
        """
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionPlan:
        """Reconstruct an ExecutionPlan from a dictionary representation.

        Nested task dictionaries are deserialized via Task.from_dict.
        Missing optional keys fall back to dataclass defaults.

        Args:
            data: Dictionary containing plan fields. Must include
                'plan_id' at minimum.

        Returns:
            A new ExecutionPlan instance with deserialized Task objects.
        """
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
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )

    @classmethod
    def create_new(cls, goal: str, phase: int = 0) -> ExecutionPlan:
        """Create a new execution plan with a unique ID."""
        return cls(plan_id=f"plan_{uuid.uuid4().hex[:8]}", goal=goal, phase=phase)


# Backward-compatible alias — callers may import Plan from contracts
Plan = ExecutionPlan


@dataclass
class AgentResult:
    """Result from an agent's execution.

    Enhanced with task tracking, status, issue reporting, and metric
    fields to support budget accounting and quality dashboards.
    """

    success: bool
    output: str | dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    provenance: list[dict] = field(default_factory=list)
    # -- Enhanced fields (session 2A) --
    task_id: str = ""  # Task this result corresponds to
    status: InferenceStatus = InferenceStatus.SUCCESS  # Outcome classification
    issues: list[dict[str, Any]] = field(default_factory=list)  # Quality issues
    metrics: dict[str, Any] = field(default_factory=dict)  # tokens, latency, cost
    output_type: str = ""  # Semantic type: "code", "plan", "report", etc.

    def __repr__(self) -> str:
        return (
            f"AgentResult(success={self.success!r}, task_id={self.task_id!r}, "
            f"status={self.status.value!r}, errors={self.errors!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the agent result to a plain dictionary.

        Returns:
            Dictionary containing success status, output, metadata,
            errors, and provenance chain.
        """
        return dataclass_to_dict(self)


@dataclass
class VerificationResult:
    """Result from verification of an agent's output."""

    passed: bool
    issues: list[dict[str, Any]] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    score: float = 0.0

    def __repr__(self) -> str:
        return f"VerificationResult(passed={self.passed!r}, score={self.score!r}, issues={len(self.issues)})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the verification result to a plain dictionary.

        Returns:
            Dictionary containing pass/fail status, issues found,
            improvement suggestions, and the overall quality score.
        """
        return dataclass_to_dict(self)


# ── Active agents (v0.5.0 — 3-agent factory pipeline) ────────────────
# Foreman orchestrates; Worker executes; Inspector gates quality.
ACTIVE_AGENT_TYPES: frozenset[AgentType] = frozenset({
    AgentType.FOREMAN,
    AgentType.WORKER,
    AgentType.INSPECTOR,
})

# Registry of the 3 active factory-pipeline agents
AGENT_REGISTRY: dict[AgentType, AgentSpec] = {
    AgentType.FOREMAN: AgentSpec(
        agent_type=AgentType.FOREMAN,
        name="Foreman",
        description=("Planning, goal decomposition, Worker assignment, user interaction, context management"),
        default_model="qwen2.5-72b",
        thinking_variant="xhigh",
        modes=["plan", "clarify", "consolidate", "summarise", "prune", "extract"],
        jurisdiction=[
            "vetinari/agents/planner_agent.py",
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
        can_delegate_to=[AgentType.WORKER.value, AgentType.INSPECTOR.value],
        max_delegation_depth=5,
        quality_gate_score=AGENT_QUALITY_GATE_STRICT,
        max_tokens=8192,  # noqa: VET129 — agent spec config, not inference param
        timeout_seconds=600,
    ),
    AgentType.WORKER: AgentSpec(
        agent_type=AgentType.WORKER,
        name="Worker",
        description=(
            "Unified execution agent — research, architecture, build, and operations across 24 modes in 4 groups"
        ),
        default_model="qwen2.5-72b",
        thinking_variant="high",
        modes=[
            # Research group (8)
            "code_discovery",
            "domain_research",
            "api_lookup",
            "lateral_thinking",
            "ui_design",
            "database",
            "devops",
            "git_workflow",
            # Architecture group (5)
            "architecture",
            "risk_assessment",
            "ontological_analysis",
            "contrarian_review",
            "suggest",
            # Build group (2)
            "build",
            "image_generation",
            # Operations group (9)
            "documentation",
            "creative_writing",
            "cost_analysis",
            "experiment",
            "error_recovery",
            "synthesis",
            "improvement",
            "monitor",
            "devops_ops",
        ],
        jurisdiction=[
            "vetinari/agents/builder_agent.py",
            "vetinari/agents/consolidated/",
            "vetinari/research/",
            "vetinari/architecture/",
            "vetinari/templates/",
            "docs/",
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
            "architecture_decision_support",
            "risk_and_tradeoff_analysis",
            "ontological_analysis",
            "contrarian_review",
            "code_scaffolding",
            "image_generation",
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
        can_delegate_to=[AgentType.INSPECTOR.value],
        max_delegation_depth=3,
        quality_gate_score=QUALITY_GATE_HIGH,
        max_tokens=32768,  # noqa: VET129 — agent spec config, not inference param
        timeout_seconds=600,
    ),
    # Decision: Inspector uses 14B+ model for security_audit (CWE classification,
    # trust boundary tracing) — 7B is insufficient for deep semantic analysis.
    # Code review and test generation also benefit from stronger reasoning.
    AgentType.INSPECTOR: AgentSpec(
        agent_type=AgentType.INSPECTOR,
        name="Inspector",
        description=(
            "Independent quality gate — code review, security audit, "
            "test generation, simplification. Gate decisions are authoritative."
        ),
        default_model="qwen3-30b-a3b",
        thinking_variant="high",
        modes=[
            "code_review",
            "security_audit",
            "test_generation",
            "simplification",
        ],
        jurisdiction=[
            "vetinari/agents/consolidated/quality_agent.py",
            "tests/",
        ],
        capabilities=[
            "code_review",
            "security_audit",
            "test_generation",
            "code_simplification",
        ],
        can_delegate_to=[AgentType.WORKER.value],
        max_delegation_depth=2,
        quality_gate_score=AGENT_QUALITY_GATE_STRICT,
        max_tokens=4096,  # noqa: VET129 — agent spec config, not inference param
        timeout_seconds=300,
    ),
}


def get_agent_spec(agent_type: AgentType) -> AgentSpec | None:
    """Get the specification for an agent type, or None if not registered."""
    return AGENT_REGISTRY.get(agent_type)


def get_all_agent_specs() -> list[AgentSpec]:
    return list(AGENT_REGISTRY.values())


def get_enabled_agents() -> list[AgentSpec]:
    return [spec for spec in AGENT_REGISTRY.values() if spec.enabled]
