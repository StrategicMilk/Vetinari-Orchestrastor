"""Architecture Constraints.

=========================
Defines delegation rules, mode-to-agent validation, and task-type-to-agent
validation.  These constraints enforce the hierarchical delegator-specialist
pattern that research (Google, Cognizant, Microsoft) confirms as optimal.

Key rules:
- PLANNER can delegate to all agents, max depth 5.
- No agent may delegate to PLANNER (prevents circular orchestration).
- QUALITY has special permission to request re-execution from BUILDER
  (maker-checker pattern, max 3 iterations).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ArchitectureConstraint:
    """Delegation and mode constraints for a single agent type."""

    agent_type: str  # AgentType.value
    can_delegate_to: list[str] = field(default_factory=list)
    cannot_delegate_to: list[str] = field(default_factory=list)
    max_delegation_depth: int = 3
    allowed_modes: list[str] = field(default_factory=list)
    allowed_task_types: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Default architecture constraints (maps to current 6 consolidated agents)
# ---------------------------------------------------------------------------

ARCHITECTURE_CONSTRAINTS: dict[str, ArchitectureConstraint] = {
    # Core orchestration
    "PLANNER": ArchitectureConstraint(
        agent_type="PLANNER",
        can_delegate_to=[
            "PLANNER",
            "CONSOLIDATED_RESEARCHER",
            "CONSOLIDATED_ORACLE",
            "BUILDER",
            "QUALITY",
            "OPERATIONS",
            "ORCHESTRATOR",
            "ARCHITECT",
            "RESEARCHER",
            "PONDER",
        ],
        cannot_delegate_to=[],
        max_delegation_depth=5,
        allowed_modes=["decompose", "schedule", "specification"],
        allowed_task_types=["planning", "analysis", "general"],
    ),
    # Building
    "BUILDER": ArchitectureConstraint(
        agent_type="BUILDER",
        can_delegate_to=["CONSOLIDATED_RESEARCHER"],
        cannot_delegate_to=["PLANNER"],
        max_delegation_depth=2,
        allowed_modes=["implement", "scaffold", "refactor"],
        allowed_task_types=["implementation", "coding", "testing"],
    ),
    # Quality (maker-checker: can request re-execution from BUILDER)
    "QUALITY": ArchitectureConstraint(
        agent_type="QUALITY",
        can_delegate_to=["BUILDER"],
        cannot_delegate_to=["PLANNER"],
        max_delegation_depth=1,
        allowed_modes=["code_review", "test_generation", "security_audit", "simplification"],
        allowed_task_types=["verification", "code_review", "testing", "security"],
    ),
    # --- Consolidated agents ---
    "ORCHESTRATOR": ArchitectureConstraint(
        agent_type="ORCHESTRATOR",
        can_delegate_to=[
            "CONSOLIDATED_RESEARCHER",
            "CONSOLIDATED_ORACLE",
            "BUILDER",
            "ARCHITECT",
            "QUALITY",
            "OPERATIONS",
        ],
        cannot_delegate_to=[],
        max_delegation_depth=4,
        allowed_modes=["clarify", "consolidate", "summarise", "prune", "extract", "monitor"],
        allowed_task_types=["clarification", "memory", "monitoring", "general"],
    ),
    "CONSOLIDATED_RESEARCHER": ArchitectureConstraint(
        agent_type="CONSOLIDATED_RESEARCHER",
        can_delegate_to=[],
        cannot_delegate_to=["PLANNER", "ORCHESTRATOR"],
        max_delegation_depth=1,
        allowed_modes=[
            "code_discovery",
            "domain_research",
            "api_lookup",
            "lateral_thinking",
            "ui_design",
            "data_engineering",
        ],
        allowed_task_types=["research", "analysis", "ui", "design", "data"],
    ),
    "CONSOLIDATED_ORACLE": ArchitectureConstraint(
        agent_type="CONSOLIDATED_ORACLE",
        can_delegate_to=[],
        cannot_delegate_to=["PLANNER", "ORCHESTRATOR"],
        max_delegation_depth=1,
        allowed_modes=["architecture", "risk_assessment", "ontological_analysis", "contrarian_review"],
        allowed_task_types=["analysis", "architecture"],
    ),
    "ARCHITECT": ArchitectureConstraint(
        agent_type="ARCHITECT",
        can_delegate_to=["BUILDER"],
        cannot_delegate_to=["PLANNER", "ORCHESTRATOR"],
        max_delegation_depth=2,
        allowed_modes=["ui_design", "database", "devops", "git_workflow"],
        allowed_task_types=["design", "infrastructure", "architecture"],
    ),
    # Note: QUALITY already defined above with maker-checker pattern
    "OPERATIONS": ArchitectureConstraint(
        agent_type="OPERATIONS",
        can_delegate_to=["BUILDER"],
        cannot_delegate_to=["PLANNER", "ORCHESTRATOR"],
        max_delegation_depth=1,
        allowed_modes=[
            "documentation",
            "creative_writing",
            "cost_analysis",
            "experiment",
            "error_recovery",
            "synthesis",
            "image_generation",
            "improvement",
        ],
        allowed_task_types=["documentation", "creative", "operations", "error_recovery", "cost_analysis"],
    ),
}


# ---------------------------------------------------------------------------
# Loop prevention and cost-aware delegation constraints
# ---------------------------------------------------------------------------

MAX_AGENT_RETRIES_PER_TASK = 3  # Prevent infinite retry loops
MAX_DELEGATION_CHAIN_LENGTH = 5  # Max depth already enforced per-agent above
PREFER_LOCAL_FOR_SIMPLE_TASKS = True  # Route simple tasks to local models
CLOUD_ESCALATION_THRESHOLD = 0.7  # Escalate to cloud when local quality < 0.7


def validate_delegation(from_agent: str, to_agent: str, current_depth: int = 0) -> tuple[bool, str]:
    """Check if delegation from one agent to another is allowed.

    Returns ``(allowed, reason)`` where *reason* explains the denial.

    Args:
        from_agent: The from agent.
        to_agent: The to agent.
        current_depth: The current depth.

    Returns:
        True if successful, False otherwise.
    """
    constraint = ARCHITECTURE_CONSTRAINTS.get(from_agent)
    if constraint is None:
        # Unknown agents are allowed by default (no constraint defined)
        return True, "no constraint defined"

    if to_agent in constraint.cannot_delegate_to:
        return False, f"{from_agent} is explicitly forbidden from delegating to {to_agent}"

    if constraint.can_delegate_to and to_agent not in constraint.can_delegate_to:
        return False, f"{to_agent} is not in {from_agent}'s allowed delegation list"

    if current_depth >= constraint.max_delegation_depth:
        return False, f"delegation depth {current_depth} exceeds max {constraint.max_delegation_depth} for {from_agent}"

    return True, "allowed"


def get_constraint(agent_type: str) -> ArchitectureConstraint | None:
    """Get the architecture constraint for an agent type."""
    return ARCHITECTURE_CONSTRAINTS.get(agent_type)
