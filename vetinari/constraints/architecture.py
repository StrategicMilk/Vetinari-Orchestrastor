"""
Architecture Constraints
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
from typing import Dict, List, Optional, Tuple


@dataclass
class ArchitectureConstraint:
    """Delegation and mode constraints for a single agent type."""

    agent_type: str                                  # AgentType.value
    can_delegate_to: List[str] = field(default_factory=list)
    cannot_delegate_to: List[str] = field(default_factory=list)
    max_delegation_depth: int = 3
    allowed_modes: List[str] = field(default_factory=list)
    allowed_task_types: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Default architecture constraints (maps to current 22 agents AND future 8)
# ---------------------------------------------------------------------------

ARCHITECTURE_CONSTRAINTS: Dict[str, ArchitectureConstraint] = {
    # Core orchestration
    "PLANNER": ArchitectureConstraint(
        agent_type="PLANNER",
        can_delegate_to=[
            # Legacy agents
            "EXPLORER", "ORACLE", "LIBRARIAN", "RESEARCHER",
            "EVALUATOR", "SYNTHESIZER", "BUILDER", "UI_PLANNER",
            "SECURITY_AUDITOR", "DATA_ENGINEER", "DOCUMENTATION_AGENT",
            "COST_PLANNER", "TEST_AUTOMATION", "DEVOPS", "IMAGE_GENERATOR",
            # Consolidated agents (Phase 3)
            "ORCHESTRATOR", "CONSOLIDATED_RESEARCHER", "CONSOLIDATED_ORACLE",
            "ARCHITECT", "QUALITY", "OPERATIONS",
        ],
        cannot_delegate_to=[],
        max_delegation_depth=5,
        allowed_modes=["decompose", "schedule", "specification"],
        allowed_task_types=["planning", "analysis", "general"],
    ),
    # Building
    "BUILDER": ArchitectureConstraint(
        agent_type="BUILDER",
        can_delegate_to=["RESEARCHER", "EXPLORER"],
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
    "EVALUATOR": ArchitectureConstraint(
        agent_type="EVALUATOR",
        can_delegate_to=["BUILDER"],
        cannot_delegate_to=["PLANNER"],
        max_delegation_depth=1,
        allowed_modes=["code_review", "test_generation", "security_audit"],
        allowed_task_types=["verification", "code_review", "testing"],
    ),
    "SECURITY_AUDITOR": ArchitectureConstraint(
        agent_type="SECURITY_AUDITOR",
        can_delegate_to=[],
        cannot_delegate_to=["PLANNER"],
        max_delegation_depth=1,
        allowed_modes=["security_audit"],
        allowed_task_types=["security", "verification"],
    ),
    # Research
    "RESEARCHER": ArchitectureConstraint(
        agent_type="RESEARCHER",
        can_delegate_to=["EXPLORER", "LIBRARIAN"],
        cannot_delegate_to=["PLANNER", "BUILDER"],
        max_delegation_depth=2,
        allowed_modes=["code_discovery", "api_lookup", "domain_research", "lateral_thinking"],
        allowed_task_types=["research", "analysis"],
    ),
    "EXPLORER": ArchitectureConstraint(
        agent_type="EXPLORER",
        can_delegate_to=[],
        cannot_delegate_to=["PLANNER"],
        max_delegation_depth=1,
        allowed_modes=["code_discovery"],
        allowed_task_types=["research", "analysis"],
    ),
    "LIBRARIAN": ArchitectureConstraint(
        agent_type="LIBRARIAN",
        can_delegate_to=[],
        cannot_delegate_to=["PLANNER"],
        max_delegation_depth=1,
        allowed_modes=["api_lookup", "domain_research"],
        allowed_task_types=["research"],
    ),
    # Oracle
    "ORACLE": ArchitectureConstraint(
        agent_type="ORACLE",
        can_delegate_to=[],
        cannot_delegate_to=["PLANNER"],
        max_delegation_depth=1,
        allowed_modes=["architecture", "risk_assessment", "ontological_analysis", "contrarian_review"],
        allowed_task_types=["analysis", "architecture"],
    ),
    # Operations / Support (leaf agents)
    "SYNTHESIZER": ArchitectureConstraint(
        agent_type="SYNTHESIZER",
        can_delegate_to=[],
        cannot_delegate_to=["PLANNER"],
        max_delegation_depth=1,
        allowed_modes=["synthesis", "creative_writing"],
        allowed_task_types=["creative", "documentation"],
    ),
    "DOCUMENTATION_AGENT": ArchitectureConstraint(
        agent_type="DOCUMENTATION_AGENT",
        can_delegate_to=[],
        cannot_delegate_to=["PLANNER"],
        max_delegation_depth=1,
        allowed_modes=["documentation"],
        allowed_task_types=["documentation"],
    ),
    "TEST_AUTOMATION": ArchitectureConstraint(
        agent_type="TEST_AUTOMATION",
        can_delegate_to=[],
        cannot_delegate_to=["PLANNER"],
        max_delegation_depth=1,
        allowed_modes=["test_generation"],
        allowed_task_types=["testing"],
    ),
    "UI_PLANNER": ArchitectureConstraint(
        agent_type="UI_PLANNER",
        can_delegate_to=["BUILDER"],
        cannot_delegate_to=["PLANNER"],
        max_delegation_depth=2,
        allowed_modes=["ui_design"],
        allowed_task_types=["ui", "design"],
    ),
    "DEVOPS": ArchitectureConstraint(
        agent_type="DEVOPS",
        can_delegate_to=[],
        cannot_delegate_to=["PLANNER"],
        max_delegation_depth=1,
        allowed_modes=["devops"],
        allowed_task_types=["devops", "deployment"],
    ),
    "DATA_ENGINEER": ArchitectureConstraint(
        agent_type="DATA_ENGINEER",
        can_delegate_to=[],
        cannot_delegate_to=["PLANNER"],
        max_delegation_depth=1,
        allowed_modes=["database"],
        allowed_task_types=["data", "database"],
    ),
    "IMAGE_GENERATOR": ArchitectureConstraint(
        agent_type="IMAGE_GENERATOR",
        can_delegate_to=[],
        cannot_delegate_to=["PLANNER"],
        max_delegation_depth=1,
        allowed_modes=["image_generation"],
        allowed_task_types=["image"],
    ),
    "COST_PLANNER": ArchitectureConstraint(
        agent_type="COST_PLANNER",
        can_delegate_to=[],
        cannot_delegate_to=["PLANNER"],
        max_delegation_depth=1,
        allowed_modes=["cost_analysis"],
        allowed_task_types=["cost_analysis"],
    ),
    "ERROR_RECOVERY": ArchitectureConstraint(
        agent_type="ERROR_RECOVERY",
        can_delegate_to=["BUILDER", "PLANNER"],
        cannot_delegate_to=[],
        max_delegation_depth=2,
        allowed_modes=["error_recovery"],
        allowed_task_types=["error_recovery", "general"],
    ),

    # --- Consolidated agents (Phase 3) ---

    "ORCHESTRATOR": ArchitectureConstraint(
        agent_type="ORCHESTRATOR",
        can_delegate_to=[
            "CONSOLIDATED_RESEARCHER", "CONSOLIDATED_ORACLE",
            "BUILDER", "ARCHITECT", "QUALITY", "OPERATIONS",
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
        allowed_modes=["code_discovery", "domain_research", "api_lookup", "lateral_thinking"],
        allowed_task_types=["research", "analysis"],
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
    # Note: QUALITY already defined above (lines 62-69) with maker-checker pattern
    "OPERATIONS": ArchitectureConstraint(
        agent_type="OPERATIONS",
        can_delegate_to=[],
        cannot_delegate_to=["PLANNER", "ORCHESTRATOR"],
        max_delegation_depth=1,
        allowed_modes=[
            "documentation", "creative_writing", "cost_analysis", "experiment",
            "error_recovery", "synthesis", "image_generation", "improvement",
        ],
        allowed_task_types=["documentation", "creative", "operations", "error_recovery", "cost_analysis"],
    ),
}


def validate_delegation(
    from_agent: str, to_agent: str, current_depth: int = 0
) -> Tuple[bool, str]:
    """Check if delegation from one agent to another is allowed.

    Returns ``(allowed, reason)`` where *reason* explains the denial.
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


def get_constraint(agent_type: str) -> Optional[ArchitectureConstraint]:
    """Get the architecture constraint for an agent type."""
    return ARCHITECTURE_CONSTRAINTS.get(agent_type)
