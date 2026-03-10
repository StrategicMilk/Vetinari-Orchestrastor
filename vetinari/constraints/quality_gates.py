"""
Quality Gates
==============
Defines per-agent quality thresholds that MUST be met before output is
accepted.  Currently ``VerificationResult.score`` exists but is never used
to block execution — this module adds enforcement.

Enforcement point: ``base_agent.complete_task()``
1. Run ``verify()`` to get ``VerificationResult``
2. Check ``result.score >= gate.min_verification_score``
3. If FAIL and retries remaining: re-execute with feedback injected
4. If FAIL and no retries: log violation, return with ``quality_gate_failed=True``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class QualityGate:
    """Quality threshold for a specific agent (optionally per-mode)."""

    agent_type: str                        # AgentType.value
    mode: Optional[str] = None             # If None, applies to all modes
    min_verification_score: float = 0.5
    min_heuristic_score: float = 0.3
    max_retry_on_failure: int = 2
    require_passing_verification: bool = True
    require_schema_validation: bool = True


# ---------------------------------------------------------------------------
# Default quality gates per agent
# ---------------------------------------------------------------------------

QUALITY_GATES: Dict[str, QualityGate] = {
    # Building — higher bar, must pass verification
    "BUILDER": QualityGate(
        agent_type="BUILDER",
        min_verification_score=0.6,
        require_passing_verification=True,
        max_retry_on_failure=2,
    ),
    # Quality agents — highest bar
    "EVALUATOR": QualityGate(
        agent_type="EVALUATOR",
        min_verification_score=0.7,
        max_retry_on_failure=1,
    ),
    "SECURITY_AUDITOR": QualityGate(
        agent_type="SECURITY_AUDITOR",
        min_verification_score=0.7,
        max_retry_on_failure=1,
    ),
    "TEST_AUTOMATION": QualityGate(
        agent_type="TEST_AUTOMATION",
        min_verification_score=0.6,
        max_retry_on_failure=2,
    ),
    # Research — moderate bar
    "RESEARCHER": QualityGate(
        agent_type="RESEARCHER",
        min_verification_score=0.5,
        max_retry_on_failure=2,
    ),
    "EXPLORER": QualityGate(
        agent_type="EXPLORER",
        min_verification_score=0.4,
        max_retry_on_failure=2,
    ),
    "LIBRARIAN": QualityGate(
        agent_type="LIBRARIAN",
        min_verification_score=0.5,
        max_retry_on_failure=2,
    ),
    # Oracle — high bar for architecture decisions
    "ORACLE": QualityGate(
        agent_type="ORACLE",
        min_verification_score=0.6,
        max_retry_on_failure=1,
    ),
    # Operations — moderate bar, more lenient for creative work
    "SYNTHESIZER": QualityGate(
        agent_type="SYNTHESIZER",
        min_verification_score=0.5,
        max_retry_on_failure=2,
    ),
    "DOCUMENTATION_AGENT": QualityGate(
        agent_type="DOCUMENTATION_AGENT",
        min_verification_score=0.5,
        max_retry_on_failure=2,
    ),
    # Planning — moderate bar
    "PLANNER": QualityGate(
        agent_type="PLANNER",
        min_verification_score=0.5,
        max_retry_on_failure=2,
    ),
    # Infrastructure agents
    "UI_PLANNER": QualityGate(
        agent_type="UI_PLANNER",
        min_verification_score=0.5,
        max_retry_on_failure=2,
    ),
    "DEVOPS": QualityGate(
        agent_type="DEVOPS",
        min_verification_score=0.5,
        max_retry_on_failure=2,
    ),
    "DATA_ENGINEER": QualityGate(
        agent_type="DATA_ENGINEER",
        min_verification_score=0.5,
        max_retry_on_failure=2,
    ),

    # --- Consolidated agents (Phase 3) ---

    "ORCHESTRATOR": QualityGate(
        agent_type="ORCHESTRATOR",
        min_verification_score=0.5,
        max_retry_on_failure=2,
    ),
    "CONSOLIDATED_RESEARCHER": QualityGate(
        agent_type="CONSOLIDATED_RESEARCHER",
        min_verification_score=0.5,
        max_retry_on_failure=2,
    ),
    "CONSOLIDATED_ORACLE": QualityGate(
        agent_type="CONSOLIDATED_ORACLE",
        min_verification_score=0.6,
        max_retry_on_failure=1,
    ),
    "ARCHITECT": QualityGate(
        agent_type="ARCHITECT",
        min_verification_score=0.5,
        max_retry_on_failure=2,
    ),
    "QUALITY": QualityGate(
        agent_type="QUALITY",
        min_verification_score=0.7,
        max_retry_on_failure=1,
    ),
    # Mode-specific gates for QUALITY agent
    "QUALITY:security_audit": QualityGate(
        agent_type="QUALITY",
        mode="security_audit",
        min_verification_score=0.7,
        max_retry_on_failure=1,
        require_schema_validation=True,
    ),
    "QUALITY:code_review": QualityGate(
        agent_type="QUALITY",
        mode="code_review",
        min_verification_score=0.6,
        max_retry_on_failure=2,
    ),
    "OPERATIONS": QualityGate(
        agent_type="OPERATIONS",
        min_verification_score=0.5,
        max_retry_on_failure=2,
    ),
    # Mode-specific gates for OPERATIONS agent
    "OPERATIONS:creative_writing": QualityGate(
        agent_type="OPERATIONS",
        mode="creative_writing",
        min_verification_score=0.4,
        max_retry_on_failure=2,
    ),
}

# Default gate for any agent not explicitly listed
_DEFAULT_GATE = QualityGate(agent_type="DEFAULT", min_verification_score=0.4)


# ---------------------------------------------------------------------------
# Criticality-based adaptive quality thresholds
# ---------------------------------------------------------------------------

QUALITY_GATES_BY_CRITICALITY: Dict[str, Dict] = {
    "critical": {"min_score": 0.85, "require_human_review": True, "max_retries": 1},
    "high":     {"min_score": 0.75, "require_human_review": False, "max_retries": 2},
    "medium":   {"min_score": 0.60, "require_human_review": False, "max_retries": 2},
    "low":      {"min_score": 0.40, "require_human_review": False, "max_retries": 3},
}


def get_criticality_gate(criticality: str) -> Dict:
    """Get quality gate thresholds for a given task criticality level."""
    return QUALITY_GATES_BY_CRITICALITY.get(
        criticality.lower(), QUALITY_GATES_BY_CRITICALITY["medium"]
    )


def get_quality_gate(agent_type: str, mode: Optional[str] = None) -> QualityGate:
    """Get the quality gate for an agent (optionally for a specific mode).

    Mode-specific gates take priority over agent-level gates.
    """
    # Check for mode-specific gate first
    if mode:
        mode_key = f"{agent_type}:{mode}"
        if mode_key in QUALITY_GATES:
            return QUALITY_GATES[mode_key]
    return QUALITY_GATES.get(agent_type, _DEFAULT_GATE)


def check_quality_gate(
    agent_type: str, score: float, mode: Optional[str] = None
) -> Tuple[bool, str]:
    """Check if an output score passes the quality gate.

    Returns ``(passed, reason)``.
    """
    gate = get_quality_gate(agent_type, mode)
    if score >= gate.min_verification_score:
        return True, f"score {score:.2f} >= threshold {gate.min_verification_score:.2f}"
    return False, (
        f"score {score:.2f} below threshold {gate.min_verification_score:.2f} "
        f"for {agent_type}"
    )
