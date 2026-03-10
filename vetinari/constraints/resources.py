"""
Per-Agent Resource Constraints
================================
Defines token limits, retry budgets, timeouts, cost caps, and parallelism
limits for each agent type.

Enforcement points:
- ``base_agent.prepare_task()``: inject resource constraints into task context
- ``base_agent._infer()``: check token limit before LLM call
- ``DurableExecutionEngine``: check timeout + cost budget per task node
- ``TwoLayerOrchestrator``: check delegation call count per agent
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ResourceConstraint:
    """Per-agent resource limits."""

    agent_type: str                  # AgentType.value
    max_tokens: int = 4096
    max_retries: int = 3
    timeout_seconds: int = 120
    max_cost_usd: float = 0.50
    max_delegation_calls: int = 5
    max_parallel_tasks: int = 3


# ---------------------------------------------------------------------------
# Default resource limits per agent
# ---------------------------------------------------------------------------

AGENT_RESOURCE_LIMITS: Dict[str, ResourceConstraint] = {
    # Core orchestration — needs larger context for decomposition
    "PLANNER": ResourceConstraint(
        agent_type="PLANNER",
        max_tokens=8192,
        timeout_seconds=180,
        max_delegation_calls=10,
        max_parallel_tasks=1,
    ),
    # Building — moderate context, limited retries
    "BUILDER": ResourceConstraint(
        agent_type="BUILDER",
        max_tokens=4096,
        max_retries=2,
        timeout_seconds=120,
        max_delegation_calls=3,
    ),
    # Quality — needs to review code, moderate context
    "EVALUATOR": ResourceConstraint(
        agent_type="EVALUATOR",
        max_tokens=4096,
        timeout_seconds=60,
        max_retries=2,
        max_delegation_calls=2,
    ),
    "SECURITY_AUDITOR": ResourceConstraint(
        agent_type="SECURITY_AUDITOR",
        max_tokens=4096,
        timeout_seconds=90,
        max_retries=2,
        max_delegation_calls=1,
    ),
    "TEST_AUTOMATION": ResourceConstraint(
        agent_type="TEST_AUTOMATION",
        max_tokens=4096,
        timeout_seconds=120,
        max_retries=2,
    ),
    # Research — moderate context, more retries allowed
    "RESEARCHER": ResourceConstraint(
        agent_type="RESEARCHER",
        max_tokens=4096,
        timeout_seconds=120,
        max_retries=3,
        max_delegation_calls=3,
    ),
    "EXPLORER": ResourceConstraint(
        agent_type="EXPLORER",
        max_tokens=4096,
        timeout_seconds=90,
        max_retries=2,
        max_delegation_calls=1,
    ),
    "LIBRARIAN": ResourceConstraint(
        agent_type="LIBRARIAN",
        max_tokens=4096,
        timeout_seconds=90,
        max_retries=2,
        max_delegation_calls=1,
    ),
    # Oracle — needs large context for architecture decisions
    "ORACLE": ResourceConstraint(
        agent_type="ORACLE",
        max_tokens=8192,
        timeout_seconds=180,
        max_retries=2,
        max_delegation_calls=1,
    ),
    # Operations — docs need more tokens and time
    "SYNTHESIZER": ResourceConstraint(
        agent_type="SYNTHESIZER",
        max_tokens=8192,
        timeout_seconds=180,
        max_retries=2,
    ),
    "DOCUMENTATION_AGENT": ResourceConstraint(
        agent_type="DOCUMENTATION_AGENT",
        max_tokens=16384,
        timeout_seconds=300,
        max_retries=2,
    ),
    "COST_PLANNER": ResourceConstraint(
        agent_type="COST_PLANNER",
        max_tokens=4096,
        timeout_seconds=60,
        max_retries=2,
    ),
    # Infrastructure
    "UI_PLANNER": ResourceConstraint(
        agent_type="UI_PLANNER",
        max_tokens=8192,
        timeout_seconds=120,
        max_retries=2,
    ),
    "DEVOPS": ResourceConstraint(
        agent_type="DEVOPS",
        max_tokens=4096,
        timeout_seconds=120,
        max_retries=2,
    ),
    "DATA_ENGINEER": ResourceConstraint(
        agent_type="DATA_ENGINEER",
        max_tokens=4096,
        timeout_seconds=120,
        max_retries=2,
    ),
    "IMAGE_GENERATOR": ResourceConstraint(
        agent_type="IMAGE_GENERATOR",
        max_tokens=4096,
        timeout_seconds=180,
        max_retries=2,
    ),
    "ERROR_RECOVERY": ResourceConstraint(
        agent_type="ERROR_RECOVERY",
        max_tokens=4096,
        timeout_seconds=120,
        max_retries=3,
        max_delegation_calls=3,
    ),
    "VERSION_CONTROL": ResourceConstraint(
        agent_type="VERSION_CONTROL",
        max_tokens=4096,
        timeout_seconds=60,
        max_retries=2,
    ),
    "EXPERIMENTATION_MANAGER": ResourceConstraint(
        agent_type="EXPERIMENTATION_MANAGER",
        max_tokens=4096,
        timeout_seconds=120,
        max_retries=2,
    ),
    "IMPROVEMENT": ResourceConstraint(
        agent_type="IMPROVEMENT",
        max_tokens=4096,
        timeout_seconds=120,
        max_retries=2,
    ),
    "USER_INTERACTION": ResourceConstraint(
        agent_type="USER_INTERACTION",
        max_tokens=4096,
        timeout_seconds=60,
        max_retries=2,
    ),
    "CONTEXT_MANAGER": ResourceConstraint(
        agent_type="CONTEXT_MANAGER",
        max_tokens=4096,
        timeout_seconds=60,
        max_retries=2,
    ),
    "PONDER": ResourceConstraint(
        agent_type="PONDER",
        max_tokens=8192,
        timeout_seconds=180,
        max_retries=2,
    ),

    # --- Consolidated agents (Phase 3) ---

    "ORCHESTRATOR": ResourceConstraint(
        agent_type="ORCHESTRATOR",
        max_tokens=4096,
        timeout_seconds=120,
        max_retries=2,
        max_delegation_calls=5,
        max_parallel_tasks=2,
    ),
    "CONSOLIDATED_RESEARCHER": ResourceConstraint(
        agent_type="CONSOLIDATED_RESEARCHER",
        max_tokens=4096,
        timeout_seconds=120,
        max_retries=3,
        max_delegation_calls=2,
        max_parallel_tasks=2,
    ),
    "CONSOLIDATED_ORACLE": ResourceConstraint(
        agent_type="CONSOLIDATED_ORACLE",
        max_tokens=8192,
        timeout_seconds=180,
        max_retries=2,
        max_delegation_calls=1,
        max_parallel_tasks=1,
    ),
    "ARCHITECT": ResourceConstraint(
        agent_type="ARCHITECT",
        max_tokens=8192,
        timeout_seconds=120,
        max_retries=2,
        max_delegation_calls=2,
        max_parallel_tasks=1,
    ),
    "QUALITY": ResourceConstraint(
        agent_type="QUALITY",
        max_tokens=4096,
        timeout_seconds=120,
        max_retries=2,
        max_delegation_calls=2,
        max_parallel_tasks=2,
    ),
    "OPERATIONS": ResourceConstraint(
        agent_type="OPERATIONS",
        max_tokens=16384,
        timeout_seconds=300,
        max_retries=2,
        max_delegation_calls=1,
        max_parallel_tasks=3,
    ),
}

# Default for any agent type not explicitly listed
_DEFAULT_CONSTRAINT = ResourceConstraint(agent_type="DEFAULT")


def get_resource_constraint(agent_type: str) -> ResourceConstraint:
    """Get resource constraints for an agent type (returns defaults if unlisted)."""
    return AGENT_RESOURCE_LIMITS.get(agent_type, _DEFAULT_CONSTRAINT)
