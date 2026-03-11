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
    # Oracle — needs large context for architecture decisions
    "ORACLE": ResourceConstraint(
        agent_type="ORACLE",
        max_tokens=8192,
        timeout_seconds=180,
        max_retries=2,
        max_delegation_calls=1,
    ),

    # --- Consolidated agents ---

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
