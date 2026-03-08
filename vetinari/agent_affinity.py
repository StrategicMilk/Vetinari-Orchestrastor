"""
Vetinari Agent-Model Affinity Mapping
======================================
Maps agent types to model capability requirements so the routing layer
can select the best available model for each agent's task type.

On hardware with a 5090 (32 GB VRAM), the primary model is typically a
32B VL model that has vision + reasoning + coding + uncensored thinking.
Smaller specialist models (7B coders, etc.) can be loaded simultaneously
when VRAM allows.

Usage::

    from vetinari.agent_affinity import get_affinity, pick_model_for_agent

    affinity = get_affinity(AgentType.UI_PLANNER)
    # -> AffinityProfile(required=["vision"], preferred=["coding"], ...)

    model_id = pick_model_for_agent(AgentType.BUILDER, available_models)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from vetinari.agents.contracts import AgentType


@dataclass
class AffinityProfile:
    """Capability requirements and preferences for an agent type."""
    agent_type: AgentType
    # Capabilities the chosen model MUST have (hard filter)
    required_capabilities: List[str] = field(default_factory=list)
    # Capabilities preferred (soft bonus in scoring)
    preferred_capabilities: List[str] = field(default_factory=list)
    # Minimum context window needed for typical tasks
    min_context_window: int = 8192
    # Whether vision capability is essential (True = reject non-VL models)
    requires_vision: bool = False
    # Latency preference: "fast" | "medium" | "any"
    latency_preference: str = "any"
    # Whether this agent benefits from uncensored / thinking models
    prefers_uncensored: bool = False
    # Fallback agent types if preferred not available
    fallback_agents: List[AgentType] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Affinity table — one entry per AgentType
# ---------------------------------------------------------------------------

_AFFINITY_TABLE: Dict[AgentType, AffinityProfile] = {
    # ── Planner: needs strong reasoning + large context for complex decomposition
    AgentType.PLANNER: AffinityProfile(
        agent_type=AgentType.PLANNER,
        required_capabilities=["reasoning"],
        preferred_capabilities=["analysis", "coding"],
        min_context_window=16384,
        latency_preference="medium",
        prefers_uncensored=True,  # uncensored thinking helps thorough analysis
    ),

    # ── Explorer: fast search, codebase scanning
    AgentType.EXPLORER: AffinityProfile(
        agent_type=AgentType.EXPLORER,
        required_capabilities=["coding"],
        preferred_capabilities=["reasoning"],
        min_context_window=8192,
        latency_preference="fast",
    ),

    # ── Oracle: architecture decisions, deep reasoning
    AgentType.ORACLE: AffinityProfile(
        agent_type=AgentType.ORACLE,
        required_capabilities=["reasoning"],
        preferred_capabilities=["analysis", "coding"],
        min_context_window=16384,
        latency_preference="medium",
        prefers_uncensored=True,
    ),

    # ── Librarian: documentation lookup, library research
    AgentType.LIBRARIAN: AffinityProfile(
        agent_type=AgentType.LIBRARIAN,
        required_capabilities=["reasoning"],
        preferred_capabilities=["coding"],
        min_context_window=8192,
        latency_preference="medium",
    ),

    # ── Researcher: multi-source synthesis, long outputs
    AgentType.RESEARCHER: AffinityProfile(
        agent_type=AgentType.RESEARCHER,
        required_capabilities=["reasoning"],
        preferred_capabilities=["analysis"],
        min_context_window=16384,
        latency_preference="any",
        prefers_uncensored=True,
    ),

    # ── Evaluator: code review and quality scoring
    AgentType.EVALUATOR: AffinityProfile(
        agent_type=AgentType.EVALUATOR,
        required_capabilities=["coding"],
        preferred_capabilities=["reasoning", "analysis"],
        min_context_window=16384,
        latency_preference="medium",
    ),

    # ── Synthesizer: combines multiple agent outputs
    AgentType.SYNTHESIZER: AffinityProfile(
        agent_type=AgentType.SYNTHESIZER,
        required_capabilities=["reasoning"],
        preferred_capabilities=["analysis"],
        min_context_window=16384,
        latency_preference="medium",
    ),

    # ── Builder: code generation, scaffolding
    AgentType.BUILDER: AffinityProfile(
        agent_type=AgentType.BUILDER,
        required_capabilities=["coding"],
        preferred_capabilities=["reasoning"],
        min_context_window=8192,
        latency_preference="medium",
    ),

    # ── UI Planner: UI/UX design — REQUIRES vision capability
    AgentType.UI_PLANNER: AffinityProfile(
        agent_type=AgentType.UI_PLANNER,
        required_capabilities=["coding"],  # vision is checked separately via requires_vision
        preferred_capabilities=["vision", "reasoning"],
        min_context_window=8192,
        requires_vision=True,
        latency_preference="medium",
        fallback_agents=[AgentType.BUILDER],
    ),

    # ── Security Auditor: security analysis, must handle sensitive content
    AgentType.SECURITY_AUDITOR: AffinityProfile(
        agent_type=AgentType.SECURITY_AUDITOR,
        required_capabilities=["coding"],
        preferred_capabilities=["reasoning", "analysis"],
        min_context_window=8192,
        latency_preference="medium",
        prefers_uncensored=True,  # needs to discuss attack vectors openly
    ),

    # ── Data Engineer: schema/migration design
    AgentType.DATA_ENGINEER: AffinityProfile(
        agent_type=AgentType.DATA_ENGINEER,
        required_capabilities=["coding"],
        preferred_capabilities=["reasoning"],
        min_context_window=8192,
        latency_preference="medium",
    ),

    # ── Documentation Agent: large output, writing quality
    AgentType.DOCUMENTATION_AGENT: AffinityProfile(
        agent_type=AgentType.DOCUMENTATION_AGENT,
        required_capabilities=["reasoning"],
        preferred_capabilities=["coding"],
        min_context_window=16384,
        latency_preference="any",
    ),

    # ── Cost Planner: reasoning about costs/models
    AgentType.COST_PLANNER: AffinityProfile(
        agent_type=AgentType.COST_PLANNER,
        required_capabilities=["reasoning"],
        preferred_capabilities=["analysis"],
        min_context_window=4096,
        latency_preference="fast",
    ),

    # ── Test Automation: code generation for tests
    AgentType.TEST_AUTOMATION: AffinityProfile(
        agent_type=AgentType.TEST_AUTOMATION,
        required_capabilities=["coding"],
        preferred_capabilities=["reasoning"],
        min_context_window=8192,
        latency_preference="medium",
    ),

    # ── Experimentation Manager: analytics / reporting
    AgentType.EXPERIMENTATION_MANAGER: AffinityProfile(
        agent_type=AgentType.EXPERIMENTATION_MANAGER,
        required_capabilities=["reasoning"],
        preferred_capabilities=["analysis"],
        min_context_window=8192,
        latency_preference="any",
    ),

    # ── Improvement Agent: meta-reasoning about the system
    AgentType.IMPROVEMENT: AffinityProfile(
        agent_type=AgentType.IMPROVEMENT,
        required_capabilities=["reasoning"],
        preferred_capabilities=["analysis", "coding"],
        min_context_window=16384,
        latency_preference="any",
        prefers_uncensored=True,
    ),

    # ── User Interaction: conversational, fast responses
    AgentType.USER_INTERACTION: AffinityProfile(
        agent_type=AgentType.USER_INTERACTION,
        required_capabilities=["reasoning"],
        preferred_capabilities=[],
        min_context_window=4096,
        latency_preference="fast",
    ),

    # ── DevOps: CI/CD, Docker, IaC knowledge
    AgentType.DEVOPS: AffinityProfile(
        agent_type=AgentType.DEVOPS,
        required_capabilities=["coding"],
        preferred_capabilities=["reasoning"],
        min_context_window=8192,
        latency_preference="medium",
    ),

    # ── Version Control: git operations
    AgentType.VERSION_CONTROL: AffinityProfile(
        agent_type=AgentType.VERSION_CONTROL,
        required_capabilities=["coding"],
        preferred_capabilities=["reasoning"],
        min_context_window=8192,
        latency_preference="fast",
    ),

    # ── Error Recovery: analyse failures, resilience
    AgentType.ERROR_RECOVERY: AffinityProfile(
        agent_type=AgentType.ERROR_RECOVERY,
        required_capabilities=["reasoning"],
        preferred_capabilities=["coding", "analysis"],
        min_context_window=8192,
        latency_preference="medium",
        prefers_uncensored=True,
    ),

    # ── Context Manager: memory consolidation, summarisation
    AgentType.CONTEXT_MANAGER: AffinityProfile(
        agent_type=AgentType.CONTEXT_MANAGER,
        required_capabilities=["reasoning"],
        preferred_capabilities=["analysis"],
        min_context_window=32768,  # needs large context to consolidate memories
        latency_preference="any",
    ),

    # ── Image Generator: vision output, image synthesis
    AgentType.IMAGE_GENERATOR: AffinityProfile(
        agent_type=AgentType.IMAGE_GENERATOR,
        required_capabilities=["vision"],
        preferred_capabilities=["reasoning"],
        min_context_window=4096,
        requires_vision=True,
        latency_preference="any",
        fallback_agents=[AgentType.UI_PLANNER],
    ),

    # ── Ponder: deep reflective reasoning, self-evaluation
    AgentType.PONDER: AffinityProfile(
        agent_type=AgentType.PONDER,
        required_capabilities=["reasoning"],
        preferred_capabilities=["analysis"],
        min_context_window=16384,
        latency_preference="any",
        prefers_uncensored=True,
    ),

    # ── Orchestrator (consolidated): top-level coordination, task routing
    AgentType.ORCHESTRATOR: AffinityProfile(
        agent_type=AgentType.ORCHESTRATOR,
        required_capabilities=["reasoning"],
        preferred_capabilities=["analysis", "coding"],
        min_context_window=32768,
        latency_preference="medium",
        prefers_uncensored=True,
    ),

    # ── Consolidated Researcher: deep multi-source research with synthesis
    AgentType.CONSOLIDATED_RESEARCHER: AffinityProfile(
        agent_type=AgentType.CONSOLIDATED_RESEARCHER,
        required_capabilities=["reasoning"],
        preferred_capabilities=["analysis"],
        min_context_window=32768,
        latency_preference="any",
        prefers_uncensored=True,
    ),

    # ── Consolidated Oracle: architecture + system design decisions
    AgentType.CONSOLIDATED_ORACLE: AffinityProfile(
        agent_type=AgentType.CONSOLIDATED_ORACLE,
        required_capabilities=["reasoning"],
        preferred_capabilities=["analysis", "coding"],
        min_context_window=32768,
        latency_preference="medium",
        prefers_uncensored=True,
    ),

    # ── Architect (consolidated): system boundaries, interface design
    AgentType.ARCHITECT: AffinityProfile(
        agent_type=AgentType.ARCHITECT,
        required_capabilities=["reasoning"],
        preferred_capabilities=["analysis", "coding"],
        min_context_window=32768,
        latency_preference="medium",
        prefers_uncensored=True,
    ),

    # ── Quality (consolidated): review, test, security — low temperature for precision
    AgentType.QUALITY: AffinityProfile(
        agent_type=AgentType.QUALITY,
        required_capabilities=["coding"],
        preferred_capabilities=["reasoning", "analysis"],
        min_context_window=16384,
        latency_preference="medium",
    ),

    # ── Operations (consolidated): devops, scheduling, monitoring
    AgentType.OPERATIONS: AffinityProfile(
        agent_type=AgentType.OPERATIONS,
        required_capabilities=["coding"],
        preferred_capabilities=["reasoning"],
        min_context_window=8192,
        latency_preference="medium",
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_affinity(agent_type: AgentType) -> AffinityProfile:
    """Return the AffinityProfile for an agent type."""
    return _AFFINITY_TABLE.get(
        agent_type,
        AffinityProfile(
            agent_type=agent_type,
            required_capabilities=["reasoning"],
            min_context_window=4096,
        ),
    )


def pick_model_for_agent(
    agent_type: AgentType,
    available_models: List[Dict[str, Any]],
    default: Optional[str] = None,
) -> Optional[str]:
    """Select the best available model for an agent based on its affinity profile.

    Args:
        agent_type:       The agent needing a model
        available_models: List of model dicts from the ModelRegistry
        default:          Fallback model_id if no suitable match found

    Returns:
        The model_id of the best-matched model, or ``default``.
    """
    affinity = get_affinity(agent_type)
    candidates = []

    for model in available_models:
        caps = set(model.get("capabilities", model.get("tags", [])))
        ctx = model.get("context_window", model.get("context_len", model.get("context_length", 4096)))

        # Hard filters
        if ctx < affinity.min_context_window:
            continue
        if affinity.requires_vision and "vision" not in caps:
            continue
        # Required capabilities — all must be present
        if affinity.required_capabilities:
            if not all(rc in caps for rc in affinity.required_capabilities):
                continue

        # Score: count preferred capabilities present
        score = sum(1 for pc in affinity.preferred_capabilities if pc in caps)

        # Bonus for latency preference match
        latency = model.get("latency_hint", "medium")
        if affinity.latency_preference != "any" and latency == affinity.latency_preference:
            score += 0.5

        # Bonus for uncensored preference
        if affinity.prefers_uncensored and ("uncensored" in caps or "uncensored" in model.get("model_id", "").lower()):
            score += 1.0

        # Prefer local models
        if model.get("privacy_level", "") == "local":
            score += 0.5

        candidates.append((score, model.get("model_id", model.get("id", ""))))

    if not candidates:
        return default

    # Return highest-scoring candidate (stable sort: prefer first in list on tie)
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1] or default


def get_all_affinities() -> Dict[AgentType, AffinityProfile]:
    """Return the complete affinity table."""
    return dict(_AFFINITY_TABLE)
