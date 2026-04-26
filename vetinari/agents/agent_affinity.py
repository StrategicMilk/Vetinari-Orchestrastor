"""Vetinari Agent-Model Affinity Mapping.

======================================
Maps agent types to model capability requirements so the routing layer
can select the best available model for each agent's task type.

On hardware with a 5090 (32 GB VRAM), the primary model is typically a
32B VL model that has vision + reasoning + coding + uncensored thinking.
Smaller specialist models (7B coders, etc.) can be loaded simultaneously
when VRAM allows.

Usage::

    from vetinari.agents.agent_affinity import get_affinity, pick_model_for_agent

    affinity = get_affinity(AgentType.WORKER)
    # -> AffinityProfile(required=["reasoning"], preferred=["analysis"], ...)

    model_id = pick_model_for_agent(AgentType.WORKER, available_models)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vetinari.constants import AFFINITY_LATENCY_BONUS, AFFINITY_LOCAL_BONUS, AFFINITY_UNCENSORED_BONUS
from vetinari.knowledge import get_family_profile
from vetinari.types import AgentType

# Canonical capability names — all variants map to the canonical form so
# model tags and affinity profiles can use different spellings interchangeably.
_CAPABILITY_ALIASES: dict[str, str] = {
    "code": "coding",
    "code_gen": "coding",
    "code_generation": "coding",
    "coder": "coding",
    "programming": "coding",
    "dev": "coding",
    "reason": "reasoning",
    "think": "reasoning",
    "thinking": "reasoning",
    "advanced": "reasoning",
    "analyze": "analysis",
    "analyse": "analysis",
    "research": "analysis",
    "fast": "low_latency",
    "speed": "low_latency",
    "turbo": "low_latency",
    "small": "low_latency",
    "writing": "creative",
    "story": "creative",
    "general": "general",
    "generalPurpose": "general",
}


def _normalize_capability(cap: str) -> str:
    """Normalize a capability name to its canonical form.

    Args:
        cap: Raw capability string from model tags or affinity profile.

    Returns:
        Canonical capability name, or the original lowercased if not in the alias map.
    """
    return _CAPABILITY_ALIASES.get(cap.lower(), cap.lower())


# Maps substrings found in model IDs to their family slug in model_families.yaml.
# Ordered from most-specific to least-specific so "qwen2.5" matches "qwen2" before "qwen".
_FAMILY_SLUG_PATTERNS: list[tuple[str, str]] = [
    ("qwen2", "qwen2"),
    ("qwen", "qwen2"),
    ("deepseek-r", "deepseek"),
    ("deepseek", "deepseek"),
    ("llama", "llama"),
    ("mistral", "mistral"),
    ("mixtral", "mixtral"),
    ("phi-4", "phi"),
    ("phi", "phi"),
    ("gemma-3", "gemma"),
    ("gemma", "gemma"),
    ("falcon", "falcon"),
    ("command", "cohere"),
]


def _extract_family_slug(model: dict[str, Any]) -> str:
    """Heuristically derive a model_families.yaml family slug from a model dict.

    Checks the model's explicit ``family`` field first, then falls back to
    substring matching on ``model_id``.

    Args:
        model: Model dict from the ModelRegistry.

    Returns:
        Family slug string (e.g. ``'qwen2'``, ``'llama'``), or empty string
        if no known family is detected.
    """
    # Prefer explicit family field when present
    explicit = model.get("family", "")
    if explicit:
        return explicit.lower()

    model_id = model.get("model_id", model.get("id", "")).lower()
    for substring, slug in _FAMILY_SLUG_PATTERNS:
        if substring in model_id:
            return slug
    return ""


def _get_family_capability_bonus(model: dict[str, Any], preferred_caps: list[str]) -> float:
    """Score a model's family-level capability match against preferred capabilities.

    Uses model_families.yaml knowledge to award a fractional bonus for each
    preferred capability the model's family is proficient at — even when the
    model's tag list doesn't explicitly declare it (e.g. Qwen2 models are
    strong coders but may not carry a ``coding`` tag).

    The bonus is capped at 0.5 per capability so it supplements but never
    overrides the tag-based score.

    Args:
        model: Model dict from the ModelRegistry.
        preferred_caps: Normalized preferred capability names from AffinityProfile.

    Returns:
        Float bonus score to add to the tag-based score.
    """
    family_slug = _extract_family_slug(model)
    if not family_slug:
        return 0.0

    profile = get_family_profile(family_slug)
    if not profile:
        return 0.0

    capabilities = profile.get("capabilities", {})
    if not capabilities:
        return 0.0

    bonus = 0.0
    for cap in preferred_caps:
        normalized = _normalize_capability(cap)
        cap_data = capabilities.get(normalized)
        if cap_data is None:
            continue

        if isinstance(cap_data, dict):
            # Aggregate sub-capability scores (e.g. coding.python, coding.debugging)
            values = [v for v in cap_data.values() if isinstance(v, (int, float))]
            if values:
                avg = sum(values) / len(values)
                bonus += avg * 0.5  # scale to keep bonus < tag-based score
        elif isinstance(cap_data, (int, float)):
            bonus += float(cap_data) * 0.5

    return bonus


@dataclass
class AffinityProfile:
    """Capability requirements and preferences for an agent type."""

    agent_type: AgentType
    # Capabilities the chosen model MUST have (hard filter)
    required_capabilities: list[str] = field(default_factory=list)
    # Capabilities preferred (soft bonus in scoring)
    preferred_capabilities: list[str] = field(default_factory=list)
    # Minimum context window needed for typical tasks
    min_context_window: int = 8192
    # Whether vision capability is essential (True = reject non-VL models)
    requires_vision: bool = False
    # Latency preference: "fast" | "medium" | "any"
    latency_preference: str = "any"
    # Whether this agent benefits from uncensored / thinking models
    prefers_uncensored: bool = False
    # Fallback agent types if preferred not available
    fallback_agents: list[AgentType] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"AffinityProfile(agent_type={self.agent_type!r}, required_capabilities={self.required_capabilities!r})"


# ---------------------------------------------------------------------------
# Affinity table — one entry per active AgentType (3-tier model)
# ---------------------------------------------------------------------------

_AFFINITY_TABLE: dict[AgentType, AffinityProfile] = {
    # ── Foreman: plans, decomposes goals, orchestrates execution.
    # Needs strong reasoning + large context for complex decomposition and routing.
    AgentType.FOREMAN: AffinityProfile(
        agent_type=AgentType.FOREMAN,
        required_capabilities=["reasoning"],
        preferred_capabilities=["analysis", "coding"],
        min_context_window=32768,
        latency_preference="medium",
        prefers_uncensored=True,  # uncensored thinking helps thorough analysis
    ),
    # ── Worker: executes all implementation, research, and operations tasks.
    # Handles code generation, research, data engineering, devops, docs, and more.
    # Vision preferred for UI/design tasks but not hard-required; tasks vary widely.
    AgentType.WORKER: AffinityProfile(
        agent_type=AgentType.WORKER,
        required_capabilities=["reasoning"],
        preferred_capabilities=["analysis", "coding", "vision"],
        min_context_window=32768,
        requires_vision=False,  # vision preferred but not hard-required; tasks vary
        latency_preference="any",
        prefers_uncensored=True,
    ),
    # ── Inspector: reviews quality, runs security audits, generates tests.
    # Needs coding capability for deep code analysis; uncensored for attack vectors.
    AgentType.INSPECTOR: AffinityProfile(
        agent_type=AgentType.INSPECTOR,
        required_capabilities=["coding"],
        preferred_capabilities=["reasoning", "analysis"],
        min_context_window=16384,
        latency_preference="medium",
        prefers_uncensored=True,  # needed for security audit thinking
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_affinity(agent_type: AgentType) -> AffinityProfile:
    """Return the AffinityProfile for an agent type.

    Args:
        agent_type: The agent type whose affinity profile is requested.

    Returns:
        The AffinityProfile for the given agent type, or a sensible default
        if the agent type has no registered profile.
    """
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
    available_models: list[dict[str, Any]],
    default: str | None = None,
) -> str | None:
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
        caps = {_normalize_capability(c) for c in model.get("capabilities", model.get("tags", []))}
        ctx = model.get("context_window", model.get("context_len", model.get("context_length", 8192)))

        # Hard filters
        if ctx < affinity.min_context_window:
            continue
        if affinity.requires_vision and "vision" not in caps:
            continue
        # Required capabilities — all must be present
        if affinity.required_capabilities and not all(
            _normalize_capability(rc) in caps for rc in affinity.required_capabilities
        ):
            continue

        # Score: count preferred capabilities present in tags
        score = sum(1 for pc in affinity.preferred_capabilities if _normalize_capability(pc) in caps)
        # Family-level capability bonus supplements tag matching for models
        # that lack explicit capability tags but belong to a proficient family
        score += _get_family_capability_bonus(model, affinity.preferred_capabilities)

        # Bonus for latency preference match
        latency = model.get("latency_hint", "medium")
        if affinity.latency_preference != "any" and latency == affinity.latency_preference:
            score += AFFINITY_LATENCY_BONUS

        # Bonus for uncensored preference
        if affinity.prefers_uncensored and ("uncensored" in caps or "uncensored" in model.get("model_id", "").lower()):
            score += AFFINITY_UNCENSORED_BONUS

        # Prefer local models
        if model.get("privacy_level", "") == "local":
            score += AFFINITY_LOCAL_BONUS

        candidates.append((score, model.get("model_id", model.get("id", ""))))

    if not candidates:
        return default

    # Return highest-scoring candidate (stable sort: prefer first in list on tie)
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1] or default


def get_all_affinities() -> dict[AgentType, AffinityProfile]:
    """Return the complete affinity table.

    Returns:
        A copy of the internal affinity table mapping each active AgentType
        to its AffinityProfile.
    """
    return dict(_AFFINITY_TABLE)
