"""Disk-aware convenience wrapper functions for Vetinari's SkillRegistry.

These thin wrappers call ``get_registry().<method>()`` so callers can use
module-level functions without explicitly fetching the singleton.  The import
of ``get_registry`` is deferred into each function body to avoid a circular
import with ``skill_registry.py``.

All names are re-exported from ``skill_registry.py`` so existing import paths
remain valid.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "activate_registry_skill",
    "check_registry_for_changes",
    "get_agent_skills",
    "get_context",
    "get_contexts_for_skill",
    "get_registry_skill",
    "get_skill_loading_level",
    "get_skill_manifest",
    "get_skill_summary",
    "list_agents",
    "list_registry_skills",
    "list_skill_summaries",
    "propose_registry_skill",
    "search_skills",
    "validate_registry",
    "validate_skill_output",
    "verify_skill_trust_elevation",
]


def list_registry_skills() -> list[dict[str, Any]]:
    """List all skills from the disk-aware registry.

    Returns:
        List of skill metadata dicts (disk + programmatic, deduplicated).
    """
    from vetinari.skills.skill_registry import get_registry

    return get_registry().list_skills()


def get_registry_skill(skill_id: str) -> dict[str, Any] | None:
    """Get skill metadata by ID from the disk-aware registry.

    Args:
        skill_id: The skill identifier.

    Returns:
        Skill metadata dict, or None if not found.
    """
    from vetinari.skills.skill_registry import get_registry

    return get_registry().get_skill(skill_id)


def get_skill_manifest(skill_id: str) -> dict[str, Any] | None:
    """Get the full manifest for a skill from the disk-aware registry.

    Args:
        skill_id: The skill identifier.

    Returns:
        Manifest dict, or None if not found.
    """
    from vetinari.skills.skill_registry import get_registry

    return get_registry().get_skill_manifest(skill_id)


def get_agent_skills(agent_id: str, env: str | None = None) -> list[dict[str, Any]]:
    """Get skills mapped to a specific agent from the disk-aware registry.

    Args:
        agent_id: The agent id or canonical agent type.
        env: Optional environment name for environment-specific overrides.

    Returns:
        List of skill mapping dicts.
    """
    from vetinari.skills.skill_registry import get_registry

    return get_registry().get_agent_skills(agent_id, env)


def get_context(context_id: str) -> dict[str, Any] | None:
    """Get a sample context by ID from the disk-aware registry.

    Args:
        context_id: The context identifier.

    Returns:
        Context dict, or None if not found.
    """
    from vetinari.skills.skill_registry import get_registry

    return get_registry().get_context(context_id)


def get_contexts_for_skill(skill_id: str) -> list[dict[str, Any]]:
    """Get all sample contexts for a specific skill from the disk-aware registry.

    Args:
        skill_id: The skill identifier.

    Returns:
        List of context dicts whose ``skill_ids`` field includes ``skill_id``.
    """
    from vetinari.skills.skill_registry import get_registry

    return get_registry().get_contexts_for_skill(skill_id)


def validate_registry() -> dict[str, list[str]]:
    """Validate the disk-aware registry and return errors and warnings.

    Returns:
        Dict with ``'errors'`` and ``'warnings'`` lists.
    """
    from vetinari.skills.skill_registry import get_registry

    return get_registry().validate()


def search_skills(query: str) -> list[dict[str, Any]]:
    """Search skills by name, description, tags, or capabilities.

    Args:
        query: Case-insensitive search string.

    Returns:
        List of matching skill metadata dicts.
    """
    from vetinari.skills.skill_registry import get_registry

    return get_registry().search_skills(query)


def list_agents() -> list[str]:
    """List all registered agent types from the disk-aware registry.

    Returns:
        Sorted list of agent type strings.
    """
    from vetinari.skills.skill_registry import get_registry

    return get_registry().list_agents()


def activate_registry_skill(skill_id: str) -> dict[str, Any] | None:
    """Elevate a skill to Level 2 (full spec) and return its complete metadata.

    Wraps ``SkillRegistry.activate_skill`` for use without an explicit
    registry reference.

    Args:
        skill_id: The skill identifier to activate.

    Returns:
        Full skill metadata dict, or None if not found.
    """
    from vetinari.skills.skill_registry import get_registry

    return get_registry().activate_skill(skill_id)


def get_skill_loading_level(skill_id: str) -> int:
    """Return the current progressive disclosure level for a skill (0/1/2).

    Wraps ``SkillRegistry.get_loading_level`` for use without an explicit
    registry reference.

    Args:
        skill_id: The skill identifier.

    Returns:
        Integer level: 0 = not loaded, 1 = summary, 2 = full spec.
    """
    from vetinari.skills.skill_registry import get_registry

    return get_registry().get_loading_level(skill_id)


def get_skill_summary(skill_id: str) -> dict[str, str] | None:
    """Return Level 1 metadata (~100 tokens) for a single skill.

    Returns id, name, description, and trust_tier without loading the full
    spec — suitable for agent startup context where token budget is tight.

    Wraps ``SkillRegistry.get_skill_summary`` for use without an explicit
    registry reference.

    Args:
        skill_id: The skill identifier.

    Returns:
        Dict with id, name, description, trust_tier — or None if not found.
    """
    from vetinari.skills.skill_registry import get_registry

    return get_registry().get_skill_summary(skill_id)


def list_skill_summaries() -> list[dict[str, str]]:
    """Return Level 1 summaries for all registered skills.

    Collects id, name, description, and trust_tier for every skill without
    loading full specs — intended for agent startup context injection.

    Wraps ``SkillRegistry.list_skill_summaries`` for use without an explicit
    registry reference.

    Returns:
        List of summary dicts with id, name, description, trust_tier.
    """
    from vetinari.skills.skill_registry import get_registry

    return get_registry().list_skill_summaries()


def validate_skill_output(skill_id: str, output: Any) -> tuple[bool, list[str]]:
    """Run registered output validators for a skill against produced output.

    Returns a pass/fail verdict and a list of failure messages.  When the
    skill has no validators registered, always returns ``(True, [])``.

    Wraps ``SkillRegistry.validate_skill_output`` for use without an explicit
    registry reference.

    Args:
        skill_id: The skill identifier whose validators to run.
        output: The output produced by the skill to validate.

    Returns:
        Tuple of (all_passed, list_of_failure_messages).
    """
    from vetinari.skills.skill_registry import get_registry

    return get_registry().validate_skill_output(skill_id, output)


def check_registry_for_changes() -> bool:
    """Check whether the central skill registry file has changed on disk.

    When the file has been modified since it was last loaded, the registry
    ``is_loaded`` flag is reset so the next access triggers a fresh load.

    Wraps ``SkillRegistry.check_for_changes`` for use without an explicit
    registry reference.

    Returns:
        True if the registry file has been modified since last load.
    """
    from vetinari.skills.skill_registry import get_registry

    return get_registry().check_for_changes()


def propose_registry_skill(
    skill_id: str,
    name: str,
    description: str,
    capabilities: list[str],
    proposed_by: str = "agent",
) -> dict[str, Any]:
    """Submit a new skill proposal for human review at T1 trust tier.

    The proposal is queued in the registry with ``status='pending_review'``
    and will not be activated until a human approves it.  Duplicate
    ``skill_id`` values are rejected immediately.

    Wraps ``SkillRegistry.propose_skill`` for use without an explicit
    registry reference.

    Args:
        skill_id: Proposed identifier for the new skill.
        name: Human-readable skill name.
        description: What the skill does.
        capabilities: List of capability strings the skill will declare.
        proposed_by: Source of the proposal (e.g. agent name or 'human').

    Returns:
        Dict with ``status`` ('rejected' or 'pending_review') and
        ``proposal`` details.
    """
    from vetinari.skills.skill_registry import get_registry

    return get_registry().propose_skill(skill_id, name, description, capabilities, proposed_by)


def verify_skill_trust_elevation(skill_id: str) -> dict[str, Any]:
    """Run the 4-gate verification chain to assess trust tier eligibility.

    Gates checked: static metadata completeness (G1), semantic capability
    declarations (G2), output schema presence (G3), and resource limit
    configuration (G4).

    Wraps ``SkillRegistry.verify_trust_elevation`` for use without an
    explicit registry reference.

    Args:
        skill_id: The skill identifier to verify.

    Returns:
        Dict with ``overall_pass`` (bool), ``gate_results`` (per-gate
        bool dict), and ``current_tier`` string.
    """
    from vetinari.skills.skill_registry import get_registry

    return get_registry().verify_trust_elevation(skill_id)
