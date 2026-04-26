"""Workflow and orchestration convenience functions for the Vetinari skill system.

Provides helpers for querying workflow templates, orchestration feature flags,
skill dependencies, and stage-to-skill mappings from the central registry.

Split from ``skill_registry.py`` to keep the registry module under 800 lines.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_workflow_template(template_name: str) -> dict[str, Any] | None:
    """Get a predefined workflow template by name.

    Args:
        template_name: The workflow template identifier.

    Returns:
        Template dict with ``stages`` list, or None if not found.
    """
    from vetinari.skills.skill_registry import get_registry

    reg = get_registry()
    return reg._registry.get("workflow_templates", {}).get(template_name)


def list_workflow_templates() -> list[str]:
    """List the names of all available workflow templates.

    Returns:
        List of template name strings.
    """
    from vetinari.skills.skill_registry import get_registry

    reg = get_registry()
    return list(reg._registry.get("workflow_templates", {}).keys())


def get_orchestration_config() -> dict[str, Any]:
    """Get the orchestration configuration flags from the central registry.

    Returns:
        Dict of orchestration feature flags, or an empty dict.
    """
    from vetinari.skills.skill_registry import get_registry

    reg = get_registry()
    return reg._registry.get("orchestration_features", {})


def get_skill_dependencies(skill_id: str) -> list[str]:
    """Get the list of skill ids that a given skill depends on.

    Args:
        skill_id: The skill identifier.

    Returns:
        List of dependency skill id strings.
    """
    from vetinari.skills.skill_registry import get_registry

    reg = get_registry()
    return reg._registry.get("skill_dependencies", {}).get(skill_id, [])


def get_skills_for_workflow_stage(stage_purpose: str) -> list[dict[str, Any]]:
    """Find skills matching a specific workflow stage purpose string.

    Case-insensitive substring match against each stage's ``purpose`` field.

    Args:
        stage_purpose: Substring to match against stage purpose descriptions.

    Returns:
        List of dicts with keys ``template``, ``skill``, ``capability``, and
        ``purpose`` for each matching stage.
    """
    from vetinari.skills.skill_registry import get_registry

    reg = get_registry()
    templates = reg._registry.get("workflow_templates", {})
    matching: list[dict[str, Any]] = []

    for template_name, template in templates.items():
        matching.extend(
            {
                "template": template_name,
                "skill": stage.get("skill"),
                "capability": stage.get("capability"),
                "purpose": stage.get("purpose"),
            }
            for stage in template.get("stages", [])
            if stage_purpose.lower() in stage.get("purpose", "").lower()
        )

    return matching
