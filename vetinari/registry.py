"""Vetinari Skill Registry.

Centralized registry for managing and discovering Vetinari skills.
Provides API surface for skill discovery, manifest loading, and context retrieval.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from vetinari.skills.skill_registry import (
    SKILL_REGISTRY as _PROGRAMMATIC_REGISTRY,
)
from vetinari.skills.skill_registry import (
    get_skill_for_agent_type as _get_programmatic_skill,
)
from vetinari.skills.skill_registry import (
    get_skills_by_capability as _get_programmatic_by_capability,
)

logger = logging.getLogger(__name__)

# Registry paths
REGISTRY_DIR = Path(__file__).parent
CENTRAL_REGISTRY = REGISTRY_DIR / "skills_registry.json"
AGENT_SKILL_MAP = REGISTRY_DIR / "config" / "agent_skill_map.json"
CONTEXT_REGISTRY = REGISTRY_DIR / "context_registry.json"


class SkillRegistry:
    """Central registry for Vetinari skills.

    Provides methods to:
    - List all available skills
    - Get skill metadata and manifests
    - Query agent-to-skill mappings
    - Retrieve sample contexts
    """

    def __init__(self, load_on_init: bool = True):
        self._registry: dict[str, Any] = {}
        self._manifests: dict[str, dict] = {}
        self._agent_map: dict[str, Any] = {}
        self._contexts: dict[str, dict] = {}
        self._loaded = False

        if load_on_init:
            self.load()

    def load(self) -> None:
        """Load all registry data from disk."""
        try:
            # Load central registry
            if CENTRAL_REGISTRY.exists():
                with open(CENTRAL_REGISTRY, encoding="utf-8") as f:
                    self._registry = json.load(f)
                logger.info("Loaded central registry with %s skills", len(self._registry.get("skills", [])))
            else:
                logger.warning("Central registry not found: %s", CENTRAL_REGISTRY)

            # Load agent-to-skill mapping
            if AGENT_SKILL_MAP.exists():
                with open(AGENT_SKILL_MAP, encoding="utf-8") as f:
                    self._agent_map = json.load(f)
                logger.info("Loaded agent skill mappings for %s agents", len(self._agent_map.get("agents", {})))
            else:
                logger.warning("Agent skill map not found: %s", AGENT_SKILL_MAP)

            # Load context registry
            if CONTEXT_REGISTRY.exists():
                with open(CONTEXT_REGISTRY, encoding="utf-8") as f:
                    context_data = json.load(f)
                    self._contexts = {ctx["id"]: ctx for ctx in context_data.get("contexts", [])}
                logger.info("Loaded %s sample contexts", len(self._contexts))
            else:
                logger.warning("Context registry not found: %s", CONTEXT_REGISTRY)

            self._loaded = True

        except Exception as e:
            logger.error("Failed to load registry: %s", e)
            raise

    @property
    def is_loaded(self) -> bool:
        """Check if registry is loaded."""
        return self._loaded

    def list_skills(self) -> list[dict[str, Any]]:
        """List all available skills with basic metadata.

        Merges disk-based skills with programmatic SkillSpec entries,
        deduplicating by skill id.
        """
        if not self._loaded:
            self.load()
        disk_skills = self._registry.get("skills", [])
        seen_ids = {s.get("id") for s in disk_skills}
        merged = list(disk_skills)
        for spec in _PROGRAMMATIC_REGISTRY.values():
            if spec.skill_id not in seen_ids:
                merged.append(spec.to_dict())
                seen_ids.add(spec.skill_id)
        return merged

    def get_skill(self, skill_id: str) -> dict[str, Any] | None:
        """Get skill metadata by ID.

        Falls back to the programmatic SkillSpec registry if the disk-based
        registry does not contain the requested skill.
        """
        if not self._loaded:
            self.load()
        skills = self._registry.get("skills", [])
        result = next((s for s in skills if s["id"] == skill_id), None)
        if result is not None:
            return result
        # Fallback: check programmatic SkillSpec registry
        spec = _PROGRAMMATIC_REGISTRY.get(skill_id)
        if spec:
            return spec.to_dict()
        return None

    def get_skill_manifest(self, skill_id: str) -> dict[str, Any] | None:
        """Get full manifest for a skill.

        Falls back to the programmatic SkillSpec registry if no disk manifest
        exists.
        """
        if skill_id in self._manifests:
            return self._manifests[skill_id]

        # Try to load manifest from disk
        manifest_path = REGISTRY_DIR / "skills" / skill_id / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
                self._manifests[skill_id] = manifest
                return manifest

        # Fallback: synthesize manifest from programmatic SkillSpec
        spec = _PROGRAMMATIC_REGISTRY.get(skill_id)
        if spec:
            manifest = spec.to_dict()
            self._manifests[skill_id] = manifest
            return manifest
        return None

    def get_skill_capabilities(self, skill_id: str) -> list[str]:
        """Get list of capabilities for a skill."""
        skill = self.get_skill(skill_id)
        if skill:
            return skill.get("capabilities", [])
        manifest = self.get_skill_manifest(skill_id)
        if manifest:
            return manifest.get("capabilities", [])
        return []

    def get_skill_permissions(self, skill_id: str) -> list[str]:
        """Get required permissions for a skill."""
        skill = self.get_skill(skill_id)
        if skill:
            return skill.get("permissions_required", [])
        manifest = self.get_skill_manifest(skill_id)
        if manifest:
            return manifest.get("required_permissions", [])
        return []

    def get_skill_by_capability(self, capability: str) -> list[dict[str, Any]]:
        """Find skills that support a specific capability.

        Searches both the disk-based registry and the programmatic SkillSpec
        registry, deduplicating by skill id.
        """
        seen_ids: set = set()
        matching = []

        # Disk-based skills
        for skill in self.list_skills():
            if capability in skill.get("capabilities", []):
                matching.append(skill)
                seen_ids.add(skill.get("id"))

        # Programmatic SkillSpec skills
        for spec in _get_programmatic_by_capability(capability):
            if spec.skill_id not in seen_ids:
                matching.append(spec.to_dict())
                seen_ids.add(spec.skill_id)

        return matching

    def list_agents(self) -> list[str]:
        """List all registered agents, including consolidated agent types."""
        if not self._loaded:
            self.load()
        disk_agents = set(self._agent_map.get("agents", {}).keys())
        # Add consolidated agent types from programmatic registry
        for spec in _PROGRAMMATIC_REGISTRY.values():
            disk_agents.add(spec.agent_type)
        return sorted(disk_agents)

    def get_agent_skills(self, agent_id: str, env: str | None = None) -> list[dict[str, Any]]:
        """Get skills mapped to a specific agent.

        Falls back to the programmatic SkillSpec when the disk-based map
        does not contain the agent.
        """
        if not self._loaded:
            self.load()

        agents = self._agent_map.get("agents", {})
        agent_config = agents.get(agent_id, {})

        # Check for environment-specific override
        if env:
            overrides = self._agent_map.get("environment_overrides", {})
            env_overrides = overrides.get(env, {})
            agent_config = env_overrides.get("agents", {}).get(agent_id, agent_config)

        disk_skills = agent_config.get("default_skills", [])
        if disk_skills:
            return disk_skills

        # Fallback: look up via programmatic SkillSpec
        spec = _get_programmatic_skill(agent_id)
        if spec:
            return [{"skill_id": spec.skill_id, "source": "programmatic"}]
        return []

    def get_context(self, context_id: str) -> dict[str, Any] | None:
        """Get sample context by ID."""
        if not self._loaded:
            self.load()
        return self._contexts.get(context_id)

    def get_contexts_for_skill(self, skill_id: str) -> list[dict[str, Any]]:
        """Get all contexts available for a specific skill."""
        if not self._loaded:
            self.load()

        contexts = []
        for ctx in self._contexts.values():
            if skill_id in ctx.get("skill_ids", []):
                contexts.append(ctx)
        return contexts

    def list_workflows(self) -> dict[str, list[dict[str, str]]]:
        """List predefined skill workflows."""
        if not self._loaded:
            self.load()
        return self._agent_map.get("workflows", {})

    def get_compatibility_matrix(self) -> dict[str, Any]:
        """Get version compatibility matrix."""
        if not self._loaded:
            self.load()
        return self._registry.get("version_matrix", {})

    def search_skills(self, query: str) -> list[dict[str, Any]]:
        """Search skills by name, description, tags, or capabilities.

        Searches both disk-based and programmatic registries via
        ``list_skills()`` which already merges both sources.
        """
        if not self._loaded:
            self.load()

        query_lower = query.lower()
        skills = self.list_skills()  # already merged

        results = []
        for skill in skills:
            if (
                query_lower in skill.get("name", "").lower()
                or query_lower in skill.get("description", "").lower()
                or any(query_lower in cap.lower() for cap in skill.get("capabilities", []))
                or any(query_lower in tag.lower() for tag in skill.get("tags", []))
            ):
                results.append(skill)

        return results

    def validate(self) -> dict[str, list[str]]:
        """Validate registry integrity.

        Returns:
            Dictionary with validation results: {'errors': [...], 'warnings': [...]}
        """
        errors = []
        warnings = []

        if not self._loaded:
            self.load()

        # Check central registry exists
        if not CENTRAL_REGISTRY.exists():
            errors.append(f"Central registry not found: {CENTRAL_REGISTRY}")

        # Check all skills have manifests
        skills = self.list_skills()
        for skill in skills:
            skill_id = skill["id"]
            manifest = self.get_skill_manifest(skill_id)
            if not manifest:
                warnings.append(f"Manifest missing for skill: {skill_id}")

        # Check agent mappings reference valid skills
        agents = self._agent_map.get("agents", {})
        for agent_id, config in agents.items():
            for skill_mapping in config.get("default_skills", []):
                skill_id = skill_mapping.get("skill_id")
                if not self.get_skill(skill_id):
                    errors.append(f"Agent '{agent_id}' references unknown skill: {skill_id}")

        return {"errors": errors, "warnings": warnings}


# Global registry instance
_global_registry: SkillRegistry | None = None


def get_registry() -> SkillRegistry:
    """Get global registry instance (singleton)."""
    global _global_registry
    if _global_registry is None:
        _global_registry = SkillRegistry()
    return _global_registry


def list_skills() -> list[dict[str, Any]]:
    """Convenience function to list all skills."""
    return get_registry().list_skills()


def get_skill(skill_id: str) -> dict[str, Any] | None:
    """Convenience function to get skill by ID."""
    return get_registry().get_skill(skill_id)


def get_skill_manifest(skill_id: str) -> dict[str, Any] | None:
    """Convenience function to get skill manifest."""
    return get_registry().get_skill_manifest(skill_id)


def get_agent_skills(agent_id: str, env: str | None = None) -> list[dict[str, Any]]:
    """Convenience function to get skills for an agent."""
    return get_registry().get_agent_skills(agent_id, env)


def get_context(context_id: str) -> dict[str, Any] | None:
    """Convenience function to get context by ID."""
    return get_registry().get_context(context_id)


def get_contexts_for_skill(skill_id: str) -> list[dict[str, Any]]:
    """Convenience function to get all contexts for a specific skill."""
    return get_registry().get_contexts_for_skill(skill_id)


def validate_registry() -> dict[str, list[str]]:
    """Convenience function to validate registry."""
    return get_registry().validate()


# Orchestration-specific methods


def get_workflow_template(template_name: str) -> dict[str, Any] | None:
    """Get a predefined workflow template by name."""
    reg = get_registry()
    if not reg.is_loaded:
        reg.load()
    templates = reg._registry.get("workflow_templates", {})
    return templates.get(template_name)


def list_workflow_templates() -> list[str]:
    """List all available workflow templates."""
    reg = get_registry()
    if not reg.is_loaded:
        reg.load()
    return list(reg._registry.get("workflow_templates", {}).keys())


def get_orchestration_config() -> dict[str, Any]:
    """Get orchestration configuration from registry."""
    reg = get_registry()
    if not reg.is_loaded:
        reg.load()
    return reg._registry.get("orchestration_features", {})


def get_skill_dependencies(skill_id: str) -> list[str]:
    """Get skills that the given skill depends on."""
    reg = get_registry()
    if not reg.is_loaded:
        reg.load()
    deps = reg._registry.get("skill_dependencies", {})
    return deps.get(skill_id, [])


def get_skills_for_workflow_stage(stage_purpose: str) -> list[dict[str, Any]]:
    """Find skills matching a specific workflow stage purpose."""
    reg = get_registry()
    if not reg.is_loaded:
        reg.load()

    templates = reg._registry.get("workflow_templates", {})
    matching = []

    for template_name, template in templates.items():
        for stage in template.get("stages", []):
            if stage_purpose.lower() in stage.get("purpose", "").lower():
                matching.append(
                    {
                        "template": template_name,
                        "skill": stage.get("skill"),
                        "capability": stage.get("capability"),
                        "purpose": stage.get("purpose"),
                    }
                )

    return matching
