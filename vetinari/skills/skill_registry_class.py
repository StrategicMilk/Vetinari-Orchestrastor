"""Disk-aware SkillRegistry class.

``SkillRegistry`` merges disk-based JSON skill manifests with the programmatic
``SKILL_REGISTRY`` (typed ``SkillSpec`` entries defined in
``vetinari.skills.skill_definitions``).  It is kept in a separate module so
that ``skill_registry.py`` stays under the 550-line ceiling while still
exposing the same public API.

Also provides:
- ``get_registry()`` — global singleton accessor
- Disk-aware convenience functions (``list_registry_skills``, ``get_registry_skill``,
  ``get_skill_manifest``, ``get_agent_skills``, ``get_context``,
  ``get_contexts_for_skill``, ``validate_registry``, ``search_skills``,
  ``list_agents``)
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from vetinari.skills.skill_definitions import SKILL_REGISTRY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry file paths
# ---------------------------------------------------------------------------

_VETINARI_PKG = Path(__file__).parent.parent
_CENTRAL_REGISTRY = _VETINARI_PKG / "skills_registry.json"
_AGENT_SKILL_MAP = _VETINARI_PKG / "config" / "agent_skill_map.json"
_CONTEXT_REGISTRY = _VETINARI_PKG / "context_registry.json"


# ---------------------------------------------------------------------------
# SkillRegistry class
# ---------------------------------------------------------------------------


class SkillRegistry:
    """Central registry for Vetinari skills.

    Merges disk-based JSON skill manifests with the programmatic
    ``SKILL_REGISTRY`` (typed ``SkillSpec`` entries).  Disk entries take
    precedence over programmatic ones when both define the same skill id.

    Provides methods to:
    - List all available skills
    - Get skill metadata and manifests
    - Query agent-to-skill mappings
    - Retrieve sample contexts
    - Search skills by name, description, tags, or capabilities
    - Validate registry integrity
    - Enumerate workflow templates
    """

    def __init__(self, load_on_init: bool = True) -> None:
        """Initialise the registry, optionally loading from disk immediately.

        Args:
            load_on_init: When True (default) call ``load()`` in ``__init__``.
        """
        self._registry: dict[str, Any] = {}
        self._manifests: dict[str, dict[str, Any]] = {}
        self._agent_map: dict[str, Any] = {}
        self._contexts: dict[str, dict[str, Any]] = {}
        self.is_loaded = False
        self._loading_levels: dict[str, int] = {}  # Progressive disclosure level per skill

        if load_on_init:
            self.load()

    def load(self) -> None:
        """Load all registry data from disk.

        Raises:
            Exception: Re-raises any exception encountered while reading
                registry files from disk.
        """
        try:
            if _CENTRAL_REGISTRY.exists():
                with Path(_CENTRAL_REGISTRY).open(encoding="utf-8") as f:
                    self._registry = json.load(f)
                logger.info(
                    "Loaded central registry with %s skills",
                    len(self._registry.get("skills", [])),
                )
            else:
                logger.warning("Central registry not found: %s", _CENTRAL_REGISTRY)

            if _AGENT_SKILL_MAP.exists():
                with Path(_AGENT_SKILL_MAP).open(encoding="utf-8") as f:
                    self._agent_map = json.load(f)
                logger.info(
                    "Loaded agent skill mappings for %s agents",
                    len(self._agent_map.get("agents", {})),
                )
            else:
                logger.warning("Agent skill map not found: %s", _AGENT_SKILL_MAP)

            if _CONTEXT_REGISTRY.exists():
                with Path(_CONTEXT_REGISTRY).open(encoding="utf-8") as f:
                    context_data = json.load(f)
                    self._contexts = {ctx["id"]: ctx for ctx in context_data.get("contexts", [])}
                logger.info("Loaded %s sample contexts", len(self._contexts))
            else:
                logger.warning("Context registry not found: %s", _CONTEXT_REGISTRY)

            self.is_loaded = True

        except Exception as exc:
            logger.error("Failed to load registry: %s", exc)
            raise

    def list_skills(self) -> list[dict[str, Any]]:
        """List all available skills with basic metadata.

        Merges disk-based skills with programmatic ``SkillSpec`` entries,
        deduplicating by skill id.

        Returns:
            List of skill metadata dicts.  Disk entries come first; programmatic
            entries are appended for any skill id not already present.
        """
        if not self.is_loaded:
            self.load()
        disk_skills = self._registry.get("skills", [])
        seen_ids = {s.get("id") for s in disk_skills}
        merged = list(disk_skills)
        for spec in SKILL_REGISTRY.values():
            if spec.skill_id not in seen_ids:
                merged.append(spec.to_dict())
                seen_ids.add(spec.skill_id)
        return merged

    def get_skill(self, skill_id: str) -> dict[str, Any] | None:
        """Get skill metadata by ID.

        Falls back to the programmatic ``SKILL_REGISTRY`` when the disk-based
        registry does not contain the requested skill.

        Args:
            skill_id: The skill identifier to look up.

        Returns:
            Skill metadata dict, or None if not found in either source.
        """
        if not self.is_loaded:
            self.load()
        skills = self._registry.get("skills", [])
        result = next((s for s in skills if s["id"] == skill_id), None)
        if result is not None:
            return result
        spec = SKILL_REGISTRY.get(skill_id)
        if spec:
            return spec.to_dict()
        return None

    def get_skill_manifest(self, skill_id: str) -> dict[str, Any] | None:
        """Get full manifest for a skill.

        Checks the manifest cache first, then tries a per-skill JSON file on
        disk, then falls back to the programmatic ``SKILL_REGISTRY``, and
        finally synthesises a minimal manifest from the disk skill list entry.

        Args:
            skill_id: The skill identifier.

        Returns:
            Manifest dict, or None if no source provides one.
        """
        if skill_id in self._manifests:
            return self._manifests[skill_id]

        manifest_path = _VETINARI_PKG / "skills" / skill_id / "manifest.json"
        if manifest_path.exists():
            with Path(manifest_path).open(encoding="utf-8") as f:
                manifest = json.load(f)
                self._manifests[skill_id] = manifest
                return manifest

        spec = SKILL_REGISTRY.get(skill_id)
        if spec:
            manifest = spec.to_dict()
            self._manifests[skill_id] = manifest
            return manifest

        skill = self.get_skill(skill_id)
        if skill:
            manifest = {
                "skill_id": skill.get("skill_id") or skill.get("id"),
                "name": skill.get("name", skill_id),
                "description": skill.get("description", ""),
                "capabilities": skill.get("capabilities", []),
                "permissions": skill.get("permissions_required", []),
            }
            self._manifests[skill_id] = manifest
            return manifest
        return None

    def get_skill_capabilities(self, skill_id: str) -> list[str]:
        """Get the list of capabilities declared for a skill.

        Args:
            skill_id: The skill identifier.

        Returns:
            List of capability strings, or an empty list when the skill is
            not found.
        """
        skill = self.get_skill(skill_id)
        if skill:
            return skill.get("capabilities", [])
        manifest = self.get_skill_manifest(skill_id)
        if manifest:
            return manifest.get("capabilities", [])
        return []

    def get_skill_permissions(self, skill_id: str) -> list[str]:
        """Get required permissions for a skill.

        Args:
            skill_id: The skill identifier.

        Returns:
            List of required permission strings, or an empty list when the
            skill is not found.
        """
        skill = self.get_skill(skill_id)
        if skill:
            return skill.get("permissions_required", [])
        manifest = self.get_skill_manifest(skill_id)
        if manifest:
            return manifest.get("required_permissions", [])
        return []

    def get_skill_by_capability(self, capability: str) -> list[dict[str, Any]]:
        """Find skills that support a specific capability.

        Searches both the disk-based registry and the programmatic
        ``SKILL_REGISTRY``, deduplicating by skill id.

        Args:
            capability: The capability string to search for.

        Returns:
            List of skill metadata dicts that declare the given capability.
        """
        from vetinari.skills.skill_registry import get_skills_by_capability

        seen_ids: set[str] = set()
        matching: list[dict[str, Any]] = []

        for skill in self.list_skills():
            if capability in skill.get("capabilities", []):
                matching.append(skill)
                seen_ids.add(str(skill.get("id") or skill.get("skill_id")))

        for spec in get_skills_by_capability(capability):
            if spec.skill_id not in seen_ids:
                matching.append(spec.to_dict())
                seen_ids.add(spec.skill_id)

        return matching

    def list_agents(self) -> list[str]:
        """List all registered agent types that have at least one skill.

        Merges disk-based agent ids (from the agent-skill map) with the
        canonical agent type strings in the programmatic ``SKILL_REGISTRY``.

        Returns:
            Sorted list of agent type strings.
        """
        if not self.is_loaded:
            self.load()
        agent_set = set(self._agent_map.get("agents", {}).keys())
        agent_set.update(spec.agent_type for spec in SKILL_REGISTRY.values())
        return sorted(agent_set)

    def get_agent_skills(
        self,
        agent_id: str,
        env: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get skills mapped to a specific agent.

        Checks the disk-based agent-skill map first (with optional
        environment-specific overrides), then falls back to the programmatic
        ``SKILL_REGISTRY`` via ``get_skill_for_agent_type()``.

        Args:
            agent_id: The agent id or canonical agent type (e.g. ``'WORKER'``).
            env: Optional environment name for environment-specific overrides.

        Returns:
            List of skill mapping dicts.  Each dict contains at least
            ``skill_id``; disk entries may include additional metadata.
        """
        from vetinari.skills.skill_registry import get_skill_for_agent_type

        if not self.is_loaded:
            self.load()

        agents = self._agent_map.get("agents", {})
        agent_config = agents.get(agent_id, {})

        if env:
            overrides = self._agent_map.get("environment_overrides", {})
            env_overrides = overrides.get(env, {})
            agent_config = env_overrides.get("agents", {}).get(agent_id, agent_config)

        disk_skills = agent_config.get("default_skills", [])
        if disk_skills:
            return disk_skills

        spec = get_skill_for_agent_type(agent_id)
        if spec:
            return [{"skill_id": spec.skill_id, "source": "programmatic"}]
        return []

    def get_context(self, context_id: str) -> dict[str, Any] | None:
        """Get a sample context by ID.

        Args:
            context_id: The context identifier.

        Returns:
            Context dict, or None if not found.
        """
        if not self.is_loaded:
            self.load()
        return self._contexts.get(context_id)

    def get_contexts_for_skill(self, skill_id: str) -> list[dict[str, Any]]:
        """Get all sample contexts available for a specific skill.

        Args:
            skill_id: The skill identifier.

        Returns:
            List of context dicts whose ``skill_ids`` field includes
            ``skill_id``.
        """
        if not self.is_loaded:
            self.load()
        return [ctx for ctx in self._contexts.values() if skill_id in ctx.get("skill_ids", [])]

    def list_workflows(self) -> dict[str, list[dict[str, str]]]:
        """List predefined skill workflows from the agent-skill map.

        Returns:
            Dict mapping workflow name to ordered list of step dicts.
        """
        if not self.is_loaded:
            self.load()
        return self._agent_map.get("workflows", {})

    def get_compatibility_matrix(self) -> dict[str, Any]:
        """Get the version compatibility matrix from the central registry.

        Returns:
            Dict containing version compatibility information, or an empty
            dict when no matrix is defined.
        """
        if not self.is_loaded:
            self.load()
        return self._registry.get("version_matrix", {})

    def search_skills(self, query: str) -> list[dict[str, Any]]:
        """Search skills by name, description, tags, or capabilities.

        Searches the merged skill list (disk + programmatic) via
        ``list_skills()``.  Results are ranked by match quality so that
        name matches appear before description matches, which appear before
        capability/tag matches.

        Scoring:
          - Name match: 3 points
          - Description match: 2 points
          - Capability or tag match: 1 point each (additive)

        Args:
            query: Case-insensitive search string matched against skill name,
                description, capabilities, and tags.

        Returns:
            List of matching skill metadata dicts, ordered best-match first.
        """
        if not self.is_loaded:
            self.load()

        query_lower = query.lower()
        scored: list[tuple[int, dict[str, Any]]] = []
        for skill in self.list_skills():
            score = 0
            if query_lower in skill.get("name", "").lower():
                score += 3
            if query_lower in skill.get("description", "").lower():
                score += 2
            for cap in skill.get("capabilities", []):
                if query_lower in cap.lower():
                    score += 1
            for tag in skill.get("tags", []):
                if query_lower in tag.lower():
                    score += 1
            if score > 0:
                scored.append((score, skill))

        # Sort descending by score, preserving original order for ties
        scored.sort(key=lambda t: t[0], reverse=True)
        return [skill for _, skill in scored]

    def validate(self) -> dict[str, list[str]]:
        """Validate registry integrity.

        Checks that the central registry file exists, that all listed skills
        have manifests, and that agent mappings reference known skill ids.

        Returns:
            Dict with two keys: ``'errors'`` (list of blocking problems) and
            ``'warnings'`` (list of non-blocking advisories).
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not self.is_loaded:
            self.load()

        if not _CENTRAL_REGISTRY.exists():
            errors.append(f"Central registry not found: {_CENTRAL_REGISTRY}")

        for skill in self.list_skills():
            skill_id = skill.get("id") or skill.get("skill_id", "")
            manifest = self.get_skill_manifest(skill_id)
            if not manifest:
                warnings.append(f"Manifest missing for skill: {skill_id}")

        agents = self._agent_map.get("agents", {})
        for agent_id, config in agents.items():
            for skill_mapping in config.get("default_skills", []):
                ref_skill_id = skill_mapping.get("skill_id")
                if not self.get_skill(ref_skill_id):
                    errors.append(
                        f"Agent '{agent_id}' references unknown skill: {ref_skill_id}",
                    )

        return {"errors": errors, "warnings": warnings}

    def get_skill_summary(self, skill_id: str) -> dict[str, str] | None:
        """Level 1 metadata (~100 tokens): id, name, description, trust_tier.

        Args:
            skill_id: The skill identifier.

        Returns:
            Dict with id, name, description, trust_tier — or None if not found.
        """
        spec = SKILL_REGISTRY.get(skill_id)
        if spec:
            self._loading_levels.setdefault(skill_id, 1)
            return {
                "id": spec.skill_id,
                "name": spec.name,
                "description": spec.description,
                "trust_tier": spec.trust_tier,
            }
        return None

    def activate_skill(self, skill_id: str) -> dict[str, Any] | None:
        """Elevate to Level 2 (full spec) and return complete metadata.

        Args:
            skill_id: The skill identifier.

        Returns:
            Full skill metadata dict, or None if not found.
        """
        result = self.get_skill(skill_id)
        if result is not None:
            self._loading_levels[skill_id] = 2
        return result

    def get_loading_level(self, skill_id: str) -> int:
        """Return the current progressive disclosure level (0/1/2/3).

        Args:
            skill_id: The skill identifier.

        Returns:
            Integer level: 0 = not loaded, 1 = summary, 2 = full spec.
        """
        return self._loading_levels.get(skill_id, 0)

    def list_skill_summaries(self) -> list[dict[str, str]]:
        """Level 1 summaries for all skills — suitable for agent startup context.

        Returns:
            List of summary dicts with id, name, description, trust_tier.
        """
        summaries = []
        for spec in SKILL_REGISTRY.values():
            summaries.append({
                "id": spec.skill_id,
                "name": spec.name,
                "description": spec.description,
                "trust_tier": spec.trust_tier,
            })
            self._loading_levels.setdefault(spec.skill_id, 1)
        return summaries

    def validate_skill_output(self, skill_id: str, output: Any) -> tuple[bool, list[str]]:
        """Run output validators for a skill against produced output.

        Args:
            skill_id: The skill identifier whose validators to run.
            output: The output produced by the skill to validate.

        Returns:
            Tuple of (all_passed, list_of_failure_messages).
        """
        spec = SKILL_REGISTRY.get(skill_id)
        if not spec or not spec.output_validators:
            return True, []
        failures = []
        for validator in spec.output_validators:
            try:
                if not validator(output):
                    failures.append(f"Validator {validator.__name__} rejected output")
            except Exception as exc:
                failures.append(f"Validator raised {type(exc).__name__}: {exc}")
        if failures:
            logger.warning("Skill %s output validation failed: %s", skill_id, "; ".join(failures))
        return len(failures) == 0, failures

    def check_for_changes(self) -> bool:
        """Check skill files for changes; reset is_loaded if the registry was modified.

        Returns:
            True if the registry file has been modified since last load.
        """
        if not _CENTRAL_REGISTRY.exists():
            return False
        try:
            current_mtime = _CENTRAL_REGISTRY.stat().st_mtime
            last_mtime = getattr(self, "_last_registry_mtime", 0.0)
            if current_mtime > last_mtime:
                self.is_loaded = False
                self._last_registry_mtime = current_mtime
                return True
        except OSError:
            logger.warning("Could not stat registry file %s — treating as unchanged", _CENTRAL_REGISTRY)
        return False

    def propose_skill(
        self,
        skill_id: str,
        name: str,
        description: str,
        capabilities: list[str],
        proposed_by: str = "agent",
    ) -> dict[str, Any]:
        """Propose a new skill for human review (enters at T1 trust tier).

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
        if skill_id in SKILL_REGISTRY:
            return {"status": "rejected", "reason": f"Skill '{skill_id}' already exists"}
        proposal: dict[str, Any] = {
            "skill_id": skill_id,
            "name": name,
            "description": description,
            "capabilities": capabilities,
            "trust_tier": "t1_untrusted",
            "proposed_by": proposed_by,
            "status": "pending_review",
        }
        if not hasattr(self, "_pending_proposals"):
            self._pending_proposals: list[dict[str, Any]] = []
        self._pending_proposals.append(proposal)
        logger.info("Skill proposal received: %s (from %s)", skill_id, proposed_by)
        return {"status": "pending_review", "proposal": proposal}

    def verify_trust_elevation(self, skill_id: str) -> dict[str, Any]:
        """Run 4-gate verification chain for trust tier elevation.

        Args:
            skill_id: The skill identifier to verify.

        Returns:
            Dict with ``overall_pass`` (bool), ``gate_results`` (per-gate
            bool dict), and ``current_tier`` string.
        """
        spec = SKILL_REGISTRY.get(skill_id)
        if not spec:
            return {"overall_pass": False, "error": f"Skill {skill_id} not found"}
        gates = {
            "g1_static": bool(spec.skill_id and spec.name and spec.description and spec.modes),
            "g2_semantic": len(spec.capabilities) > 0,
            "g3_behavioral": bool(spec.output_schema),
            "g4_permissions": spec.max_tokens > 0 and spec.timeout_seconds > 0,
        }
        return {
            "overall_pass": all(gates.values()),
            "gate_results": gates,
            "current_tier": spec.trust_tier,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_global_registry: SkillRegistry | None = None
_global_registry_lock = threading.Lock()


def get_registry() -> SkillRegistry:
    """Return the global ``SkillRegistry`` singleton.

    The registry is lazily created and loaded on first access.

    Returns:
        The shared ``SkillRegistry`` instance.
    """
    global _global_registry
    if _global_registry is None:
        with _global_registry_lock:
            if _global_registry is None:
                _global_registry = SkillRegistry()
    return _global_registry
