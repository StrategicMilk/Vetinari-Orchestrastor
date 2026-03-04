"""
Vetinari Skill Registry

Centralized registry for managing and discovering Vetinari skills.
Provides API surface for skill discovery, manifest loading, and context retrieval.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Registry paths
REGISTRY_DIR = Path(__file__).parent
CENTRAL_REGISTRY = REGISTRY_DIR / "skills_registry.json"
AGENT_SKILL_MAP = REGISTRY_DIR / "config" / "agent_skill_map.json"
CONTEXT_REGISTRY = REGISTRY_DIR / "context_registry.json"


class SkillRegistry:
    """
    Central registry for Vetinari skills.
    
    Provides methods to:
    - List all available skills
    - Get skill metadata and manifests
    - Query agent-to-skill mappings
    - Retrieve sample contexts
    """
    
    def __init__(self, load_on_init: bool = True):
        self._registry: Dict[str, Any] = {}
        self._manifests: Dict[str, Dict] = {}
        self._agent_map: Dict[str, Any] = {}
        self._contexts: Dict[str, Dict] = {}
        self._loaded = False
        
        if load_on_init:
            self.load()
    
    def load(self) -> None:
        """Load all registry data from disk."""
        try:
            # Load central registry
            if CENTRAL_REGISTRY.exists():
                with open(CENTRAL_REGISTRY, 'r', encoding='utf-8') as f:
                    self._registry = json.load(f)
                logger.info(f"Loaded central registry with {len(self._registry.get('skills', []))} skills")
            else:
                logger.warning(f"Central registry not found: {CENTRAL_REGISTRY}")
            
            # Load agent-to-skill mapping
            if AGENT_SKILL_MAP.exists():
                with open(AGENT_SKILL_MAP, 'r', encoding='utf-8') as f:
                    self._agent_map = json.load(f)
                logger.info(f"Loaded agent skill mappings for {len(self._agent_map.get('agents', {}))} agents")
            else:
                logger.warning(f"Agent skill map not found: {AGENT_SKILL_MAP}")
            
            # Load context registry
            if CONTEXT_REGISTRY.exists():
                with open(CONTEXT_REGISTRY, 'r', encoding='utf-8') as f:
                    context_data = json.load(f)
                    self._contexts = {ctx['id']: ctx for ctx in context_data.get('contexts', [])}
                logger.info(f"Loaded {len(self._contexts)} sample contexts")
            else:
                logger.warning(f"Context registry not found: {CONTEXT_REGISTRY}")
            
            self._loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            raise
    
    @property
    def is_loaded(self) -> bool:
        """Check if registry is loaded."""
        return self._loaded
    
    def list_skills(self) -> List[Dict[str, Any]]:
        """List all available skills with basic metadata."""
        if not self._loaded:
            self.load()
        return self._registry.get('skills', [])
    
    def get_skill(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """Get skill metadata by ID."""
        if not self._loaded:
            self.load()
        skills = self._registry.get('skills', [])
        return next((s for s in skills if s['id'] == skill_id), None)
    
    def get_skill_manifest(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """Get full manifest for a skill."""
        if skill_id in self._manifests:
            return self._manifests[skill_id]
        
        # Try to load manifest from disk
        manifest_path = REGISTRY_DIR / "skills" / skill_id / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
                self._manifests[skill_id] = manifest
                return manifest
        
        return None
    
    def get_skill_capabilities(self, skill_id: str) -> List[str]:
        """Get list of capabilities for a skill."""
        skill = self.get_skill(skill_id)
        if skill:
            return skill.get('capabilities', [])
        manifest = self.get_skill_manifest(skill_id)
        if manifest:
            return manifest.get('capabilities', [])
        return []
    
    def get_skill_permissions(self, skill_id: str) -> List[str]:
        """Get required permissions for a skill."""
        skill = self.get_skill(skill_id)
        if skill:
            return skill.get('permissions_required', [])
        manifest = self.get_skill_manifest(skill_id)
        if manifest:
            return manifest.get('required_permissions', [])
        return []
    
    def get_skill_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """Find skills that support a specific capability."""
        skills = self.list_skills()
        matching = []
        for skill in skills:
            if capability in skill.get('capabilities', []):
                matching.append(skill)
        return matching
    
    def list_agents(self) -> List[str]:
        """List all registered agents."""
        if not self._loaded:
            self.load()
        return list(self._agent_map.get('agents', {}).keys())
    
    def get_agent_skills(self, agent_id: str, env: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get skills mapped to a specific agent."""
        if not self._loaded:
            self.load()
        
        agents = self._agent_map.get('agents', {})
        agent_config = agents.get(agent_id, {})
        
        # Check for environment-specific override
        if env:
            overrides = self._agent_map.get('environment_overrides', {})
            env_overrides = overrides.get(env, {})
            agent_config = env_overrides.get('agents', {}).get(agent_id, agent_config)
        
        return agent_config.get('default_skills', [])
    
    def get_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Get sample context by ID."""
        if not self._loaded:
            self.load()
        return self._contexts.get(context_id)
    
    def get_contexts_for_skill(self, skill_id: str) -> List[Dict[str, Any]]:
        """Get all contexts available for a specific skill."""
        if not self._loaded:
            self.load()
        
        contexts = []
        for ctx in self._contexts.values():
            if skill_id in ctx.get('skill_ids', []):
                contexts.append(ctx)
        return contexts
    
    def list_workflows(self) -> Dict[str, List[Dict[str, str]]]:
        """List predefined skill workflows."""
        if not self._loaded:
            self.load()
        return self._agent_map.get('workflows', {})
    
    def get_compatibility_matrix(self) -> Dict[str, Any]:
        """Get version compatibility matrix."""
        if not self._loaded:
            self.load()
        return self._registry.get('version_matrix', {})
    
    def search_skills(self, query: str) -> List[Dict[str, Any]]:
        """Search skills by name, description, or tags."""
        if not self._loaded:
            self.load()
        
        query_lower = query.lower()
        skills = self.list_skills()
        
        results = []
        for skill in skills:
            # Search in name, description, and capabilities
            if (query_lower in skill.get('name', '').lower() or
                query_lower in skill.get('description', '').lower() or
                any(query_lower in cap.lower() for cap in skill.get('capabilities', []))):
                results.append(skill)
        
        return results
    
    def validate(self) -> Dict[str, List[str]]:
        """
        Validate registry integrity.
        
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
            skill_id = skill['id']
            manifest = self.get_skill_manifest(skill_id)
            if not manifest:
                warnings.append(f"Manifest missing for skill: {skill_id}")
        
        # Check agent mappings reference valid skills
        agents = self._agent_map.get('agents', {})
        for agent_id, config in agents.items():
            for skill_mapping in config.get('default_skills', []):
                skill_id = skill_mapping.get('skill_id')
                if not self.get_skill(skill_id):
                    errors.append(f"Agent '{agent_id}' references unknown skill: {skill_id}")
        
        return {'errors': errors, 'warnings': warnings}


# Global registry instance
_global_registry: Optional[SkillRegistry] = None


def get_registry() -> SkillRegistry:
    """Get global registry instance (singleton)."""
    global _global_registry
    if _global_registry is None:
        _global_registry = SkillRegistry()
    return _global_registry


def list_skills() -> List[Dict[str, Any]]:
    """Convenience function to list all skills."""
    return get_registry().list_skills()


def get_skill(skill_id: str) -> Optional[Dict[str, Any]]:
    """Convenience function to get skill by ID."""
    return get_registry().get_skill(skill_id)


def get_skill_manifest(skill_id: str) -> Optional[Dict[str, Any]]:
    """Convenience function to get skill manifest."""
    return get_registry().get_skill_manifest(skill_id)


def get_agent_skills(agent_id: str, env: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convenience function to get skills for an agent."""
    return get_registry().get_agent_skills(agent_id, env)


def get_context(context_id: str) -> Optional[Dict[str, Any]]:
    """Convenience function to get context by ID."""
    return get_registry().get_context(context_id)


def get_contexts_for_skill(skill_id: str) -> List[Dict[str, Any]]:
    """Convenience function to get all contexts for a specific skill."""
    return get_registry().get_contexts_for_skill(skill_id)


def validate_registry() -> Dict[str, List[str]]:
    """Convenience function to validate registry."""
    return get_registry().validate()


# Orchestration-specific methods

def get_workflow_template(template_name: str) -> Optional[Dict[str, Any]]:
    """Get a predefined workflow template by name."""
    reg = get_registry()
    if not reg.is_loaded:
        reg.load()
    templates = reg._registry.get('workflow_templates', {})
    return templates.get(template_name)


def list_workflow_templates() -> List[str]:
    """List all available workflow templates."""
    reg = get_registry()
    if not reg.is_loaded:
        reg.load()
    return list(reg._registry.get('workflow_templates', {}).keys())


def get_orchestration_config() -> Dict[str, Any]:
    """Get orchestration configuration from registry."""
    reg = get_registry()
    if not reg.is_loaded:
        reg.load()
    return reg._registry.get('orchestration_features', {})


def get_skill_dependencies(skill_id: str) -> List[str]:
    """Get skills that the given skill depends on."""
    reg = get_registry()
    if not reg.is_loaded:
        reg.load()
    deps = reg._registry.get('skill_dependencies', {})
    return deps.get(skill_id, [])


def get_skills_for_workflow_stage(stage_purpose: str) -> List[Dict[str, Any]]:
    """Find skills matching a specific workflow stage purpose."""
    reg = get_registry()
    if not reg.is_loaded:
        reg.load()
    
    templates = reg._registry.get('workflow_templates', {})
    matching = []
    
    for template_name, template in templates.items():
        for stage in template.get('stages', []):
            if stage_purpose.lower() in stage.get('purpose', '').lower():
                matching.append({
                    'template': template_name,
                    'skill': stage.get('skill'),
                    'capability': stage.get('capability'),
                    'purpose': stage.get('purpose')
                })
    
    return matching
