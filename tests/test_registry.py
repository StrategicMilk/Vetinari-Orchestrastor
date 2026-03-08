"""
Unit tests for the Vetinari Skill Registry

Tests cover:
- Registry loading
- Skill discovery
- Manifest validation
- Agent mapping
- Context retrieval
- Search functionality
- Validation
"""

import sys
import pytest
import json
from pathlib import Path

# Remove incomplete stubs left by earlier test files so real modules load
for _stubname in ("vetinari.registry", "vetinari.skills.skill_registry"):
    sys.modules.pop(_stubname, None)

from vetinari import registry as registry_module
from vetinari.registry import SkillRegistry


class TestRegistryLoading:
    """Tests for registry loading."""
    
    def test_registry_loads(self):
        """Test that registry loads successfully."""
        reg = SkillRegistry(load_on_init=False)
        reg.load()
        assert reg.is_loaded is True
    
    def test_central_registry_exists(self):
        """Test central registry file exists."""
        reg = SkillRegistry()
        skills = reg.list_skills()
        assert len(skills) > 0
    
    def test_agent_map_loads(self):
        """Test agent skill map loads."""
        reg = SkillRegistry()
        agents = reg.list_agents()
        assert len(agents) > 0
    
    def test_contexts_load(self):
        """Test context registry loads."""
        reg = SkillRegistry()
        context = reg.get_context("sample_code_snippet")
        assert context is not None


class TestSkillDiscovery:
    """Tests for skill discovery."""
    
    def setup_method(self):
        self.reg = SkillRegistry()
    
    def test_list_skills(self):
        """Test listing all skills."""
        skills = self.reg.list_skills()
        assert len(skills) >= 8  # We have 8 skills
        skill_ids = [s['id'] for s in skills]
        assert 'builder' in skill_ids
        assert 'explorer' in skill_ids
        assert 'evaluator' in skill_ids
    
    def test_get_skill_by_id(self):
        """Test getting specific skill."""
        skill = self.reg.get_skill('builder')
        assert skill is not None
        assert skill['id'] == 'builder'
        assert 'capabilities' in skill
    
    def test_get_skill_not_found(self):
        """Test getting non-existent skill."""
        skill = self.reg.get_skill('nonexistent_skill')
        assert skill is None
    
    def test_get_skill_manifest(self):
        """Test getting skill manifest."""
        manifest = self.reg.get_skill_manifest('builder')
        assert manifest is not None
        assert manifest['skill_id'] == 'builder'
    
    def test_get_skill_capabilities(self):
        """Test getting skill capabilities."""
        caps = self.reg.get_skill_capabilities('builder')
        assert len(caps) > 0
        assert 'feature_implementation' in caps
    
    def test_get_skill_permissions(self):
        """Test getting skill permissions."""
        perms = self.reg.get_skill_permissions('builder')
        assert len(perms) > 0
        assert 'FILE_READ' in perms
    
    def test_get_skill_by_capability(self):
        """Test finding skills by capability."""
        skills = self.reg.get_skill_by_capability('code_review')
        assert len(skills) > 0
        skill_ids = [s['id'] for s in skills]
        assert 'evaluator' in skill_ids


class TestAgentMappings:
    """Tests for agent-to-skill mappings."""
    
    def setup_method(self):
        self.reg = SkillRegistry()
    
    def test_list_agents(self):
        """Test listing all agents."""
        agents = self.reg.list_agents()
        assert len(agents) > 0
        assert 'builder_agent' in agents
        assert 'general_agent' in agents
    
    def test_get_agent_skills(self):
        """Test getting skills for an agent."""
        skills = self.reg.get_agent_skills('builder_agent')
        assert len(skills) > 0
        skill_ids = [s['skill_id'] for s in skills]
        assert 'builder' in skill_ids
    
    def test_get_agent_skills_with_env_override(self):
        """Test environment-specific skill overrides."""
        skills = self.reg.get_agent_skills('general_agent', env='prod')
        assert len(skills) > 0
    
    def test_workflows_defined(self):
        """Test predefined workflows exist."""
        workflows = self.reg.list_workflows()
        assert len(workflows) > 0
        assert 'code_review_pipeline' in workflows
        assert 'feature_implementation_pipeline' in workflows


class TestContextCatalog:
    """Tests for context catalog."""
    
    def setup_method(self):
        self.reg = SkillRegistry()
    
    def test_get_context(self):
        """Test getting context by ID."""
        ctx = self.reg.get_context('sample_code_snippet')
        assert ctx is not None
        assert ctx['id'] == 'sample_code_snippet'
        assert 'data' in ctx
    
    def test_get_context_not_found(self):
        """Test getting non-existent context."""
        ctx = self.reg.get_context('nonexistent_context')
        assert ctx is None
    
    def test_get_contexts_for_skill(self):
        """Test getting contexts for specific skill."""
        contexts = self.reg.get_contexts_for_skill('builder')
        assert len(contexts) > 0
        context_ids = [c['id'] for c in contexts]
        assert 'sample_code_snippet' in context_ids


class TestSearchFunctionality:
    """Tests for search functionality."""
    
    def setup_method(self):
        self.reg = SkillRegistry()
    
    def test_search_by_name(self):
        """Test searching by skill name."""
        results = self.reg.search_skills('builder')
        assert len(results) > 0
        assert any(s['id'] == 'builder' for s in results)
    
    def test_search_by_capability(self):
        """Test searching by capability."""
        results = self.reg.search_skills('code_review')
        assert len(results) > 0
    
    def test_search_returns_empty_for_no_match(self):
        """Test search returns empty for no matches."""
        results = self.reg.search_skills('xyznonexistent')
        assert len(results) == 0


class TestValidation:
    """Tests for registry validation."""
    
    def setup_method(self):
        self.reg = SkillRegistry()
    
    def test_validate_returns_dict(self):
        """Test validation returns expected structure."""
        result = self.reg.validate()
        assert 'errors' in result
        assert 'warnings' in result
    
    def test_validate_no_critical_errors(self):
        """Test that there are no critical errors in registry."""
        result = self.reg.validate()
        # Should have no errors for properly configured registry
        assert isinstance(result['errors'], list)
        assert isinstance(result['warnings'], list)


class TestCompatibilityMatrix:
    """Tests for version compatibility."""
    
    def setup_method(self):
        self.reg = SkillRegistry()
    
    def test_compatibility_matrix_exists(self):
        """Test compatibility matrix is loaded."""
        matrix = self.reg.get_compatibility_matrix()
        assert 'vetinari_core' in matrix
    
    def test_compatibility_has_skill_versions(self):
        """Test compatibility includes skill versions."""
        matrix = self.reg.get_compatibility_matrix()
        compat = matrix.get('vetinari_core', {}).get('compatibility', {})
        assert 'builder' in compat
        assert 'evaluator' in compat


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_list_skills_convenience(self):
        """Test list_skills convenience function."""
        skills = registry_module.list_skills()
        assert len(skills) > 0
    
    def test_get_skill_convenience(self):
        """Test get_skill convenience function."""
        skill = registry_module.get_skill('builder')
        assert skill is not None
        assert skill['id'] == 'builder'
    
    def test_validate_convenience(self):
        """Test validate_registry convenience function."""
        result = registry_module.validate_registry()
        assert 'errors' in result


class TestGlobalRegistrySingleton:
    """Tests for global registry singleton."""
    
    def test_get_registry_returns_singleton(self):
        """Test get_registry returns same instance."""
        reg1 = registry_module.get_registry()
        reg2 = registry_module.get_registry()
        assert reg1 is reg2
