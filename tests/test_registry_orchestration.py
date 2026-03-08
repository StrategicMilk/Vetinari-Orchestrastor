"""
Integration tests for Vetinari Skill Registry Orchestration Features

Tests cover:
- Workflow template discovery
- Skill dependencies
- Orchestration config
- Cross-skill workflow execution
"""

import sys
import pytest

# Remove incomplete stubs left by earlier test files so real modules load
for _stubname in ("vetinari.registry", "vetinari.skills.skill_registry"):
    sys.modules.pop(_stubname, None)

from vetinari import registry
from vetinari.registry import (
    get_workflow_template,
    list_workflow_templates,
    get_orchestration_config,
    get_skill_dependencies,
    get_skills_for_workflow_stage
)


class TestWorkflowTemplates:
    """Tests for workflow template discovery."""
    
    def test_list_workflow_templates(self):
        """Test listing all workflow templates."""
        templates = list_workflow_templates()
        assert len(templates) > 0
        assert "full_stack_feature" in templates
        assert "security_audit" in templates
        assert "quick_fix" in templates
    
    def test_get_workflow_template(self):
        """Test getting a specific workflow template."""
        template = get_workflow_template("full_stack_feature")
        assert template is not None
        assert "stages" in template
        assert len(template["stages"]) > 0
    
    def test_get_workflow_template_not_found(self):
        """Test getting non-existent template."""
        template = get_workflow_template("nonexistent_template")
        assert template is None
    
    def test_full_stack_feature_template_structure(self):
        """Test full_stack_feature template has correct structure."""
        template = get_workflow_template("full_stack_feature")
        assert template["description"] == "Complete feature implementation from research to deployment-ready code"
        
        stages = template["stages"]
        assert len(stages) == 8
        
        # Check first and last stages
        assert stages[0]["skill"] == "explorer"
        assert stages[0]["capability"] == "grep_search"
        assert stages[-1]["skill"] == "synthesizer"
        assert stages[-1]["capability"] == "report_generation"


class TestOrchestrationConfig:
    """Tests for orchestration configuration."""
    
    def test_get_orchestration_config(self):
        """Test getting orchestration configuration."""
        config = get_orchestration_config()
        assert config is not None
        assert config.get("enabled") is True
        assert "default_workflow_template" in config
        assert "max_workflow_depth" in config
        assert "retry_policy" in config
    
    def test_retry_policy_config(self):
        """Test retry policy configuration."""
        config = get_orchestration_config()
        retry = config.get("retry_policy", {})
        assert retry.get("max_retries") == 3
        assert retry.get("backoff_multiplier") == 2


class TestSkillDependencies:
    """Tests for skill dependency mapping."""
    
    def test_get_skill_dependencies(self):
        """Test getting skill dependencies."""
        deps = get_skill_dependencies("builder")
        assert "explorer" in deps
        assert "librarian" in deps
    
    def test_get_skill_dependencies_none(self):
        """Test getting dependencies for skill with no deps."""
        deps = get_skill_dependencies("explorer")
        assert deps == []
    
    def test_all_skills_have_dependency_info(self):
        """Test all registered skills have dependency info."""
        skills = registry.list_skills()
        for skill in skills:
            deps = get_skill_dependencies(skill["id"])
            assert isinstance(deps, list)


class TestWorkflowStageMatching:
    """Tests for workflow stage purpose matching."""
    
    def test_find_skills_for_purpose(self):
        """Test finding skills matching a workflow purpose."""
        results = get_skills_for_workflow_stage("Analyze")
        assert len(results) > 0
        assert any(r["skill"] == "explorer" for r in results)
    
    def test_find_skills_for_security(self):
        """Test finding skills for security-related purposes."""
        results = get_skills_for_workflow_stage("Security")
        assert len(results) > 0
        assert any(r["skill"] == "evaluator" for r in results)
    
    def test_find_skills_for_report(self):
        """Test finding skills for reporting purposes."""
        results = get_skills_for_workflow_stage("report")
        assert len(results) > 0
        assert any(r["skill"] == "synthesizer" for r in results)


class TestRegistryWorkflowIntegration:
    """Integration tests combining registry and workflow features."""
    
    def test_workflow_references_valid_skills(self):
        """Test all skills referenced in workflows exist in registry."""
        templates = list_workflow_templates()
        all_skills = {s["id"] for s in registry.list_skills()}
        
        for template_name in templates:
            template = get_workflow_template(template_name)
            for stage in template.get("stages", []):
                skill_id = stage.get("skill")
                assert skill_id in all_skills, f"Template {template_name} references unknown skill {skill_id}"
    
    def test_skill_dependencies_exist(self):
        """Test all dependent skills exist in registry."""
        all_skills = {s["id"] for s in registry.list_skills()}
        
        skills = registry.list_skills()
        for skill in skills:
            deps = get_skill_dependencies(skill["id"])
            for dep in deps:
                assert dep in all_skills, f"Skill {skill['id']} depends on unknown skill {dep}"


class TestContextForWorkflows:
    """Tests for context availability for workflows."""
    
    def test_get_context_for_skill(self):
        """Test getting context for specific skill."""
        ctx = registry.get_context("sample_code_snippet")
        assert ctx is not None
        assert "builder" in ctx.get("skill_ids", [])
    
    def test_get_contexts_for_skill(self):
        """Test getting all contexts for a skill."""
        contexts = registry.get_contexts_for_skill("builder")
        assert len(contexts) > 0
    
    def test_workflow_contexts_available(self):
        """Test contexts needed for workflows are available."""
        template = get_workflow_template("full_stack_feature")
        required_contexts = set()
        
        for stage in template.get("stages", []):
            skill_id = stage.get("skill")
            contexts = registry.get_contexts_for_skill(skill_id)
            for ctx in contexts:
                required_contexts.add(ctx["id"])
        
        # Should have at least some contexts available
        assert len(required_contexts) > 0
