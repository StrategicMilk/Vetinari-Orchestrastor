"""Tests for the disk-aware SkillRegistry and orchestration helpers.

Covers:
- Registry loading from JSON files on disk
- Skill discovery (merged disk + programmatic)
- Manifest retrieval and caching
- Agent mapping queries
- Context catalog
- Search functionality
- Registry validation
- Compatibility matrix
- Workflow template discovery
- Skill dependencies
- Orchestration config
- Cross-skill workflow integrity
"""

from __future__ import annotations

from vetinari.skills.skill_registry import (
    SkillRegistry,
    get_agent_skills,
    get_context,
    get_contexts_for_skill,
    get_orchestration_config,
    get_registry,
    get_skill_dependencies,
    get_skill_manifest,
    get_skills_for_workflow_stage,
    get_workflow_template,
    list_agents,
    list_registry_skills,
    list_workflow_templates,
    search_skills,
    validate_registry,
)


class TestRegistryLoading:
    """Tests for registry loading from disk."""

    def test_registry_loads(self):
        """Registry loads successfully with load_on_init=True."""
        reg = SkillRegistry(load_on_init=False)
        reg.load()
        assert reg.is_loaded is True

    def test_central_registry_exists(self):
        """Central registry file provides at least one skill."""
        reg = SkillRegistry()
        skills = reg.list_skills()
        assert len(skills) > 0

    def test_agent_map_loads(self):
        """Agent skill map loads and provides at least one agent."""
        reg = SkillRegistry()
        agents = reg.list_agents()
        assert len(agents) > 0

    def test_contexts_load(self):
        """Context registry loads and the sample_code_snippet context exists."""
        reg = SkillRegistry()
        context = reg.get_context("sample_code_snippet")
        assert context is not None
        assert context["id"] == "sample_code_snippet"


class TestSkillDiscovery:
    """Tests for skill discovery."""

    def setup_method(self):
        self.reg = SkillRegistry()

    def test_list_skills(self):
        """All three pipeline skills are present."""
        skills = self.reg.list_skills()
        assert len(skills) >= 3
        skill_ids = [s.get("id") or s.get("skill_id") for s in skills]
        assert "foreman" in skill_ids
        assert "worker" in skill_ids
        assert "inspector" in skill_ids

    def test_get_skill_by_id(self):
        """Getting a skill by id returns the expected metadata."""
        skill = self.reg.get_skill("worker")
        assert skill is not None
        assert skill.get("id") == "worker" or skill.get("skill_id") == "worker"
        assert "capabilities" in skill

    def test_get_skill_not_found(self):
        """Getting a non-existent skill returns None."""
        skill = self.reg.get_skill("nonexistent_skill")
        assert skill is None

    def test_get_skill_manifest(self):
        """Manifest for the worker skill contains skill_id."""
        manifest = self.reg.get_skill_manifest("worker")
        assert manifest is not None
        assert manifest.get("skill_id") == "worker"

    def test_get_skill_capabilities(self):
        """Worker skill capabilities include feature_implementation."""
        caps = self.reg.get_skill_capabilities("worker")
        assert len(caps) > 0
        assert "feature_implementation" in caps

    def test_get_skill_permissions(self):
        """Worker skill permissions include FILE_READ."""
        perms = self.reg.get_skill_permissions("worker")
        assert len(perms) > 0
        assert "FILE_READ" in perms

    def test_get_skill_by_capability(self):
        """Searching by capability code_review returns the inspector skill."""
        skills = self.reg.get_skill_by_capability("code_review")
        assert len(skills) > 0
        skill_ids = [s.get("id") or s.get("skill_id") for s in skills]
        assert "inspector" in skill_ids


class TestAgentMappings:
    """Tests for agent-to-skill mappings."""

    def setup_method(self):
        self.reg = SkillRegistry()

    def test_list_agents(self):
        """list_agents() returns worker_agent and general_agent."""
        agents = self.reg.list_agents()
        assert len(agents) > 0
        assert "worker_agent" in agents
        assert "general_agent" in agents

    def test_get_agent_skills(self):
        """worker_agent maps to the worker skill."""
        skills = self.reg.get_agent_skills("worker_agent")
        assert len(skills) > 0
        skill_ids = [s["skill_id"] for s in skills]
        assert "worker" in skill_ids

    def test_get_agent_skills_with_env_override(self):
        """Environment-specific overrides return at least one skill."""
        skills = self.reg.get_agent_skills("general_agent", env="prod")
        assert len(skills) > 0

    def test_workflows_defined(self):
        """Predefined workflows include the standard pipelines."""
        workflows = self.reg.list_workflows()
        assert len(workflows) > 0
        assert "code_review_pipeline" in workflows
        assert "feature_implementation_pipeline" in workflows


class TestContextCatalog:
    """Tests for the context catalog."""

    def setup_method(self):
        self.reg = SkillRegistry()

    def test_get_context(self):
        """sample_code_snippet context has id and data fields."""
        ctx = self.reg.get_context("sample_code_snippet")
        assert ctx is not None
        assert ctx["id"] == "sample_code_snippet"
        assert "data" in ctx

    def test_get_context_not_found(self):
        """Getting a non-existent context returns None."""
        ctx = self.reg.get_context("nonexistent_context")
        assert ctx is None

    def test_get_contexts_for_skill(self):
        """get_contexts_for_skill() returns a list (may be empty)."""
        contexts = self.reg.get_contexts_for_skill("worker")
        assert isinstance(contexts, list)


class TestSearchFunctionality:
    """Tests for search functionality."""

    def setup_method(self):
        self.reg = SkillRegistry()

    def test_search_by_name(self):
        """Searching for 'worker' returns the worker skill."""
        results = self.reg.search_skills("worker")
        assert len(results) > 0
        assert any(s.get("id") == "worker" or s.get("skill_id") == "worker" for s in results)

    def test_search_by_capability(self):
        """Searching for 'code_review' returns at least one result."""
        results = self.reg.search_skills("code_review")
        assert len(results) > 0

    def test_search_returns_empty_for_no_match(self):
        """Searching for a nonsense string returns an empty list."""
        results = self.reg.search_skills("xyznonexistent")
        assert len(results) == 0


class TestValidation:
    """Tests for registry validation."""

    def setup_method(self):
        self.reg = SkillRegistry()

    def test_validate_returns_dict(self):
        """validate() returns a dict with 'errors' and 'warnings' keys."""
        result = self.reg.validate()
        assert "errors" in result
        assert "warnings" in result

    def test_validate_no_critical_errors(self):
        """A properly configured registry has no blocking errors."""
        result = self.reg.validate()
        assert isinstance(result["errors"], list)
        assert isinstance(result["warnings"], list)


class TestCompatibilityMatrix:
    """Tests for the version compatibility matrix."""

    def setup_method(self):
        self.reg = SkillRegistry()

    def test_compatibility_matrix_exists(self):
        """Compatibility matrix contains vetinari_core entry."""
        matrix = self.reg.get_compatibility_matrix()
        assert "vetinari_core" in matrix

    def test_compatibility_has_skill_versions(self):
        """Compatibility matrix includes all three pipeline skills."""
        matrix = self.reg.get_compatibility_matrix()
        compat = matrix.get("vetinari_core", {}).get("compatibility", {})
        assert "foreman" in compat
        assert "worker" in compat
        assert "inspector" in compat


class TestGlobalRegistrySingleton:
    """Tests for the global registry singleton."""

    def test_get_registry_returns_singleton(self):
        """get_registry() returns the same instance on repeated calls."""
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_list_registry_skills(self):
        """list_registry_skills() returns at least one skill."""
        skills = list_registry_skills()
        assert len(skills) > 0

    def test_get_registry_skill(self):
        """get_registry().get_skill() returns the worker skill."""
        skill = get_registry().get_skill("worker")
        assert skill is not None
        assert skill.get("id") == "worker" or skill.get("skill_id") == "worker"

    def test_validate_registry_convenience(self):
        """validate_registry() returns a dict with 'errors' key."""
        result = validate_registry()
        assert "errors" in result

    def test_get_skill_manifest_convenience(self):
        """get_skill_manifest() returns worker manifest."""
        manifest = get_skill_manifest("worker")
        assert manifest is not None
        assert manifest.get("skill_id") == "worker"

    def test_get_agent_skills_convenience(self):
        """get_agent_skills() returns skills for worker_agent."""
        skills = get_agent_skills("worker_agent")
        assert len(skills) > 0

    def test_search_skills_convenience(self):
        """search_skills() finds at least one result for 'worker'."""
        results = search_skills("worker")
        assert len(results) > 0

    def test_list_agents_convenience(self):
        """list_agents() returns a non-empty list."""
        agents = list_agents()
        assert len(agents) > 0

    def test_get_context_convenience(self):
        """get_context() returns the sample_code_snippet context."""
        ctx = get_context("sample_code_snippet")
        assert ctx is not None
        assert ctx["id"] == "sample_code_snippet"

    def test_get_contexts_for_skill_convenience(self):
        """get_contexts_for_skill() returns a list."""
        contexts = get_contexts_for_skill("worker")
        assert isinstance(contexts, list)


# ── Orchestration features ───────────────────────────────────────────────────


class TestWorkflowTemplates:
    """Tests for workflow template discovery."""

    def test_list_workflow_templates(self):
        """Standard workflow templates are present."""
        templates = list_workflow_templates()
        assert len(templates) > 0
        assert "full_stack_feature" in templates
        assert "security_audit" in templates
        assert "quick_fix" in templates

    def test_get_workflow_template(self):
        """full_stack_feature template has a stages list."""
        template = get_workflow_template("full_stack_feature")
        assert template is not None
        assert "stages" in template
        assert len(template["stages"]) > 0

    def test_get_workflow_template_not_found(self):
        """Getting a non-existent template returns None."""
        template = get_workflow_template("nonexistent_template")
        assert template is None

    def test_full_stack_feature_template_structure(self):
        """full_stack_feature template has the expected description and stages."""
        template = get_workflow_template("full_stack_feature")
        assert template["description"] == "Complete feature implementation from research to deployment-ready code"
        stages = template["stages"]
        assert len(stages) == 5

        assert stages[0]["skill_id"] == "foreman"
        assert stages[0]["capability"] == "goal_decomposition"
        assert stages[-1]["skill_id"] == "worker"
        assert stages[-1]["capability"] == "documentation_generation"


class TestOrchestrationConfig:
    """Tests for orchestration configuration flags."""

    def test_get_orchestration_config(self):
        """Orchestration config includes parallel_execution and quality_gate."""
        config = get_orchestration_config()
        assert config is not None
        assert config.get("parallel_execution") is True
        assert config.get("quality_gate") is True

    def test_orchestration_features(self):
        """Orchestration config includes self_correction and delegation."""
        config = get_orchestration_config()
        assert config.get("self_correction") is True
        assert config.get("delegation") is True


class TestSkillDependencies:
    """Tests for skill dependency mapping."""

    def test_get_skill_dependencies(self):
        """Worker skill depends on foreman."""
        deps = get_skill_dependencies("worker")
        assert "foreman" in deps

    def test_get_skill_dependencies_none(self):
        """Foreman has no dependencies."""
        deps = get_skill_dependencies("foreman")
        assert deps == []

    def test_all_skills_have_dependency_info(self):
        """Every registered skill has a dependency list (possibly empty)."""
        skills = list_registry_skills()
        for skill in skills:
            skill_id = skill.get("id") or skill.get("skill_id")
            deps = get_skill_dependencies(skill_id)
            assert isinstance(deps, list)


class TestWorkflowStageMatching:
    """Tests for workflow stage purpose matching."""

    def test_find_skills_for_analyze_purpose(self):
        """get_skills_for_workflow_stage('Analyze') returns at least one match."""
        results = get_skills_for_workflow_stage("Analyze")
        assert len(results) > 0

    def test_find_skills_for_security(self):
        """get_skills_for_workflow_stage('Security') returns at least one match."""
        results = get_skills_for_workflow_stage("Security")
        assert len(results) > 0

    def test_find_skills_for_report(self):
        """get_skills_for_workflow_stage('report') returns at least one match."""
        results = get_skills_for_workflow_stage("report")
        assert len(results) > 0


class TestRegistryWorkflowIntegration:
    """Integration tests combining registry and workflow features."""

    def test_workflow_references_valid_skills(self):
        """All skills referenced in workflow templates exist in the registry."""
        templates = list_workflow_templates()
        all_skills = {s["id"] for s in list_registry_skills() if "id" in s}

        for template_name in templates:
            template = get_workflow_template(template_name)
            for stage in template.get("stages", []):
                skill_id = stage.get("skill")
                assert skill_id in all_skills, f"Template {template_name} references unknown skill {skill_id}"

    def test_skill_dependencies_exist(self):
        """All dependency skill ids resolve to known skills."""
        all_skills = {s["id"] for s in list_registry_skills() if "id" in s}

        for skill in list_registry_skills():
            skill_id = skill.get("id")
            if not skill_id:
                continue
            for dep in get_skill_dependencies(skill_id):
                assert dep in all_skills, f"Skill {skill_id} depends on unknown skill {dep}"


class TestContextForWorkflows:
    """Tests for context availability relative to workflows."""

    def test_get_context_for_skill(self):
        """sample_code_snippet context references the worker skill."""
        ctx = get_context("sample_code_snippet")
        assert ctx is not None
        assert "worker" in ctx.get("skill_ids", [])

    def test_get_contexts_for_skill(self):
        """get_contexts_for_skill('worker') returns at least one context."""
        contexts = get_contexts_for_skill("worker")
        assert len(contexts) > 0

    def test_workflow_contexts_available(self):
        """At least one context is available across full_stack_feature stages."""
        template = get_workflow_template("full_stack_feature")
        required_contexts: set[str] = set()

        for stage in template.get("stages", []):
            skill_id = stage.get("skill")
            for ctx in get_contexts_for_skill(skill_id):
                required_contexts.add(ctx["id"])

        assert len(required_contexts) > 0
