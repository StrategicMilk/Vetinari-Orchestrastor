"""
Tests for deferred Phase 7 and Phase 8 items.

Phase 7 deferred:
- 7.1: SkillSpec entries for consolidated agents
- 7.2-7.3: Legacy registry/map updates for consolidated agents
- 7.4: Prompt assembler role defs for consolidated agents
- 7.7-7.8: Planner prompt + affinity table for consolidated agents
- 7.9: AgentGraph capability-based routing via SkillSpec

Phase 8 deferred:
- 8.2: Per-agent output schema validation
- 8.5/8.6: Document and code style constraints
- 8.9: rules.yaml expansion with consolidated agent identifiers
"""

import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Phase 7.1: SkillSpec entries for consolidated agents
# ---------------------------------------------------------------------------


class TestSkillSpecConsolidatedAgents(unittest.TestCase):
    """Every consolidated agent must have a SkillSpec entry."""

    def test_all_consolidated_agents_have_skill_specs(self):
        from vetinari.skills.skill_registry import SKILL_REGISTRY
        required = ["planner", "orchestrator", "researcher", "oracle",
                     "builder", "architect", "quality", "operations"]
        for skill_id in required:
            self.assertIn(skill_id, SKILL_REGISTRY, f"Missing SkillSpec: {skill_id}")

    def test_skill_specs_have_required_fields(self):
        from vetinari.skills.skill_registry import SKILL_REGISTRY
        for skill_id, spec in SKILL_REGISTRY.items():
            self.assertTrue(spec.skill_id, f"{skill_id}: empty skill_id")
            self.assertTrue(spec.name, f"{skill_id}: empty name")
            self.assertTrue(spec.description, f"{skill_id}: empty description")
            self.assertTrue(spec.modes, f"{skill_id}: no modes")
            self.assertTrue(spec.capabilities, f"{skill_id}: no capabilities")
            self.assertTrue(spec.input_schema, f"{skill_id}: no input_schema")
            self.assertTrue(spec.output_schema, f"{skill_id}: no output_schema")

    def test_skill_specs_validate(self):
        from vetinari.skills.skill_registry import validate_all
        errors = validate_all()
        self.assertEqual(errors, [], f"Skill spec validation errors: {errors}")

    def test_legacy_agent_type_mapping_complete(self):
        from vetinari.skills.skill_registry import _LEGACY_AGENT_TO_SKILL
        legacy_types = [
            "PLANNER", "USER_INTERACTION", "CONTEXT_MANAGER",
            "EXPLORER", "RESEARCHER", "LIBRARIAN",
            "ORACLE", "PONDER", "BUILDER",
            "UI_PLANNER", "DATA_ENGINEER", "DEVOPS", "VERSION_CONTROL",
            "EVALUATOR", "SECURITY_AUDITOR", "TEST_AUTOMATION",
            "SYNTHESIZER", "DOCUMENTATION_AGENT", "COST_PLANNER",
            "EXPERIMENTATION_MANAGER", "IMPROVEMENT", "ERROR_RECOVERY",
            "IMAGE_GENERATOR",
        ]
        for lt in legacy_types:
            self.assertIn(lt, _LEGACY_AGENT_TO_SKILL,
                          f"Missing legacy mapping for {lt}")

    def test_get_skill_for_agent_type(self):
        from vetinari.skills.skill_registry import get_skill_for_agent_type
        # Legacy type -> consolidated skill
        spec = get_skill_for_agent_type("EXPLORER")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.skill_id, "researcher")

        spec = get_skill_for_agent_type("SECURITY_AUDITOR")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.skill_id, "quality")

    def test_get_skills_by_capability(self):
        from vetinari.skills.skill_registry import get_skills_by_capability
        specs = get_skills_by_capability("feature_implementation")
        self.assertTrue(len(specs) >= 1)
        self.assertTrue(any(s.skill_id == "builder" for s in specs))

    def test_get_skills_by_tag(self):
        from vetinari.skills.skill_registry import get_skills_by_tag
        specs = get_skills_by_tag("security")
        self.assertTrue(len(specs) >= 1)
        self.assertTrue(any(s.skill_id == "quality" for s in specs))


# ---------------------------------------------------------------------------
# Phase 7.4: Prompt assembler role defs for consolidated agents
# ---------------------------------------------------------------------------


class TestPromptAssemblerConsolidated(unittest.TestCase):
    """Prompt assembler has role definitions for consolidated agents."""

    def test_consolidated_role_defs_exist(self):
        from vetinari.prompts.assembler import _ROLE_DEFS
        consolidated = [
            "ORCHESTRATOR", "CONSOLIDATED_RESEARCHER", "CONSOLIDATED_ORACLE",
            "ARCHITECT", "QUALITY", "OPERATIONS",
        ]
        for agent_type in consolidated:
            self.assertIn(agent_type, _ROLE_DEFS,
                          f"Missing role def for {agent_type}")

    def test_legacy_role_defs_still_exist(self):
        from vetinari.prompts.assembler import _ROLE_DEFS
        # PLANNER and BUILDER remain as direct entries; legacy aliases
        # (EXPLORER, ORACLE, RESEARCHER) are now served by consolidated agents.
        core = ["PLANNER", "BUILDER", "CONSOLIDATED_RESEARCHER", "CONSOLIDATED_ORACLE"]
        for agent_type in core:
            self.assertIn(agent_type, _ROLE_DEFS,
                          f"Missing role def for {agent_type}")

    def test_build_prompt_with_consolidated_agent(self):
        from vetinari.prompts.assembler import get_prompt_assembler
        asm = get_prompt_assembler()
        result = asm.build(
            agent_type="QUALITY",
            task_type="review",
            task_description="Review this code for bugs",
        )
        self.assertIn("system", result)
        self.assertIn("user", result)
        self.assertIn("Quality", result["system"])

    def test_build_prompt_with_unknown_agent_uses_fallback(self):
        from vetinari.prompts.assembler import get_prompt_assembler
        asm = get_prompt_assembler()
        result = asm.build(
            agent_type="UNKNOWN_TYPE",
            task_type="general",
            task_description="Do something",
        )
        self.assertIn("system", result)
        self.assertIn("specialist", result["system"].lower())


# ---------------------------------------------------------------------------
# Phase 7.9: AgentGraph capability-based routing
# ---------------------------------------------------------------------------


class TestAgentGraphCapabilityRouting(unittest.TestCase):
    """AgentGraph can route by capability via SkillSpec."""

    def setUp(self):
        from vetinari.orchestration.agent_graph import AgentGraph
        self.graph = AgentGraph()
        # Don't call initialize() — mock agents instead

    def test_get_agent_by_capability_no_agents(self):
        result = self.graph.get_agent_by_capability("feature_implementation")
        self.assertIsNone(result)

    def test_get_agent_by_capability_with_mock_agent(self):
        from vetinari.types import AgentType
        mock_agent = MagicMock()
        self.graph._agents[AgentType.BUILDER] = mock_agent

        result = self.graph.get_agent_by_capability("feature_implementation")
        self.assertIs(result, mock_agent)

    def test_get_agent_by_capability_unknown(self):
        result = self.graph.get_agent_by_capability("teleportation")
        self.assertIsNone(result)

    def test_get_skill_spec(self):
        from vetinari.types import AgentType
        spec = self.graph.get_skill_spec(AgentType.BUILDER)
        self.assertIsNotNone(spec)
        self.assertEqual(spec.skill_id, "builder")

    def test_get_skill_spec_legacy_type(self):
        from vetinari.types import AgentType
        spec = self.graph.get_skill_spec(AgentType.EXPLORER)
        self.assertIsNotNone(spec)
        self.assertEqual(spec.skill_id, "researcher")

    def test_get_agents_for_task_type(self):
        from vetinari.types import AgentType
        mock_agent = MagicMock()
        self.graph._agents[AgentType.BUILDER] = mock_agent

        agents = self.graph.get_agents_for_task_type("feature_implementation")
        self.assertIn(AgentType.BUILDER, agents)

    def test_get_agents_for_task_type_empty(self):
        agents = self.graph.get_agents_for_task_type("nonexistent_mode")
        self.assertEqual(agents, [])


# ---------------------------------------------------------------------------
# Phase 8.2: Output schema validation
# ---------------------------------------------------------------------------


class TestOutputSchemaValidation(unittest.TestCase):
    """AgentGraph validates output against SkillSpec output_schema."""

    def setUp(self):
        from vetinari.orchestration.agent_graph import AgentGraph
        self.graph = AgentGraph()

    def test_valid_output_no_issues(self):
        from vetinari.types import AgentType
        output = {"code": "print('hi')", "language": "python"}
        issues = self.graph._validate_output_schema(AgentType.BUILDER, output)
        self.assertEqual(issues, [])

    def test_missing_required_field(self):
        from vetinari.types import AgentType
        output = {"language": "python"}  # missing "code"
        issues = self.graph._validate_output_schema(AgentType.BUILDER, output)
        self.assertTrue(any("code" in i for i in issues))

    def test_wrong_type_detected(self):
        from vetinari.types import AgentType
        output = {"code": 42, "language": "python"}  # code should be string
        issues = self.graph._validate_output_schema(AgentType.BUILDER, output)
        self.assertTrue(any("type" in i.lower() for i in issues))

    def test_non_dict_output_skipped(self):
        from vetinari.types import AgentType
        issues = self.graph._validate_output_schema(AgentType.BUILDER, "just a string")
        self.assertEqual(issues, [])

    def test_unknown_agent_type_no_issues(self):
        from vetinari.types import AgentType
        # IMAGE_GENERATOR may not have a SkillSpec
        output = {"whatever": True}
        issues = self.graph._validate_output_schema(AgentType.IMAGE_GENERATOR, output)
        # Should not crash, may return issues or empty
        self.assertIsInstance(issues, list)


# ---------------------------------------------------------------------------
# Phase 8.5/8.6: Style constraints
# ---------------------------------------------------------------------------


class TestStyleConstraints(unittest.TestCase):
    """Document and code style constraints."""

    def test_code_style_detects_todo(self):
        from vetinari.constraints.style import validate_output_style
        issues = validate_output_style("x = 1  # TODO fix later", "code")
        rule_ids = [i["rule_id"] for i in issues]
        self.assertIn("code-no-todo", rule_ids)

    def test_code_style_detects_bare_except(self):
        from vetinari.constraints.style import validate_output_style
        issues = validate_output_style("try:\n    pass\nexcept:\n    pass", "code")
        rule_ids = [i["rule_id"] for i in issues]
        self.assertIn("code-no-bare-except", rule_ids)

    def test_code_style_detects_hardcoded_secret(self):
        from vetinari.constraints.style import validate_output_style
        issues = validate_output_style('api_key = "sk-1234567890abcdef"', "code")
        rule_ids = [i["rule_id"] for i in issues]
        self.assertIn("code-no-hardcoded-secrets", rule_ids)

    def test_doc_style_detects_placeholder(self):
        from vetinari.constraints.style import validate_output_style
        issues = validate_output_style("This section is TBD.", "documentation")
        rule_ids = [i["rule_id"] for i in issues]
        self.assertIn("doc-no-placeholder", rule_ids)

    def test_clean_code_passes(self):
        from vetinari.constraints.style import validate_output_style
        issues = validate_output_style("def add(a, b):\n    return a + b", "code")
        warning_issues = [i for i in issues if i["severity"] == "warning"]
        self.assertEqual(warning_issues, [])

    def test_style_domain_mapping(self):
        from vetinari.constraints.style import get_style_domain
        self.assertEqual(get_style_domain("BUILDER"), "code")
        self.assertEqual(get_style_domain("PLANNER"), "documentation")
        self.assertEqual(get_style_domain("OPERATIONS", "creative_writing"), "creative")
        self.assertEqual(get_style_domain("QUALITY", "security_audit"), "code")

    def test_get_style_rules(self):
        from vetinari.constraints.style import get_style_rules
        code_rules = get_style_rules("code")
        self.assertIsNotNone(code_rules)
        self.assertEqual(code_rules.domain, "code")
        self.assertTrue(len(code_rules.rules) > 0)

    def test_creative_style_no_line_limit(self):
        from vetinari.constraints.style import get_style_rules
        creative = get_style_rules("creative")
        self.assertEqual(creative.max_line_length, 0)

    def test_forbidden_phrases(self):
        from vetinari.constraints.style import validate_output_style
        issues = validate_output_style("This is a HACK to fix it", "code")
        rule_ids = [i["rule_id"] for i in issues]
        self.assertIn("forbidden-phrase", rule_ids)

    def test_empty_text_no_issues(self):
        from vetinari.constraints.style import validate_output_style
        issues = validate_output_style("", "code")
        self.assertEqual(issues, [])

    def test_style_constraint_exports(self):
        from vetinari.constraints import (
            STYLE_CONSTRAINTS, StyleConstraint, StyleRule,
            get_style_domain, get_style_rules, validate_output_style,
        )
        self.assertIsNotNone(STYLE_CONSTRAINTS)
        self.assertTrue(callable(validate_output_style))


# ---------------------------------------------------------------------------
# SkillSpec serialization
# ---------------------------------------------------------------------------


class TestSkillSpecSerialization(unittest.TestCase):
    """SkillSpec to_dict / from_dict round-trip."""

    def test_round_trip(self):
        from vetinari.skills.skill_spec import SkillSpec
        spec = SkillSpec(
            skill_id="test", name="Test", description="Test skill",
            modes=["a", "b"], capabilities=["c"],
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )
        d = spec.to_dict()
        restored = SkillSpec.from_dict(d)
        self.assertEqual(restored.skill_id, "test")
        self.assertEqual(restored.modes, ["a", "b"])
        self.assertEqual(restored.capabilities, ["c"])


# ---------------------------------------------------------------------------
# Phase 7.2-7.3: Legacy registry fallback to programmatic SkillSpec
# ---------------------------------------------------------------------------


class TestLegacyRegistryFallback(unittest.TestCase):
    """Legacy SkillRegistry falls back to programmatic SkillSpec."""

    def setUp(self):
        from vetinari.registry import SkillRegistry
        self.reg = SkillRegistry(load_on_init=False)
        # Force loaded state without disk files (empty registry)
        self.reg._loaded = True

    def test_get_skill_fallback_to_programmatic(self):
        """get_skill() returns SkillSpec data when disk has no match."""
        result = self.reg.get_skill("builder")
        self.assertIsNotNone(result)
        self.assertEqual(result["skill_id"], "builder")

    def test_get_skill_manifest_fallback(self):
        """get_skill_manifest() returns SkillSpec data as manifest."""
        result = self.reg.get_skill_manifest("quality")
        self.assertIsNotNone(result)
        self.assertEqual(result["skill_id"], "quality")
        self.assertIn("capabilities", result)

    def test_get_skill_manifest_caches(self):
        """Programmatic manifest is cached after first lookup."""
        m1 = self.reg.get_skill_manifest("researcher")
        m2 = self.reg.get_skill_manifest("researcher")
        self.assertIs(m1, m2)

    def test_get_skill_by_capability_includes_programmatic(self):
        """get_skill_by_capability() finds programmatic specs."""
        results = self.reg.get_skill_by_capability("feature_implementation")
        skill_ids = [r.get("skill_id") or r.get("id") for r in results]
        self.assertIn("builder", skill_ids)

    def test_list_skills_includes_programmatic(self):
        """list_skills() merges programmatic specs."""
        skills = self.reg.list_skills()
        skill_ids = [s.get("skill_id") or s.get("id") for s in skills]
        for expected in ["planner", "orchestrator", "researcher", "oracle",
                         "builder", "architect", "quality", "operations"]:
            self.assertIn(expected, skill_ids, f"Missing: {expected}")

    def test_list_agents_includes_consolidated(self):
        """list_agents() includes consolidated agent types."""
        agents = self.reg.list_agents()
        for agent_type in ["PLANNER", "ORCHESTRATOR", "RESEARCHER", "ORACLE",
                           "BUILDER", "ARCHITECT", "QUALITY", "OPERATIONS"]:
            self.assertIn(agent_type, agents, f"Missing agent: {agent_type}")

    def test_get_agent_skills_fallback(self):
        """get_agent_skills() returns programmatic skill for consolidated agents."""
        skills = self.reg.get_agent_skills("BUILDER")
        self.assertTrue(len(skills) >= 1)
        self.assertEqual(skills[0]["skill_id"], "builder")

    def test_get_agent_skills_legacy_fallback(self):
        """get_agent_skills() maps legacy agent types via SkillSpec."""
        skills = self.reg.get_agent_skills("EXPLORER")
        self.assertTrue(len(skills) >= 1)
        self.assertEqual(skills[0]["skill_id"], "researcher")

    def test_search_skills_finds_programmatic(self):
        """search_skills() searches programmatic specs too."""
        results = self.reg.search_skills("security")
        skill_ids = [r.get("skill_id") or r.get("id") for r in results]
        self.assertIn("quality", skill_ids)

    def test_search_skills_by_tag(self):
        """search_skills() can match on tags from programmatic specs."""
        results = self.reg.search_skills("architecture")
        skill_ids = [r.get("skill_id") or r.get("id") for r in results]
        self.assertIn("architect", skill_ids)

    def test_get_skill_unknown_returns_none(self):
        """get_skill() returns None for truly unknown skills."""
        result = self.reg.get_skill("nonexistent_skill_xyz")
        self.assertIsNone(result)

    def test_get_skill_capabilities_fallback(self):
        """get_skill_capabilities() works via fallback."""
        caps = self.reg.get_skill_capabilities("builder")
        self.assertIn("feature_implementation", caps)


# ---------------------------------------------------------------------------
# Phase 7.7-7.8: Planner prompt + affinity table
# ---------------------------------------------------------------------------


class TestPlannerConsolidatedAgents(unittest.TestCase):
    """Planner prompt references consolidated agents and affinity table."""

    def test_system_prompt_has_consolidated_agents(self):
        from vetinari.agents.planner_agent import PlannerAgent
        planner = PlannerAgent()
        prompt = planner.get_system_prompt()
        for agent in ["PLANNER", "RESEARCHER", "ORACLE",
                       "BUILDER", "QUALITY", "OPERATIONS"]:
            self.assertIn(agent, prompt,
                          f"Consolidated agent {agent} missing from planner prompt")

    def test_system_prompt_has_affinity_table(self):
        from vetinari.agents.planner_agent import PlannerAgent
        planner = PlannerAgent()
        prompt = planner.get_system_prompt()
        self.assertIn("Affinity Table", prompt)
        # Check key affinity mappings
        self.assertIn("code/implement/build", prompt)
        self.assertIn("research/explore/discover", prompt)
        self.assertIn("review/test/security", prompt)

    def test_system_prompt_lists_active_agents(self):
        """System prompt lists the active consolidated agents."""
        from vetinari.agents.planner_agent import PlannerAgent
        planner = PlannerAgent()
        prompt = planner.get_system_prompt()
        self.assertIn("Active agents", prompt)
        self.assertIn("PLANNER", prompt)
        self.assertIn("BUILDER", prompt)

    def test_decompose_available_agents_includes_consolidated(self):
        """_decompose_goal_llm available_agents includes consolidated types."""
        from vetinari.agents.planner_agent import PlannerAgent
        import inspect
        source = inspect.getsource(PlannerAgent._decompose_goal_llm)
        for agent in ["PLANNER", "CONSOLIDATED_RESEARCHER", "CONSOLIDATED_ORACLE",
                       "BUILDER", "QUALITY", "OPERATIONS"]:
            self.assertIn(f'"{agent}"', source,
                          f"{agent} missing from _decompose_goal_llm available_agents")

    def test_planner_prefers_consolidated_agents_rule(self):
        """System prompt lists only the 6 consolidated agents."""
        from vetinari.agents.planner_agent import PlannerAgent
        planner = PlannerAgent()
        prompt = planner.get_system_prompt()
        self.assertIn("6 consolidated", prompt)


# ---------------------------------------------------------------------------
# Phase 8.9: rules.yaml expansion
# ---------------------------------------------------------------------------


class TestRulesYamlConsolidated(unittest.TestCase):
    """rules.yaml includes consolidated agent identifiers."""

    def setUp(self):
        import yaml
        from pathlib import Path
        rules_path = Path(__file__).parent.parent / "rules.yaml"
        with open(rules_path, 'r', encoding='utf-8') as f:
            self.rules = yaml.safe_load(f)

    def test_agents_section_exists(self):
        self.assertIn("agents", self.rules)
        self.assertIsInstance(self.rules["agents"], dict)

    def test_all_consolidated_agents_present(self):
        agents = self.rules["agents"]
        for agent in ["PLANNER", "ORCHESTRATOR", "RESEARCHER", "ORACLE",
                       "BUILDER", "ARCHITECT", "QUALITY", "OPERATIONS"]:
            self.assertIn(agent, agents,
                          f"Missing consolidated agent in rules.yaml: {agent}")

    def test_agent_rules_are_lists(self):
        agents = self.rules["agents"]
        for agent_type, rules in agents.items():
            self.assertIsInstance(rules, list,
                                 f"{agent_type} rules should be a list")
            self.assertTrue(len(rules) >= 2,
                            f"{agent_type} should have at least 2 rules")

    def test_builder_has_verification_score(self):
        agents = self.rules["agents"]
        builder_rules = " ".join(agents["BUILDER"])
        self.assertIn("verification score", builder_rules.lower())

    def test_quality_highest_bar(self):
        agents = self.rules["agents"]
        quality_rules = " ".join(agents["QUALITY"])
        self.assertIn("0.7", quality_rules)

    def test_operations_extended_budget(self):
        agents = self.rules["agents"]
        ops_rules = " ".join(agents["OPERATIONS"])
        self.assertIn("16384", ops_rules)

    def test_researcher_replaces_legacy(self):
        agents = self.rules["agents"]
        researcher_rules = " ".join(agents["RESEARCHER"])
        self.assertIn("EXPLORER", researcher_rules)
        self.assertIn("LIBRARIAN", researcher_rules)

    def test_global_section_preserved(self):
        self.assertIn("global", self.rules)
        self.assertIn("global_system_prompt", self.rules)
        self.assertIn("models", self.rules)


# ---------------------------------------------------------------------------
# Phase 7.9H: Permission enforcement in AgentGraph + Blackboard
# ---------------------------------------------------------------------------


class TestPermissionEnforcementAgentGraph(unittest.TestCase):
    """7.9H: Permission enforcement in _execute_task_node and Blackboard.claim."""

    def test_agent_graph_has_permission_check_in_execute(self):
        """_execute_task_node source contains permission enforcement."""
        import inspect
        from vetinari.orchestration.agent_graph import AgentGraph
        src = inspect.getsource(AgentGraph._execute_task_node)
        self.assertIn("enforce_permission", src)
        self.assertIn("MODEL_INFERENCE", src)

    def test_agent_graph_returns_failure_on_permission_denied(self):
        """When permission is denied, _execute_task_node returns failure."""
        from vetinari.orchestration.agent_graph import AgentGraph, TaskNode
        from vetinari.agents.contracts import AgentType, Task, AgentResult

        graph = AgentGraph()
        mock_agent = MagicMock()
        mock_agent.execute.return_value = AgentResult(success=True, output="ok")
        mock_agent.verify.return_value = MagicMock(passed=True, issues=[])
        graph._agents[AgentType.BUILDER] = mock_agent

        task = Task(id="t1", description="Test", assigned_agent=AgentType.BUILDER)
        node = TaskNode(task=task)

        # Patch enforce_permission to raise PermissionError
        with patch(
            "vetinari.execution_context.get_context_manager"
        ) as mock_ctx:
            mock_mgr = MagicMock()
            mock_mgr.enforce_permission.side_effect = PermissionError("denied")
            mock_ctx.return_value = mock_mgr

            result = graph._execute_task_node(node)
            self.assertFalse(result.success)
            self.assertIn("Permission denied", result.errors[0])

    def test_agent_graph_allows_when_no_context_manager(self):
        """When context manager is not configured, execution proceeds normally."""
        from vetinari.orchestration.agent_graph import AgentGraph, TaskNode
        from vetinari.agents.contracts import AgentType, Task, AgentResult

        graph = AgentGraph()
        mock_agent = MagicMock()
        mock_agent.execute.return_value = AgentResult(success=True, output="ok")
        mock_agent.verify.return_value = MagicMock(passed=True, issues=[], score=1.0)
        mock_agent._incorporate_prior_results = MagicMock()
        graph._agents[AgentType.BUILDER] = mock_agent

        task = Task(id="t1", description="Test", assigned_agent=AgentType.BUILDER)
        node = TaskNode(task=task)

        with patch(
            "vetinari.execution_context.get_context_manager",
            side_effect=ImportError("not available"),
        ):
            result = graph._execute_task_node(node)
            self.assertTrue(result.success)

    def test_blackboard_claim_checks_permission(self):
        """Blackboard.claim checks MODEL_INFERENCE permission."""
        import inspect
        from vetinari.blackboard import Blackboard
        src = inspect.getsource(Blackboard.claim)
        self.assertIn("MODEL_INFERENCE", src)
        self.assertIn("check_permission", src)

    def test_blackboard_claim_denied_returns_none(self):
        """When permission is denied, claim returns None."""
        from vetinari.blackboard import Blackboard, EntryState

        board = Blackboard()
        entry_id = board.post("test", "code_search", "BUILDER")

        with patch("vetinari.execution_context.get_context_manager") as mock_ctx:
            mock_mgr = MagicMock()
            mock_mgr.check_permission.return_value = False
            mock_ctx.return_value = mock_mgr

            result = board.claim(entry_id, "EXPLORER")
            self.assertIsNone(result)

    def test_blackboard_claim_allowed_succeeds(self):
        """When permission is allowed, claim succeeds."""
        from vetinari.blackboard import Blackboard, EntryState

        board = Blackboard()
        entry_id = board.post("test", "code_search", "BUILDER")

        with patch("vetinari.execution_context.get_context_manager") as mock_ctx:
            mock_mgr = MagicMock()
            mock_mgr.check_permission.return_value = True
            mock_ctx.return_value = mock_mgr

            result = board.claim(entry_id, "EXPLORER")
            self.assertIsNotNone(result)
            self.assertEqual(result.claimed_by, "EXPLORER")
            self.assertEqual(result.state, EntryState.CLAIMED)


# ---------------------------------------------------------------------------
# Phase 7.9I: Dependency results incorporation
# ---------------------------------------------------------------------------


class TestDependencyResultsIncorporation(unittest.TestCase):
    """7.9I: BaseAgent._incorporate_prior_results and AgentGraph wiring."""

    def test_base_agent_has_incorporate_method(self):
        """BaseAgent has _incorporate_prior_results method."""
        from vetinari.agents.base_agent import BaseAgent
        self.assertTrue(hasattr(BaseAgent, "_incorporate_prior_results"))

    def test_incorporate_extracts_dependency_results(self):
        """_incorporate_prior_results extracts context.dependency_results."""
        from vetinari.agents.planner_agent import PlannerAgent
        from vetinari.agents.contracts import AgentTask, AgentType

        agent = PlannerAgent()
        task = AgentTask(
            task_id="t2",
            agent_type=AgentType.PLANNER,
            description="Test",
            prompt="Test",
            context={
                "dependency_results": {
                    "t1": {"success": True, "output_summary": "spec ready"},
                }
            },
        )
        results = agent._incorporate_prior_results(task)
        self.assertIn("t1", results)
        self.assertTrue(results["t1"]["success"])

    def test_incorporate_returns_empty_when_no_deps(self):
        """Returns empty dict when no dependency_results in context."""
        from vetinari.agents.planner_agent import PlannerAgent
        from vetinari.agents.contracts import AgentTask, AgentType

        agent = PlannerAgent()
        task = AgentTask(
            task_id="t1",
            agent_type=AgentType.PLANNER,
            description="Test",
            prompt="Test",
            context={},
        )
        results = agent._incorporate_prior_results(task)
        self.assertEqual(results, {})

    def test_agent_graph_calls_incorporate(self):
        """AgentGraph._execute_task_node calls _incorporate_prior_results."""
        import inspect
        from vetinari.orchestration.agent_graph import AgentGraph
        src = inspect.getsource(AgentGraph._execute_task_node)
        self.assertIn("_incorporate_prior_results", src)


# ---------------------------------------------------------------------------
# Phase 7.9J: Dynamic graph modification (inject_task)
# ---------------------------------------------------------------------------


class TestInjectTask(unittest.TestCase):
    """7.9J: AgentGraph.inject_task for mid-execution DAG changes."""

    def _make_graph_with_plan(self):
        from vetinari.orchestration.agent_graph import AgentGraph, TaskNode, ExecutionPlan
        from vetinari.agents.contracts import AgentType, Plan, Task

        graph = AgentGraph()
        plan = Plan.create_new("Test goal")
        t1 = Task(id="t1", description="First", assigned_agent=AgentType.EXPLORER)
        t2 = Task(id="t2", description="Second", assigned_agent=AgentType.BUILDER,
                   dependencies=["t1"])

        exec_plan = ExecutionPlan(plan_id=plan.plan_id, original_plan=plan)
        exec_plan.nodes["t1"] = TaskNode(task=t1, dependents={"t2"})
        exec_plan.nodes["t2"] = TaskNode(task=t2, dependencies={"t1"})
        exec_plan.execution_order = ["t1", "t2"]

        graph._execution_plans[plan.plan_id] = exec_plan
        return graph, plan, exec_plan

    def test_inject_task_success(self):
        """inject_task inserts a new task between existing ones."""
        from vetinari.agents.contracts import AgentType, Task

        graph, plan, exec_plan = self._make_graph_with_plan()
        new_task = Task(id="t_review", description="Review",
                        assigned_agent=AgentType.QUALITY)

        result = graph.inject_task(plan.plan_id, new_task, "t1")
        self.assertTrue(result)
        self.assertIn("t_review", exec_plan.nodes)

        # t_review should depend on t1
        self.assertIn("t1", exec_plan.nodes["t_review"].dependencies)

        # t2 should now depend on t_review (not t1)
        self.assertIn("t_review", exec_plan.nodes["t2"].dependencies)
        self.assertNotIn("t1", exec_plan.nodes["t2"].dependencies)

    def test_inject_task_nonexistent_plan(self):
        """inject_task returns False for unknown plan_id."""
        from vetinari.orchestration.agent_graph import AgentGraph
        from vetinari.agents.contracts import AgentType, Task

        graph = AgentGraph()
        task = Task(id="tx", description="X", assigned_agent=AgentType.BUILDER)
        self.assertFalse(graph.inject_task("no_such_plan", task, "t1"))

    def test_inject_task_nonexistent_after(self):
        """inject_task returns False when after_task_id not in plan."""
        from vetinari.agents.contracts import AgentType, Task

        graph, plan, _ = self._make_graph_with_plan()
        task = Task(id="tx", description="X", assigned_agent=AgentType.BUILDER)
        self.assertFalse(graph.inject_task(plan.plan_id, task, "nonexistent"))

    def test_inject_task_duplicate_id(self):
        """inject_task returns False when task.id already exists."""
        from vetinari.agents.contracts import AgentType, Task

        graph, plan, _ = self._make_graph_with_plan()
        task = Task(id="t1", description="Dup", assigned_agent=AgentType.BUILDER)
        self.assertFalse(graph.inject_task(plan.plan_id, task, "t1"))

    def test_inject_task_updates_execution_order(self):
        """inject_task rebuilds execution_order with new task."""
        from vetinari.agents.contracts import AgentType, Task

        graph, plan, exec_plan = self._make_graph_with_plan()
        new_task = Task(id="t_mid", description="Mid",
                        assigned_agent=AgentType.QUALITY)
        graph.inject_task(plan.plan_id, new_task, "t1")

        # t_mid should appear between t1 and t2
        order = exec_plan.execution_order
        self.assertIn("t_mid", order)
        self.assertLess(order.index("t1"), order.index("t_mid"))
        self.assertLess(order.index("t_mid"), order.index("t2"))


# ---------------------------------------------------------------------------
# Phase 7.9K: Maker-checker pattern
# ---------------------------------------------------------------------------


class TestMakerChecker(unittest.TestCase):
    """7.9K: QUALITY reviews BUILDER output with feedback loop."""

    def test_maker_checker_approval(self):
        """QUALITY approves BUILDER output on first attempt."""
        from vetinari.orchestration.agent_graph import AgentGraph
        from vetinari.agents.contracts import AgentType, Task, AgentResult

        graph = AgentGraph()
        mock_quality = MagicMock()
        mock_quality.execute.return_value = AgentResult(success=True, output={"score": 0.9})
        mock_quality.verify.return_value = MagicMock(passed=True, issues=[], score=0.9)
        graph._agents[AgentType.QUALITY] = mock_quality
        graph._agents[AgentType.BUILDER] = MagicMock()

        task = Task(id="t1", description="Build X", assigned_agent=AgentType.BUILDER)
        result = AgentResult(success=True, output="code here")

        final = graph._apply_maker_checker(task, result)
        self.assertTrue(final.success)
        self.assertTrue(final.metadata["maker_checker"]["approved"])
        self.assertEqual(final.metadata["maker_checker"]["iterations"], 1)

    def test_maker_checker_rejection_retries(self):
        """QUALITY rejects, BUILDER gets feedback and retries."""
        from vetinari.orchestration.agent_graph import AgentGraph
        from vetinari.agents.contracts import AgentType, Task, AgentResult

        graph = AgentGraph()

        # QUALITY rejects first, approves second
        mock_quality = MagicMock()
        reject_verify = MagicMock(passed=False, issues=[{"message": "missing tests"}], score=0.3)
        approve_verify = MagicMock(passed=True, issues=[], score=0.9)
        mock_quality.execute.return_value = AgentResult(success=True, output={"score": 0.9})
        mock_quality.verify.side_effect = [reject_verify, approve_verify]
        graph._agents[AgentType.QUALITY] = mock_quality

        mock_builder = MagicMock()
        mock_builder.execute.return_value = AgentResult(success=True, output="fixed code")
        graph._agents[AgentType.BUILDER] = mock_builder

        task = Task(id="t1", description="Build X", assigned_agent=AgentType.BUILDER)
        result = AgentResult(success=True, output="initial code")

        final = graph._apply_maker_checker(task, result)
        self.assertTrue(final.metadata["maker_checker"]["approved"])
        self.assertEqual(final.metadata["maker_checker"]["iterations"], 2)
        # BUILDER should have been called with feedback
        builder_call = mock_builder.execute.call_args
        self.assertIn("MAKER-CHECKER FEEDBACK", builder_call[0][0].description)

    def test_maker_checker_max_iterations(self):
        """After max iterations without approval, returns not-approved result."""
        from vetinari.orchestration.agent_graph import AgentGraph
        from vetinari.agents.contracts import AgentType, Task, AgentResult

        graph = AgentGraph()
        mock_quality = MagicMock()
        mock_quality.execute.return_value = AgentResult(success=True, output={})
        mock_quality.verify.return_value = MagicMock(
            passed=False, issues=[{"message": "still bad"}], score=0.2
        )
        graph._agents[AgentType.QUALITY] = mock_quality

        mock_builder = MagicMock()
        mock_builder.execute.return_value = AgentResult(success=True, output="code")
        graph._agents[AgentType.BUILDER] = mock_builder

        task = Task(id="t1", description="Build", assigned_agent=AgentType.BUILDER)
        result = AgentResult(success=True, output="code")

        final = graph._apply_maker_checker(task, result)
        self.assertFalse(final.metadata["maker_checker"]["approved"])
        self.assertEqual(
            final.metadata["maker_checker"]["iterations"],
            AgentGraph._MAKER_CHECKER_MAX_ITERATIONS,
        )

    def test_maker_checker_triggered_for_builder(self):
        """_execute_task_node triggers maker-checker for BUILDER when QUALITY registered."""
        import inspect
        from vetinari.orchestration.agent_graph import AgentGraph
        src = inspect.getsource(AgentGraph._execute_task_node)
        self.assertIn("_apply_maker_checker", src)
        self.assertIn("_quality_reviewed_agents", src)

    def test_maker_checker_skips_without_quality(self):
        """Without QUALITY agent, maker-checker returns original result."""
        from vetinari.orchestration.agent_graph import AgentGraph
        from vetinari.agents.contracts import AgentType, Task, AgentResult

        graph = AgentGraph()
        graph._agents[AgentType.BUILDER] = MagicMock()
        # No QUALITY agent registered

        task = Task(id="t1", description="Build", assigned_agent=AgentType.BUILDER)
        result = AgentResult(success=True, output="code")

        final = graph._apply_maker_checker(task, result)
        # Should return original result unchanged
        self.assertEqual(final.output, "code")


# ---------------------------------------------------------------------------
# Phase 7.9A: AgentGraph wired into TwoLayerOrchestrator
# ---------------------------------------------------------------------------


class TestAgentGraphInTwoLayer(unittest.TestCase):
    """7.9A: TwoLayerOrchestrator.execute_with_agent_graph method."""

    def test_method_exists(self):
        """TwoLayerOrchestrator has execute_with_agent_graph method."""
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator
        self.assertTrue(hasattr(TwoLayerOrchestrator, "execute_with_agent_graph"))

    def test_method_returns_agent_graph_backend(self):
        """execute_with_agent_graph returns result with backend='agent_graph'."""
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        orch = TwoLayerOrchestrator()

        with patch(
            "vetinari.orchestration.agent_graph.get_agent_graph"
        ) as mock_ag:
            mock_graph = MagicMock()
            # Simulate AgentGraph.execute_plan returning results
            mock_result = MagicMock(success=True, output="done", errors=[])
            mock_graph.execute_plan.return_value = {"t1": mock_result}
            mock_ag.return_value = mock_graph

            # Also mock PlanGenerator
            mock_exec_graph = MagicMock()
            mock_exec_graph.plan_id = "plan-1"
            mock_exec_graph.nodes = {}
            orch.plan_generator.generate_plan = MagicMock(
                return_value=mock_exec_graph
            )

            result = orch.execute_with_agent_graph("Build a thing")
            self.assertEqual(result["backend"], "agent_graph")

    def test_fallback_on_agent_graph_failure(self):
        """Falls back to generate_and_execute when AgentGraph unavailable."""
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        orch = TwoLayerOrchestrator()

        with patch(
            "vetinari.orchestration.agent_graph.get_agent_graph",
            side_effect=ImportError("not available"),
        ):
            with patch.object(orch, "generate_and_execute") as mock_fallback:
                mock_fallback.return_value = {"plan_id": "p1", "backend": "durable"}
                result = orch.execute_with_agent_graph("Build a thing")
                mock_fallback.assert_called_once()


# ---------------------------------------------------------------------------
# Phase 7.9B: Blackboard inter-agent delegation patterns
# ---------------------------------------------------------------------------


class TestBlackboardDelegation(unittest.TestCase):
    """7.9B: Blackboard helper methods for inter-agent delegation."""

    def test_request_help_posts_and_returns(self):
        """request_help posts an entry and waits for result."""
        from vetinari.blackboard import Blackboard

        board = Blackboard()
        # Patch claim/complete/get_result chain
        with patch.object(board, "get_result", return_value="found it"):
            result = board.request_help(
                "BUILDER", "code_search", "Find async patterns"
            )
            self.assertEqual(result, "found it")

    def test_escalate_error_creates_high_priority_entry(self):
        """escalate_error creates a priority-1 error_recovery entry."""
        from vetinari.blackboard import Blackboard

        board = Blackboard()
        entry_id = board.escalate_error(
            "BUILDER", "t1", "NullPointerException"
        )
        entry = board.get_entry(entry_id)
        self.assertEqual(entry.priority, 1)
        self.assertEqual(entry.request_type, "error_recovery")
        self.assertIn("NullPointerException", entry.content)

    def test_request_consensus_creates_entry(self):
        """request_consensus posts an architecture_decision entry."""
        from vetinari.blackboard import Blackboard

        board = Blackboard()
        entry_id = board.request_consensus(
            "PLANNER", "Which DB?", ["PostgreSQL", "SQLite"]
        )
        entry = board.get_entry(entry_id)
        self.assertEqual(entry.request_type, "architecture_decision")
        self.assertTrue(entry.metadata["consensus_request"])
        self.assertEqual(entry.metadata["options"], ["PostgreSQL", "SQLite"])

    def test_blackboard_has_all_delegation_methods(self):
        """Blackboard has request_help, escalate_error, request_consensus."""
        from vetinari.blackboard import Blackboard
        self.assertTrue(hasattr(Blackboard, "request_help"))
        self.assertTrue(hasattr(Blackboard, "escalate_error"))
        self.assertTrue(hasattr(Blackboard, "request_consensus"))

    def test_escalate_error_includes_context(self):
        """escalate_error metadata includes original_task_id and error."""
        from vetinari.blackboard import Blackboard

        board = Blackboard()
        entry_id = board.escalate_error(
            "QUALITY", "t5", "type error",
            context={"file": "main.py"},
        )
        entry = board.get_entry(entry_id)
        self.assertEqual(entry.metadata["original_task_id"], "t5")
        self.assertEqual(entry.metadata["error"], "type error")
        self.assertEqual(entry.metadata["file"], "main.py")


# ---------------------------------------------------------------------------
# Topological sort helper
# ---------------------------------------------------------------------------


class TestTopologicalSort(unittest.TestCase):
    """Test AgentGraph._topological_sort."""

    def test_linear_chain(self):
        from vetinari.orchestration.agent_graph import AgentGraph, TaskNode
        from vetinari.agents.contracts import AgentType, Task

        t1 = Task(id="t1", description="A", assigned_agent=AgentType.BUILDER)
        t2 = Task(id="t2", description="B", assigned_agent=AgentType.BUILDER, dependencies=["t1"])
        t3 = Task(id="t3", description="C", assigned_agent=AgentType.BUILDER, dependencies=["t2"])

        nodes = {
            "t1": TaskNode(task=t1),
            "t2": TaskNode(task=t2, dependencies={"t1"}),
            "t3": TaskNode(task=t3, dependencies={"t2"}),
        }
        # Set up dependents (reverse edges) required by instance method
        nodes["t1"].dependents.add("t2")
        nodes["t2"].dependents.add("t3")

        graph = AgentGraph()
        graph.initialize()
        order = graph._topological_sort(nodes)
        self.assertEqual(order, ["t1", "t2", "t3"])

    def test_parallel_tasks(self):
        from vetinari.orchestration.agent_graph import AgentGraph, TaskNode
        from vetinari.agents.contracts import AgentType, Task

        t1 = Task(id="t1", description="A", assigned_agent=AgentType.BUILDER)
        t2 = Task(id="t2", description="B", assigned_agent=AgentType.BUILDER)
        t3 = Task(id="t3", description="C", assigned_agent=AgentType.BUILDER,
                   dependencies=["t1", "t2"])

        nodes = {
            "t1": TaskNode(task=t1),
            "t2": TaskNode(task=t2),
            "t3": TaskNode(task=t3, dependencies={"t1", "t2"}),
        }
        # Set up dependents (reverse edges)
        nodes["t1"].dependents.add("t3")
        nodes["t2"].dependents.add("t3")

        graph = AgentGraph()
        graph.initialize()
        order = graph._topological_sort(nodes)
        # t1 and t2 before t3
        self.assertLess(order.index("t1"), order.index("t3"))
        self.assertLess(order.index("t2"), order.index("t3"))


if __name__ == "__main__":
    unittest.main()
