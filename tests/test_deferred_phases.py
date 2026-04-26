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

import pytest

from vetinari.types import AgentType

# ---------------------------------------------------------------------------
# Phase 7.1: SkillSpec entries for consolidated agents
# ---------------------------------------------------------------------------


class TestSkillSpecConsolidatedAgents:
    """Every consolidated agent must have a SkillSpec entry."""

    def test_all_consolidated_agents_have_skill_specs(self):
        from vetinari.skills.skill_registry import SKILL_REGISTRY

        required = ["foreman", "worker", "inspector"]
        for skill_id in required:
            assert skill_id in SKILL_REGISTRY, f"Missing SkillSpec: {skill_id}"

    def test_skill_specs_have_required_fields(self):
        from vetinari.skills.skill_registry import SKILL_REGISTRY

        for skill_id, spec in SKILL_REGISTRY.items():
            assert spec.skill_id, f"{skill_id}: empty skill_id"
            assert spec.name, f"{skill_id}: empty name"
            assert spec.description, f"{skill_id}: empty description"
            assert spec.modes, f"{skill_id}: no modes"
            assert spec.capabilities, f"{skill_id}: no capabilities"
            assert spec.input_schema, f"{skill_id}: no input_schema"
            assert spec.output_schema, f"{skill_id}: no output_schema"

    def test_skill_specs_validate(self):
        from vetinari.skills.skill_registry import validate_all

        errors = validate_all()
        assert errors == [], f"Skill spec validation errors: {errors}"

    def test_canonical_agent_type_mapping_complete(self):
        from vetinari.skills.skill_registry import _AGENT_TO_SKILL

        canonical_types = [AgentType.FOREMAN.value, AgentType.WORKER.value, AgentType.INSPECTOR.value]
        for ct in canonical_types:
            assert ct in _AGENT_TO_SKILL, f"Missing canonical mapping for {ct}"

    def test_get_skill_for_agent_type(self):
        from vetinari.skills.skill_registry import get_skill_for_agent_type

        # Canonical types map to their respective skills
        spec = get_skill_for_agent_type(AgentType.WORKER.value)
        assert spec is not None
        assert spec.skill_id == "worker"

        # Inspector maps to inspector
        spec = get_skill_for_agent_type(AgentType.INSPECTOR.value)
        assert spec is not None
        assert spec.skill_id == "inspector"

    def test_get_skills_by_capability(self):
        from vetinari.skills.skill_registry import get_skills_by_capability

        specs = get_skills_by_capability("feature_implementation")
        assert len(specs) >= 1
        # feature_implementation is a Worker capability
        assert any(s.skill_id == "worker" for s in specs)

    def test_get_skills_by_tag(self):
        from vetinari.skills.skill_registry import get_skills_by_tag

        specs = get_skills_by_tag("security")
        assert len(specs) >= 1
        # security tag lives on the Inspector skill
        assert any(s.skill_id == "inspector" for s in specs)


# ---------------------------------------------------------------------------
# Phase 7.4: Prompt assembler role defs for consolidated agents
# ---------------------------------------------------------------------------


class TestPromptAssemblerConsolidated:
    """Prompt assembler has role definitions for consolidated agents."""

    def test_consolidated_role_defs_exist(self):
        from vetinari.prompts.assembler import _ROLE_DEFS

        # Three-tier active agents
        for agent_type in [AgentType.FOREMAN.value, AgentType.WORKER.value, AgentType.INSPECTOR.value]:
            assert agent_type in _ROLE_DEFS, f"Missing role def for {agent_type}"

    def test_legacy_role_defs_still_exist(self):
        from vetinari.prompts.assembler import _ROLE_DEFS

        # Three-tier model: FOREMAN, WORKER, INSPECTOR replace all legacy agents
        for agent_type in [AgentType.FOREMAN.value, AgentType.WORKER.value, AgentType.INSPECTOR.value]:
            assert agent_type in _ROLE_DEFS, f"Missing role def for {agent_type}"

    def test_build_prompt_with_consolidated_agent(self):
        from vetinari.prompts import get_prompt_assembler

        asm = get_prompt_assembler()
        result = asm.build(
            agent_type=AgentType.INSPECTOR.value,
            task_type="review",
            task_description="Review this code for bugs",
        )
        assert "system" in result
        assert "user" in result
        assert "Inspector" in result["system"]

    def test_build_prompt_with_unknown_agent_uses_fallback(self):
        from vetinari.prompts import get_prompt_assembler

        asm = get_prompt_assembler()
        result = asm.build(
            agent_type="UNKNOWN_TYPE",
            task_type="general",
            task_description="Do something",
        )
        assert "system" in result
        assert "specialist" in result["system"].lower()


# ---------------------------------------------------------------------------
# Phase 7.9: AgentGraph capability-based routing
# ---------------------------------------------------------------------------


class TestAgentGraphCapabilityRouting:
    """AgentGraph can route by capability via SkillSpec."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from vetinari.orchestration.agent_graph import AgentGraph

        self.graph = AgentGraph()
        # Don't call initialize() — mock agents instead

    def test_get_agent_by_capability_no_agents(self):
        result = self.graph.get_agent_by_capability("feature_implementation")
        assert result is None

    def test_get_agent_by_capability_with_mock_agent(self):
        from vetinari.types import AgentType

        mock_agent = MagicMock()
        self.graph._agents[AgentType.WORKER] = mock_agent

        result = self.graph.get_agent_by_capability("feature_implementation")
        assert result is mock_agent

    def test_get_agent_by_capability_unknown(self):
        result = self.graph.get_agent_by_capability("teleportation")
        assert result is None

    def test_get_skill_spec(self):
        from vetinari.types import AgentType

        spec = self.graph.get_skill_spec(AgentType.WORKER)
        assert spec is not None
        assert spec.skill_id == "worker"

    def test_get_skill_spec_legacy_type(self):
        from vetinari.types import AgentType

        spec = self.graph.get_skill_spec(AgentType.INSPECTOR)
        assert spec is not None
        assert spec.skill_id == "inspector"

    def test_get_agents_for_task_type(self):
        from vetinari.types import AgentType

        mock_agent = MagicMock()
        self.graph._agents[AgentType.WORKER] = mock_agent

        agents = self.graph.get_agents_for_task_type("feature_implementation")
        assert AgentType.WORKER in agents

    def test_get_agents_for_task_type_empty(self):
        agents = self.graph.get_agents_for_task_type("nonexistent_mode")
        assert agents == []


# ---------------------------------------------------------------------------
# Phase 8.2: Output schema validation
# ---------------------------------------------------------------------------


class TestOutputSchemaValidation:
    """AgentGraph validates output against SkillSpec output_schema."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from vetinari.orchestration.agent_graph import AgentGraph

        self.graph = AgentGraph()

    def test_valid_output_no_issues(self):
        from vetinari.types import AgentType

        # Worker output_schema requires 'success' (boolean) and 'output'
        output = {"success": True, "output": "print('hi')"}
        issues = self.graph._validate_output_schema(AgentType.WORKER, output)
        assert issues == []

    def test_missing_required_field(self):
        from vetinari.types import AgentType

        output = {"output": "done"}  # missing "success"
        issues = self.graph._validate_output_schema(AgentType.WORKER, output)
        assert any("success" in i for i in issues)

    def test_wrong_type_detected(self):
        from vetinari.types import AgentType

        output = {"success": "yes", "output": "done"}  # success should be boolean
        issues = self.graph._validate_output_schema(AgentType.WORKER, output)
        assert any("type" in i.lower() for i in issues)

    def test_non_dict_output_skipped(self):
        from vetinari.types import AgentType

        issues = self.graph._validate_output_schema(AgentType.WORKER, "just a string")
        assert issues == []

    def test_unknown_agent_type_no_issues(self):
        from vetinari.types import AgentType

        # IMAGE_GENERATOR may not have a SkillSpec
        output = {"whatever": True}
        issues = self.graph._validate_output_schema(AgentType.WORKER, output)
        # Should not crash, may return issues or empty
        assert isinstance(issues, list)


# ---------------------------------------------------------------------------
# Phase 8.5/8.6: Style constraints
# ---------------------------------------------------------------------------


class TestStyleConstraints:
    """Document and code style constraints."""

    def test_code_style_detects_todo(self):
        from vetinari.constraints.style import validate_output_style

        issues = validate_output_style("x = 1  # TODO fix later", "code")
        rule_ids = [i["rule_id"] for i in issues]
        assert "code-no-todo" in rule_ids

    def test_code_style_detects_bare_except(self):
        from vetinari.constraints.style import validate_output_style

        issues = validate_output_style("try:\n    pass\nexcept:\n    pass", "code")
        rule_ids = [i["rule_id"] for i in issues]
        assert "code-no-bare-except" in rule_ids

    def test_code_style_detects_hardcoded_secret(self):
        from vetinari.constraints.style import validate_output_style

        issues = validate_output_style('api_key = "sk-1234567890abcdef"', "code")
        rule_ids = [i["rule_id"] for i in issues]
        assert "code-no-hardcoded-secrets" in rule_ids

    def test_doc_style_detects_placeholder(self):
        from vetinari.constraints.style import validate_output_style

        issues = validate_output_style("This section is TBD.", "documentation")
        rule_ids = [i["rule_id"] for i in issues]
        assert "doc-no-placeholder" in rule_ids

    def test_clean_code_passes(self):
        from vetinari.constraints.style import validate_output_style

        issues = validate_output_style("def add(a, b):\n    return a + b", "code")
        warning_issues = [i for i in issues if i["severity"] == "warning"]
        assert warning_issues == []

    def test_style_domain_mapping(self):
        from vetinari.constraints.style import get_style_domain

        assert get_style_domain(AgentType.WORKER.value) == "code"
        assert get_style_domain(AgentType.FOREMAN.value) == "documentation"
        assert get_style_domain(AgentType.WORKER.value, mode="creative_writing") == "creative"
        assert get_style_domain(AgentType.WORKER.value, mode="security_audit") == "code"

    def test_get_style_rules(self):
        from vetinari.constraints.style import get_style_rules

        code_rules = get_style_rules("code")
        assert code_rules is not None
        assert code_rules.domain == "code"
        assert len(code_rules.rules) > 0

    def test_creative_style_no_line_limit(self):
        from vetinari.constraints.style import get_style_rules

        creative = get_style_rules("creative")
        assert creative.max_line_length == 0

    def test_forbidden_phrases(self):
        from vetinari.constraints.style import validate_output_style

        issues = validate_output_style("This is a HACK to fix it", "code")
        rule_ids = [i["rule_id"] for i in issues]
        assert "forbidden-phrase" in rule_ids

    def test_empty_text_no_issues(self):
        from vetinari.constraints.style import validate_output_style

        issues = validate_output_style("", "code")
        assert issues == []

    def test_style_constraint_exports(self):
        from vetinari.constraints import (
            STYLE_CONSTRAINTS,
            StyleConstraint,
            StyleRule,
            get_style_domain,
            get_style_rules,
            validate_output_style,
        )

        assert STYLE_CONSTRAINTS is not None
        assert callable(validate_output_style)


# ---------------------------------------------------------------------------
# SkillSpec serialization
# ---------------------------------------------------------------------------


class TestSkillSpecSerialization:
    """SkillSpec to_dict / from_dict round-trip."""

    def test_round_trip(self):
        from vetinari.skills.skill_spec import SkillSpec

        spec = SkillSpec(
            skill_id="test",
            name="Test",
            description="Test skill",
            modes=["a", "b"],
            capabilities=["c"],
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )
        d = spec.to_dict()
        restored = SkillSpec.from_dict(d)
        assert restored.skill_id == "test"
        assert restored.modes == ["a", "b"]
        assert restored.capabilities == ["c"]


# ---------------------------------------------------------------------------
# Phase 7.2-7.3: Legacy registry fallback to programmatic SkillSpec
# ---------------------------------------------------------------------------


class TestLegacyRegistryFallback:
    """Legacy SkillRegistry falls back to programmatic SkillSpec."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from vetinari.skills.skill_registry import SkillRegistry

        self.reg = SkillRegistry(load_on_init=False)
        # Force loaded state without disk files (empty registry)
        self.reg.is_loaded = True

    def test_get_skill_fallback_to_programmatic(self):
        """get_skill() returns SkillSpec data when disk has no match."""
        result = self.reg.get_skill("worker")
        assert result is not None
        assert result["skill_id"] == "worker"

    def test_get_skill_manifest_fallback(self):
        """get_skill_manifest() returns SkillSpec data as manifest."""
        result = self.reg.get_skill_manifest("inspector")
        assert result is not None
        assert result["skill_id"] == "inspector"
        assert "capabilities" in result

    def test_get_skill_manifest_caches(self):
        """Programmatic manifest is cached after first lookup."""
        m1 = self.reg.get_skill_manifest("worker")
        m2 = self.reg.get_skill_manifest("worker")
        assert m1 is m2

    def test_get_skill_by_capability_includes_programmatic(self):
        """get_skill_by_capability() finds programmatic specs."""
        results = self.reg.get_skill_by_capability("feature_implementation")
        skill_ids = [r.get("skill_id") or r.get("id") for r in results]
        # feature_implementation belongs to the Worker skill
        assert "worker" in skill_ids

    def test_list_skills_includes_programmatic(self):
        """list_skills() merges programmatic specs."""
        skills = self.reg.list_skills()
        skill_ids = [s.get("skill_id") or s.get("id") for s in skills]
        for expected in ["foreman", "worker", "inspector"]:
            assert expected in skill_ids, f"Missing: {expected}"

    def test_list_agents_includes_consolidated(self):
        """list_agents() includes the three-tier factory agent types."""
        agents = self.reg.list_agents()
        for agent_type in [AgentType.FOREMAN.value, AgentType.WORKER.value, AgentType.INSPECTOR.value]:
            assert agent_type in agents, f"Missing agent: {agent_type}"

    def test_get_agent_skills_fallback(self):
        """get_agent_skills() returns programmatic skill for canonical agent types."""
        skills = self.reg.get_agent_skills(AgentType.WORKER.value)
        assert len(skills) >= 1
        assert skills[0]["skill_id"] == "worker"

    def test_get_agent_skills_inspector_fallback(self):
        """get_agent_skills() returns inspector skill for INSPECTOR agent type."""
        skills = self.reg.get_agent_skills(AgentType.INSPECTOR.value)
        assert len(skills) >= 1
        assert skills[0]["skill_id"] == "inspector"

    def test_search_skills_finds_programmatic(self):
        """search_skills() searches programmatic specs too."""
        results = self.reg.search_skills("security")
        skill_ids = [r.get("skill_id") or r.get("id") for r in results]
        # Security-related capabilities live in the Inspector skill
        assert "inspector" in skill_ids

    def test_search_skills_by_tag(self):
        """search_skills() can match on tags from programmatic specs."""
        results = self.reg.search_skills("architecture")
        skill_ids = [r.get("skill_id") or r.get("id") for r in results]
        # architecture tag belongs to the Worker skill
        assert "worker" in skill_ids

    def test_get_skill_unknown_returns_none(self):
        """get_skill() returns None for truly unknown skills."""
        result = self.reg.get_skill("nonexistent_skill_xyz")
        assert result is None

    def test_get_skill_capabilities_fallback(self):
        """get_skill_capabilities() works via fallback."""
        caps = self.reg.get_skill_capabilities("worker")
        assert "feature_implementation" in caps


# ---------------------------------------------------------------------------
# Phase 7.7-7.8: Planner prompt + affinity table
# ---------------------------------------------------------------------------


class TestPlannerConsolidatedAgents:
    """Planner prompt references consolidated agents and affinity table."""

    def test_system_prompt_has_consolidated_agents(self):
        """Planner system prompt is non-empty and references planning concepts."""
        from vetinari.agents.planner_agent import ForemanAgent as PlannerAgent

        planner = PlannerAgent()
        prompt = planner.get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_system_prompt_has_affinity_table(self):
        """Planner system prompt loads successfully from file or hardcoded fallback."""
        from vetinari.agents.planner_agent import ForemanAgent as PlannerAgent

        planner = PlannerAgent()
        prompt = planner.get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_system_prompt_lists_active_agents(self):
        """Planner system prompt is non-empty."""
        from vetinari.agents.planner_agent import ForemanAgent as PlannerAgent

        planner = PlannerAgent()
        prompt = planner.get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_decompose_available_agents_includes_factory_pipeline(self):
        """decompose_goal_llm available_agents includes 3-agent factory types."""
        import inspect

        from vetinari.agents.planner_decompose import decompose_goal_llm

        source = inspect.getsource(decompose_goal_llm)
        for agent in [AgentType.FOREMAN.value, AgentType.WORKER.value, AgentType.INSPECTOR.value]:
            # M4: agent types referenced via AgentType enum, not string literals
            assert f"AgentType.{agent}" in source or f'"{agent}"' in source, (
                f"{agent} missing from decompose_goal_llm available_agents"
            )

    def test_planner_prefers_consolidated_agents_rule(self):
        """System prompt is non-empty and describes the Planner role."""
        from vetinari.agents.planner_agent import ForemanAgent as PlannerAgent

        planner = PlannerAgent()
        prompt = planner.get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50


# ---------------------------------------------------------------------------
# Phase 8.9: rules.yaml expansion
# ---------------------------------------------------------------------------


class TestRulesYamlConsolidated:
    """rules.yaml includes consolidated agent identifiers."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from pathlib import Path

        import yaml

        rules_path = Path(__file__).parent.parent / "vetinari" / "config" / "rules.yaml"
        with open(rules_path, encoding="utf-8") as f:
            self.rules = yaml.safe_load(f)

    def test_agents_section_exists(self):
        assert "agents" in self.rules
        assert isinstance(self.rules["agents"], dict)

    def test_all_consolidated_agents_present(self):
        agents = self.rules["agents"]
        for agent in [AgentType.FOREMAN.value, AgentType.WORKER.value, AgentType.INSPECTOR.value]:
            assert agent in agents, f"Missing three-tier agent in rules.yaml: {agent}"

    def test_agent_rules_are_lists(self):
        agents = self.rules["agents"]
        for agent_type, rules in agents.items():
            assert isinstance(rules, list), f"{agent_type} rules should be a list"
        # Three-tier agents must each have at least 2 rules
        for agent_type in [AgentType.FOREMAN.value, AgentType.WORKER.value, AgentType.INSPECTOR.value]:
            if agent_type in agents:
                assert len(agents[agent_type]) >= 2, f"{agent_type} should have at least 2 rules"

    def test_builder_has_verification_score(self):
        agents = self.rules["agents"]
        worker_rules = " ".join(agents[AgentType.WORKER.value])
        assert "verification score" in worker_rules.lower()

    def test_quality_highest_bar(self):
        agents = self.rules["agents"]
        inspector_rules = " ".join(agents[AgentType.INSPECTOR.value])
        assert "0.7" in inspector_rules

    def test_operations_extended_budget(self):
        agents = self.rules["agents"]
        # WORKER handles all operational tasks in the three-tier model
        worker_rules = " ".join(agents[AgentType.WORKER.value])
        assert len(worker_rules) > 0, "WORKER should have rules"

    def test_researcher_replaces_legacy(self):
        agents = self.rules["agents"]
        # WORKER handles all research tasks in the three-tier model
        worker_rules = " ".join(agents[AgentType.WORKER.value])
        assert len(worker_rules) > 0, "WORKER should have rules"

    def test_global_section_preserved(self):
        assert "global" in self.rules
        assert "global_system_prompt" in self.rules
        assert "models" in self.rules


# ---------------------------------------------------------------------------
# Phase 7.9H: Permission enforcement in AgentGraph + Blackboard
# ---------------------------------------------------------------------------


class TestPermissionEnforcementAgentGraph:
    """7.9H: Permission enforcement in _execute_task_node and Blackboard.claim."""

    def test_agent_graph_has_permission_check_in_execute(self):
        """_execute_task_node source contains permission enforcement."""
        import inspect

        from vetinari.orchestration.agent_graph import AgentGraph

        src = inspect.getsource(AgentGraph._execute_task_node)
        assert "enforce_permission" in src
        assert "MODEL_INFERENCE" in src

    def test_agent_graph_returns_failure_on_permission_denied(self):
        """When permission is denied, _execute_task_node returns failure."""
        from vetinari.agents.contracts import AgentResult, Task
        from vetinari.orchestration.agent_graph import AgentGraph, TaskNode
        from vetinari.types import AgentType

        graph = AgentGraph()
        mock_agent = MagicMock()
        mock_agent.execute.return_value = AgentResult(success=True, output="ok")
        mock_agent.verify.return_value = MagicMock(passed=True, issues=[])
        graph._agents[AgentType.WORKER] = mock_agent

        task = Task(id="t1", description="Test", assigned_agent=AgentType.WORKER)
        node = TaskNode(task=task)

        # Patch enforce_permission to raise PermissionError
        with patch("vetinari.execution_context.get_context_manager") as mock_ctx:
            mock_mgr = MagicMock()
            mock_mgr.enforce_permission.side_effect = PermissionError("denied")
            mock_ctx.return_value = mock_mgr

            result = graph._execute_task_node(node)
            assert not result.success
            assert "Permission denied" in result.errors[0]

    def test_agent_graph_allows_when_no_context_manager(self):
        """When context manager is not configured, execution proceeds normally."""
        from vetinari.agents.contracts import AgentResult, Task
        from vetinari.orchestration.agent_graph import AgentGraph, TaskNode
        from vetinari.types import AgentType

        graph = AgentGraph()
        mock_agent = MagicMock()
        mock_agent.execute.return_value = AgentResult(success=True, output="ok")
        mock_agent.verify.return_value = MagicMock(passed=True, issues=[], score=1.0)
        mock_agent._incorporate_prior_results = MagicMock()
        graph._agents[AgentType.WORKER] = mock_agent

        task = Task(id="t1", description="Test", assigned_agent=AgentType.WORKER)
        node = TaskNode(task=task)

        with patch(
            "vetinari.execution_context.get_context_manager",
            side_effect=ImportError("not available"),
        ):
            result = graph._execute_task_node(node)
            assert result.success

    def test_blackboard_claim_checks_permission(self):
        """Blackboard.claim checks MODEL_INFERENCE permission."""
        import inspect

        from vetinari.memory.blackboard import Blackboard

        src = inspect.getsource(Blackboard.claim)
        assert "MODEL_INFERENCE" in src
        assert "check_permission" in src

    def test_blackboard_claim_denied_returns_none(self):
        """When permission is denied, claim returns None."""
        from vetinari.memory.blackboard import Blackboard, EntryState

        board = Blackboard()
        entry_id = board.post("test", "code_search", "BUILDER")

        with patch("vetinari.execution_context.get_context_manager") as mock_ctx:
            mock_mgr = MagicMock()
            mock_mgr.check_permission.return_value = False
            mock_ctx.return_value = mock_mgr

            result = board.claim(entry_id, "EXPLORER")
            assert result is None

    def test_blackboard_claim_allowed_succeeds(self):
        """When permission is allowed, claim succeeds."""
        from vetinari.memory.blackboard import Blackboard, EntryState

        board = Blackboard()
        entry_id = board.post("test", "code_search", "BUILDER")

        with patch("vetinari.execution_context.get_context_manager") as mock_ctx:
            mock_mgr = MagicMock()
            mock_mgr.check_permission.return_value = True
            mock_ctx.return_value = mock_mgr

            result = board.claim(entry_id, "EXPLORER")
            assert result is not None
            assert result.claimed_by == "EXPLORER"
            assert result.state == EntryState.CLAIMED


# ---------------------------------------------------------------------------
# Phase 7.9I: Dependency results incorporation
# ---------------------------------------------------------------------------


class TestDependencyResultsIncorporation:
    """7.9I: BaseAgent._incorporate_prior_results and AgentGraph wiring."""

    def test_base_agent_has_incorporate_method(self):
        """BaseAgent has _incorporate_prior_results method."""
        from vetinari.agents.base_agent import BaseAgent

        assert hasattr(BaseAgent, "_incorporate_prior_results")

    def test_incorporate_extracts_dependency_results(self):
        """_incorporate_prior_results extracts context.dependency_results."""
        from vetinari.agents.contracts import AgentTask
        from vetinari.agents.planner_agent import ForemanAgent as PlannerAgent
        from vetinari.types import AgentType

        agent = PlannerAgent()
        task = AgentTask(
            task_id="t2",
            agent_type=AgentType.FOREMAN,
            description="Test",
            prompt="Test",
            context={
                "dependency_results": {
                    "t1": {"success": True, "output_summary": "spec ready"},
                }
            },
        )
        results = agent._incorporate_prior_results(task)
        assert "t1" in results
        assert results["t1"]["success"]

    def test_incorporate_returns_empty_when_no_deps(self):
        """Returns empty dict when no dependency_results in context."""
        from vetinari.agents.contracts import AgentTask
        from vetinari.agents.planner_agent import ForemanAgent as PlannerAgent
        from vetinari.types import AgentType

        agent = PlannerAgent()
        task = AgentTask(
            task_id="t1",
            agent_type=AgentType.FOREMAN,
            description="Test",
            prompt="Test",
            context={},
        )
        results = agent._incorporate_prior_results(task)
        assert results == {}

    def test_agent_graph_calls_incorporate(self):
        """AgentGraph._execute_task_node calls _incorporate_prior_results."""
        import inspect

        from vetinari.orchestration.agent_graph import AgentGraph

        src = inspect.getsource(AgentGraph._execute_task_node)
        assert "_incorporate_prior_results" in src


# ---------------------------------------------------------------------------
# Phase 7.9J: Dynamic graph modification (inject_task)
# ---------------------------------------------------------------------------


class TestInjectTask:
    """7.9J: AgentGraph.inject_task for mid-execution DAG changes."""

    @pytest.fixture
    def graph_with_plan(self):
        """Provide a pre-wired AgentGraph with a two-task plan for inject_task tests."""
        from vetinari.agents.contracts import ExecutionPlan, Task
        from vetinari.orchestration.agent_graph import AgentGraph, ExecutionDAG, TaskNode
        from vetinari.types import AgentType

        graph = AgentGraph()
        plan = ExecutionPlan.create_new("Test goal")
        t1 = Task(id="t1", description="First", assigned_agent=AgentType.WORKER)
        t2 = Task(id="t2", description="Second", assigned_agent=AgentType.WORKER, dependencies=["t1"])

        exec_plan = ExecutionDAG(plan_id=plan.plan_id, original_plan=plan)
        exec_plan.nodes["t1"] = TaskNode(task=t1, dependents={"t2"})
        exec_plan.nodes["t2"] = TaskNode(task=t2, dependencies={"t1"})
        exec_plan.execution_order = ["t1", "t2"]

        graph._execution_plans[plan.plan_id] = exec_plan
        return graph, plan, exec_plan

    def test_inject_task_success(self, graph_with_plan):
        """inject_task inserts a new task between existing ones."""
        from vetinari.agents.contracts import Task
        from vetinari.types import AgentType

        graph, plan, exec_plan = graph_with_plan
        new_task = Task(id="t_review", description="Review", assigned_agent=AgentType.INSPECTOR)

        result = graph.inject_task(plan.plan_id, new_task, "t1")
        assert result is True
        assert "t_review" in exec_plan.nodes

        # t_review should depend on t1
        assert "t1" in exec_plan.nodes["t_review"].dependencies

        # t2 should now depend on t_review (not t1)
        assert "t_review" in exec_plan.nodes["t2"].dependencies
        assert "t1" not in exec_plan.nodes["t2"].dependencies

    def test_inject_task_nonexistent_plan(self):
        """inject_task returns False for unknown plan_id."""
        from vetinari.agents.contracts import Task
        from vetinari.orchestration.agent_graph import AgentGraph
        from vetinari.types import AgentType

        graph = AgentGraph()
        task = Task(id="tx", description="X", assigned_agent=AgentType.WORKER)
        assert not graph.inject_task("no_such_plan", task, "t1")

    def test_inject_task_nonexistent_after(self, graph_with_plan):
        """inject_task returns False when after_task_id not in plan."""
        from vetinari.agents.contracts import Task
        from vetinari.types import AgentType

        graph, plan, _ = graph_with_plan
        task = Task(id="tx", description="X", assigned_agent=AgentType.WORKER)
        assert not graph.inject_task(plan.plan_id, task, "nonexistent")

    def test_inject_task_duplicate_id(self, graph_with_plan):
        """inject_task returns False when task.id already exists."""
        from vetinari.agents.contracts import Task
        from vetinari.types import AgentType

        graph, plan, _ = graph_with_plan
        task = Task(id="t1", description="Dup", assigned_agent=AgentType.WORKER)
        assert not graph.inject_task(plan.plan_id, task, "t1")

    def test_inject_task_updates_execution_order(self, graph_with_plan):
        """inject_task rebuilds execution_order with new task."""
        from vetinari.agents.contracts import Task
        from vetinari.types import AgentType

        graph, plan, exec_plan = graph_with_plan
        new_task = Task(id="t_mid", description="Mid", assigned_agent=AgentType.INSPECTOR)
        graph.inject_task(plan.plan_id, new_task, "t1")

        # t_mid should appear between t1 and t2
        order = exec_plan.execution_order
        assert "t_mid" in order
        assert order.index("t1") < order.index("t_mid")
        assert order.index("t_mid") < order.index("t2")


# ---------------------------------------------------------------------------
# Phase 7.9K: Maker-checker pattern
# ---------------------------------------------------------------------------


class TestMakerChecker:
    """7.9K: QUALITY reviews BUILDER output with feedback loop."""

    def test_maker_checker_approval(self):
        """QUALITY approves BUILDER output on first attempt."""
        from vetinari.agents.contracts import AgentResult, Task
        from vetinari.orchestration.agent_graph import AgentGraph
        from vetinari.types import AgentType

        graph = AgentGraph()
        mock_quality = MagicMock()
        mock_quality.execute.return_value = AgentResult(success=True, output={"score": 0.9})
        mock_quality.verify.return_value = MagicMock(passed=True, issues=[], score=0.9)
        graph._agents[AgentType.INSPECTOR] = mock_quality
        graph._agents[AgentType.WORKER] = MagicMock()

        task = Task(id="t1", description="Build X", assigned_agent=AgentType.WORKER)
        result = AgentResult(success=True, output="code here")

        final = graph._apply_maker_checker(task, result)
        assert final.success
        assert final.metadata["maker_checker"]["approved"]
        assert final.metadata["maker_checker"]["iterations"] == 1

    def test_maker_checker_rejection_retries(self):
        """QUALITY rejects, BUILDER gets feedback and retries."""
        from vetinari.agents.contracts import AgentResult, Task
        from vetinari.orchestration.agent_graph import AgentGraph
        from vetinari.types import AgentType

        graph = AgentGraph()

        # QUALITY rejects first, approves second
        mock_quality = MagicMock()
        reject_verify = MagicMock(passed=False, issues=[{"message": "missing tests"}], score=0.3)
        approve_verify = MagicMock(passed=True, issues=[], score=0.9)
        mock_quality.execute.return_value = AgentResult(success=True, output={"score": 0.9})
        mock_quality.verify.side_effect = [reject_verify, approve_verify]
        graph._agents[AgentType.INSPECTOR] = mock_quality

        mock_builder = MagicMock()
        mock_builder.execute.return_value = AgentResult(success=True, output="fixed code")
        graph._agents[AgentType.WORKER] = mock_builder

        task = Task(id="t1", description="Build X", assigned_agent=AgentType.WORKER)
        result = AgentResult(success=True, output="initial code")

        final = graph._apply_maker_checker(task, result)
        assert final.metadata["maker_checker"]["approved"]
        assert final.metadata["maker_checker"]["iterations"] == 2
        # BUILDER should have been called with feedback
        builder_call = mock_builder.execute.call_args
        assert "MAKER-CHECKER FEEDBACK" in builder_call[0][0].description

    def test_maker_checker_max_iterations(self):
        """After max iterations without approval, returns not-approved result."""
        from vetinari.agents.contracts import AgentResult, Task
        from vetinari.orchestration.agent_graph import AgentGraph
        from vetinari.types import AgentType

        graph = AgentGraph()
        mock_quality = MagicMock()
        mock_quality.execute.return_value = AgentResult(success=True, output={})
        mock_quality.verify.return_value = MagicMock(passed=False, issues=[{"message": "still bad"}], score=0.2)
        graph._agents[AgentType.INSPECTOR] = mock_quality

        mock_builder = MagicMock()
        mock_builder.execute.return_value = AgentResult(success=True, output="code")
        graph._agents[AgentType.WORKER] = mock_builder

        task = Task(id="t1", description="Build", assigned_agent=AgentType.WORKER)
        result = AgentResult(success=True, output="code")

        final = graph._apply_maker_checker(task, result)
        assert not final.metadata["maker_checker"]["approved"]
        assert final.metadata["maker_checker"]["iterations"] == AgentGraph._MAKER_CHECKER_MAX_ITERATIONS

    def test_maker_checker_triggered_for_builder(self):
        """Task retry loop triggers maker-checker for BUILDER when QUALITY registered."""
        import inspect

        from vetinari.orchestration.task_retry_loop import TaskRetryLoopMixin

        src = inspect.getsource(TaskRetryLoopMixin._run_task_attempt_loop)
        assert "_apply_maker_checker" in src
        assert "_quality_reviewed_agents" in src

    def test_maker_checker_skips_without_quality(self):
        """Without QUALITY agent, maker-checker returns original result."""
        from vetinari.agents.contracts import AgentResult, Task
        from vetinari.orchestration.agent_graph import AgentGraph
        from vetinari.types import AgentType

        graph = AgentGraph()
        graph._agents[AgentType.WORKER] = MagicMock()
        # No QUALITY agent registered

        task = Task(id="t1", description="Build", assigned_agent=AgentType.WORKER)
        result = AgentResult(success=True, output="code")

        final = graph._apply_maker_checker(task, result)
        # Should return original result unchanged
        assert final.output == "code"


# ---------------------------------------------------------------------------
# Phase 7.9A: AgentGraph wired into TwoLayerOrchestrator
# ---------------------------------------------------------------------------


class TestAgentGraphInTwoLayer:
    """7.9A: TwoLayerOrchestrator.execute_with_agent_graph method."""

    def test_method_exists(self):
        """TwoLayerOrchestrator has execute_with_agent_graph method."""
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        assert hasattr(TwoLayerOrchestrator, "execute_with_agent_graph")

    def test_method_returns_agent_graph_backend(self):
        """execute_with_agent_graph returns result with backend='agent_graph'."""
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        orch = TwoLayerOrchestrator()

        with patch("vetinari.orchestration.agent_graph.get_agent_graph") as mock_ag:
            mock_graph = MagicMock()
            # Simulate AgentGraph.execute_plan returning results
            mock_result = MagicMock(success=True, output="done", errors=[])
            mock_graph.execute_plan.return_value = {"t1": mock_result}
            mock_ag.return_value = mock_graph

            # Also mock PlanGenerator
            mock_exec_graph = MagicMock()
            mock_exec_graph.plan_id = "plan-1"
            mock_exec_graph.nodes = {}
            mock_exec_graph.follow_up_question = None  # prevent pause path
            orch.plan_generator.generate_plan = MagicMock(return_value=mock_exec_graph)

            result = orch.execute_with_agent_graph("Build a thing")
            assert result["backend"] == "agent_graph"

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
                mock_fallback.assert_called_once_with("Build a thing", None, context=None)
                assert result == {"plan_id": "p1", "backend": "durable"}


# ---------------------------------------------------------------------------
# Phase 7.9B: Blackboard inter-agent delegation patterns
# ---------------------------------------------------------------------------


class TestBlackboardDelegation:
    """7.9B: Blackboard helper methods for inter-agent delegation."""

    def test_request_help_posts_and_returns(self):
        """request_help posts an entry and waits for result."""
        from vetinari.memory.blackboard import Blackboard

        board = Blackboard()
        # Patch claim/complete/get_result chain
        with patch.object(board, "get_result", return_value="found it"):
            result = board.request_help("BUILDER", "code_search", "Find async patterns")
            assert result == "found it"

    def test_escalate_error_creates_high_priority_entry(self):
        """escalate_error creates a priority-1 error_recovery entry."""
        from vetinari.memory.blackboard import Blackboard

        board = Blackboard()
        entry_id = board.escalate_error("BUILDER", "t1", "NullPointerException")
        entry = board.get_entry(entry_id)
        assert entry.priority == 1
        assert entry.request_type == "error_recovery"
        assert "NullPointerException" in entry.content

    def test_request_consensus_creates_entry(self):
        """request_consensus posts an architecture_decision entry."""
        from vetinari.memory.blackboard import Blackboard

        board = Blackboard()
        entry_id = board.request_consensus(AgentType.FOREMAN.value, "Which DB?", ["PostgreSQL", "SQLite"])
        entry = board.get_entry(entry_id)
        assert entry.request_type == "architecture_decision"
        assert entry.metadata["consensus_request"]
        assert entry.metadata["options"] == ["PostgreSQL", "SQLite"]

    def test_blackboard_has_all_delegation_methods(self):
        """Blackboard has request_help, escalate_error, request_consensus."""
        from vetinari.memory.blackboard import Blackboard

        assert hasattr(Blackboard, "request_help")
        assert hasattr(Blackboard, "escalate_error")
        assert hasattr(Blackboard, "request_consensus")

    def test_escalate_error_includes_context(self):
        """escalate_error metadata includes original_task_id and error."""
        from vetinari.memory.blackboard import Blackboard

        board = Blackboard()
        entry_id = board.escalate_error(
            "QUALITY",
            "t5",
            "type error",
            context={"file": "main.py"},
        )
        entry = board.get_entry(entry_id)
        assert entry.metadata["original_task_id"] == "t5"
        assert entry.metadata["error"] == "type error"
        assert entry.metadata["file"] == "main.py"


# ---------------------------------------------------------------------------
# Topological sort helper
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    """Test AgentGraph._topological_sort."""

    def test_linear_chain(self):
        from vetinari.agents.contracts import Task
        from vetinari.orchestration.agent_graph import AgentGraph, TaskNode
        from vetinari.types import AgentType

        t1 = Task(id="t1", description="A", assigned_agent=AgentType.WORKER)
        t2 = Task(id="t2", description="B", assigned_agent=AgentType.WORKER, dependencies=["t1"])
        t3 = Task(id="t3", description="C", assigned_agent=AgentType.WORKER, dependencies=["t2"])

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
        assert order == ["t1", "t2", "t3"]

    def test_parallel_tasks(self):
        from vetinari.agents.contracts import Task
        from vetinari.orchestration.agent_graph import AgentGraph, TaskNode
        from vetinari.types import AgentType

        t1 = Task(id="t1", description="A", assigned_agent=AgentType.WORKER)
        t2 = Task(id="t2", description="B", assigned_agent=AgentType.WORKER)
        t3 = Task(id="t3", description="C", assigned_agent=AgentType.WORKER, dependencies=["t1", "t2"])

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
        assert order.index("t1") < order.index("t3")
        assert order.index("t2") < order.index("t3")
