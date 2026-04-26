"""Tests for Phase 5a: Auto-derive SkillSpec from agent metadata.

Verifies that MultiModeAgent.to_skill_spec() generates correct specs
and that auto_populate_from_agents() merges with hand-written specs.
"""

from __future__ import annotations

from vetinari.agents.contracts import AgentResult, AgentTask
from vetinari.agents.multi_mode_agent import MultiModeAgent
from vetinari.skills.skill_spec import SkillSpec
from vetinari.types import AgentType

# ---------------------------------------------------------------------------
# Test agent subclass
# ---------------------------------------------------------------------------


class _TestMultiModeAgent(MultiModeAgent):
    """A test agent for skill derivation."""

    MODES = {
        "alpha": "_execute_alpha",
        "beta": "_execute_beta",
    }
    DEFAULT_MODE = "alpha"
    MODE_KEYWORDS = {
        "alpha": ["analyze", "inspect"],
        "beta": ["build", "compile"],
    }

    def __init__(self):
        super().__init__(AgentType.WORKER)

    def _execute_alpha(self, task: AgentTask) -> AgentResult:
        return AgentResult(success=True, output="alpha done")

    def _execute_beta(self, task: AgentTask) -> AgentResult:
        return AgentResult(success=True, output="beta done")


# ===================================================================
# MultiModeAgent.to_skill_spec tests
# ===================================================================


class TestToSkillSpec:
    def test_returns_skill_spec(self):
        spec = _TestMultiModeAgent.to_skill_spec()
        assert isinstance(spec, SkillSpec)

    def test_modes_match_class_modes(self):
        spec = _TestMultiModeAgent.to_skill_spec()
        assert spec.modes == ["alpha", "beta"]

    def test_capabilities_include_modes_and_keywords(self):
        spec = _TestMultiModeAgent.to_skill_spec()
        assert "alpha" in spec.capabilities
        assert "beta" in spec.capabilities
        assert "analyze" in spec.capabilities
        assert "build" in spec.capabilities

    def test_no_duplicate_capabilities(self):
        spec = _TestMultiModeAgent.to_skill_spec()
        assert len(spec.capabilities) == len(set(spec.capabilities))

    def test_description_from_docstring(self):
        spec = _TestMultiModeAgent.to_skill_spec()
        assert "test agent" in spec.description.lower()

    def test_auto_derived_tag(self):
        spec = _TestMultiModeAgent.to_skill_spec()
        assert "auto-derived" in spec.tags

    def test_agent_type_populated(self):
        spec = _TestMultiModeAgent.to_skill_spec()
        assert spec.agent_type  # non-empty


# ===================================================================
# Real agent to_skill_spec tests
# ===================================================================


class TestRealAgentSkillSpecs:
    def test_builder_agent_derives_spec(self):
        from vetinari.agents.builder_agent import BuilderAgent

        spec = BuilderAgent.to_skill_spec()
        assert isinstance(spec, SkillSpec)
        assert len(spec.modes) > 0

    def test_quality_agent_derives_spec(self):
        from vetinari.agents.consolidated.quality_agent import InspectorAgent

        spec = InspectorAgent.to_skill_spec()
        assert isinstance(spec, SkillSpec)
        assert "code_review" in spec.modes

    def test_researcher_agent_derives_spec(self):
        from vetinari.agents.consolidated.researcher_agent import ConsolidatedResearcherAgent

        spec = ConsolidatedResearcherAgent.to_skill_spec()
        assert isinstance(spec, SkillSpec)
        assert "code_discovery" in spec.modes


# ===================================================================
# auto_populate_from_agents tests
# ===================================================================


class TestAutoPopulateFromAgents:
    def test_returns_dict_of_skill_specs(self):
        from vetinari.skills.skill_registry import auto_populate_from_agents

        result = auto_populate_from_agents()
        assert isinstance(result, dict)
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, SkillSpec)

    def test_discovers_available_agents(self):
        from vetinari.skills.skill_registry import auto_populate_from_agents

        result = auto_populate_from_agents()
        # Must discover at least the core 3 agent types (Foreman, Worker, Inspector)
        assert len(result) >= 3

    def test_merges_with_existing_hand_written_specs(self):
        from vetinari.skills.skill_registry import SKILL_REGISTRY, auto_populate_from_agents

        result = auto_populate_from_agents()
        # Builder exists in hand-written registry
        if "builder" in SKILL_REGISTRY:
            merged = result.get("builder")
            assert merged is not None
            hand_written = SKILL_REGISTRY["builder"]
            # Hand-written standards should be preserved
            assert merged.standards == hand_written.standards

    def test_auto_derived_tag_present(self):
        from vetinari.skills.skill_registry import auto_populate_from_agents

        result = auto_populate_from_agents()
        for spec in result.values():
            assert "auto-derived" in spec.tags

    def test_hand_written_name_preserved(self):
        from vetinari.skills.skill_registry import SKILL_REGISTRY, auto_populate_from_agents

        result = auto_populate_from_agents()
        if "builder" in SKILL_REGISTRY:
            assert result["builder"].name == SKILL_REGISTRY["builder"].name
