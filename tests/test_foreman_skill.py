"""Tests for ForemanSkillTool — planning, clarification, and orchestration.

Covers all 6 modes (plan, clarify, consolidate, summarise, prune, extract),
mode validation, error handling, metadata, and ToolResult contract.
"""

from __future__ import annotations

import pytest

from vetinari.agents.contracts import Task
from vetinari.skills.foreman_skill import (
    ForemanMode,
    ForemanResult,
    ForemanSkillTool,
    make_plan_task,
)
from vetinari.tool_interface import ToolResult
from vetinari.types import AgentType

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def foreman():
    """Create a fresh ForemanSkillTool instance."""
    return ForemanSkillTool()


# ═══════════════════════════════════════════════════════════════════════════
# Initialization and Metadata
# ═══════════════════════════════════════════════════════════════════════════


class TestForemanInitialization:
    """Tests for ForemanSkillTool initialization and metadata."""

    def test_initialization(self, foreman):
        """ForemanSkillTool initializes with correct metadata."""
        assert foreman.metadata.name == "foreman"
        assert foreman.metadata.version == "2.0.0"

    def test_description_is_meaningful(self, foreman):
        """Description is meaningful (not a stub)."""
        assert len(foreman.metadata.description) > 20
        assert "planning" in foreman.metadata.description.lower()

    def test_parameters_include_goal(self, foreman):
        """The 'goal' parameter is required."""
        param_names = [p.name for p in foreman.metadata.parameters]
        assert "goal" in param_names
        goal_param = next(p for p in foreman.metadata.parameters if p.name == "goal")
        assert goal_param.required is True

    def test_parameters_include_mode(self, foreman):
        """The 'mode' parameter has all valid modes."""
        param_names = [p.name for p in foreman.metadata.parameters]
        assert "mode" in param_names
        mode_param = next(p for p in foreman.metadata.parameters if p.name == "mode")
        assert set(mode_param.allowed_values) == {m.value for m in ForemanMode}

    def test_parameters_include_context(self, foreman):
        """The 'context' parameter is optional."""
        param_names = [p.name for p in foreman.metadata.parameters]
        assert "context" in param_names

    def test_tags(self, foreman):
        """Tags include planning-related keywords."""
        assert "planning" in foreman.metadata.tags
        assert "orchestration" in foreman.metadata.tags


# ═══════════════════════════════════════════════════════════════════════════
# Mode Execution — Happy Path
# ═══════════════════════════════════════════════════════════════════════════


class TestForemanModes:
    """Tests for each Foreman mode."""

    def test_plan_mode(self, foreman):
        """Plan mode returns a successful ToolResult with goal."""
        result = foreman.execute(goal="Build user auth system", mode="plan")
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output is not None
        assert result.output["goal"] == "Build user auth system"
        assert result.metadata["mode"] == "plan"
        assert result.metadata["agent"] == AgentType.FOREMAN.value

    def test_clarify_mode(self, foreman):
        """Clarify mode returns a successful result."""
        result = foreman.execute(goal="Add caching", mode="clarify")
        assert result.success is True
        assert result.output["goal"] == "Add caching"
        assert result.metadata["mode"] == "clarify"

    def test_consolidate_mode(self, foreman):
        """Consolidate mode returns a summary."""
        result = foreman.execute(goal="Merge contexts", mode="consolidate")
        assert result.success is True
        assert result.output["summary"] == "Context consolidated"

    def test_summarise_mode(self, foreman):
        """Summarise mode returns a successful result."""
        result = foreman.execute(goal="Session summary", mode="summarise")
        assert result.success is True
        assert result.metadata["mode"] == "summarise"

    def test_prune_mode(self, foreman):
        """Prune mode returns a successful result."""
        result = foreman.execute(goal="Reduce context", mode="prune")
        assert result.success is True
        assert result.metadata["mode"] == "prune"

    def test_extract_mode(self, foreman):
        """Extract mode returns a successful result."""
        result = foreman.execute(goal="Extract patterns", mode="extract")
        assert result.success is True
        assert result.metadata["mode"] == "extract"

    def test_default_mode_is_plan(self, foreman):
        """When no mode is specified, defaults to 'plan'."""
        result = foreman.execute(goal="Do something")
        assert result.success is True
        assert result.metadata["mode"] == "plan"


# ═══════════════════════════════════════════════════════════════════════════
# Error Handling
# ═══════════════════════════════════════════════════════════════════════════


class TestForemanErrors:
    """Tests for error handling in ForemanSkillTool."""

    def test_invalid_mode(self, foreman):
        """Invalid mode returns error ToolResult."""
        result = foreman.execute(goal="test", mode="nonexistent")
        assert result.success is False
        assert result.error is not None
        assert "Unknown mode" in result.error

    def test_empty_goal(self, foreman):
        """Empty goal still succeeds (foreman can plan with empty input)."""
        result = foreman.execute(goal="")
        assert result.success is True

    def test_missing_goal_uses_default(self, foreman):
        """Missing goal parameter uses empty string default."""
        result = foreman.execute()
        assert result.success is True

    def test_extra_kwargs_ignored(self, foreman):
        """Extra keyword arguments are gracefully ignored."""
        result = foreman.execute(goal="test", mode="plan", unknown_param="value")
        assert result.success is True


# ═══════════════════════════════════════════════════════════════════════════
# Context Parameter
# ═══════════════════════════════════════════════════════════════════════════


class TestForemanContext:
    """Tests for context parameter handling."""

    def test_context_accepted(self, foreman):
        """Context parameter is accepted without error."""
        result = foreman.execute(
            goal="Plan refactor",
            mode="plan",
            context={"prior_plan": "existing plan here"},
        )
        assert result.success is True

    def test_empty_context(self, foreman):
        """Empty context dict works fine."""
        result = foreman.execute(goal="Plan", mode="plan", context={})
        assert result.success is True


# ═══════════════════════════════════════════════════════════════════════════
# Thinking Mode Metadata
# ═══════════════════════════════════════════════════════════════════════════


class TestForemanThinkingModes:
    """Tests for thinking mode assignment per mode."""

    def test_plan_uses_xhigh_thinking(self, foreman):
        """Plan mode uses XHIGH thinking budget."""
        result = foreman.execute(goal="Complex plan", mode="plan")
        assert result.output["metadata"]["thinking_mode"] == "xhigh"

    def test_clarify_uses_high_thinking(self, foreman):
        """Clarify mode uses HIGH thinking budget."""
        result = foreman.execute(goal="Questions", mode="clarify")
        assert result.output["metadata"]["thinking_mode"] == "high"

    def test_prune_uses_low_thinking(self, foreman):
        """Prune mode uses LOW thinking budget."""
        result = foreman.execute(goal="Prune", mode="prune")
        assert result.output["metadata"]["thinking_mode"] == "low"


# ═══════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════


class TestMakePlanTask:
    """Tests for make_plan_task factory (replaces retired PlanTask)."""

    def test_make_plan_task_defaults(self):
        """make_plan_task produces a Task with sensible defaults."""
        task = make_plan_task("T-001", "Test task", AgentType.WORKER.value)
        assert isinstance(task, Task)
        assert task.id == "T-001"
        assert task.assigned_agent == AgentType.WORKER
        assert task.dependencies == []
        assert task.inputs == []
        assert task.outputs == []

    def test_make_plan_task_with_metadata(self):
        """Planning metadata (effort, mode) is embedded in Task.metadata."""
        task = make_plan_task(
            "T-002",
            "Build feature",
            AgentType.WORKER,
            dependencies=["T-001"],
            effort="L",
            mode="build",
        )
        d = task.to_dict()
        assert d["id"] == "T-002"
        assert d["dependencies"] == ["T-001"]
        assert d["metadata"]["effort"] == "L"
        assert d["metadata"]["mode"] == "build"

    def test_make_plan_task_full_fields(self):
        """make_plan_task with all fields populated serializes correctly."""
        task = make_plan_task(
            "T-003",
            "Research APIs",
            AgentType.WORKER.value,
            dependencies=["T-001", "T-002"],
            inputs=["requirements.md"],
            outputs=["api_report.md"],
            effort="XL",
            acceptance_criteria="Report covers all endpoints",
            mode="api_lookup",
        )
        d = task.to_dict()
        assert len(d["dependencies"]) == 2
        assert d["inputs"] == ["requirements.md"]
        assert d["metadata"]["acceptance_criteria"] == "Report covers all endpoints"


class TestForemanResult:
    """Tests for ForemanResult dataclass."""

    def test_foreman_result_defaults(self):
        """ForemanResult has empty defaults."""
        r = ForemanResult()
        assert r.plan_id == ""
        assert r.goal == ""
        assert r.tasks == []
        assert r.risks == []
        assert r.questions == []
        assert r.summary == ""
        assert r.metadata == {}

    def test_foreman_result_to_dict(self):
        """ForemanResult.to_dict() serializes all fields."""
        task = make_plan_task("T-1", "Do thing", AgentType.WORKER.value)
        r = ForemanResult(
            plan_id="P-001",
            goal="Build auth",
            tasks=[task],
            risks=["Scope creep"],
            questions=["Which OAuth provider?"],
            summary="Plan ready",
            metadata={"version": 1},
        )
        d = r.to_dict()
        assert d["plan_id"] == "P-001"
        assert len(d["tasks"]) == 1
        assert d["tasks"][0]["id"] == "T-1"
        assert d["risks"] == ["Scope creep"]
        assert d["questions"] == ["Which OAuth provider?"]


# ═══════════════════════════════════════════════════════════════════════════
# ForemanMode Enum
# ═══════════════════════════════════════════════════════════════════════════


class TestForemanModeEnum:
    """Tests for ForemanMode enum."""

    def test_has_six_modes(self):
        """ForemanMode has exactly 6 values."""
        assert len(ForemanMode) == 6

    def test_expected_modes(self):
        """All expected modes exist."""
        expected = {"plan", "clarify", "consolidate", "summarise", "prune", "extract"}
        actual = {m.value for m in ForemanMode}
        assert actual == expected

    def test_mode_is_str_enum(self):
        """ForemanMode values are strings."""
        for mode in ForemanMode:
            assert isinstance(mode.value, str)


# ═══════════════════════════════════════════════════════════════════════════
# All Modes Execute Successfully
# ═══════════════════════════════════════════════════════════════════════════


class TestForemanAllModes:
    """Verify every mode executes without error."""

    @pytest.mark.parametrize("mode", [m.value for m in ForemanMode])
    def test_all_modes_succeed(self, foreman, mode):
        """Every mode returns success=True with a valid goal."""
        result = foreman.execute(goal="Test goal for mode execution", mode=mode)
        assert result.success is True
        assert result.metadata["mode"] == mode
        assert result.metadata["agent"] == AgentType.FOREMAN.value
