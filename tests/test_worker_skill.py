"""Tests for WorkerSkillTool — all-purpose execution across 25 modes.

Covers mode resolution, mode group detection, constraint checking,
delegation to component skills, metadata, and ToolResult contract.
"""

from __future__ import annotations

import pytest

from vetinari.skills.worker_skill import (
    GROUP_THINKING_BUDGET,
    MODE_TO_GROUP,
    WorkerModeGroup,
    WorkerResult,
    WorkerSkillTool,
)
from vetinari.tool_interface import ToolResult
from vetinari.types import AgentType, ThinkingMode

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def worker():
    """Create a fresh WorkerSkillTool instance."""
    return WorkerSkillTool()


# ═══════════════════════════════════════════════════════════════════════════
# Initialization and Metadata
# ═══════════════════════════════════════════════════════════════════════════


class TestWorkerInitialization:
    """Tests for WorkerSkillTool initialization and metadata."""

    def test_initialization(self, worker):
        """WorkerSkillTool initializes with correct metadata."""
        assert worker.metadata.name == "worker"
        assert worker.metadata.version == "2.0.0"

    def test_description_is_meaningful(self, worker):
        """Description is meaningful."""
        assert len(worker.metadata.description) > 20
        assert "execution" in worker.metadata.description.lower()

    def test_task_parameter_required(self, worker):
        """The 'task' parameter is required."""
        param_names = [p.name for p in worker.metadata.parameters]
        assert "task" in param_names
        task_param = next(p for p in worker.metadata.parameters if p.name == "task")
        assert task_param.required is True

    def test_mode_parameter_lists_all_modes(self, worker):
        """The 'mode' parameter lists all valid modes."""
        mode_param = next(p for p in worker.metadata.parameters if p.name == "mode")
        assert set(mode_param.allowed_values) == set(MODE_TO_GROUP.keys())

    def test_tags_cover_all_groups(self, worker):
        """Tags cover all 4 mode groups."""
        expected = {"research", "architecture", "build", "operations"}
        assert expected.issubset(set(worker.metadata.tags))

    def test_all_modes_class_attribute(self, worker):
        """ALL_MODES contains all mode keys."""
        assert set(worker.ALL_MODES) == set(MODE_TO_GROUP.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Mode Groups
# ═══════════════════════════════════════════════════════════════════════════


class TestModeGroups:
    """Tests for mode-to-group mapping."""

    def test_research_modes(self):
        """All research modes map to RESEARCH group."""
        research_modes = [
            "code_discovery",
            "domain_research",
            "api_lookup",
            "lateral_thinking",
            "ui_design",
            "database",
            "devops",
            "git_workflow",
        ]
        for mode in research_modes:
            assert MODE_TO_GROUP[mode] == WorkerModeGroup.RESEARCH

    def test_architecture_modes(self):
        """All architecture modes map to ARCHITECTURE group."""
        arch_modes = [
            "architecture",
            "risk_assessment",
            "ontological_analysis",
            "contrarian_review",
            "suggest",
        ]
        for mode in arch_modes:
            assert MODE_TO_GROUP[mode] == WorkerModeGroup.ARCHITECTURE

    def test_build_modes(self):
        """Build modes map to BUILD group."""
        assert MODE_TO_GROUP["build"] == WorkerModeGroup.BUILD
        assert MODE_TO_GROUP["image_generation"] == WorkerModeGroup.BUILD

    def test_operations_modes(self):
        """All operations modes map to OPERATIONS group."""
        ops_modes = [
            "documentation",
            "creative_writing",
            "cost_analysis",
            "experiment",
            "error_recovery",
            "synthesis",
            "improvement",
            "monitor",
            "devops_ops",
        ]
        for mode in ops_modes:
            assert MODE_TO_GROUP[mode] == WorkerModeGroup.OPERATIONS

    def test_total_mode_count(self):
        """Total modes across all groups: 8 research + 5 arch + 2 build + 9 ops = 24."""
        assert len(MODE_TO_GROUP) == 24

    def test_get_mode_group_static(self, worker):
        """get_mode_group() returns correct group name."""
        assert WorkerSkillTool.get_mode_group("build") == "build"
        assert WorkerSkillTool.get_mode_group("architecture") == "architecture"
        assert WorkerSkillTool.get_mode_group("code_discovery") == "research"
        assert WorkerSkillTool.get_mode_group("documentation") == "operations"
        assert WorkerSkillTool.get_mode_group("nonexistent") is None


# ═══════════════════════════════════════════════════════════════════════════
# Mode Resolution
# ═══════════════════════════════════════════════════════════════════════════


class TestModeResolution:
    """Tests for keyword-based mode resolution."""

    def test_research_keywords(self, worker):
        """Research keywords resolve to code_discovery."""
        assert worker._resolve_mode("research the codebase") == "code_discovery"
        assert worker._resolve_mode("explore the auth module") == "code_discovery"

    def test_architecture_keywords(self, worker):
        """Architecture keywords resolve to architecture."""
        assert worker._resolve_mode("design the system architecture") == "architecture"
        assert worker._resolve_mode("create an ADR") == "architecture"

    def test_security_keywords(self, worker):
        """Security keywords resolve to architecture."""
        assert worker._resolve_mode("check for security vulnerabilities") == "architecture"

    def test_database_keywords(self, worker):
        """Database keywords resolve to database."""
        assert worker._resolve_mode("analyze the database schema") == "database"
        assert worker._resolve_mode("write a SQL migration") == "database"

    def test_build_keywords(self, worker):
        """Build keywords resolve to build."""
        assert worker._resolve_mode("implement the new feature") == "build"
        assert worker._resolve_mode("create the login page") == "build"

    def test_documentation_keywords(self, worker):
        """Documentation keywords resolve to documentation."""
        assert worker._resolve_mode("generate the README") == "documentation"
        assert worker._resolve_mode("write a changelog") == "documentation"

    def test_cost_keywords(self, worker):
        """Cost keywords resolve to cost_analysis."""
        assert worker._resolve_mode("analyze token cost") == "cost_analysis"

    def test_default_is_build(self, worker):
        """Unknown tasks default to build mode."""
        assert worker._resolve_mode("do something vague") == "build"

    def test_git_keywords(self, worker):
        """Git keywords resolve to git_workflow."""
        assert worker._resolve_mode("run git blame on auth.py") == "git_workflow"

    def test_monitor_keywords(self, worker):
        """Monitor keywords resolve to monitor."""
        assert worker._resolve_mode("set up monitoring alerts") == "monitor"


# ═══════════════════════════════════════════════════════════════════════════
# Execution — Happy Path
# ═══════════════════════════════════════════════════════════════════════════


class TestWorkerExecution:
    """Tests for WorkerSkillTool execution."""

    def test_explicit_mode(self, worker):
        """Explicit mode is used when provided."""
        result = worker.execute(task="Build feature", mode="build")
        assert result.success is True
        assert result.metadata["mode"] == "build"
        assert result.metadata["mode_group"] == "build"
        assert result.metadata["agent"] == AgentType.WORKER.value

    def test_auto_resolve_mode(self, worker):
        """Mode is auto-resolved from task description when omitted."""
        result = worker.execute(task="design the architecture")
        assert result.success is True
        assert result.metadata["mode"] == "architecture"

    def test_output_is_worker_result_dict(self, worker):
        """Output is a serialized WorkerResult."""
        result = worker.execute(task="test", mode="build")
        assert result.output is not None
        assert "success" in result.output
        assert "metadata" in result.output
        assert "errors" in result.output
        assert "warnings" in result.output

    def test_files_parameter_passed_through(self, worker):
        """Files parameter is included in result metadata."""
        result = worker.execute(task="Review", mode="build", files=["main.py"])
        assert result.output["metadata"]["files"] == ["main.py"]

    def test_thinking_mode_override(self, worker):
        """Explicit thinking_mode overrides group default."""
        result = worker.execute(task="Quick build", mode="build", thinking_mode="low")
        assert result.output["metadata"]["thinking_mode"] == "low"


# ═══════════════════════════════════════════════════════════════════════════
# Error Handling
# ═══════════════════════════════════════════════════════════════════════════


class TestWorkerErrors:
    """Tests for error handling in WorkerSkillTool."""

    def test_invalid_mode(self, worker):
        """Invalid mode returns error ToolResult."""
        result = worker.execute(task="test", mode="nonexistent_mode")
        assert result.success is False
        assert "Unknown mode" in result.error

    def test_empty_task(self, worker):
        """Empty task still succeeds (mode determines behavior)."""
        result = worker.execute(task="", mode="build")
        assert result.success is True

    def test_missing_task_uses_default(self, worker):
        """Missing task parameter uses empty string."""
        result = worker.execute()
        assert result.success is True


# ═══════════════════════════════════════════════════════════════════════════
# Thinking Budgets
# ═══════════════════════════════════════════════════════════════════════════


class TestThinkingBudgets:
    """Tests for thinking budget per mode group."""

    def test_research_gets_medium(self):
        """Research group gets MEDIUM thinking budget."""
        assert GROUP_THINKING_BUDGET[WorkerModeGroup.RESEARCH] == ThinkingMode.MEDIUM

    def test_architecture_gets_high(self):
        """Architecture group gets HIGH thinking budget."""
        assert GROUP_THINKING_BUDGET[WorkerModeGroup.ARCHITECTURE] == ThinkingMode.HIGH

    def test_build_gets_high(self):
        """Build group gets HIGH thinking budget."""
        assert GROUP_THINKING_BUDGET[WorkerModeGroup.BUILD] == ThinkingMode.HIGH

    def test_operations_gets_medium(self):
        """Operations group gets MEDIUM thinking budget."""
        assert GROUP_THINKING_BUDGET[WorkerModeGroup.OPERATIONS] == ThinkingMode.MEDIUM


# ═══════════════════════════════════════════════════════════════════════════
# Delegation — Architecture and Operations
# ═══════════════════════════════════════════════════════════════════════════


class TestDelegation:
    """Tests for delegation to component skill tools."""

    def test_architecture_mode_delegates(self, worker):
        """Architecture mode result includes delegation metadata."""
        result = worker.execute(task="Design the auth module", mode="architecture")
        assert result.success is True
        output = result.output
        assert output["metadata"]["delegation"] == "architect_skill"
        assert output["metadata"]["mode_group"] == "architecture"

    def test_operations_mode_delegates(self, worker):
        """Operations mode result includes delegation metadata."""
        result = worker.execute(task="Write docs for the API", mode="documentation")
        assert result.success is True
        output = result.output
        assert output["metadata"]["delegation"] == "operations_skill"
        assert output["metadata"]["mode_group"] == "operations"

    def test_build_mode_uses_pipeline(self, worker):
        """Build mode uses agent_pipeline (no component delegation)."""
        result = worker.execute(task="Implement feature", mode="build")
        assert result.success is True
        assert result.output["metadata"]["delegation"] == "agent_pipeline"

    def test_research_mode_uses_pipeline(self, worker):
        """Research mode uses agent_pipeline (no component delegation)."""
        result = worker.execute(task="Explore codebase", mode="code_discovery")
        assert result.success is True
        assert result.output["metadata"]["delegation"] == "agent_pipeline"

    def test_architecture_delegation_failure_handled(self, worker):
        """Architecture delegation handles component tool exceptions."""
        from unittest.mock import MagicMock, patch

        mock_tool = MagicMock()
        mock_tool.execute.side_effect = RuntimeError("Architect exploded")
        worker._architect_tool = mock_tool
        result = worker.execute(task="Design system", mode="architecture")
        assert result.success is False
        assert "Architect exploded" in result.error

    def test_operations_delegation_failure_handled(self, worker):
        """Operations delegation handles component tool exceptions."""
        from unittest.mock import MagicMock

        mock_tool = MagicMock()
        mock_tool.execute.side_effect = RuntimeError("Operations broke")
        worker._operations_tool = mock_tool
        result = worker.execute(task="Generate docs", mode="documentation")
        assert result.success is False
        assert "Operations broke" in result.error

    def test_architecture_tool_cached(self, worker):
        """ArchitectSkillTool is created once and reused."""
        worker.execute(task="Design A", mode="architecture")
        first_tool = worker._architect_tool
        worker.execute(task="Design B", mode="risk_assessment")
        assert worker._architect_tool is first_tool

    def test_operations_tool_cached(self, worker):
        """OperationsSkillTool is created once and reused."""
        worker.execute(task="Write docs A", mode="documentation")
        first_tool = worker._operations_tool
        worker.execute(task="Write docs B", mode="synthesis")
        assert worker._operations_tool is first_tool


# ═══════════════════════════════════════════════════════════════════════════
# Constraint Checking
# ═══════════════════════════════════════════════════════════════════════════


class TestConstraints:
    """Tests for mode-group constraint checking."""

    def test_constraint_check_returns_empty(self, worker):
        """Constraint check returns empty list for valid modes."""
        errors = worker._check_constraints("build", WorkerModeGroup.BUILD)
        assert errors == []

    def test_research_mode_check_does_not_error(self, worker):
        """Research mode constraint check handles missing context manager gracefully."""
        errors = worker._check_constraints("code_discovery", WorkerModeGroup.RESEARCH)
        assert errors == []

    def test_architecture_mode_check_does_not_error(self, worker):
        """Architecture mode constraint check handles missing context gracefully."""
        errors = worker._check_constraints("architecture", WorkerModeGroup.ARCHITECTURE)
        assert errors == []


# ═══════════════════════════════════════════════════════════════════════════
# WorkerResult Dataclass
# ═══════════════════════════════════════════════════════════════════════════


class TestWorkerResult:
    """Tests for WorkerResult dataclass."""

    def test_defaults(self):
        """WorkerResult has sensible defaults."""
        r = WorkerResult()
        assert r.success is True
        assert r.output is None
        assert r.files_changed == []
        assert r.errors == []
        assert r.warnings == []
        assert r.metadata == {}
        assert r.provenance == []

    def test_to_dict(self):
        """WorkerResult.to_dict() serializes all fields."""
        r = WorkerResult(
            success=True,
            output={"key": "value"},
            files_changed=["main.py"],
            errors=[],
            warnings=["minor issue"],
            metadata={"mode": "build"},
            provenance=[{"source": "agent"}],
        )
        d = r.to_dict()
        assert d["success"] is True
        assert d["output"] == {"key": "value"}
        assert d["files_changed"] == ["main.py"]
        assert d["warnings"] == ["minor issue"]
        assert d["provenance"] == [{"source": "agent"}]

    def test_failure_result(self):
        """WorkerResult can represent failures."""
        r = WorkerResult(success=False, errors=["Something broke"])
        assert r.success is False
        assert r.errors == ["Something broke"]


# ═══════════════════════════════════════════════════════════════════════════
# WorkerModeGroup Enum
# ═══════════════════════════════════════════════════════════════════════════


class TestWorkerModeGroupEnum:
    """Tests for WorkerModeGroup enum."""

    def test_has_four_groups(self):
        """WorkerModeGroup has exactly 4 values."""
        assert len(WorkerModeGroup) == 4

    def test_expected_groups(self):
        """All expected groups exist."""
        expected = {"research", "architecture", "build", "operations"}
        actual = {g.value for g in WorkerModeGroup}
        assert actual == expected


# ═══════════════════════════════════════════════════════════════════════════
# All Modes Execute Successfully
# ═══════════════════════════════════════════════════════════════════════════


class TestWorkerAllModes:
    """Verify every mode executes without error."""

    @pytest.mark.parametrize("mode", list(MODE_TO_GROUP.keys()))
    def test_all_modes_succeed(self, worker, mode):
        """Every mode returns success=True with a valid task."""
        result = worker.execute(task="Execute mode test", mode=mode)
        assert result.success is True
        assert result.metadata["mode"] == mode
        assert result.metadata["agent"] == AgentType.WORKER.value
