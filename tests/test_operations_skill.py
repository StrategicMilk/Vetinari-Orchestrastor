"""
Unit tests for the unified Operations Skill Tool.

Tests cover:
- Tool initialization and metadata
- All 8 operations modes (documentation, creative_writing, cost_analysis, experiment,
  error_recovery, synthesis, image_generation, improvement)
- Output format variations
- Thinking mode depth
- Parameter validation
- Dataclass serialization
- Error cases
"""

import pytest
from unittest.mock import Mock

from vetinari.skills.operations_skill import (
    OperationsSkillTool,
    OperationsMode,
    OutputFormat,
    ThinkingMode,
    OperationsRequest,
    Section,
    OperationsResult,
)
from vetinari.execution_context import (
    ToolPermission,
    ExecutionMode,
    ExecutionContext,
)
from vetinari.tool_interface import ToolResult


class TestOperationsSkillToolMetadata:
    """Tests for operations skill metadata and initialization."""

    def test_initialization(self):
        """Test OperationsSkillTool initializes with correct metadata."""
        tool = OperationsSkillTool()

        assert tool.metadata.name == "operations"
        assert "documentation" in tool.metadata.tags
        assert "operations" in tool.metadata.tags
        assert "recovery" in tool.metadata.tags
        assert ToolPermission.FILE_READ in tool.metadata.required_permissions
        assert ToolPermission.FILE_WRITE in tool.metadata.required_permissions
        assert ToolPermission.MODEL_INFERENCE in tool.metadata.required_permissions

    def test_allowed_execution_modes(self):
        """Test allowed execution modes."""
        tool = OperationsSkillTool()

        assert ExecutionMode.EXECUTION in tool.metadata.allowed_modes
        assert ExecutionMode.PLANNING in tool.metadata.allowed_modes

    def test_tool_parameters(self):
        """Test tool parameters are properly defined."""
        tool = OperationsSkillTool()

        param_names = {p.name for p in tool.metadata.parameters}
        assert "mode" in param_names
        assert "content" in param_names
        assert "context" in param_names
        assert "output_format" in param_names
        assert "thinking_mode" in param_names

    def test_mode_parameter_has_all_modes(self):
        """Test mode parameter has all OperationsMode values."""
        tool = OperationsSkillTool()

        mode_param = next(p for p in tool.metadata.parameters if p.name == "mode")

        assert mode_param.required is True
        assert all(m.value in mode_param.allowed_values for m in OperationsMode)

    def test_output_format_parameter(self):
        """Test output_format parameter defaults and values."""
        tool = OperationsSkillTool()

        fmt_param = next(
            p for p in tool.metadata.parameters if p.name == "output_format"
        )

        assert fmt_param.required is False
        assert fmt_param.default == "markdown"
        assert all(f.value in fmt_param.allowed_values for f in OutputFormat)

    def test_version(self):
        """Test tool version."""
        tool = OperationsSkillTool()
        assert tool.metadata.version == "1.1.0"


class TestOperationsSkillToolExecution:
    """Tests for operations skill execution logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = OperationsSkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock(spec=ExecutionContext)
        self.mock_ctx_manager.current_context = self.mock_context
        self.mock_context.execution_mode = ExecutionMode.EXECUTION
        self.mock_context.pre_execution_hooks = []
        self.mock_context.post_execution_hooks = []
        self.tool._context_manager = self.mock_ctx_manager

    def test_documentation_mode(self):
        """Test documentation generation mode."""
        result = self.tool.execute(
            mode="documentation",
            content="Vetinari skill system API",
        )

        assert result.success is True
        output = result.output
        assert output["success"] is True
        assert output["type"] == "documentation"
        assert len(output["sections"]) >= 4
        section_titles = [s["title"] for s in output["sections"]]
        assert "Overview" in section_titles
        assert "Usage" in section_titles
        assert "API Reference" in section_titles

    def test_documentation_high_thinking_adds_sections(self):
        """Test documentation with high thinking mode adds extra sections."""
        result = self.tool.execute(
            mode="documentation",
            content="API docs",
            thinking_mode="high",
        )

        assert result.success is True
        section_titles = [s["title"] for s in result.output["sections"]]
        assert "Troubleshooting" in section_titles
        assert "Migration Guide" in section_titles

    def test_documentation_low_thinking_fewer_sections(self):
        """Test documentation with low thinking mode has fewer sections."""
        result = self.tool.execute(
            mode="documentation",
            content="API docs",
            thinking_mode="low",
        )

        assert result.success is True
        section_titles = [s["title"] for s in result.output["sections"]]
        assert "Troubleshooting" not in section_titles

    def test_creative_writing(self):
        """Test creative writing mode."""
        result = self.tool.execute(
            mode="creative_writing",
            content="A story about AI agents collaborating",
        )

        assert result.success is True
        assert result.output["type"] == "creative_writing"
        assert result.output["metadata"]["word_count"] > 0

    def test_cost_analysis(self):
        """Test cost analysis mode."""
        result = self.tool.execute(
            mode="cost_analysis",
            content="Monthly infrastructure spending review",
        )

        assert result.success is True
        assert result.output["type"] == "cost_analysis"
        section_titles = [s["title"] for s in result.output["sections"]]
        assert "Current Cost" in section_titles
        assert "Recommendations" in section_titles

    def test_experiment(self):
        """Test experiment design mode."""
        result = self.tool.execute(
            mode="experiment",
            content="A/B test for new prompt template",
        )

        assert result.success is True
        assert result.output["type"] == "experiment"
        section_titles = [s["title"] for s in result.output["sections"]]
        assert "Hypothesis" in section_titles
        assert "Methodology" in section_titles
        assert "Success Criteria" in section_titles
        assert "Rollback Plan" in section_titles

    def test_error_recovery(self):
        """Test error recovery mode."""
        result = self.tool.execute(
            mode="error_recovery",
            content="Database connection timeout in production",
        )

        assert result.success is True
        assert result.output["type"] == "error_recovery"
        section_titles = [s["title"] for s in result.output["sections"]]
        assert "Root Cause Analysis" in section_titles
        assert "Fix Steps" in section_titles
        assert "Prevention" in section_titles
        # Should include production warning
        assert len(result.output["warnings"]) > 0

    def test_synthesis(self):
        """Test information synthesis mode."""
        result = self.tool.execute(
            mode="synthesis",
            content="Combine research findings from 3 papers",
        )

        assert result.success is True
        assert result.output["type"] == "synthesis"
        section_titles = [s["title"] for s in result.output["sections"]]
        assert "Summary" in section_titles
        assert "Key Insights" in section_titles
        assert "Conflicts" in section_titles

    def test_image_generation(self):
        """Test image generation mode."""
        result = self.tool.execute(
            mode="image_generation",
            content="Dashboard mockup with analytics charts",
        )

        assert result.success is True
        assert result.output["type"] == "image_generation"
        assert result.output["metadata"]["status"] == "prompt_prepared"
        # Should warn about external API requirement
        assert len(result.output["warnings"]) > 0

    def test_improvement(self):
        """Test improvement analysis mode."""
        result = self.tool.execute(
            mode="improvement",
            content="Analyze codebase for tech debt",
        )

        assert result.success is True
        assert result.output["type"] == "improvement"
        section_titles = [s["title"] for s in result.output["sections"]]
        assert "Current State" in section_titles
        assert "Tech Debt" in section_titles
        assert "Quick Wins" in section_titles
        assert "Strategic Improvements" in section_titles

    def test_invalid_mode(self):
        """Test execution with invalid mode returns error."""
        result = self.tool.execute(
            mode="nonexistent",
            content="something",
        )

        assert result.success is False
        assert "Invalid mode" in result.error

    def test_missing_content(self):
        """Test execution without content returns error."""
        result = self.tool.execute(
            mode="documentation",
        )

        assert result.success is False
        assert "content" in result.error.lower()

    def test_invalid_thinking_mode_falls_back(self):
        """Test invalid thinking mode falls back to medium."""
        result = self.tool.execute(
            mode="documentation",
            content="Test docs",
            thinking_mode="invalid",
        )

        # The implementation falls back to MEDIUM for invalid thinking mode
        assert result.success is True

    def test_invalid_output_format_falls_back(self):
        """Test invalid output format falls back to markdown."""
        result = self.tool.execute(
            mode="documentation",
            content="Test docs",
            output_format="invalid",
        )

        # The implementation falls back to MARKDOWN for invalid format
        assert result.success is True

    def test_context_parameter(self):
        """Test context parameter is accepted."""
        result = self.tool.execute(
            mode="error_recovery",
            content="Fix deployment issue",
            context="Running on Kubernetes cluster",
        )

        assert result.success is True

    def test_metadata_in_result(self):
        """Test metadata is included in ToolResult."""
        result = self.tool.execute(
            mode="documentation",
            content="API docs",
            output_format="json",
        )

        assert result.metadata["mode"] == "documentation"
        assert result.metadata["output_format"] == "json"
        assert "sections_count" in result.metadata


class TestOperationsModeEnum:
    """Tests for OperationsMode enum."""

    def test_all_modes_have_values(self):
        """Test all modes have valid string values."""
        for mode in OperationsMode:
            assert isinstance(mode.value, str)
            assert len(mode.value) > 0

    def test_expected_modes_exist(self):
        """Test expected modes are defined."""
        expected = {
            "documentation", "creative_writing", "cost_analysis", "experiment",
            "error_recovery", "synthesis", "image_generation", "improvement",
        }
        actual = {m.value for m in OperationsMode}
        assert actual == expected

    def test_mode_count(self):
        """Test correct number of modes."""
        assert len(OperationsMode) == 8


class TestOutputFormatEnum:
    """Tests for OutputFormat enum."""

    def test_all_formats(self):
        """Test all output formats exist."""
        expected = {"markdown", "html", "plain", "json"}
        actual = {f.value for f in OutputFormat}
        assert actual == expected


class TestOperationsDataclasses:
    """Tests for operations dataclasses."""

    def test_request_defaults(self):
        """Test OperationsRequest default values."""
        req = OperationsRequest(
            mode=OperationsMode.DOCUMENTATION,
            content="Test content",
        )

        assert req.context is None
        assert req.output_format == OutputFormat.MARKDOWN
        assert req.thinking_mode == ThinkingMode.MEDIUM

    def test_request_full(self):
        """Test OperationsRequest with all fields."""
        req = OperationsRequest(
            mode=OperationsMode.COST_ANALYSIS,
            content="Analyze costs",
            context="Q4 budget review",
            output_format=OutputFormat.JSON,
            thinking_mode=ThinkingMode.HIGH,
        )

        assert req.mode == OperationsMode.COST_ANALYSIS
        assert req.context == "Q4 budget review"
        assert req.output_format == OutputFormat.JSON

    def test_request_to_dict(self):
        """Test OperationsRequest serialization."""
        req = OperationsRequest(
            mode=OperationsMode.EXPERIMENT,
            content="Run experiment",
            thinking_mode=ThinkingMode.XHIGH,
        )

        d = req.to_dict()
        assert d["mode"] == "experiment"
        assert d["content"] == "Run experiment"
        assert d["thinking_mode"] == "xhigh"
        assert d["output_format"] == "markdown"
        assert d["context"] is None

    def test_section_to_dict(self):
        """Test Section serialization."""
        section = Section(title="Overview", content="This is an overview", order=1)

        d = section.to_dict()
        assert d["title"] == "Overview"
        assert d["content"] == "This is an overview"
        assert d["order"] == 1

    def test_section_defaults(self):
        """Test Section default values."""
        section = Section(title="Test", content="Content")

        assert section.order == 0

    def test_result_success(self):
        """Test OperationsResult success case."""
        result = OperationsResult(
            success=True,
            content="Generated content",
            content_type="documentation",
            output_format=OutputFormat.MARKDOWN,
            sections=[Section(title="Main", content="Body", order=1)],
        )

        assert result.success is True
        assert len(result.sections) == 1
        assert result.warnings == []
        assert result.metadata == {}

    def test_result_to_dict(self):
        """Test OperationsResult serialization."""
        result = OperationsResult(
            success=True,
            content="Test content",
            content_type="synthesis",
            output_format=OutputFormat.HTML,
            sections=[
                Section(title="A", content="Body A", order=1),
                Section(title="B", content="Body B", order=2),
            ],
            metadata={"source_count": 3},
            warnings=["Review needed"],
        )

        d = result.to_dict()
        assert d["success"] is True
        assert d["content"] == "Test content"
        assert d["type"] == "synthesis"
        assert d["format"] == "html"
        assert len(d["sections"]) == 2
        assert d["metadata"]["source_count"] == 3
        assert d["warnings"] == ["Review needed"]

    def test_result_failure(self):
        """Test OperationsResult failure case."""
        result = OperationsResult(
            success=False,
            content="Unknown mode: invalid",
        )

        assert result.success is False
        assert result.sections == []


class TestOperationsSkillToolParameterValidation:
    """Tests for input parameter validation."""

    def test_valid_parameters(self):
        """Test valid parameters pass validation."""
        tool = OperationsSkillTool()

        is_valid, error = tool.validate_inputs({
            "mode": "documentation",
            "content": "Generate docs",
        })

        assert is_valid is True
        assert error is None

    def test_missing_required_mode(self):
        """Test missing mode is caught."""
        tool = OperationsSkillTool()

        is_valid, error = tool.validate_inputs({
            "content": "Some content",
        })

        assert is_valid is False
        assert "mode" in error.lower()

    def test_missing_required_content(self):
        """Test missing content is caught."""
        tool = OperationsSkillTool()

        is_valid, error = tool.validate_inputs({
            "mode": "documentation",
        })

        assert is_valid is False
        assert "content" in error.lower()

    def test_invalid_mode_value(self):
        """Test invalid mode value is caught."""
        tool = OperationsSkillTool()

        is_valid, error = tool.validate_inputs({
            "mode": "nonexistent",
            "content": "test",
        })

        assert is_valid is False

    def test_optional_params_omitted(self):
        """Test optional parameters can be omitted."""
        tool = OperationsSkillTool()

        is_valid, error = tool.validate_inputs({
            "mode": "synthesis",
            "content": "Synthesize data",
        })

        assert is_valid is True


class TestOperationsSkillToolEdgeCases:
    """Tests for edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = OperationsSkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock(spec=ExecutionContext)
        self.mock_ctx_manager.current_context = self.mock_context
        self.mock_context.execution_mode = ExecutionMode.EXECUTION
        self.mock_context.pre_execution_hooks = []
        self.mock_context.post_execution_hooks = []
        self.tool._context_manager = self.mock_ctx_manager

    def test_empty_content(self):
        """Test with empty content returns error."""
        result = self.tool.execute(
            mode="documentation",
            content="",
        )

        assert result.success is False

    def test_very_long_content(self):
        """Test with very long content."""
        result = self.tool.execute(
            mode="synthesis",
            content="A" * 10000,
        )

        assert result.success is True

    def test_special_characters_in_content(self):
        """Test with special characters."""
        result = self.tool.execute(
            mode="creative_writing",
            content="Write about <HTML> & \"quotes\" in code",
        )

        assert result.success is True

    def test_unicode_content(self):
        """Test with unicode content."""
        result = self.tool.execute(
            mode="documentation",
            content="Document the 日本語 API with émojis 🚀",
        )

        assert result.success is True

    def test_all_modes_execute_successfully(self):
        """Test all modes execute without errors."""
        for mode in OperationsMode:
            result = self.tool.execute(
                mode=mode.value,
                content=f"Test content for {mode.value}",
            )
            assert result.success is True, f"Mode {mode.value} failed"

    def test_all_output_formats_accepted(self):
        """Test all output formats are accepted."""
        for fmt in OutputFormat:
            result = self.tool.execute(
                mode="documentation",
                content="Test docs",
                output_format=fmt.value,
            )
            assert result.success is True, f"Format {fmt.value} failed"

    def test_all_thinking_modes_accepted(self):
        """Test all thinking modes are accepted."""
        for tm in ThinkingMode:
            result = self.tool.execute(
                mode="documentation",
                content="Test docs",
                thinking_mode=tm.value,
            )
            assert result.success is True, f"ThinkingMode {tm.value} failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
