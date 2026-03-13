"""
Unit tests for the builder skill Tool wrapper.

Tests cover:
- Tool initialization and metadata
- All builder capabilities (feature, refactoring, testing, error handling, code generation, debugging)
- Parameter validation
- Permission checking
- Execution mode handling (PLANNING vs EXECUTION)
- Error cases
"""

from unittest.mock import Mock

import pytest

from vetinari.execution_context import (
    ExecutionContext,
    ExecutionMode,
    ToolPermission,
)
from vetinari.skills.builder import (
    BuilderCapability,
    BuilderSkillTool,
    ImplementationRequest,
    ImplementationResult,
    ThinkingMode,
)


class TestBuilderSkillToolMetadata:
    """Tests for builder skill metadata and initialization."""

    def test_initialization(self):
        """Test BuilderSkillTool initialization."""
        tool = BuilderSkillTool()

        assert tool.metadata.name == "builder"
        assert "implementation" in tool.metadata.tags
        assert "coding" in tool.metadata.tags
        assert ToolPermission.FILE_READ in tool.metadata.required_permissions
        assert ToolPermission.FILE_WRITE in tool.metadata.required_permissions
        assert ToolPermission.MODEL_INFERENCE in tool.metadata.required_permissions

    def test_allowed_execution_modes(self):
        """Test allowed execution modes."""
        tool = BuilderSkillTool()

        assert ExecutionMode.EXECUTION in tool.metadata.allowed_modes
        assert ExecutionMode.PLANNING in tool.metadata.allowed_modes

    def test_tool_parameters(self):
        """Test tool parameters are properly defined."""
        tool = BuilderSkillTool()

        param_names = {p.name for p in tool.metadata.parameters}
        assert "capability" in param_names
        assert "description" in param_names
        assert "context" in param_names
        assert "thinking_mode" in param_names
        assert "requirements" in param_names

    def test_capability_parameter_validation(self):
        """Test capability parameter validation."""
        tool = BuilderSkillTool()

        capability_param = next(
            p for p in tool.metadata.parameters if p.name == "capability"
        )

        assert capability_param.required is True
        assert all(c.value in capability_param.allowed_values for c in BuilderCapability)

    def test_thinking_mode_parameter_validation(self):
        """Test thinking_mode parameter validation."""
        tool = BuilderSkillTool()

        thinking_param = next(
            p for p in tool.metadata.parameters if p.name == "thinking_mode"
        )

        assert thinking_param.required is False
        assert thinking_param.default == "medium"
        assert all(m.value in thinking_param.allowed_values for m in ThinkingMode)


class TestBuilderSkillToolExecution:
    """Tests for builder skill execution logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = BuilderSkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock(spec=ExecutionContext)
        self.mock_ctx_manager.current_context = self.mock_context
        self.mock_ctx_manager.current_mode = ExecutionMode.EXECUTION

        # Set up default context
        self.mock_context.execution_mode = ExecutionMode.EXECUTION
        self.mock_context.pre_execution_hooks = []
        self.mock_context.post_execution_hooks = []

        self.tool._context_manager = self.mock_ctx_manager

    def test_feature_implementation_execution_mode(self):
        """Test feature implementation in EXECUTION mode."""
        self.mock_context.execution_mode = ExecutionMode.EXECUTION

        result = self.tool.execute(
            capability="feature_implementation",
            description="Create a user authentication system",
            thinking_mode="medium",
        )

        assert result.success is True
        assert result.output is not None
        assert result.output["code"] is None  # Placeholder implementation
        assert result.output["explanation"] is not None
        assert "Feature Implementation" in result.output["explanation"]

    def test_feature_implementation_planning_mode(self):
        """Test feature implementation in PLANNING mode."""
        self.mock_context.mode = ExecutionMode.PLANNING

        result = self.tool.execute(
            capability="feature_implementation",
            description="Create a user authentication system",
            thinking_mode="high",
        )

        assert result.success is True
        assert "Planning mode" in result.output["explanation"]
        assert "EXECUTION mode" in result.output["explanation"]
        assert len(result.output["warnings"]) > 0

    def test_refactoring_with_context(self):
        """Test refactoring capability with code context."""
        code_context = "def foo(x, y, z):\n    return x + y + z"

        result = self.tool.execute(
            capability="refactoring",
            description="Improve readability",
            context=code_context,
            thinking_mode="medium",
        )

        assert result.success is True
        assert "Refactoring" in result.output["explanation"]
        assert code_context in result.output["explanation"]

    def test_refactoring_without_context(self):
        """Test refactoring fails without code context."""
        result = self.tool.execute(
            capability="refactoring",
            description="Improve readability",
        )

        assert result.success is False
        assert "requires context" in result.output["explanation"].lower()
        assert "Refactoring" in result.output["explanation"]

    def test_test_writing_with_context(self):
        """Test test writing capability with code context."""
        code_context = "def add(a, b):\n    return a + b"

        result = self.tool.execute(
            capability="test_writing",
            description="Write unit tests",
            context=code_context,
            thinking_mode="high",
        )

        assert result.success is True
        assert "Test Writing" in result.output["explanation"]
        assert result.output["tests_added"] == 6

    def test_test_writing_without_context(self):
        """Test test writing fails without code context."""
        result = self.tool.execute(
            capability="test_writing",
            description="Write unit tests",
        )

        assert result.success is False
        assert "requires context" in result.output["explanation"].lower()

    def test_error_handling_with_context(self):
        """Test error handling capability with code context."""
        code_context = "result = some_function()"

        result = self.tool.execute(
            capability="error_handling",
            description="Add error handling",
            context=code_context,
        )

        assert result.success is True
        assert "Error Handling" in result.output["explanation"]
        assert "try-catch" in result.output["explanation"]

    def test_error_handling_without_context(self):
        """Test error handling fails without code context."""
        result = self.tool.execute(
            capability="error_handling",
            description="Add error handling",
        )

        assert result.success is False
        assert "requires context" in result.output["explanation"].lower()

    def test_code_generation_execution_mode(self):
        """Test code generation in EXECUTION mode."""
        result = self.tool.execute(
            capability="code_generation",
            description="Generate CRUD endpoints",
            requirements=["RESTful API", "async/await"],
        )

        assert result.success is True
        assert "Code Generation" in result.output["explanation"]
        assert "RESTful API" in result.output["explanation"]

    def test_code_generation_planning_mode(self):
        """Test code generation in PLANNING mode."""
        self.mock_context.mode = ExecutionMode.PLANNING

        result = self.tool.execute(
            capability="code_generation",
            description="Generate CRUD endpoints",
        )

        assert result.success is True
        assert "Planning mode" in result.output["explanation"]

    def test_debugging_with_context(self):
        """Test debugging capability with code context."""
        code_context = "if x = 5:\n    print('error')"

        result = self.tool.execute(
            capability="debugging",
            description="Fix syntax errors",
            context=code_context,
        )

        assert result.success is True
        assert "Debugging" in result.output["explanation"]
        assert "root cause" in result.output["explanation"]

    def test_debugging_without_context(self):
        """Test debugging fails without code context."""
        result = self.tool.execute(
            capability="debugging",
            description="Fix syntax errors",
        )

        assert result.success is False
        assert "requires context" in result.output["explanation"].lower()

    def test_invalid_capability(self):
        """Test execution with invalid capability."""
        result = self.tool.execute(
            capability="invalid_capability",
            description="Do something",
        )

        assert result.success is False
        assert result.output is None
        assert "Invalid capability" in result.error

    def test_invalid_thinking_mode(self):
        """Test execution with invalid thinking mode."""
        result = self.tool.execute(
            capability="feature_implementation",
            description="Create something",
            thinking_mode="invalid_mode",
        )

        assert result.success is False
        assert result.output is None
        assert "Invalid thinking_mode" in result.error

    def test_thinking_mode_affects_explanation(self):
        """Test that thinking mode affects the generated explanation."""
        modes_and_expectations = [
            ("low", "Quick implementation"),
            ("medium", "Full feature with basic tests"),
            ("high", "Complete implementation"),
            ("xhigh", "Production-ready"),
        ]

        for mode, expected_text in modes_and_expectations:
            result = self.tool.execute(
                capability="feature_implementation",
                description="Implement something",
                thinking_mode=mode,
            )

            assert result.success is True
            assert expected_text in result.output["explanation"]

    def test_requirements_included_in_output(self):
        """Test that requirements are included in output."""
        requirements = ["async support", "type hints", "error handling"]

        result = self.tool.execute(
            capability="code_generation",
            description="Generate API endpoints",
            requirements=requirements,
        )

        assert result.success is True
        output_explanation = result.output["explanation"]
        assert any(req in output_explanation for req in requirements)


class TestImplementationRequest:
    """Tests for ImplementationRequest dataclass."""

    def test_creation_with_defaults(self):
        """Test creating request with default values."""
        request = ImplementationRequest(
            capability=BuilderCapability.FEATURE_IMPLEMENTATION,
            description="Build a feature",
        )

        assert request.capability == BuilderCapability.FEATURE_IMPLEMENTATION
        assert request.description == "Build a feature"
        assert request.context is None
        assert request.thinking_mode == ThinkingMode.MEDIUM
        assert request.requirements == []

    def test_creation_with_all_fields(self):
        """Test creating request with all fields."""
        reqs = ["async", "error handling"]
        request = ImplementationRequest(
            capability=BuilderCapability.REFACTORING,
            description="Refactor module",
            context="original code",
            thinking_mode=ThinkingMode.HIGH,
            requirements=reqs,
        )

        assert request.capability == BuilderCapability.REFACTORING
        assert request.context == "original code"
        assert request.thinking_mode == ThinkingMode.HIGH
        assert request.requirements == reqs

    def test_to_dict(self):
        """Test converting request to dictionary."""
        request = ImplementationRequest(
            capability=BuilderCapability.TEST_WRITING,
            description="Write tests",
            thinking_mode=ThinkingMode.XHIGH,
            requirements=["100% coverage"],
        )

        result_dict = request.to_dict()

        assert result_dict["capability"] == "test_writing"
        assert result_dict["description"] == "Write tests"
        assert result_dict["thinking_mode"] == "xhigh"
        assert result_dict["requirements"] == ["100% coverage"]


class TestImplementationResult:
    """Tests for ImplementationResult dataclass."""

    def test_success_result(self):
        """Test creating successful result."""
        result = ImplementationResult(
            success=True,
            code="def foo(): pass",
            explanation="Function created",
            files_affected=["module.py"],
            tests_added=3,
        )

        assert result.success is True
        assert result.code == "def foo(): pass"
        assert len(result.files_affected) == 1
        assert result.tests_added == 3

    def test_failure_result(self):
        """Test creating failure result."""
        result = ImplementationResult(
            success=False,
            explanation="Could not generate code",
            warnings=["Invalid input"],
        )

        assert result.success is False
        assert result.code is None
        assert "Invalid input" in result.warnings

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = ImplementationResult(
            success=True,
            code="code here",
            explanation="Done",
            files_affected=["a.py", "b.py"],
            tests_added=2,
            warnings=["Note 1"],
        )

        result_dict = result.to_dict()

        assert result_dict["success"] is True
        assert result_dict["code"] == "code here"
        assert len(result_dict["files_affected"]) == 2
        assert result_dict["tests_added"] == 2


class TestBuilderSkillToolParameterValidation:
    """Tests for input parameter validation."""

    def test_missing_required_capability(self):
        """Test that missing capability is caught by validation."""
        tool = BuilderSkillTool()

        is_valid, error = tool.validate_inputs({"description": "test"})

        assert is_valid is False
        assert "capability" in error.lower()

    def test_missing_required_description(self):
        """Test that missing description is caught by validation."""
        tool = BuilderSkillTool()

        is_valid, error = tool.validate_inputs({"capability": "feature_implementation"})

        assert is_valid is False
        assert "description" in error.lower()

    def test_invalid_capability_value(self):
        """Test that invalid capability value is caught."""
        tool = BuilderSkillTool()

        is_valid, error = tool.validate_inputs({
            "capability": "invalid_capability",
            "description": "test",
        })

        assert is_valid is False
        assert "capability" in error.lower()

    def test_invalid_thinking_mode_value(self):
        """Test that invalid thinking_mode value is caught."""
        tool = BuilderSkillTool()

        is_valid, error = tool.validate_inputs({
            "capability": "feature_implementation",
            "description": "test",
            "thinking_mode": "invalid_mode",
        })

        assert is_valid is False
        assert "thinking_mode" in error.lower()

    def test_valid_parameters(self):
        """Test that valid parameters pass validation."""
        tool = BuilderSkillTool()

        is_valid, error = tool.validate_inputs({
            "capability": "feature_implementation",
            "description": "test",
            "thinking_mode": "high",
            "context": "code",
            "requirements": ["test"],
        })

        assert is_valid is True
        assert error is None

    def test_optional_context_omitted(self):
        """Test that optional context can be omitted."""
        tool = BuilderSkillTool()

        is_valid, error = tool.validate_inputs({
            "capability": "feature_implementation",
            "description": "test",
        })

        assert is_valid is True
        assert error is None


class TestBuilderCapabilityEnum:
    """Tests for BuilderCapability enum."""

    def test_all_capabilities_have_values(self):
        """Test that all capabilities have proper values."""
        capabilities = [
            BuilderCapability.FEATURE_IMPLEMENTATION,
            BuilderCapability.REFACTORING,
            BuilderCapability.TEST_WRITING,
            BuilderCapability.ERROR_HANDLING,
            BuilderCapability.CODE_GENERATION,
            BuilderCapability.DEBUGGING,
        ]

        for cap in capabilities:
            assert cap.value is not None
            assert isinstance(cap.value, str)
            assert len(cap.value) > 0


class TestThinkingModeEnum:
    """Tests for ThinkingMode enum."""

    def test_all_thinking_modes_have_values(self):
        """Test that all thinking modes have proper values."""
        modes = [
            ThinkingMode.LOW,
            ThinkingMode.MEDIUM,
            ThinkingMode.HIGH,
            ThinkingMode.XHIGH,
        ]

        for mode in modes:
            assert mode.value is not None
            assert isinstance(mode.value, str)
            assert len(mode.value) > 0

    def test_thinking_mode_ordering(self):
        """Test thinking mode values are meaningful."""
        assert ThinkingMode.LOW.value == "low"
        assert ThinkingMode.MEDIUM.value == "medium"
        assert ThinkingMode.HIGH.value == "high"
        assert ThinkingMode.XHIGH.value == "xhigh"


class TestBuilderSkillToolEdgeCases:
    """Tests for edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = BuilderSkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock(spec=ExecutionContext)
        self.mock_ctx_manager.current_context = self.mock_context
        self.mock_ctx_manager.current_mode = ExecutionMode.EXECUTION

        self.mock_context.execution_mode = ExecutionMode.EXECUTION
        self.mock_context.pre_execution_hooks = []
        self.mock_context.post_execution_hooks = []

        self.tool._context_manager = self.mock_ctx_manager

    def test_empty_description(self):
        """Test with empty description."""
        result = self.tool.execute(
            capability="feature_implementation",
            description="",
        )

        # Should still succeed, just with empty description
        assert result.success is True

    def test_empty_requirements_list(self):
        """Test with empty requirements."""
        result = self.tool.execute(
            capability="code_generation",
            description="Generate code",
            requirements=[],
        )

        assert result.success is True

    def test_very_long_description(self):
        """Test with very long description."""
        long_desc = "A" * 10000

        result = self.tool.execute(
            capability="feature_implementation",
            description=long_desc,
        )

        assert result.success is True

    def test_large_context(self):
        """Test with large code context."""
        large_code = "def func():\n    pass\n" * 1000

        result = self.tool.execute(
            capability="refactoring",
            description="Refactor",
            context=large_code,
        )

        assert result.success is True

    def test_special_characters_in_description(self):
        """Test with special characters in description."""
        result = self.tool.execute(
            capability="feature_implementation",
            description="Add <special> & \"quoted\" characters",
        )

        assert result.success is True

    def test_unicode_characters(self):
        """Test with unicode characters."""
        result = self.tool.execute(
            capability="feature_implementation",
            description="Build 日本語 ñ feature with émojis 🚀",
        )

        assert result.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
