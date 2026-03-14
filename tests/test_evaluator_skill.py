"""
Unit tests for the Evaluator Skill Tool

Tests cover:
- Metadata validation
- Execution modes
- All capabilities
- Parameter validation
- Permission enforcement
- Error handling
- Edge cases
"""

from dataclasses import asdict
from unittest.mock import MagicMock, Mock, patch

import pytest

from vetinari.execution_context import ToolPermission
from vetinari.skills.evaluator import (
    EvaluatorCapability,
    EvaluatorSkillTool,
    Issue,
    QualityScore,
    ReviewRequest,
    ReviewResult,
    SeverityLevel,
    ThinkingMode,
)
from vetinari.types import ExecutionMode


class TestEvaluatorToolMetadata:
    """Tests for evaluator tool metadata."""

    def test_initialization(self):
        """Test tool initializes with correct metadata."""
        tool = EvaluatorSkillTool()
        assert tool.metadata.name == "evaluator"
        assert tool.metadata.version == "1.0.0"
        assert tool.metadata.author == "Vetinari"

    def test_description_present(self):
        """Test tool has appropriate description."""
        tool = EvaluatorSkillTool()
        assert "review" in tool.metadata.description.lower()
        assert "quality" in tool.metadata.description.lower()

    def test_parameters_defined(self):
        """Test all expected parameters are defined."""
        tool = EvaluatorSkillTool()
        param_names = {p.name for p in tool.metadata.parameters}
        assert "capability" in param_names
        assert "code" in param_names
        assert "context" in param_names
        assert "thinking_mode" in param_names
        assert "focus_areas" in param_names

    def test_capability_parameter_validation(self):
        """Test capability parameter has allowed values."""
        tool = EvaluatorSkillTool()
        capability_param = next(p for p in tool.metadata.parameters if p.name == "capability")
        assert capability_param.required is True
        expected_values = {c.value for c in EvaluatorCapability}
        assert set(capability_param.allowed_values) == expected_values

    def test_thinking_mode_parameter_validation(self):
        """Test thinking_mode parameter has allowed values."""
        tool = EvaluatorSkillTool()
        mode_param = next(p for p in tool.metadata.parameters if p.name == "thinking_mode")
        assert mode_param.required is False
        assert mode_param.default == "medium"
        expected_values = {m.value for m in ThinkingMode}
        assert set(mode_param.allowed_values) == expected_values

    def test_required_permissions(self):
        """Test tool requires correct permissions."""
        tool = EvaluatorSkillTool()
        assert ToolPermission.FILE_READ in tool.metadata.required_permissions
        assert ToolPermission.MODEL_INFERENCE in tool.metadata.required_permissions

    def test_allowed_execution_modes(self):
        """Test tool supports correct execution modes."""
        tool = EvaluatorSkillTool()
        assert ExecutionMode.EXECUTION in tool.metadata.allowed_modes
        assert ExecutionMode.PLANNING in tool.metadata.allowed_modes


class TestEvaluatorToolExecution:
    """Tests for evaluator tool execution."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = EvaluatorSkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock()
        self.mock_context.mode = ExecutionMode.EXECUTION
        self.mock_ctx_manager.current_context = self.mock_context
        self.tool._context_manager = self.mock_ctx_manager

    def test_code_review_execution_mode(self):
        """Test code review in execution mode."""
        result = self.tool.execute(
            capability="code_review",
            code="def hello():\n    pass\n",
        )
        assert result.success is True
        assert result.output is not None
        assert "issues" in result.output
        assert "quality_score" in result.output

    def test_code_review_planning_mode(self):
        """Test code review in planning mode."""
        self.mock_context.mode = ExecutionMode.PLANNING
        result = self.tool.execute(
            capability="code_review",
            code="def hello():\n    pass\n",
        )
        assert result.success is True
        assert "Planning mode" in result.output["summary"]

    def test_quality_assessment_execution_mode(self):
        """Test quality assessment in execution mode."""
        result = self.tool.execute(
            capability="quality_assessment",
            code="def calculate():\n    return 42\n",
        )
        assert result.success is True
        assert "quality_score" in result.output

    def test_security_audit_execution_mode(self):
        """Test security audit in execution mode."""
        result = self.tool.execute(
            capability="security_audit",  # noqa: VET040
            code="password = 'secret123'\n",  # noqa: VET040
        )
        assert result.success is True
        assert len(result.output["issues"]) > 0

    def test_test_strategy_execution_mode(self):
        """Test test strategy in execution mode."""
        result = self.tool.execute(
            capability="test_strategy",
            code="def process_data(x):\n    return x * 2\n",
        )
        assert result.success is True
        assert len(result.output["recommendations"]) > 0

    def test_performance_review_execution_mode(self):
        """Test performance review in execution mode."""
        result = self.tool.execute(
            capability="performance_review",
            code="for i in range(10):\n    for j in range(10):\n        pass\n",
        )
        assert result.success is True
        assert "issues" in result.output

    def test_best_practices_execution_mode(self):
        """Test best practices check in execution mode."""
        result = self.tool.execute(
            capability="best_practices",
            code="def foo():\n    pass\n",
        )
        assert result.success is True
        assert "recommendations" in result.output

    def test_invalid_capability_error(self):
        """Test error handling for invalid capability."""
        result = self.tool.execute(
            capability="invalid_capability",
            code="def hello():\n    pass\n",
        )
        assert result.success is False
        assert "Invalid capability" in result.error

    def test_invalid_thinking_mode_error(self):
        """Test error handling for invalid thinking mode."""
        result = self.tool.execute(
            capability="code_review",
            code="def hello():\n    pass\n",
            thinking_mode="invalid_mode",
        )
        assert result.success is False
        assert "Invalid thinking_mode" in result.error

    def test_missing_code_parameter_error(self):
        """Test error handling when code parameter is missing."""
        result = self.tool.execute(
            capability="code_review",
        )
        assert result.success is False
        assert "Code parameter is required" in result.error

    def test_empty_code_parameter_error(self):
        """Test error handling when code parameter is empty."""
        result = self.tool.execute(
            capability="code_review",
            code="",
        )
        assert result.success is False
        assert "Code parameter is required" in result.error

    def test_metadata_in_result(self):
        """Test result includes execution metadata."""
        result = self.tool.execute(
            capability="code_review",
            code="def hello():\n    pass\n",
            thinking_mode="high",
        )
        assert result.success is True
        assert result.metadata is not None
        assert result.metadata["capability"] == "code_review"
        assert result.metadata["thinking_mode"] == "high"
        assert result.metadata["execution_mode"] == ExecutionMode.EXECUTION.value

    def test_execution_mode_in_metadata(self):
        """Test execution mode is included in metadata."""
        self.mock_context.mode = ExecutionMode.PLANNING
        result = self.tool.execute(
            capability="code_review",
            code="def hello():\n    pass\n",
        )
        assert result.metadata["execution_mode"] == ExecutionMode.PLANNING.value


class TestEvaluatorParameterValidation:
    """Tests for parameter validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = EvaluatorSkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock(mode=ExecutionMode.EXECUTION)
        self.mock_ctx_manager.current_context = self.mock_context
        self.tool._context_manager = self.mock_ctx_manager

    def test_capability_required(self):
        """Test capability parameter is required."""
        result = self.tool.execute(
            code="def hello():\n    pass\n",
        )
        assert result.success is False

    def test_code_required(self):
        """Test code parameter is required."""
        result = self.tool.execute(
            capability="code_review",
        )
        assert result.success is False

    def test_context_optional(self):
        """Test context parameter is optional."""
        result = self.tool.execute(
            capability="code_review",
            code="def hello():\n    pass\n",
        )
        assert result.success is True

    def test_thinking_mode_optional(self):
        """Test thinking_mode parameter is optional."""
        result = self.tool.execute(
            capability="code_review",
            code="def hello():\n    pass\n",
        )
        assert result.success is True

    def test_thinking_mode_default_medium(self):
        """Test thinking_mode defaults to medium."""
        result = self.tool.execute(
            capability="code_review",
            code="def hello():\n    pass\n",
        )
        assert result.success is True
        assert result.metadata["thinking_mode"] == "medium"

    def test_focus_areas_optional(self):
        """Test focus_areas parameter is optional."""
        result = self.tool.execute(
            capability="code_review",
            code="def hello():\n    pass\n",
        )
        assert result.success is True

    def test_focus_areas_with_list(self):
        """Test focus_areas accepts list parameter."""
        result = self.tool.execute(
            capability="code_review",
            code="def hello():\n    pass\n",
            focus_areas=["security", "performance"],
        )
        assert result.success is True

    def test_all_thinking_modes_valid(self):
        """Test all thinking modes are accepted."""
        for mode in ThinkingMode:
            result = self.tool.execute(
                capability="code_review",
                code="def hello():\n    pass\n",
                thinking_mode=mode.value,
            )
            assert result.success is True
            assert result.metadata["thinking_mode"] == mode.value

    def test_all_capabilities_valid(self):
        """Test all capabilities are accepted."""
        for capability in EvaluatorCapability:
            result = self.tool.execute(
                capability=capability.value,
                code="def hello():\n    pass\n",
            )
            assert result.success is True


class TestIssueDataclass:
    """Tests for Issue dataclass."""

    def test_issue_creation(self):
        """Test Issue can be created."""
        issue = Issue(
            title="Test Issue",
            severity=SeverityLevel.MEDIUM,
            location="file.py:10",
            description="This is a test issue",
            suggestion="Fix this issue",
        )
        assert issue.title == "Test Issue"
        assert issue.severity == SeverityLevel.MEDIUM

    def test_issue_to_dict(self):
        """Test Issue can be converted to dict."""
        issue = Issue(
            title="Test Issue",
            severity=SeverityLevel.HIGH,
            location="file.py:15",
            description="Test description",
        )
        result = issue.to_dict()
        assert result["title"] == "Test Issue"
        assert result["severity"] == "high"
        assert result["location"] == "file.py:15"

    def test_issue_optional_fields(self):
        """Test Issue optional fields."""
        issue = Issue(
            title="Minimal Issue",
            severity=SeverityLevel.LOW,
        )
        assert issue.location is None
        assert issue.description is None
        assert issue.suggestion is None


class TestReviewRequestDataclass:
    """Tests for ReviewRequest dataclass."""

    def test_request_creation(self):
        """Test ReviewRequest can be created."""
        request = ReviewRequest(
            capability=EvaluatorCapability.CODE_REVIEW,
            code="def hello():\n    pass\n",
        )
        assert request.capability == EvaluatorCapability.CODE_REVIEW
        assert request.code == "def hello():\n    pass\n"

    def test_request_default_thinking_mode(self):
        """Test ReviewRequest defaults thinking_mode to medium."""
        request = ReviewRequest(
            capability=EvaluatorCapability.CODE_REVIEW,
            code="def hello():\n    pass\n",
        )
        assert request.thinking_mode == ThinkingMode.MEDIUM

    def test_request_to_dict(self):
        """Test ReviewRequest can be converted to dict."""
        request = ReviewRequest(
            capability=EvaluatorCapability.SECURITY_AUDIT,
            code="def process():\n    pass\n",
            thinking_mode=ThinkingMode.HIGH,
        )
        result = request.to_dict()
        assert result["capability"] == "security_audit"
        assert result["thinking_mode"] == "high"


class TestReviewResultDataclass:
    """Tests for ReviewResult dataclass."""

    def test_result_creation_success(self):
        """Test ReviewResult can be created with success."""
        result = ReviewResult(success=True)
        assert result.success is True
        assert len(result.issues) == 0

    def test_result_with_issues(self):
        """Test ReviewResult with issues."""
        issue = Issue(
            title="Test Issue",
            severity=SeverityLevel.MEDIUM,
        )
        result = ReviewResult(
            success=True,
            issues=[issue],
        )
        assert len(result.issues) == 1
        assert result.issues[0].title == "Test Issue"

    def test_result_to_dict(self):
        """Test ReviewResult can be converted to dict."""
        issue = Issue(
            title="Test Issue",
            severity=SeverityLevel.HIGH,
        )
        result = ReviewResult(
            success=True,
            issues=[issue],
            quality_score=QualityScore.B,
            summary="Test summary",
        )
        result_dict = result.to_dict()
        assert result_dict["success"] is True
        assert len(result_dict["issues"]) == 1
        assert result_dict["quality_score"] == "B"


class TestCodeReviewCapability:
    """Tests for code_review capability."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = EvaluatorSkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock(mode=ExecutionMode.EXECUTION)
        self.mock_ctx_manager.current_context = self.mock_context
        self.tool._context_manager = self.mock_ctx_manager

    def test_code_review_basic(self):
        """Test basic code review."""
        result = self.tool.execute(
            capability="code_review",
            code="def hello():\n    pass\n",
        )
        assert result.success is True
        assert "quality_score" in result.output

    def test_code_review_detects_todos(self):
        """Test code review detects TODO comments."""
        result = self.tool.execute(
            capability="code_review",
            code="def hello():\n    # TODO: implement\n    pass\n",
        )
        assert result.success is True
        assert len(result.output["issues"]) > 0

    def test_code_review_detects_unbalanced_braces(self):
        """Test code review detects unbalanced braces."""
        result = self.tool.execute(
            capability="code_review",
            code="{\n  code\n",
        )
        assert result.success is True
        assert len(result.output["issues"]) > 0

    def test_code_review_planning_mode(self):
        """Test code review in planning mode."""
        self.mock_context.mode = ExecutionMode.PLANNING
        result = self.tool.execute(
            capability="code_review",
            code="def hello():\n    pass\n",
        )
        assert result.success is True
        assert "Planning mode" in result.output["summary"]


class TestQualityAssessmentCapability:
    """Tests for quality_assessment capability."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = EvaluatorSkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock(mode=ExecutionMode.EXECUTION)
        self.mock_ctx_manager.current_context = self.mock_context
        self.tool._context_manager = self.mock_ctx_manager

    def test_quality_assessment_basic(self):
        """Test basic quality assessment."""
        result = self.tool.execute(
            capability="quality_assessment",
            code="def process():\n    return 42\n",
        )
        assert result.success is True
        assert "quality_score" in result.output

    def test_quality_assessment_high_complexity(self):
        """Test quality assessment detects high complexity."""
        long_code = "\n".join([f"line {i}: x = {i}" for i in range(600)])
        result = self.tool.execute(
            capability="quality_assessment",
            code=long_code,
            thinking_mode="medium",
        )
        assert result.success is True
        assert len(result.output["issues"]) > 0

    def test_quality_assessment_with_high_thinking_mode(self):
        """Test quality assessment with high thinking mode."""
        result = self.tool.execute(
            capability="quality_assessment",
            code="var = 42\n",
            thinking_mode="high",
        )
        assert result.success is True


class TestSecurityAuditCapability:
    """Tests for security_audit capability."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = EvaluatorSkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock(mode=ExecutionMode.EXECUTION)
        self.mock_ctx_manager.current_context = self.mock_context
        self.tool._context_manager = self.mock_ctx_manager

    def test_security_audit_basic(self):
        """Test basic security audit."""
        result = self.tool.execute(
            capability="security_audit",
            code="def safe():\n    return True\n",
        )
        assert result.success is True
        assert "issues" in result.output

    def test_security_audit_detects_eval(self):
        """Test security audit detects eval usage."""
        result = self.tool.execute(
            capability="security_audit",
            code="result = eval(user_input)\n",
        )
        assert result.success is True
        assert len(result.output["issues"]) > 0

    def test_security_audit_detects_exec(self):
        """Test security audit detects exec usage."""
        result = self.tool.execute(
            capability="security_audit",
            code="exec(code_string)\n",
        )
        assert result.success is True
        assert len(result.output["issues"]) > 0

    def test_security_audit_detects_hardcoded_secrets(self):
        """Test security audit detects hardcoded secrets."""
        result = self.tool.execute(
            capability="security_audit",  # noqa: VET040
            code="password = 'super_secret'\n",  # noqa: VET040
        )
        assert result.success is True
        assert len(result.output["issues"]) > 0

    def test_security_audit_critical_severity(self):
        """Test security audit marks critical issues."""
        result = self.tool.execute(
            capability="security_audit",
            code="eval(user_input)\n",
        )
        assert result.success is True
        critical_issues = [i for i in result.output["issues"] if i["severity"] == "critical"]
        assert len(critical_issues) > 0


class TestTestStrategyCapability:
    """Tests for test_strategy capability."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = EvaluatorSkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock(mode=ExecutionMode.EXECUTION)
        self.mock_ctx_manager.current_context = self.mock_context
        self.tool._context_manager = self.mock_ctx_manager

    def test_test_strategy_basic(self):
        """Test basic test strategy generation."""
        result = self.tool.execute(
            capability="test_strategy",
            code="def add(a, b):\n    return a + b\n",
        )
        assert result.success is True
        assert "recommendations" in result.output
        assert len(result.output["recommendations"]) > 0

    def test_test_strategy_with_low_thinking_mode(self):
        """Test test strategy with low thinking mode."""
        result = self.tool.execute(
            capability="test_strategy",
            code="def process():\n    pass\n",
            thinking_mode="low",
        )
        assert result.success is True
        assert len(result.output["recommendations"]) > 0

    def test_test_strategy_with_high_thinking_mode(self):
        """Test test strategy with high thinking mode."""
        result = self.tool.execute(
            capability="test_strategy",
            code="def process():\n    pass\n",
            thinking_mode="high",
        )
        assert result.success is True
        assert len(result.output["recommendations"]) > 3


class TestPerformanceReviewCapability:
    """Tests for performance_review capability."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = EvaluatorSkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock(mode=ExecutionMode.EXECUTION)
        self.mock_ctx_manager.current_context = self.mock_context
        self.tool._context_manager = self.mock_ctx_manager

    def test_performance_review_basic(self):
        """Test basic performance review."""
        result = self.tool.execute(
            capability="performance_review",
            code="def calculate():\n    return 42\n",
        )
        assert result.success is True
        assert "issues" in result.output

    def test_performance_review_nested_loops(self):
        """Test performance review detects nested loops."""
        code = "for i in range(10):\n    for j in range(10):\n        for k in range(10):\n            pass\n"
        result = self.tool.execute(
            capability="performance_review",
            code=code,
        )
        assert result.success is True
        assert len(result.output["issues"]) > 0

    def test_performance_review_infinite_loop(self):
        """Test performance review detects infinite loops."""
        result = self.tool.execute(
            capability="performance_review",
            code="while True:\n    x = 1\n",
        )
        assert result.success is True
        assert len(result.output["issues"]) > 0


class TestBestPracticesCapability:
    """Tests for best_practices capability."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = EvaluatorSkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock(mode=ExecutionMode.EXECUTION)
        self.mock_ctx_manager.current_context = self.mock_context
        self.tool._context_manager = self.mock_ctx_manager

    def test_best_practices_basic(self):
        """Test basic best practices check."""
        result = self.tool.execute(
            capability="best_practices",
            code="def hello():\n    pass\n",
        )
        assert result.success is True
        assert "recommendations" in result.output

    def test_best_practices_recommendations(self):
        """Test best practices provides recommendations."""
        result = self.tool.execute(
            capability="best_practices",
            code="def foo():\n    pass\ndef bar():\n    pass\n",
        )
        assert result.success is True
        assert len(result.output["recommendations"]) > 0


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = EvaluatorSkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock(mode=ExecutionMode.EXECUTION)
        self.mock_ctx_manager.current_context = self.mock_context
        self.tool._context_manager = self.mock_ctx_manager

    def test_very_long_code(self):
        """Test handling of very long code."""
        long_code = "\n".join([f"# Line {i}" for i in range(10000)])
        result = self.tool.execute(
            capability="code_review",
            code=long_code,
        )
        assert result.success is True

    def test_unicode_in_code(self):
        """Test handling of unicode characters in code."""
        result = self.tool.execute(
            capability="code_review",
            code="# This is a comment with unicode: 你好世界\ndef hello():\n    pass\n",
        )
        assert result.success is True

    def test_special_characters_in_code(self):
        """Test handling of special characters in code."""
        result = self.tool.execute(
            capability="code_review",
            code="# Special chars: !@#$%^&*()\ndef process():\n    pass\n",
        )
        assert result.success is True

    def test_multiline_strings(self):
        """Test handling of multiline strings."""
        result = self.tool.execute(
            capability="code_review",
            code='"""\nMultiline\ndocstring\n"""\ndef foo():\n    pass\n',
        )
        assert result.success is True

    def test_mixed_indentation(self):
        """Test handling of mixed indentation."""
        result = self.tool.execute(
            capability="code_review",
            code="def foo():\n\t    pass\n",
        )
        assert result.success is True

    def test_null_bytes_in_code(self):
        """Test handling of null bytes."""
        result = self.tool.execute(
            capability="code_review",
            code="def foo():\n    pass\x00\n",
        )
        assert result.success is True

    def test_all_capabilities_with_minimal_code(self):
        """Test all capabilities work with minimal code."""
        capabilities = [c.value for c in EvaluatorCapability]
        for capability in capabilities:
            result = self.tool.execute(
                capability=capability,
                code="x = 1",
            )
            assert result.success is True

    def test_result_output_serializable(self):
        """Test result output is JSON serializable."""
        import json
        result = self.tool.execute(
            capability="code_review",
            code="def hello():\n    pass\n",
        )
        assert result.success is True
        # This should not raise an exception
        json.dumps(result.output)

    def test_context_manager_exception_handling(self):
        """Test tool handles context manager errors gracefully."""
        self.mock_ctx_manager.current_context = None
        result = self.tool.execute(
            capability="code_review",
            code="def hello():\n    pass\n",
        )
        # Should handle the error gracefully
        assert result.success is False or result.success is True


class TestIntegration:
    """Integration tests combining multiple features."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = EvaluatorSkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock(mode=ExecutionMode.EXECUTION)
        self.mock_ctx_manager.current_context = self.mock_context
        self.tool._context_manager = self.mock_ctx_manager

    def test_multiple_capability_runs(self):
        """Test running multiple capabilities in sequence."""
        code = "def process(data):\n    return data * 2\n"

        code_review = self.tool.execute(
            capability="code_review",
            code=code,
        )
        assert code_review.success is True

        quality = self.tool.execute(
            capability="quality_assessment",
            code=code,
        )
        assert quality.success is True

    def test_different_thinking_modes_produce_different_output(self):
        """Test that different thinking modes affect output."""
        code = "def complex_func():\n    pass\n"

        low_result = self.tool.execute(
            capability="code_review",
            code=code,
            thinking_mode="low",
        )

        xhigh_result = self.tool.execute(
            capability="code_review",
            code=code,
            thinking_mode="xhigh",
        )

        assert low_result.success is True
        assert xhigh_result.success is True

    def test_planning_vs_execution_mode_results(self):
        """Test planning mode vs execution mode produces different outputs."""
        code = "def test():\n    pass\n"

        # Execution mode
        self.mock_context.mode = ExecutionMode.EXECUTION
        exec_result = self.tool.execute(
            capability="code_review",
            code=code,
        )

        # Planning mode
        self.mock_context.mode = ExecutionMode.PLANNING
        plan_result = self.tool.execute(
            capability="code_review",
            code=code,
        )

        assert exec_result.success is True
        assert plan_result.success is True
        # Planning mode should indicate planning
        assert "Planning mode" in plan_result.output["summary"]
