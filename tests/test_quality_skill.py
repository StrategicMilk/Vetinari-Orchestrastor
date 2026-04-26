"""
Unit tests for the unified Quality Skill Tool.

Tests cover:
- Tool initialization and metadata
- All 6 quality modes (code_review, security_audit, test_generation, simplification,
  performance_review, best_practices)
- Security audit OWASP coverage
- Severity filtering
- Grading system
- Parameter validation
- Dataclass serialization
- Error cases
"""

from unittest.mock import Mock

import pytest

from vetinari.execution_context import (
    ExecutionContext,
    ExecutionMode,
    ToolPermission,
)
from vetinari.skills.quality_skill import (
    OWASP_TOP_10,
    QualityGrade,
    QualityIssue,
    QualityMode,
    QualityResult,
    QualitySkillTool,
    SeverityLevel,
    ThinkingMode,
)


class TestQualitySkillToolMetadata:
    """Tests for quality skill metadata and initialization."""

    def test_initialization(self):
        """Test QualitySkillTool initializes with correct metadata."""
        tool = QualitySkillTool()

        assert tool.metadata.name == "quality"
        assert "review" in tool.metadata.tags
        assert "security" in tool.metadata.tags
        assert "testing" in tool.metadata.tags
        assert ToolPermission.FILE_READ in tool.metadata.required_permissions
        assert ToolPermission.MODEL_INFERENCE in tool.metadata.required_permissions

    def test_allowed_execution_modes(self):
        """Test allowed execution modes."""
        tool = QualitySkillTool()

        assert ExecutionMode.EXECUTION in tool.metadata.allowed_modes
        assert ExecutionMode.PLANNING in tool.metadata.allowed_modes

    def test_tool_parameters(self):
        """Test tool parameters are properly defined."""
        tool = QualitySkillTool()

        param_names = {p.name for p in tool.metadata.parameters}
        assert "mode" in param_names
        assert "code" in param_names
        assert "context" in param_names
        assert "thinking_mode" in param_names
        assert "severity_threshold" in param_names

    def test_mode_parameter_has_all_modes(self):
        """Test mode parameter has all QualityMode values."""
        tool = QualitySkillTool()

        mode_param = next(p for p in tool.metadata.parameters if p.name == "mode")

        assert mode_param.required is True
        assert all(m.value in mode_param.allowed_values for m in QualityMode)

    def test_severity_threshold_parameter(self):
        """Test severity_threshold parameter defaults and values."""
        tool = QualitySkillTool()

        sev_param = next(p for p in tool.metadata.parameters if p.name == "severity_threshold")

        assert sev_param.required is False
        assert sev_param.default == "low"
        assert all(s.value in sev_param.allowed_values for s in SeverityLevel)

    def test_version(self):
        """Test tool version."""
        tool = QualitySkillTool()
        assert tool.metadata.version == "1.1.0"


class TestQualitySkillToolExecution:
    """Tests for quality skill execution logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = QualitySkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock(spec=ExecutionContext)
        self.mock_ctx_manager.current_context = self.mock_context
        self.mock_context.execution_mode = ExecutionMode.EXECUTION
        self.mock_context.pre_execution_hooks = []
        self.mock_context.post_execution_hooks = []
        self.tool._context_manager = self.mock_ctx_manager

    def test_code_review(self):
        """Test code review mode."""
        result = self.tool.execute(
            mode="code_review",
            code="def foo(x):\n    return x + 1\n",
        )

        assert result.success is True
        output = result.output
        assert output["success"] is True
        assert "quality_grade" in output
        assert output["quality_grade"] in ["A", "B", "C", "D", "F"]
        assert "summary" in output

    def test_code_review_finds_issues(self):
        """Test code review detects common issues."""
        code_with_issues = "def process():\n    # TODO: fix this\n    x = 1\n    return x\n"

        result = self.tool.execute(
            mode="code_review",
            code=code_with_issues,
        )

        assert result.success is True
        # Should find TODO/FIXME issue
        issues = result.output["issues"]
        assert len(issues) > 0

    def test_security_audit(self):
        """Test security audit mode."""
        result = self.tool.execute(
            mode="security_audit",
            code="import json\ndata = json.loads(input_str)\n",
        )

        assert result.success is True
        output = result.output
        assert "owasp_coverage" in output["metrics"]

    def test_security_audit_detects_dangerous_patterns(self):
        """Test security audit detects dangerous code patterns."""
        dangerous_code = "import yaml\ndata = yaml.load(content)\n"

        result = self.tool.execute(
            mode="security_audit",
            code=dangerous_code,
        )

        assert result.success is True
        issues = result.output["issues"]
        # Should detect yaml.load as unsafe
        assert any("yaml" in str(i).lower() for i in issues)

    def test_security_audit_detects_debug_mode(self):
        """Test security audit detects debug=True."""
        code = "app = Flask(__name__)\napp.run(debug=True)\n"

        result = self.tool.execute(
            mode="security_audit",
            code=code,
        )

        assert result.success is True
        issues = result.output["issues"]
        assert any("debug" in str(i).lower() for i in issues)

    def test_test_generation(self):
        """Test test generation mode."""
        result = self.tool.execute(
            mode="test_generation",
            code="def add(a, b):\n    return a + b\n",
        )

        assert result.success is True
        output = result.output
        assert "summary" in output

    def test_simplification(self):
        """Test simplification mode."""
        result = self.tool.execute(
            mode="simplification",
            code="x = True\nif x == True:\n    return True\nelse:\n    return False\n",
        )

        assert result.success is True

    def test_performance_review(self):
        """Test performance review mode."""
        result = self.tool.execute(
            mode="performance_review",
            code="for i in range(100):\n    for j in range(100):\n        process(i, j)\n",
        )

        assert result.success is True

    def test_best_practices(self):
        """Test best practices mode."""
        result = self.tool.execute(
            mode="best_practices",
            code="class MyClass:\n    def do_stuff(self):\n        pass\n",
        )

        assert result.success is True

    def test_invalid_mode(self):
        """Test execution with invalid mode returns error."""
        result = self.tool.execute(
            mode="nonexistent_mode",
            code="pass",
        )

        assert result.success is False
        assert "Invalid mode" in result.error

    def test_missing_code(self):
        """Test execution without code returns error."""
        result = self.tool.execute(
            mode="code_review",
        )

        assert result.success is False
        assert "code" in result.error.lower()

    def test_invalid_thinking_mode(self):
        """Test execution with invalid thinking mode."""
        result = self.tool.execute(
            mode="code_review",
            code="x = 1",
            thinking_mode="invalid",
        )

        assert result.success is False
        assert "thinking_mode" in result.error

    def test_severity_threshold_filtering(self):
        """Test severity threshold filters issues."""
        code_with_issues = "# TODO fix\nimport os\ndef f():\n    pass\n"

        # Low threshold should return more issues
        result_low = self.tool.execute(
            mode="code_review",
            code=code_with_issues,
            severity_threshold="low",
        )

        # Critical threshold should return fewer or equal issues
        result_critical = self.tool.execute(
            mode="code_review",
            code=code_with_issues,
            severity_threshold="critical",
        )

        assert result_low.success is True
        assert result_critical.success is True
        # Critical filter should have <= issues than low filter
        assert len(result_critical.output["issues"]) <= len(result_low.output["issues"])

    def test_context_parameter_accepted(self):
        """Test context parameter is accepted."""
        result = self.tool.execute(
            mode="code_review",
            code="x = 1",
            context="Part of auth module",
        )

        assert result.success is True

    def test_metadata_in_result(self):
        """Test metadata is included in ToolResult."""
        result = self.tool.execute(
            mode="security_audit",
            code="x = 1",
            thinking_mode="high",
        )

        assert result.metadata["mode"] == "security_audit"
        assert result.metadata["thinking_mode"] == "high"
        assert "issues_found" in result.metadata


class TestQualityModeEnum:
    """Tests for QualityMode enum."""

    def test_all_modes_have_values(self):
        """Test all modes have valid string values."""
        for mode in QualityMode:
            assert isinstance(mode.value, str)
            assert len(mode.value) > 0

    def test_expected_modes_exist(self):
        """Test expected modes are defined."""
        expected = {
            "code_review",
            "security_audit",
            "test_generation",
            "simplification",
            "performance_review",
            "best_practices",
        }
        actual = {m.value for m in QualityMode}
        assert actual == expected

    def test_mode_count(self):
        """Test correct number of modes."""
        assert len(QualityMode) == 6


class TestSeverityLevelEnum:
    """Tests for SeverityLevel enum."""

    def test_all_levels(self):
        """Test all severity levels exist."""
        expected = {"critical", "high", "medium", "low", "info"}
        actual = {s.value for s in SeverityLevel}
        assert actual == expected

    def test_ordering_semantics(self):
        """Test severity levels have meaningful values."""
        assert SeverityLevel.CRITICAL.value == "critical"
        assert SeverityLevel.INFO.value == "info"


class TestOWASPCoverage:
    """Tests for OWASP Top 10 coverage."""

    def test_owasp_list_has_10_entries(self):
        """Test OWASP list has exactly 10 entries."""
        assert len(OWASP_TOP_10) == 10

    def test_owasp_entries_are_strings(self):
        """Test all OWASP entries are non-empty strings."""
        for entry in OWASP_TOP_10:
            assert isinstance(entry, str)
            assert len(entry) > 0

    def test_security_audit_reports_owasp_coverage(self):
        """Test security audit metadata includes OWASP coverage count."""
        tool = QualitySkillTool()
        mock_ctx_manager = Mock()
        mock_context = Mock(spec=ExecutionContext)
        mock_ctx_manager.current_context = mock_context
        mock_context.execution_mode = ExecutionMode.EXECUTION
        mock_context.pre_execution_hooks = []
        mock_context.post_execution_hooks = []
        tool._context_manager = mock_ctx_manager

        result = tool.execute(
            mode="security_audit",
            code="x = 1",
        )

        assert result.success is True
        assert "owasp_coverage" in result.output["metrics"]
        coverage = result.output["metrics"]["owasp_coverage"]
        assert isinstance(coverage, int)
        assert 0 <= coverage <= 10


class TestQualityDataclasses:
    """Tests for quality dataclasses."""

    def test_issue_to_dict(self):
        """Test QualityIssue serialization."""
        issue = QualityIssue(
            title="SQL Injection",
            severity=SeverityLevel.CRITICAL,
            description="Unsanitized input in query",
            location="db.py:42",
            suggestion="Use parameterized queries",
            cwe_id="CWE-89",
            owasp_category="A03:2021 Injection",
        )

        d = issue.to_dict()
        assert d["title"] == "SQL Injection"
        assert d["severity"] == "critical"
        assert d["location"] == "db.py:42"
        assert d["cwe_id"] == "CWE-89"
        assert d["owasp_category"] == "A03:2021 Injection"

    def test_issue_defaults(self):
        """Test QualityIssue default values."""
        issue = QualityIssue(
            title="Minor issue",
            severity=SeverityLevel.LOW,
            description="A minor issue",
        )

        assert issue.location is None
        assert issue.suggestion is None
        assert issue.cwe_id is None
        assert issue.owasp_category is None

    def test_issue_to_dict_optional_fields_omitted(self):
        """Test QualityIssue.to_dict omits None optional fields."""
        issue = QualityIssue(
            title="Test",
            severity=SeverityLevel.INFO,
            description="Test desc",
        )

        d = issue.to_dict()
        assert "location" not in d
        assert "cwe_id" not in d
        assert "owasp_category" not in d

    def test_result_success(self):
        """Test QualityResult success case."""
        result = QualityResult(
            success=True,
            issues=[],
            grade=QualityGrade.A,
            summary="No issues found",
        )

        assert result.success is True
        assert result.grade == QualityGrade.A
        assert result.recommendations == []

    def test_result_to_dict(self):
        """Test QualityResult serialization."""
        issue = QualityIssue(title="Test", severity=SeverityLevel.MEDIUM, description="Desc")
        result = QualityResult(
            success=True,
            issues=[issue],
            grade=QualityGrade.B,
            score=85.0,
            summary="Minor issues",
            recommendations=["Fix tests"],
            metrics={"lines_reviewed": 100},
        )

        d = result.to_dict()
        assert d["success"] is True
        assert len(d["issues"]) == 1
        assert d["quality_grade"] == "B"
        assert d["score"] == 85.0
        assert d["summary"] == "Minor issues"
        assert d["recommendations"] == ["Fix tests"]
        assert d["metrics"]["lines_reviewed"] == 100

    def test_result_defaults(self):
        """Test QualityResult default values."""
        result = QualityResult(success=False)

        assert result.issues == []
        assert result.grade is None
        assert result.score == 0.0
        assert result.summary is None
        assert result.recommendations == []
        assert result.tests is None
        assert result.metrics == {}

    def test_result_to_dict_none_grade(self):
        """Test QualityResult.to_dict with None grade."""
        result = QualityResult(success=True)
        d = result.to_dict()
        assert d["quality_grade"] is None


class TestQualitySkillToolParameterValidation:
    """Tests for input parameter validation."""

    def test_valid_parameters(self):
        """Test valid parameters pass validation."""
        tool = QualitySkillTool()

        is_valid, error = tool.validate_inputs({
            "mode": "code_review",
            "code": "x = 1",
        })

        assert is_valid is True
        assert error is None

    def test_missing_required_mode(self):
        """Test missing mode is caught."""
        tool = QualitySkillTool()

        is_valid, error = tool.validate_inputs({
            "code": "x = 1",
        })

        assert is_valid is False
        assert "mode" in error.lower()

    def test_missing_required_code(self):
        """Test missing code is caught."""
        tool = QualitySkillTool()

        is_valid, error = tool.validate_inputs({
            "mode": "code_review",
        })

        assert is_valid is False
        assert "code" in error.lower()

    def test_invalid_mode_value(self):
        """Test invalid mode value is caught."""
        tool = QualitySkillTool()

        is_valid, _error = tool.validate_inputs({
            "mode": "nonexistent",
            "code": "x = 1",
        })

        assert is_valid is False

    def test_optional_params_omitted(self):
        """Test optional parameters can be omitted."""
        tool = QualitySkillTool()

        is_valid, _error = tool.validate_inputs({
            "mode": "security_audit",
            "code": "import os",
        })

        assert is_valid is True


class TestQualitySkillToolEdgeCases:
    """Tests for edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = QualitySkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock(spec=ExecutionContext)
        self.mock_ctx_manager.current_context = self.mock_context
        self.mock_context.execution_mode = ExecutionMode.EXECUTION
        self.mock_context.pre_execution_hooks = []
        self.mock_context.post_execution_hooks = []
        self.tool._context_manager = self.mock_ctx_manager

    def test_empty_code(self):
        """Test with empty code string."""
        result = self.tool.execute(
            mode="code_review",
            code="",
        )

        assert result.success is False

    def test_very_long_code(self):
        """Test with very long code input."""
        long_code = "x = 1\n" * 5000

        result = self.tool.execute(
            mode="code_review",
            code=long_code,
        )

        assert result.success is True

    def test_all_modes_execute_successfully(self):
        """Test all modes execute without errors."""
        for mode in QualityMode:
            result = self.tool.execute(
                mode=mode.value,
                code="def foo():\n    return 42\n",
            )
            assert result.success is True, f"Mode {mode.value} failed"

    def test_all_thinking_modes_accepted(self):
        """Test all thinking modes are accepted."""
        for tm in ThinkingMode:
            result = self.tool.execute(
                mode="code_review",
                code="x = 1",
                thinking_mode=tm.value,
            )
            assert result.success is True, f"ThinkingMode {tm.value} failed"

    def test_all_severity_thresholds_accepted(self):
        """Test all severity thresholds are accepted."""
        for sev in SeverityLevel:
            result = self.tool.execute(
                mode="code_review",
                code="x = 1",
                severity_threshold=sev.value,
            )
            assert result.success is True, f"Severity {sev.value} failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
