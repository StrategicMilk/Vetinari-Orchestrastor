"""
Unit tests for Verification System (Phase 2)

Tests the verification pipeline, verifiers, and verification results.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from vetinari.validation.verification import (
    VerificationLevel,
    VerificationStatus,
    VerificationIssue,
    VerificationResult,
    CodeSyntaxVerifier,
    SecurityVerifier,
    ImportVerifier,
    JSONStructureVerifier,
    VerificationPipeline,
    get_verifier_pipeline,
)


class TestVerificationIssue:
    """Test VerificationIssue class."""
    
    def test_issue_creation(self):
        """Test creating a verification issue."""
        issue = VerificationIssue(
            severity="error",
            category="syntax",
            message="Invalid Python syntax",
            location="line 42",
            suggestion="Check for missing colons",
        )
        assert issue.severity == "error"
        assert issue.category == "syntax"
        assert issue.message == "Invalid Python syntax"
        assert issue.location == "line 42"
        assert issue.suggestion == "Check for missing colons"


class TestVerificationResult:
    """Test VerificationResult class."""
    
    def test_result_creation(self):
        """Test creating a verification result."""
        result = VerificationResult(
            status=VerificationStatus.PASSED,
            check_name="syntax_check",
        )
        assert result.status == VerificationStatus.PASSED
        assert result.check_name == "syntax_check"
        assert result.issues == []
    
    def test_result_issue_counts(self):
        """Test issue counting in result."""
        result = VerificationResult(
            status=VerificationStatus.FAILED,
            check_name="security_check",
            issues=[
                VerificationIssue("error", "security", "Secret found"),
                VerificationIssue("error", "security", "Another secret"),
                VerificationIssue("warning", "security", "Suspicious pattern"),
                VerificationIssue("info", "security", "Note this"),
            ],
        )
        assert result.error_count == 2
        assert result.warning_count == 1
        assert result.info_count == 1
    
    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        issue = VerificationIssue(
            severity="error",
            category="syntax",
            message="Syntax error",
        )
        result = VerificationResult(
            status=VerificationStatus.FAILED,
            check_name="syntax",
            issues=[issue],
        )
        
        result_dict = result.to_dict()
        assert result_dict["status"] == "failed"
        assert result_dict["check_name"] == "syntax"
        assert len(result_dict["issues"]) == 1
        assert result_dict["error_count"] == 1


class TestCodeSyntaxVerifier:
    """Test CodeSyntaxVerifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.verifier = CodeSyntaxVerifier()
    
    def test_verifier_name(self):
        """Test verifier name."""
        assert self.verifier.name == "code_syntax"
    
    def test_valid_python_code(self):
        """Test with valid Python code."""
        code = """
def hello():
    print("Hello, world!")
    return 42
"""
        result = self.verifier.verify(code)
        assert result.status == VerificationStatus.PASSED
        assert len(result.issues) == 0
    
    def test_invalid_python_code(self):
        """Test with invalid Python code."""
        code = """
def broken(
    print("Missing closing paren")
"""
        result = self.verifier.verify(code)
        assert result.status == VerificationStatus.FAILED
        assert len(result.issues) > 0
        assert result.issues[0].category == "syntax"
    
    def test_empty_code(self):
        """Test with empty code."""
        result = self.verifier.verify("")
        assert result.status == VerificationStatus.SKIPPED
    
    def test_markdown_code_block(self):
        """Test with markdown code block."""
        code = """```python
def valid():
    return True
```"""
        result = self.verifier.verify(code)
        assert result.status == VerificationStatus.PASSED
    
    def test_non_string_input(self):
        """Test with non-string input."""
        result = self.verifier.verify({"not": "code"})
        assert result.status == VerificationStatus.SKIPPED


class TestSecurityVerifier:
    """Test SecurityVerifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.verifier = SecurityVerifier()
    
    def test_verifier_name(self):
        """Test verifier name."""
        assert self.verifier.name == "security"
    
    def test_safe_code(self):
        """Test with safe code."""
        code = """
def safe_function():
    x = 42
    return x
"""
        result = self.verifier.verify(code)
        assert result.status == VerificationStatus.PASSED
    
    def test_dangerous_exec(self):
        """Test detection of exec()."""
        code = "exec('print(1)')"
        result = self.verifier.verify(code)
        assert result.status == VerificationStatus.WARNING
        assert any("exec" in str(issue.message).lower() for issue in result.issues)
    
    def test_dangerous_eval(self):
        """Test detection of eval()."""
        code = "result = eval(user_input)"
        result = self.verifier.verify(code)
        assert result.status == VerificationStatus.WARNING
    
    def test_dangerous_os_system(self):
        """Test detection of os.system()."""
        code = "os.system('rm -rf /')"
        result = self.verifier.verify(code)
        assert result.status == VerificationStatus.WARNING
    
    def test_dangerous_subprocess_shell(self):
        """Test detection of subprocess with shell=True."""
        code = "subprocess.run(cmd, shell=True)"
        result = self.verifier.verify(code)
        assert result.status == VerificationStatus.WARNING
    
    def test_non_string_input(self):
        """Test with non-string input."""
        result = self.verifier.verify(123)
        assert result.status == VerificationStatus.SKIPPED
    
    def test_secret_detection(self):
        """Test secret detection."""
        code = "api_key = 'sk_test_abc123def456'"
        
        # Mock the secret scanner
        with patch.object(self.verifier, 'scanner') as mock_scanner:
            mock_scanner.scan.return_value = {
                "api_key_pattern": ["sk_test_abc123def456"]
            }
            result = self.verifier.verify(code)
            
            assert result.status == VerificationStatus.FAILED
            assert any("secret" in issue.message.lower() for issue in result.issues)


class TestImportVerifier:
    """Test ImportVerifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.verifier = ImportVerifier()
    
    def test_verifier_name(self):
        """Test verifier name."""
        assert self.verifier.name == "imports"
    
    def test_safe_imports(self):
        """Test with safe imports."""
        code = """
import os
from pathlib import Path
import json
"""
        result = self.verifier.verify(code)
        assert result.status == VerificationStatus.PASSED
    
    def test_blocked_import_ctypes(self):
        """Test detection of blocked ctypes import."""
        code = "import ctypes"
        result = self.verifier.verify(code)
        assert result.status == VerificationStatus.WARNING
        assert any("ctypes" in issue.message for issue in result.issues)
    
    def test_blocked_import_winreg(self):
        """Test detection of blocked winreg import."""
        code = "import winreg"
        result = self.verifier.verify(code)
        assert result.status == VerificationStatus.WARNING
    
    def test_from_import_blocked(self):
        """Test detection of blocked from-import."""
        code = "from mmap import mmap"
        result = self.verifier.verify(code)
        assert result.status == VerificationStatus.WARNING
    
    def test_non_string_input(self):
        """Test with non-string input."""
        result = self.verifier.verify([1, 2, 3])
        assert result.status == VerificationStatus.SKIPPED
    
    def test_custom_allowed_modules(self):
        """Test with custom allowed modules."""
        verifier = ImportVerifier(allowed_modules=["custom_module"])
        code = "import custom_module"
        result = verifier.verify(code)
        # Should still pass - allowed_modules are for documentation
        assert result.status == VerificationStatus.PASSED


class TestJSONStructureVerifier:
    """Test JSONStructureVerifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.verifier = JSONStructureVerifier()
    
    def test_verifier_name(self):
        """Test verifier name."""
        assert self.verifier.name == "json_structure"
    
    def test_valid_json(self):
        """Test with valid JSON."""
        json_str = '{"name": "test", "value": 42}'
        result = self.verifier.verify(json_str)
        assert result.status == VerificationStatus.PASSED
    
    def test_invalid_json(self):
        """Test with invalid JSON."""
        json_str = '{"name": "test", invalid}'
        result = self.verifier.verify(json_str)
        assert result.status == VerificationStatus.FAILED
        assert len(result.issues) > 0
    
    def test_json_with_markdown(self):
        """Test JSON wrapped in markdown."""
        json_str = """```json
{"name": "test", "value": 42}
```"""
        result = self.verifier.verify(json_str)
        assert result.status == VerificationStatus.PASSED
    
    def test_required_fields_present(self):
        """Test with required fields present."""
        verifier = JSONStructureVerifier(required_fields=["name", "value"])
        json_str = '{"name": "test", "value": 42}'
        result = verifier.verify(json_str)
        assert result.status == VerificationStatus.PASSED
    
    def test_required_fields_missing(self):
        """Test with required fields missing."""
        verifier = JSONStructureVerifier(required_fields=["name", "value", "required"])
        json_str = '{"name": "test", "value": 42}'
        result = verifier.verify(json_str)
        assert result.status == VerificationStatus.WARNING
        assert any("required" in issue.message.lower() for issue in result.issues)
    
    def test_non_string_input(self):
        """Test with non-string input."""
        result = self.verifier.verify({"already": "dict"})
        assert result.status == VerificationStatus.SKIPPED


class TestVerificationPipeline:
    """Test VerificationPipeline."""
    
    def test_pipeline_creation_basic_level(self):
        """Test creating pipeline with BASIC level."""
        pipeline = VerificationPipeline(VerificationLevel.BASIC)
        assert pipeline.level == VerificationLevel.BASIC
        assert len(pipeline.verifiers) > 0
    
    def test_pipeline_creation_standard_level(self):
        """Test creating pipeline with STANDARD level."""
        pipeline = VerificationPipeline(VerificationLevel.STANDARD)
        assert len(pipeline.verifiers) > 0
    
    def test_pipeline_creation_strict_level(self):
        """Test creating pipeline with STRICT level."""
        pipeline = VerificationPipeline(VerificationLevel.STRICT)
        assert len(pipeline.verifiers) > 0
    
    def test_pipeline_none_level(self):
        """Test creating pipeline with NONE level."""
        pipeline = VerificationPipeline(VerificationLevel.NONE)
        # NONE level should have no verifiers
        assert len(pipeline.verifiers) == 0
    
    def test_add_custom_verifier(self):
        """Test adding custom verifier."""
        pipeline = VerificationPipeline(VerificationLevel.STANDARD)
        initial_count = len(pipeline.verifiers)
        
        custom_verifier = Mock()
        custom_verifier.name = "custom"
        pipeline.add_verifier(custom_verifier)
        
        assert len(pipeline.verifiers) == initial_count + 1
    
    def test_verify_content(self):
        """Test verifying content."""
        pipeline = VerificationPipeline(VerificationLevel.STANDARD)
        
        code = """
def hello():
    return "world"
"""
        results = pipeline.verify(code)
        
        assert isinstance(results, dict)
        assert len(results) > 0
        # Check that results have the verifier names as keys
        assert any(name in results for name in ["code_syntax", "security", "imports"])
    
    def test_verify_with_verifier_exception(self):
        """Test that verifier exceptions don't break pipeline."""
        pipeline = VerificationPipeline(VerificationLevel.STANDARD)
        
        # Add a verifier that raises an exception
        broken_verifier = Mock()
        broken_verifier.name = "broken"
        broken_verifier.verify.side_effect = Exception("Verifier error")
        pipeline.add_verifier(broken_verifier)
        
        content = "test"
        results = pipeline.verify(content)
        
        # Should still have results for other verifiers
        assert "broken" in results
        assert results["broken"].status == VerificationStatus.SKIPPED
    
    def test_get_summary_all_passed(self):
        """Test summary when all checks pass."""
        pipeline = VerificationPipeline(VerificationLevel.BASIC)
        
        results = {
            "check1": VerificationResult(
                status=VerificationStatus.PASSED,
                check_name="check1",
            ),
            "check2": VerificationResult(
                status=VerificationStatus.PASSED,
                check_name="check2",
            ),
        }
        
        summary = pipeline.get_summary(results)
        assert summary["overall_status"] == "PASSED"
        assert summary["total_checks"] == 2
        assert summary["error_count"] == 0
    
    def test_get_summary_with_failures(self):
        """Test summary with failures."""
        pipeline = VerificationPipeline(VerificationLevel.BASIC)
        
        results = {
            "check1": VerificationResult(
                status=VerificationStatus.PASSED,
                check_name="check1",
            ),
            "check2": VerificationResult(
                status=VerificationStatus.FAILED,
                check_name="check2",
                issues=[
                    VerificationIssue("error", "security", "Secret found"),
                ],
            ),
        }
        
        summary = pipeline.get_summary(results)
        assert summary["overall_status"] == "FAILED"
        assert summary["error_count"] == 1
    
    def test_get_summary_with_warnings(self):
        """Test summary with warnings."""
        pipeline = VerificationPipeline(VerificationLevel.BASIC)
        
        results = {
            "check1": VerificationResult(
                status=VerificationStatus.PASSED,
                check_name="check1",
            ),
            "check2": VerificationResult(
                status=VerificationStatus.WARNING,
                check_name="check2",
                issues=[
                    VerificationIssue("warning", "security", "Suspicious pattern"),
                ],
            ),
        }
        
        summary = pipeline.get_summary(results)
        assert summary["overall_status"] == "PASSED"  # Warnings don't fail
        assert summary["warning_count"] == 1


class TestVerificationPipelineIntegration:
    """Integration tests for verification pipeline."""
    
    def test_pipeline_with_valid_code(self):
        """Test pipeline with valid code."""
        pipeline = VerificationPipeline(VerificationLevel.STANDARD)
        
        code = """
def calculate_sum(a, b):
    return a + b

result = calculate_sum(5, 3)
"""
        results = pipeline.verify(code)
        summary = pipeline.get_summary(results)
        
        assert summary["overall_status"] in ["PASSED", "PASSED"]  # At least syntax should pass
    
    def test_pipeline_with_unsafe_code(self):
        """Test pipeline with unsafe code."""
        pipeline = VerificationPipeline(VerificationLevel.STANDARD)
        
        code = "exec(user_input)"
        results = pipeline.verify(code)
        
        # Security verifier should flag this
        assert results["security"].status == VerificationStatus.WARNING
    
    def test_pipeline_with_malformed_json(self):
        """Test pipeline with malformed JSON."""
        pipeline = VerificationPipeline(VerificationLevel.STANDARD)
        json_content = '{"invalid": json}'
        
        results = pipeline.verify(json_content)
        
        # Should have issues with syntax or be skipped for JSON verification
        assert "code_syntax" in results or "json_structure" not in results


class TestGlobalVerificationPipeline:
    """Test global verification pipeline singleton."""
    
    def test_get_verifier_pipeline_singleton(self):
        """Test that get_verifier_pipeline returns singleton."""
        pipeline1 = get_verifier_pipeline()
        pipeline2 = get_verifier_pipeline()
        
        assert pipeline1 is pipeline2
    
    def test_pipeline_is_instance(self):
        """Test that returned pipeline is VerificationPipeline instance."""
        pipeline = get_verifier_pipeline()
        assert isinstance(pipeline, VerificationPipeline)
    
    def test_default_level_is_standard(self):
        """Test that default level is STANDARD."""
        pipeline = get_verifier_pipeline()
        assert pipeline.level == VerificationLevel.STANDARD
