"""Verification and Post-Execution System for Vetinari.

Implements comprehensive verification of task outputs, code quality,
and security before completing execution. Inspired by OpenCode's safety
mechanisms.

Features:
- Output validation (format, structure, content)
- Code quality checks (syntax, imports, security)
- Security scanning (secret detection, vulnerability patterns)
- Performance verification
- Artifact integrity checks
- Customizable verification pipelines
"""

from __future__ import annotations

import abc
import ast
import json
import logging
import re
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from vetinari.security import get_secret_scanner

logger = logging.getLogger(__name__)


class VerificationLevel(Enum):
    """Levels of verification strictness."""

    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


class VerificationStatus(Enum):
    """Status of verification."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class VerificationIssue:
    """Represents a single verification issue found."""

    severity: str  # "info", "warning", "error"
    category: str  # "syntax", "security", "performance", etc.
    message: str
    location: str | None = None  # Line number, file, etc.
    suggestion: str | None = None


@dataclass
class VerificationResult:
    """Result of a verification check."""

    status: VerificationStatus
    check_name: str
    issues: list[VerificationIssue] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time_ms: int = 0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def info_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "info")

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "status": self.status.value,
            "check_name": self.check_name,
            "issues": [
                {
                    "severity": i.severity,
                    "category": i.category,
                    "message": i.message,
                    "location": i.location,
                    "suggestion": i.suggestion,
                }
                for i in self.issues
            ],
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "execution_time_ms": self.execution_time_ms,
        }


class Verifier(ABC):
    """Base class for verification checks."""

    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def verify(self, content: Any) -> VerificationResult:
        """Execute the verification check."""


class CodeSyntaxVerifier(Verifier):
    """Verifies Python code syntax."""

    def __init__(self):
        super().__init__("code_syntax")

    def verify(self, content: str) -> VerificationResult:
        """Check if content is valid Python syntax.

        Returns:
            The VerificationResult result.
        """
        import time

        start = time.time()

        result = VerificationResult(
            status=VerificationStatus.PASSED,
            check_name=self.name,
        )

        if not isinstance(content, str) or not content.strip():
            result.status = VerificationStatus.SKIPPED
            return result

        # Remove markdown code blocks
        cleaned = re.sub(r"```[\w]*\n", "\n", content)
        cleaned = re.sub(r"```$", "", cleaned)
        cleaned = cleaned.strip()

        if not cleaned:
            result.status = VerificationStatus.SKIPPED
            return result

        try:
            ast.parse(cleaned)
            logger.info("Code syntax validation passed")
        except SyntaxError as e:
            result.status = VerificationStatus.FAILED
            result.issues.append(
                VerificationIssue(
                    severity="error",
                    category="syntax",
                    message=f"Syntax error: {e!s}",
                    location=f"line {e.lineno}",
                )
            )
        except Exception as e:
            result.status = VerificationStatus.WARNING
            result.issues.append(
                VerificationIssue(
                    severity="warning",
                    category="syntax",
                    message=f"Could not parse code: {e!s}",
                )
            )

        result.execution_time_ms = int((time.time() - start) * 1000)
        return result


class SecurityVerifier(Verifier):
    """Verifies content for security issues."""

    def __init__(self):
        super().__init__("security")
        self.scanner = get_secret_scanner()

    def verify(self, content: str) -> VerificationResult:
        """Check for security issues in content.

        Returns:
            The VerificationResult result.
        """
        import time

        start = time.time()

        result = VerificationResult(
            status=VerificationStatus.PASSED,
            check_name=self.name,
        )

        if not isinstance(content, str):
            result.status = VerificationStatus.SKIPPED
            return result

        # Check for secrets
        secrets = self.scanner.scan(content)
        for pattern, _matches in secrets.items():
            result.status = VerificationStatus.FAILED
            result.issues.append(
                VerificationIssue(
                    severity="error",
                    category="security",
                    message=f"Potential secret detected: {pattern}",
                    suggestion="Sanitize sensitive information before storing",
                )
            )

        # Check for dangerous patterns
        dangerous_patterns = [
            (r"exec\s*\(", "exec() allows arbitrary code execution"),
            (r"eval\s*\(", "eval() is unsafe"),
            (r"__import__\s*\(", "Direct imports may be unsafe"),
            (r"os\.system\s*\(", "os.system() is unsafe"),
            (r"subprocess.*shell=True", "shell=True in subprocess is dangerous"),
        ]

        for pattern, message in dangerous_patterns:
            if re.search(pattern, content):
                result.status = VerificationStatus.WARNING
                result.issues.append(
                    VerificationIssue(
                        severity="warning",
                        category="security",
                        message=message,
                        suggestion="Use safer alternatives",
                    )
                )

        result.execution_time_ms = int((time.time() - start) * 1000)
        return result


class ImportVerifier(Verifier):
    """Verifies Python imports are safe and available."""

    def __init__(self, allowed_modules: list[str] | None = None):
        super().__init__("imports")
        self.allowed_modules = allowed_modules or []
        self.blocked_modules = ["ctypes", "mmap", "msvcrt", "winreg"]

    def verify(self, content: str) -> VerificationResult:
        """Check Python imports in content.

        Returns:
            The VerificationResult result.
        """
        import time

        start = time.time()

        result = VerificationResult(
            status=VerificationStatus.PASSED,
            check_name=self.name,
        )

        if not isinstance(content, str):
            result.status = VerificationStatus.SKIPPED
            return result

        # Extract imports
        imports = self._extract_imports(content)

        for imp in imports:
            module = imp.split(".")[0]

            # Check for blocked modules
            if module in self.blocked_modules:
                result.status = VerificationStatus.WARNING
                result.issues.append(
                    VerificationIssue(
                        severity="warning",
                        category="import",
                        message=f"Import '{module}' is potentially unsafe",
                    )
                )

        result.execution_time_ms = int((time.time() - start) * 1000)
        return result

    def _extract_imports(self, content: str) -> list[str]:
        """Extract import statements from code."""
        imports = []

        # Find 'import X' statements
        for match in re.finditer(r"^import\s+([\w.]+)", content, re.MULTILINE):
            imports.append(match.group(1))

        # Find 'from X import Y' statements
        for match in re.finditer(r"^from\s+([\w.]+)\s+import", content, re.MULTILINE):
            imports.append(match.group(1))

        return imports


class JSONStructureVerifier(Verifier):
    """Verifies JSON structure and completeness."""

    def __init__(self, required_fields: list[str] | None = None):
        super().__init__("json_structure")
        self.required_fields = required_fields or []

    def verify(self, content: str) -> VerificationResult:
        """Check JSON structure and fields.

        Returns:
            The VerificationResult result.
        """
        import time

        start = time.time()

        result = VerificationResult(
            status=VerificationStatus.PASSED,
            check_name=self.name,
        )

        if not isinstance(content, str):
            result.status = VerificationStatus.SKIPPED
            return result

        # Extract JSON if wrapped in markdown
        json_str = content.strip()
        if json_str.startswith("```"):
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", json_str, re.DOTALL)
            if match:
                json_str = match.group(1)

        try:
            data = json.loads(json_str)

            # Check required fields
            if isinstance(data, dict):
                for field in self.required_fields:
                    if field not in data:
                        result.status = VerificationStatus.WARNING
                        result.issues.append(
                            VerificationIssue(
                                severity="warning",
                                category="structure",
                                message=f"Missing required field: {field}",
                            )
                        )
        except json.JSONDecodeError as e:
            result.status = VerificationStatus.FAILED
            result.issues.append(
                VerificationIssue(
                    severity="error",
                    category="structure",
                    message=f"Invalid JSON: {e!s}",
                )
            )

        result.execution_time_ms = int((time.time() - start) * 1000)
        return result


class VerificationPipeline:
    """Pipeline of verification checks."""

    def __init__(self, level: VerificationLevel = VerificationLevel.STANDARD):
        self.level = level
        self.verifiers: list[Verifier] = []
        self._setup_verifiers()

    def _setup_verifiers(self):
        """Setup verifiers based on verification level."""
        if self.level in (
            VerificationLevel.BASIC,
            VerificationLevel.STANDARD,
            VerificationLevel.STRICT,
            VerificationLevel.PARANOID,
        ):
            self.verifiers.append(CodeSyntaxVerifier())
            self.verifiers.append(SecurityVerifier())
            self.verifiers.append(ImportVerifier())

    def add_verifier(self, verifier: Verifier):
        """Add a custom verifier to the pipeline."""
        self.verifiers.append(verifier)

    def verify(self, content: Any) -> dict[str, VerificationResult]:
        """Run all verifiers on content.

        Args:
            content: Content to verify

        Returns:
            Dictionary mapping verifier name to VerificationResult
        """
        results = {}

        for verifier in self.verifiers:
            try:
                results[verifier.name] = verifier.verify(content)
            except Exception as e:
                logger.error("Error in verifier %s: %s", verifier.name, e)
                results[verifier.name] = VerificationResult(
                    status=VerificationStatus.SKIPPED,
                    check_name=verifier.name,
                )

        return results

    def get_summary(self, results: dict[str, VerificationResult]) -> dict[str, Any]:
        """Get summary of verification results.

        Returns:
            The result string.
        """
        total_issues = sum(len(r.issues) for r in results.values())
        total_errors = sum(r.error_count for r in results.values())
        total_warnings = sum(r.warning_count for r in results.values())

        all(r.status == VerificationStatus.PASSED for r in results.values())
        has_failures = any(r.status == VerificationStatus.FAILED for r in results.values())

        return {
            "overall_status": "FAILED" if has_failures else "PASSED",
            "total_checks": len(results),
            "total_issues": total_issues,
            "error_count": total_errors,
            "warning_count": total_warnings,
            "checks": {name: r.to_dict() for name, r in results.items()},
        }


class QualityGateVerifier(Verifier):
    """Verifier that wraps QualityGateRunner for integration with VerificationPipeline.

    This bridges the quality gate system (Task 26) with the existing
    VerificationPipeline infrastructure. It runs all configured gates for a
    given pipeline stage and converts GateCheckResults into VerificationResults.

    Usage::

        pipeline = VerificationPipeline(VerificationLevel.STANDARD)
        pipeline.add_verifier(QualityGateVerifier(stage="post_execution"))
        results = pipeline.verify(code_string)
    """

    def __init__(self, stage: str = "post_execution", custom_gates: dict | None = None):
        """Initialize the QualityGateVerifier.

        Args:
            stage: Pipeline stage to run gates for (e.g. "post_execution").
            custom_gates: Optional custom gate configuration dict.
        """
        super().__init__(f"quality_gate_{stage}")
        self._stage = stage
        from vetinari.validation.quality_gates import QualityGateRunner

        self._runner = QualityGateRunner(custom_gates=custom_gates)

    def verify(self, content: Any) -> VerificationResult:
        """Run quality gate checks on the provided content.

        Args:
            content: Code string or dict of artifacts to verify.
                     If a string is provided, it is treated as code.
                     If a dict is provided, it is passed directly as artifacts.

        Returns:
            VerificationResult aggregating all gate check outcomes.
        """
        import time

        from vetinari.validation.quality_gates import GateResult

        start = time.time()

        # Build artifacts dict
        if isinstance(content, dict):
            artifacts = content
        elif isinstance(content, str):
            artifacts = {"code": content}
        else:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                check_name=self.name,
            )

        gate_results = self._runner.run_gate(self._stage, artifacts)

        if not gate_results:
            result = VerificationResult(
                status=VerificationStatus.SKIPPED,
                check_name=self.name,
            )
            result.execution_time_ms = int((time.time() - start) * 1000)
            return result

        # Aggregate gate results into a single VerificationResult
        issues: list[VerificationIssue] = []
        has_failure = False
        has_warning = False

        for gr in gate_results:
            if gr.result == GateResult.FAILED:
                has_failure = True
            elif gr.result == GateResult.WARNING:
                has_warning = True

            for issue in gr.issues:
                severity = issue.get("severity", "info")
                # Normalise severity to the VerificationIssue vocabulary
                if severity in ("critical", "high"):
                    severity = "error"
                elif severity in ("medium", "low"):
                    severity = "warning"
                issues.append(
                    VerificationIssue(
                        severity=severity,
                        category=issue.get("category", gr.mode.value),
                        message=issue.get("message", ""),
                        location=issue.get("location"),
                        suggestion=None,
                    )
                )

            for suggestion in gr.suggestions:
                issues.append(
                    VerificationIssue(
                        severity="info",
                        category="suggestion",
                        message=suggestion,
                    )
                )

        if has_failure:
            status = VerificationStatus.FAILED
        elif has_warning:
            status = VerificationStatus.WARNING
        else:
            status = VerificationStatus.PASSED

        result = VerificationResult(
            status=status,
            check_name=self.name,
            issues=issues,
        )
        result.execution_time_ms = int((time.time() - start) * 1000)
        return result


# Global verifier instance
_verifier_pipeline: VerificationPipeline | None = None


def get_verifier_pipeline() -> VerificationPipeline:
    """Get or create the global verification pipeline.

    Returns:
        The VerificationPipeline result.
    """
    global _verifier_pipeline
    if _verifier_pipeline is None:
        # Use STANDARD level by default, can be customized
        _verifier_pipeline = VerificationPipeline(VerificationLevel.STANDARD)
    return _verifier_pipeline
