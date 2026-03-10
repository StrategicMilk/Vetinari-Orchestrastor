"""
Unified Quality Skill Tool
============================
Consolidated skill tool for the QUALITY agent role.

Unifies capabilities from 3 legacy agents:
  - EVALUATOR -> code_review, quality_assessment, best_practices modes
  - SECURITY_AUDITOR -> security_audit mode
  - TEST_AUTOMATION -> test_generation mode

Plus simplification and performance_review modes.

This module defines a security pattern scanner. The patterns listed in
SECURITY_PATTERNS are vulnerability signatures to DETECT in user code,
not patterns this module itself uses.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging
from enum import Enum

from vetinari.tool_interface import (
    Tool,
    ToolMetadata,
    ToolResult,
    ToolParameter,
    ToolCategory,
)
from vetinari.execution_context import ToolPermission, ExecutionMode
from vetinari.types import ThinkingMode, SeverityLevel, QualityGrade  # canonical enums

logger = logging.getLogger(__name__)


class QualityMode(str, Enum):
    """Modes of the unified quality skill."""
    CODE_REVIEW = "code_review"
    SECURITY_AUDIT = "security_audit"
    TEST_GENERATION = "test_generation"
    SIMPLIFICATION = "simplification"
    PERFORMANCE_REVIEW = "performance_review"
    BEST_PRACTICES = "best_practices"


OWASP_TOP_10 = [
    "A01:2021 - Broken Access Control",
    "A02:2021 - Cryptographic Failures",
    "A03:2021 - Injection",
    "A04:2021 - Insecure Design",
    "A05:2021 - Security Misconfiguration",
    "A06:2021 - Vulnerable and Outdated Components",
    "A07:2021 - Identification and Authentication Failures",
    "A08:2021 - Software and Data Integrity Failures",
    "A09:2021 - Security Logging and Monitoring Failures",
    "A10:2021 - Server-Side Request Forgery (SSRF)",
]


def _build_security_patterns() -> Dict[str, Dict[str, str]]:
    """Build the vulnerability detection patterns.

    These are signatures we SEARCH FOR in user code to flag potential
    security issues.  This module never invokes these patterns itself.
    """
    return {
        # A02 — Cryptographic Failures
        "yaml.load(": {
            "severity": "high", "cwe": "CWE-502",
            "owasp": "A02:2021",
            "desc": "Unsafe YAML load — use yaml.safe_load()",
        },
        "md5(": {
            "severity": "medium", "cwe": "CWE-328",
            "owasp": "A02:2021",
            "desc": "MD5 is cryptographically broken",
        },
        # A05 — Security Misconfiguration
        "debug=True": {
            "severity": "high", "cwe": "CWE-489",
            "owasp": "A05:2021",
            "desc": "Debug mode enabled in configuration",
        },
        # A07 — Identification / Authentication Failures
        "password =": {
            "severity": "critical", "cwe": "CWE-798",
            "owasp": "A07:2021",
            "desc": "Possible hardcoded password assignment",
        },
        "api_key =": {
            "severity": "critical", "cwe": "CWE-798",
            "owasp": "A07:2021",
            "desc": "Possible hardcoded API key assignment",
        },
        "secret =": {
            "severity": "critical", "cwe": "CWE-798",
            "owasp": "A07:2021",
            "desc": "Possible hardcoded secret assignment",
        },
    }


SECURITY_PATTERNS = _build_security_patterns()


@dataclass
class QualityIssue:
    """A single quality or security issue."""
    title: str
    severity: SeverityLevel
    description: str
    location: Optional[str] = None
    suggestion: Optional[str] = None
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "title": self.title,
            "severity": self.severity.value,
            "description": self.description,
            "suggestion": self.suggestion,
        }
        if self.location:
            result["location"] = self.location
        if self.cwe_id:
            result["cwe_id"] = self.cwe_id
        if self.owasp_category:
            result["owasp_category"] = self.owasp_category
        return result


@dataclass
class QualityResult:
    """Result of a quality operation."""
    success: bool
    issues: List[QualityIssue] = field(default_factory=list)
    grade: Optional[QualityGrade] = None
    score: float = 0.0
    summary: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)
    tests: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "issues": [i.to_dict() for i in self.issues],
            "quality_grade": self.grade.value if self.grade else None,
            "score": self.score,
            "summary": self.summary,
            "recommendations": self.recommendations,
            "tests": self.tests,
            "metrics": self.metrics,
        }


class QualitySkillTool(Tool):
    """
    Unified tool for the QUALITY consolidated agent.

    Replaces: EvaluatorSkillTool, SecurityAuditorSkill, TestAutomationSkill.

    Provides code review, security auditing, test generation, performance
    review, and simplification through a standardized Tool interface.
    """

    def __init__(self) -> None:
        metadata = ToolMetadata(
            name="quality",
            description=(
                "Code review, security audit, test generation, performance review, "
                "and simplification. Use for any code quality assessment."
            ),
            category=ToolCategory.SEARCH_ANALYSIS,
            version="1.1.0",
            author="Vetinari",
            parameters=[
                ToolParameter(
                    name="mode", type=str,
                    description="Quality mode to use",
                    required=True,
                    allowed_values=[m.value for m in QualityMode],
                ),
                ToolParameter(
                    name="code", type=str,
                    description="Code to review, audit, or generate tests for",
                    required=True,
                ),
                ToolParameter(
                    name="context", type=str,
                    description="File path, PR description, or context",
                    required=False,
                ),
                ToolParameter(
                    name="thinking_mode", type=str,
                    description="Review depth (low/medium/high/xhigh)",
                    required=False, default="medium",
                    allowed_values=[m.value for m in ThinkingMode],
                ),
                ToolParameter(
                    name="focus_areas", type=list,
                    description="Specific areas to prioritize",
                    required=False,
                ),
                ToolParameter(
                    name="severity_threshold", type=str,
                    description="Minimum severity to report",
                    required=False, default="low",
                    allowed_values=[s.value for s in SeverityLevel],
                ),
            ],
            required_permissions=[
                ToolPermission.FILE_READ,
                ToolPermission.MODEL_INFERENCE,
            ],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=["quality", "review", "security", "testing", "audit", "performance"],
        )
        super().__init__(metadata)

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute a quality operation."""
        try:
            mode_str = kwargs.get("mode")
            code = kwargs.get("code")
            context = kwargs.get("context")
            thinking_mode_str = kwargs.get("thinking_mode", "medium")
            focus_areas = kwargs.get("focus_areas", [])
            severity_threshold_str = kwargs.get("severity_threshold", "low")

            if not code:
                return ToolResult(success=False, output=None, error="code parameter is required")

            try:
                mode = QualityMode(mode_str)
            except ValueError:
                return ToolResult(success=False, output=None, error=f"Invalid mode: {mode_str}")

            try:
                thinking_mode = ThinkingMode(thinking_mode_str)
            except ValueError:
                return ToolResult(success=False, output=None, error=f"Invalid thinking_mode: {thinking_mode_str}")

            result = self._run_mode(mode, code, context, thinking_mode, focus_areas)

            # Filter by severity threshold
            severity_order = list(SeverityLevel)
            try:
                threshold = SeverityLevel(severity_threshold_str)
                threshold_idx = severity_order.index(threshold)
                result.issues = [
                    i for i in result.issues
                    if severity_order.index(i.severity) <= threshold_idx
                ]
            except ValueError:
                logger.debug("Invalid severity threshold value: %s", severity_threshold_str, exc_info=True)

            return ToolResult(
                success=result.success,
                output=result.to_dict(),
                error=None if result.success else "Quality assessment failed",
                metadata={
                    "mode": mode.value,
                    "thinking_mode": thinking_mode.value,
                    "issues_found": len(result.issues),
                    "quality_grade": result.grade.value if result.grade else None,
                },
            )
        except Exception as e:
            logger.error("Quality tool failed: %s", e, exc_info=True)
            return ToolResult(success=False, output=None, error=str(e))

    # ------------------------------------------------------------------
    # Mode dispatcher
    # ------------------------------------------------------------------

    def _run_mode(
        self, mode: QualityMode, code: str, context: Optional[str],
        thinking_mode: ThinkingMode, focus_areas: List[str],
    ) -> QualityResult:
        dispatch = {
            QualityMode.CODE_REVIEW: self._code_review,
            QualityMode.SECURITY_AUDIT: self._security_audit,
            QualityMode.TEST_GENERATION: self._generate_tests,
            QualityMode.SIMPLIFICATION: self._simplification_review,
            QualityMode.PERFORMANCE_REVIEW: self._performance_review,
            QualityMode.BEST_PRACTICES: self._best_practices,
        }
        handler = dispatch.get(mode)
        if handler is None:
            return QualityResult(success=False, summary=f"Unknown mode: {mode.value}")
        return handler(code, context, thinking_mode)

    # ------------------------------------------------------------------
    # MODE: code_review
    # ------------------------------------------------------------------

    def _code_review(
        self, code: str, context: Optional[str], thinking_mode: ThinkingMode,
    ) -> QualityResult:
        logger.info("Code review (depth: %s)", thinking_mode.value)
        issues: List[QualityIssue] = []
        recommendations: List[str] = []
        lines = code.split("\n")
        line_count = len(lines)

        if code.count("{") != code.count("}"):
            issues.append(QualityIssue(
                title="Unbalanced braces", severity=SeverityLevel.HIGH,
                description="Opening and closing braces do not match",
                suggestion="Verify brace pairing throughout the code",
            ))
        if code.count("(") != code.count(")"):
            issues.append(QualityIssue(
                title="Unbalanced parentheses", severity=SeverityLevel.HIGH,
                description="Opening and closing parentheses do not match",
                suggestion="Check parenthesis pairing",
            ))
        if "TODO" in code or "FIXME" in code:
            issues.append(QualityIssue(
                title="Unresolved TODOs/FIXMEs", severity=SeverityLevel.MEDIUM,
                description="Code contains TODO or FIXME comments",
                suggestion="Address all outstanding TODOs and FIXMEs before merge",
            ))
        if line_count > 300:
            issues.append(QualityIssue(
                title="File too long", severity=SeverityLevel.MEDIUM,
                description=f"File has {line_count} lines — exceeds 300-line guideline",
                suggestion="Break into smaller, focused modules",
            ))

        func_defs = [l for l in lines if l.strip().startswith(("def ", "async def "))]
        if len(func_defs) > 10:
            issues.append(QualityIssue(
                title="Too many functions", severity=SeverityLevel.LOW,
                description=f"{len(func_defs)} functions in a single file",
                suggestion="Split into multiple modules by responsibility",
            ))

        if thinking_mode in (ThinkingMode.HIGH, ThinkingMode.XHIGH):
            if "import *" in code:
                issues.append(QualityIssue(
                    title="Wildcard import", severity=SeverityLevel.MEDIUM,
                    description="Wildcard imports pollute namespace",
                    suggestion="Use explicit imports",
                ))
            recommendations.extend([
                "Add type hints to all function signatures",
                "Ensure docstrings on all public functions and classes",
            ])

        crit = sum(1 for i in issues if i.severity == SeverityLevel.CRITICAL)
        high = sum(1 for i in issues if i.severity == SeverityLevel.HIGH)
        score = max(0, 100 - crit * 25 - high * 10 - len(issues) * 3)
        grade = (QualityGrade.F if crit else
                 QualityGrade.D if high > 2 else
                 QualityGrade.C if len(issues) > 5 else
                 QualityGrade.B if issues else QualityGrade.A)

        return QualityResult(
            success=True, issues=issues, grade=grade, score=score,
            summary=f"Code review: {len(issues)} issue(s) in {line_count} lines. Grade: {grade.value}",
            recommendations=recommendations,
            metrics={"lines_of_code": line_count, "function_count": len(func_defs)},
        )

    # ------------------------------------------------------------------
    # MODE: security_audit
    # ------------------------------------------------------------------

    def _security_audit(
        self, code: str, context: Optional[str], thinking_mode: ThinkingMode,
    ) -> QualityResult:
        logger.info("Security audit (depth: %s)", thinking_mode.value)
        issues: List[QualityIssue] = []
        code_lower = code.lower()

        for pattern, info in SECURITY_PATTERNS.items():
            if pattern.lower() in code_lower:
                issues.append(QualityIssue(
                    title=f"Security: {info['desc']}",
                    severity=SeverityLevel(info["severity"]),
                    description=f"Pattern '{pattern}' detected",
                    suggestion=f"Review usage. See {info['cwe']}",
                    cwe_id=info["cwe"],
                    owasp_category=info["owasp"],
                ))

        if "except:" in code and "except Exception" not in code:
            issues.append(QualityIssue(
                title="Bare except clause", severity=SeverityLevel.MEDIUM,
                description="Bare except catches SystemExit and KeyboardInterrupt",
                suggestion="Use 'except Exception as e:' and log the error",
                cwe_id="CWE-396", owasp_category="A09:2021",
            ))

        covered = set(i.owasp_category for i in issues if i.owasp_category)
        crit = sum(1 for i in issues if i.severity == SeverityLevel.CRITICAL)
        score = max(0, 100 - crit * 30 - len(issues) * 5)
        grade = (QualityGrade.F if crit else
                 QualityGrade.C if len(issues) > 3 else
                 QualityGrade.B if issues else QualityGrade.A)

        return QualityResult(
            success=True, issues=issues, grade=grade, score=score,
            summary=f"Security audit: {len(issues)} issue(s). {len(covered)} OWASP categories covered.",
            recommendations=[
                f"OWASP coverage: {len(covered)}/{len(OWASP_TOP_10)} categories",
                "Use parameterized queries to prevent injection (A03)",
                "Store secrets in environment variables (A07)",
                "Enable security logging for auth events (A09)",
            ],
            metrics={"security_issues_count": len(issues), "owasp_coverage": len(covered)},
        )

    # ------------------------------------------------------------------
    # MODE: test_generation
    # ------------------------------------------------------------------

    def _generate_tests(
        self, code: str, context: Optional[str], thinking_mode: ThinkingMode,
    ) -> QualityResult:
        logger.info("Test generation (depth: %s)", thinking_mode.value)
        funcs = [l.strip() for l in code.split("\n") if l.strip().startswith(("def ", "async def "))]
        classes = [l.strip() for l in code.split("\n") if l.strip().startswith("class ")]

        categories = ["happy_path", "edge_cases", "error_cases"]
        if thinking_mode in (ThinkingMode.HIGH, ThinkingMode.XHIGH):
            categories.extend(["integration", "performance"])

        est = len(funcs) * len(categories)
        return QualityResult(
            success=True, issues=[], grade=QualityGrade.B, score=80,
            summary=f"Test strategy: {est} tests across {len(categories)} categories.",
            recommendations=[
                "Follow Arrange-Act-Assert pattern",
                f"Cover {len(funcs)} functions and {len(classes)} classes",
                f"Categories: {', '.join(categories)}",
                "Mock external dependencies",
                "Naming: test_<fn>_<scenario>_<expected>",
            ],
            metrics={"testable_functions": len(funcs), "testable_classes": len(classes),
                     "estimated_tests": est, "test_categories": categories},
        )

    # ------------------------------------------------------------------
    # MODE: simplification
    # ------------------------------------------------------------------

    def _simplification_review(
        self, code: str, context: Optional[str], thinking_mode: ThinkingMode,
    ) -> QualityResult:
        logger.info("Simplification review (depth: %s)", thinking_mode.value)
        issues: List[QualityIssue] = []
        lines = code.split("\n")

        if len(lines) > 200:
            issues.append(QualityIssue(
                title="Long file", severity=SeverityLevel.LOW,
                description=f"{len(lines)} lines — candidate for splitting",
                suggestion="Split by single responsibility principle",
            ))

        max_indent = max((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
        if max_indent > 20:
            issues.append(QualityIssue(
                title="Deep nesting", severity=SeverityLevel.MEDIUM,
                description=f"Max indentation: {max_indent // 4} levels",
                suggestion="Use early returns, guard clauses, or extract helpers",
            ))

        return QualityResult(
            success=True, issues=issues,
            grade=QualityGrade.B if len(issues) <= 2 else QualityGrade.C,
            score=max(0, 100 - len(issues) * 10),
            summary=f"Simplification: {len(issues)} opportunities.",
            recommendations=["Early return pattern", "Extract complex conditionals",
                             "Self-documenting code over comments", "Data-driven patterns"],
        )

    # ------------------------------------------------------------------
    # MODE: performance_review
    # ------------------------------------------------------------------

    def _performance_review(
        self, code: str, context: Optional[str], thinking_mode: ThinkingMode,
    ) -> QualityResult:
        logger.info("Performance review (depth: %s)", thinking_mode.value)
        issues: List[QualityIssue] = []
        for_count = code.count("for ")

        if for_count > 2:
            issues.append(QualityIssue(
                title="Potential quadratic complexity", severity=SeverityLevel.MEDIUM,
                description=f"{for_count} for-loops — check for nested iteration",
                suggestion="Use dict/set lookups or algorithmic optimization",
            ))
        if "while True" in code:
            issues.append(QualityIssue(
                title="Infinite loop risk", severity=SeverityLevel.HIGH,
                description="while True needs clear exit conditions",
                suggestion="Add max_iterations guard and termination condition",
            ))

        return QualityResult(
            success=True, issues=issues,
            grade=QualityGrade.B if len(issues) <= 1 else QualityGrade.C,
            score=max(0, 100 - len(issues) * 15),
            summary=f"Performance: {len(issues)} potential issue(s).",
            recommendations=["Profile before optimizing", "Use dict/set for lookups",
                             "functools.lru_cache for expensive computations",
                             "Generators for large datasets"],
            metrics={"loop_count": for_count},
        )

    # ------------------------------------------------------------------
    # MODE: best_practices
    # ------------------------------------------------------------------

    def _best_practices(
        self, code: str, context: Optional[str], thinking_mode: ThinkingMode,
    ) -> QualityResult:
        logger.info("Best practices (depth: %s)", thinking_mode.value)
        issues: List[QualityIssue] = []

        if "except:" in code:
            issues.append(QualityIssue(
                title="Bare except", severity=SeverityLevel.MEDIUM,
                description="Catches SystemExit and KeyboardInterrupt",
                suggestion="Use 'except Exception as e:'",
            ))
        if "def " in code and "=[]" in code.replace(" ", ""):
            issues.append(QualityIssue(
                title="Mutable default argument", severity=SeverityLevel.HIGH,
                description="Mutable defaults are shared between calls",
                suggestion="Use None default, create list inside function",
            ))
        if "global " in code:
            issues.append(QualityIssue(
                title="Global variable usage", severity=SeverityLevel.MEDIUM,
                description="Globals make code harder to test",
                suggestion="Use parameters or dependency injection",
            ))

        grade = (QualityGrade.A if not issues else
                 QualityGrade.B if len(issues) <= 2 else QualityGrade.C)
        return QualityResult(
            success=True, issues=issues, grade=grade,
            score=max(0, 100 - len(issues) * 10),
            summary=f"Best practices: {len(issues)} issue(s).",
            recommendations=["SOLID principles", "Composition over inheritance",
                             "Consistent type hints", "Docstrings on public APIs",
                             "Functions under 50 lines"],
        )
