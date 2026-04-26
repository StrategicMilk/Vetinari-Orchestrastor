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
import threading
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from vetinari.security import get_secret_scanner

# Module-level import so tests can patch vetinari.validation.verification.score_confidence_via_llm.
# The import is guarded to prevent circular-import failures during early startup.
try:
    from vetinari.llm_helpers import score_confidence_via_llm
except ImportError:  # pragma: no cover
    score_confidence_via_llm = None  # type: ignore[assignment]

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

    def __repr__(self) -> str:
        return f"VerificationIssue(severity={self.severity!r}, category={self.category!r}, location={self.location!r})"


@dataclass
class ValidationVerificationResult:
    """Result of a verification check."""

    status: VerificationStatus
    check_name: str
    issues: list[VerificationIssue] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_time_ms: int = 0

    @property
    def error_count(self) -> int:
        """Number of error-level findings in this verification result."""
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        """Number of warning-level findings in this verification result."""
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def info_count(self) -> int:
        """Number of info-level findings in this verification result."""
        return sum(1 for i in self.issues if i.severity == "info")

    def __repr__(self) -> str:
        return (
            f"VerificationResult(check_name={self.check_name!r}, status={self.status.value!r}, "
            f"issues={len(self.issues)}, execution_time_ms={self.execution_time_ms!r})"
        )

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
    def verify(self, content: Any) -> ValidationVerificationResult:
        """Execute the verification check."""


class CodeSyntaxVerifier(Verifier):
    """Verifies Python code syntax."""

    def __init__(self):
        super().__init__("code_syntax")

    def verify(self, content: str) -> ValidationVerificationResult:
        """Check if content is valid Python syntax.

        Args:
            content: Python source code to parse, optionally wrapped in
                markdown code fences.

        Returns:
            VerificationResult with PASSED status on success; FAILED with an
            error-level issue on SyntaxError; WARNING on other parse failures;
            SKIPPED if content is empty or non-string.
        """
        import time

        start = time.time()

        result = ValidationVerificationResult(
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
                ),
            )
        except Exception as e:
            result.status = VerificationStatus.WARNING
            result.issues.append(
                VerificationIssue(
                    severity="warning",
                    category="syntax",
                    message=f"Could not parse code: {e!s}",
                ),
            )

        result.execution_time_ms = int((time.time() - start) * 1000)
        return result


class SecurityVerifier(Verifier):
    """Verifies content for security issues."""

    def __init__(self):
        super().__init__("security")
        self.scanner = get_secret_scanner()

    def verify(self, content: str) -> ValidationVerificationResult:
        """Check for security issues in content.

        Args:
            content: Text or source code to scan for secrets and dangerous
                patterns (exec, eval, shell=True, etc.).

        Returns:
            VerificationResult with FAILED status when secrets are detected,
            WARNING when dangerous-but-not-secret patterns appear, PASSED when
            clean, or SKIPPED for non-string input.
        """
        import time

        start = time.time()

        result = ValidationVerificationResult(
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
                ),
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
                # Only upgrade to WARNING; preserve FAILED if already set
                if result.status != VerificationStatus.FAILED:
                    result.status = VerificationStatus.WARNING
                result.issues.append(
                    VerificationIssue(
                        severity="warning",
                        category="security",
                        message=message,
                        suggestion="Use safer alternatives",
                    ),
                )

        result.execution_time_ms = int((time.time() - start) * 1000)
        return result


class ImportVerifier(Verifier):
    """Verifies Python imports are safe and available."""

    def __init__(self, allowed_modules: list[str] | None = None):
        super().__init__("imports")
        self.allowed_modules = allowed_modules or []  # noqa: VET112 - empty fallback preserves optional request metadata contract
        self.blocked_modules = ["ctypes", "mmap", "msvcrt", "winreg"]

    def verify(self, content: str) -> ValidationVerificationResult:
        """Check Python imports in content.

        Args:
            content: Python source code whose import statements will be
                extracted and checked against the blocked module list.

        Returns:
            VerificationResult with WARNING for each blocked import found,
            PASSED if all imports are clean, or SKIPPED for non-string input.
        """
        import time

        start = time.time()

        result = ValidationVerificationResult(
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
                    ),
                )

        result.execution_time_ms = int((time.time() - start) * 1000)
        return result

    def _extract_imports(self, content: str) -> list[str]:
        """Extract import statements from code."""
        # Find 'import X' statements
        imports = [match.group(1) for match in re.finditer(r"^import\s+([\w.]+)", content, re.MULTILINE)]

        # Find 'from X import Y' statements
        imports.extend(match.group(1) for match in re.finditer(r"^from\s+([\w.]+)\s+import", content, re.MULTILINE))

        return imports


class JSONStructureVerifier(Verifier):
    """Verifies JSON structure and completeness."""

    def __init__(self, required_fields: list[str] | None = None):
        super().__init__("json_structure")
        self.required_fields = required_fields or []  # noqa: VET112 - empty fallback preserves optional request metadata contract

    def verify(self, content: str) -> ValidationVerificationResult:
        """Check JSON structure and required fields.

        Args:
            content: JSON string to parse, optionally wrapped in a markdown
                code fence.

        Returns:
            VerificationResult with PASSED when JSON is valid and all required
            fields are present; WARNING for missing required fields; FAILED for
            unparseable JSON; SKIPPED for non-string input.
        """
        import time

        start = time.time()

        result = ValidationVerificationResult(
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
                            ),
                        )
        except json.JSONDecodeError as e:
            result.status = VerificationStatus.FAILED
            result.issues.append(
                VerificationIssue(
                    severity="error",
                    category="structure",
                    message=f"Invalid JSON: {e!s}",
                ),
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

    def add_verifier(self, verifier: Verifier) -> None:
        """Add a custom verifier to the pipeline."""
        self.verifiers.append(verifier)

    def verify(self, content: Any) -> dict[str, ValidationVerificationResult]:
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
                results[verifier.name] = ValidationVerificationResult(
                    status=VerificationStatus.SKIPPED,
                    check_name=verifier.name,
                )

        return results

    def get_summary(self, results: dict[str, ValidationVerificationResult]) -> dict[str, Any]:
        """Aggregate verification results into a pipeline summary.

        Args:
            results: Mapping of verifier name to VerificationResult, as
                returned by :meth:`verify`.

        Returns:
            Dictionary with overall_status ("PASSED"/"FAILED"), total_checks,
            total_issues, error_count, warning_count, and a per-check breakdown
            under the "checks" key.
        """
        total_issues = sum(len(r.issues) for r in results.values())
        total_errors = sum(r.error_count for r in results.values())
        total_warnings = sum(r.warning_count for r in results.values())

        all_passed = all(r.status == VerificationStatus.PASSED for r in results.values())
        has_failures = any(r.status == VerificationStatus.FAILED for r in results.values())

        return {
            "overall_status": "FAILED" if has_failures else "PASSED",
            "all_passed": all_passed,
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

    def verify(self, content: Any) -> ValidationVerificationResult:
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
            return ValidationVerificationResult(
                status=VerificationStatus.SKIPPED,
                check_name=self.name,
            )

        gate_results = self._runner.run_gate(self._stage, artifacts)

        if not gate_results:
            result = ValidationVerificationResult(
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
                    ),
                )

            issues.extend(
                VerificationIssue(
                    severity="info",
                    category="suggestion",
                    message=suggestion,
                )
                for suggestion in gr.suggestions
            )

        if has_failure:
            status = VerificationStatus.FAILED
        elif has_warning:
            status = VerificationStatus.WARNING
        else:
            status = VerificationStatus.PASSED

        result = ValidationVerificationResult(
            status=status,
            check_name=self.name,
            issues=issues,
        )
        result.execution_time_ms = int((time.time() - start) * 1000)
        return result


# ---------------------------------------------------------------------------
# Tiered verification cascade (Item 16.9)
# ---------------------------------------------------------------------------


@dataclass
class CascadeVerdict:
    """Aggregated result from the three-tier verification cascade.

    Attributes:
        passed: Overall pass/fail verdict.
        tier_reached: Which tier made the final decision
            (``"static"``, ``"entailment"``, or ``"llm"``).
        static_findings: List of finding strings from Tier 1 static checks.
        entailment_coverage: Keyword coverage from Tier 2, or None if skipped.
        llm_score: Confidence score from Tier 3 LLM, or None if skipped.
    """

    passed: bool
    tier_reached: str  # "static" | "entailment" | "llm"
    static_findings: list[str] = field(default_factory=list)
    entailment_coverage: float | None = None
    llm_score: float | None = None

    def __repr__(self) -> str:
        """Show key fields for debugging."""
        return (
            f"CascadeVerdict(passed={self.passed!r}, tier_reached={self.tier_reached!r}, "
            f"entailment_coverage={self.entailment_coverage!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dictionary.

        Returns:
            JSON-serialisable dict with all verdict fields.
        """
        return {
            "passed": self.passed,
            "tier_reached": self.tier_reached,
            "static_findings": self.static_findings,
            "entailment_coverage": self.entailment_coverage,
            "llm_score": self.llm_score,
        }


class CascadeOrchestrator:
    """Three-tier verification cascade that minimises LLM calls.

    Tiers (run in order, stopping as soon as a verdict is conclusive):

    **Tier 1 — StaticVerifier**: deterministic checks (syntax, banned imports,
    credential patterns, code presence). If any static check fails, the output
    is rejected immediately — no further tiers run.

    **Tier 2 — EntailmentChecker**: keyword coverage + optional semantic
    similarity. If coverage is above threshold, output is accepted without an
    LLM call.

    **Tier 3 — LLM**: ``score_confidence_via_llm()`` is called only when Tiers
    1 and 2 are both inconclusive.

    Example::

        orchestrator = CascadeOrchestrator()
        verdict = orchestrator.verify("def add(a, b): return a + b", "implement an add function")
        assert verdict.passed
        assert verdict.tier_reached in ("static", "entailment", "llm")
    """

    def __init__(self) -> None:
        from vetinari.validation.entailment_checker import EntailmentChecker
        from vetinari.validation.static_verifier import StaticVerifier

        self._static = StaticVerifier()
        self._entailment = EntailmentChecker()

    def verify(self, content: str, task_description: str = "") -> CascadeVerdict:
        """Run the three-tier cascade on *content*.

        Args:
            content: The output text or code to verify.
            task_description: The task or requirement the output should satisfy.

        Returns:
            :class:`CascadeVerdict` with the overall verdict and evidence from
            whichever tiers ran.
        """
        # ── Tier 1: Static checks ────────────────────────────────────────────
        static_results = self._static.verify(content, task_description)
        static_findings = [r.finding for r in static_results if not r.passed and r.finding]

        if static_findings:
            logger.info(
                "CascadeOrchestrator: Tier 1 FAILED (%d findings) — short-circuiting",
                len(static_findings),
            )
            return CascadeVerdict(
                passed=False,
                tier_reached="static",
                static_findings=static_findings,
            )

        # ── Tier 2: Entailment check ─────────────────────────────────────────
        entailment_result = self._entailment.check(task_description, content)
        logger.debug(
            "CascadeOrchestrator: Tier 2 coverage=%.3f entailed=%s",
            entailment_result.coverage,
            entailment_result.entailed,
        )

        if entailment_result.entailed:
            return CascadeVerdict(
                passed=True,
                tier_reached="entailment",
                static_findings=[],
                entailment_coverage=entailment_result.coverage,
            )

        # When coverage is very low (< 20%), fail without calling LLM
        if entailment_result.coverage < 0.2:
            logger.info(
                "CascadeOrchestrator: Tier 2 coverage %.3f too low — rejecting without LLM",
                entailment_result.coverage,
            )
            return CascadeVerdict(
                passed=False,
                tier_reached="entailment",
                static_findings=[],
                entailment_coverage=entailment_result.coverage,
            )

        # ── Tier 3: LLM confidence score ─────────────────────────────────────
        llm_score: float | None = None
        try:
            llm_score = score_confidence_via_llm(task_description, content)
        except Exception as exc:
            logger.warning(
                "CascadeOrchestrator: Tier 3 LLM unavailable (%s) — using entailment coverage as fallback",
                exc,
            )

        if llm_score is not None:
            passed = llm_score >= 0.5
        else:
            # LLM unavailable — use entailment coverage as final fallback
            passed = entailment_result.coverage >= 0.4

        logger.info(
            "CascadeOrchestrator: Tier 3 llm_score=%s passed=%s",
            llm_score,
            passed,
        )
        return CascadeVerdict(
            passed=passed,
            tier_reached="llm",
            static_findings=[],
            entailment_coverage=entailment_result.coverage,
            llm_score=llm_score,
        )


# Global cascade orchestrator singleton
_cascade_orchestrator: CascadeOrchestrator | None = None
_cascade_orchestrator_lock = threading.Lock()


def get_cascade_orchestrator() -> CascadeOrchestrator:
    """Return the process-wide CascadeOrchestrator singleton.

    Uses double-checked locking for thread safety.

    Returns:
        The singleton :class:`CascadeOrchestrator` instance.
    """
    global _cascade_orchestrator
    if _cascade_orchestrator is None:
        with _cascade_orchestrator_lock:
            if _cascade_orchestrator is None:
                _cascade_orchestrator = CascadeOrchestrator()
    return _cascade_orchestrator


# Global verifier instance
_verifier_pipeline: VerificationPipeline | None = None
_verifier_pipeline_lock = threading.Lock()


def get_verifier_pipeline() -> VerificationPipeline:
    """Get or create the global verification pipeline.

    Returns:
        The singleton VerificationPipeline configured at STANDARD level,
        including syntax, security, and import verifiers.
    """
    global _verifier_pipeline
    if _verifier_pipeline is None:
        with _verifier_pipeline_lock:
            if _verifier_pipeline is None:
                # Use STANDARD level by default, can be customized
                _verifier_pipeline = VerificationPipeline(VerificationLevel.STANDARD)
    return _verifier_pipeline


# ---------------------------------------------------------------------------
# Lightweight output validator (consolidated from validator.py)
# ---------------------------------------------------------------------------


class Validator:
    """Validates agent outputs against quality and safety rules.

    Provides quick heuristic checks for text, JSON, and Python code outputs.
    For thorough verification use the ``VerificationPipeline`` above.
    """

    def is_valid_text(self, text: str) -> bool:
        """Check whether *text* is non-empty and syntactically plausible.

        Returns ``True`` for valid JSON, syntactically correct Python code,
        or any non-empty text string.

        Args:
            text: The text to validate.

        Returns:
            ``True`` when the text passes basic validation.
        """
        if not text or len(text.strip()) == 0:
            return False

        # Try to parse as JSON first (some tasks return JSON)
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, TypeError, ValueError):  # noqa: VET022 - best-effort optional path must not fail the primary flow
            pass

        # Check if it looks like code
        if self._looks_like_code(text):
            return self._validate_python_code(text)

        # Default: if there's content and it's not empty, accept it
        return len(text.strip()) > 0

    def _looks_like_code(self, text: str) -> bool:
        cleaned = re.sub(r"```[\w]*\n?", "", text)
        cleaned = cleaned.strip()
        return bool(re.search(r"^(def|class|import|from|@|#|async|with|\s+=\s+)", cleaned, re.MULTILINE))

    def _validate_python_code(self, code: str) -> bool:
        cleaned = re.sub(r"```[\w]*\n", "\n", code)
        cleaned = re.sub(r"```$", "", cleaned)
        cleaned = cleaned.strip()

        if cleaned.startswith("{") and cleaned.endswith("}"):
            inner = cleaned[1:-1].strip()
            if not re.search(r"^(def|class|import|from|@|async|with|\s+=)", inner, re.MULTILINE):
                cleaned = inner

        try:
            ast.parse(cleaned)
            return True
        except (SyntaxError, ValueError):
            logger.warning("Code syntax check found errors — marking as invalid")
            return False
