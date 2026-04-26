"""Inspector Skill Tool.

==============================
Skill tool for the INSPECTOR agent — independent quality gate.

Covers 4 modes:
  - code_review: 5-pass review (correctness, style, security, perf, maintainability)
  - security_audit: OWASP Top 10, CWE mapping, secrets detection
  - test_generation: Coverage analysis, gap-filling test generation
  - simplification: Dead code detection, complexity reduction

The Inspector is the final gate in the factory pipeline. Its decisions
cannot be overridden by any other agent — only humans can bypass the gate.

Standards enforced (from skill_registry):
  - STD-INS-001: 5-dimension review coverage
  - STD-INS-002: OWASP Top 10 + CWE mapping
  - STD-INS-003: Credential/secrets scanning
  - STD-INS-004: Happy/edge/error test coverage
  - STD-INS-005: Severity + actionable descriptions
  - STD-INS-006: Read-only constraint
  - STD-INS-007: Objective gate criteria
  - STD-INS-008: Human-only gate override
"""

from __future__ import annotations

import dataclasses
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from vetinari.agents.contracts import LLMJudgment, OutcomeSignal, Provenance
from vetinari.execution_context import ToolPermission
from vetinari.tool_interface import (
    Tool,
    ToolCategory,
    ToolMetadata,
    ToolParameter,
    ToolResult,
)
from vetinari.types import AgentType, EvidenceBasis, ExecutionMode

logger = logging.getLogger(__name__)

# ── Module-level compiled patterns ───────────────────────────────────────────
_FUNC_DEF_RE = re.compile(r"def\s+(\w+)\s*\(")


class InspectorMode(str, Enum):
    """Modes of the Inspector skill tool."""

    CODE_REVIEW = "code_review"
    SECURITY_AUDIT = "security_audit"
    TEST_GENERATION = "test_generation"
    SIMPLIFICATION = "simplification"


@dataclass
class ReviewIssue:
    """A single issue found during review."""

    severity: str  # critical, high, medium, low, info
    description: str
    file: str = ""
    line: int = 0
    category: str = ""
    cwe: str = ""
    owasp: str = ""
    suggestion: str = ""

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"ReviewIssue(severity={self.severity!r}, category={self.category!r}, file={self.file!r})"


@dataclass
class InspectorResult:
    """Result from Inspector skill execution."""

    passed: bool = True
    issues: list[ReviewIssue] = field(default_factory=list)
    grade: str = "A"  # A, B, C, D, F
    score: float = 1.0  # 0.0 to 1.0
    suggestions: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    self_check_passed: bool | None = None

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"InspectorResult(passed={self.passed!r}, grade={self.grade!r}, score={self.score!r})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary with review result fields.
        """
        result: dict[str, Any] = {
            "passed": self.passed,
            "issues": [dataclasses.asdict(i) for i in self.issues],
            "grade": self.grade,
            "score": self.score,
            "suggestions": self.suggestions,
            "metrics": self.metrics,
        }
        if self.self_check_passed is not None:
            result["self_check_passed"] = self.self_check_passed
        return result


class InspectorSkillTool(Tool):
    """Skill tool for the Inspector agent — independent quality gate.

    The Inspector is the last stage of the factory pipeline. It performs
    read-only review of Worker output and issues pass/fail gate decisions
    that cannot be overridden by any other agent. Only humans can bypass
    the Inspector's gate.

    QualitySkillTool is used as a supplementary checker: Inspector heuristics
    run first and remain the primary gate; Quality findings are merged in
    afterwards as additive signal only.
    """

    def __init__(self):
        self._quality_tool: Any = None  # lazily populated by _get_quality_tool()
        super().__init__(
            metadata=ToolMetadata(
                name="inspector",
                description=(
                    "Independent quality gate — code review, security audit, test generation, and code simplification"
                ),
                category=ToolCategory.SEARCH_ANALYSIS,
                version="2.0.0",
                parameters=[
                    ToolParameter(
                        name="code",
                        type=str,
                        description="Code or content to review",
                        required=True,
                    ),
                    ToolParameter(
                        name="mode",
                        type=str,
                        description="Review mode",
                        required=False,
                        default="code_review",
                        allowed_values=[m.value for m in InspectorMode],
                    ),
                    ToolParameter(
                        name="context",
                        type=dict,
                        description="Review context (PR description, self_check results)",
                        required=False,
                    ),
                    ToolParameter(
                        name="focus_areas",
                        type=list,
                        description="Specific areas to focus the review on",
                        required=False,
                    ),
                    ToolParameter(
                        name="thinking_mode",
                        type=str,
                        description="Thinking budget tier",
                        required=False,
                        allowed_values=["low", "medium", "high", "xhigh"],
                    ),
                ],
                required_permissions=[
                    ToolPermission.FILE_READ,
                    ToolPermission.MODEL_INFERENCE,
                ],
                allowed_modes=[ExecutionMode.EXECUTION],
                tags=["quality", "security", "review", "gate"],
            ),
        )

    def execute(self, **kwargs) -> ToolResult:
        """Execute the Inspector skill in the specified mode.

        Args:
            **kwargs: Must include 'code'; optionally 'mode', 'context', 'focus_areas'.

        Returns:
            ToolResult containing the Inspector's review result.
        """
        code = kwargs.get("code", "")
        mode_str = kwargs.get("mode", "code_review")
        context = kwargs.get("context", {})
        focus_areas = kwargs.get("focus_areas", [])

        try:
            mode = InspectorMode(mode_str)
        except ValueError:
            logger.warning("Invalid InspectorMode %r in tool call — returning error to caller", mode_str)
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown mode: {mode_str}. Valid modes: {[m.value for m in InspectorMode]}",
            )

        logger.info("Inspector executing mode=%s code_length=%d", mode.value, len(code))

        # Check self_check_passed from context (Phase 5.24 integration)
        self_check_passed = context.get("self_check_passed")
        self_check_issues = context.get("self_check_issues", [])
        if self_check_passed is False:
            logger.info(
                "Inspector: self_check failed with %d issues — applying deeper review",
                len(self_check_issues),
            )

        try:
            result = self._execute_mode(mode, code, context, focus_areas)
            # Incorporate self_check results into gate decision
            if self_check_passed is not None:
                result.self_check_passed = self_check_passed
                if not self_check_passed and result.passed:
                    # Self-check failed but review passed — add advisory
                    result.suggestions.append(
                        "Agent self-check flagged issues that were not caught in review: "
                        + "; ".join(self_check_issues[:3]),
                    )

            logger.info(
                "Inspector completed mode=%s passed=%s grade=%s score=%.2f",
                mode.value,
                result.passed,
                result.grade,
                result.score,
            )

            # Build OutcomeSignal from heuristic result so callers get a
            # provenance-bearing verdict rather than a bare pass/fail dict.
            outcome = _inspector_result_to_signal(result, mode.value)

            return ToolResult(
                success=True,
                output=result.to_dict(),
                metadata={
                    "mode": mode.value,
                    "agent": AgentType.INSPECTOR.value,
                    "passed": result.passed,
                    "grade": result.grade,
                    "outcome_signal": {
                        "passed": outcome.passed,
                        "score": outcome.score,
                        "basis": outcome.basis.value,
                        "issues": list(outcome.issues),
                    },
                },
            )
        except Exception as exc:
            logger.error("Inspector mode=%s failed: %s", mode.value, exc)
            return ToolResult(success=False, output=None, error=str(exc))

    def _execute_mode(
        self,
        mode: InspectorMode,
        code: str,
        context: dict[str, Any],
        focus_areas: list[str],
    ) -> InspectorResult:
        """Route to the appropriate mode handler.

        Args:
            mode: The review mode.
            code: Code or content to review.
            context: Review context.
            focus_areas: Specific areas to focus on.

        Returns:
            InspectorResult from the mode handler.
        """
        handler = {
            InspectorMode.CODE_REVIEW: self._code_review,
            InspectorMode.SECURITY_AUDIT: self._security_audit,
            InspectorMode.TEST_GENERATION: self._test_generation,
            InspectorMode.SIMPLIFICATION: self._simplification,
        }[mode]
        return handler(code, context, focus_areas)

    def _code_review(self, code: str, context: dict[str, Any], focus_areas: list[str]) -> InspectorResult:
        """5-pass code review: correctness, style, security, performance, maintainability.

        Delegates to the InspectorAgent for LLM-powered analysis when available,
        falls back to heuristic scanning.
        """
        issues: list[ReviewIssue] = []

        # Heuristic pass: check for common issues
        lines = code.split("\n") if code else []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if "print(" in stripped and not stripped.startswith("#"):
                issues.append(
                    ReviewIssue(
                        severity="medium",
                        description="print statement found in production code — use logging module",
                        line=i,
                        category="style",
                        suggestion="Replace with logger.info() or logger.debug()",
                    ),
                )
            if "except:" in stripped and "except Exception" not in stripped:
                issues.append(
                    ReviewIssue(
                        severity="high",
                        description="Bare except clause — catches SystemExit and KeyboardInterrupt",
                        line=i,
                        category="correctness",
                        suggestion="Use 'except Exception as e:' with proper error handling",
                    ),
                )
            if "TODO" in stripped and "#" in stripped:
                issues.append(
                    ReviewIssue(
                        severity="low",
                        description="TODO comment found in production code",
                        line=i,
                        category="completeness",
                        suggestion="Resolve or convert to tracked issue",
                    ),
                )

        # Deny-pattern pass: check for dangerous code patterns from standards config.
        try:
            from vetinari.config.standards_loader import get_standards_loader

            deny_findings = get_standards_loader().evaluate_deny_patterns(code)
            issues.extend(
                ReviewIssue(
                    severity=finding.get("severity", "high"),
                    description=finding.get("description", f"Deny pattern matched: {finding.get('pattern', '')}"),
                    category="security",
                    suggestion=f"Remove or replace the matched pattern: {finding.get('match', '')}",
                )
                for finding in deny_findings
            )
        except Exception as exc:
            logger.warning("Inspector: deny pattern evaluation failed: %s", exc)

        # Supplementary pass: delegate to QualitySkillTool for richer detection.
        heuristic_count = len(issues)
        supplementary_count = 0
        try:
            quality_result = self._get_quality_tool().execute(mode="code_review", code=code, thinking_mode="medium")
            issues = self._merge_quality_issues(issues, quality_result, dedup_field="description")
            supplementary_count = len(issues) - heuristic_count
        except Exception as exc:
            logger.error(
                "Inspector: QualitySkillTool code_review failed — using heuristic results only: %s",
                exc,
            )

        # Cascade verification pass — static + entailment before LLM scoring
        task_description = context.get("task_description", "")
        try:
            from vetinari.validation.verification import get_cascade_orchestrator

            verdict = get_cascade_orchestrator().verify(code, task_description)
            if not verdict.passed and verdict.tier_reached == "static":
                for finding in verdict.static_findings:
                    issues.append(
                        ReviewIssue(
                            severity="high",
                            description=finding,
                            category="static_verification",
                            suggestion="Fix the static verification failure before resubmitting",
                        )
                    )
                logger.info(
                    "Inspector: cascade Tier 1 added %d static finding(s)",
                    len(verdict.static_findings),
                )
        except Exception as exc:
            logger.warning("Inspector: cascade verification unavailable (%s) — skipping cascade tier", exc)

        # Score and grade — zero-tolerance: any issue fails the gate
        critical_count = sum(1 for i in issues if i.severity == "critical")
        high_count = sum(1 for i in issues if i.severity == "high")
        medium_count = sum(1 for i in issues if i.severity == "medium")
        low_count = sum(1 for i in issues if i.severity == "low")

        score = max(
            0.0,
            1.0 - (critical_count * 0.3 + high_count * 0.15 + medium_count * 0.05 + low_count * 0.02),
        )
        grade = self._score_to_grade(score)
        passed = len(issues) == 0

        return InspectorResult(
            passed=passed,
            issues=issues,
            grade=grade,
            score=round(score, 3),
            metrics={
                "total_issues": len(issues),
                "critical": critical_count,
                "high": high_count,
                "medium": medium_count,
                "low": low_count,
                "lines_reviewed": len(lines),
                "supplementary_issues": supplementary_count,
            },
        )

    def _security_audit(self, code: str, context: dict[str, Any], focus_areas: list[str]) -> InspectorResult:
        """Security audit with OWASP Top 10 and CWE mapping.

        Scans for hardcoded credentials, injection vulnerabilities, insecure
        deserialization, and other security patterns.
        """
        issues: list[ReviewIssue] = []

        # Pattern-based security scan
        security_patterns = [
            ("password\\s*=\\s*[\"']", "Hardcoded password", "CWE-798", "A07:2021"),
            ("api_key\\s*=\\s*[\"']", "Hardcoded API key", "CWE-798", "A07:2021"),
            ("secret\\s*=\\s*[\"']", "Hardcoded secret", "CWE-798", "A07:2021"),
            ("shell\\s*=\\s*True", "Shell injection risk", "CWE-78", "A03:2021"),
            ("yaml\\.load\\(", "Unsafe YAML loading", "CWE-502", "A08:2021"),
            ("eval\\(", "Code injection via eval()", "CWE-95", "A03:2021"),
            ("pickle\\.loads?\\(", "Insecure deserialization", "CWE-502", "A08:2021"),
            ("verify\\s*=\\s*False", "SSL verification disabled", "CWE-295", "A07:2021"),
        ]

        import re

        for pattern, desc, cwe, owasp in security_patterns:
            for i, line in enumerate(code.split("\n"), 1):
                if re.search(pattern, line) and not line.strip().startswith("#"):
                    issues.append(
                        ReviewIssue(
                            severity="high",
                            description=desc,
                            line=i,
                            category="security",
                            cwe=cwe,
                            owasp=owasp,
                            suggestion=f"Review and remediate: {desc}",
                        ),
                    )

        # Supplementary pass: delegate to QualitySkillTool for broader security detection.
        heuristic_security_count = len(issues)
        supplementary_security_count = 0
        try:
            quality_result = self._get_quality_tool().execute(mode="security_audit", code=code, thinking_mode="medium")
            issues = self._merge_quality_issues(issues, quality_result, dedup_field="cwe")
            supplementary_security_count = len(issues) - heuristic_security_count
        except Exception as exc:
            logger.error(
                "Inspector: QualitySkillTool security_audit failed — using heuristic results only: %s",
                exc,
            )

        score = max(0.0, 1.0 - len(issues) * 0.15)
        passed = len(issues) == 0

        return InspectorResult(
            passed=passed,
            issues=issues,
            grade=self._score_to_grade(score),
            score=round(score, 3),
            metrics={
                "security_issues": len(issues),
                "patterns_checked": len(security_patterns),
                "supplementary_security_issues": supplementary_security_count,
            },
        )

    def _test_generation(self, code: str, context: dict[str, Any], focus_areas: list[str]) -> InspectorResult:
        """Identify untested paths and generate gap-filling test suggestions.

        Analyzes code to find functions without corresponding tests and
        suggests test cases for happy path, edge cases, and error paths.
        """
        # Find functions that might need tests
        functions = _FUNC_DEF_RE.findall(code)
        public_functions = [f for f in functions if not f.startswith("_")]

        suggestions: list[str] = []
        for func in public_functions:
            suggestions.extend((f"Add test for {func}() — happy path", f"Add test for {func}() — error/edge cases"))

        has_gaps = len(suggestions) > 0
        if not has_gaps:
            grade = "A"
            score = 0.95
        elif len(suggestions) <= 4:
            grade = "B"
            score = 0.75
        elif len(suggestions) <= 8:
            grade = "C"
            score = 0.55
        else:
            grade = "D"
            score = 0.35

        return InspectorResult(
            passed=not has_gaps,
            grade=grade,
            score=score,
            suggestions=suggestions,
            metrics={
                "total_functions": len(functions),
                "public_functions": len(public_functions),
                "test_suggestions": len(suggestions),
            },
        )

    def _simplification(self, code: str, context: dict[str, Any], focus_areas: list[str]) -> InspectorResult:
        """Identify dead code, over-abstraction, and complexity reduction opportunities.

        Analyzes code for YAGNI violations, unnecessary abstractions, and
        overly complex logic that could be simplified.
        """
        issues: list[ReviewIssue] = []
        lines = code.split("\n") if code else []

        # Check for common complexity indicators
        nesting_depth = 0
        max_nesting = 0
        for line in lines:
            stripped = line.strip()
            if stripped.endswith(":") and any(
                stripped.startswith(kw) for kw in ["if ", "for ", "while ", "try:", "with "]
            ):
                nesting_depth += 1
                max_nesting = max(max_nesting, nesting_depth)
            elif stripped in ("", "pass") or (not stripped.startswith(" " * (nesting_depth * 4))):
                nesting_depth = max(0, nesting_depth - 1)

        if max_nesting > 4:
            issues.append(
                ReviewIssue(
                    severity="medium",
                    description=f"Deep nesting detected (depth {max_nesting}) — consider extracting helper functions",
                    category="complexity",
                    suggestion="Break complex nested logic into smaller, named functions",
                ),
            )

        # Check for overly long functions
        current_func = None
        func_start = 0
        for i, line in enumerate(lines, 1):
            if line.strip().startswith("def "):
                if current_func and i - func_start > 50:
                    issues.append(
                        ReviewIssue(
                            severity="low",
                            description=f"Function '{current_func}' is {i - func_start} lines — consider splitting",
                            line=func_start,
                            category="complexity",
                        ),
                    )
                current_func = line.strip().split("(")[0].replace("def ", "")
                func_start = i

        score = max(0.0, 1.0 - len(issues) * 0.1)

        return InspectorResult(
            passed=len(issues) == 0,
            issues=issues,
            grade=self._score_to_grade(score),
            score=round(score, 3),
            metrics={
                "max_nesting_depth": max_nesting,
                "total_lines": len(lines),
                "simplification_opportunities": len(issues),
            },
        )

    def _get_quality_tool(self) -> Any:
        """Lazily create and cache a QualitySkillTool instance.

        The import is deferred to avoid a circular import at module load time
        and to keep QualitySkillTool as an optional dependency.

        Returns:
            A cached QualitySkillTool instance.
        """
        if self._quality_tool is None:
            from vetinari.skills.quality_skill import QualitySkillTool

            self._quality_tool = QualitySkillTool()
        return self._quality_tool

    def _merge_quality_issues(
        self,
        existing: list[ReviewIssue],
        quality_result: Any,
        dedup_field: str = "description",
    ) -> list[ReviewIssue]:
        """Merge QualitySkillTool findings into Inspector's issue list.

        Extracts QualityIssue dicts from a ToolResult, converts them to
        ReviewIssue objects, and appends only those whose deduplication key
        does not already appear in the existing list.

        Args:
            existing: Inspector's current list of ReviewIssue objects.
            quality_result: ToolResult returned by QualitySkillTool.execute().
            dedup_field: Field name used for deduplication — ``"description"``
                for code-review merges, ``"cwe"`` for security-audit merges.

        Returns:
            New list containing ``existing`` issues followed by any
            non-duplicate issues from the quality result.
        """
        if not (quality_result and quality_result.success and quality_result.output):
            return existing

        raw_issues: list[dict[str, Any]] = quality_result.output.get("issues", [])
        if not raw_issues:
            return existing

        # Build a set of existing dedup keys for O(1) look-up.
        if dedup_field == "description":
            existing_keys: set[str] = {i.description.lower() for i in existing}
        else:  # "cwe"
            existing_keys = {i.cwe.lower() for i in existing if i.cwe}

        merged = list(existing)
        for raw in raw_issues:
            # Map QualityIssue fields to ReviewIssue fields.
            description: str = raw.get("description") or raw.get("title", "")
            cwe: str = raw.get("cwe_id", "")
            owasp: str = raw.get("owasp_category", "")

            if dedup_field == "description":
                key = description.lower()
            else:
                key = cwe.lower() if cwe else description.lower()

            if key in existing_keys:
                continue

            existing_keys.add(key)
            merged.append(
                ReviewIssue(
                    severity=raw.get("severity", "low"),
                    description=description,
                    category=raw.get("category", ""),
                    cwe=cwe,
                    owasp=owasp,
                    suggestion=raw.get("suggestion", "") or "",
                ),
            )

        return merged

    @staticmethod
    def _score_to_grade(score: float) -> str:
        """Convert a numeric score to a letter grade.

        Args:
            score: Score between 0.0 and 1.0.

        Returns:
            Letter grade (A, B, C, D, or F).
        """
        if score >= 0.9:
            return "A"
        if score >= 0.8:
            return "B"
        if score >= 0.7:
            return "C"
        if score >= 0.6:
            return "D"
        return "F"


# -- OutcomeSignal helpers ---------------------------------------------------


def _inspector_result_to_signal(result: InspectorResult, mode: str) -> OutcomeSignal:
    """Convert an InspectorResult to a heuristic-based OutcomeSignal.

    Produces a ``LLM_JUDGMENT`` basis signal because the Inspector's heuristic
    scoring is a model-adjacent quality assessment, not a deterministic tool
    invocation.  Callers that have separate tool signals should merge them via
    ``aggregate_outcome_signals()``.

    Args:
        result: The InspectorResult produced by an inspection mode handler.
        mode: The InspectorMode value string for provenance labelling.

    Returns:
        OutcomeSignal with basis=LLM_JUDGMENT and populated provenance.
    """
    issues: tuple[str, ...] = tuple(
        f"{i.severity.upper()}: {i.description}" + (f" (line {i.line})" if i.line else "") for i in result.issues
    )
    suggestions: tuple[str, ...] = tuple(result.suggestions)

    judgment = LLMJudgment(
        model_id="inspector_heuristic",
        summary=f"Inspector {mode} grade={result.grade} score={result.score:.3f}",
        score=result.score,
        reasoning="; ".join(issues[:5]) if issues else "no issues",
    )

    return OutcomeSignal(
        passed=result.passed,
        score=result.score,
        basis=EvidenceBasis.LLM_JUDGMENT,
        llm_judgment=judgment,
        issues=issues,
        suggestions=suggestions,
        provenance=Provenance(
            source=f"vetinari.skills.inspector_skill.{mode}",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            model_id="inspector_heuristic",
        ),
    )
