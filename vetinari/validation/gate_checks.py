"""Quality gate check implementations for QualityGateRunner.

Contains the five gate check methods (quality, security, coverage,
architecture, prevention) extracted from ``quality_gates.py`` to keep
each module under the 550-line ceiling.

``QualityGateRunner`` in ``quality_gates.py`` inherits ``_GateCheckMixin``
to get all check implementations without duplication.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from vetinari.validation.gate_types import GateCheckResult, GateResult, QualityGateConfig, VerificationMode

logger = logging.getLogger(__name__)


class _GateCheckMixin:
    """Mixin providing all gate check implementations for QualityGateRunner.

    This class is not meant to be instantiated directly; it is mixed into
    ``QualityGateRunner`` to keep the check implementations in a separate
    module from the runner's orchestration logic.
    """

    # ------------------------------------------------------------------
    # Public check methods (called by _run_single_gate dispatch table)
    # ------------------------------------------------------------------

    def check_quality(self, artifacts: dict[str, Any], config: QualityGateConfig) -> GateCheckResult:
        """Run quality verification (style, complexity, best practices).

        Inspects ``artifacts["code"]`` for common quality issues using
        lightweight heuristic analysis. Falls back gracefully when the
        code key is absent.

        Args:
            artifacts: Dict containing at least an optional ``"code"`` key
                with the source code string to inspect.
            config: The quality gate configuration for this check.

        Returns:
            GateCheckResult reflecting quality heuristics outcome.
        """
        code = artifacts.get("code", "")
        issues: list[dict[str, Any]] = []
        suggestions: list[str] = []
        score = 1.0

        if not code:
            return GateCheckResult(
                gate_name=config.name,
                mode=config.mode,
                result=GateResult.WARNING,
                score=0.5,
                issues=[{"severity": "warning", "message": "No code artifacts to check"}],
                suggestions=["Provide code artifacts for quality analysis"],
            )

        # 1. Function length (rough proxy for complexity)
        long_functions = self._check_long_functions(code)
        if long_functions:
            penalty = min(0.3, len(long_functions) * 0.1)
            score -= penalty
            issues.extend(
                {
                    "severity": "warning",
                    "category": "complexity",
                    "message": f"Function '{fn}' appears to be overly long",
                }
                for fn in long_functions
            )
            suggestions.append("Break large functions into smaller, focused helpers")

        # 2. Missing docstrings
        missing_docs = self._check_missing_docstrings(code)
        if missing_docs:
            penalty = min(0.2, len(missing_docs) * 0.05)
            score -= penalty
            issues.extend(
                {
                    "severity": "info",
                    "category": "documentation",
                    "message": f"Function '{fn}' is missing a docstring",
                }
                for fn in missing_docs
            )
            suggestions.append("Add docstrings to all public functions")

        # 3. Bare except clauses
        bare_excepts = len(re.findall(r"except\s*:", code))
        if bare_excepts:
            score -= min(0.2, bare_excepts * 0.1)
            issues.append(
                {
                    "severity": "warning",
                    "category": "best_practices",
                    "message": f"Found {bare_excepts} bare except clause(s)",
                },
            )
            suggestions.append("Catch specific exception types instead of bare except")

        # 4. TODO/FIXME/HACK markers
        markers = len(re.findall(r"#\s*(TODO|FIXME|HACK|XXX)\b", code, re.IGNORECASE))
        if markers:
            score -= min(0.1, markers * 0.02)
            issues.append(
                {
                    "severity": "info",
                    "category": "maintenance",
                    "message": f"Found {markers} TODO/FIXME/HACK marker(s)",
                },
            )

        score = max(0.0, min(1.0, score))
        result_enum = self._score_to_result(score, config.min_score)

        return GateCheckResult(
            gate_name=config.name,
            mode=config.mode,
            result=result_enum,
            score=round(score, 3),
            issues=issues,
            suggestions=suggestions,
        )

    def check_security(self, artifacts: dict[str, Any], config: QualityGateConfig) -> GateCheckResult:
        """Run security verification.

        Checks ``artifacts["code"]`` for dangerous patterns, potential
        secrets, and unsafe practices.

        Args:
            artifacts: Dict containing at least an optional ``"code"`` key.
            config: The quality gate configuration for this check.

        Returns:
            GateCheckResult reflecting the security scan outcome.
        """
        code = artifacts.get("code", "")
        issues: list[dict[str, Any]] = []
        suggestions: list[str] = []
        score = 1.0

        if not code:
            return GateCheckResult(
                gate_name=config.name,
                mode=config.mode,
                result=GateResult.WARNING,
                score=0.5,
                issues=[{"severity": "warning", "message": "No code artifacts for security check"}],
                suggestions=["Provide code artifacts for security analysis"],
            )

        dangerous_patterns = [
            (r"eval\s*\(", "eval() allows arbitrary code execution", "critical"),
            (r"exec\s*\(", "exec() allows arbitrary code execution", "critical"),
            (r"__import__\s*\(", "Dynamic __import__() may be unsafe", "high"),
            (r"os\.system\s*\(", "os.system() is vulnerable to shell injection", "high"),
            (r"subprocess.*shell\s*=\s*True", "subprocess with shell=True is dangerous", "high"),
            (r"pickle\.loads?\s*\(", "pickle deserialization can execute arbitrary code", "high"),
            (r"yaml\.load\s*\((?!.*Loader)", "yaml.load without Loader is unsafe", "medium"),
            (r"input\s*\(", "input() in production code may be unintended", "low"),
        ]

        severity_penalties = {"critical": 0.3, "high": 0.2, "medium": 0.1, "low": 0.05}

        for pattern, message, severity in dangerous_patterns:
            matches = re.findall(pattern, code)
            if matches:
                penalty = severity_penalties.get(severity, 0.1)
                score -= penalty
                issues.append(
                    {
                        "severity": severity,
                        "category": "security",
                        "message": message,
                        "count": len(matches),
                    },
                )
                suggestions.append(f"Review and mitigate: {message}")

        # Check for potential hardcoded secrets
        secret_patterns = [
            (r'(?:password|passwd|pwd)\s*=\s*["\'][^"\']+["\']', "Possible hardcoded password"),
            (r'(?:api_key|apikey|api_secret)\s*=\s*["\'][^"\']+["\']', "Possible hardcoded API key"),
            (r'(?:secret|token)\s*=\s*["\'][A-Za-z0-9+/=]{20,}["\']', "Possible hardcoded secret/token"),
        ]

        for pattern, message in secret_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                score -= 0.25
                issues.append(
                    {
                        "severity": "critical",
                        "category": "secrets",
                        "message": message,
                    },
                )
                suggestions.append("Move secrets to environment variables or a secrets manager")

        score = max(0.0, min(1.0, score))
        result_enum = self._score_to_result(score, config.min_score)

        return GateCheckResult(
            gate_name=config.name,
            mode=config.mode,
            result=result_enum,
            score=round(score, 3),
            issues=issues,
            suggestions=suggestions,
        )

    def check_coverage(self, artifacts: dict[str, Any], config: QualityGateConfig) -> GateCheckResult:
        """Run coverage verification.

        Checks ``artifacts["tests"]`` for test existence and
        ``artifacts["coverage_percent"]`` for coverage threshold.
        Also checks ``artifacts["code"]`` for testable functions
        without corresponding tests.

        Args:
            artifacts: Dict optionally containing ``"tests"``, ``"code"``,
                and ``"coverage_percent"`` keys.
            config: The quality gate configuration for this check.

        Returns:
            GateCheckResult reflecting the coverage outcome.
        """
        tests = artifacts.get("tests", "")
        code = artifacts.get("code", "")
        coverage_pct = artifacts.get("coverage_percent")
        issues: list[dict[str, Any]] = []
        suggestions: list[str] = []
        score = 1.0

        # Check that tests exist
        if not tests:
            score -= 0.4
            issues.append(
                {
                    "severity": "error",
                    "category": "coverage",
                    "message": "No test artifacts provided",
                },
            )
            suggestions.append("Write tests for the implemented code")

        # Check coverage percentage if provided
        if coverage_pct is not None:
            try:
                cov = float(coverage_pct)
                if cov < 50:
                    score -= 0.3
                    issues.append(
                        {
                            "severity": "error",
                            "category": "coverage",
                            "message": f"Test coverage is {cov}%, below 50% minimum",
                        },
                    )
                elif cov < 70:
                    score -= 0.15
                    issues.append(
                        {
                            "severity": "warning",
                            "category": "coverage",
                            "message": f"Test coverage is {cov}%, below 70% target",
                        },
                    )
                elif cov < 80:
                    score -= 0.05
                    issues.append(
                        {
                            "severity": "info",
                            "category": "coverage",
                            "message": f"Test coverage is {cov}%, consider improving to 80%+",
                        },
                    )
            except (ValueError, TypeError):
                logger.warning("Could not parse coverage value from analysis output")

        # Heuristic: count test functions vs code functions
        if code and tests:
            code_fns = set(re.findall(r"def\s+(\w+)\s*\(", code))
            test_fns_raw = set(re.findall(r"def\s+(test_\w+)\s*\(", tests))
            # Try to match test function names to code function names
            tested_fns: set[str] = set()
            for code_fn in code_fns:
                if code_fn.startswith("_"):
                    continue  # Skip private functions
                for test_fn in test_fns_raw:
                    if code_fn.lower() in test_fn.lower():
                        tested_fns.add(code_fn)
                        break

            public_fns = {f for f in code_fns if not f.startswith("_")}
            untested = public_fns - tested_fns
            if untested and public_fns:
                ratio = len(untested) / len(public_fns)
                score -= min(0.3, ratio * 0.3)
                issues.append(
                    {
                        "severity": "warning",
                        "category": "coverage",
                        "message": f"{len(untested)} public function(s) appear untested: {', '.join(sorted(untested)[:5])}",
                    },
                )
                suggestions.append("Add tests for untested public functions")

        score = max(0.0, min(1.0, score))
        result_enum = self._score_to_result(score, config.min_score)

        return GateCheckResult(
            gate_name=config.name,
            mode=config.mode,
            result=result_enum,
            score=round(score, 3),
            issues=issues,
            suggestions=suggestions,
        )

    def check_architecture(self, artifacts: dict[str, Any], config: QualityGateConfig) -> GateCheckResult:
        """Run architecture verification.

        Checks ``artifacts["code"]`` for architectural consistency including
        circular imports, layer violations, and naming conventions.

        Args:
            artifacts: Dict optionally containing ``"code"`` and
                ``"architecture"`` (dict with optional ``"package_name"``
                and ``"forbidden_patterns"`` keys).
            config: The quality gate configuration for this check.

        Returns:
            GateCheckResult reflecting the architecture analysis outcome.
        """
        code = artifacts.get("code", "")
        architecture = artifacts.get("architecture", {})
        issues: list[dict[str, Any]] = []
        suggestions: list[str] = []
        score = 1.0

        if not code:
            return GateCheckResult(
                gate_name=config.name,
                mode=config.mode,
                result=GateResult.WARNING,
                score=0.5,
                issues=[{"severity": "warning", "message": "No code artifacts for architecture check"}],
                suggestions=["Provide code artifacts for architecture analysis"],
            )

        # 1. Check for wildcard imports (anti-pattern)
        wildcard_imports = re.findall(r"^from\s+\S+\s+import\s+\*", code, re.MULTILINE)
        if wildcard_imports:
            score -= min(0.15, len(wildcard_imports) * 0.05)
            issues.append(
                {
                    "severity": "warning",
                    "category": "architecture",
                    "message": f"Found {len(wildcard_imports)} wildcard import(s)",
                },
            )
            suggestions.append("Replace wildcard imports with explicit imports")

        # 2. Check for circular import patterns (heuristic: same-package back-imports)
        imports = re.findall(r"^(?:from|import)\s+([\w.]+)", code, re.MULTILINE)
        if architecture.get("package_name"):
            pkg = architecture["package_name"]
            back_imports = [i for i in imports if i.startswith(pkg)]
            if len(back_imports) > 5:
                score -= 0.1
                issues.append(
                    {
                        "severity": "warning",
                        "category": "architecture",
                        "message": f"High internal coupling: {len(back_imports)} intra-package imports",
                    },
                )
                suggestions.append("Consider reducing coupling between modules")

        # 3. Check class count per file (God-module detection)
        classes = re.findall(r"^class\s+(\w+)", code, re.MULTILINE)
        if len(classes) > 5:
            score -= 0.1
            issues.append(
                {
                    "severity": "warning",
                    "category": "architecture",
                    "message": f"File defines {len(classes)} classes, consider splitting",
                },
            )
            suggestions.append("Split large modules into focused, single-responsibility files")

        # 4. Check for forbidden patterns from architecture config
        forbidden = architecture.get("forbidden_patterns", [])
        for pattern_info in forbidden:
            pat = pattern_info if isinstance(pattern_info, str) else pattern_info.get("pattern", "")
            if pat and re.search(pat, code):
                score -= 0.2
                msg = (
                    pattern_info.get("message", f"Forbidden pattern found: {pat}")
                    if isinstance(pattern_info, dict)
                    else f"Forbidden pattern found: {pat}"
                )
                issues.append(
                    {
                        "severity": "error",
                        "category": "architecture",
                        "message": msg,
                    },
                )

        score = max(0.0, min(1.0, score))
        result_enum = self._score_to_result(score, config.min_score)

        return GateCheckResult(
            gate_name=config.name,
            mode=config.mode,
            result=result_enum,
            score=round(score, 3),
            issues=issues,
            suggestions=suggestions,
        )

    def check_prevention(self, artifacts: dict[str, Any], config: QualityGateConfig) -> GateCheckResult:
        """Run pre-execution prevention checks (poka-yoke).

        Validates task inputs before Builder executes using PreventionGate logic.
        Checks: acceptance criteria present, referenced files exist,
        context completeness, model capability, token budget, concurrent conflicts.

        Args:
            artifacts: Task input artifacts. Expected keys:
                ``task_description``, ``acceptance_criteria``, ``referenced_files``,
                ``model_capabilities``, ``required_capabilities``,
                ``estimated_tokens``, ``token_budget``, ``active_file_scopes``.
            config: The quality gate configuration for this check.

        Returns:
            GateCheckResult reflecting the prevention gate outcome.
        """
        from vetinari.validation.prevention import PreventionGate

        task_description: str = artifacts.get("task_description", "")
        acceptance_criteria: list[str] = artifacts.get("acceptance_criteria", [])
        referenced_files: list[str] = artifacts.get("referenced_files", [])
        model_capabilities: set[str] = artifacts.get("model_capabilities", set())
        required_capabilities: set[str] = artifacts.get("required_capabilities", set())
        estimated_tokens: int = artifacts.get("estimated_tokens", 0)
        token_budget: int = artifacts.get("token_budget", 100_000)
        active_file_scopes: set[str] = artifacts.get("active_file_scopes", set())

        gate = PreventionGate()
        prevention_result = gate.validate(
            task_description=task_description,
            acceptance_criteria=acceptance_criteria,
            referenced_files=referenced_files,
            model_capabilities=model_capabilities,
            required_capabilities=required_capabilities,
            estimated_tokens=estimated_tokens,
            token_budget=token_budget,
            active_file_scopes=active_file_scopes,
        )

        issues: list[dict[str, Any]] = []
        if prevention_result.passed:
            score = 1.0
        else:
            failure_count = len(prevention_result.failures)
            # Each failure deducts equally; at least 6 failures → score 0.0
            score = max(0.0, 1.0 - failure_count / 6.0)
            issues.extend(
                {"severity": "error", "category": "prevention", "message": failure.reason}
                for failure in prevention_result.failures
            )

        result_enum = self._score_to_result(score, config.min_score)
        return GateCheckResult(
            gate_name=config.name,
            mode=config.mode,
            result=result_enum,
            score=round(score, 3),
            issues=issues,
            suggestions=(
                [f"Recommended action: {prevention_result.recommendation}"] if not prevention_result.passed else []
            ),
            metadata={"recommendation": prevention_result.recommendation},
        )

    # ------------------------------------------------------------------
    # Internal helpers used by the check methods above
    # ------------------------------------------------------------------

    def _run_single_gate(self, config: QualityGateConfig, artifacts: dict[str, Any]) -> GateCheckResult:
        """Dispatch to the appropriate check method based on verification mode.

        Args:
            config: Gate configuration specifying the mode.
            artifacts: The artifacts to check.

        Returns:
            GateCheckResult from the dispatched check method.
        """
        dispatch = {
            VerificationMode.VERIFY_QUALITY: self.check_quality,
            VerificationMode.SECURITY: self.check_security,
            VerificationMode.VERIFY_COVERAGE: self.check_coverage,
            VerificationMode.VERIFY_ARCHITECTURE: self.check_architecture,
            VerificationMode.PRE_EXECUTION: self.check_prevention,
        }
        handler = dispatch.get(config.mode)
        if handler is None:
            return GateCheckResult(
                gate_name=config.name,
                mode=config.mode,
                result=GateResult.WARNING,
                score=0.5,
                issues=[
                    {
                        "severity": "warning",
                        "message": f"No handler for verification mode: {config.mode.value}",
                    },
                ],
            )
        return handler(artifacts, config)

    @staticmethod
    def _score_to_result(score: float, min_score: float) -> GateResult:
        """Convert a numeric score to a GateResult enum value.

        A score below ``min_score * 0.7`` is FAILED; between that and
        ``min_score`` is WARNING; at or above ``min_score`` is PASSED.

        Args:
            score: Numeric score (0.0-1.0).
            min_score: Minimum passing score from gate config.

        Returns:
            GateResult enum value.
        """
        if score >= min_score:
            return GateResult.PASSED
        if score >= min_score * 0.7:
            return GateResult.WARNING
        return GateResult.FAILED

    @staticmethod
    def _check_long_functions(code: str, max_lines: int = 50) -> list[str]:
        """Return names of functions that exceed max_lines.

        Args:
            code: Source code string to scan.
            max_lines: Line count above which a function is considered too long.

        Returns:
            List of function names that exceed the limit.
        """
        long_fns = []
        lines = code.split("\n")
        current_fn = None
        fn_start = 0

        for i, line in enumerate(lines):
            match = re.match(r"^(\s*)def\s+(\w+)\s*\(", line)
            if match:
                if current_fn is not None and (i - fn_start) > max_lines:
                    long_fns.append(current_fn)
                current_fn = match.group(2)
                fn_start = i

        # Check last function
        if current_fn is not None and (len(lines) - fn_start) > max_lines:
            long_fns.append(current_fn)

        return long_fns

    @staticmethod
    def _check_missing_docstrings(code: str) -> list[str]:
        """Return names of public functions missing docstrings.

        Args:
            code: Source code string to scan.

        Returns:
            List of public function names (not prefixed with ``_``) that
            have no docstring on the line(s) immediately following their
            ``def`` statement.
        """
        missing = []
        lines = code.split("\n")

        for i, line in enumerate(lines):
            match = re.match(r"^\s*def\s+(\w+)\s*\(", line)
            if match:
                fn_name = match.group(1)
                if fn_name.startswith("_"):
                    continue  # Skip private functions
                found_docstring = False
                for j in range(i + 1, min(i + 5, len(lines))):
                    stripped = lines[j].strip()
                    if not stripped:
                        continue
                    if stripped.startswith(('"""', "'''")):
                        found_docstring = True
                    break
                if not found_docstring:
                    missing.append(fn_name)

        return missing
