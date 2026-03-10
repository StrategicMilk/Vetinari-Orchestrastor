"""
Quality Gates for Vetinari Agentic Verification System (Task 26)

Implements verification as TESTER agent modes that serve as quality gates
between execution stages. Uses modes within existing agents rather than
creating new agent types.

Features:
- VerificationMode enum for TESTER agent verification modes
- QualityGateConfig for configuring individual gates
- GateCheckResult for gate check outcomes
- QualityGateRunner for running gates between pipeline stages
"""

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VerificationMode(Enum):
    """Modes for the TESTER agent verification.

    Each mode represents a different verification strategy that the
    TESTER (Evaluator) agent can operate in during quality gate checks.
    """
    VERIFY_QUALITY = "verify_quality"           # Style, complexity, best practices
    SECURITY = "security"                        # Security checks
    VERIFY_COVERAGE = "verify_coverage"         # Test existence and pass rate
    VERIFY_ARCHITECTURE = "verify_architecture" # Consistency with project arch


class GateResult(Enum):
    """Outcome of a quality gate check."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class QualityGateConfig:
    """Configuration for a quality gate.

    Attributes:
        name: Human-readable gate name.
        mode: The VerificationMode to use for this gate.
        required: If True, failure blocks the pipeline.
        min_score: Minimum score to pass (0.0-1.0).
        timeout_seconds: Maximum time allowed for the check.
        auto_fix: If True, attempt auto-remediation on failure.
    """
    name: str
    mode: VerificationMode
    required: bool = True
    min_score: float = 0.6
    timeout_seconds: int = 60
    auto_fix: bool = False


@dataclass
class GateCheckResult:
    """Result of a quality gate check.

    Attributes:
        gate_name: Name of the gate that produced this result.
        mode: The verification mode used.
        result: PASSED, FAILED, or WARNING.
        score: Numeric score (0.0-1.0).
        issues: List of issue dictionaries found during the check.
        suggestions: List of improvement suggestions.
        metadata: Additional metadata about the check.
    """
    gate_name: str
    mode: VerificationMode
    result: GateResult
    score: float
    issues: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a serializable dictionary."""
        return {
            "gate_name": self.gate_name,
            "mode": self.mode.value,
            "result": self.result.value,
            "score": self.score,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "metadata": self.metadata,
        }


class QualityGateRunner:
    """Runs quality gates between pipeline stages.

    Each pipeline stage (post_planning, post_execution, post_testing,
    pre_assembly) has a set of configured quality gates. The runner
    executes each gate's verification checks and collects results.

    The PIPELINE_GATES class variable defines the default gate
    configuration for each stage. Custom gates can be provided at
    construction time to override or extend defaults.
    """

    PIPELINE_GATES: Dict[str, List[QualityGateConfig]] = {
        "post_planning": [
            QualityGateConfig(
                "architecture_check",
                VerificationMode.VERIFY_ARCHITECTURE,
                min_score=0.7,
            ),
        ],
        "post_execution": [
            QualityGateConfig(
                "quality_check",
                VerificationMode.VERIFY_QUALITY,
                min_score=0.6,
            ),
            QualityGateConfig(
                "security_check",
                VerificationMode.SECURITY,
                min_score=0.8,
                required=True,
            ),
        ],
        "post_testing": [
            QualityGateConfig(
                "coverage_check",
                VerificationMode.VERIFY_COVERAGE,
                min_score=0.5,
            ),
        ],
        "pre_assembly": [
            QualityGateConfig(
                "final_quality",
                VerificationMode.VERIFY_QUALITY,
                min_score=0.7,
            ),
            QualityGateConfig(
                "final_security",
                VerificationMode.SECURITY,
                min_score=0.9,
            ),
        ],
    }

    def __init__(self, custom_gates: Optional[Dict[str, List[QualityGateConfig]]] = None):
        """Initialize the runner with optional custom gate configurations.

        Args:
            custom_gates: Optional dict mapping stage names to gate configs.
                          Merges with (and overrides) the default PIPELINE_GATES.
        """
        self._gates: Dict[str, List[QualityGateConfig]] = dict(self.PIPELINE_GATES)
        if custom_gates:
            self._gates.update(custom_gates)
        self._history: List[GateCheckResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_gate(self, stage: str, artifacts: Dict[str, Any]) -> List[GateCheckResult]:
        """Run all gates for a pipeline stage.

        Args:
            stage: Pipeline stage name (e.g. "post_execution").
            artifacts: Dictionary of artifacts to verify. Expected keys
                       depend on the gate mode but typically include
                       "code", "tests", "architecture", etc.

        Returns:
            List of GateCheckResult for each gate in the stage.
            Returns an empty list if the stage has no configured gates.
        """
        gate_configs = self._gates.get(stage, [])
        if not gate_configs:
            logger.debug("No gates configured for stage '%s'", stage)
            return []

        results: List[GateCheckResult] = []
        for config in gate_configs:
            start = time.time()
            try:
                result = self._run_single_gate(config, artifacts)
            except Exception as exc:
                logger.error(
                    "Gate '%s' raised an exception: %s", config.name, exc
                )
                result = GateCheckResult(
                    gate_name=config.name,
                    mode=config.mode,
                    result=GateResult.FAILED,
                    score=0.0,
                    issues=[{"severity": "error", "message": f"Gate error: {exc}"}],
                )
            elapsed_ms = int((time.time() - start) * 1000)
            result.metadata["execution_time_ms"] = elapsed_ms
            result.metadata["stage"] = stage
            result.metadata["required"] = config.required

            results.append(result)
            self._history.append(result)

        return results

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the full history of gate check results.

        Returns:
            List of serialized GateCheckResult dictionaries.
        """
        return [r.to_dict() for r in self._history]

    def get_gates_for_stage(self, stage: str) -> List[QualityGateConfig]:
        """Return the gate configs for a given stage."""
        return list(self._gates.get(stage, []))

    def stage_passed(self, results: List[GateCheckResult]) -> bool:
        """Return True if all *required* gates in the results passed.

        A gate is considered passing if its result is PASSED or WARNING.
        Only gates marked as ``required`` can cause a stage failure.
        """
        for r in results:
            is_required = r.metadata.get("required", True)
            if is_required and r.result == GateResult.FAILED:
                return False
        return True

    # ------------------------------------------------------------------
    # Gate-specific check implementations
    # ------------------------------------------------------------------

    def check_quality(
        self, artifacts: Dict[str, Any], config: QualityGateConfig
    ) -> GateCheckResult:
        """Run quality verification (style, complexity, best practices).

        Inspects ``artifacts["code"]`` for common quality issues using
        lightweight heuristic analysis. Falls back gracefully when the
        code key is absent.
        """
        code = artifacts.get("code", "")
        issues: List[Dict[str, Any]] = []
        suggestions: List[str] = []
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

        # --- Heuristic checks ---

        # 1. Function length (rough proxy for complexity)
        long_functions = self._check_long_functions(code)
        if long_functions:
            penalty = min(0.3, len(long_functions) * 0.1)
            score -= penalty
            for fn in long_functions:
                issues.append({
                    "severity": "warning",
                    "category": "complexity",
                    "message": f"Function '{fn}' appears to be overly long",
                })
            suggestions.append("Break large functions into smaller, focused helpers")

        # 2. Missing docstrings
        missing_docs = self._check_missing_docstrings(code)
        if missing_docs:
            penalty = min(0.2, len(missing_docs) * 0.05)
            score -= penalty
            for fn in missing_docs:
                issues.append({
                    "severity": "info",
                    "category": "documentation",
                    "message": f"Function '{fn}' is missing a docstring",
                })
            suggestions.append("Add docstrings to all public functions")

        # 3. Bare except clauses
        bare_excepts = len(re.findall(r'except\s*:', code))
        if bare_excepts:
            score -= min(0.2, bare_excepts * 0.1)
            issues.append({
                "severity": "warning",
                "category": "best_practices",
                "message": f"Found {bare_excepts} bare except clause(s)",
            })
            suggestions.append("Catch specific exception types instead of bare except")

        # 4. TODO/FIXME/HACK markers
        markers = len(re.findall(r'#\s*(TODO|FIXME|HACK|XXX)\b', code, re.IGNORECASE))
        if markers:
            score -= min(0.1, markers * 0.02)
            issues.append({
                "severity": "info",
                "category": "maintenance",
                "message": f"Found {markers} TODO/FIXME/HACK marker(s)",
            })

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

    def check_security(
        self, artifacts: Dict[str, Any], config: QualityGateConfig
    ) -> GateCheckResult:
        """Run security verification.

        Checks ``artifacts["code"]`` for dangerous patterns, potential
        secrets, and unsafe practices.
        """
        code = artifacts.get("code", "")
        issues: List[Dict[str, Any]] = []
        suggestions: List[str] = []
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

        # --- Security heuristics ---

        dangerous_patterns = [
            (r'eval\s*\(', "eval() allows arbitrary code execution", "critical"),
            (r'exec\s*\(', "exec() allows arbitrary code execution", "critical"),
            (r'__import__\s*\(', "Dynamic __import__() may be unsafe", "high"),
            (r'os\.system\s*\(', "os.system() is vulnerable to shell injection", "high"),
            (r'subprocess.*shell\s*=\s*True', "subprocess with shell=True is dangerous", "high"),
            (r'pickle\.loads?\s*\(', "pickle deserialization can execute arbitrary code", "high"),
            (r'yaml\.load\s*\((?!.*Loader)', "yaml.load without Loader is unsafe", "medium"),
            (r'input\s*\(', "input() in production code may be unintended", "low"),
        ]

        severity_penalties = {"critical": 0.3, "high": 0.2, "medium": 0.1, "low": 0.05}

        for pattern, message, severity in dangerous_patterns:
            matches = re.findall(pattern, code)
            if matches:
                penalty = severity_penalties.get(severity, 0.1)
                score -= penalty
                issues.append({
                    "severity": severity,
                    "category": "security",
                    "message": message,
                    "count": len(matches),
                })
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
                issues.append({
                    "severity": "critical",
                    "category": "secrets",
                    "message": message,
                })
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

    def check_coverage(
        self, artifacts: Dict[str, Any], config: QualityGateConfig
    ) -> GateCheckResult:
        """Run coverage verification.

        Checks ``artifacts["tests"]`` for test existence and
        ``artifacts["coverage_percent"]`` for coverage threshold.
        Also checks ``artifacts["code"]`` for testable functions
        without corresponding tests.
        """
        tests = artifacts.get("tests", "")
        code = artifacts.get("code", "")
        coverage_pct = artifacts.get("coverage_percent")
        issues: List[Dict[str, Any]] = []
        suggestions: List[str] = []
        score = 1.0

        # Check that tests exist
        if not tests:
            score -= 0.4
            issues.append({
                "severity": "error",
                "category": "coverage",
                "message": "No test artifacts provided",
            })
            suggestions.append("Write tests for the implemented code")

        # Check coverage percentage if provided
        if coverage_pct is not None:
            try:
                cov = float(coverage_pct)
                if cov < 50:
                    score -= 0.3
                    issues.append({
                        "severity": "error",
                        "category": "coverage",
                        "message": f"Test coverage is {cov}%, below 50% minimum",
                    })
                elif cov < 70:
                    score -= 0.15
                    issues.append({
                        "severity": "warning",
                        "category": "coverage",
                        "message": f"Test coverage is {cov}%, below 70% target",
                    })
                elif cov < 80:
                    score -= 0.05
                    issues.append({
                        "severity": "info",
                        "category": "coverage",
                        "message": f"Test coverage is {cov}%, consider improving to 80%+",
                    })
            except (ValueError, TypeError):
                pass

        # Heuristic: count test functions vs code functions
        if code and tests:
            code_fns = set(re.findall(r'def\s+(\w+)\s*\(', code))
            test_fns_raw = set(re.findall(r'def\s+(test_\w+)\s*\(', tests))
            # Try to match test function names to code function names
            tested_fns = set()
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
                issues.append({
                    "severity": "warning",
                    "category": "coverage",
                    "message": f"{len(untested)} public function(s) appear untested: {', '.join(sorted(untested)[:5])}",
                })
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

    def check_architecture(
        self, artifacts: Dict[str, Any], config: QualityGateConfig
    ) -> GateCheckResult:
        """Run architecture verification.

        Checks ``artifacts["code"]`` for architectural consistency
        including circular imports, layer violations, and naming
        conventions.
        """
        code = artifacts.get("code", "")
        architecture = artifacts.get("architecture", {})
        issues: List[Dict[str, Any]] = []
        suggestions: List[str] = []
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
        wildcard_imports = re.findall(r'^from\s+\S+\s+import\s+\*', code, re.MULTILINE)
        if wildcard_imports:
            score -= min(0.15, len(wildcard_imports) * 0.05)
            issues.append({
                "severity": "warning",
                "category": "architecture",
                "message": f"Found {len(wildcard_imports)} wildcard import(s)",
            })
            suggestions.append("Replace wildcard imports with explicit imports")

        # 2. Check for circular import patterns (heuristic: same-package back-imports)
        imports = re.findall(r'^(?:from|import)\s+([\w.]+)', code, re.MULTILINE)
        if architecture.get("package_name"):
            pkg = architecture["package_name"]
            back_imports = [i for i in imports if i.startswith(pkg)]
            if len(back_imports) > 5:
                score -= 0.1
                issues.append({
                    "severity": "warning",
                    "category": "architecture",
                    "message": f"High internal coupling: {len(back_imports)} intra-package imports",
                })
                suggestions.append("Consider reducing coupling between modules")

        # 3. Check class count per file (God-module detection)
        classes = re.findall(r'^class\s+(\w+)', code, re.MULTILINE)
        if len(classes) > 5:
            score -= 0.1
            issues.append({
                "severity": "warning",
                "category": "architecture",
                "message": f"File defines {len(classes)} classes, consider splitting",
            })
            suggestions.append("Split large modules into focused, single-responsibility files")

        # 4. Check for forbidden patterns from architecture config
        forbidden = architecture.get("forbidden_patterns", [])
        for pattern_info in forbidden:
            pat = pattern_info if isinstance(pattern_info, str) else pattern_info.get("pattern", "")
            if pat and re.search(pat, code):
                score -= 0.2
                msg = pattern_info.get("message", f"Forbidden pattern found: {pat}") if isinstance(pattern_info, dict) else f"Forbidden pattern found: {pat}"
                issues.append({
                    "severity": "error",
                    "category": "architecture",
                    "message": msg,
                })

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_single_gate(
        self, config: QualityGateConfig, artifacts: Dict[str, Any]
    ) -> GateCheckResult:
        """Dispatch to the appropriate check method based on mode."""
        dispatch = {
            VerificationMode.VERIFY_QUALITY: self.check_quality,
            VerificationMode.SECURITY: self.check_security,
            VerificationMode.VERIFY_COVERAGE: self.check_coverage,
            VerificationMode.VERIFY_ARCHITECTURE: self.check_architecture,
        }
        handler = dispatch.get(config.mode)
        if handler is None:
            return GateCheckResult(
                gate_name=config.name,
                mode=config.mode,
                result=GateResult.WARNING,
                score=0.5,
                issues=[{
                    "severity": "warning",
                    "message": f"No handler for verification mode: {config.mode.value}",
                }],
            )
        return handler(artifacts, config)

    @staticmethod
    def _score_to_result(score: float, min_score: float) -> GateResult:
        """Convert a numeric score to a GateResult enum value."""
        if score >= min_score:
            return GateResult.PASSED
        elif score >= min_score * 0.7:
            return GateResult.WARNING
        else:
            return GateResult.FAILED

    @staticmethod
    def _check_long_functions(code: str, max_lines: int = 50) -> List[str]:
        """Return names of functions that exceed max_lines."""
        long_fns = []
        lines = code.split("\n")
        current_fn = None
        fn_start = 0

        for i, line in enumerate(lines):
            match = re.match(r'^(\s*)def\s+(\w+)\s*\(', line)
            if match:
                if current_fn is not None:
                    if (i - fn_start) > max_lines:
                        long_fns.append(current_fn)
                current_fn = match.group(2)
                fn_start = i

        # Check last function
        if current_fn is not None and (len(lines) - fn_start) > max_lines:
            long_fns.append(current_fn)

        return long_fns

    @staticmethod
    def _check_missing_docstrings(code: str) -> List[str]:
        """Return names of functions missing docstrings."""
        missing = []
        lines = code.split("\n")

        for i, line in enumerate(lines):
            match = re.match(r'^\s*def\s+(\w+)\s*\(', line)
            if match:
                fn_name = match.group(1)
                if fn_name.startswith("_"):
                    continue  # Skip private functions
                # Check if next non-empty line is a docstring
                found_docstring = False
                for j in range(i + 1, min(i + 5, len(lines))):
                    stripped = lines[j].strip()
                    if not stripped:
                        continue
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        found_docstring = True
                    break
                if not found_docstring:
                    missing.append(fn_name)

        return missing
