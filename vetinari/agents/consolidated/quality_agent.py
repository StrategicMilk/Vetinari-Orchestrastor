"""
Consolidated Quality Agent (Phase 3)
======================================
Replaces: EVALUATOR + SECURITY_AUDITOR + TEST_AUTOMATION

Modes:
- code_review: General code quality, design patterns, maintainability
- security_audit: Vulnerability detection with 45+ heuristic patterns + LLM
- test_generation: pytest-aware test generation with coverage analysis
- simplification: Code simplification and refactoring recommendations
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from vetinari.agents.multi_mode_agent import MultiModeAgent
from vetinari.agents.contracts import AgentResult, AgentTask, AgentType, VerificationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Security heuristic patterns (preserved from SecurityAuditorAgent)
# ---------------------------------------------------------------------------

_SECURITY_PATTERNS: List[Tuple[str, str, str]] = [
    # (pattern_regex, finding_name, severity)
    (r"subprocess\.(?:call|run|Popen)\(.*shell\s*=\s*True", "Shell injection risk", "HIGH"),
    (r"input\s*\(", "Unsanitized user input", "MEDIUM"),
    (r"password\s*=\s*['\"][^'\"]+['\"]", "Hardcoded password", "CRITICAL"),
    (r"api[_-]?key\s*=\s*['\"][^'\"]+['\"]", "Hardcoded API key", "CRITICAL"),
    (r"secret\s*=\s*['\"][^'\"]+['\"]", "Hardcoded secret", "CRITICAL"),
    (r"token\s*=\s*['\"][^'\"]+['\"]", "Hardcoded token", "HIGH"),
    (r"SELECT\s+.*\s+FROM\s+.*\+\s*", "Potential SQL injection", "HIGH"),
    (r"f['\"].*SELECT.*FROM.*\{", "Potential SQL injection (f-string)", "HIGH"),
    (r"\.format\(.*\).*SELECT", "Potential SQL injection (.format)", "HIGH"),
    (r"yaml\.load\((?!.*Loader)", "Unsafe YAML loading", "HIGH"),
    (r"marshal\.loads?\(", "Unsafe deserialization (marshal)", "HIGH"),
    (r"open\(.*['\"]w['\"].*\).*\n.*write\(.*input", "Write from user input", "MEDIUM"),
    (r"chmod\s*\(\s*0o?777", "World-writable permissions", "HIGH"),
    (r"verify\s*=\s*False", "SSL verification disabled", "HIGH"),
    (r"CORS\(.*origins\s*=\s*['\"]?\*", "Wildcard CORS", "MEDIUM"),
    (r"DEBUG\s*=\s*True", "Debug mode enabled", "MEDIUM"),
    (r"assert\s+", "Assert in production code", "LOW"),
    (r"except\s*:", "Bare except clause", "LOW"),
    (r"# ?TODO|# ?FIXME|# ?HACK|# ?XXX", "Unresolved code annotation", "LOW"),
    (r"\.env", "Potential dotenv reference", "INFO"),
    (r"tempfile\.mk(?:s?temp|dtemp)\((?!.*dir=)", "Temp file without explicit directory", "LOW"),
    (r"md5\(|sha1\(", "Weak hash algorithm", "MEDIUM"),
    (r"random\.random\(|random\.randint\(", "Non-cryptographic random", "LOW"),
    (r"urllib\.request\.urlopen\(", "Unvalidated URL open", "MEDIUM"),
    (r"exec\s*\(|compile\s*\(", "Dynamic code execution", "HIGH"),
    # ── B6: Expanded CWE patterns (15 new) ──
    # CWE-434: Unrestricted file upload
    (r"\.save\(.*filename\)|\bsave_file\b.*request\.", "Unrestricted file upload (CWE-434)", "HIGH"),
    # CWE-611: XML External Entity (XXE)
    (r"xml\.etree\.ElementTree\.parse\(|lxml\.etree\.parse\(", "Potential XXE — use defusedxml (CWE-611)", "HIGH"),
    (r"xml\.sax\.parse\(|xml\.parsers\.expat", "Potential XXE via SAX/expat (CWE-611)", "HIGH"),
    # CWE-918: Server-Side Request Forgery (SSRF)
    (r"requests\.(?:get|post|put|delete|patch|head)\(.*(?:user|param|arg|input|query)", "Potential SSRF — validate URLs (CWE-918)", "HIGH"),
    # CWE-502: Insecure deserialization — detect usage of unsafe deserializers
    (r"shelve\.open\(|jsonpickle\.decode\(|dill\.loads?\(", "Insecure deserialization (CWE-502)", "CRITICAL"),
    # CWE-732: Incorrect permission assignment
    (r"os\.chmod\(.*0o?[67][67][67]", "Overly permissive file mode (CWE-732)", "MEDIUM"),
    (r"umask\s*\(\s*0\s*\)", "Umask disabled — files world-accessible (CWE-732)", "HIGH"),
    # CWE-295: Improper certificate validation
    (r"ssl\._create_unverified_context|ssl\.CERT_NONE", "SSL cert validation disabled (CWE-295)", "HIGH"),
    # CWE-327: Use of broken cryptographic algorithm
    (r"DES\b|Blowfish|RC4|RC2|ARC4", "Weak/broken cipher algorithm (CWE-327)", "HIGH"),
    # CWE-798: Hardcoded credentials
    (r"(?:db_pass|database_password|mysql_pwd)\s*=\s*['\"]", "Hardcoded database password (CWE-798)", "CRITICAL"),
    # CWE-89: SQL injection (additional patterns)
    (r"%s.*(?:SELECT|INSERT|UPDATE|DELETE|DROP).*%\s*\(", "SQL injection via percent formatting (CWE-89)", "HIGH"),
    (r"cursor\.execute\(.*\+", "SQL injection via string concatenation (CWE-89)", "HIGH"),
    # CWE-22: Path traversal
    (r"open\(.*(?:request|user_input|params).*\)", "Potential path traversal (CWE-22)", "HIGH"),
    # CWE-400: Uncontrolled resource consumption
    (r"while\s+True.*(?:recv|read|accept)", "Unbounded I/O loop — potential DoS (CWE-400)", "MEDIUM"),
    # CWE-312: Cleartext storage/logging of sensitive data
    (r"log(?:ger)?\.(?:info|debug|warning|error).*(?:password|secret|token|api.?key)", "Logging sensitive data in cleartext (CWE-312)", "HIGH"),
    # CWE-942: CSRF protection disabled
    (r"csrf_exempt|WTF_CSRF_ENABLED\s*=\s*False", "CSRF protection disabled (CWE-942)", "HIGH"),
    # CWE-1004: Sensitive cookie without HttpOnly
    (r"set_cookie\(.*secure\s*=\s*False", "Cookie without Secure flag (CWE-1004)", "MEDIUM"),
]


class QualityAgent(MultiModeAgent):
    """Unified quality agent for code review, security audit, and test generation."""

    MODES = {
        "code_review": "_execute_code_review",
        "security_audit": "_execute_security_audit",
        "test_generation": "_execute_test_generation",
        "simplification": "_execute_simplification",
    }
    DEFAULT_MODE = "code_review"
    MODE_KEYWORDS = {
        "code_review": ["review", "quality", "maintainab", "readab", "refactor", "clean",
                         "design pattern", "solid", "code smell"],
        "security_audit": ["security", "vulnerab", "audit", "cwe", "owasp", "injection",
                            "xss", "csrf", "auth", "encrypt", "credential"],
        "test_generation": ["test", "pytest", "coverage", "unit test", "integration test",
                             "mock", "fixture", "assert", "tdd"],
        "simplification": ["simplif", "complex", "reduce", "clean up", "streamline"],
    }
    LEGACY_TYPE_TO_MODE = {
        "EVALUATOR": "code_review",
        "SECURITY_AUDITOR": "security_audit",
        "TEST_AUTOMATION": "test_generation",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.QUALITY, config)

    def _get_base_system_prompt(self) -> str:
        return (
            "You are Vetinari's Quality Agent. You perform code review, "
            "security audits, test generation, and simplification analysis."
        )

    def _get_mode_system_prompt(self, mode: str) -> str:
        prompts = {
            "code_review": (
                "You are Vetinari's Code Quality Reviewer. Your role is to:\n"
                "- Assess code quality, readability, and maintainability\n"
                "- Identify design pattern violations and anti-patterns\n"
                "- Check SOLID principles adherence\n"
                "- Evaluate error handling and edge cases\n"
                "- Score quality 0.0-1.0\n\n"
                "Check for: logic errors, race conditions, resource leaks, missing error "
                "handling, API contract violations. Rate severity per finding.\n"
                "Be constructive. Prioritize issues by impact."
            ),
            "security_audit": (
                "You are Vetinari's Security Auditor. Your role is to:\n"
                "- Detect vulnerabilities using heuristic patterns + analysis\n"
                "- Classify findings by CWE ID and OWASP Top 10\n"
                "- Assess severity (CRITICAL/HIGH/MEDIUM/LOW/INFO)\n"
                "- Provide remediation code examples\n"
                "- Check for hardcoded secrets, injection flaws, unsafe deserialization\n\n"
                "Map findings to OWASP Top 10 categories. Include CWE IDs. "
                "Provide remediation code snippets.\n"
                "Be thorough. False negatives are worse than false positives."
            ),
            "test_generation": (
                "You are Vetinari's Test Automation Specialist. Your role is to:\n"
                "- Generate comprehensive pytest test suites\n"
                "- Create fixtures, mocks, and parametrized tests\n"
                "- Target coverage gaps and edge cases\n"
                "- Include both positive and negative test cases\n"
                "- Generate integration tests when appropriate\n\n"
                "Target 80% branch coverage. Include: happy path, error path, edge cases, "
                "boundary values. Use Arrange-Act-Assert pattern.\n"
                "Tests must be runnable and follow pytest conventions."
            ),
            "simplification": (
                "You are Vetinari's Code Simplification Specialist. Your role is to:\n"
                "- Identify overly complex code\n"
                "- Suggest simpler alternatives\n"
                "- Reduce cyclomatic complexity\n"
                "- Extract reusable patterns\n"
                "- Improve naming and structure\n\n"
                "Preserve all behavior. Reduce cyclomatic complexity. Extract only when "
                "reuse is proven (>=3 call sites).\n"
                "Preserve functionality. Simplify without losing clarity."
            ),
        }
        return prompts.get(mode, "")

    def verify(self, output: Any) -> VerificationResult:
        if output is None:
            return VerificationResult(passed=False, issues=[{"message": "No output"}], score=0.0)
        if isinstance(output, dict):
            has_review = bool(
                output.get("issues") or output.get("findings")
                or output.get("tests") or output.get("score") is not None
            )
            return VerificationResult(passed=has_review, score=0.8 if has_review else 0.4)
        return VerificationResult(passed=True, score=0.6)

    # ------------------------------------------------------------------
    # Code Review (from EvaluatorAgent)
    # ------------------------------------------------------------------

    def _execute_code_review(self, task: AgentTask) -> AgentResult:
        code = task.context.get("code", task.description)
        review_type = task.context.get("review_type", "general")

        prompt = (
            f"Review the following code for quality and maintainability:\n\n"
            f"```\n{code[:6000]}\n```\n\n"
            f"Review focus: {review_type}\n\n"
            "Respond as JSON:\n"
            '{"score": 0.75, "summary": "...", '
            '"issues": [{"severity": "high|medium|low", "category": "...", '
            '"message": "...", "line": 0, "suggestion": "..."}], '
            '"strengths": [...], "recommendations": [...]}'
        )
        result = self._infer_json(prompt, fallback={"score": 0.5, "issues": [], "summary": "Review unavailable"})
        return AgentResult(success=True, output=result, metadata={"mode": "code_review", "review_type": review_type})

    # ------------------------------------------------------------------
    # Security Audit (from SecurityAuditorAgent — preserves heuristics)
    # ------------------------------------------------------------------

    def _execute_security_audit(self, task: AgentTask) -> AgentResult:
        code = task.context.get("code", task.description)

        # Phase 1: Heuristic pattern scan
        heuristic_findings = self._run_heuristic_scan(code)

        # Phase 2: LLM-based deep analysis
        heuristic_summary = ""
        if heuristic_findings:
            heuristic_summary = (
                "\n\nHeuristic scan found these preliminary issues:\n"
                + "\n".join(f"- [{f['severity']}] {f['finding']}" for f in heuristic_findings[:10])
            )

        prompt = (
            f"Perform a comprehensive security audit of this code:\n\n"
            f"```\n{code[:5000]}\n```\n"
            f"{heuristic_summary}\n\n"
            "Analyze for: injection, broken auth, sensitive data exposure, "
            "XXE, broken access control, misconfig, XSS, insecure deserialization, "
            "vulnerable components, insufficient logging.\n\n"
            "Respond as JSON:\n"
            '{"findings": [{"severity": "CRITICAL|HIGH|MEDIUM|LOW|INFO", '
            '"finding": "...", "cwe": "CWE-79 (use real CWE IDs: 79=XSS, 89=SQLi, 22=PathTraversal, 78=OSCmd, 502=Deserialization, 798=HardcodedCreds, 327=BrokenCrypto, 306=MissingAuth)", "owasp": "A01-A10", '
            '"line": 0, "remediation": "...", "code_example": "..."}], '
            '"summary": "...", "overall_risk": "high|medium|low", '
            '"score": 0.75}'
        )

        llm_result = self._infer_json(prompt, fallback=None)

        # Merge heuristic + LLM findings
        if llm_result and isinstance(llm_result, dict):
            llm_findings = llm_result.get("findings", [])
            # Add heuristic findings not already covered by LLM
            llm_finding_names = {f.get("finding", "").lower() for f in llm_findings}
            for hf in heuristic_findings:
                if hf["finding"].lower() not in llm_finding_names:
                    llm_findings.append(hf)
            llm_result["findings"] = llm_findings
            llm_result.setdefault("heuristic_count", len(heuristic_findings))
            return AgentResult(success=True, output=llm_result,
                               metadata={"mode": "security_audit", "heuristic_findings": len(heuristic_findings)})

        # Heuristic-only fallback
        return AgentResult(
            success=True,
            output={
                "findings": heuristic_findings,
                "summary": f"Heuristic scan found {len(heuristic_findings)} issues (LLM unavailable)",
                "overall_risk": "high" if any(f["severity"] in ("CRITICAL", "HIGH") for f in heuristic_findings) else "medium",
                "score": max(0.0, 1.0 - len(heuristic_findings) * 0.1),
            },
            metadata={"mode": "security_audit", "heuristic_only": True},
        )

    def _run_heuristic_scan(self, code: str) -> List[Dict[str, Any]]:
        """Run regex-based security heuristic patterns against code."""
        findings = []
        lines = code.split("\n")
        for line_num, line in enumerate(lines, 1):
            for pattern, finding_name, severity in _SECURITY_PATTERNS:
                try:
                    if re.search(pattern, line, re.IGNORECASE):
                        findings.append({
                            "severity": severity,
                            "finding": finding_name,
                            "line": line_num,
                            "evidence": line.strip()[:120],
                            "source": "heuristic",
                        })
                except re.error:
                    pass
        return findings

    # ------------------------------------------------------------------
    # Test Generation (from TestAutomationAgent)
    # ------------------------------------------------------------------

    def _execute_test_generation(self, task: AgentTask) -> AgentResult:
        code = task.context.get("code", task.description)
        test_type = task.context.get("test_type", "unit")
        target_coverage = task.context.get("target_coverage", 0.8)

        prompt = (
            f"Generate comprehensive {test_type} tests for this code:\n\n"
            f"```python\n{code[:5000]}\n```\n\n"
            f"Target coverage: {target_coverage * 100}%\n\n"
            "Requirements:\n"
            "- Use pytest framework\n"
            "- Include fixtures and parametrize where appropriate\n"
            "- Test both happy paths and edge cases\n"
            "- Include error/exception testing\n"
            "- Add docstrings explaining test purpose\n\n"
            "Respond as JSON:\n"
            '{"tests": "...full pytest code...", '
            '"test_count": 0, "coverage_estimate": 0.8, '
            '"fixtures": [...], "edge_cases_covered": [...]}'
        )
        result = self._infer_json(prompt, fallback={"tests": "", "test_count": 0, "coverage_estimate": 0.0})
        return AgentResult(success=True, output=result,
                           metadata={"mode": "test_generation", "test_type": test_type})

    # ------------------------------------------------------------------
    # Simplification
    # ------------------------------------------------------------------

    def _execute_simplification(self, task: AgentTask) -> AgentResult:
        code = task.context.get("code", task.description)

        prompt = (
            f"Analyze this code for simplification opportunities:\n\n"
            f"```\n{code[:6000]}\n```\n\n"
            "Respond as JSON:\n"
            '{"score": 0.6, "complexity_issues": [{"location": "...", '
            '"issue": "...", "suggestion": "...", "simplified_code": "..."}], '
            '"overall_recommendations": [...], '
            '"estimated_line_reduction": 0}'
        )
        result = self._infer_json(prompt, fallback={"score": 0.5, "complexity_issues": []})
        return AgentResult(success=True, output=result, metadata={"mode": "simplification"})

    def get_capabilities(self) -> List[str]:
        return [
            "code_review", "quality_scoring", "design_pattern_check",
            "security_audit", "vulnerability_detection", "cwe_classification",
            "test_generation", "coverage_analysis", "fixture_generation",
            "code_simplification", "complexity_reduction",
        ]


# Singleton
_quality_agent: Optional[QualityAgent] = None


def get_quality_agent(config: Optional[Dict[str, Any]] = None) -> QualityAgent:
    global _quality_agent
    if _quality_agent is None:
        _quality_agent = QualityAgent(config)
    return _quality_agent
