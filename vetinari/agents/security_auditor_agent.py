"""
Vetinari Security Auditor Agent

LLM-powered security auditing agent that performs real code analysis,
vulnerability detection (heuristic + LLM), and generates actionable
remediation guidance based on actual artifacts.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult,
)

logger = logging.getLogger(__name__)

# Patterns that indicate security issues — checked before LLM to give it context
_HEURISTIC_CHECKS = [
    (r"eval\s*\(", "critical", "Use of eval() — arbitrary code execution risk"),
    (r"exec\s*\(", "critical", "Use of exec() — arbitrary code execution risk"),
    (r"pickle\.loads?\s*\(", "high", "Unsafe pickle deserialisation"),
    (r"subprocess\.call.*shell\s*=\s*True", "high", "Shell injection risk via subprocess"),
    (r"os\.system\s*\(", "high", "OS command injection risk"),
    (r"(?i)password\s*=\s*['\"][^'\"]{3,}", "high", "Hardcoded password detected"),
    (r"(?i)api_key\s*=\s*['\"][^'\"]{8,}", "high", "Hardcoded API key detected"),
    (r"(?i)secret\s*=\s*['\"][^'\"]{6,}", "medium", "Hardcoded secret detected"),
    (r"sql\s*=.*%s|sql\s*=.*format\(|sql\s*=.*\+", "high", "Potential SQL injection via string formatting"),
    (r"innerHTML\s*=\s*[^'\"]*\+", "medium", "Potential XSS via innerHTML assignment"),
    (r"cursor\.execute\([^,]+%s", "high", "SQL injection risk — use parameterised queries"),
    (r"DEBUG\s*=\s*True", "medium", "Debug mode enabled in production code"),
    (r"verify\s*=\s*False", "medium", "SSL verification disabled"),
    (r"md5\s*\(|hashlib\.md5", "low", "Weak MD5 hash — use SHA-256 or bcrypt"),
    (r"random\s*\.\s*random\(\)|random\.randint", "low", "Insecure random — use secrets module for security-sensitive ops"),
]


class SecurityAuditorAgent(BaseAgent):
    """Security Auditor agent — heuristic + LLM-powered vulnerability analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.SECURITY_AUDITOR, config)

    def get_system_prompt(self) -> str:
        return """You are Vetinari's Security Auditor — a senior application security engineer.
Your job is to perform a thorough security review of the provided code and artifacts.

You MUST analyse the ACTUAL code content provided. Do not produce generic advice.
Reference specific line numbers, function names, and patterns found in the code.

Required output (JSON):
{
  "verdict": "pass" | "conditional" | "fail",
  "security_score": 0-100,
  "security_issues": [
    {
      "id": "SEC-001",
      "severity": "critical" | "high" | "medium" | "low" | "info",
      "category": "injection" | "auth" | "crypto" | "config" | "data_exposure" | "dependency" | "logic",
      "title": "...",
      "description": "...",
      "location": "file:line or component",
      "cwe": "CWE-XXX",
      "evidence": "the actual code snippet",
      "remediation": "specific fix with code example"
    }
  ],
  "recommendations": ["..."],
  "remediation_steps": ["prioritised, actionable steps"],
  "compliance_notes": {
    "owasp_top10": [],
    "gdpr_concerns": [],
    "pci_concerns": []
  },
  "positive_findings": ["things done well"],
  "summary": "..."
}

Security scoring:
- 90-100: No significant issues
- 70-89: Minor issues only
- 50-69: Moderate issues requiring attention
- 30-49: High severity issues present
- 0-29: Critical issues — do not deploy
"""

    def get_capabilities(self) -> List[str]:
        return [
            "policy_compliance_check",
            "vulnerability_scanning",
            "access_control_review",
            "data_protection_analysis",
            "encryption_verification",
            "compliance_reporting",
        ]

    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the security audit using heuristics + LLM analysis."""
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"],
            )

        task = self.prepare_task(task)

        try:
            outputs = task.context.get("outputs", [])
            policy_level = task.context.get("policy_level", "standard")
            artifacts = task.context.get("artifacts", {})

            # Combine all code content for analysis
            code_content = self._collect_code_content(outputs, artifacts, task.description)

            # Run heuristic checks first
            heuristic_issues = self._run_heuristic_checks(code_content)

            # Search for current vulnerability patterns
            vuln_context = ""
            try:
                search_results = self._search(
                    "OWASP top 10 2024 vulnerability detection Python web application"
                )
                if search_results:
                    vuln_context = "\n".join([r.get("snippet", "") for r in search_results[:2]])
            except Exception:
                pass

            # Build prompt with actual code content
            code_excerpt = code_content[:3000] if len(code_content) > 3000 else code_content
            heuristic_summary = (
                "\n".join(
                    [f"- [{i['severity'].upper()}] {i['title']} at {i['location']}" for i in heuristic_issues]
                )
                if heuristic_issues
                else "No heuristic issues detected."
            )

            prompt = (
                f"Perform a security audit. Policy level: {policy_level}\n\n"
                f"CODE TO AUDIT:\n```\n{code_excerpt}\n```\n\n"
                f"Pre-detected heuristic issues (include these in your analysis):\n{heuristic_summary}\n\n"
                f"Security context:\n{vuln_context[:500]}\n\n"
                "Return a complete JSON security audit report."
            )

            audit = self._infer_json(
                prompt=prompt,
                fallback=self._fallback_audit(heuristic_issues, policy_level),
            )

            # Merge in any heuristic issues not captured by LLM
            existing_titles = {i.get("title", "") for i in audit.get("security_issues", [])}
            for h_issue in heuristic_issues:
                if h_issue["title"] not in existing_titles:
                    audit.setdefault("security_issues", []).append(h_issue)

            # Ensure required keys
            audit.setdefault("verdict", self._compute_verdict(audit.get("security_issues", [])))
            audit.setdefault("security_score", self._compute_score(audit.get("security_issues", [])))
            audit.setdefault("recommendations", [])
            audit.setdefault("remediation_steps", [])
            audit.setdefault("compliance_notes", {})
            audit.setdefault("positive_findings", [])
            audit.setdefault("summary", f"Security audit complete. Verdict: {audit['verdict'].upper()}")

            result = AgentResult(
                success=True,
                output=audit,
                metadata={
                    "artifacts_audited": len(outputs),
                    "policy_level": policy_level,
                    "verdict": audit.get("verdict"),
                    "issues_found": len(audit.get("security_issues", [])),
                    "security_score": audit.get("security_score"),
                },
            )
            task = self.complete_task(task, result)
            return result

        except Exception as e:
            self._log("error", f"Security audit failed: {e}")
            return AgentResult(success=False, output=None, errors=[str(e)])

    def verify(self, output: Any) -> VerificationResult:
        issues = []
        score = 1.0

        if not isinstance(output, dict):
            issues.append({"type": "invalid_type", "message": "Output must be a dict"})
            return VerificationResult(passed=False, issues=issues, score=0.0)

        if output.get("verdict") not in ("pass", "fail", "conditional"):
            issues.append({"type": "invalid_verdict", "message": "Invalid verdict value"})
            score -= 0.3
        if "security_issues" not in output:
            issues.append({"type": "missing_issues", "message": "Security issues list missing"})
            score -= 0.2
        if not output.get("recommendations"):
            issues.append({"type": "missing_recommendations", "message": "Recommendations missing"})
            score -= 0.15
        if not output.get("remediation_steps"):
            issues.append({"type": "missing_remediation", "message": "Remediation steps missing"})
            score -= 0.15

        passed = score >= 0.5
        return VerificationResult(passed=passed, issues=issues, score=max(0.0, score))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_code_content(
        self, outputs: List[Any], artifacts: Dict[str, Any], description: str
    ) -> str:
        """Collect all code content into a single string for analysis."""
        parts = []
        if description:
            parts.append(f"Task: {description}")
        for item in outputs:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for k, v in item.items():
                    if isinstance(v, str) and len(v) > 20:
                        parts.append(f"# {k}\n{v}")
        for k, v in artifacts.items():
            if isinstance(v, str) and len(v) > 20:
                parts.append(f"# {k}\n{v[:1000]}")
        return "\n\n".join(parts)

    def _run_heuristic_checks(self, code: str) -> List[Dict[str, Any]]:
        """Run pattern-based heuristic checks against code content."""
        issues = []
        for i, (pattern, severity, title) in enumerate(_HEURISTIC_CHECKS, 1):
            matches = list(re.finditer(pattern, code))
            for match in matches[:3]:  # Limit to 3 examples per pattern
                line_num = code[: match.start()].count("\n") + 1
                issues.append(
                    {
                        "id": f"HEUR-{i:03d}",
                        "severity": severity,
                        "category": "code_pattern",
                        "title": title,
                        "description": f"Pattern '{pattern}' detected",
                        "location": f"line {line_num}",
                        "evidence": match.group(0)[:80],
                        "remediation": "Review and remediate this security pattern",
                    }
                )
        return issues

    def _compute_verdict(self, issues: List[Dict]) -> str:
        severities = {i.get("severity", "info") for i in issues}
        if "critical" in severities:
            return "fail"
        if "high" in severities or len(issues) > 5:
            return "conditional"
        return "pass"

    def _compute_score(self, issues: List[Dict]) -> int:
        score = 100
        weights = {"critical": 30, "high": 15, "medium": 8, "low": 3, "info": 0}
        for issue in issues:
            score -= weights.get(issue.get("severity", "info"), 0)
        return max(0, score)

    def _fallback_audit(self, heuristic_issues: List[Dict], policy_level: str) -> Dict[str, Any]:
        verdict = self._compute_verdict(heuristic_issues)
        return {
            "verdict": verdict,
            "security_score": self._compute_score(heuristic_issues),
            "security_issues": heuristic_issues,
            "recommendations": [
                "Review all heuristically-detected patterns",
                "Add input validation to all user-facing endpoints",
                "Use environment variables for secrets",
                "Enable security linting in CI/CD (bandit, semgrep)",
                "Implement rate limiting and authentication",
            ],
            "remediation_steps": [
                "1. Fix any critical/high severity issues immediately",
                "2. Replace hardcoded secrets with environment variables",
                "3. Add parameterised queries for any SQL operations",
                "4. Enable HTTPS and validate SSL certificates",
                "5. Add security headers (CSP, HSTS, X-Frame-Options)",
            ],
            "compliance_notes": {
                "owasp_top10": ["Review against A01-A10 checklist"],
                "gdpr_concerns": ["Ensure PII is encrypted at rest and in transit"],
            },
            "positive_findings": [],
            "summary": f"Heuristic security scan complete. Verdict: {verdict.upper()}. Found {len(heuristic_issues)} potential issues.",
        }


_security_auditor_agent: Optional[SecurityAuditorAgent] = None


def get_security_auditor_agent(config: Optional[Dict[str, Any]] = None) -> SecurityAuditorAgent:
    global _security_auditor_agent
    if _security_auditor_agent is None:
        _security_auditor_agent = SecurityAuditorAgent(config)
    return _security_auditor_agent
