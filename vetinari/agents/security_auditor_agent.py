"""
Vetinari Security Auditor Agent

The Security Auditor agent is responsible for enforcing safety, policy compliance,
and vulnerability checks.
"""

from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult
)


class SecurityAuditorAgent(BaseAgent):
    """Security Auditor agent - safety, policy compliance, vulnerability checks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.SECURITY_AUDITOR, config)
        
    def get_system_prompt(self) -> str:
        return """You are Vetinari's Security Auditor. Review plans, artifacts, and code for policy 
compliance and safety. Flag policy breaches and provide remediation steps.

You must:
1. Check for policy compliance
2. Identify security vulnerabilities
3. Review data handling practices
4. Assess access control patterns
5. Verify encryption usage
6. Flag sensitive data exposure

Output format must include verdict, issues (with severity), recommendations, and remediation_steps."""
    
    def get_capabilities(self) -> List[str]:
        return [
            "policy_compliance_check",
            "vulnerability_scanning",
            "access_control_review",
            "data_protection_analysis",
            "encryption_verification",
            "compliance_reporting"
        ]
    
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the security audit task.
        
        Args:
            task: The task containing outputs to audit
            
        Returns:
            AgentResult containing the audit results
        """
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"]
            )
        
        task = self.prepare_task(task)
        
        try:
            outputs = task.context.get("outputs", [])
            policy_level = task.context.get("policy_level", "standard")
            
            # Perform security audit (simulated - in production would use actual tools)
            audit = self._audit_security(outputs, policy_level)
            
            return AgentResult(
                success=True,
                output=audit,
                metadata={
                    "artifacts_audited": len(outputs),
                    "policy_level": policy_level,
                    "verdict": audit.get("verdict")
                }
            )
            
        except Exception as e:
            self._log("error", f"Security audit failed: {str(e)}")
            return AgentResult(
                success=False,
                output=None,
                errors=[str(e)]
            )
    
    def verify(self, output: Any) -> VerificationResult:
        """Verify the audit output meets quality standards.
        
        Args:
            output: The audit to verify
            
        Returns:
            VerificationResult with pass/fail status
        """
        issues = []
        score = 1.0
        
        if not isinstance(output, dict):
            issues.append({"type": "invalid_type", "message": "Output must be a dict"})
            score -= 0.5
            return VerificationResult(passed=False, issues=issues, score=score)
        
        # Check for verdict
        if output.get("verdict") not in ["pass", "fail", "conditional"]:
            issues.append({"type": "invalid_verdict", "message": "Invalid verdict value"})
            score -= 0.3
        
        # Check for issues
        if "security_issues" not in output:
            issues.append({"type": "missing_issues", "message": "Security issues list missing"})
            score -= 0.2
        
        # Check for recommendations
        if not output.get("recommendations"):
            issues.append({"type": "missing_recommendations", "message": "Recommendations missing"})
            score -= 0.15
        
        # Check for remediation steps
        if not output.get("remediation_steps"):
            issues.append({"type": "missing_remediation", "message": "Remediation steps missing"})
            score -= 0.15
        
        passed = score >= 0.5
        return VerificationResult(passed=passed, issues=issues, score=max(0, score))
    
    def _audit_security(self, outputs: List[str], policy_level: str) -> Dict[str, Any]:
        """Perform security audit on outputs.
        
        Args:
            outputs: List of outputs to audit
            policy_level: Policy compliance level (standard, strict, relaxed)
            
        Returns:
            Dictionary containing audit results
        """
        # This is a simplified implementation
        # In production, this would use actual security scanning tools
        
        security_issues = []
        recommendations = []
        remediation_steps = []
        
        # Simulate security scanning
        security_issues.append({
            "severity": "low",
            "issue": "Hardcoded configuration values",
            "location": "config.py:10-15",
            "details": "API keys should not be hardcoded in source"
        })
        
        security_issues.append({
            "severity": "medium",
            "issue": "Missing input validation",
            "location": "api.py:45",
            "details": "User input not validated before processing"
        })
        
        # Add recommendations based on policy level
        if policy_level == "strict":
            security_issues.append({
                "severity": "low",
                "issue": "Insufficient logging",
                "location": "security_logging",
                "details": "Audit trails not comprehensive enough"
            })
        
        # Generate remediation steps
        remediation_steps = [
            "Move hardcoded config values to environment variables",
            "Add input validation to all API endpoints",
            "Implement comprehensive security logging",
            "Enable rate limiting on API endpoints",
            "Add CORS policy configuration"
        ]
        
        # Determine verdict
        critical_issues = [i for i in security_issues if i.get("severity") == "critical"]
        verdict = "fail" if critical_issues else ("conditional" if len(security_issues) > 2 else "pass")
        
        # Generate recommendations
        recommendations = [
            "Review and implement security best practices",
            "Add security testing to CI/CD pipeline",
            "Implement regular security audits",
            "Provide security training for developers"
        ]
        
        return {
            "verdict": verdict,
            "security_issues": security_issues,
            "recommendations": recommendations,
            "remediation_steps": remediation_steps,
            "policy_level": policy_level,
            "summary": f"Security audit completed. Verdict: {verdict.upper()}. Found {len(security_issues)} issues."
        }


# Singleton instance
_security_auditor_agent: Optional[SecurityAuditorAgent] = None


def get_security_auditor_agent(config: Optional[Dict[str, Any]] = None) -> SecurityAuditorAgent:
    """Get the singleton Security Auditor agent instance."""
    global _security_auditor_agent
    if _security_auditor_agent is None:
        _security_auditor_agent = SecurityAuditorAgent(config)
    return _security_auditor_agent
