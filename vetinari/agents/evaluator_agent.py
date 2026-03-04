"""
Vetinari Evaluator Agent

The Evaluator agent is responsible for code quality, security checks, 
and testability evaluation.
"""

from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult
)


class EvaluatorAgent(BaseAgent):
    """Evaluator agent - code quality, security, and testability evaluation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.EVALUATOR, config)
        self._quality_threshold = self._config.get("quality_threshold", 0.7)
        
    def get_system_prompt(self) -> str:
        return """You are Vetinari's Evaluator. Evaluate outputs for quality, security, and testability. 
Provide a pass/fail verdict and a list of actionable improvements with rationale.

You must:
1. Check code quality (style, design, documentation)
2. Identify security vulnerabilities
3. Assess testability and coverage
4. Evaluate architecture and design patterns
5. Provide specific improvement suggestions
6. Assign quality scores

Output format must include verdict (pass/fail), quality_score, findings, security_issues, and improvements."""
    
    def get_capabilities(self) -> List[str]:
        return [
            "code_quality_analysis",
            "security_evaluation",
            "testability_assessment",
            "architecture_review",
            "documentation_check",
            "performance_analysis"
        ]
    
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the evaluation task.
        
        Args:
            task: The task containing artifacts to evaluate
            
        Returns:
            AgentResult containing the evaluation results
        """
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"]
            )
        
        task = self.prepare_task(task)
        
        try:
            artifacts = task.context.get("artifacts", [])
            evaluation_focus = task.context.get("focus", "all")
            
            # Perform evaluation (simulated - in production would use actual code analysis tools)
            evaluation = self._evaluate_artifacts(artifacts, evaluation_focus)
            
            return AgentResult(
                success=True,
                output=evaluation,
                metadata={
                    "artifacts_evaluated": len(artifacts),
                    "focus": evaluation_focus,
                    "verdict": evaluation.get("verdict")
                }
            )
            
        except Exception as e:
            self._log("error", f"Evaluation failed: {str(e)}")
            return AgentResult(
                success=False,
                output=None,
                errors=[str(e)]
            )
    
    def verify(self, output: Any) -> VerificationResult:
        """Verify the evaluation output meets quality standards.
        
        Args:
            output: The evaluation to verify
            
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
        verdict = output.get("verdict")
        if verdict not in ["pass", "fail"]:
            issues.append({"type": "invalid_verdict", "message": "Verdict must be 'pass' or 'fail'"})
            score -= 0.3
        
        # Check for quality score
        if "quality_score" not in output:
            issues.append({"type": "missing_score", "message": "Quality score is missing"})
            score -= 0.2
        
        # Check for findings
        if not output.get("findings"):
            issues.append({"type": "no_findings", "message": "No evaluation findings"})
            score -= 0.2
        
        # Check for improvements
        if not output.get("improvements"):
            issues.append({"type": "no_improvements", "message": "No improvement suggestions"})
            score -= 0.1
        
        passed = score >= 0.5
        return VerificationResult(passed=passed, issues=issues, score=max(0, score))
    
    def _evaluate_artifacts(self, artifacts: List[str], focus: str) -> Dict[str, Any]:
        """Evaluate provided artifacts.
        
        Args:
            artifacts: List of artifacts to evaluate
            focus: Focus area (code, security, testability, or all)
            
        Returns:
            Dictionary containing evaluation results
        """
        # This is a simplified implementation
        # In production, this would use actual code analysis tools
        
        findings = []
        security_issues = []
        improvements = []
        
        # Simulate code quality findings
        if focus in ["code", "all"]:
            findings.extend([
                {"area": "code_style", "message": "Code follows PEP 8 guidelines", "score": 0.9},
                {"area": "design_patterns", "message": "Good use of design patterns", "score": 0.85}
            ])
            improvements.append({
                "area": "code_style",
                "issue": "Some lines exceed 88 characters",
                "suggestion": "Apply black formatter for consistency"
            })
        
        # Simulate security evaluation
        if focus in ["security", "all"]:
            security_issues.append({
                "severity": "low",
                "type": "hardcoded_secret",
                "location": "config.py:42",
                "recommendation": "Use environment variables for secrets"
            })
            improvements.append({
                "area": "security",
                "issue": "Potential SQL injection vulnerability",
                "suggestion": "Use parameterized queries"
            })
        
        # Simulate testability assessment
        if focus in ["testability", "all"]:
            findings.append({"area": "test_coverage", "message": "Test coverage at 75%", "score": 0.75})
            improvements.append({
                "area": "testability",
                "issue": "Some functions are not testable",
                "suggestion": "Refactor functions to reduce coupling"
            })
        
        # Calculate overall quality score
        quality_scores = [f.get("score", 0.7) for f in findings]
        overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.7
        
        verdict = "pass" if overall_quality >= self._quality_threshold else "fail"
        
        return {
            "verdict": verdict,
            "quality_score": round(overall_quality, 2),
            "findings": findings,
            "security_issues": security_issues,
            "improvements": improvements,
            "summary": f"Evaluation complete. Overall quality: {round(overall_quality*100)}%. Verdict: {verdict.upper()}"
        }


# Singleton instance
_evaluator_agent: Optional[EvaluatorAgent] = None


def get_evaluator_agent(config: Optional[Dict[str, Any]] = None) -> EvaluatorAgent:
    """Get the singleton Evaluator agent instance."""
    global _evaluator_agent
    if _evaluator_agent is None:
        _evaluator_agent = EvaluatorAgent(config)
    return _evaluator_agent
