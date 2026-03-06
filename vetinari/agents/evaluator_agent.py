"""
Vetinari Evaluator Agent

The Evaluator agent is responsible for code quality, security checks,
and testability evaluation. Supports verification modes (Task 26) for
quality gate checks between pipeline stages.
"""

import logging
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult
)
from vetinari.validation.quality_gates import (
    GateCheckResult,
    GateResult,
    QualityGateConfig,
    QualityGateRunner,
    VerificationMode,
)

logger = logging.getLogger(__name__)


class EvaluatorAgent(BaseAgent):
    """Evaluator agent - code quality, security, and testability evaluation.

    Supports verification modes (Task 26) for quality gate checks. When a
    ``mode`` key is present in ``task.context``, the agent routes to the
    appropriate quality gate check instead of the default LLM evaluation.

    Supported modes (from VerificationMode):
        - verify_quality: Style, complexity, best practices
        - security: Security pattern scanning
        - verify_coverage: Test existence and pass rate
        - verify_architecture: Consistency with project architecture
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.EVALUATOR, config)
        self._quality_threshold = self._config.get("quality_threshold", 0.7)
        self._gate_runner = QualityGateRunner()

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
            "performance_analysis",
            "quality_gate_verification",
        ]

    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the evaluation task.

        When ``task.context["mode"]`` is set to a valid VerificationMode
        value string, the agent runs the corresponding quality gate check
        instead of the default LLM evaluation. The gate artifacts are
        taken from ``task.context["artifacts"]`` (expected to be a dict).

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

        # Check for verification mode routing (Task 26)
        mode_str = task.context.get("mode")
        if mode_str:
            return self._execute_verification_mode(task, mode_str)

        try:
            artifacts = task.context.get("artifacts", [])
            evaluation_focus = task.context.get("focus", "all")

            # Perform evaluation via LLM + heuristic analysis
            evaluation = self._evaluate_artifacts(artifacts, evaluation_focus)

            result = AgentResult(
                success=True,
                output=evaluation,
                metadata={
                    "artifacts_evaluated": len(artifacts),
                    "focus": evaluation_focus,
                    "verdict": evaluation.get("verdict")
                }
            )
            self.complete_task(task, result)
            return result

        except Exception as e:
            self._log("error", f"Evaluation failed: {str(e)}")
            return AgentResult(
                success=False,
                output=None,
                errors=[str(e)]
            )

    def _execute_verification_mode(self, task: AgentTask, mode_str: str) -> AgentResult:
        """Route to the appropriate quality gate check based on mode.

        Args:
            task: The agent task with context containing artifacts.
            mode_str: The verification mode string (e.g. "verify_quality").

        Returns:
            AgentResult wrapping the GateCheckResult.
        """
        try:
            mode = VerificationMode(mode_str)
        except ValueError:
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Unknown verification mode: {mode_str}. "
                        f"Valid modes: {[m.value for m in VerificationMode]}"],
            )

        # Build gate config from task context
        min_score = task.context.get("min_score", self._quality_threshold)
        gate_config = QualityGateConfig(
            name=f"{mode.value}_gate",
            mode=mode,
            min_score=min_score,
            required=task.context.get("required", True),
            auto_fix=task.context.get("auto_fix", False),
        )

        # Artifacts should be a dict for the gate runner
        artifacts = task.context.get("artifacts", {})
        if isinstance(artifacts, list):
            # Convert list of strings to a dict with "code" key
            artifacts = {"code": "\n".join(str(a) for a in artifacts)}

        try:
            dispatch = {
                VerificationMode.VERIFY_QUALITY: self._gate_runner.check_quality,
                VerificationMode.SECURITY: self._gate_runner.check_security,
                VerificationMode.VERIFY_COVERAGE: self._gate_runner.check_coverage,
                VerificationMode.VERIFY_ARCHITECTURE: self._gate_runner.check_architecture,
            }
            handler = dispatch[mode]
            gate_result: GateCheckResult = handler(artifacts, gate_config)

            # Convert GateResult to evaluation-style output
            verdict = "pass" if gate_result.result == GateResult.PASSED else "fail"
            output = {
                "verdict": verdict,
                "quality_score": gate_result.score,
                "mode": mode.value,
                "gate_result": gate_result.result.value,
                "findings": [
                    {"area": issue.get("category", "general"), "message": issue.get("message", ""), "score": gate_result.score}
                    for issue in gate_result.issues
                ],
                "security_issues": [
                    issue for issue in gate_result.issues
                    if issue.get("category") in ("security", "secrets")
                ],
                "improvements": [
                    {"area": "general", "issue": s, "suggestion": s}
                    for s in gate_result.suggestions
                ],
                "summary": f"Verification mode '{mode.value}' completed with score {gate_result.score}",
            }

            result = AgentResult(
                success=True,
                output=output,
                metadata={
                    "mode": mode.value,
                    "gate_result": gate_result.to_dict(),
                    "verdict": verdict,
                },
            )
            self.complete_task(task, result)
            return result

        except Exception as e:
            self._log("error", f"Verification mode '{mode_str}' failed: {e}")
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Verification mode '{mode_str}' failed: {e}"],
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
        """Evaluate provided artifacts using LLM analysis.

        Uses the LLM to perform thorough code quality, security, and
        testability assessment. Falls back to default scores if the LLM
        is unavailable.
        """
        import json as _json

        artifact_content = "\n---\n".join(str(a) for a in artifacts) if artifacts else "(no artifacts provided)"

        prompt = f"""Evaluate the following artifact(s) for quality.

EVALUATION FOCUS: {focus}
QUALITY THRESHOLD: {self._quality_threshold}

ARTIFACTS:
{artifact_content[:3000]}

Produce a JSON evaluation report with this exact structure:
{{
  "verdict": "pass" or "fail",
  "quality_score": 0.0-1.0,
  "findings": [
    {{"area": "code_style|design_patterns|test_coverage|architecture", "message": "...", "score": 0.0-1.0}}
  ],
  "security_issues": [
    {{"severity": "low|medium|high|critical", "type": "...", "location": "...", "recommendation": "..."}}
  ],
  "improvements": [
    {{"area": "...", "issue": "...", "suggestion": "..."}}
  ],
  "summary": "Brief evaluation summary"
}}

Be specific and actionable. Do not make up issues that are not evidenced by the artifacts."""

        result = self._infer_json(prompt)

        if result and isinstance(result, dict) and "verdict" in result:
            quality_score = float(result.get("quality_score", 0.7))
            # Respect LLM's qualitative verdict first; only override if score is
            # clearly below threshold (LLM may flag critical issues with high score)
            llm_verdict = result.get("verdict", "").lower()
            has_critical = any(
                si.get("severity") == "critical"
                for si in result.get("security_issues", [])
            )
            if has_critical:
                # Critical security issues always fail regardless of score
                result["verdict"] = "fail"
            elif llm_verdict in ("pass", "fail"):
                # If LLM gave a clear verdict, respect it but reconcile with score
                if quality_score < self._quality_threshold:
                    result["verdict"] = "fail"
                # else keep LLM's verdict (it may be "fail" for qualitative reasons)
            else:
                result["verdict"] = "pass" if quality_score >= self._quality_threshold else "fail"
            result["quality_score"] = round(quality_score, 2)
            return result

        # Fallback: return inconclusive result (never rubber-stamp as passing)
        self._log("warning", "LLM evaluation unavailable, returning inconclusive assessment")
        return {
            "verdict": "inconclusive",
            "quality_score": 0.5,
            "findings": [{"area": "general", "message": "LLM unavailable — manual review required", "score": 0.5}],
            "security_issues": [],
            "improvements": [{"area": "general", "issue": "Manual review required", "suggestion": "Review artifacts manually before proceeding"}],
            "summary": "Inconclusive evaluation (LLM unavailable) — manual review required"
        }


# Singleton instance
_evaluator_agent: Optional[EvaluatorAgent] = None


def get_evaluator_agent(config: Optional[Dict[str, Any]] = None) -> EvaluatorAgent:
    """Get the singleton Evaluator agent instance."""
    global _evaluator_agent
    if _evaluator_agent is None:
        _evaluator_agent = EvaluatorAgent(config)
    return _evaluator_agent
