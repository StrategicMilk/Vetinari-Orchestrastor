"""Vetinari Tester Agent — consolidated from Test Automation + Security Auditor + Evaluator."""
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import AgentResult, AgentTask, AgentType, VerificationResult

import logging

logger = logging.getLogger(__name__)


class TesterAgent(BaseAgent):
    """Tester agent — test generation, security audits, and code quality evaluation.

    Absorbs:
        - TestAutomationAgent: test generation, coverage improvement, validation
        - SecurityAuditorAgent: vulnerability checks, policy compliance, safety
        - EvaluatorAgent: code quality, testability, evaluation scoring
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.TESTER, config)

    def get_system_prompt(self) -> str:
        return """You are Vetinari's Tester. You combine automated test generation, security
auditing, and code quality evaluation into a single unified capability.

Your responsibilities:
1. Generate comprehensive test suites (unit, integration, e2e) with high coverage
2. Identify security vulnerabilities, injection risks, and compliance issues
3. Evaluate code quality, maintainability, and testability scores
4. Propose concrete fixes for identified issues
5. Report coverage gaps and prioritise remediation

Output must include: test_cases or security_findings or evaluation_score depending on task type."""

    def get_capabilities(self) -> List[str]:
        return [
            "test_generation",
            "coverage_analysis",
            "security_audit",
            "vulnerability_detection",
            "code_quality_evaluation",
            "testability_scoring",
        ]

    def execute(self, task: AgentTask) -> AgentResult:
        """Execute task, delegating to test automation, security auditor, or evaluator."""
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"],
            )

        task = self.prepare_task(task)
        desc = (task.description or "").lower()

        try:
            if any(kw in desc for kw in ("security", "vulnerability", "audit", "exploit", "injection", "xss", "sqli", "compliance", "owasp")):
                result = self._delegate_to_security_auditor(task)
            elif any(kw in desc for kw in ("evaluate", "quality", "score", "metric", "rating", "assess quality", "code review")):
                result = self._delegate_to_evaluator(task)
            elif any(kw in desc for kw in ("test", "coverage", "unit", "integration", "e2e", "pytest", "unittest", "spec")):
                result = self._delegate_to_test_automation(task)
            else:
                result = self._execute_default(task)

            self.complete_task(task, result)
            return result

        except Exception as e:
            self._log("error", f"TesterAgent execution failed: {e}")
            return AgentResult(success=False, output=None, errors=[str(e)])

    def _delegate_to_test_automation(self, task: AgentTask) -> AgentResult:
        from vetinari.agents.test_automation_agent import TestAutomationAgent
        agent = TestAutomationAgent(self._config)
        agent._adapter_manager = self._adapter_manager
        agent._web_search = self._web_search
        agent._initialized = self._initialized
        return agent.execute(task)

    def _delegate_to_security_auditor(self, task: AgentTask) -> AgentResult:
        from vetinari.agents.security_auditor_agent import SecurityAuditorAgent
        agent = SecurityAuditorAgent(self._config)
        agent._adapter_manager = self._adapter_manager
        agent._web_search = self._web_search
        agent._initialized = self._initialized
        return agent.execute(task)

    def _delegate_to_evaluator(self, task: AgentTask) -> AgentResult:
        from vetinari.agents.evaluator_agent import EvaluatorAgent
        agent = EvaluatorAgent(self._config)
        agent._adapter_manager = self._adapter_manager
        agent._web_search = self._web_search
        agent._initialized = self._initialized
        return agent.execute(task)

    def _execute_default(self, task: AgentTask) -> AgentResult:
        goal = task.prompt or task.description
        prompt = f"""Generate tests, security findings, and quality evaluation for:\n\n{goal}\n\nReturn JSON with test_cases, security_findings, and evaluation_score."""
        output = self._infer_json(prompt)
        if output is None:
            output = {
                "test_cases": [{"name": "test_basic", "type": "unit", "description": f"Basic test for {goal}"}],
                "security_findings": [],
                "evaluation_score": 0.0,
                "fallback": True,
            }
        return AgentResult(success=True, output=output, metadata={"mode": "default"})

    def verify(self, output: Any) -> VerificationResult:
        issues = []
        score = 1.0
        if not isinstance(output, dict):
            return VerificationResult(passed=False, issues=[{"type": "invalid_type", "message": "Output must be a dict"}], score=0.5)
        has_content = any(output.get(k) for k in ("test_cases", "security_findings", "evaluation_score"))
        if not has_content:
            issues.append({"type": "no_content", "message": "No test cases, findings, or scores"})
            score -= 0.5
        return VerificationResult(passed=score >= 0.5, issues=issues, score=max(0.0, score))


def get_tester_agent(config: Optional[Dict[str, Any]] = None) -> TesterAgent:
    return TesterAgent(config)
