"""Vetinari Meta Agent — consolidated from Improvement + Experimentation Manager."""
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import AgentResult, AgentTask, AgentType, VerificationResult

import logging

logger = logging.getLogger(__name__)


class MetaAgent(BaseAgent):
    """Meta agent — system performance analysis, optimisation recommendations, and experiment tracking.

    Absorbs:
        - ImprovementAgent: meta-analysis of system performance, optimisation recommendations
        - ExperimentationManagerAgent: experiment tracking, versioning, reproducibility
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.META, config)

    def get_system_prompt(self) -> str:
        return """You are Vetinari's Meta agent. You analyse system-wide performance and manage
experiments to continuously improve the orchestration system itself.

Your responsibilities:
1. Review agent performance metrics and identify bottlenecks
2. Recommend prompt improvements, model swaps, and workflow optimisations
3. Track experiments with hypothesis, methodology, and results
4. Ensure experiment reproducibility through versioning and parameter logging
5. Synthesise learnings into actionable system improvement plans
6. A/B test configurations and report statistical significance

Output must include improvement_recommendations or experiment_results depending on task type."""

    def get_capabilities(self) -> List[str]:
        return [
            "performance_analysis",
            "optimisation_recommendations",
            "prompt_improvement",
            "experiment_tracking",
            "experiment_versioning",
            "reproducibility_management",
            "ab_testing",
            "metrics_synthesis",
        ]

    def execute(self, task: AgentTask) -> AgentResult:
        """Execute task, delegating to improvement or experimentation manager sub-agent."""
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"],
            )

        task = self.prepare_task(task)
        desc = (task.description or "").lower()

        try:
            if any(kw in desc for kw in ("experiment", "ab test", "a/b", "hypothesis", "trial", "run", "track", "reproduce", "version", "parameter")):
                result = self._delegate_to_experimentation_manager(task)
            elif any(kw in desc for kw in ("improve", "optimise", "optimize", "performance", "metric", "bottleneck", "analyse system", "analyze system", "recommend")):
                result = self._delegate_to_improvement(task)
            else:
                result = self._execute_default(task)

            self.complete_task(task, result)
            return result

        except Exception as e:
            self._log("error", f"MetaAgent execution failed: {e}")
            return AgentResult(success=False, output=None, errors=[str(e)])

    def _delegate_to_improvement(self, task: AgentTask) -> AgentResult:
        from vetinari.agents.improvement_agent import ImprovementAgent
        agent = ImprovementAgent(self._config)
        agent._adapter_manager = self._adapter_manager
        agent._web_search = self._web_search
        agent._initialized = self._initialized
        return agent.execute(task)

    def _delegate_to_experimentation_manager(self, task: AgentTask) -> AgentResult:
        from vetinari.agents.experimentation_manager_agent import ExperimentationManagerAgent
        agent = ExperimentationManagerAgent(self._config)
        agent._adapter_manager = self._adapter_manager
        agent._web_search = self._web_search
        agent._initialized = self._initialized
        return agent.execute(task)

    def _execute_default(self, task: AgentTask) -> AgentResult:
        goal = task.prompt or task.description
        prompt = f"""Provide system improvement analysis and experiment tracking for:\n\n{goal}\n\nReturn JSON with improvement_recommendations and experiment_results fields."""
        output = self._infer_json(prompt)
        if output is None:
            output = {
                "improvement_recommendations": [
                    {"area": "performance", "suggestion": f"Profile and optimise critical path for: {goal}", "priority": "high"},
                ],
                "experiment_results": {"status": "no_active_experiments", "next_steps": ["Define baseline metrics", "Design hypothesis"]},
            }
        return AgentResult(success=True, output=output, metadata={"mode": "default"})

    def verify(self, output: Any) -> VerificationResult:
        issues = []
        score = 1.0
        if not isinstance(output, dict):
            return VerificationResult(passed=False, issues=[{"type": "invalid_type", "message": "Output must be a dict"}], score=0.5)
        has_content = any(output.get(k) for k in ("improvement_recommendations", "experiment_results", "recommendations", "experiments", "metrics"))
        if not has_content:
            issues.append({"type": "no_content", "message": "No improvement recommendations or experiment results found"})
            score -= 0.5
        return VerificationResult(passed=score >= 0.5, issues=issues, score=max(0.0, score))


def get_meta_agent(config: Optional[Dict[str, Any]] = None) -> MetaAgent:
    return MetaAgent(config)
