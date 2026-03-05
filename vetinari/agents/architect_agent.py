"""Vetinari Architect Agent — consolidated from Oracle (architecture, risk, debugging) + Cost Planner (cost analysis, model selection, efficiency)."""
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import AgentResult, AgentTask, AgentType, VerificationResult

import logging

logger = logging.getLogger(__name__)


class ArchitectAgent(BaseAgent):
    """Architect agent — architectural decisions, risk assessment, cost analysis, model selection.

    Absorbs:
        - OracleAgent: architecture design, risk assessment, debugging strategies
        - CostPlannerAgent: cost analysis, model selection, efficiency optimisation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.ARCHITECT, config)

    def get_system_prompt(self) -> str:
        return """You are Vetinari's Architect. You combine deep architectural expertise with
cost-aware decision making to design robust, efficient systems.

Your responsibilities:
1. Propose and evaluate architectural options with clear trade-offs
2. Identify risks (likelihood, impact, mitigation strategy) for each option
3. Analyse model selection, token costs, and compute efficiency
4. Recommend the most cost-effective implementation path
5. Provide concrete implementation guidelines
6. Debug architectural issues and propose corrective strategies

Output must include: architecture_vision, risks, recommended_guidelines, cost_analysis."""

    def get_capabilities(self) -> List[str]:
        return [
            "architecture_design",
            "risk_assessment",
            "tradeoff_analysis",
            "debugging_strategy",
            "cost_analysis",
            "model_selection",
            "efficiency_optimisation",
        ]

    def execute(self, task: AgentTask) -> AgentResult:
        """Execute task, delegating to OracleAgent or CostPlannerAgent based on keywords."""
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"],
            )

        task = self.prepare_task(task)
        desc = (task.description or "").lower()

        try:
            if any(kw in desc for kw in ("cost", "price", "budget", "model select", "token", "efficiency", "cheap", "expensive")):
                result = self._delegate_to_cost_planner(task)
            elif any(kw in desc for kw in ("architect", "design", "risk", "debug", "trade-off", "tradeoff", "structure", "pattern", "guideline")):
                result = self._delegate_to_oracle(task)
            else:
                result = self._execute_default(task)

            self.complete_task(task, result)
            return result

        except Exception as e:
            self._log("error", f"ArchitectAgent execution failed: {e}")
            return AgentResult(success=False, output=None, errors=[str(e)])

    def _delegate_to_oracle(self, task: AgentTask) -> AgentResult:
        from vetinari.agents.oracle_agent import OracleAgent
        agent = OracleAgent(self._config)
        agent._adapter_manager = self._adapter_manager
        agent._web_search = self._web_search
        agent._initialized = self._initialized
        return agent.execute(task)

    def _delegate_to_cost_planner(self, task: AgentTask) -> AgentResult:
        from vetinari.agents.cost_planner_agent import CostPlannerAgent
        agent = CostPlannerAgent(self._config)
        agent._adapter_manager = self._adapter_manager
        agent._web_search = self._web_search
        agent._initialized = self._initialized
        return agent.execute(task)

    def _execute_default(self, task: AgentTask) -> AgentResult:
        goal = task.prompt or task.description
        prompt = f"""Provide architectural guidance and cost analysis for:\n\n{goal}\n\nInclude architecture_vision, risks, recommended_guidelines, and cost_analysis in JSON."""
        output = self._infer_json(prompt)
        if output is None:
            output = {
                "architecture_vision": f"Modular architecture for: {goal}",
                "risks": [{"risk": "Scope creep", "likelihood": 0.5, "impact": 0.6, "mitigation": "Define MVP clearly"}],
                "recommended_guidelines": ["Use CI/CD", "Write tests", "Document decisions"],
                "cost_analysis": {"estimated_tokens": "unknown", "recommended_model": self.default_model},
            }
        return AgentResult(success=True, output=output, metadata={"mode": "default"})

    def verify(self, output: Any) -> VerificationResult:
        issues = []
        score = 1.0
        if not isinstance(output, dict):
            return VerificationResult(passed=False, issues=[{"type": "invalid_type", "message": "Output must be a dict"}], score=0.5)
        if "architecture_vision" not in output:
            issues.append({"type": "missing_vision", "message": "No architecture vision"})
            score -= 0.3
        if not output.get("risks"):
            issues.append({"type": "no_risks", "message": "No risks identified"})
            score -= 0.2
        if not output.get("recommended_guidelines"):
            issues.append({"type": "no_guidelines", "message": "No guidelines"})
            score -= 0.2
        return VerificationResult(passed=score >= 0.5, issues=issues, score=max(0.0, score))


def get_architect_agent(config: Optional[Dict[str, Any]] = None) -> ArchitectAgent:
    return ArchitectAgent(config)
