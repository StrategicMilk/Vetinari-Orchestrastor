"""Vetinari Resilience Agent — consolidated from Error Recovery + Image Generator."""
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import AgentResult, AgentTask, AgentType, VerificationResult

import logging

logger = logging.getLogger(__name__)


class ResilienceAgent(BaseAgent):
    """Resilience agent — failure analysis, retry strategies, fallback planning, and asset generation.

    Absorbs:
        - ErrorRecoveryAgent: failure analysis, retry strategies, circuit breaking, fallback planning
        - ImageGeneratorAgent: logo, icon, UI mockup, diagram, and asset generation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.RESILIENCE, config)

    def get_system_prompt(self) -> str:
        return """You are Vetinari's Resilience agent. You handle both system failure recovery
and creative asset generation to keep projects robust and visually complete.

Your responsibilities:
1. Analyse failures and propose retry strategies with backoff policies
2. Design circuit breakers and fallback mechanisms
3. Create failure recovery playbooks and runbooks
4. Generate images, logos, icons, UI mockups, and diagrams via Stable Diffusion or SVG
5. Produce prompt specifications for visual assets when direct generation is unavailable

Output must include recovery_strategy or asset_specification depending on task type."""

    def get_capabilities(self) -> List[str]:
        return [
            "failure_analysis",
            "retry_strategy",
            "circuit_breaking",
            "fallback_planning",
            "image_generation",
            "logo_creation",
            "diagram_generation",
            "asset_specification",
        ]

    def execute(self, task: AgentTask) -> AgentResult:
        """Execute task, delegating to error recovery or image generator sub-agent."""
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"],
            )

        task = self.prepare_task(task)
        desc = (task.description or "").lower()

        try:
            if any(kw in desc for kw in ("image", "logo", "icon", "mockup", "diagram", "visual", "asset", "svg", "png", "illustration", "generate image")):
                result = self._delegate_to_image_generator(task)
            elif any(kw in desc for kw in ("error", "failure", "recover", "retry", "fallback", "circuit", "resilience", "exception", "crash", "outage")):
                result = self._delegate_to_error_recovery(task)
            else:
                result = self._execute_default(task)

            self.complete_task(task, result)
            return result

        except Exception as e:
            self._log("error", f"ResilienceAgent execution failed: {e}")
            return AgentResult(success=False, output=None, errors=[str(e)])

    def _delegate_to_error_recovery(self, task: AgentTask) -> AgentResult:
        from vetinari.agents.error_recovery_agent import ErrorRecoveryAgent
        agent = ErrorRecoveryAgent(self._config)
        agent._adapter_manager = self._adapter_manager
        agent._web_search = self._web_search
        agent._initialized = self._initialized
        return agent.execute(task)

    def _delegate_to_image_generator(self, task: AgentTask) -> AgentResult:
        from vetinari.agents.image_generator_agent import ImageGeneratorAgent
        agent = ImageGeneratorAgent(self._config)
        agent._adapter_manager = self._adapter_manager
        agent._web_search = self._web_search
        agent._initialized = self._initialized
        return agent.execute(task)

    def _execute_default(self, task: AgentTask) -> AgentResult:
        goal = task.prompt or task.description
        prompt = f"""Provide resilience strategy and asset recommendations for:\n\n{goal}\n\nReturn JSON with recovery_strategy and asset_specification fields."""
        output = self._infer_json(prompt)
        if output is None:
            output = {
                "recovery_strategy": {
                    "approach": "exponential_backoff",
                    "max_retries": 3,
                    "fallback": f"Graceful degradation for: {goal}",
                },
                "asset_specification": {"type": "diagram", "description": f"Architecture diagram for {goal}"},
            }
        return AgentResult(success=True, output=output, metadata={"mode": "default"})

    def verify(self, output: Any) -> VerificationResult:
        issues = []
        score = 1.0
        if not isinstance(output, dict):
            return VerificationResult(passed=False, issues=[{"type": "invalid_type", "message": "Output must be a dict"}], score=0.5)
        has_content = any(output.get(k) for k in ("recovery_strategy", "asset_specification", "image_url", "svg", "retry_policy", "fallback_plan"))
        if not has_content:
            issues.append({"type": "no_content", "message": "No recovery strategy or asset specification found"})
            score -= 0.5
        return VerificationResult(passed=score >= 0.5, issues=issues, score=max(0.0, score))


def get_resilience_agent(config: Optional[Dict[str, Any]] = None) -> ResilienceAgent:
    return ResilienceAgent(config)
