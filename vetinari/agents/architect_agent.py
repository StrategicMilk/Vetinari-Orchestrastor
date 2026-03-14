"""Vetinari Architect Agent — consolidated from Oracle (architecture, risk, debugging) + Cost Planner (cost analysis, model selection, efficiency)."""

from __future__ import annotations

import logging
from typing import Any

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import AgentResult, AgentTask, VerificationResult
from vetinari.types import AgentType

logger = logging.getLogger(__name__)


class ArchitectAgent(BaseAgent):
    """Architect agent — architectural decisions, risk assessment, cost analysis, model selection.

    Absorbs:
        - OracleAgent: architecture design, risk assessment, debugging strategies
        - CostPlannerAgent: cost analysis, model selection, efficiency optimisation
    """

    def __init__(self, config: dict[str, Any] | None = None):
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

    def get_capabilities(self) -> list[str]:
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
        """Execute task, delegating to OracleAgent or CostPlannerAgent based on keywords.

        Returns:
            The AgentResult result.
        """
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"],
            )

        task = self.prepare_task(task)
        desc = (task.description or "").lower()

        try:
            if any(kw in desc for kw in ("suggest", "improve", "enhancement", "recommendation", "creative")):
                result = self._generate_suggestions(task)
            elif any(
                kw in desc
                for kw in ("cost", "price", "budget", "model select", "token", "efficiency", "cheap", "expensive")
            ):
                result = self._delegate_to_cost_planner(task)
            elif any(
                kw in desc
                for kw in (
                    "architect",
                    "design",
                    "risk",
                    "debug",
                    "trade-off",
                    "tradeoff",
                    "structure",
                    "pattern",
                    "guideline",
                )
            ):
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

    def _generate_suggestions(self, task: AgentTask) -> AgentResult:
        """Generate creative improvement suggestions based on project context.

        Called at strategic insertion points (pre-decomposition, mid-execution,
        post-execution) to propose enhancements the user might want.
        """
        goal = task.prompt or task.description
        context = task.context or {}
        completed_outputs = context.get("completed_outputs", [])

        prompt = f"""Based on this project context, suggest 3-5 improvements:

Goal: {goal}

{("Completed so far: " + ", ".join(str(o)[:100] for o in completed_outputs[:5])) if completed_outputs else ""}.

For each suggestion provide:
- title: Short name
- rationale: Why this improvement matters
- expected_impact: What benefit it brings (low/medium/high)
- effort: Implementation effort (low/medium/high)
- category: One of "performance", "security", "ux", "architecture", "testing", "documentation"

Return as JSON array. Only suggest improvements with confidence > 0.7.
Avoid generic advice — be specific to this project."""

        suggestions = self._infer_json(prompt)
        if not suggestions or not isinstance(suggestions, list):
            suggestions = self._fallback_suggestions(goal)

        # Filter low-confidence suggestions
        filtered = [s for s in suggestions if isinstance(s, dict) and s.get("expected_impact") != "low"][:5]

        return AgentResult(
            success=True,
            output={"suggestions": filtered, "insertion_point": context.get("insertion_point", "post_execution")},
            metadata={"mode": "suggest", "suggestion_count": len(filtered)},
        )

    # Each rule: (exclude_keywords, require_keywords, suggestion_dict)
    # exclude_keywords: if ANY present in goal, skip this suggestion
    # require_keywords: if ANY present in goal, include this suggestion (empty = always include)
    _SUGGESTION_RULES: list[tuple[tuple[str, ...], tuple[str, ...], dict[str, str]]] = [
        (
            ("test",),
            (),
            {
                "title": "Add comprehensive test coverage",
                "rationale": "No testing mentioned — tests catch regressions early",
                "expected_impact": "high",
                "effort": "medium",
                "category": "testing",
            },
        ),
        (
            ("security", "auth"),
            (),
            {
                "title": "Security review pass",
                "rationale": "No explicit security requirements — proactive auditing prevents vulnerabilities",
                "expected_impact": "high",
                "effort": "low",
                "category": "security",
            },
        ),
        (
            ("doc", "readme"),
            (),
            {
                "title": "Add API documentation",
                "rationale": "Documentation helps future maintainers and API consumers",
                "expected_impact": "medium",
                "effort": "low",
                "category": "documentation",
            },
        ),
        (
            (),
            ("api", "web", "server", "endpoint"),
            {
                "title": "Add rate limiting and input validation",
                "rationale": "Public-facing APIs need protection against abuse",
                "expected_impact": "high",
                "effort": "medium",
                "category": "security",
            },
        ),
        (
            (),
            ("database", "sql", "data"),
            {
                "title": "Add database migration strategy",
                "rationale": "Schema changes need a forward migration path",
                "expected_impact": "medium",
                "effort": "medium",
                "category": "architecture",
            },
        ),
    ]

    _DEFAULT_SUGGESTION: dict[str, str] = {
        "title": "Add error handling and logging",
        "rationale": "Structured logging and error recovery improve reliability",
        "expected_impact": "medium",
        "effort": "low",
        "category": "architecture",
    }

    def _fallback_suggestions(self, goal: str) -> list[dict[str, str]]:
        """Generate basic improvement suggestions without LLM inference.

        Uses rule-based keyword matching against the goal to surface relevant
        suggestions for testing, security, documentation, and architecture.

        Args:
            goal: The project goal text to match suggestions against.

        Returns:
            Up to 5 relevant suggestion dicts, or a single default suggestion.
        """
        goal_lower = goal.lower()
        suggestions = []
        for exclude_kws, require_kws, suggestion in self._SUGGESTION_RULES:
            if exclude_kws and any(kw in goal_lower for kw in exclude_kws):
                continue
            if require_kws and not any(kw in goal_lower for kw in require_kws):
                continue
            suggestions.append(suggestion)

        return suggestions[:5] if suggestions else [self._DEFAULT_SUGGESTION]

    def _execute_default(self, task: AgentTask) -> AgentResult:
        goal = task.prompt or task.description
        prompt = f"""Provide architectural guidance and cost analysis for:\n\n{goal}\n\nInclude architecture_vision, risks, recommended_guidelines, and cost_analysis in JSON."""
        output = self._infer_json(prompt)
        if output is None:
            output = {
                "architecture_vision": f"Modular architecture for: {goal}",
                "risks": [
                    {"risk": "Scope creep", "likelihood": 0.5, "impact": 0.6, "mitigation": "Define MVP clearly"}
                ],
                "recommended_guidelines": ["Use CI/CD", "Write tests", "Document decisions"],
                "cost_analysis": {"estimated_tokens": "unknown", "recommended_model": self.default_model},
            }
        return AgentResult(success=True, output=output, metadata={"mode": "default"})

    def verify(self, output: Any) -> VerificationResult:
        """Verify.

        Returns:
            The VerificationResult result.
        """
        issues = []
        score = 1.0
        if not isinstance(output, dict):
            return VerificationResult(
                passed=False, issues=[{"type": "invalid_type", "message": "Output must be a dict"}], score=0.5
            )
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


def get_architect_agent(config: dict[str, Any] | None = None) -> ArchitectAgent:
    return ArchitectAgent(config)
