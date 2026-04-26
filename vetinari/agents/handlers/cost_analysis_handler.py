"""Cost analysis mode handler for the Operations Agent.

Extracts the cost analysis and model comparison logic from OperationsAgent
into a standalone handler class. Calculates per-task token costs, compares
model pricing tiers, and recommends cost-efficient model selections.
"""

from __future__ import annotations

import logging
import types
from typing import Any

from vetinari.agents.contracts import AgentResult, AgentTask
from vetinari.agents.handlers import BaseHandler
from vetinari.constants import TRUNCATE_CONTENT_ANALYSIS

logger = logging.getLogger(__name__)

# Model pricing table -- immutable mapping to prevent accidental mutation.
# Keys are model identifiers; values contain per-1k-token costs and tier.
MODEL_PRICING = types.MappingProxyType({
    "qwen2.5-coder-7b": types.MappingProxyType({"input_per_1k": 0.0001, "output_per_1k": 0.0002, "tier": "small"}),
    "qwen2.5-72b": types.MappingProxyType({"input_per_1k": 0.001, "output_per_1k": 0.002, "tier": "large"}),
    "qwen3-30b-a3b": types.MappingProxyType({"input_per_1k": 0.0005, "output_per_1k": 0.001, "tier": "medium"}),
    "qwen2.5-vl-32b": types.MappingProxyType({"input_per_1k": 0.0005, "output_per_1k": 0.001, "tier": "medium"}),
    "claude-3.5-sonnet": types.MappingProxyType({"input_per_1k": 0.003, "output_per_1k": 0.015, "tier": "premium"}),
    "gpt-4o": types.MappingProxyType({"input_per_1k": 0.005, "output_per_1k": 0.015, "tier": "premium"}),
    "gemini-1.5-pro": types.MappingProxyType({"input_per_1k": 0.00125, "output_per_1k": 0.005, "tier": "large"}),
})


class CostAnalysisHandler(BaseHandler):
    """Handler for the 'cost_analysis' mode of OperationsAgent.

    Performs model cost comparisons using a static pricing table, and falls
    back to LLM-assisted general cost analysis when the analysis type is not
    a direct model comparison.
    """

    def __init__(self) -> None:
        super().__init__(
            mode_name="cost_analysis",
            description="Calculate token costs, compare models, recommend cost-efficient selections",
        )

    def get_system_prompt(self) -> str:
        """Return the cost-analyst system prompt.

        Returns:
            A multi-section prompt defining the cost analyst's responsibilities,
            analysis framework, optimisation strategies, and output format.
        """
        return (
            "You are Vetinari's Cost Analyst -- an expert in AI/ML economics, token pricing,\n"
            "and infrastructure cost optimisation. You help teams make data-driven decisions\n"
            "about model selection, deployment strategy, and resource allocation.\n\n"
            "## Core Responsibilities\n"
            "- Calculate per-task token costs across all available models and providers\n"
            "- Compare local inference vs cloud API costs with full TCO analysis\n"
            "- Recommend cost-efficient model selections that meet quality thresholds\n"
            "- Forecast monthly/quarterly spending based on usage patterns\n"
            "- Identify cost anomalies and optimisation opportunities\n\n"
            "## Analysis Framework\n"
            "- Always include: input token cost, output token cost, latency, quality score\n"
            "- Calculate cost-per-quality-point ($/quality) for meaningful comparisons\n"
            "- Factor in hidden costs: retry overhead, escalation chains, batch vs real-time\n"
            "- Compare tiers: small (7B, <$0.001/1k), medium (30B, ~$0.001/1k),\n"
            "  large (72B, ~$0.002/1k), premium (cloud APIs, $0.003-0.015/1k)\n"
            "- Local models: amortised hardware cost = $0 per token after purchase\n\n"
            "## Optimisation Strategies\n"
            "- Cascade routing: start cheap, escalate only on low confidence (saves 40-60%)\n"
            "- Batch processing: queue non-urgent work for 50% API discount\n"
            "- Prompt caching: reuse system prompts to cut input costs (Anthropic: 90% reduction)\n"
            "- Token budgeting: set per-task max_tokens to prevent runaway costs\n"
            "- SLM preprocessing: use small models for classification/routing before expensive inference\n\n"
            "## Output Format\n"
            "Return JSON with 'comparisons' (array of model cost breakdowns), 'recommendation'\n"
            "(cheapest adequate model), 'estimated_savings', and 'forecast' when applicable.\n"
            "Always include concrete dollar amounts, not just relative comparisons."
        )

    def execute(self, task: AgentTask, context: dict[str, Any]) -> AgentResult:
        """Perform cost analysis for the given task.

        Supports two analysis types:
        - 'model_comparison': deterministic cost calculation across specified
          models using the static pricing table.
        - Any other type: heuristic token estimation followed by LLM-assisted
          analysis via the 'infer_json' callable in the execution context.

        Args:
            task: The agent task containing the cost analysis request.
            context: Execution context; should contain an 'infer_json' callable
                with signature ``(prompt: str, fallback: Any) -> dict`` for
                non-comparison analysis types.

        Returns:
            An AgentResult with cost comparisons, recommendations, and
            estimated savings in the output field.
        """
        task_context = task.context or {}
        analysis_type = task_context.get("analysis_type", "model_comparison")

        if analysis_type == "model_comparison":
            return self._execute_model_comparison(task_context)

        return self._execute_general_analysis(task, context)

    def _execute_model_comparison(self, task_context: dict[str, Any]) -> AgentResult:
        """Run a deterministic model cost comparison.

        Args:
            task_context: Task context containing optional 'models' list and
                'estimated_tokens' count.

        Returns:
            An AgentResult with sorted cost comparisons and the cheapest model
            as the recommendation.
        """
        models = task_context.get("models", list(MODEL_PRICING.keys()))
        estimated_tokens = task_context.get("estimated_tokens", 10000)

        comparisons = []
        for model_id in models:
            pricing = MODEL_PRICING.get(
                model_id,
                {"input_per_1k": 0.001, "output_per_1k": 0.002, "tier": "unknown"},
            )
            input_cost = (estimated_tokens / 1000) * pricing["input_per_1k"]
            output_cost = (estimated_tokens / 1000) * pricing["output_per_1k"]
            comparisons.append(
                {
                    "model": model_id,
                    "tier": pricing["tier"],
                    "input_cost": round(input_cost, 4),
                    "output_cost": round(output_cost, 4),
                    "total_cost": round(input_cost + output_cost, 4),
                },
            )
        comparisons.sort(key=lambda c: c["total_cost"])

        return AgentResult(
            success=True,
            output={
                "comparisons": comparisons,
                "recommendation": comparisons[0]["model"] if comparisons else "unknown",
                "estimated_tokens": estimated_tokens,
                "cheapest": comparisons[0] if comparisons else None,
                "most_expensive": comparisons[-1] if comparisons else None,
            },
            metadata={"mode": "cost_analysis", "analysis_type": "model_comparison"},
        )

    def _execute_general_analysis(self, task: AgentTask, context: dict[str, Any]) -> AgentResult:
        """Run a heuristic + LLM-assisted general cost analysis.

        Args:
            task: The agent task with a description to analyse.
            context: Execution context with optional 'infer_json' callable.

        Returns:
            An AgentResult with token estimates, model recommendations, and
            savings guidance.
        """
        description = task.description or ""
        word_count = len(description.split())
        estimated_tokens = int(word_count * 1.3)  # rough word-to-token ratio

        recommendations = []
        for model_id, pricing in sorted(
            MODEL_PRICING.items(),
            key=lambda x: x[1].get("input_per_1k", 0),
        ):
            cost = (estimated_tokens / 1000) * (pricing["input_per_1k"] + pricing["output_per_1k"])
            recommendations.append(
                {
                    "model": model_id,
                    "tier": pricing["tier"],
                    "estimated_cost": round(cost, 6),
                },
            )

        fallback: dict[str, Any] = {
            "analysis": f"Estimated {estimated_tokens} tokens based on {word_count} words",
            "recommendations": recommendations[:3],
            "estimated_savings": "Use local models for 10-100x cost reduction vs cloud APIs",
        }

        prompt = (
            f"Perform cost analysis for:\n{description[:TRUNCATE_CONTENT_ANALYSIS]}\n\n"
            "Respond as JSON:\n"
            '{"analysis": "...", "recommendations": [...], "estimated_savings": "..."}'
        )

        infer_json = context.get("infer_json")
        if infer_json is not None:
            result = infer_json(prompt, fallback=fallback)
            return AgentResult(
                success=True,
                output=result,
                metadata={"mode": "cost_analysis"},
            )
        else:
            self._logger.warning("No infer_json callable in context, using fallback")
            return AgentResult(
                success=False,
                output=fallback,
                metadata={"mode": "cost_analysis", "_is_fallback": True},
            )
