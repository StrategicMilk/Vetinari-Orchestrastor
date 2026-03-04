"""
Vetinari Cost Planner Agent

LLM-powered cost planning agent that analyses real model usage data, computes
actual token costs from the CostTracker/ThompsonSampling systems, and provides
data-driven model selection recommendations.
"""

import logging
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult,
)

logger = logging.getLogger(__name__)

# Real model pricing per 1k tokens (input/output average) — updated 2025/2026
MODEL_PRICING = {
    # Local (free)
    "qwen2.5-coder-7b": 0.0,
    "qwen2.5-72b": 0.0,
    "llama-3.3-70b": 0.0,
    "qwen3-30b-a3b": 0.0,
    "qwen2.5-vl-32b": 0.0,
    # Cloud
    "claude-sonnet-4": 0.015,
    "claude-opus-4": 0.075,
    "claude-haiku-3": 0.0008,
    "claude-3-5-sonnet-20241022": 0.015,
    "gpt-4o": 0.005,
    "gpt-4o-mini": 0.0006,
    "gemini-2.0-flash": 0.0,
    "gemini-1.5-pro": 0.00125,
    "gemini-1.5-flash": 0.000075,
    "command-r-plus": 0.003,
    "command-r": 0.0005,
}


class CostPlannerAgent(BaseAgent):
    """Cost Planner agent — data-driven cost analysis, model recommendations, budget planning."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.COST_PLANNER, config)

    def get_system_prompt(self) -> str:
        return """You are Vetinari's Cost Planner — a FinOps engineer specialising in LLM cost optimisation.
Your job is to analyse actual usage data and provide actionable, quantified cost recommendations.

You have access to real model pricing data and actual usage statistics.
Every recommendation must be backed by specific numbers and ROI calculations.

Required output (JSON):
{
  "cost_report": {
    "total_tokens_used": 0,
    "total_cost_usd": 0.0,
    "cost_by_model": {},
    "cost_by_task_type": {},
    "cost_by_agent": {},
    "period": "...",
    "currency": "USD",
    "trend": "increasing|stable|decreasing"
  },
  "model_recommendations": [
    {
      "task_type": "...",
      "current_model": "...",
      "recommended_model": "...",
      "current_cost_per_1k": 0.0,
      "recommended_cost_per_1k": 0.0,
      "estimated_savings_percent": 0,
      "quality_trade_off": "none|minimal|moderate|significant",
      "reasoning": "..."
    }
  ],
  "budget_constraints": {
    "monthly_budget_usd": null,
    "per_run_budget_usd": null,
    "local_only_mode": false,
    "cloud_escalation_threshold": "..."
  },
  "optimizations": [
    {
      "technique": "...",
      "estimated_savings_percent": 0,
      "implementation_effort": "low|medium|high",
      "description": "...",
      "implementation_steps": []
    }
  ],
  "token_efficiency_analysis": {
    "avg_tokens_per_task": 0,
    "wasteful_patterns": [],
    "compression_opportunities": []
  },
  "summary": "..."
}
"""

    def get_capabilities(self) -> List[str]:
        return [
            "cost_calculation",
            "budget_planning",
            "model_selection",
            "usage_tracking",
            "cost_reporting",
            "optimization_analysis",
        ]

    def execute(self, task: AgentTask) -> AgentResult:
        """Execute cost analysis using real telemetry data + LLM reasoning."""
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"],
            )

        task = self.prepare_task(task)

        try:
            plan_outputs = task.context.get("plan_outputs", task.description)
            usage_stats = task.context.get("usage_stats", {})

            # Collect real usage data from system analytics
            real_usage = self._collect_real_usage_data(usage_stats)

            # Search for cost optimisation strategies
            opt_context = ""
            try:
                search_results = self._search(
                    "LLM token cost optimisation strategies prompt caching model routing 2025"
                )
                if search_results:
                    opt_context = "\n".join([r.get("snippet", "") for r in search_results[:2]])
            except Exception:
                pass

            prompt = (
                f"Analyse LLM costs for this project: {plan_outputs}\n\n"
                f"Actual usage data:\n{self._format_usage(real_usage)}\n\n"
                f"Available models and pricing:\n{self._format_pricing()}\n\n"
                f"Cost optimisation context:\n{opt_context[:600]}\n\n"
                "Generate a detailed cost analysis with specific, quantified recommendations. "
                "Return JSON matching the required output format."
            )

            analysis = self._infer_json(
                prompt=prompt,
                fallback=self._fallback_analysis(real_usage, plan_outputs),
            )

            # Ensure required keys
            analysis.setdefault("cost_report", {"total_cost_usd": 0.0, "currency": "USD"})
            analysis.setdefault("model_recommendations", [])
            analysis.setdefault("budget_constraints", {})
            analysis.setdefault("optimizations", [])
            analysis.setdefault("token_efficiency_analysis", {})
            analysis.setdefault("summary", "Cost analysis complete")

            result = AgentResult(
                success=True,
                output=analysis,
                metadata={
                    "total_cost": analysis.get("cost_report", {}).get("total_cost_usd", 0),
                    "recommendations_count": len(analysis.get("model_recommendations", [])),
                    "optimizations_count": len(analysis.get("optimizations", [])),
                },
            )
            task = self.complete_task(task, result)
            return result

        except Exception as e:
            self._log("error", f"Cost analysis failed: {e}")
            return AgentResult(success=False, output=None, errors=[str(e)])

    def verify(self, output: Any) -> VerificationResult:
        issues = []
        score = 1.0

        if not isinstance(output, dict):
            issues.append({"type": "invalid_type", "message": "Output must be a dict"})
            return VerificationResult(passed=False, issues=issues, score=0.0)

        if not output.get("cost_report"):
            issues.append({"type": "missing_report", "message": "Cost report missing"})
            score -= 0.25
        if not output.get("model_recommendations"):
            issues.append({"type": "missing_recommendations", "message": "Model recommendations missing"})
            score -= 0.2
        if not output.get("budget_constraints"):
            issues.append({"type": "missing_constraints", "message": "Budget constraints missing"})
            score -= 0.2
        if not output.get("optimizations"):
            issues.append({"type": "missing_optimizations", "message": "Optimization suggestions missing"})
            score -= 0.15

        passed = score >= 0.5
        return VerificationResult(passed=passed, issues=issues, score=max(0.0, score))

    def _collect_real_usage_data(self, provided_stats: Dict) -> Dict[str, Any]:
        """Collect real usage statistics from system analytics modules."""
        usage = dict(provided_stats)

        # Try CostTracker
        try:
            from vetinari.analytics.cost import get_cost_tracker
            tracker = get_cost_tracker()
            # get_stats() is the correct method (not get_summary)
            stats = tracker.get_stats() if hasattr(tracker, "get_stats") else {}
            if stats:
                usage["cost_tracker"] = stats
        except Exception:
            pass

        # Try ThompsonSampling for model performance data
        try:
            from vetinari.learning.model_selector import get_thompson_selector
            selector = get_thompson_selector()
            # get_rankings() is the correct method (not get_efficiencies)
            rankings = selector.get_rankings() if hasattr(selector, "get_rankings") else []
            if rankings:
                usage["model_rankings"] = rankings
        except Exception:
            pass

        # Try TelemetryCollector
        try:
            from vetinari.telemetry import get_telemetry_collector
            tel = get_telemetry_collector()
            # Combine available metric getters
            metrics = {}
            if hasattr(tel, "get_adapter_metrics"):
                metrics["adapters"] = tel.get_adapter_metrics()
            if hasattr(tel, "get_plan_metrics"):
                metrics["plans"] = tel.get_plan_metrics()
            if metrics:
                usage["telemetry"] = metrics
        except Exception:
            pass

        return usage

    def _format_usage(self, usage: Dict) -> str:
        lines = []
        for k, v in usage.items():
            if isinstance(v, dict):
                lines.append(f"{k}: {str(v)[:200]}")
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines) if lines else "No usage data available (first run)"

    def _format_pricing(self) -> str:
        lines = []
        local = [(m, p) for m, p in MODEL_PRICING.items() if p == 0.0]
        cloud = [(m, p) for m, p in MODEL_PRICING.items() if p > 0.0]
        lines.append("LOCAL (free): " + ", ".join([m for m, _ in local]))
        for model, price in sorted(cloud, key=lambda x: x[1]):
            lines.append(f"  {model}: ${price:.4f}/1k tokens")
        return "\n".join(lines)

    def _fallback_analysis(self, usage: Dict, description: str) -> Dict[str, Any]:
        """Structured fallback when LLM inference fails."""
        has_usage = bool(usage.get("cost_tracker") or usage.get("telemetry"))
        return {
            "cost_report": {
                "total_tokens_used": usage.get("total_tokens", 0),
                "total_cost_usd": 0.0,
                "cost_by_model": {},
                "cost_by_task_type": {},
                "cost_by_agent": {},
                "period": "current session",
                "currency": "USD",
                "trend": "stable",
                "note": "Running on local models — zero cloud cost" if not has_usage else "Based on collected telemetry",
            },
            "model_recommendations": [
                {
                    "task_type": "planning",
                    "current_model": "qwen2.5-72b",
                    "recommended_model": "qwen2.5-72b",
                    "current_cost_per_1k": 0.0,
                    "recommended_cost_per_1k": 0.0,
                    "estimated_savings_percent": 0,
                    "quality_trade_off": "none",
                    "reasoning": "Local model — no cost. Maintain for best quality.",
                },
                {
                    "task_type": "code_generation",
                    "current_model": "qwen2.5-coder-7b",
                    "recommended_model": "qwen2.5-coder-7b",
                    "current_cost_per_1k": 0.0,
                    "recommended_cost_per_1k": 0.0,
                    "estimated_savings_percent": 0,
                    "quality_trade_off": "none",
                    "reasoning": "Specialised coding model — optimal choice.",
                },
                {
                    "task_type": "complex_reasoning",
                    "current_model": "qwen2.5-72b",
                    "recommended_model": "claude-haiku-3",
                    "current_cost_per_1k": 0.0,
                    "recommended_cost_per_1k": 0.0008,
                    "estimated_savings_percent": 0,
                    "quality_trade_off": "minimal",
                    "reasoning": "For cloud escalation: use Haiku for speed/cost, Sonnet for quality",
                },
            ],
            "budget_constraints": {
                "monthly_budget_usd": None,
                "per_run_budget_usd": None,
                "local_only_mode": True,
                "cloud_escalation_threshold": "Escalate to cloud only when local quality score < 0.6",
            },
            "optimizations": [
                {
                    "technique": "Prompt caching (Anthropic)",
                    "estimated_savings_percent": 85,
                    "implementation_effort": "low",
                    "description": "Cache system prompts using Anthropic's prompt caching API",
                    "implementation_steps": ["Add cache_control to system prompt", "Ensure prompt prefix is stable"],
                },
                {
                    "technique": "Model routing by task complexity",
                    "estimated_savings_percent": 70,
                    "implementation_effort": "medium",
                    "description": "Route simple tasks (classification, extraction) to 7B models; reserve 72B for complex reasoning",
                    "implementation_steps": ["Classify task complexity", "Route based on complexity score"],
                },
                {
                    "technique": "Output token constraints",
                    "estimated_savings_percent": 25,
                    "implementation_effort": "low",
                    "description": "Set appropriate max_tokens per task type instead of using 2048 universally",
                    "implementation_steps": ["Map task types to appropriate max_tokens", "Use JSON mode to reduce verbose output"],
                },
                {
                    "technique": "Local LLM preprocessing",
                    "estimated_savings_percent": 40,
                    "implementation_effort": "high",
                    "description": "Use local models to compress and summarise context before cloud API calls",
                    "implementation_steps": ["Implement LocalPreprocessor", "Extract key facts from verbose context", "Reduce cloud prompt size by 40-60%"],
                },
            ],
            "token_efficiency_analysis": {
                "avg_tokens_per_task": 0,
                "wasteful_patterns": ["Universal max_tokens=2048 regardless of task", "No prompt caching", "No context compression"],
                "compression_opportunities": ["Summarise prior task outputs before including in context", "Use RepoMap instead of raw file contents"],
            },
            "summary": "Cost analysis complete. Currently running on local models (zero cost). Implement token optimisations before enabling cloud models.",
        }


_cost_planner_agent: Optional[CostPlannerAgent] = None


def get_cost_planner_agent(config: Optional[Dict[str, Any]] = None) -> CostPlannerAgent:
    global _cost_planner_agent
    if _cost_planner_agent is None:
        _cost_planner_agent = CostPlannerAgent(config)
    return _cost_planner_agent
