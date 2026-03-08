"""
Improvement Agent - Vetinari Meta-Agent for Self-Improvement

A meta-agent that periodically reviews system performance and
generates actionable improvement recommendations. Can automatically
implement low-risk changes or present high-risk changes for human approval.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult,
)

logger = logging.getLogger(__name__)


class ImprovementAgent(BaseAgent):
    """
    Meta-agent that monitors system performance and drives self-improvement.

    Analyzes:
    - Model performance trends across task types
    - Quality score distributions by agent
    - SLA compliance patterns
    - Cost efficiency metrics
    - Anomaly detection patterns
    - Workflow success rates by domain

    Generates recommendations:
    - Model routing adjustments (auto-applies low-risk changes)
    - Prompt variation suggestions (requires A/B test setup)
    - Workflow strategy updates (requires human approval)
    - Configuration tuning (via AutoTuner)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.IMPROVEMENT, config)
        self._review_interval_hours = self._config.get("review_interval_hours", 1)
        self._auto_apply_threshold = self._config.get("auto_apply_threshold", 0.7)

    def get_system_prompt(self) -> str:
        return """You are Vetinari's Improvement Agent -- a meta-analyst that reviews system
performance and recommends improvements.

You have access to:
- Model performance history (success rates, latency, quality scores)
- Workflow execution patterns (depth, breadth, domain distributions)
- Cost trends per model and task type
- SLA compliance data
- Anomaly detection history

Your role:
1. Identify performance bottlenecks and patterns
2. Generate specific, actionable improvement recommendations
3. Prioritize changes by expected impact and risk
4. Flag changes that require human approval
5. Track improvement actions and their outcomes

Output must be actionable: what to change, why, expected impact, and risk level."""

    def get_capabilities(self) -> List[str]:
        return [
            "performance_analysis",
            "model_recommendation",
            "prompt_optimization",
            "workflow_optimization",
            "cost_analysis",
            "sla_monitoring",
            "anomaly_analysis",
        ]

    def execute(self, task: AgentTask) -> AgentResult:
        """Execute a performance review and generate improvement recommendations."""
        if not self.validate_task(task):
            return AgentResult(
                success=False, output=None,
                errors=[f"Invalid task for {self._agent_type.value}"]
            )
        task = self.prepare_task(task)

        try:
            review_type = task.context.get("review_type", "full")
            metrics = self._collect_metrics()
            recommendations = self._generate_recommendations(metrics)
            auto_applied = self._apply_safe_changes(recommendations)

            output = {
                "review_type": review_type,
                "timestamp": datetime.now().isoformat(),
                "metrics_summary": self._summarize_metrics(metrics),
                "recommendations": recommendations,
                "auto_applied": auto_applied,
                "pending_approval": [r for r in recommendations if r.get("risk") == "high"],
            }

            task = self.complete_task(task, AgentResult(success=True, output=output))
            return AgentResult(
                success=True,
                output=output,
                metadata={
                    "recommendations_count": len(recommendations),
                    "auto_applied_count": len(auto_applied),
                }
            )
        except Exception as e:
            self._log("error", f"Improvement review failed: {e}")
            return AgentResult(success=False, output=None, errors=[str(e)])

    def verify(self, output: Any) -> VerificationResult:
        if not isinstance(output, dict):
            return VerificationResult(passed=False, issues=[{"type": "invalid_output"}], score=0.0)
        has_recs = bool(output.get("recommendations"))
        score = 0.9 if has_recs else 0.5
        return VerificationResult(passed=score >= 0.5, issues=[], score=score)

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics from all learning subsystems."""
        metrics: Dict[str, Any] = {}

        # Thompson Sampling arm states
        try:
            from vetinari.learning.model_selector import get_thompson_selector
            selector = get_thompson_selector()
            metrics["model_arms"] = {k: {"mean": v.mean, "pulls": v.total_pulls}
                                     for k, v in selector._arms.items()}
        except Exception:
            metrics["model_arms"] = {}

        # Workflow patterns
        try:
            from vetinari.learning.workflow_learner import get_workflow_learner
            metrics["workflow_patterns"] = get_workflow_learner().get_all_patterns()
        except Exception:
            metrics["workflow_patterns"] = []

        # Quality scoring history
        try:
            from vetinari.learning.quality_scorer import get_quality_scorer
            history = get_quality_scorer().get_history()
            if history:
                metrics["avg_quality"] = sum(s.overall_score for s in history) / len(history)
                metrics["quality_samples"] = len(history)
        except Exception:
            logger.debug("Failed to collect quality scoring history", exc_info=True)

        # Telemetry
        try:
            from vetinari.telemetry import get_telemetry_collector
            tel = get_telemetry_collector()
            tel_data = {}
            if hasattr(tel, "get_adapter_metrics"):
                tel_data["adapters"] = tel.get_adapter_metrics()
            if hasattr(tel, "get_memory_metrics"):
                tel_data["memory"] = tel.get_memory_metrics()
            if hasattr(tel, "get_plan_metrics"):
                tel_data["plans"] = tel.get_plan_metrics()
            if tel_data:
                metrics["telemetry"] = tel_data
        except Exception:
            logger.debug("Failed to collect telemetry metrics", exc_info=True)

        return metrics

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate improvement recommendations from collected metrics."""
        recommendations: List[Dict[str, Any]] = []

        # 1. Identify underperforming models
        for key, arm in metrics.get("model_arms", {}).items():
            if arm["pulls"] >= 10 and arm["mean"] < 0.5:
                model, task_type = key.split(":", 1) if ":" in key else (key, "general")
                recommendations.append({
                    "type": "model_routing",
                    "priority": "medium",
                    "risk": "low",
                    "action": f"Reduce routing weight for model '{model}' on '{task_type}' tasks",
                    "rationale": f"Quality score {arm['mean']:.2f} below threshold after {arm['pulls']} pulls",
                    "auto_apply": True,
                    "parameters": {"model_id": model, "task_type": task_type, "action": "reduce_weight"},
                })

        # 2. Suggest prompt evolution for low-quality agents
        avg_quality = metrics.get("avg_quality", 0.7)
        if avg_quality < 0.6:
            recommendations.append({
                "type": "prompt_evolution",
                "priority": "high",
                "risk": "medium",
                "action": "Trigger prompt A/B testing for lowest-performing agents",
                "rationale": f"System average quality {avg_quality:.2f} below 0.60 threshold",
                "auto_apply": False,
                "parameters": {"action": "generate_variants"},
            })

        # 3. Workflow optimization suggestions
        for pattern in metrics.get("workflow_patterns", []):
            if pattern.get("sample_count", 0) > 5 and pattern.get("success_rate", 1.0) < 0.5:
                recommendations.append({
                    "type": "workflow_strategy",
                    "priority": "high",
                    "risk": "high",
                    "action": f"Review decomposition strategy for '{pattern['domain']}' domain",
                    "rationale": f"Success rate {pattern['success_rate']:.2f} below 0.50 for '{pattern['domain']}'",
                    "auto_apply": False,
                    "parameters": {"domain": pattern["domain"]},
                })

        # 4. Auto-tuner run
        try:
            from vetinari.learning.auto_tuner import get_auto_tuner
            tuner_actions = get_auto_tuner().run_cycle()
            for action in tuner_actions:
                recommendations.append({
                    "type": "system_tuning",
                    "priority": "low",
                    "risk": "low" if action.auto_applied else "medium",
                    "action": f"Adjust {action.parameter}: {action.old_value} → {action.new_value}",
                    "rationale": action.rationale,
                    "auto_apply": action.auto_applied,
                    "parameters": {"parameter": action.parameter, "value": action.new_value},
                })
        except Exception as e:
            logger.debug("AutoTuner cycle failed: %s", e)

        # 5. LLM-generated improvements if metrics show issues
        if len(recommendations) > 0 or avg_quality < 0.65:
            llm_recs = self._generate_llm_recommendations(metrics, recommendations)
            recommendations.extend(llm_recs)

        return recommendations

    def _generate_llm_recommendations(
        self, metrics: Dict[str, Any], existing: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Use LLM to generate additional improvement ideas."""
        try:
            import json
            prompt = f"""You are analyzing an AI orchestration system's performance metrics.

METRICS:
{json.dumps(metrics, default=str)[:1000]}

EXISTING RECOMMENDATIONS:
{json.dumps(existing[:3], indent=2)[:500]}

Generate 2-3 additional specific improvement recommendations as JSON:
[
  {{
    "type": "model_routing|prompt_evolution|workflow_strategy|tooling",
    "priority": "low|medium|high",
    "risk": "low|medium|high",
    "action": "Specific action to take",
    "rationale": "Why this improvement matters",
    "auto_apply": false,
    "parameters": {{}}
  }}
]

Focus on actionable, specific improvements based on the data."""

            result = self._infer_json(prompt)
            if result and isinstance(result, list):
                return result[:3]
        except Exception as e:
            logger.debug("LLM recommendations failed: %s", e)
        return []

    def _apply_safe_changes(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Auto-apply low-risk changes."""
        applied: List[Dict[str, Any]] = []
        for rec in recommendations:
            if not rec.get("auto_apply") or rec.get("risk") == "high":
                continue
            try:
                rec_type = rec.get("type")
                params = rec.get("parameters", {})
                if rec_type == "model_routing":
                    self._apply_model_routing_change(params)
                    applied.append(rec)
            except Exception as e:
                logger.debug("Failed to auto-apply recommendation: %s", e)
        return applied

    def _apply_model_routing_change(self, params: Dict[str, Any]) -> None:
        """Apply a model routing weight change."""
        try:
            from vetinari.dynamic_model_router import get_model_router
            router = get_model_router()
            model_id = params.get("model_id")
            if model_id and hasattr(router, "get_performance_cache"):
                key = f"{model_id}:{params.get('task_type', 'general')}"
                cache = router.get_performance_cache(key)
                # Reduce success_rate to lower routing probability
                cache["success_rate"] = max(0.1, cache.get("success_rate", 0.5) * 0.8)
                router.update_performance_cache(key, cache)
        except Exception as e:
            logger.debug("Model routing change failed: %s", e)

    def _summarize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create a human-readable metrics summary."""
        arms = metrics.get("model_arms", {})
        return {
            "model_count": len(set(k.split(":")[0] for k in arms)),
            "total_model_pulls": sum(v["pulls"] for v in arms.values()),
            "avg_model_quality": (
                sum(v["mean"] * v["pulls"] for v in arms.values()) /
                max(sum(v["pulls"] for v in arms.values()), 1)
            ) if arms else 0.7,
            "workflow_domains": len(metrics.get("workflow_patterns", [])),
            "quality_samples": metrics.get("quality_samples", 0),
        }


# Singleton
_improvement_agent: Optional[ImprovementAgent] = None


def get_improvement_agent(config: Optional[Dict[str, Any]] = None) -> ImprovementAgent:
    """Get the singleton Improvement agent instance."""
    global _improvement_agent
    if _improvement_agent is None:
        _improvement_agent = ImprovementAgent(config)
    return _improvement_agent
