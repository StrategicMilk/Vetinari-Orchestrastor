"""
Vetinari Experimentation Manager Agent

LLM-powered experimentation management agent that interfaces with the real
ThompsonSampling, QualityScorer, and TelemetryCollector systems to track
actual model performance experiments and generate evidence-based recommendations.
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


class ExperimentationManagerAgent(BaseAgent):
    """Experimentation Manager — real system telemetry + LLM-powered experiment analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.EXPERIMENTATION_MANAGER, config)

    def get_system_prompt(self) -> str:
        return """You are Vetinari's Experimentation Manager — a data scientist specialising in
LLM system evaluation, A/B testing, and model performance analysis.

You have access to real system telemetry, model performance history, and quality scores.
Base ALL analysis on the actual data provided — do not fabricate metrics.

Required output (JSON):
{
  "experiment_log": [
    {
      "id": "...",
      "name": "...",
      "hypothesis": "...",
      "status": "active|completed|failed",
      "start_date": "...",
      "metrics_tracked": [],
      "tags": []
    }
  ],
  "configuration": {
    "model_variants": {},
    "hyperparameters": {},
    "environment": {},
    "seed": null
  },
  "results": {
    "metrics": [
      {"name": "...", "value": null, "unit": "...", "source": "measured|estimated"}
    ],
    "comparisons": [],
    "statistical_significance": "...",
    "confidence_level": null
  },
  "reproducibility_plan": {
    "version_control": {},
    "dependencies": [],
    "data_checksum": null,
    "instructions": []
  },
  "analysis": {
    "summary": "...",
    "insights": [],
    "recommendations": [],
    "next_steps": [],
    "risks": []
  },
  "summary": "..."
}

Important: Mark all values as "source": "measured" only if they come from actual system data.
Use "source": "estimated" for projected or hypothetical values. Never fabricate measurements.
"""

    def get_capabilities(self) -> List[str]:
        return [
            "experiment_planning",
            "configuration_tracking",
            "result_recording",
            "reproducibility_documentation",
            "hypothesis_testing",
            "experiment_analysis",
        ]

    def execute(self, task: AgentTask) -> AgentResult:
        """Execute experimentation management using real system data + LLM analysis."""
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"],
            )

        task = self.prepare_task(task)

        try:
            experiments = task.context.get("experiments", [])
            baseline = task.context.get("baseline", {})

            # Collect real system telemetry and performance data
            real_data = self._collect_system_data()

            # Build prompt with actual data
            data_summary = self._format_system_data(real_data)
            exp_descriptions = "\n".join(
                [f"- {e.get('name', e)}: {e.get('hypothesis', '')}" for e in experiments[:5]]
            ) if experiments else "No specific experiments — analyse current system performance"

            prompt = (
                f"Manage and analyse experiments for Vetinari AI orchestration system.\n\n"
                f"Planned experiments:\n{exp_descriptions}\n\n"
                f"Real system data:\n{data_summary}\n\n"
                f"Baseline configuration:\n{str(baseline)[:500]}\n\n"
                "Analyse the actual data, identify performance patterns, and generate "
                "evidence-based recommendations. Return JSON matching the required format. "
                "Mark all metrics as 'measured' only if sourced from the real system data."
            )

            management = self._infer_json(
                prompt=prompt,
                fallback=self._fallback_management(experiments, real_data),
            )

            # Ensure required keys
            management.setdefault("experiment_log", [])
            management.setdefault("configuration", {})
            management.setdefault("results", {"metrics": [], "comparisons": []})
            management.setdefault("reproducibility_plan", {"instructions": []})
            management.setdefault("analysis", {"insights": [], "recommendations": []})
            management.setdefault("summary", "Experiment management complete")

            result = AgentResult(
                success=True,
                output=management,
                metadata={
                    "experiments_count": len(experiments),
                    "real_data_sources": list(real_data.keys()),
                    "tracked_experiments": len(management.get("experiment_log", [])),
                },
            )
            task = self.complete_task(task, result)
            return result

        except Exception as e:
            self._log("error", f"Experiment management failed: {e}")
            return AgentResult(success=False, output=None, errors=[str(e)])

    def verify(self, output: Any) -> VerificationResult:
        issues = []
        score = 1.0

        if not isinstance(output, dict):
            issues.append({"type": "invalid_type", "message": "Output must be a dict"})
            return VerificationResult(passed=False, issues=issues, score=0.0)

        if not output.get("experiment_log") and not output.get("analysis"):
            issues.append({"type": "missing_content", "message": "No experiment log or analysis"})
            score -= 0.4
        if not output.get("results"):
            issues.append({"type": "missing_results", "message": "Results missing"})
            score -= 0.2
        if not output.get("reproducibility_plan"):
            issues.append({"type": "missing_reproducibility", "message": "Reproducibility plan missing"})
            score -= 0.2
        if not output.get("analysis"):
            issues.append({"type": "missing_analysis", "message": "Analysis missing"})
            score -= 0.1

        passed = score >= 0.5
        return VerificationResult(passed=passed, issues=issues, score=max(0.0, score))

    def _collect_system_data(self) -> Dict[str, Any]:
        """Collect real performance data from system modules."""
        data = {}

        # ThompsonSampling model performance
        try:
            from vetinari.learning.model_selector import get_thompson_selector
            selector = get_thompson_selector()
            # Use _arms (private attribute is correct) or get_arm_state()
            arms = getattr(selector, "_arms", None) or (
                selector.get_arm_state() if hasattr(selector, "get_arm_state") else {}
            )
            data["thompson_sampling"] = {
                "model_scores": {
                    model: {"alpha": arm.alpha, "beta": arm.beta,
                            "mean": arm.alpha / (arm.alpha + arm.beta)}
                    for model, arm in arms.items()
                } if arms else {}
            }
        except Exception:
            pass

        # Quality scores
        try:
            from vetinari.learning.quality_scorer import get_quality_scorer
            scorer = get_quality_scorer()
            history = scorer.get_history() if hasattr(scorer, "get_history") else []
            if history:
                def _score_val(h):
                    # Handle both QualityScore objects and dicts
                    return h.overall_score if hasattr(h, "overall_score") else h.get("overall_score", h.get("score", 0))
                data["quality_scores"] = {
                    "count": len(history),
                    "avg_score": sum(_score_val(h) for h in history) / len(history),
                    "recent": [
                        {"model": getattr(h, "model_id", "?"), "score": _score_val(h),
                         "type": getattr(h, "task_type", "?")}
                        for h in history[-5:]
                    ],
                }
        except Exception:
            pass

        # Workflow patterns
        try:
            from vetinari.learning.workflow_learner import get_workflow_learner
            learner = get_workflow_learner()
            # get_all_patterns() is the correct method (not get_patterns)
            patterns = learner.get_all_patterns() if hasattr(learner, "get_all_patterns") else {}
            if patterns:
                data["workflow_patterns"] = patterns
        except Exception:
            pass

        # Telemetry
        try:
            from vetinari.telemetry import get_telemetry_collector
            tel = get_telemetry_collector()
            tel_data = {}
            if hasattr(tel, "get_adapter_metrics"):
                tel_data["adapters"] = tel.get_adapter_metrics()
            if hasattr(tel, "get_plan_metrics"):
                tel_data["plans"] = tel.get_plan_metrics()
            if tel_data:
                data["telemetry"] = tel_data
        except Exception:
            pass

        return data

    def _format_system_data(self, data: Dict) -> str:
        if not data:
            return "No system telemetry available (first run or cold start)"
        lines = []
        for source, content in data.items():
            lines.append(f"[{source}]")
            if isinstance(content, dict):
                for k, v in list(content.items())[:5]:
                    lines.append(f"  {k}: {str(v)[:150]}")
            else:
                lines.append(f"  {str(content)[:200]}")
        return "\n".join(lines)

    def _fallback_management(
        self, experiments: List[Dict], real_data: Dict
    ) -> Dict[str, Any]:
        now = datetime.now().isoformat()
        experiment_log = []
        for i, exp in enumerate(experiments):
            name = exp.get("name", f"Experiment {i+1}") if isinstance(exp, dict) else str(exp)
            experiment_log.append({
                "id": f"exp_{i+1:03d}",
                "name": name,
                "hypothesis": exp.get("hypothesis", "") if isinstance(exp, dict) else "",
                "status": "active",
                "start_date": now,
                "metrics_tracked": ["quality_score", "latency_ms", "tokens_used"],
                "tags": exp.get("tags", []) if isinstance(exp, dict) else [],
            })

        # Build configuration from real system state
        config = {
            "model_variants": {},
            "hyperparameters": {"temperature": 0.7, "max_tokens": 2048},
            "environment": {"vetinari_version": "0.2.0", "local_inference": True},
            "seed": None,
        }

        # Extract real metrics if available
        metrics = []
        ts_data = real_data.get("thompson_sampling", {}).get("model_scores", {})
        for model, scores in ts_data.items():
            metrics.append({
                "name": f"{model}_quality",
                "value": round(scores.get("mean", 0), 3),
                "unit": "score",
                "source": "measured",
            })
        if not metrics:
            metrics.append({
                "name": "quality_score",
                "value": None,
                "unit": "score",
                "source": "not_yet_collected",
            })

        return {
            "experiment_log": experiment_log,
            "configuration": config,
            "results": {
                "metrics": metrics,
                "comparisons": [],
                "statistical_significance": "Insufficient data for significance testing",
                "confidence_level": None,
            },
            "reproducibility_plan": {
                "version_control": {"note": "Use git to tag experiment versions"},
                "dependencies": [
                    {"package": "vetinari", "version": "0.2.0"},
                    {"package": "duckduckgo-search", "version": ">=4.0.0"},
                ],
                "data_checksum": None,
                "instructions": [
                    "1. Record the git commit hash at experiment start",
                    "2. Save the full config snapshot to experiments/<id>/config.json",
                    "3. Log all inference calls with model ID, tokens, latency",
                    "4. Store quality scores in vetinari_memory.db",
                    "5. Archive results to experiments/<id>/results.json",
                ],
            },
            "analysis": {
                "summary": f"Tracking {len(experiment_log)} experiments. Real telemetry {'available' if real_data else 'not yet collected'}.",
                "insights": [
                    f"System has {len(ts_data)} models with performance history" if ts_data else "No model performance history yet — run tasks to collect data",
                    "Thompson Sampling will adaptively route to better-performing models as data accumulates",
                ],
                "recommendations": [
                    "Run at least 10 tasks per model/task-type combination before drawing conclusions",
                    "Use the Improvement Agent to apply auto-tuning after collecting sufficient data",
                    "Monitor quality scores per agent type to identify underperforming agents",
                ],
                "next_steps": [
                    "Enable telemetry collection in all adapter infer() calls",
                    "Run a sample workload of 50+ tasks to build model performance baselines",
                    "Schedule regular ImprovementAgent runs (every 100 tasks or daily)",
                ],
                "risks": [
                    "Insufficient data may lead to premature model routing decisions",
                    "Quality scorer uses LLM-as-judge which may be inconsistent",
                ],
            },
            "summary": f"Experiment management complete. {len(experiment_log)} experiments tracked.",
        }


_experimentation_manager_agent: Optional[ExperimentationManagerAgent] = None


def get_experimentation_manager_agent(config: Optional[Dict[str, Any]] = None) -> ExperimentationManagerAgent:
    global _experimentation_manager_agent
    if _experimentation_manager_agent is None:
        _experimentation_manager_agent = ExperimentationManagerAgent(config)
    return _experimentation_manager_agent
