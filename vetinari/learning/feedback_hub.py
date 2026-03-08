"""
Unified Feedback Hub - Vetinari Integration Layer (I14)

Central dispatcher that routes feedback events to all relevant learning
subsystems in one call. Replaces scattered try/except fan-out blocks
with a single, consistent entry point.

Usage::

    from vetinari.learning.feedback_hub import get_feedback_hub

    hub = get_feedback_hub()
    hub.on_task_complete(
        task_id="t_123",
        model_id="qwen3-32b",
        agent_type="BUILDER",
        task_type="coding",
        quality_score=0.85,
        success=True,
        output_summary="Generated UserRepository class",
        task_description="Build a user repository with CRUD ops",
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FeedbackHub:
    """
    Unified feedback dispatcher.

    Routes task completion, benchmark, and approval events to:
    - FeedbackLoop (model performance + router cache)
    - EpisodeMemory (episodic storage for similarity retrieval)
    - WorkflowLearner (decomposition pattern learning)
    - PromptEvolver (prompt variant quality tracking)
    - QualityScorer (scoring pipeline)
    - AnomalyDetector (anomaly monitoring)

    All calls are wrapped in try/except so a failure in one subsystem
    never blocks the others.
    """

    def on_task_complete(
        self,
        task_id: str,
        model_id: str,
        agent_type: str,
        task_type: str,
        quality_score: float,
        success: bool,
        output_summary: str = "",
        task_description: str = "",
        latency_ms: int = 0,
        cost_usd: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, bool]:
        """
        Fan out a task completion event to all learning subsystems.

        Args:
            task_id: Unique task identifier.
            model_id: Model that executed the task.
            agent_type: Agent type string (e.g. "BUILDER").
            task_type: Task category (e.g. "coding").
            quality_score: Overall quality 0.0-1.0.
            success: Whether the task succeeded.
            output_summary: Truncated output for episode storage.
            task_description: Original task description.
            latency_ms: Execution latency in milliseconds.
            cost_usd: Estimated cost in USD.
            metadata: Optional extra metadata dict.

        Returns:
            Dict mapping subsystem name to success boolean.
        """
        results: Dict[str, bool] = {}

        # 1. FeedbackLoop (model performance + router + Thompson)
        try:
            from vetinari.learning.feedback_loop import get_feedback_loop
            get_feedback_loop().record_outcome(
                task_id=task_id,
                model_id=model_id,
                task_type=task_type,
                quality_score=quality_score,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                success=success,
            )
            results["feedback_loop"] = True
        except Exception as e:
            logger.debug(f"[FeedbackHub] FeedbackLoop failed: {e}")
            results["feedback_loop"] = False

        # 2. EpisodeMemory (episodic storage)
        try:
            from vetinari.learning.episode_memory import get_episode_memory
            get_episode_memory().record(
                task_description=task_description or task_id,
                agent_type=agent_type,
                task_type=task_type,
                output_summary=output_summary[:500] if output_summary else "",
                quality_score=quality_score,
                success=success,
                model_id=model_id,
                metadata=metadata,
            )
            results["episode_memory"] = True
        except Exception as e:
            logger.debug(f"[FeedbackHub] EpisodeMemory failed: {e}")
            results["episode_memory"] = False

        # 3. PromptEvolver (variant quality tracking with model affinity)
        try:
            from vetinari.learning.prompt_evolver import get_prompt_evolver
            get_prompt_evolver().record_result(
                agent_type=task_type,
                variant_id=f"{task_type}_baseline",
                quality=quality_score,
                model_id=model_id,
            )
            results["prompt_evolver"] = True
        except Exception as e:
            logger.debug(f"[FeedbackHub] PromptEvolver failed: {e}")
            results["prompt_evolver"] = False

        # 4. AnomalyDetector (monitor for anomalous quality/latency)
        try:
            from vetinari.analytics.anomaly import get_anomaly_detector
            detector = get_anomaly_detector()
            detector.detect(f"task.quality.{task_type}", quality_score)
            if latency_ms > 0:
                detector.detect(f"task.latency.{task_type}", float(latency_ms))
            results["anomaly_detector"] = True
        except Exception as e:
            logger.debug(f"[FeedbackHub] AnomalyDetector failed: {e}")
            results["anomaly_detector"] = False

        # 5. Cost tracking
        if cost_usd > 0:
            try:
                from vetinari.analytics.cost import get_cost_tracker
                get_cost_tracker().record(
                    provider="local" if "local" in model_id.lower() else "cloud",
                    model=model_id,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=cost_usd,
                )
                results["cost_tracker"] = True
            except Exception as e:
                logger.debug(f"[FeedbackHub] CostTracker failed: {e}")
                results["cost_tracker"] = False

        return results

    def on_benchmark_complete(
        self,
        model_id: str,
        benchmark_results: List[Dict[str, Any]],
    ) -> Dict[str, bool]:
        """
        Fan out benchmark completion events to all learning subsystems.

        Args:
            model_id: Model that was benchmarked.
            benchmark_results: List of per-agent benchmark result dicts,
                each with keys: pass_rate, task_type, suite_name, n_trials, avg_score.

        Returns:
            Dict mapping subsystem name to success boolean.
        """
        results: Dict[str, bool] = {}

        # 1. FeedbackLoop (benchmark-weighted model performance)
        try:
            from vetinari.learning.feedback_loop import get_feedback_loop
            feedback = get_feedback_loop()
            for br in benchmark_results:
                feedback.record_benchmark_outcome(
                    model_id=model_id,
                    benchmark_result=br,
                )
            results["feedback_loop"] = True
        except Exception as e:
            logger.debug(f"[FeedbackHub] FeedbackLoop benchmark failed: {e}")
            results["feedback_loop"] = False

        # 2. WorkflowLearner (extract decomposition patterns)
        try:
            from vetinari.learning.workflow_learner import get_workflow_learner
            learner = get_workflow_learner()
            for br in benchmark_results:
                learner.learn_from_benchmark(br)
            results["workflow_learner"] = True
        except Exception as e:
            logger.debug(f"[FeedbackHub] WorkflowLearner benchmark failed: {e}")
            results["workflow_learner"] = False

        return results

    def on_plan_approved(
        self,
        plan_id: str,
        goal: str,
        approved: bool,
        approver: str,
        risk_score: float = 0.0,
        subtask_count: int = 0,
    ) -> Dict[str, bool]:
        """
        Fan out plan approval events.

        Args:
            plan_id: The plan identifier.
            goal: The plan's goal text.
            approved: Whether it was approved or rejected.
            approver: Who approved/rejected.
            risk_score: Plan risk score.
            subtask_count: Number of subtasks in the plan.

        Returns:
            Dict mapping subsystem name to success boolean.
        """
        results: Dict[str, bool] = {}

        # 1. FeedbackLoop
        try:
            from vetinari.learning.feedback_loop import get_feedback_loop
            get_feedback_loop().record_outcome(
                task_id=plan_id,
                model_id="plan_approval",
                task_type="plan_approval",
                quality_score=1.0 if approved else 0.0,
                success=approved,
            )
            results["feedback_loop"] = True
        except Exception as e:
            logger.debug(f"[FeedbackHub] FeedbackLoop plan approval failed: {e}")
            results["feedback_loop"] = False

        # 2. EpisodeMemory
        try:
            from vetinari.learning.episode_memory import get_episode_memory
            get_episode_memory().record(
                task_description=f"Plan approval: {goal[:200]}",
                agent_type="PLANNER",
                task_type="plan_approval",
                output_summary=f"{'Approved' if approved else 'Rejected'} by {approver}. "
                               f"Risk: {risk_score:.2f}. Subtasks: {subtask_count}",
                quality_score=1.0 if approved else 0.3,
                success=approved,
                model_id="plan_mode",
                metadata={"plan_id": plan_id, "approver": approver, "risk_score": risk_score},
            )
            results["episode_memory"] = True
        except Exception as e:
            logger.debug(f"[FeedbackHub] EpisodeMemory plan approval failed: {e}")
            results["episode_memory"] = False

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Collect stats from all connected subsystems."""
        stats: Dict[str, Any] = {}

        subsystems = [
            ("feedback_loop", "vetinari.learning.feedback_loop", "get_feedback_loop"),
            ("episode_memory", "vetinari.learning.episode_memory", "get_episode_memory"),
            ("workflow_learner", "vetinari.learning.workflow_learner", "get_workflow_learner"),
            ("prompt_evolver", "vetinari.learning.prompt_evolver", "get_prompt_evolver"),
            ("quality_scorer", "vetinari.learning.quality_scorer", "get_quality_scorer"),
        ]

        for name, module_path, getter_name in subsystems:
            try:
                import importlib
                mod = importlib.import_module(module_path)
                getter = getattr(mod, getter_name)
                instance = getter()
                if hasattr(instance, "get_stats"):
                    stats[name] = instance.get_stats()
                else:
                    stats[name] = {"status": "connected"}
            except Exception as e:
                stats[name] = {"status": "unavailable", "error": str(e)}

        return stats


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_feedback_hub: Optional[FeedbackHub] = None


def get_feedback_hub() -> FeedbackHub:
    """Return the global FeedbackHub singleton."""
    global _feedback_hub
    if _feedback_hub is None:
        _feedback_hub = FeedbackHub()
    return _feedback_hub
