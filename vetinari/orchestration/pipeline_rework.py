"""RCA-driven rework routing for the pipeline quality layer.

Contains ``ReworkDecision`` and the two mixin methods that handle quality
rejections: ``_handle_quality_rejection`` (root-cause analysis to decision)
and ``_execute_rework_decision`` (decision to concrete corrective action).

These are extracted from ``pipeline_quality.py`` to keep that file under
the 550-line ceiling.  They are composed back into ``PipelineQualityMixin``
via ``pipeline_quality.py``'s mixin inheritance.

Not part of the public API — access via ``PipelineQualityMixin``.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from vetinari.constants import MAX_REWORK_CYCLES
from vetinari.types import StatusEnum

logger = logging.getLogger(__name__)


class ReworkDecision(Enum):
    """Decision on how to handle a quality rejection (Dept 7.8).

    Each value represents a distinct rework routing path based on root
    cause analysis of the defect.
    """

    RETRY_SAME_AGENT = "retry_same_agent"
    RETRY_DIFFERENT_MODEL = "retry_different_model"
    REPLAN = "replan"
    REPLAN_WIDER_SCOPE = "replan_wider_scope"
    RESEARCH_THEN_RETRY = "research_then_retry"
    ESCALATE_TO_USER = "escalate_to_user"


class PipelineReworkMixin:
    """Mixin providing RCA-driven rework routing for the pipeline.

    Mixed into ``PipelineQualityMixin`` (and therefore into
    ``TwoLayerOrchestrator``). Methods access ``self`` attributes such as
    ``execution_engine`` and ``_route_model_for_task`` that are defined
    on ``TwoLayerOrchestrator``.
    """

    def _select_rework_by_quality_score(
        self,
        task_id: str,
        quality_score: float,
    ) -> ReworkDecision:
        """Route rework based on quality score percentiles (ADR: quality-conditioned rework).

        Higher scores get lighter interventions, lower scores get heavier ones.
        This complements RCA-based routing by using the quality *magnitude*
        rather than just the root cause *category*.

        Score bands:
          0.70-0.85 -> RETRY_SAME_AGENT with critique feedback
          0.50-0.70 -> RETRY_DIFFERENT_MODEL (try a stronger model)
          0.30-0.50 -> RESEARCH_THEN_RETRY (gather more context first)
          < 0.30    -> ESCALATE_TO_USER (too low for automated recovery)

        Args:
            task_id: The failed task's ID (for logging).
            quality_score: Quality score between 0.0 and 1.0.

        Returns:
            A ReworkDecision based on the quality score band.
        """
        if quality_score >= 0.70:
            decision = ReworkDecision.RETRY_SAME_AGENT
        elif quality_score >= 0.50:
            decision = ReworkDecision.RETRY_DIFFERENT_MODEL
        elif quality_score >= 0.30:
            decision = ReworkDecision.RESEARCH_THEN_RETRY
        else:
            decision = ReworkDecision.ESCALATE_TO_USER

        logger.info(
            "[QualityRework] Task %s quality_score=%.3f -> %s",
            task_id,
            quality_score,
            decision.value,
        )
        return decision

    def _handle_quality_rejection(
        self,
        task_id: str,
        result: dict[str, Any],
        rework_count: int,
    ) -> ReworkDecision:
        """Route corrective action based on root cause analysis AND quality score.

        First checks quality score for score-based routing (13.4). If no
        quality score is present, falls back to RCA category routing.

        Args:
            task_id: The failed task's ID.
            result: The quality result dict containing root_cause metadata.
            rework_count: Number of times this task has been reworked.

        Returns:
            A ReworkDecision enum value indicating the rework routing path.
        """
        if rework_count >= MAX_REWORK_CYCLES:
            logger.warning(
                "[RCA] Max rework cycles reached for task %s (rework_count=%d) — escalating",
                task_id,
                rework_count,
            )
            return ReworkDecision.ESCALATE_TO_USER

        # Quality-score-based routing (13.4) takes precedence when score available
        if isinstance(result, dict):
            quality_score = result.get("quality_score") or result.get("score")
            if isinstance(quality_score, (int, float)) and 0.0 <= quality_score <= 1.0:
                return self._select_rework_by_quality_score(task_id, float(quality_score))

        root_cause = result.get("root_cause") if isinstance(result, dict) else None

        if not root_cause:
            logger.info(
                "[RCA] No root_cause in result for task %s — defaulting to retry_same_agent",
                task_id,
            )
            return ReworkDecision.RETRY_SAME_AGENT

        category = root_cause.get("category", "")

        _routing: dict[str, ReworkDecision] = {
            "bad_spec": ReworkDecision.REPLAN,
            "wrong_model": ReworkDecision.RETRY_DIFFERENT_MODEL,
            "hallucination": ReworkDecision.RETRY_SAME_AGENT,
            "context": ReworkDecision.RESEARCH_THEN_RETRY,
            "integration": ReworkDecision.REPLAN_WIDER_SCOPE,
            "complexity": ReworkDecision.REPLAN,
            "prompt": ReworkDecision.RETRY_SAME_AGENT,
        }

        decision = _routing.get(category, ReworkDecision.RETRY_SAME_AGENT)
        logger.info(
            "[RCA] Task %s quality rejection routed: category=%s, decision=%s",
            task_id,
            category,
            decision.value,
        )

        # Log rework/retry decision to audit trail (US-023)
        try:
            from vetinari.audit import get_audit_logger

            get_audit_logger().log_decision(
                decision_type="retry_action",
                choice=decision.value,
                reasoning=f"Root cause category '{category}' for task {task_id} (rework #{rework_count})",
                alternatives=[d.value for d in ReworkDecision if d != decision],
                context={
                    "task_id": task_id,
                    "root_cause_category": category,
                    "rework_count": rework_count,
                },
            )
        except Exception:
            logger.warning("Audit logging failed during rework routing", exc_info=True)

        return decision

    def _execute_rework_decision(
        self,
        decision: ReworkDecision,
        task_id: str,
        task_result: Any,
        graph: Any,
        task_handler: Any | None = None,
    ) -> dict[str, Any] | None:
        """Execute a rework routing decision for a failed task.

        Translates the abstract ReworkDecision into concrete recovery action:
        retry the same task, reassign to a different model, replan, or
        escalate. Returns the rework outcome or None if no action was taken.

        Args:
            decision: The ReworkDecision from root cause analysis.
            task_id: The failed task's identifier.
            task_result: The result metadata from the failed execution.
            graph: The execution graph containing the task.
            task_handler: Optional handler for re-executing tasks.

        Returns:
            Dict with ``action``, ``task_id``, and ``outcome`` keys, or None
            if the decision is ESCALATE_TO_USER (requires human intervention).
        """
        node = graph.nodes.get(task_id)
        if node is None:
            logger.warning("[Rework] Task %s not found in graph — skipping", task_id)
            return None

        if decision == ReworkDecision.ESCALATE_TO_USER:
            logger.info("[Rework] Task %s escalated to user — no automatic action", task_id)
            return {"action": "escalate", "task_id": task_id, "outcome": "awaiting_user"}

        if decision in (
            ReworkDecision.RETRY_SAME_AGENT,
            ReworkDecision.RESEARCH_THEN_RETRY,
        ):
            # Reset the task for retry with enriched context from rejection
            node.status = StatusEnum.PENDING
            rejection_feedback = ""
            if isinstance(task_result, dict):
                rejection_feedback = task_result.get("summary", "") or task_result.get("reason", "")
                root_cause = task_result.get("root_cause", {})
                if isinstance(root_cause, dict):
                    corrective = root_cause.get("corrective_action", "")
                    if corrective:
                        rejection_feedback += f" Corrective action: {corrective}"

            # Generate LLM retry brief with 3 specific changes (~300 tokens)
            retry_brief = ""
            try:
                from vetinari.llm_helpers import generate_retry_brief

                retry_brief = (
                    generate_retry_brief(
                        error_description=rejection_feedback or "Task output rejected",
                        inspector_feedback=rejection_feedback,
                    )
                    or ""
                )
            except Exception:
                logger.warning("LLM retry brief unavailable — using raw rejection feedback without briefing")

            combined_feedback = rejection_feedback
            if retry_brief:
                combined_feedback = f"{rejection_feedback}\n\nRETRY BRIEF:\n{retry_brief}"

            if combined_feedback:
                node.input_data["rework_feedback"] = combined_feedback
                node.description = (
                    f"{node.description}\n\nPREVIOUS ATTEMPT FAILED — apply this feedback:\n{combined_feedback}"
                )
            node.error = ""
            if decision == ReworkDecision.RESEARCH_THEN_RETRY:
                node.input_data["rework_hint"] = "research_context_before_retry"
            logger.info(
                "[Rework] Task %s reset for retry (decision=%s, feedback=%s)",
                task_id,
                decision.value,
                bool(rejection_feedback),
            )

        elif decision == ReworkDecision.RETRY_DIFFERENT_MODEL:
            # Reset and request a different model assignment
            node.status = StatusEnum.PENDING
            node.error = ""
            current_model = node.input_data.get("assigned_model", "default")
            node.input_data["excluded_models"] = [current_model]
            new_model = self._route_model_for_task(node)  # type: ignore[attr-defined]
            node.input_data["assigned_model"] = new_model
            node.input_data["rework_feedback"] = f"Previous model ({current_model}) produced rejected output"
            logger.info(
                "[Rework] Task %s reassigned from %s to %s",
                task_id,
                current_model,
                new_model,
            )

            # Log model swap decision (US-023)
            try:
                from vetinari.audit import get_audit_logger

                get_audit_logger().log_decision(
                    decision_type="model_swap",
                    choice=str(new_model),
                    reasoning=f"Switched from {current_model} after quality rejection on task {task_id}",
                    alternatives=[current_model],
                    context={"task_id": task_id, "previous_model": current_model},
                )
            except Exception:
                logger.warning("Audit logging failed during model swap", exc_info=True)

        elif decision in (ReworkDecision.REPLAN, ReworkDecision.REPLAN_WIDER_SCOPE):
            # Replan: regenerate subtasks for the failed task's scope
            logger.info("[Rework] Replanning for task %s (decision=%s)", task_id, decision.value)
            node.status = StatusEnum.PENDING
            node.error = ""
            if decision == ReworkDecision.REPLAN_WIDER_SCOPE:
                node.input_data["rework_hint"] = "widen_scope"

        # Re-execute the single reset task via the durable engine
        handler = (
            task_handler
            or self.execution_engine._task_handlers.get(node.task_type)  # type: ignore[attr-defined]
            or self.execution_engine._task_handlers.get("default")  # type: ignore[attr-defined]
        )

        if handler is None:
            logger.warning("[Rework] No handler available for task %s — cannot re-execute", task_id)
            return {"action": decision.value, "task_id": task_id, "outcome": "no_handler"}

        result = self.execution_engine._execute_task(graph, node)  # type: ignore[attr-defined]
        self.execution_engine._save_checkpoint(graph.plan_id, graph)  # type: ignore[attr-defined]

        outcome = result.get("status", "unknown")
        logger.info("[Rework] Task %s rework complete: action=%s, outcome=%s", task_id, decision.value, outcome)
        return {"action": decision.value, "task_id": task_id, "outcome": outcome}
