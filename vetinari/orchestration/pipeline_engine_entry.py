"""Pipeline engine entry helpers for event emission and queue admission.

This module keeps the public ``PipelineEngineMixin`` small by holding the
pipeline's notification hook and request-admission entrypoint. The admitted
request still flows back into ``PipelineEngineMixin._execute_pipeline`` for
the full stage ordering.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from vetinari.orchestration.pipeline_events import (
    EventSeverity,
    PipelineEvent,
    PipelineStage,
)
from vetinari.orchestration.request_routing import (
    PRIORITY_CUSTOM,
    PRIORITY_EXPRESS,
    PRIORITY_STANDARD,
    QueueFullError,
    RequestQueue,
)

logger = logging.getLogger(__name__)

# Map pipeline stages to value stream timing events for analytics wiring.
# Stages not listed here emit no timing event.
_STAGE_TO_TIMING: dict[PipelineStage, str] = {
    PipelineStage.INTAKE: "task_queued",
    PipelineStage.PLAN_GEN: "task_dispatched",
    PipelineStage.EXECUTION: "task_dispatched",
    PipelineStage.REVIEW: "task_completed",
    PipelineStage.ASSEMBLY: "task_completed",
}


class _PipelineEngineEntryMixin:
    """Provide pipeline event emission and request queue admission."""

    def _emit(
        self,
        stage: PipelineStage,
        event_type: str,
        data: dict[str, Any] | None = None,
        severity: EventSeverity = EventSeverity.INFO,
    ) -> None:
        """Emit a pipeline event to the registered event handler.

        Args:
            stage: Which pipeline stage is emitting.
            event_type: Discriminator within the stage.
            data: Optional payload dict.
            severity: Event severity level.
        """
        try:
            event_data = dict(data) if data is not None else {}
            trace_id = getattr(self, "_current_trace_id", None)
            if trace_id and "trace_id" not in event_data:
                event_data["trace_id"] = trace_id
            event = PipelineEvent(
                stage=stage,
                event_type=event_type,
                data=event_data,
                severity=severity,
            )
            self._event_handler.on_event(event)  # type: ignore[attr-defined]
        except Exception:
            logger.warning("Event emission failed for %s/%s", stage.value, event_type, exc_info=True)

        timing_event = _STAGE_TO_TIMING.get(stage, "")
        if timing_event:
            try:
                from vetinari.analytics.wiring import record_pipeline_event

                event_payload = data if data is not None else {}
                execution_id = str(event_payload.get("exec_id", ""))
                if not execution_id:
                    execution_id = "unknown"
                record_pipeline_event(
                    execution_id=execution_id,
                    task_id=str(event_payload.get("task_id", "")),
                    agent_type=str(event_payload.get("agent_type", "")),
                    timing_event=timing_event,
                    metadata={"stage": stage.value, "event_type": event_type},
                )
            except Exception as exc:
                logger.warning("Pipeline event recording skipped for stage %s: %s", stage, exc)

    def generate_and_execute(
        self,
        goal: str,
        constraints: dict[str, Any] | None = None,
        task_handler: Callable[..., Any] | None = None,
        context: dict[str, Any] | None = None,
        project_id: str | None = None,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """Run the assembly-line pipeline for one user goal.

        Args:
            goal: User goal to execute.
            constraints: Optional planning constraints.
            task_handler: Optional task execution callback.
            context: Optional mutable pipeline context.
            project_id: Optional project identifier.
            model_id: Optional model identifier for routing and tracing.

        Returns:
            Pipeline result dict with plan, execution, review, final output,
            stage details, and optional verification data.
        """
        stages: dict[str, Any] = {}
        start_time = time.time()
        if not context:
            context = {}

        correlation_context = None
        try:
            from vetinari.structured_logging import CorrelationContext

            correlation_context = CorrelationContext()
            correlation_context.__enter__()
        except (ImportError, AttributeError):
            logger.warning("Failed to initialize CorrelationContext for pipeline", exc_info=True)

        pipeline_span = None
        try:
            from vetinari.observability.otel_genai import get_genai_tracer

            genai_tracer = get_genai_tracer()
            pipeline_span = genai_tracer.start_agent_span(
                agent_name="pipeline",
                operation="orchestrate",
                model=model_id if model_id is not None else "",
            )
            pipeline_span.attributes["goal"] = goal[:200]
        except (ImportError, AttributeError):
            logger.warning("GenAI tracer unavailable for pipeline span")

        intake_tier = None
        intake_features = None
        try:
            from vetinari.orchestration.intake import get_request_intake

            intake = get_request_intake()
            intake_tier, intake_features = intake.classify_with_features(goal, context)
            context["intake_tier"] = intake_tier.value
            context["intake_confidence"] = intake_features.confidence
            context["intake_pattern_key"] = intake_features.pattern_key
            stages["intake"] = {
                "tier": intake_tier.value,
                "confidence": intake_features.confidence,
                "word_count": intake_features.word_count,
                "cross_cutting": intake_features.cross_cutting_keywords,
            }
            logger.info(
                "[Pipeline] Stage 0: Intake classified as %s (confidence=%.2f)",
                intake_tier.value,
                intake_features.confidence,
            )
        except Exception:
            logger.warning("Intake classification unavailable, proceeding with full pipeline", exc_info=True)

        if not hasattr(self, "_request_queue"):
            self._request_queue = RequestQueue()  # type: ignore[attr-defined]
        queue_priority = PRIORITY_STANDARD
        if intake_tier is not None:
            try:
                from vetinari.orchestration.intake import Tier

                tier_priority_map = {
                    Tier.EXPRESS: PRIORITY_EXPRESS,
                    Tier.STANDARD: PRIORITY_STANDARD,
                    Tier.CUSTOM: PRIORITY_CUSTOM,
                }
                queue_priority = tier_priority_map.get(intake_tier, PRIORITY_STANDARD)
            except Exception:
                logger.warning("Intake tier priority mapping failed", exc_info=True)
        try:
            exec_id = self._request_queue.enqueue(goal, context, priority=queue_priority)  # type: ignore[attr-defined]
        except QueueFullError:
            logger.warning("[Pipeline] Backpressure: request rejected (queue full)")
            return {
                "status": "rejected",
                "error": "too_many_requests",
                "message": "Server is at capacity. Please retry later.",
                "http_status": 429,
            }
        context["_exec_id"] = exec_id

        dequeued = self._request_queue.dequeue()  # type: ignore[attr-defined]
        if dequeued is None:
            logger.info("[Pipeline] Request %s queued (at concurrency limit)", exec_id)
            return {
                "status": "queued",
                "exec_id": exec_id,
                "queue_depth": self._request_queue.depth,  # type: ignore[attr-defined]
                "active_count": self._request_queue.active_count,  # type: ignore[attr-defined]
            }

        try:
            return self._execute_pipeline(
                goal,
                constraints,
                context,
                stages,
                start_time,
                correlation_context,
                pipeline_span,
                intake_tier,
                intake_features,
                task_handler,
                project_id,
                model_id,
            )
        finally:
            self._request_queue.complete(exec_id)  # type: ignore[attr-defined]
