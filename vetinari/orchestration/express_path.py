"""Express-path execution mixin for TwoLayerOrchestrator.

The express path bypasses planning entirely and routes simple goals directly
to a Builder task, then closes spans and correlation contexts.  This is the
Tier.EXPRESS fast lane from the intake classifier (Dept 4.1).

The ``ExpressPathMixin`` class is designed to be mixed into
``TwoLayerOrchestrator`` only.  It accesses instance attributes such as
``self._make_default_handler()`` and ``self._express_metrics`` that are
provided by that class.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from vetinari.types import StatusEnum

logger = logging.getLogger(__name__)


class ExpressPathMixin:
    """Express-path execution for simple goals (mixin for TwoLayerOrchestrator).

    Provides ``_execute_express`` and ``_record_express_metrics``.  Both
    methods rely on ``self._make_default_handler()`` which is defined on
    ``TwoLayerOrchestrator``.  Python resolves the attribute at call time so
    no forward reference is needed.
    """

    def _execute_express(
        self,
        goal: str,
        context: dict[str, Any],
        stages: dict[str, Any],
        start_time: float,
        corr_ctx: Any | None,
        pipeline_span: Any | None,
        *,
        task_handler: Callable | None = None,
    ) -> dict[str, Any]:
        """Execute Express tier: Builder -> Quality, skip planning.

        Creates a single synthetic ``TaskNode``, runs it through the provided
        (or default) handler, records metrics, and returns a standard pipeline
        result dict.  Span and correlation context are always closed in the
        ``finally`` block.

        Args:
            goal: The enriched goal string.
            context: The pipeline context.
            stages: The stages dict for recording progress.
            start_time: Pipeline start timestamp.
            corr_ctx: Optional CorrelationContext.
            pipeline_span: Optional OTel span.
            task_handler: Optional user-provided task handler.

        Returns:
            Pipeline result dict with keys ``plan_id``, ``goal``, ``backend``,
            ``tier``, ``completed``, ``failed``, ``outputs``, ``final_output``,
            ``stages``, and ``total_time_ms``.
        """
        handler = task_handler or self._make_default_handler()  # type: ignore[attr-defined]
        try:
            # Single Builder task, no planning decomposition.
            # Create a synthetic TaskNode so the handler signature is satisfied.
            from vetinari.orchestration.execution_graph import ExecutionTaskNode

            express_task = ExecutionTaskNode(
                id=f"express-{int(start_time)}", description=goal, task_type="implementation"
            )
            result = handler(express_task)
            # A handler can signal failure via a structured dict {"success": False, ...}.
            # Checking bool(result) alone would treat that truthy dict as success, so
            # we inspect the "success" key explicitly when it is present (fail-closed).
            if isinstance(result, dict) and "success" in result:
                express_success = bool(result["success"])
            else:
                express_success = bool(result)
            stages["express_execution"] = {"success": express_success}

            # Track express lane metrics for Dept 4.1
            self._record_express_metrics(express_success, start_time)

            return {
                "plan_id": f"express-{int(start_time)}",
                "goal": goal,
                "backend": "express",
                "tier": "express",
                StatusEnum.COMPLETED.value: 1 if express_success else 0,
                StatusEnum.FAILED.value: 0 if express_success else 1,
                "outputs": {"express_task": result},
                "final_output": result,
                "stages": stages,
                "total_time_ms": int((time.time() - start_time) * 1000),
            }
        except Exception as e:  # Broad: task handler is user-supplied; any failure mode is possible
            logger.warning("[Pipeline] Express execution failed: %s", e)
            stages["express_execution"] = {"success": False, "error": str(e)}
            # Track express lane failure
            self._record_express_metrics(False, start_time)
            return {
                "plan_id": f"express-{int(start_time)}",
                "goal": goal,
                "backend": "express",
                "tier": "express",
                StatusEnum.COMPLETED.value: 0,
                StatusEnum.FAILED.value: 1,
                "error": str(e),
                "stages": stages,
                "total_time_ms": int((time.time() - start_time) * 1000),
            }
        finally:
            if pipeline_span is not None:
                try:
                    from vetinari.observability.otel_genai import get_genai_tracer

                    get_genai_tracer().end_agent_span(pipeline_span, status="ok")
                except (ImportError, AttributeError):
                    logger.warning("Failed to close GenAI span", exc_info=True)
            if corr_ctx is not None:
                import contextlib as _cl

                with _cl.suppress(Exception):
                    corr_ctx.__exit__(None, None, None)

    def _record_express_metrics(self, success: bool, start_time: float) -> None:
        """Record express lane metrics for success rate tracking.

        Updates an internal ``_express_metrics`` dict (created on first call)
        with total, success, and failed counts, then logs the current rates.

        Args:
            success: Whether the express execution succeeded.
            start_time: Pipeline start timestamp for latency calculation.
        """
        try:
            latency_ms = int((time.time() - start_time) * 1000)
            # Wire WO-13: use "failed" not "promoted" — the counter tracks
            # express-lane failures, not tier promotions.  These are distinct
            # events: a promotion sends the request to the full pipeline, whereas
            # this counter increments when the express handler raises an exception.
            if not hasattr(self, "_express_metrics"):
                self._express_metrics: dict[str, int] = {"total": 0, "success": 0, StatusEnum.FAILED.value: 0}
            self._express_metrics["total"] += 1
            if success:
                self._express_metrics["success"] += 1
            else:
                self._express_metrics[StatusEnum.FAILED.value] += 1
            rate = self._express_metrics["success"] / max(self._express_metrics["total"], 1)
            logger.info(
                "[Express Metrics] total=%d, success=%d, failed=%d, success_rate=%.2f, latency_ms=%d",
                self._express_metrics["total"],
                self._express_metrics["success"],
                self._express_metrics[StatusEnum.FAILED.value],
                rate,
                latency_ms,
            )
        except (ArithmeticError, AttributeError):
            logger.warning("Express metrics recording failed", exc_info=True)
