"""Plan execution visualization API — native Litestar handlers. Native Litestar equivalents (ADR-0066). URL paths identical to Flask.

Provides DAG rendering, cost accumulation, quality gate indicators, and
human-in-the-loop approval endpoints for plan execution monitoring.

Endpoints:
    GET  /api/plans/{plan_id}/visualization        — DAG structure for a plan
    GET  /api/plans/{plan_id}/visualization/stream — SSE real-time updates
    POST /api/plans/{plan_id}/approve-gate         — approve/reject a quality gate
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Response, get, post
    from litestar.response import ServerSentEvent

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_visualization_handlers() -> list[Any]:
    """Create all Litestar route handlers for the plan visualization API.

    Called by ``vetinari.web.litestar_app.create_app()`` to register these
    handlers in the main Litestar application.

    Returns:
        List of Litestar route handler functions, or empty list when
        Litestar is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        logger.debug("Litestar not available — visualization API handlers not registered")
        return []

    # -- GET /api/plans/{plan_id}/visualization --------------------------------

    @get("/api/plans/{plan_id:str}/visualization", media_type=MediaType.JSON)
    async def api_plan_visualization(plan_id: str) -> dict[str, Any]:
        """Return the DAG structure for a plan as JSON.

        Includes nodes (tasks), edges (dependencies), and summary statistics
        for cost, progress, and quality gates.  Enriches with subtask
        breakdowns from ``PlanModeEngine`` when available.

        Args:
            plan_id: The plan identifier.

        Returns:
            JSON DAG structure with optional subtasks_by_depth and
            subtasks_by_domain enrichment fields.

        Raises:
            FileNotFoundError: When the plan is not found (mapped to 404).
        """
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.visualization import PlanVisualizationBuilder

        try:
            from vetinari.planning import get_plan_manager

            plan_manager = get_plan_manager()
            plan = plan_manager.get_plan(plan_id)
            if not plan:
                return litestar_error_response(  # type: ignore[return-value]
                    "Plan not found", 404
                )

            builder = PlanVisualizationBuilder()
            dag = builder.build_from_plan(plan)
            response: dict[str, Any] = dag.to_dict()

            # Enrich with subtask breakdowns from plan_types.Plan when available.
            try:
                from vetinari.planning.plan_mode import PlanModeEngine
                from vetinari.planning.plan_types import TaskDomain

                typed_plan = PlanModeEngine().get_plan(plan_id)
                if typed_plan is not None:
                    max_depth = max((s.depth for s in typed_plan.subtasks), default=0)
                    subtasks_by_depth: dict[str, list[dict[str, Any]]] = {}
                    for depth in range(max_depth + 1):
                        subtasks_at_depth = typed_plan.get_subtasks_by_depth(depth)
                        if subtasks_at_depth:
                            subtasks_by_depth[str(depth)] = [s.to_dict() for s in subtasks_at_depth]

                    subtasks_by_domain: dict[str, list[dict[str, Any]]] = {}
                    for domain in TaskDomain:
                        subtasks_in_domain = typed_plan.get_subtasks_by_domain(domain)
                        if subtasks_in_domain:
                            subtasks_by_domain[domain.value] = [s.to_dict() for s in subtasks_in_domain]

                    response["subtasks_by_depth"] = subtasks_by_depth
                    response["subtasks_by_domain"] = subtasks_by_domain
            except Exception:
                logger.warning(
                    "Subtask breakdown unavailable for plan %s — skipping enrichment",
                    plan_id,
                )

            return response
        except Exception:
            logger.exception("Failed to build plan visualization for %s", plan_id)
            return litestar_error_response("Internal server error", 500)  # type: ignore[return-value]

    # -- GET /api/plans/{plan_id}/visualization/stream (SSE) ------------------

    @get("/api/plans/{plan_id:str}/visualization/stream", sync_to_thread=False)
    def api_plan_visualization_stream(plan_id: str) -> Response | ServerSentEvent:
        """SSE endpoint for real-time plan execution visualization updates.

        Checks that the plan exists before opening the SSE stream so clients
        receive a 404 JSON response immediately rather than an empty stream
        that never emits any meaningful events.

        Emits events: task_started, task_completed, task_failed,
        quality_gate_result, quality_gate_pending, cost_update.  A connected
        event is sent immediately on subscription.  Keepalive heartbeats are
        emitted when the queue is idle.

        Args:
            plan_id: The plan identifier.

        Returns:
            SSE event stream, or a 404 JSON Response when the plan is not found.
        """
        from vetinari.web.responses import litestar_error_response

        # Validate plan exists before opening the SSE stream — clients should get a
        # clean 404 rather than an empty event stream that never progresses.
        try:
            from vetinari.planning import get_plan_manager

            plan_manager = get_plan_manager()
            if not plan_manager.get_plan(plan_id):
                return litestar_error_response("Plan not found", code=404)
        except Exception:
            logger.warning(
                "Could not verify plan %s exists for SSE stream — returning 404 to client",
                plan_id,
            )
            return litestar_error_response("Plan not found", code=404)

        from vetinari.web.visualization import _get_viz_queue, _remove_viz_queue

        async def _generate():  # type: ignore[return]
            """Yield SSE frames from the per-plan visualization queue.

            Yields:
                Dicts with ``data`` key for real events, or ``comment`` key
                for keepalives — both understood by Litestar's SSE encoder.
            """
            import asyncio
            import json
            import queue as _queue

            from vetinari.constants import SSE_MESSAGE_TIMEOUT

            q = _get_viz_queue(plan_id)
            try:
                yield {"data": json.dumps({"type": "connected", "plan_id": plan_id})}

                loop = asyncio.get_running_loop()
                while True:
                    try:
                        deadline = loop.time() + SSE_MESSAGE_TIMEOUT
                        while True:
                            try:
                                msg = q.get_nowait()
                                break
                            except _queue.Empty:
                                remaining = deadline - loop.time()
                                if remaining <= 0:
                                    raise
                                await asyncio.sleep(min(0.1, remaining))
                        if msg is None:
                            yield {"data": json.dumps({"type": "done"})}
                            break
                        yield {"event": msg["event"], "data": msg["data"]}
                    except _queue.Empty:
                        yield {"comment": "keepalive"}
            finally:
                _remove_viz_queue(plan_id, q)

        return ServerSentEvent(
            _generate(),
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # -- POST /api/plans/{plan_id}/approve-gate --------------------------------

    @post("/api/plans/{plan_id:str}/approve-gate", media_type=MediaType.JSON)
    async def api_approve_gate(
        plan_id: str, data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Approve or reject a quality gate checkpoint for human-in-the-loop control.

        Request body: ``{"action": "approve" | "reject", "reason": "..."}``

        Args:
            plan_id: The plan identifier.
            data: JSON request body with ``action`` and optional ``reason``.

        Returns:
            Updated gate status JSON with ``plan_id``, ``gate_status``,
            ``task_id``, and ``action`` fields.
        """
        import time as _time

        from vetinari.web.request_validation import json_object_body
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.visualization import (
            _pending_gates,
            _pending_gates_lock,
            push_visualization_event,
        )

        try:
            body = json_object_body(data)
            if body is None:
                return litestar_error_response(  # type: ignore[return-value]
                    "Request body must be a JSON object", 400
                )
            action = body.get("action", "")
            reason = body.get("reason", "")

            if action not in ("approve", "reject"):
                return litestar_error_response(  # type: ignore[return-value]
                    "action must be 'approve' or 'reject'", 400
                )

            with _pending_gates_lock:
                gate = _pending_gates.get(plan_id)
                if not gate:
                    return litestar_error_response(  # type: ignore[return-value]
                        "No pending quality gate for this plan", 404
                    )

                # Verify the underlying plan still exists — a gate for an orphaned
                # plan_id would mutate state that can never be consumed.
                try:
                    from vetinari.planning import get_plan_manager

                    _pm = get_plan_manager()
                    if not _pm.get_plan(plan_id):
                        return litestar_error_response(  # type: ignore[return-value]
                            f"Plan '{plan_id}' not found — cannot approve gate for nonexistent plan",
                            code=404,
                        )
                except Exception as exc:
                    logger.warning(
                        "Could not verify plan %s exists during gate approval — rejecting request: %s",
                        plan_id,
                        exc,
                    )
                    return litestar_error_response(  # type: ignore[return-value]
                        "Could not verify plan exists", code=500
                    )

                gate["status"] = "approved" if action == "approve" else "rejected"
                gate["reason"] = reason
                gate["resolved_at"] = _time.time()

            # Emit SSE event for the gate decision.
            push_visualization_event(
                plan_id,
                "quality_gate_result",
                {
                    "plan_id": plan_id,
                    "task_id": gate["task_id"],
                    "action": action,
                    "reason": reason,
                },
            )

            # Pause the plan if the gate was rejected.
            if action == "reject":
                try:
                    from vetinari.planning import get_plan_manager

                    plan_manager = get_plan_manager()
                    plan = plan_manager.get_plan(plan_id)
                    if plan:
                        plan.status = "paused"
                        plan_manager._save_plan(plan)
                except Exception as exc:
                    logger.warning(
                        "Could not pause plan %s after gate rejection — plan state may be stale: %s",
                        plan_id,
                        exc,
                    )

            return {
                "plan_id": plan_id,
                "gate_status": gate["status"],
                "task_id": gate["task_id"],
                "action": action,
            }
        except Exception:
            logger.exception("Failed to process gate approval for %s", plan_id)
            return litestar_error_response("Internal server error", 500)  # type: ignore[return-value]

    return [
        api_plan_visualization,
        api_plan_visualization_stream,
        api_approve_gate,
    ]
