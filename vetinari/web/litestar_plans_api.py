"""Plan management and lifecycle handlers.

Native Litestar equivalents of the routes previously registered by
``vetinari.web.plans_api``. Part of Flask->Litestar migration (ADR-0066).
URL paths are identical to the Flask originals.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, delete, get, post, put
    from litestar.params import Parameter
    from litestar.response import Response

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_plans_api_handlers() -> list[Any]:
    """Create Litestar handlers for plan management and lifecycle transitions.

    Replicates the eleven routes from ``vetinari.web.plans_api``:
    ``POST /api/v1/plan`` (PlanModeEngine), ``POST /api/v1/plans``,
    ``GET /api/v1/plans``, ``GET /api/v1/plans/{plan_id}``,
    ``PUT /api/v1/plans/{plan_id}``, ``DELETE /api/v1/plans/{plan_id}``,
    ``POST /api/v1/plans/{plan_id}/start``,
    ``POST /api/v1/plans/{plan_id}/pause``,
    ``POST /api/v1/plans/{plan_id}/resume``,
    ``POST /api/v1/plans/{plan_id}/cancel``, and
    ``GET /api/v1/plans/{plan_id}/status``.

    Returns an empty list when Litestar is not installed, so the factory is
    safe to call in Flask-only environments.

    Returns:
        List of Litestar route handler objects ready to register on a Router
        or Application.  Empty when Litestar is unavailable.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.responses import litestar_error_response
    from vetinari.web.shared import get_orchestrator, validate_path_param

    @post("/api/v1/plan", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_create_plan(data: dict[str, Any]) -> Any:
        """Create a plan using the PlanModeEngine planning engine.

        Admin-only.

        Args:
            data: Request body with required ``goal`` (str) and optional
                ``system_prompt`` (str) keys.

        Returns:
            JSON with ``status`` and ``plan`` dict on success, or HTTP 400
            when the goal is missing or no models are available.
        """
        goal: str = data.get("goal", "")
        if not goal:
            return litestar_error_response("goal is required", 400)

        if not isinstance(goal, str):
            return litestar_error_response("goal must be a string", 422)

        from vetinari.planning.plan_mode import PlanModeEngine
        from vetinari.planning.plan_types import PlanGenerationRequest

        # Attempt model discovery but do not hard-fail when no models are
        # available — fall through to the planner which may have a fallback.
        try:
            orb = get_orchestrator()
            orb.model_pool.discover_models()
        except Exception:
            logger.warning("api_create_plan: model pool discovery failed — proceeding with planner fallback")

        engine = PlanModeEngine()
        plan = engine.generate_plan(PlanGenerationRequest(goal=goal))
        return {"status": "ok", "plan": plan.to_dict()}

    @post("/api/v1/plans", media_type=MediaType.JSON, guards=[admin_guard], status_code=201)
    async def api_plan_create(data: dict[str, Any]) -> dict[str, Any]:
        """Create a new plan via the plan manager.

        Admin-only.

        Args:
            data: Request body.  Supported fields:
                title (str): Plan title.
                prompt (str): Plan prompt / description.
                created_by (str): Creator identifier (default ``"user"``).
                waves (list): Optional wave data for the plan.

        Returns:
            Plan dict with HTTP 201 on success.
        """
        title = data.get("title", "")
        prompt = data.get("prompt", "")

        if not title or not isinstance(title, str):
            return litestar_error_response("title must be a non-empty string", 422)
        if not isinstance(prompt, str):
            return litestar_error_response("prompt must be a string", 422)

        from vetinari.planning import get_plan_manager

        plan_manager = get_plan_manager()
        plan = plan_manager.create_plan(
            title=title,
            prompt=prompt,
            created_by=data.get("created_by", "user"),
            waves_data=data.get("waves"),
        )
        return plan.to_dict()

    @get("/api/v1/plans", media_type=MediaType.JSON)
    async def api_plans_list(
        status: str | None = Parameter(query="status", default=None),
        limit: int = Parameter(query="limit", default=50),
        offset: int = Parameter(query="offset", default=0),
    ) -> dict[str, Any]:
        """Return a paginated list of plans, optionally filtered by status.

        Args:
            status: Optional status filter.
            limit: Maximum number of plans to return (default 50, max 200).
            offset: Number of plans to skip (default 0).

        Returns:
            JSON with ``plans``, ``total``, ``limit``, and ``offset`` keys.
        """
        from vetinari.planning import get_plan_manager

        plan_manager = get_plan_manager()

        # Clamp to the same bounds as the Flask original
        limit = max(1, min(limit, 200))
        offset = max(0, min(offset, 10000))

        # Compute total from the full unsliced filtered set, not the page.
        # list_plans() returns only the requested page, so len(page) != total.
        all_plans = list(plan_manager.plans.values())
        if status:
            all_plans = [p for p in all_plans if p.status == status]
        total = len(all_plans)

        plans = plan_manager.list_plans(status=status, limit=limit, offset=offset)
        return {
            "plans": [p.to_dict() for p in plans],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    @get("/api/v1/plans/{plan_id:str}", media_type=MediaType.JSON)
    async def api_plan_get(plan_id: str) -> Any:
        """Retrieve a single plan by its identifier.

        Args:
            plan_id: The plan identifier.

        Returns:
            Plan dict on success, or HTTP 404 when not found.
        """
        from vetinari.planning import get_plan_manager

        plan_manager = get_plan_manager()
        plan = plan_manager.get_plan(plan_id)
        if not plan:
            return litestar_error_response("Plan not found", 404)
        return plan.to_dict()

    @put("/api/v1/plans/{plan_id:str}", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_plan_update(plan_id: str, data: dict[str, Any]) -> Any:
        """Update an existing plan's fields.

        Admin-only.

        Args:
            plan_id: The plan identifier.
            data: Request body containing any plan fields to update.

        Returns:
            Updated plan dict on success, HTTP 400 for traversal-style
            identifiers, or HTTP 404 when not found.
        """
        if not validate_path_param(plan_id):
            return litestar_error_response("Invalid plan_id", 400)

        from vetinari.planning import get_plan_manager

        plan_manager = get_plan_manager()
        plan = plan_manager.update_plan(plan_id, data)
        if not plan:
            return litestar_error_response("Plan not found", 404)
        return plan.to_dict()

    @delete("/api/v1/plans/{plan_id:str}", guards=[admin_guard], status_code=200)
    async def api_plan_delete(plan_id: str) -> Response:
        """Delete a plan permanently.

        Admin-only.

        Args:
            plan_id: The plan identifier.

        Returns:
            Empty response with HTTP 204 on success, HTTP 400 for
            traversal-style identifiers, or HTTP 404 when not found.
        """
        if not validate_path_param(plan_id):
            return litestar_error_response("Invalid plan_id", 400)

        from vetinari.planning import get_plan_manager

        plan_manager = get_plan_manager()
        if not plan_manager.delete_plan(plan_id):
            return litestar_error_response("Plan not found", 404)
        return Response(content=None, status_code=204)

    @post(
        "/api/v1/plans/{plan_id:str}/start",
        media_type=MediaType.JSON,
        guards=[admin_guard],
    )
    async def api_plan_start(plan_id: str) -> Any:
        """Transition a plan to the running state.

        Admin-only.

        Args:
            plan_id: The plan identifier.

        Returns:
            JSON with ``plan_id``, ``status``, and ``started_at`` on success,
            or HTTP 404 when not found.
        """
        from vetinari.planning import get_plan_manager

        plan_manager = get_plan_manager()
        plan = plan_manager.start_plan(plan_id)
        if not plan:
            return litestar_error_response("Plan not found", 404)
        return {"plan_id": plan.plan_id, "status": plan.status, "started_at": plan.updated_at}

    @post(
        "/api/v1/plans/{plan_id:str}/pause",
        media_type=MediaType.JSON,
        guards=[admin_guard],
    )
    async def api_plan_pause(plan_id: str) -> Any:
        """Pause a running plan.

        Admin-only.

        Args:
            plan_id: The plan identifier.

        Returns:
            JSON with ``plan_id``, ``status``, and ``paused_at`` on success,
            or HTTP 404 when not found.
        """
        from vetinari.planning import get_plan_manager

        plan_manager = get_plan_manager()
        plan = plan_manager.pause_plan(plan_id)
        if not plan:
            return litestar_error_response("Plan not found", 404)
        # X-Compatibility-Only: true signals that pause is a metadata-only
        # operation; no agent work is interrupted.
        return Response(
            content={"plan_id": plan.plan_id, "status": plan.status, "paused_at": plan.updated_at},
            headers={"X-Compatibility-Only": "true"},
            status_code=200,
            media_type=MediaType.JSON,
        )

    @post(
        "/api/v1/plans/{plan_id:str}/resume",
        media_type=MediaType.JSON,
        guards=[admin_guard],
    )
    async def api_plan_resume(plan_id: str) -> Any:
        """Resume a paused plan.

        Admin-only.

        Args:
            plan_id: The plan identifier.

        Returns:
            JSON with ``plan_id``, ``status``, and ``resumed_at`` on success,
            or HTTP 404 when not found.
        """
        from vetinari.planning import get_plan_manager

        plan_manager = get_plan_manager()
        plan = plan_manager.resume_plan(plan_id)
        if not plan:
            return litestar_error_response("Plan not found", 404)
        # X-Compatibility-Only: true signals that resume is a metadata-only
        # operation; no agent work is started immediately by this call.
        return Response(
            content={"plan_id": plan.plan_id, "status": plan.status, "resumed_at": plan.updated_at},
            headers={"X-Compatibility-Only": "true"},
            status_code=200,
            media_type=MediaType.JSON,
        )

    @post(
        "/api/v1/plans/{plan_id:str}/cancel",
        media_type=MediaType.JSON,
        guards=[admin_guard],
    )
    async def api_plan_cancel(plan_id: str) -> Any:
        """Cancel an active plan.

        Admin-only.

        Args:
            plan_id: The plan identifier.

        Returns:
            JSON with ``plan_id``, ``status``, and ``cancelled_at`` on
            success, or HTTP 404 when not found.
        """
        from vetinari.planning import get_plan_manager

        plan_manager = get_plan_manager()
        plan = plan_manager.cancel_plan(plan_id)
        if not plan:
            return litestar_error_response("Plan not found", 404)
        # X-Compatibility-Only: true signals that cancel is a metadata-only
        # operation; no in-flight agent work is interrupted by this call.
        return Response(
            content={"plan_id": plan.plan_id, "status": plan.status, "cancelled_at": plan.updated_at},
            headers={"X-Compatibility-Only": "true"},
            status_code=200,
            media_type=MediaType.JSON,
        )

    @get("/api/v1/plans/{plan_id:str}/status", media_type=MediaType.JSON)
    async def api_plan_status(plan_id: str) -> Any:
        """Return a concise progress snapshot for a plan.

        Args:
            plan_id: The plan identifier.

        Returns:
            JSON with ``plan_id``, ``status``, ``current_wave``,
            ``completed_tasks``, ``running_tasks``, ``pending_tasks``,
            ``failed_tasks``, and ``progress_percent``, or HTTP 404 when
            not found.
        """
        from vetinari.planning import get_plan_manager
        from vetinari.types import StatusEnum

        plan_manager = get_plan_manager()
        plan = plan_manager.get_plan(plan_id)
        if not plan:
            return litestar_error_response("Plan not found", 404)

        return {
            "plan_id": plan.plan_id,
            "status": plan.status,
            "current_wave": plan.current_wave.wave_id if plan.current_wave else None,
            "completed_tasks": plan.completed_tasks,
            "running_tasks": sum(1 for w in plan.waves for t in w.tasks if t.status == StatusEnum.RUNNING.value),
            "pending_tasks": sum(1 for w in plan.waves for t in w.tasks if t.status == StatusEnum.PENDING.value),
            "failed_tasks": sum(1 for w in plan.waves for t in w.tasks if t.status == StatusEnum.FAILED.value),
            "progress_percent": plan.progress_percent,
        }

    return [
        api_create_plan,
        api_plan_create,
        api_plans_list,
        api_plan_get,
        api_plan_update,
        api_plan_delete,
        api_plan_start,
        api_plan_pause,
        api_plan_resume,
        api_plan_cancel,
        api_plan_status,
    ]
