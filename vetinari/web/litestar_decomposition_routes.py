"""Task decomposition API handlers.

Native Litestar equivalents of the routes previously registered by
``decomposition_routes``. Part of Flask->Litestar migration (ADR-0066).
URL paths identical to Flask originals.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Response, get, post
    from litestar.params import Parameter

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_decomposition_routes_handlers() -> list[Any]:
    """Return Litestar route handler instances for the decomposition API.

    Returns an empty list when Litestar is not installed so the caller can
    safely call this in environments that only have Flask.

    Returns:
        List of Litestar route handler objects covering all decomposition endpoints.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.responses import litestar_error_response

    @get("/api/v1/decomposition/templates")
    async def api_decomposition_templates(
        keywords: str | None = Parameter(query="keywords", default=None),
        agent_type: str | None = Parameter(query="agent_type", default=None),
        dod_level: str | None = Parameter(query="dod_level", default=None),
    ) -> dict[str, Any]:
        """List available task decomposition templates.

        Optionally filtered by keywords (comma-separated), agent type, or
        DoD level.

        Args:
            keywords: Comma-separated keyword filter string.
            agent_type: Filter templates by the target agent type.
            dod_level: Filter templates by Definition-of-Done level.

        Returns:
            Dict with ``templates`` list and ``total`` count.
            Returns 503 when the decomposition engine is unavailable.
        """
        try:
            from vetinari.planning.decomposition import decomposition_engine

            keyword_list = keywords.split(",") if keywords else None
            templates = decomposition_engine.get_templates(
                keywords=keyword_list,
                agent_type=agent_type,
                dod_level=dod_level,
            )
            serialized = [t if isinstance(t, dict) else t.__dict__ for t in templates]
            return {"templates": serialized, "total": len(templates)}
        except Exception as exc:
            logger.warning("api_decomposition_templates: decomposition engine unavailable — returning 503: %s", exc)
            return litestar_error_response("Decomposition system unavailable", code=503)

    @get("/api/v1/decomposition/dod-dor")
    async def api_decomposition_dod_dor(
        level: str = Parameter(query="level", default="Standard"),
    ) -> dict[str, Any]:
        """Return Definition of Done and Definition of Ready criteria for a quality level.

        Args:
            level: Quality level — one of ``Light``, ``Standard``, or ``Hard``.

        Returns:
            Dict with ``dod_criteria``, ``dor_criteria``, and available ``levels``.
            Returns 503 when the decomposition engine is unavailable.
        """
        try:
            from vetinari.planning.decomposition import decomposition_engine

            return {
                "dod_criteria": decomposition_engine.get_dod_criteria(level),
                "dor_criteria": decomposition_engine.get_dor_criteria(level),
                "levels": ["Light", "Standard", "Hard"],
            }
        except Exception as exc:
            logger.warning("api_decomposition_dod_dor: decomposition engine unavailable — returning 503: %s", exc)
            return litestar_error_response("Decomposition system unavailable", code=503)

    @get("/api/v1/decomposition/knobs")
    async def api_decomposition_knobs() -> dict[str, Any]:
        """Return recursion control knobs including seed mix, seed rate, and depth bounds.

        Returns:
            Dict with ``recursion_knobs``, ``seed_mix``, ``seed_rate``,
            ``default_max_depth``, ``min_max_depth``, and ``max_max_depth``.
            Returns 503 when the decomposition agent module is unavailable.
        """
        try:
            from vetinari.agents.decomposition_agent import (
                DEFAULT_MAX_DEPTH,
                MAX_MAX_DEPTH,
                MIN_MAX_DEPTH,
                RECURSION_KNOBS,
                SEED_MIX,
                SEED_RATE,
            )

            return {
                "recursion_knobs": RECURSION_KNOBS,
                "seed_mix": SEED_MIX,
                "seed_rate": SEED_RATE,
                "default_max_depth": DEFAULT_MAX_DEPTH,
                "min_max_depth": MIN_MAX_DEPTH,
                "max_max_depth": MAX_MAX_DEPTH,
            }
        except Exception as exc:
            logger.warning("api_decomposition_knobs: decomposition agent module unavailable — returning 503: %s", exc)
            return litestar_error_response("Decomposition system unavailable", code=503)

    @get("/api/v1/decomposition/history")
    async def api_decomposition_history(
        plan_id: str | None = Parameter(query="plan_id", default=None),
    ) -> dict[str, Any]:
        """Return decomposition history events, optionally filtered by plan ID.

        Args:
            plan_id: Optional plan identifier to filter history events.

        Returns:
            Dict with ``history`` list of event dicts and ``total`` count.
            Returns 503 when the decomposition engine is unavailable.
        """
        try:
            from vetinari.planning.decomposition import decomposition_engine

            history = decomposition_engine.get_decomposition_history(plan_id)
            return {
                "history": [
                    {
                        "event_id": e.event_id,
                        "plan_id": e.plan_id,
                        "task_id": e.task_id,
                        "depth": e.depth,
                        "seeds_used": e.seeds_used,
                        "subtasks_created": e.subtasks_created,
                        "timestamp": e.timestamp,
                    }
                    for e in history
                ],
                "total": len(history),
            }
        except Exception as exc:
            logger.warning("api_decomposition_history: decomposition engine unavailable — returning 503: %s", exc)
            return litestar_error_response("Decomposition system unavailable", code=503)

    @get("/api/v1/decomposition/seed-config")
    async def api_decomposition_seed_config() -> dict[str, Any]:
        """Return the current seed mix ratios and depth limits from the decomposition engine.

        Returns:
            Dict with ``seed_mix``, ``seed_rate``, ``default_max_depth``,
            ``min_max_depth``, and ``max_max_depth``.
            Returns 503 when the decomposition engine is unavailable.
        """
        try:
            from vetinari.planning.decomposition import decomposition_engine

            return {
                "seed_mix": decomposition_engine.SEED_MIX,
                "seed_rate": decomposition_engine.SEED_RATE,
                "default_max_depth": decomposition_engine.DEFAULT_MAX_DEPTH,
                "min_max_depth": decomposition_engine.MIN_MAX_DEPTH,
                "max_max_depth": decomposition_engine.MAX_MAX_DEPTH,
            }
        except Exception as exc:
            logger.warning("api_decomposition_seed_config: decomposition engine unavailable — returning 503: %s", exc)
            return litestar_error_response("Decomposition system unavailable", code=503)

    @post("/api/v1/decomposition/decompose", guards=[admin_guard])
    async def api_decomposition_decompose(data: dict[str, Any]) -> Response:
        """Decompose a task prompt into subtasks using the decomposition engine.

        Args:
            data: Request body with ``task_prompt``, optional ``parent_task_id``,
                ``depth``, ``max_depth``, and ``plan_id``.

        Returns:
            Dict with ``subtasks``, ``count``, ``depth``, and ``max_depth``,
            or 400 when depth/max_depth are non-numeric.
        """
        from vetinari.planning.decomposition import decomposition_engine
        from vetinari.web.request_validation import body_depth_exceeded, body_has_oversized_key
        from vetinari.web.responses import litestar_error_response

        if body_depth_exceeded(data):
            return litestar_error_response("Request body nesting depth exceeds maximum", code=400)
        if body_has_oversized_key(data):
            return litestar_error_response("Request body contains oversized key", code=400)

        # task_prompt is required — reject bodies that omit it or provide empty string
        if "task_prompt" not in data:
            return litestar_error_response("task_prompt is required", code=422)
        task_prompt = data["task_prompt"]
        if not isinstance(task_prompt, str):
            return litestar_error_response("task_prompt must be a string", code=422)
        if not task_prompt.strip():
            return litestar_error_response("task_prompt must not be empty", code=422)

        parent_task_id = data.get("parent_task_id", "root")
        try:
            depth = int(data.get("depth", 0))
            max_depth = int(data.get("max_depth", 14))
        except (ValueError, TypeError):
            logger.warning("Decomposition handler received non-numeric depth/max_depth — returning 400")
            return litestar_error_response("depth and max_depth must be integers", code=400)

        plan_id = data.get("plan_id", "default")

        # Clamp max_depth to the supported range [12, 16]
        if max_depth < 12:
            max_depth = 12
        elif max_depth > 16:
            max_depth = 16

        subtasks = decomposition_engine.decompose_task(
            task_prompt=task_prompt,
            parent_task_id=parent_task_id,
            depth=depth,
            max_depth=max_depth,
            plan_id=plan_id,
        )
        result = {"subtasks": subtasks, "count": len(subtasks), "depth": depth, "max_depth": max_depth}
        return Response(content=result, status_code=200, media_type=MediaType.JSON)

    @post("/api/v1/decomposition/decompose-agent", guards=[admin_guard])
    async def api_decomposition_decompose_agent(data: dict[str, Any]) -> Response:
        """Decompose a task using the decomposition agent, creating a plan if needed.

        Args:
            data: Request body with ``plan_id`` (required) and optional ``prompt``.

        Returns:
            Dict with the agent decomposition result, or 400 when ``plan_id``
            is missing.
        """
        from vetinari.agents.decomposition_agent import decomposition_agent
        from vetinari.planning import get_plan_manager

        plan_manager = get_plan_manager()
        plan_id = data.get("plan_id")
        prompt = data.get("prompt", "")

        if not plan_id:
            return litestar_error_response("plan_id required", code=400)
        if not isinstance(plan_id, str):
            return litestar_error_response("plan_id must be a string", code=422)

        plan = plan_manager.get_plan(plan_id)
        if not plan:
            return litestar_error_response("Plan not found", code=404)

        result = decomposition_agent.decompose_from_prompt(plan, prompt)
        return Response(content=result, status_code=200, media_type=MediaType.JSON)

    return [
        api_decomposition_templates,
        api_decomposition_dod_dor,
        api_decomposition_knobs,
        api_decomposition_history,
        api_decomposition_seed_config,
        api_decomposition_decompose,
        api_decomposition_decompose_agent,
    ]
