"""Ponder (model deliberation) API handlers.

Native Litestar equivalents of the routes previously registered by
``ponder_routes``. Part of Flask->Litestar migration (ADR-0066).
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


def create_ponder_routes_handlers() -> list[Any]:
    """Return Litestar route handler instances for the ponder API.

    Returns an empty list when Litestar is not installed so the caller can
    safely call this in environments that only have Flask.

    Returns:
        List of Litestar route handler objects covering all ponder endpoints.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.responses import litestar_error_response

    @post("/api/ponder/choose-model", guards=[admin_guard])
    async def api_ponder_choose_model(data: dict[str, Any]) -> Response:
        """Rank available models for a task description using the ponder engine.

        Accepts a JSON body with ``task_description`` (required) and ``top_n``
        (default 3). Runs capability and Thompson-sampling scoring against
        candidate models and returns a ranked list.

        Args:
            data: Request body with ``task_description`` and optional ``top_n``.

        Returns:
            JSON object with ranked model results, or 400 when
            ``task_description`` is missing, or 503 when the ponder
            service is unavailable.
        """
        from vetinari.models.ponder import rank_models

        task_description = data.get("task_description", "")
        top_n = data.get("top_n", 3)

        if not task_description:
            return litestar_error_response("task_description required", code=400)

        try:
            result = rank_models(task_description, top_n)
        except Exception as e:
            logger.warning(
                "Ponder choose-model unavailable — %s",
                type(e).__name__,
                exc_info=True,
            )
            return litestar_error_response("Ponder service unavailable", 503)
        return Response(content=result, status_code=200, media_type=MediaType.JSON)

    @get("/api/ponder/templates")
    async def api_ponder_templates(
        version: str = Parameter(query="version", default="v1"),
    ) -> Response:
        """Return the ponder template prompts for a given template version.

        Instantiates a PonderEngine for the requested version and returns the
        full list of template prompt strings for UI preview.

        Args:
            version: Template version to load (default ``"v1"``).

        Returns:
            JSON Response with ``templates`` list, ``total`` count, and
            ``version`` string, or 500 if the ponder engine fails to initialise.
        """
        try:
            from vetinari.models.ponder import PonderEngine

            engine = PonderEngine()
            templates = engine.get_template_prompts()
            return Response(
                content={"templates": templates, "total": len(templates), "version": version},
                status_code=200,
                media_type=MediaType.JSON,
            )
        except Exception:
            logger.warning(
                "Failed to load ponder templates for version %s — returning error to client",
                version,
            )
            return litestar_error_response("Ponder template service unavailable", code=500)

    @get("/api/ponder/models")
    async def api_ponder_models() -> Response:
        """Return all models currently available to the ponder engine.

        Queries the ponder module's model discovery layer, which reads from
        the LM Studio server and any statically configured model entries.

        Returns:
            JSON Response with a ``models`` list and ``total`` count, or 500
            if the model discovery layer cannot be reached.
        """
        try:
            from vetinari.models.ponder import get_available_models

            models = get_available_models()
            return Response(
                content={"models": models, "total": len(models)},
                status_code=200,
                media_type=MediaType.JSON,
            )
        except Exception:
            logger.warning(
                "Failed to retrieve available ponder models — returning error to client",
            )
            return litestar_error_response("Ponder model discovery unavailable", code=500)

    @get("/api/ponder/health")
    async def api_ponder_health() -> Response:
        """Return the health status of the ponder subsystem.

        Checks that the ponder engine can reach the LM Studio server and that
        at least one model is available.

        Returns:
            JSON Response with ponder health fields, or 500 if the health
            check itself cannot complete.
        """
        try:
            from vetinari.models.ponder import get_ponder_health

            return Response(
                content=get_ponder_health(),
                status_code=200,
                media_type=MediaType.JSON,
            )
        except Exception:
            logger.warning(
                "Failed to retrieve ponder health status — returning error to client",
            )
            return litestar_error_response("Ponder health check unavailable", code=500)

    @post("/api/ponder/plan/{plan_id:str}", guards=[admin_guard])
    async def api_ponder_run_plan(plan_id: str) -> Response:
        """Run ponder model analysis for the given plan and return ranked results.

        Triggers a full ponder pass for ``plan_id``, evaluating each candidate
        model against the plan's task descriptions.

        Args:
            plan_id: Identifier of the plan to run ponder analysis on.

        Returns:
            JSON object with ponder results on success, 400 when the ponder
            pass reports failure, or 500 when the ponder engine raises.
        """
        try:
            from vetinari.models.ponder import ponder_project_for_plan

            result = ponder_project_for_plan(plan_id)
            if not result.get("success", False):
                return Response(content=result, status_code=400, media_type=MediaType.JSON)
            return Response(content=result, status_code=200, media_type=MediaType.JSON)
        except Exception:
            logger.warning(
                "Failed to run ponder analysis for plan %s — returning error to client",
                plan_id,
            )
            return litestar_error_response("Ponder analysis failed", code=500)

    @get("/api/ponder/plan/{plan_id:str}")
    async def api_ponder_get_plan(plan_id: str) -> Response:
        """Retrieve previously stored ponder results for a plan.

        Looks up cached ponder output for ``plan_id`` so the UI can display
        model rankings from a prior analysis pass without re-running inference.

        Args:
            plan_id: Identifier of the plan whose ponder results to retrieve.

        Returns:
            JSON Response with ponder results, or 500 if the results store
            cannot be reached.
        """
        try:
            from vetinari.models.ponder import get_ponder_results_for_plan

            return Response(
                content=get_ponder_results_for_plan(plan_id),
                status_code=200,
                media_type=MediaType.JSON,
            )
        except Exception:
            logger.warning(
                "Failed to retrieve ponder results for plan %s — returning error to client",
                plan_id,
            )
            return litestar_error_response("Ponder results unavailable", code=500)

    return [
        api_ponder_choose_model,
        api_ponder_templates,
        api_ponder_models,
        api_ponder_health,
        api_ponder_run_plan,
        api_ponder_get_plan,
    ]
