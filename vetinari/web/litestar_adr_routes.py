"""ADR (Architecture Decision Records) API handlers.

Native Litestar equivalents of the routes previously registered by ``adr_routes``.
Part of Flask->Litestar migration (ADR-0066). URL paths identical to Flask originals.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Response, get, post, put
    from litestar.params import Parameter

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_adr_routes_handlers() -> list[Any]:
    """Return Litestar route handler instances for the ADR API.

    Returns an empty list when Litestar is not installed so the caller can
    safely call this in environments that only have Flask.

    Returns:
        List of Litestar route handler objects covering all ADR endpoints.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.responses import litestar_error_response

    @get("/api/adr")
    async def api_adr_list(
        status: str | None = Parameter(query="status", default=None),
        category: str | None = Parameter(query="category", default=None),
        limit: int = Parameter(query="limit", default=100, ge=1, le=500),
    ) -> Response:
        """List ADRs with optional filtering by status, category, and limit.

        Args:
            status: Filter by ADR lifecycle status (e.g. ``"accepted"``).
            category: Filter by decision category (e.g. ``"architecture"``).
            limit: Maximum number of results to return (default 100, max 500).

        Returns:
            JSON Response with ``adrs`` list and ``total`` count, or 500 when
            the ADR system cannot be reached.
        """
        try:
            from vetinari.adr import get_adr_system

            adrs = get_adr_system().list_adrs(status=status, category=category, limit=limit)
            return Response(
                content={"adrs": [a.to_dict() for a in adrs], "total": len(adrs)},
                status_code=200,
                media_type=MediaType.JSON,
            )
        except Exception:
            logger.warning("ADR system unavailable for list — returning error to client")
            return litestar_error_response("ADR system unavailable", code=500)

    @get("/api/adr/statistics")
    async def api_adr_statistics() -> Response:
        """Return aggregate statistics across all ADRs (counts by status and category).

        Returns:
            JSON Response with counts broken down by status and category, or
            500 when the ADR system cannot be reached.
        """
        try:
            from vetinari.adr import get_adr_system

            return Response(
                content=get_adr_system().get_statistics(),
                status_code=200,
                media_type=MediaType.JSON,
            )
        except Exception:
            logger.warning("ADR system unavailable for statistics — returning error to client")
            return litestar_error_response("ADR system unavailable", code=500)

    @get("/api/adr/recent")
    async def api_adr_recent(
        n: int = Parameter(query="n", default=5, ge=1, le=50),
    ) -> Response:
        """Return the most recently created ADRs for quick contextual reference.

        Args:
            n: Number of recent ADRs to return (default 5, max 50).

        Returns:
            JSON Response with ``adrs`` list and ``total`` count, or 500 when
            the ADR system cannot be reached.
        """
        try:
            from vetinari.adr import get_adr_system

            adrs = get_adr_system().get_recent_decisions(n)
            return Response(
                content={"adrs": [a.to_dict() for a in adrs], "total": len(adrs)},
                status_code=200,
                media_type=MediaType.JSON,
            )
        except Exception:
            logger.warning("ADR system unavailable for recent decisions — returning error to client")
            return litestar_error_response("ADR system unavailable", code=500)

    @get("/api/adr/is-high-stakes")
    async def api_adr_is_high_stakes(
        category: str = Parameter(query="category", default="architecture"),
    ) -> Response:
        """Check whether a given ADR category is considered high-stakes.

        Args:
            category: The ADR category to check (default ``"architecture"``).

        Returns:
            JSON Response with ``is_high_stakes`` boolean and the queried
            ``category``, or 500 when the ADR system cannot be reached.
        """
        try:
            from vetinari.adr import get_adr_system

            is_high_stakes = get_adr_system().is_high_stakes(category)
            return Response(
                content={"is_high_stakes": is_high_stakes, "category": category},
                status_code=200,
                media_type=MediaType.JSON,
            )
        except Exception:
            logger.warning(
                "ADR system unavailable for high-stakes check on category %s — returning error to client",
                category,
            )
            return litestar_error_response("ADR system unavailable", code=500)

    @get("/api/adr/{adr_id:str}")
    async def api_adr_get(adr_id: str) -> Response:
        """Retrieve a single ADR by its ID.

        Args:
            adr_id: The unique identifier of the ADR to retrieve.

        Returns:
            JSON representation of the ADR, or 404 if not found, or 500 when
            the ADR system cannot be reached.
        """
        try:
            from vetinari.adr import get_adr_system

            adr = get_adr_system().get_adr(adr_id)
            if not adr:
                return litestar_error_response("ADR not found", code=404)
            return Response(content=adr.to_dict(), status_code=200, media_type=MediaType.JSON)
        except Exception:
            logger.warning(
                "ADR system unavailable for get %s — returning error to client",
                adr_id,
            )
            return litestar_error_response("ADR system unavailable", code=500)

    @post("/api/adr", guards=[admin_guard])
    async def api_adr_create(data: dict[str, Any]) -> Response:
        """Create a new ADR from the supplied JSON body.

        Requires admin privileges. The request body must include ``title``.
        Optional fields: ``category``, ``context``, ``decision``,
        ``consequences``, ``created_by``.

        Args:
            data: Request body dict containing ADR fields.

        Returns:
            JSON representation of the newly created ADR, or an error response.
        """
        from vetinari.adr import get_adr_system

        title_val = data.get("title")
        if not isinstance(title_val, str) or not title_val:
            return litestar_error_response("'title' must be a non-empty string", code=400)

        adr = get_adr_system().create_adr(
            title=title_val,
            category=data.get("category", "architecture"),
            context=data.get("context", ""),
            decision=data.get("decision", ""),
            consequences=data.get("consequences", ""),
            created_by=data.get("created_by", "user"),
        )
        return Response(content=adr.to_dict(), status_code=200, media_type=MediaType.JSON)

    # Fields the caller may update — read-only system fields (adr_id, created_at,
    # updated_at) are excluded so unknown-key-only bodies are correctly rejected.
    _ADR_UPDATABLE_FIELDS: frozenset[str] = frozenset({
        "title",
        "category",
        "context",
        "decision",
        "status",
        "consequences",
        "related_adrs",
        "created_by",
        "notes",
    })

    @put("/api/adr/{adr_id:str}", guards=[admin_guard])
    async def api_adr_update(adr_id: str, data: dict[str, Any]) -> Response:
        """Update fields on an existing ADR.

        Requires admin privileges. Only the fields supplied in the JSON body
        are updated; omitted fields are left unchanged.

        Args:
            adr_id: The unique identifier of the ADR to update.
            data: Request body dict containing the fields to update.  Must
                contain at least one recognized ADR field (title, category,
                context, decision, status, consequences, related_adrs,
                created_by, or notes); bodies with only unknown keys return 400.

        Returns:
            JSON representation of the updated ADR, or 400 if no valid fields
            are present, or 404 if not found.
        """
        recognized = {k: v for k, v in data.items() if k in _ADR_UPDATABLE_FIELDS}
        if not recognized:
            return litestar_error_response(
                "Request body must contain at least one updatable ADR field "
                "(title, category, context, decision, status, consequences, related_adrs, created_by, or notes)",
                code=400,
            )
        from vetinari.adr import get_adr_system

        adr = get_adr_system().update_adr(adr_id, recognized)
        if not adr:
            return litestar_error_response("ADR not found", code=404)
        return Response(content=adr.to_dict(), status_code=200, media_type=MediaType.JSON)

    # Only field accepted in the deprecate body — everything else is rejected.
    _DEPRECATE_VALID_FIELDS: frozenset[str] = frozenset({"replacement_id"})

    @post("/api/adr/{adr_id:str}/deprecate", guards=[admin_guard])
    async def api_adr_deprecate(adr_id: str, data: dict[str, Any]) -> Response:
        """Mark an ADR as deprecated, optionally linking a replacement ADR.

        Requires admin privileges. Accepts an optional ``replacement_id`` in the
        JSON body to record which newer ADR supersedes this one.

        Args:
            adr_id: The unique identifier of the ADR to deprecate.
            data: Request body dict optionally containing ``replacement_id``.

        Returns:
            JSON representation of the deprecated ADR, or 404 if not found.
        """
        from vetinari.adr import get_adr_system

        # Reject empty bodies and bodies that contain only unrecognised keys —
        # callers that send {} or {"foo": "bar"} would otherwise silently
        # deprecate the ADR with no replacement_id, masking typos and
        # wrong-endpoint mistakes.
        if not data:
            return litestar_error_response(
                "Request body must not be empty — provide 'replacement_id' to link a superseding ADR",
                code=422,
            )
        unknown_keys = set(data.keys()) - _DEPRECATE_VALID_FIELDS
        if unknown_keys and "replacement_id" not in data:
            return litestar_error_response(
                "Request body may only contain 'replacement_id' (a string ADR identifier)",
                code=400,
            )

        replacement_id = data.get("replacement_id")
        # replacement_id must be a string ADR identifier or absent — reject dicts/lists
        # which would silently pass through and corrupt the deprecation record.
        if replacement_id is not None and not isinstance(replacement_id, str):
            return litestar_error_response(
                "replacement_id must be a string ADR identifier (e.g. 'ADR-0042') or null",
                code=400,
            )

        adr = get_adr_system().deprecate_adr(adr_id, replacement_id)
        if not adr:
            return litestar_error_response("ADR not found", code=404)
        return Response(content=adr.to_dict(), status_code=200, media_type=MediaType.JSON)

    @post("/api/adr/propose", guards=[admin_guard])
    async def api_adr_propose(data: dict[str, Any]) -> Response:
        """Generate an ADR proposal with options from a free-text context description.

        Requires admin privileges. The JSON body must include a non-empty
        ``context`` describing the decision space, and may include ``num_options``
        (default 3).

        Args:
            data: Request body dict with ``context`` and optional ``num_options``.

        Returns:
            JSON Response with ``question``, ``options``, ``recommended`` index, and
            ``rationale`` for the recommended option, or 400 if ``context`` is missing.
        """
        from vetinari.adr import get_adr_system

        if not data:
            return litestar_error_response("Request body must not be empty", code=400)
        context = data.get("context", "")
        # context must be a non-empty string — without it the proposal engine has
        # nothing to reason about and would generate meaningless options.
        if not context or not isinstance(context, str):
            return litestar_error_response(
                "Missing required field: context (must be a non-empty string describing the decision space)",
                code=400,
            )

        num_options_raw = data.get("num_options", 3)
        if not isinstance(num_options_raw, int):
            return litestar_error_response("'num_options' must be an integer", code=422)
        num_options = num_options_raw
        proposal = get_adr_system().generate_proposal(context, num_options)
        return Response(
            content={
                "question": proposal.question,
                "options": proposal.options,
                "recommended": proposal.recommended,
                "rationale": proposal.rationale,
            },
            status_code=200,
            media_type=MediaType.JSON,
        )

    @post("/api/adr/propose/accept", guards=[admin_guard])
    async def api_adr_propose_accept(data: dict[str, Any]) -> Response:
        """Accept a previously generated proposal and persist it as a new ADR.

        Requires admin privileges. The JSON body must include non-empty
        ``question`` and ``title`` fields, plus ``options``, ``recommended``
        index, and ``category``.

        Args:
            data: Request body dict with proposal fields plus ``title`` and
                ``category``.

        Returns:
            JSON representation of the newly created ADR, or 400 if required
            fields are missing or empty.
        """
        from vetinari.adr import ADRProposal, get_adr_system

        if not data:
            return litestar_error_response("Request body must not be empty", code=400)

        question = data.get("question", "")
        title = data.get("title", "")

        # Both question and title must be non-empty strings — an empty question or
        # default title produces a meaningless ADR record in the decision log.
        if not question or not isinstance(question, str):
            return litestar_error_response(
                "Missing required field: question (must be a non-empty string)",
                code=400,
            )
        if not title or not isinstance(title, str):
            return litestar_error_response(
                "Missing required field: title (must be a non-empty string)",
                code=400,
            )

        options = data.get("options", [])
        recommended = data.get("recommended", 0)
        category = data.get("category", "architecture")

        proposal = ADRProposal(question=question, options=options, recommended=recommended)
        adr = get_adr_system().accept_proposal(proposal, title, category)
        return Response(content=adr.to_dict(), status_code=200, media_type=MediaType.JSON)

    @post("/api/adr/from-plan", guards=[admin_guard])
    async def api_add_adr_from_plan(data: dict[str, Any]) -> Response:
        """Create an ADR through the plan engine's add_adr method.

        Expects JSON with ``adr_id``, ``title``, ``context``, ``decision``,
        and optional ``status`` (defaults to ``"proposed"``).

        Args:
            data: Request body dict containing required ADR fields.

        Returns:
            JSON with the created ADR metadata, or 400 if required fields missing.
        """
        missing = [f for f in ("adr_id", "title", "context", "decision") if not data.get(f)]
        if missing:
            return litestar_error_response(f"Missing required fields: {', '.join(missing)}", code=400)

        # Validate each required field is actually a string — dict/list values pass the
        # truthiness check above but would corrupt the ADR record downstream.
        non_string = [f for f in ("adr_id", "title", "context", "decision") if not isinstance(data.get(f), str)]
        if non_string:
            return litestar_error_response(
                f"Fields must be strings: {', '.join(non_string)}",
                code=400,
            )

        try:
            from vetinari.planning.planning import get_plan_engine

            engine = get_plan_engine()
            result = engine.add_adr(
                adr_id=data["adr_id"],
                title=data["title"],
                context=data["context"],
                decision=data["decision"],
                status=data.get("status", "proposed"),
            )
            return Response(content=result, status_code=201, media_type=MediaType.JSON)
        except Exception:
            logger.warning("Failed to create ADR via plan engine", exc_info=True)
            return litestar_error_response("ADR creation failed", code=500)

    @post("/api/adr/{adr_id:str}/link-plan", guards=[admin_guard])
    async def api_adr_link_plan(adr_id: str, data: dict[str, Any]) -> Response:
        """Link an ADR to a plan by recording it in the plan's ADR history.

        Requires admin privileges. The JSON body must include ``plan_id``.
        Optional fields: ``title``, ``context``, ``decision``, ``status``.

        Args:
            adr_id: The ADR identifier to link (e.g. ``ADR-0042``).
            data: Request body dict with ``plan_id`` and optional override fields.

        Returns:
            JSON representation of the newly created ADR history entry, or an
            error response if the ADR or plan is not found.
        """
        from vetinari.adr import get_adr_system
        from vetinari.planning.planning import PlanManager

        adr = get_adr_system().get_adr(adr_id)
        if not adr:
            return litestar_error_response(f"ADR '{adr_id}' not found", code=404)

        plan_id = data.get("plan_id", "")
        # plan_id must be a non-empty string — a list or dict value would crash
        # manager.plans.get() with a TypeError when used as a dict key.
        if not isinstance(plan_id, str) or not plan_id:
            return litestar_error_response(
                "Missing required field: plan_id (must be a non-empty string)",
                code=400,
            )

        manager = PlanManager.get_instance()
        plan = manager.plans.get(plan_id)
        if plan is None:
            return litestar_error_response(f"Plan '{plan_id}' not found", code=404)

        adr_status = adr.status.value if hasattr(adr.status, "value") else str(adr.status)
        entry = plan.add_adr(
            adr_id=adr_id,
            title=data.get("title", adr.title),
            context=data.get("context", adr.context),
            decision=data.get("decision", adr.decision),
            status=data.get("status", adr_status),
        )
        return Response(content=entry, status_code=200, media_type=MediaType.JSON)

    return [
        api_adr_list,
        api_adr_statistics,
        api_adr_recent,
        api_adr_is_high_stakes,
        api_adr_get,
        api_adr_create,
        api_adr_update,
        api_adr_deprecate,
        api_adr_propose,
        api_adr_propose_accept,
        api_add_adr_from_plan,
        api_adr_link_plan,
    ]
