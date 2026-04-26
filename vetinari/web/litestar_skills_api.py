"""Litestar handlers for skill catalog routes. Native Litestar equivalents (ADR-0066).

URL paths identical to Flask skills_api.py.  Only routes not already covered
by the inline ``_create_skills_handlers`` factory in ``litestar_app.py`` are
included here:

    GET  /api/v1/skills/catalog                  — list all catalog entries
    GET  /api/v1/skills/capabilities             — filter entries by capability
    GET  /api/v1/skills/catalog/{agent}          — catalog entries for one agent
    GET  /api/v1/skills/tags                     — filter catalog entries by tag
    GET  /api/v1/skills/summaries                — Level-1 summaries for all registry skills
    GET  /api/v1/skills/{skill_id}/summary       — Level-1 summary for one registry skill
    GET  /api/v1/skills/{skill_id}/trust          — 4-gate trust elevation check
    POST /api/v1/skills/{skill_id}/validate      — run output validators against payload
    POST /api/v1/skills/propose                  — submit a new skill proposal
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


def create_skills_api_handlers() -> list[Any]:
    """Create Litestar handlers for skill catalog and registry routes.

    Covers catalog filtering, registry summaries, trust elevation checks,
    output validation, and skill proposals.

    Returns:
        List of Litestar route handler functions, or an empty list when
        Litestar is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.responses import litestar_error_response

    @get("/api/v1/skills/catalog", media_type=MediaType.JSON)
    async def list_catalog() -> Response | list[dict[str, Any]]:
        """List all skill catalog entries parsed from SKILL.md files.

        Absolute on-disk ``file_path`` or ``path`` values on each entry are
        replaced with just the filename so internal directory structure is not
        leaked to callers.

        Returns:
            JSON array of catalog entry dictionaries with relativized paths.
        """
        try:
            from vetinari.skills.catalog_loader import _ensure_loaded

            catalog = _ensure_loaded()
        except Exception:
            logger.exception(
                "Skill catalog load failed — returning 500",
            )
            from vetinari.web.responses import litestar_error_response

            return litestar_error_response("Skills system unavailable", code=500)

        entries = []
        for entry in catalog.values():
            d = entry.to_dict()
            # Strip absolute path prefixes so internal filesystem layout is
            # not exposed to API callers.
            for key in ("file_path", "path"):
                raw = d.get(key)
                if isinstance(raw, str) and (raw.startswith("/") or (len(raw) > 2 and raw[1] == ":")):
                    from pathlib import Path as _Path

                    d[key] = _Path(raw).name
            entries.append(d)
        return entries

    @get("/api/v1/skills/capabilities", media_type=MediaType.JSON)
    async def get_by_capability(
        capability: str = Parameter(query="capability", default=""),
    ) -> Response | list[dict[str, Any]]:
        """Return catalog entries that declare a specific capability.

        Args:
            capability: The capability string to filter by (query parameter).

        Returns:
            JSON array of matching catalog entry dicts, or 400 if the
            ``capability`` query parameter is missing.
        """
        stripped = capability.strip()
        if not stripped:
            return Response(
                content={"error": "Missing required query parameter: capability"},
                status_code=400,
                media_type=MediaType.JSON,
            )

        try:
            from vetinari.skills.catalog_loader import get_catalog_by_capability

            entries = get_catalog_by_capability(stripped)
        except Exception:
            logger.exception(
                "Skill catalog capability filter failed for %r — returning 500",
                stripped,
            )
            from vetinari.web.responses import litestar_error_response

            return litestar_error_response("Skills system unavailable", code=500)

        return [e.to_dict() for e in entries]

    @get("/api/v1/skills/tags", media_type=MediaType.JSON)
    async def get_by_tag(
        tag: str = Parameter(query="tag", default=""),
    ) -> Response | list[dict[str, Any]]:
        """Return catalog entries that carry a specific tag.

        Args:
            tag: Tag string to filter by (query parameter).

        Returns:
            JSON array of matching catalog entry dicts, or 400 if the
            ``tag`` query parameter is missing.
        """
        stripped = tag.strip()
        if not stripped:
            return Response(
                content={"error": "Missing required query parameter: tag"},
                status_code=400,
                media_type=MediaType.JSON,
            )

        try:
            from vetinari.skills.catalog_loader import get_catalog_by_tag

            entries = get_catalog_by_tag(stripped)
        except Exception:
            logger.exception(
                "Skill catalog tag filter failed for %r — returning 500",
                stripped,
            )
            from vetinari.web.responses import litestar_error_response

            return litestar_error_response("Skills system unavailable", code=500)

        return [e.to_dict() for e in entries]

    @get("/api/v1/skills/catalog/{agent:str}", media_type=MediaType.JSON)
    async def get_catalog_for_agent(agent: str) -> Response | list[dict[str, Any]]:
        """Return all catalog entries for a specific agent.

        Args:
            agent: Agent name (foreman, worker, or inspector).

        Returns:
            JSON array of catalog entry dictionaries for the given agent.
        """
        try:
            from vetinari.skills.catalog_loader import get_catalog_by_agent

            entries = get_catalog_by_agent(agent)
        except Exception:
            logger.exception(
                "Skill catalog agent filter failed for agent %r — returning 500",
                agent,
            )
            from vetinari.web.responses import litestar_error_response

            return litestar_error_response("Skills system unavailable", code=500)

        return [e.to_dict() for e in entries]

    @get("/api/v1/skills/summaries", media_type=MediaType.JSON)
    async def list_summaries() -> Response | list[dict[str, str]]:
        """Return Level-1 summaries (id, name, description, trust_tier) for all registry skills.

        Suitable for agent startup context injection where token budget is tight.

        Returns:
            JSON array of summary dicts.
        """
        try:
            from vetinari.skills.skill_registry_convenience import list_skill_summaries

            return list_skill_summaries()
        except Exception:
            logger.exception(
                "Skill summaries list failed — returning 500",
            )
            from vetinari.web.responses import litestar_error_response

            return litestar_error_response("Skills system unavailable", code=500)

    @get("/api/v1/skills/{skill_id:str}/summary", media_type=MediaType.JSON)
    async def get_summary(skill_id: str) -> Response | dict[str, str]:
        """Return Level-1 summary for a single registry skill.

        Args:
            skill_id: The skill identifier (foreman, worker, or inspector).

        Returns:
            Summary dict with id, name, description, trust_tier — or 404 if
            the skill is not found.
        """
        try:
            from vetinari.skills.skill_registry_convenience import get_skill_summary

            summary = get_skill_summary(skill_id)
        except Exception:
            logger.exception(
                "Skill summary lookup failed for skill %r — returning 500",
                skill_id,
            )
            from vetinari.web.responses import litestar_error_response

            return litestar_error_response("Skills system unavailable", code=500)

        if summary is None:
            return Response(
                content={"error": f"Skill '{skill_id}' not found"},
                status_code=404,
                media_type=MediaType.JSON,
            )
        return summary

    @get("/api/v1/skills/{skill_id:str}/trust", media_type=MediaType.JSON)
    async def check_trust_elevation(skill_id: str) -> Response | dict[str, Any]:
        """Run the 4-gate trust elevation verification chain for a skill.

        Gates: G1 static metadata, G2 capability declarations, G3 output
        schema, G4 resource limits.

        Args:
            skill_id: The skill identifier to verify.

        Returns:
            Dict with ``overall_pass``, ``gate_results``, and ``current_tier``,
            or 404 when the skill is not found in the registry.
        """
        try:
            from vetinari.skills.skill_registry_convenience import verify_skill_trust_elevation

            result = verify_skill_trust_elevation(skill_id)
        except Exception:
            logger.exception(
                "Trust elevation check failed for skill %r — returning 500",
                skill_id,
            )
            from vetinari.web.responses import litestar_error_response

            return litestar_error_response("Skills system unavailable", code=500)

        # Return 404 for missing skills, consistent with get_summary and get_validation_detail.
        # verify_trust_elevation returns {"overall_pass": False, "error": "Skill X not found"}
        # when the skill_id is absent from SKILL_REGISTRY. Expose that as 404 rather than
        # a 200 with an inline error field so callers get consistent not-found semantics.
        if result is None or "error" in result:
            return Response(
                content={"error": f"Skill '{skill_id}' not found"},
                status_code=404,
                media_type=MediaType.JSON,
            )
        return result

    @post("/api/v1/skills/{skill_id:str}/validate", media_type=MediaType.JSON)
    async def validate_output(skill_id: str, data: dict[str, Any]) -> Response | dict[str, Any]:
        """Run registered output validators for a skill against a provided payload.

        The ``output`` key must be present in the request body — an empty body
        ``{}`` is rejected with 400 because there is nothing to validate.

        Args:
            skill_id: The skill whose validators to run.
            data: Request body containing ``output`` key with the value to validate.

        Returns:
            Dict with ``passed`` (bool) and ``failures`` (list of error strings).
            Returns 400 when the required ``output`` field is absent from the body.
        """
        # Require explicit output field — an absent key means there is nothing to validate.
        if "output" not in data:
            return Response(
                content={"error": "Request body must include an 'output' field"},
                status_code=400,
                media_type=MediaType.JSON,
            )

        # The output value must be a non-empty string — null, integers, lists, and
        # empty strings cannot be meaningfully validated against skill output schemas.
        _output_val = data.get("output")
        if _output_val is None or not isinstance(_output_val, str) or not _output_val.strip():
            return Response(
                content={"error": "'output' must be a non-empty string"},
                status_code=400,
                media_type=MediaType.JSON,
            )

        try:
            from vetinari.skills.skill_registry_convenience import validate_skill_output

            output = data.get("output")
            passed, failures = validate_skill_output(skill_id, output)
        except Exception:
            logger.exception(
                "Skill output validation failed for skill %r — returning 500",
                skill_id,
            )
            from vetinari.web.responses import litestar_error_response

            return litestar_error_response("Skills system unavailable", code=500)

        return {"passed": passed, "failures": failures}

    @post("/api/v1/skills/propose", media_type=MediaType.JSON)
    async def propose_skill(data: dict[str, Any]) -> Response | dict[str, Any]:
        """Submit a new skill proposal for human review.

        The proposal enters at T1 (untrusted) tier and requires human approval
        before activation.  Duplicate skill IDs are rejected with 409.

        Args:
            data: Request body with ``skill_id``, ``name``, ``description``,
                ``capabilities`` (list), and optional ``proposed_by`` (str).

        Returns:
            Dict with ``status`` and ``proposal`` details, or 400/409 on error.
        """
        from vetinari.skills.skill_registry_convenience import propose_registry_skill

        # Guard against null values — .get() returns None when key is present with null.
        for _f in ("skill_id", "name", "description", "proposed_by"):
            if _f in data and not isinstance(data[_f], str):
                return Response(
                    content={"error": f"'{_f}' must be a string"},
                    status_code=400,
                    media_type=MediaType.JSON,
                )

        skill_id = (data.get("skill_id") or "").strip()
        name = (data.get("name") or "").strip()
        description = (data.get("description") or "").strip()
        capabilities = data.get("capabilities", [])
        proposed_by = data.get("proposed_by") or "agent"

        if not skill_id or not name or not description:
            return Response(
                content={"error": "skill_id, name, and description are required"},
                status_code=400,
                media_type=MediaType.JSON,
            )
        if not isinstance(capabilities, list):
            return Response(
                content={"error": "capabilities must be a list of strings"},
                status_code=400,
                media_type=MediaType.JSON,
            )

        try:
            result = propose_registry_skill(skill_id, name, description, capabilities, proposed_by)
        except Exception as e:
            logger.warning(
                "Skill proposal failed for skill_id %r — %s",
                skill_id,
                type(e).__name__,
                exc_info=True,
            )
            return litestar_error_response("Skills service unavailable", 503)
        if result.get("status") == "rejected":
            return Response(
                content=result,
                status_code=409,
                media_type=MediaType.JSON,
            )
        return result

    @get("/api/v1/skills/{skill_id:str}/validation-detail", media_type=MediaType.JSON)
    async def get_validation_detail(skill_id: str) -> Response | dict[str, Any]:
        """Return structured enforcement details for a skill spec.

        Exposes hard constraints, error-severity standards, and per-category
        breakdowns derived from the spec's helper methods.

        Args:
            skill_id: The skill identifier (foreman, worker, or inspector).

        Returns:
            Dict with ``hard_constraints``, ``error_standards``,
            ``standards_by_category``, and ``constraints_by_category`` —
            or 404 if the skill is not found.
        """
        try:
            from vetinari.skills.skill_registry import get_skill_validation_detail

            detail = get_skill_validation_detail(skill_id)
        except Exception:
            logger.exception(
                "Skill validation detail lookup failed for skill %r — returning 500",
                skill_id,
            )
            from vetinari.web.responses import litestar_error_response

            return litestar_error_response("Skills system unavailable", code=500)

        if detail is None:
            return Response(
                content={"error": f"Skill '{skill_id}' not found"},
                status_code=404,
                media_type=MediaType.JSON,
            )
        return detail

    return [
        list_catalog,
        get_by_capability,
        get_by_tag,
        get_catalog_for_agent,
        list_summaries,
        get_summary,
        check_trust_elevation,
        validate_output,
        propose_skill,
        get_validation_detail,
    ]
