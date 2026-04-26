"""Rules configuration API handlers.

Native Litestar equivalents of the routes previously registered by ``rules_routes``.
Part of Flask->Litestar migration (ADR-0066). URL paths identical to Flask originals.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Response, get, post

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_rules_routes_handlers() -> list[Any]:
    """Return Litestar route handler instances for the rules configuration API.

    Returns an empty list when Litestar is not installed so the caller can
    safely call this in environments that only have Flask.

    Returns:
        List of Litestar route handler objects covering all rules endpoints.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.responses import litestar_error_response
    from vetinari.web.shared import validate_path_param

    @get("/api/v1/rules")
    async def api_rules_get() -> Any:
        """Return the complete rules configuration as a serialised dict.

        Reads all global rules, per-project overrides, and per-model overrides
        from the RulesManager so the UI can display the current configuration
        in a single request.

        Returns:
            Dict produced by ``RulesManager.to_dict()``, or 503 on service failure.
        """
        try:
            from vetinari.rules_manager import get_rules_manager

            rm = get_rules_manager()
            return rm.to_dict()
        except Exception:
            logger.exception(
                "Rules GET failed — rules manager unavailable, returning 500",
            )
            return litestar_error_response("Rules system unavailable", code=500)

    @get("/api/v1/rules/global")
    async def api_rules_global_get() -> Any:
        """Return the current global rules list that applies to all agents.

        Returns:
            Dict with a ``rules`` list, or 503 on service failure.
        """
        try:
            from vetinari.rules_manager import get_rules_manager

            rm = get_rules_manager()
            return {"rules": rm.get_global_rules()}
        except Exception:
            logger.exception(
                "Global rules GET failed — rules manager unavailable, returning 500",
            )
            return litestar_error_response("Rules system unavailable", code=500)

    @post("/api/v1/rules/global", guards=[admin_guard])
    async def api_rules_global_post(data: dict[str, Any]) -> Any:
        """Replace the global rules list that applies to all agents.

        Accepts ``rules`` as either a list of strings or a newline-delimited
        string and persists the result via RulesManager. An empty body (``{}``)
        is accepted and clears all global rules.

        Args:
            data: Request body dict containing the new ``rules`` value.
                Must include a ``rules`` key; empty body ``{}`` is rejected.

        Returns:
            Dict with ``status`` and the updated ``rules`` list, or 400 when
            the required ``rules`` field is absent.
        """
        # Require at least a "rules" key so an empty body does not silently succeed.
        if "rules" not in data:
            return Response(
                content={"error": "Request body must include a 'rules' field"},
                status_code=400,
                media_type=MediaType.JSON,
            )

        try:
            from vetinari.rules_manager import get_rules_manager

            rm = get_rules_manager()
            rules = data.get("rules", [])
            if isinstance(rules, str):
                rules = [r.strip() for r in rules.splitlines() if r.strip()]
            rm.set_global_rules(rules)
            return {"status": "saved", "rules": rm.get_global_rules()}
        except Exception:
            logger.exception(
                "Global rules POST failed — rules manager unavailable, returning 500",
            )
            return litestar_error_response("Rules system unavailable", code=500)

    @get("/api/v1/rules/global-prompt")
    async def api_rules_global_prompt_get() -> Any:
        """Return the current global system prompt override for all agents.

        Returns:
            Dict with a ``prompt`` string, or 503 on service failure.
        """
        try:
            from vetinari.rules_manager import get_rules_manager

            rm = get_rules_manager()
            return {"prompt": rm.get_global_system_prompt()}
        except Exception:
            logger.exception(
                "Global prompt GET failed — rules manager unavailable, returning 500",
            )
            return litestar_error_response("Rules system unavailable", code=500)

    @post("/api/v1/rules/global-prompt", guards=[admin_guard])
    async def api_rules_global_prompt_post(data: dict[str, Any]) -> Any:
        """Replace the global system prompt override for all agents.

        An empty body (``{}``) is accepted and clears the prompt override.

        Args:
            data: Request body dict with the new ``prompt`` string.

        Returns:
            Dict with ``status`` set to ``"saved"``, or 400 for malformed
            body, 422 for unrecognised keys, or 503 on service failure.
        """
        from vetinari.web.request_validation import body_depth_exceeded, body_has_oversized_key

        if body_depth_exceeded(data):
            return litestar_error_response("Request body nesting depth exceeds maximum", code=400)
        if body_has_oversized_key(data):
            return litestar_error_response("Request body contains oversized key", code=400)

        # Reject bodies with no recognised keys — catches fuzz inputs.
        _KNOWN_KEYS = {"prompt"}
        if not _KNOWN_KEYS.intersection(data):
            return litestar_error_response("Request body contains no recognised fields", code=422)

        try:
            from vetinari.rules_manager import get_rules_manager

            rm = get_rules_manager()
            rm.set_global_system_prompt(data.get("prompt", ""))
            return {"status": "saved"}
        except Exception:
            logger.exception(
                "Global prompt POST failed — rules manager unavailable, returning 500",
            )
            return litestar_error_response("Rules system unavailable", code=500)

    @get("/api/v1/rules/project/{project_id:str}")
    async def api_rules_project_get(project_id: str) -> Any:
        """Return the rules list for a specific project.

        Args:
            project_id: Identifier of the project whose rules to read.

        Returns:
            JSON with ``project_id`` and ``rules`` list, or 400 for an
            invalid identifier, or 503 on service failure.
        """
        if not validate_path_param(project_id):
            return litestar_error_response(f"Invalid project_id: {project_id!r}", 400)

        try:
            from vetinari.rules_manager import get_rules_manager

            rm = get_rules_manager()
            return {"project_id": project_id, "rules": rm.get_project_rules(project_id)}
        except Exception:
            logger.exception(
                "Project rules GET failed for project %r — rules manager unavailable, returning 500",
                project_id,
            )
            return litestar_error_response("Rules system unavailable", code=500)

    @post("/api/v1/rules/project/{project_id:str}", guards=[admin_guard])
    async def api_rules_project_post(project_id: str, data: dict[str, Any]) -> Any:
        """Replace the rules list for a specific project.

        Accepts ``rules`` as either a list of strings or a newline-delimited
        string so operators can apply project-scoped constraints.

        Args:
            project_id: Identifier of the project whose rules to update.
            data: Request body dict containing the new ``rules`` value.

        Returns:
            JSON with ``status``, ``project_id``, and updated ``rules``, or
            400 for an invalid identifier, 422 for unrecognised body keys,
            or 503 on service failure.
        """
        from vetinari.web.request_validation import body_depth_exceeded, body_has_oversized_key

        if body_depth_exceeded(data):
            return litestar_error_response("Request body nesting depth exceeds maximum", code=400)
        if body_has_oversized_key(data):
            return litestar_error_response("Request body contains oversized key", code=400)

        # Reject bodies that contain no recognised keys.
        _KNOWN_KEYS = {"rules"}
        if not _KNOWN_KEYS.intersection(data):
            return litestar_error_response("Request body contains no recognised fields", code=422)

        if not validate_path_param(project_id):
            return litestar_error_response(f"Invalid project_id: {project_id!r}", 400)

        try:
            from vetinari.rules_manager import get_rules_manager

            rm = get_rules_manager()
            rules = data.get("rules", [])
            if isinstance(rules, str):
                rules = [r.strip() for r in rules.splitlines() if r.strip()]
            rm.set_project_rules(project_id, rules)
            return {"status": "saved", "project_id": project_id, "rules": rm.get_project_rules(project_id)}
        except Exception:
            logger.exception(
                "Project rules POST failed for project %r — rules manager unavailable, returning 500",
                project_id,
            )
            return litestar_error_response("Rules system unavailable", code=500)

    @get("/api/v1/rules/model/{model_id:path}")
    async def api_rules_model_get(model_id: str) -> Any:
        """Return the rules list for a specific model.

        The ``model_id`` path parameter uses ``:path`` matching to support IDs
        that contain forward slashes (e.g. ``org/model-name``). Litestar
        captures the leading slash when the URL segment begins immediately after
        the path prefix, so any leading ``/`` is stripped before use.

        Args:
            model_id: Identifier of the model whose rules to read.

        Returns:
            Dict with ``model_id`` and ``rules`` list, or 503 on service failure.
        """
        # Litestar :path parameters can include a leading slash when the
        # captured segment starts immediately after the route prefix.
        # Strip it so the persisted key is always slash-free (e.g. "org/model").
        clean_id = model_id.lstrip("/")
        try:
            from vetinari.rules_manager import get_rules_manager

            rm = get_rules_manager()
            return {"model_id": clean_id, "rules": rm.get_model_rules(clean_id)}
        except Exception:
            logger.exception(
                "Model rules GET failed for model %r — rules manager unavailable, returning 500",
                clean_id,
            )
            return litestar_error_response("Rules system unavailable", code=500)

    @post("/api/v1/rules/model/{model_id:path}", guards=[admin_guard])
    async def api_rules_model_post(model_id: str, data: dict[str, Any]) -> Any:
        """Replace the rules list for a specific model.

        Accepts ``rules`` as either a list of strings or a newline-delimited
        string so operators can constrain individual model behaviour.
        Strips any leading slash from ``model_id`` that Litestar may inject
        when the model name contains slashes (e.g. ``"org/model"``).

        The ``model_id`` path parameter uses ``:path`` matching to support IDs
        that contain forward slashes (e.g. ``org/model-name``). Any leading
        ``/`` introduced by Litestar's path capture is stripped before use.

        Args:
            model_id: Identifier of the model whose rules to update.
            data: Request body dict containing the new ``rules`` value.

        Returns:
            Dict with ``status``, ``model_id``, and updated ``rules``, or
            422 for unrecognised body keys, or 503 on service failure.
        """
        from vetinari.web.request_validation import body_depth_exceeded, body_has_oversized_key

        if body_depth_exceeded(data):
            return litestar_error_response("Request body nesting depth exceeds maximum", code=400)
        if body_has_oversized_key(data):
            return litestar_error_response("Request body contains oversized key", code=400)

        # Reject bodies that contain no recognised keys.
        _KNOWN_KEYS = {"rules"}
        if not _KNOWN_KEYS.intersection(data):
            return litestar_error_response("Request body contains no recognised fields", code=422)

        clean_id = model_id.lstrip("/")
        try:
            from vetinari.rules_manager import get_rules_manager

            rm = get_rules_manager()
            rules = data.get("rules", [])
            if isinstance(rules, str):
                rules = [r.strip() for r in rules.splitlines() if r.strip()]
            rm.set_model_rules(clean_id, rules)
            return {"status": "saved", "model_id": clean_id, "rules": rm.get_model_rules(clean_id)}
        except Exception:
            logger.exception(
                "Model rules POST failed for model %r — rules manager unavailable, returning 500",
                clean_id,
            )
            return litestar_error_response("Rules system unavailable", code=500)

    return [
        api_rules_get,
        api_rules_global_get,
        api_rules_global_post,
        api_rules_global_prompt_get,
        api_rules_global_prompt_post,
        api_rules_project_get,
        api_rules_project_post,
        api_rules_model_get,
        api_rules_model_post,
    ]
