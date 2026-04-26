"""Local model discovery, scoring, configuration, and swap handlers.

Litestar migration of ``models_discovery_api.py``.  Provides handlers for
live local GGUF model discovery (with TTL cache), forced refresh, capability
scoring for a task description, model configuration updates, model swapping
(global and per-project), and the convenience discover endpoint.

This module is a standalone handler factory — it does NOT use the Flask
``_register(bp)`` pattern.  Call ``create_models_discovery_handlers()`` and
pass the returned list to your Litestar ``Router`` or application.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, get, post

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False

# Import shared helpers at module level so tests can patch
# ``vetinari.web.litestar_models_discovery._get_models_cached`` reliably.
# Factory-scoped imports create closures that are invisible to unittest.mock.patch.
try:
    from vetinari.web.shared import (
        PROJECT_ROOT,
        _get_models_cached,
        _infer_recommended_tasks,
        current_config,
    )
except ImportError:
    PROJECT_ROOT = None  # type: ignore[assignment]
    _get_models_cached = None  # type: ignore[assignment]
    _infer_recommended_tasks = None  # type: ignore[assignment]
    current_config = None  # type: ignore[assignment]


def create_models_discovery_handlers() -> list[Any]:
    """Create Litestar handlers for local model discovery, scoring, config, and swap.

    Returns an empty list when Litestar is not installed, so the factory is
    safe to call in environments that only have Flask available.

    Returns:
        List of Litestar route handler objects ready to register on a Router
        or Application.  Empty when Litestar is unavailable.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.responses import litestar_error_response, success_response

    @get("/api/v1/models", media_type=MediaType.JSON)
    async def api_models() -> Any:
        """Return the cached list of locally discovered models.

        Returns:
            JSON object with keys ``models`` (list), ``cached`` (bool), and
            ``count`` (int) wrapped in the standard success envelope.
            503 when the model discovery subsystem cannot be reached.
        """
        try:
            models = _get_models_cached()
            return success_response({"models": models, "cached": True, "count": len(models)})
        except Exception as exc:
            logger.warning("api_models: model discovery unavailable — returning 503: %s", exc)
            return litestar_error_response(  # type: ignore[return-value]
                "Model discovery subsystem unavailable", 503
            )

    @post("/api/v1/models/refresh", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_models_refresh() -> dict[str, Any]:
        """Force-refresh model discovery, bypassing the TTL cache.

        Returns:
            JSON object with keys ``models`` (list), ``cached`` (bool set to
            ``False``), and ``count`` (int).

        Raises:
            Returns 503 when model discovery is unavailable.
        """
        try:
            models = _get_models_cached(force=True)
            return {"models": models, "cached": False, "count": len(models)}
        except Exception as exc:
            logger.warning("api_models_refresh: model discovery unavailable — returning 503: %s", exc)
            return litestar_error_response(  # type: ignore[return-value]
                "Model discovery subsystem unavailable", 503
            )  # type: ignore[return-value]

    @post("/api/v1/score-models", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_score_models(data: dict[str, Any]) -> Any:
        """Score available models for a given task description.

        Reads ``task_description`` (str) from the JSON body and evaluates each
        discovered model against keyword-derived required capabilities.  Models
        are ranked by score descending.

        Args:
            data: Request body with optional ``task_description`` string.

        Returns:
            JSON object with key ``models`` containing scored model dicts,
            or a 400 error when no models are available, or 422 for
            unrecognised body keys.
        """
        from vetinari.web.request_validation import body_depth_exceeded, body_has_oversized_key

        if body_depth_exceeded(data):
            return litestar_error_response("Request body nesting depth exceeds maximum", 400)
        if body_has_oversized_key(data):
            return litestar_error_response("Request body contains oversized key", code=422)

        # Reject bodies that contain no recognised keys — catches repeated_keys/fuzz inputs.
        _KNOWN_KEYS = {"task_description"}
        if not _KNOWN_KEYS.intersection(data):
            return litestar_error_response("Request body contains no recognised fields", code=400)

        task_description_raw = data.get("task_description", "")
        if not isinstance(task_description_raw, str):
            return litestar_error_response("'task_description' must be a string", 400)
        task_description = task_description_raw.strip()
        if not task_description:
            return litestar_error_response("'task_description' cannot be empty", 400)

        try:
            models = _get_models_cached(force=True)
        except Exception as exc:
            logger.warning("api_score_models: model discovery unavailable — returning 503: %s", exc)
            return litestar_error_response(  # type: ignore[return-value]
                "Model discovery subsystem unavailable", 503
            )

        if not models:
            return litestar_error_response("No models available", 400)

        task_lower = task_description.lower()

        required_capabilities: list[str] = []
        if any(
            word in task_lower
            for word in [
                "code",
                "implement",
                "build",
                "create",
                "python",
                "javascript",
                "script",
                "api",
                "web",
                "function",
                "class",
            ]
        ):
            required_capabilities.append("code_gen")
        if any(word in task_lower for word in ["document", "readme", "explain", "comment", "docs", "description"]):
            required_capabilities.append("docs")
        if any(word in task_lower for word in ["chat", "conversation", "message", "respond", "reply"]):
            required_capabilities.append("chat")

        scored_models: list[dict[str, Any]] = []
        for m in models:
            model_name = m.get("name", "")
            capabilities = m.get("capabilities", [])

            score = 0
            capability_matches: list[str] = []
            for req in required_capabilities:
                if req in capabilities:
                    score += 10
                    capability_matches.append(req)

            memory_gb = m.get("memory_gb", 99)
            if memory_gb <= 2:
                score += 2
            elif memory_gb <= 8:
                score += 1

            recommended_for = _infer_recommended_tasks(capabilities)

            scored_models.append({
                "name": model_name,
                "score": score,
                "capabilities": capabilities,
                "memory_gb": memory_gb,
                "matches": capability_matches,
                "recommended_for": recommended_for,
            })

        scored_models.sort(key=lambda x: x["score"], reverse=True)
        return {"models": scored_models}

    @get("/api/v1/model-config", media_type=MediaType.JSON)
    async def api_model_config() -> dict[str, Any]:
        """Return the current model configuration settings.

        Returns:
            JSON object with ``default_models``, ``fallback_models``,
            ``uncensored_fallback_models``, and ``memory_budget_gb``.
        """
        return {
            "default_models": current_config.default_models,
            "fallback_models": current_config.fallback_models,
            "uncensored_fallback_models": current_config.uncensored_fallback_models,
            "memory_budget_gb": current_config.memory_budget_gb,
        }

    @post("/api/v1/model-config", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_update_model_config(data: dict[str, Any]) -> dict[str, Any]:
        """Update model configuration settings (admin only).

        Accepts a JSON body with any subset of ``default_models``,
        ``fallback_models``, ``uncensored_fallback_models``, and
        ``memory_budget_gb``.

        Args:
            data: Request body with the configuration keys to update.

        Returns:
            JSON with ``status`` and the full updated configuration, or 422
            for unrecognised body keys or malformed values.
        """
        from vetinari.web.request_validation import body_depth_exceeded, body_has_oversized_key

        if body_depth_exceeded(data):
            return litestar_error_response("Request body nesting depth exceeds maximum", 422)
        if body_has_oversized_key(data):
            return litestar_error_response("Request body contains oversized key", code=422)

        # Reject bodies that contain none of the recognised config keys.
        _KNOWN_KEYS = {"default_models", "fallback_models", "uncensored_fallback_models", "memory_budget_gb"}
        if not _KNOWN_KEYS.intersection(data):
            return litestar_error_response("Request body contains no recognised configuration keys", code=400)

        # Validate types for each recognised key that is present.
        # model-map fields must be dicts; memory_budget_gb must be a number.
        _model_map_keys = ("default_models", "fallback_models", "uncensored_fallback_models")
        for _mk in _model_map_keys:
            if _mk in data and not isinstance(data[_mk], dict):
                return litestar_error_response(f"'{_mk}' must be a JSON object (dict)", code=422)

        try:
            if "default_models" in data:
                current_config.default_models = data["default_models"]
            if "fallback_models" in data:
                current_config.fallback_models = data["fallback_models"]
            if "uncensored_fallback_models" in data:
                current_config.uncensored_fallback_models = data["uncensored_fallback_models"]
            if "memory_budget_gb" in data:
                memory_budget_raw = data["memory_budget_gb"]
                if not isinstance(memory_budget_raw, (int, float)):
                    return litestar_error_response("'memory_budget_gb' must be a number", 422)
                current_config.memory_budget_gb = int(memory_budget_raw)

            return {
                "status": "updated",
                "default_models": current_config.default_models,
                "fallback_models": current_config.fallback_models,
                "uncensored_fallback_models": current_config.uncensored_fallback_models,
                "memory_budget_gb": current_config.memory_budget_gb,
            }
        except Exception as exc:
            logger.warning("api_update_model_config: config update failed — returning 503: %s", exc)
            return litestar_error_response(
                "Model config subsystem unavailable — could not update configuration",
                503,
            )

    @post("/api/v1/swap-model", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_swap_model(data: dict[str, Any]) -> Any:
        """Swap the active model for a project or globally (admin only).

        Expects JSON with ``model_id`` (required) and optionally ``project_id``.
        When ``project_id`` is supplied the project's ``project.yaml`` is updated;
        otherwise the global ``current_config.active_model_id`` is set.

        Args:
            data: Request body with required ``model_id`` and optional ``project_id``.

        Returns:
            JSON with ``status`` and relevant model/project identifiers, or an
            error response when ``model_id`` is missing or the project is not found.
        """
        from vetinari.web.request_validation import body_depth_exceeded

        if body_depth_exceeded(data):
            return litestar_error_response("Request body nesting depth exceeds maximum", 400)

        import pathlib

        import yaml

        from vetinari.web.shared import validate_path_param

        project_id_raw = data.get("project_id")
        if project_id_raw is not None and not isinstance(project_id_raw, str):
            return litestar_error_response("'project_id' must be a string", 422)
        project_id: str | None = project_id_raw
        new_model_raw = data.get("model_id", "")
        if not isinstance(new_model_raw, str) or not new_model_raw:
            return litestar_error_response("'model_id' must be a non-empty string", 400)
        new_model = new_model_raw

        try:
            if project_id:
                if validate_path_param(project_id) is None:
                    return litestar_error_response(f"Invalid project_id: {project_id!r}", 400)
                project_dir = PROJECT_ROOT / "projects" / project_id
                if not project_dir.exists():
                    return litestar_error_response(f"Project not found: {project_id}", 404)

                config_file = project_dir / "project.yaml"
                if config_file.exists():
                    with pathlib.Path(config_file).open(encoding="utf-8") as f:
                        config = yaml.safe_load(f) or {}
                else:
                    config = {}

                config["active_model_id"] = new_model

                with pathlib.Path(config_file).open("w", encoding="utf-8") as f:
                    yaml.dump(config, f, allow_unicode=True)

                return {"status": "swapped", "project_id": project_id, "active_model_id": new_model}

            current_config.active_model_id = new_model
            return {"status": "swapped", "active_model_id": new_model}
        except Exception as exc:
            logger.warning("api_swap_model: model swap failed — returning 503: %s", exc)
            return litestar_error_response(  # type: ignore[return-value]
                "Model config subsystem unavailable — could not swap model", 503
            )

    @get("/api/v1/discover", media_type=MediaType.JSON)
    async def api_discover() -> Any:
        """Force model discovery and return all found models.

        Convenience alias for ``/api/v1/models/refresh`` that always bypasses
        the cache and uses a slightly different response shape.

        Returns:
            JSON with ``discovered`` (int), ``models`` (list), and ``status``.
            503 when the model discovery subsystem cannot be reached.
        """
        try:
            models = _get_models_cached(force=True)
            return {"discovered": len(models), "models": models, "status": "ok"}
        except Exception as exc:
            logger.warning("api_discover: model discovery unavailable — returning 503: %s", exc)
            return litestar_error_response(  # type: ignore[return-value]
                "Model discovery subsystem unavailable", 503
            )

    return [
        api_models,
        api_models_refresh,
        api_score_models,
        api_model_config,
        api_update_model_config,
        api_swap_model,
        api_discover,
    ]
