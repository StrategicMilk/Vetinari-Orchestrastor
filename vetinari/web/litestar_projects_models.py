"""Project model-management Litestar handlers.

Covers project model search, task model override, and project model refresh
routes extracted from ``litestar_projects_api.py``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Response, post

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def _create_projects_model_handlers() -> list[Any]:
    """Create Litestar handlers for project model-management routes.

    Returns:
        List of Litestar route handler objects, or empty list when Litestar
        is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard

    # -- Models --------------------------------------------------------------

    @post("/api/project/{project_id:str}/model-search", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_model_search(project_id: str, data: dict[str, Any]) -> Response:
        """Search for candidate models suitable for a project's task.

        Queries ``ModelDiscovery`` with the task description from the request
        body and returns ranked model candidates.  Respects per-project and
        global external-discovery flags.

        Args:
            project_id: The project for which models are being searched.
            data: JSON body with optional ``task_description``.

        Returns:
            JSON with ``status``, ``candidates`` list, and ``count``.
        """
        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import (
            ENABLE_EXTERNAL_DISCOVERY,
            _project_external_model_enabled,
            current_config,
            validate_path_param,
        )

        if not validate_path_param(project_id):
            return litestar_error_response("Invalid project ID", 400)

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        if not ENABLE_EXTERNAL_DISCOVERY:
            return litestar_error_response("External discovery globally disabled", 403)
        if not _project_external_model_enabled(project_dir):
            return litestar_error_response("External model discovery disabled for this project", 403)

        from vetinari.model_discovery import ModelDiscovery

        task_description = data.get("task_description", "")
        discovery = ModelDiscovery()
        lm_models: list[Any] = []
        try:
            from vetinari.models.model_pool import ModelPool

            model_pool = ModelPool(current_config)
            model_pool.discover_models()
            lm_models = model_pool.list_models()
        except Exception:
            logger.warning(
                "Could not get local models for model-search in project %s — using empty list",
                project_id,
            )

        candidates = discovery.search(task_description, lm_models)
        return Response(
            content={
                "status": "ok",
                "candidates": [c.to_dict() for c in candidates],
                "count": len(candidates),
            },
            status_code=200,
            media_type=MediaType.JSON,
        )

    @post(
        "/api/project/{project_id:str}/task/{task_id:str}/override",
        media_type=MediaType.JSON,
        guards=[admin_guard],
    )
    async def api_task_override(
        project_id: str, task_id: str, data: dict[str, Any]
    ) -> Response:
        """Override the model assigned to a specific task within a project.

        Reads the project YAML, locates the task by ID, sets its
        ``model_override`` field, and persists the updated config.

        Args:
            project_id: The project containing the task.
            task_id: The task whose model override is being set.
            data: JSON body with ``model_id`` string.

        Returns:
            JSON with ``status``, ``task_id``, and ``model_override``.
        """
        import pathlib

        import yaml

        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import validate_path_param

        if not validate_path_param(project_id) or not validate_path_param(task_id):
            return litestar_error_response("Invalid parameters", 400)

        model_id = data.get("model_id", "")
        if not model_id:
            return litestar_error_response("model_id is required", 422)

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        config_file = project_dir / "project.yaml"

        if not config_file.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        with pathlib.Path(config_file).open(encoding="utf-8") as f:
            config: dict[str, Any] = yaml.safe_load(f) or {}

        for task in config.get("tasks", []):
            if task.get("id") == task_id:
                task["model_override"] = model_id
                break

        with pathlib.Path(config_file).open("w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        return Response(
            content={"status": "ok", "task_id": task_id, "model_override": model_id},
            status_code=200,
            media_type=MediaType.JSON,
        )

    @post(
        "/api/project/{project_id:str}/refresh-models",
        media_type=MediaType.JSON,
        guards=[admin_guard],
    )
    async def api_refresh_models(project_id: str) -> Response:
        """Signal that the model cache for a project should be refreshed.

        With live external model discovery enabled there is no local cache to
        expire.  This endpoint acknowledges the request and informs the caller
        that live search is active.

        Args:
            project_id: The project whose model cache is being refreshed.

        Returns:
            JSON with ``status`` and a ``message`` string, or 404 if the
            project does not exist.
        """
        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import validate_path_param

        if not validate_path_param(project_id):
            return litestar_error_response("Invalid project ID", 400)

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        return Response(
            content={"status": "ok", "message": "Model cache refreshed (live search enabled)"},
            status_code=200,
            media_type=MediaType.JSON,
        )

    return [
        api_model_search,
        api_task_override,
        api_refresh_models,
    ]
