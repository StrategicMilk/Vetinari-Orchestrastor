"""Project task-management Litestar handlers.

Covers add, update, delete, and rerun task routes extracted from
``litestar_projects_api.py``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Response, delete, post, put

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def _create_projects_task_handlers() -> list[Any]:
    """Create Litestar handlers for project task-management routes.

    Returns:
        List of Litestar route handler objects, or empty list when Litestar
        is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard

    # -- Tasks ---------------------------------------------------------------

    @post("/api/project/{project_id:str}/task", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_add_task(project_id: str, data: dict[str, Any]) -> Response:
        """Add a new task to an existing project's task list.

        Args:
            project_id: The project to add the task to.
            data: JSON body with optional ``id``, ``description``, ``inputs``,
                ``outputs``, ``dependencies``, and ``model_override``.

        Returns:
            JSON with ``status`` and the newly created ``task`` dict.
        """
        import pathlib

        import yaml

        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import validate_path_param

        if not validate_path_param(project_id):
            return litestar_error_response("Invalid project ID", 400)

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        config_file = project_dir / "project.yaml"
        if not config_file.exists():
            return litestar_error_response("Project config not found", 404)

        with pathlib.Path(config_file).open(encoding="utf-8") as f:
            config: dict[str, Any] = yaml.safe_load(f) or {}

        tasks = config.get("tasks", [])
        new_id = data.get("id", f"t{len(tasks) + 1}")

        if any(t.get("id") == new_id for t in tasks):
            return litestar_error_response(f"Task ID '{new_id}' already exists", 400)

        new_task: dict[str, Any] = {
            "id": new_id,
            "description": data.get("description", ""),
            "inputs": data.get("inputs", []),
            "outputs": data.get("outputs", []),
            "dependencies": data.get("dependencies", []),
            "model_override": data.get("model_override", ""),
        }
        tasks.append(new_task)
        config["tasks"] = tasks

        with pathlib.Path(config_file).open("w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)

        return Response(
            content={"status": "added", "task": new_task},
            status_code=200,
            media_type=MediaType.JSON,
        )

    @put(
        "/api/project/{project_id:str}/task/{task_id:str}",
        media_type=MediaType.JSON,
        guards=[admin_guard],
    )
    async def api_update_task(
        project_id: str, task_id: str, data: dict[str, Any]
    ) -> Response:
        """Update mutable fields on an existing project task.

        Applies any subset of ``description``, ``inputs``, ``outputs``,
        ``dependencies``, and ``model_override`` and persists the updated YAML.

        Args:
            project_id: The project containing the task.
            task_id: The task to update.
            data: JSON body with fields to update.

        Returns:
            JSON with ``status`` and the updated ``task`` dict.
        """
        import pathlib

        import yaml

        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import validate_path_param

        if not validate_path_param(project_id) or not validate_path_param(task_id):
            return litestar_error_response("Invalid parameters", 400)

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        config_file = project_dir / "project.yaml"
        if not config_file.exists():
            return litestar_error_response("Project config not found", 404)

        with pathlib.Path(config_file).open(encoding="utf-8") as f:
            config: dict[str, Any] = yaml.safe_load(f) or {}

        tasks = config.get("tasks", [])
        task_found = False
        updated_idx = 0

        for i, task in enumerate(tasks):
            if task["id"] == task_id:
                if "description" in data:
                    tasks[i]["description"] = data["description"]
                if "inputs" in data:
                    tasks[i]["inputs"] = data["inputs"]
                if "outputs" in data:
                    tasks[i]["outputs"] = data["outputs"]
                if "dependencies" in data:
                    tasks[i]["dependencies"] = data["dependencies"]
                if "model_override" in data:
                    tasks[i]["model_override"] = data["model_override"]
                task_found = True
                updated_idx = i
                break

        if not task_found:
            return litestar_error_response("Task not found", 404)

        config["tasks"] = tasks
        with pathlib.Path(config_file).open("w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)

        return Response(
            content={"status": "updated", "task": tasks[updated_idx]},
            status_code=200,
            media_type=MediaType.JSON,
        )

    @delete(
        "/api/project/{project_id:str}/task/{task_id:str}",
        media_type=MediaType.JSON,
        guards=[admin_guard],
        status_code=200,
    )
    async def api_delete_task(project_id: str, task_id: str) -> Response:
        """Remove a task from a project and clean up dependency references.

        Args:
            project_id: The project containing the task.
            task_id: The task to delete.

        Returns:
            JSON with ``status`` and ``task_id``.
        """
        import pathlib

        import yaml

        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import validate_path_param

        if not validate_path_param(project_id) or not validate_path_param(task_id):
            return litestar_error_response("Invalid parameters", 400)

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        config_file = project_dir / "project.yaml"
        if not config_file.exists():
            return litestar_error_response("Project config not found", 404)

        with pathlib.Path(config_file).open(encoding="utf-8") as f:
            config: dict[str, Any] = yaml.safe_load(f) or {}

        tasks = config.get("tasks", [])
        tasks = [t for t in tasks if t["id"] != task_id]
        for task in tasks:
            if task_id in task.get("dependencies", []):
                task["dependencies"] = [d for d in task["dependencies"] if d != task_id]

        config["tasks"] = tasks
        with pathlib.Path(config_file).open("w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)

        return Response(
            content={"status": "deleted", "task_id": task_id},
            status_code=200,
            media_type=MediaType.JSON,
        )

    @post(
        "/api/project/{project_id:str}/task/{task_id:str}/rerun",
        media_type=MediaType.JSON,
        guards=[admin_guard],
    )
    async def api_rerun_task(
        project_id: str, task_id: str, data: dict[str, Any]
    ) -> Response:
        """Re-run a single task within a completed or partially-completed project.

        Resets the task to pending and enqueues it via the orchestration request
        queue.  Accepts an optional model override.

        Args:
            project_id: The project that owns the task.
            task_id: The task to re-run.
            data: JSON body with optional ``model`` override.

        Returns:
            JSON with ``ok``, ``task_id``, ``exec_id``, and updated task status.
        """
        import pathlib

        import yaml

        from vetinari.types import StatusEnum
        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import _push_sse_event, validate_path_param

        if not validate_path_param(project_id) or not validate_path_param(task_id):
            return litestar_error_response("Invalid parameters", 400)

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        config_path = project_dir / "project.yaml"

        if not config_path.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        with pathlib.Path(config_path).open(encoding="utf-8") as f:
            project_config: dict[str, Any] = yaml.safe_load(f) or {}

        tasks = project_config.get("tasks", [])
        target_task = None
        for t in tasks:
            if t.get("id") == task_id:
                target_task = t
                break

        if target_task is None:
            return litestar_error_response(f"Task '{task_id}' not found in project", 404)

        target_task["status"] = "pending"
        target_task["output"] = ""
        with pathlib.Path(config_path).open("w", encoding="utf-8") as f:
            yaml.dump(project_config, f, allow_unicode=True)

        model_override = data.get("model") or None

        try:
            from vetinari.orchestration.request_routing import PRIORITY_STANDARD, RequestQueue

            rq = RequestQueue()
            context: dict[str, Any] = {
                "project_id": project_id,
                "task_id": task_id,
                "retry": True,
                "model_override": model_override,
            }
            exec_id = rq.enqueue(
                goal=f"Re-run task {task_id} in project {project_id}",
                context=context,
                priority=PRIORITY_STANDARD,
            )
        except Exception:
            logger.exception(
                "Failed to enqueue rerun for project=%s task=%s — orchestrator may be unavailable",
                project_id,
                task_id,
            )
            return litestar_error_response("Orchestrator unavailable — rerun not queued", 503)

        logger.info(
            "Task %s rerun enqueued for project %s (exec_id=%s)",
            task_id,
            project_id,
            exec_id,
        )
        _push_sse_event(
            project_id,
            "task_rerun",
            {
                "project_id": project_id,
                "task_id": task_id,
                "status": StatusEnum.PENDING.value,
                "exec_id": exec_id,
            },
        )
        return Response(
            content={
                "ok": True,
                "task_id": task_id,
                "exec_id": exec_id,
                "status": StatusEnum.PENDING.value,
            },
            status_code=200,
            media_type=MediaType.JSON,
        )

    return [
        api_add_task,
        api_update_task,
        api_delete_task,
        api_rerun_task,
    ]
