"""Project lifecycle Litestar handlers.

Covers project listing, detail lookup, rename, archive, and deletion routes
extracted from ``litestar_projects_api.py`` to keep route groups cohesive.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Request, Response, delete, get, post
    from litestar.params import Parameter

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def _create_projects_lifecycle_handlers() -> list[Any]:
    """Create Litestar handlers for project lifecycle routes.

    Returns:
        List of Litestar route handler objects, or empty list when Litestar
        is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard

    # -- Lifecycle -----------------------------------------------------------

    @get("/api/projects", media_type=MediaType.JSON)
    async def api_projects(
        include_archived: str | None = Parameter(query="include_archived", default=None),
    ) -> dict[str, Any]:
        """List all projects, optionally including archived ones.

        Scans the projects directory, reads each project's YAML config and
        conversation file, and returns summary metadata with task status.

        Args:
            include_archived: Pass ``"true"`` to include archived projects in
                the result set.

        Returns:
            JSON with a ``projects`` list of project summary dicts.
        """
        import json as _json
        import pathlib

        import yaml as _yaml

        from vetinari.types import StatusEnum
        from vetinari.web import shared
        from vetinari.web.shared import _derive_project_status, _is_project_actually_running

        show_archived = (include_archived or "false").lower() == "true"
        projects_dir = shared.PROJECT_ROOT / "projects"
        projects: list[dict[str, Any]] = []

        if projects_dir.exists():
            for p in sorted(projects_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                if not p.is_dir():
                    continue
                config_file = p / "project.yaml"
                conv_file = p / "conversation.json"
                outputs_dir = p / "outputs"

                project_data: dict[str, Any] = {
                    "id": p.name,
                    "name": p.name,
                    "path": p.name,
                    "tasks": [],
                    "status": "unknown",
                    "archived": False,
                }

                if config_file.exists():
                    with config_file.open(encoding="utf-8") as _cf:
                        config = _yaml.safe_load(_cf) or {}
                    project_data["name"] = config.get("project_name", p.name)
                    from vetinari.security.redaction import redact_text

                    project_data["description"] = redact_text(str(config.get("description", "")))
                    project_data["goal"] = redact_text(str(config.get("high_level_goal", "")))
                    project_data["model"] = config.get("model", "")
                    project_data["active_model_id"] = config.get("active_model_id", "")
                    project_data["status"] = config.get("status", "unknown")
                    project_data["warnings"] = config.get("warnings", [])
                    project_data["archived"] = config.get("archived", False)

                    if project_data["archived"] and not show_archived:
                        continue

                    planned_tasks = config.get("tasks", [])

                    completed_tasks: set[str] = set()
                    if outputs_dir.exists():
                        for task_dir in outputs_dir.iterdir():
                            if task_dir.is_dir() and (task_dir / "output.txt").exists():
                                completed_tasks.add(task_dir.name)

                    project_data["status"] = _derive_project_status(
                        config.get("status", "unknown"),
                        planned_tasks,
                        completed_tasks,
                        project_id=p.name,
                    )

                    # Only mark tasks "running" when a live background thread exists.
                    # Stale YAML status='running' from crashed sessions must not
                    # create phantom running tasks in the UI.
                    actually_running = _is_project_actually_running(p.name)
                    for t in planned_tasks:
                        task_id = t.get("id", "") or t.get("subtask_id", "")
                        project_data["tasks"].append({
                            "id": task_id,
                            "description": t.get("description", ""),
                            "assigned_model": t.get("assigned_model_id", ""),
                            "status": "completed"
                            if task_id in completed_tasks
                            else (StatusEnum.RUNNING.value if actually_running else StatusEnum.PENDING.value),
                            "model_override": t.get("model_override", ""),
                        })

                if conv_file.exists():
                    with pathlib.Path(conv_file).open(encoding="utf-8") as f:
                        conv = _json.load(f)
                        project_data["message_count"] = len(conv)

                projects.append(project_data)

        return {"projects": projects}

    @get("/api/project/{project_id:str}", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_project(project_id: str) -> Response:
        """Return the full state of a single project.

        Loads the project YAML config, conversation history, and per-task
        output status, merging planned tasks with any extra output-only tasks.

        Args:
            project_id: The unique project directory name.

        Returns:
            JSON with ``id``, ``config``, ``conversation``, and ``tasks``,
            or HTTP 404 when the project does not exist.
        """
        import pathlib

        import yaml

        from vetinari.types import StatusEnum
        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import validate_path_param

        if not validate_path_param(project_id):
            return litestar_error_response("Invalid project ID", 400)

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        config_file = project_dir / "project.yaml"
        config: dict[str, Any] = {}
        if config_file.exists():
            with pathlib.Path(config_file).open(encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

        import json as _json

        conv_file = project_dir / "conversation.json"
        conversation: list[dict[str, Any]] = []
        if conv_file.exists():
            with pathlib.Path(conv_file).open(encoding="utf-8") as f:
                conversation = _json.load(f)

        tasks: list[dict[str, Any]] = []
        planned_tasks = config.get("tasks", [])

        completed_task_ids: set[str] = set()
        task_outputs: dict[str, Any] = {}
        outputs_dir = project_dir / "outputs"
        if outputs_dir.exists():
            for task_dir in sorted(outputs_dir.iterdir()):
                if task_dir.is_dir():
                    task_id_inner = task_dir.name
                    output_file = task_dir / "output.txt"
                    output = ""
                    if output_file.exists():
                        output = output_file.read_text(encoding="utf-8")
                        completed_task_ids.add(task_id_inner)
                        task_outputs[task_id_inner] = output

                    generated_dir = task_dir / "generated"
                    files: list[dict[str, Any]] = []
                    if generated_dir.exists():
                        files.extend(
                            {"name": f.name, "path": str(f.relative_to(project_dir))}
                            for f in generated_dir.iterdir()
                            if f.is_file()
                        )
                    task_outputs[task_id_inner + "_files"] = files

        project_status = config.get("status", "unknown")
        for t in planned_tasks:
            tid = t.get("id", "") or t.get("subtask_id", "")
            tasks.append({
                "id": tid,
                "description": t.get("description", ""),
                "assigned_model": t.get("assigned_model_id", ""),
                "output": task_outputs.get(tid, ""),
                "files": task_outputs.get(tid + "_files", []),
                "status": "completed"
                if tid in completed_task_ids
                else (
                    StatusEnum.RUNNING.value if project_status == StatusEnum.RUNNING.value else StatusEnum.PENDING.value
                ),
            })

        for tid in completed_task_ids:
            if not any(t["id"] == tid for t in tasks):
                tasks.append({
                    "id": tid,
                    "description": "Additional task",
                    "assigned_model": "",
                    "output": task_outputs.get(tid, ""),
                    "files": task_outputs.get(tid + "_files", []),
                    "status": StatusEnum.COMPLETED.value,
                })

        return Response(
            content={
                "id": project_id,
                "name": config.get("project_name", config.get("name", project_id)),
                "goal": config.get("goal", ""),
                "config": config,
                "conversation": conversation,
                "tasks": tasks,
            },
            status_code=200,
            media_type=MediaType.JSON,
        )

    @post("/api/project/{project_id:str}/rename", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_rename_project(project_id: str, data: dict[str, Any]) -> Response:
        """Rename a project and/or update its description in project.yaml.

        Args:
            project_id: The project to rename.
            data: JSON body with optional ``name`` and ``description`` fields.

        Returns:
            JSON with ``status: renamed``, ``project_id``, updated
            ``project_name``, and ``description``.  HTTP 404 when not found.
        """
        import pathlib
        import unicodedata

        import yaml

        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import validate_path_param

        if not validate_path_param(project_id):
            return litestar_error_response("Invalid project ID", 400)

        new_name_raw = data.get("name", "")
        if not isinstance(new_name_raw, str):
            return litestar_error_response("'name' must be a string", 422)
        new_name = unicodedata.normalize("NFC", new_name_raw)
        new_description_raw = data.get("description", "")
        if not isinstance(new_description_raw, str):
            return litestar_error_response("'description' must be a string", 422)
        new_description = unicodedata.normalize("NFC", new_description_raw)

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        config_file = project_dir / "project.yaml"
        if not config_file.exists():
            return litestar_error_response("Project config not found", 404)

        with pathlib.Path(config_file).open(encoding="utf-8") as f:
            config: dict[str, Any] = yaml.safe_load(f) or {}

        if new_name:
            config["project_name"] = new_name
        if new_description is not None:
            config["description"] = new_description

        with pathlib.Path(config_file).open("w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)

        return Response(
            content={
                "status": "renamed",
                "project_id": project_id,
                "project_name": config.get("project_name"),
                "description": config.get("description"),
            },
            status_code=200,
            media_type=MediaType.JSON,
        )

    @post("/api/project/{project_id:str}/archive", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_archive_project(project_id: str, data: dict[str, Any]) -> Response:
        """Archive or unarchive a project by toggling its archived flag.

        Args:
            project_id: The project to archive or unarchive.
            data: JSON body with optional ``archive`` boolean (default ``true``).

        Returns:
            JSON with ``status``, ``project_id``, and the new ``archived``
            boolean.  HTTP 404 when the project does not exist.
        """
        import pathlib

        import yaml

        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import validate_path_param

        if not validate_path_param(project_id):
            return litestar_error_response("Invalid project ID", 400)

        archive = data.get("archive", True)

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        config_file = project_dir / "project.yaml"
        if not config_file.exists():
            return litestar_error_response("Project config not found", 404)

        with pathlib.Path(config_file).open(encoding="utf-8") as f:
            config: dict[str, Any] = yaml.safe_load(f) or {}

        config["archived"] = archive
        config["status"] = "archived" if archive else "completed"

        with pathlib.Path(config_file).open("w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)

        return Response(
            content={
                "status": "archived" if archive else "unarchived",
                "project_id": project_id,
                "archived": archive,
            },
            status_code=200,
            media_type=MediaType.JSON,
        )

    @delete("/api/project/{project_id:str}", media_type=MediaType.JSON, guards=[admin_guard], status_code=200)
    async def api_delete_project(project_id: str, request: Request) -> Response:
        """Move a project to the recycle bin after confirming intent.

        Requires a JSON body carrying ``confirmed_by`` and ``reason`` fields.
        The project directory is moved to the recycle bin (not hard-deleted)
        so it can be restored within the configured grace window.

        Args:
            project_id: The project to delete.
            request: Litestar request object used to read the JSON body.

        Returns:
            JSON with ``status``, ``project_id``, and ``recycle_record_id``,
            or HTTP 400 if intent is missing, or HTTP 404 if not found.
        """
        from vetinari.safety.recycle import RecycleStore
        from vetinari.web import shared
        from vetinari.web.api.destructive_api import parse_confirmed_intent_or_400
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import (
            _cancel_project_task,
            _cleanup_project_state,
            _is_project_actually_running,
            validate_path_param,
        )

        if not validate_path_param(project_id):
            return litestar_error_response("Invalid project ID", 400)

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        body = await request.json()
        intent, err = parse_confirmed_intent_or_400(body)
        if err:
            return litestar_error_response(err, 400)

        if _is_project_actually_running(project_id):
            _cancel_project_task(project_id)
            # Allow the background thread to notice the cancel flag before files move
            time.sleep(0.5)
            _cleanup_project_state(project_id)

        # VET142-excluded: lifecycle-fenced deletion via RecycleStore.retire
        record = RecycleStore().retire(
            project_dir,
            reason=intent.reason,
            work_receipt_id=None,
        )
        return Response(
            content={
                "status": "deleted",
                "project_id": project_id,
                "recycle_record_id": record.record_id,
            },
            status_code=200,
            media_type=MediaType.JSON,
        )

    return [
        api_projects,
        api_project,
        api_rename_project,
        api_archive_project,
        api_delete_project,
    ]
