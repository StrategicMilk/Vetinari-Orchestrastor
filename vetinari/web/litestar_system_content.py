"""Workflow, prompts, preferences, settings, variant, download, and artifact Litestar handlers.

Native Litestar equivalents of the routes previously registered by
``system_content_api._register(bp)``.  Handles the workflow project tree view,
system prompt CRUD, user preferences, application settings, processing-depth
variant control, file downloads, and build artifact listing.

This is part of the Flask->Litestar migration (ADR-0066).  The URL paths
are identical to the Flask originals so existing clients require no changes.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

import yaml

from vetinari.web.litestar_system_content_validation import (
    _SETTINGS_VALID_KEYS,
    _atomic_write_text,
    _secure_filename,
    _validate_settings_update,
)

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Response, delete, get, post, put
    from litestar.params import Parameter
    from litestar.response import File

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_system_content_handlers() -> list[Any]:
    """Create Litestar handlers for workflow, prompts, preferences, settings, and download routes.

    Returns an empty list when Litestar is not installed so the application
    starts cleanly in environments where the optional dependency is absent.

    Returns:
        List of Litestar route handler objects, or empty list when unavailable.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard

    @get("/api/v1/workflow", media_type=MediaType.JSON)
    async def api_workflow(
        include_archived: bool | None = Parameter(query="include_archived", default=False),
        search: str | None = Parameter(query="search", default=None),
        status: str | None = Parameter(query="status", default=None),
    ) -> dict[str, Any]:
        """Return the project tree with per-task status for the workflow view.

        Supports optional query parameters ``include_archived`` (bool),
        ``search`` (str), and ``status`` (str) to filter the result set.

        Args:
            include_archived: When True, include archived projects in the result.
            search: Optional substring to filter by project name/description/goal.
            status: Optional exact status string to filter projects.

        Returns:
            JSON object with a ``projects`` key containing project dicts, each
            including task list, status, model assignment, and metadata.
        """
        from vetinari.types import StatusEnum
        from vetinari.web import shared

        if search is None:
            search = ""
        search_query = search.lower()
        if status is None:
            status = ""
        status_filter = status.lower()

        projects_dir = shared.PROJECT_ROOT / "projects"
        projects = []

        skipped_projects: list[str] = []
        if projects_dir.exists():
            try:
                project_entries = sorted(projects_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            except OSError:
                logger.warning(
                    "Could not list projects directory %s — returning empty project list",
                    projects_dir,
                )
                project_entries = []

            for p in project_entries:
                if not p.is_dir():
                    continue
                try:
                    config_file = p / "project.yaml"
                    conv_file = p / "conversation.json"
                    outputs_dir = p / "outputs"

                    # Do not expose absolute host filesystem paths — use project ID only
                    project_data: dict[str, Any] = {
                        "id": p.name,
                        "name": p.name,
                        "tasks": [],
                        "status": "unknown",
                        "archived": False,
                    }

                    if config_file.exists():
                        with pathlib.Path(config_file).open(encoding="utf-8") as f:
                            config = yaml.safe_load(f) or {}
                        project_data["name"] = config.get("project_name", p.name)
                        project_data["description"] = config.get("description", "")
                        project_data["goal"] = config.get("high_level_goal", "")
                        project_data["model"] = config.get("model", "")
                        project_data["active_model_id"] = config.get("active_model_id", "")
                        project_data["status"] = config.get("status", "unknown")
                        project_data["warnings"] = config.get("warnings", [])
                        project_data["archived"] = config.get("archived", False)

                        if project_data["archived"] and not include_archived:
                            continue

                        if search_query:
                            searchable = (
                                f"{project_data['name']} {project_data['description']} {project_data['goal']}".lower()
                            )
                            if search_query not in searchable:
                                continue

                        if status_filter and project_data["status"] != status_filter:
                            continue

                        planned_tasks = config.get("tasks", [])

                        completed_tasks: set[str] = set()
                        if outputs_dir.exists():
                            for task_dir in outputs_dir.iterdir():
                                if task_dir.is_dir():
                                    output_file = task_dir / "output.txt"
                                    if output_file.exists():
                                        completed_tasks.add(task_dir.name)

                        from vetinari.web.shared import _derive_project_status

                        project_data["status"] = _derive_project_status(
                            config.get("status", "unknown"),
                            planned_tasks,
                            completed_tasks,
                        )

                        from vetinari.web.shared import _is_project_actually_running

                        actually_running = _is_project_actually_running(p.name)
                        for t in planned_tasks:
                            task_id = t.get("id", "") or t.get("subtask_id", "")
                            project_data["tasks"].append({
                                "id": task_id,
                                "description": t.get("description", ""),
                                "assigned_model": t.get("assigned_model_id", ""),
                                "status": StatusEnum.COMPLETED.value
                                if task_id in completed_tasks
                                else (StatusEnum.RUNNING.value if actually_running else StatusEnum.PENDING.value),
                                "model_override": t.get("model_override", ""),
                            })

                    if conv_file.exists():
                        with pathlib.Path(conv_file).open(encoding="utf-8") as f:
                            conv = json.load(f)
                            project_data["message_count"] = len(conv)

                    projects.append(project_data)
                except Exception:
                    logger.warning(
                        "Could not load project %s — skipping it from workflow listing",
                        p.name,
                        exc_info=True,
                    )
                    skipped_projects.append(p.name)

        result: dict[str, Any] = {"projects": projects}
        if skipped_projects:
            result["warnings"] = [f"Project '{name}' could not be loaded and was skipped" for name in skipped_projects]
        return result

    @get("/api/v1/system-prompts", media_type=MediaType.JSON)
    async def api_system_prompts() -> dict[str, Any]:
        """List all saved system prompt templates.

        These are user-managed prompt templates stored in ``system_prompts/*.txt``.
        They serve as a library of reusable prompts for the UI, but are NOT
        the runtime inference prompt source. Runtime prompts are built by
        PromptAssembler (``vetinari/prompts/assembler.py``).

        Scans ``<project_root>/system_prompts/`` for ``.txt`` files and returns
        each prompt's stem name and full text content.

        Returns:
            JSON object with a ``prompts`` key containing a list of
            ``{"name": str, "content": str}`` dicts.
        """
        from vetinari.web import shared

        prompts_dir = shared.PROJECT_ROOT / "system_prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)

        prompts = []
        unreadable: list[str] = []
        for f in prompts_dir.glob("*.txt"):
            try:
                prompts.append({"name": f.stem, "content": f.read_text(encoding="utf-8").strip()})
            except OSError:
                logger.warning(
                    "Could not read system prompt file %s — skipping it from listing",
                    f.name,
                )
                unreadable.append(f.name)

        result: dict[str, Any] = {"prompts": prompts}
        if unreadable:
            result["warnings"] = [f"Prompt file '{name}' could not be read and was skipped" for name in unreadable]
        return result

    @post("/api/v1/system-prompts", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_save_system_prompt(data: dict[str, Any]) -> Response:
        """Persist a named system prompt template to disk.

        These are user-managed prompt templates stored in ``system_prompts/*.txt``.
        They serve as a library of reusable prompts for the UI, but are NOT
        the runtime inference prompt source. Runtime prompts are built by
        PromptAssembler (``vetinari/prompts/assembler.py``).

        Reads ``name`` and ``content`` from the JSON body, writes the content to
        ``<project_root>/system_prompts/<name>.txt``, creating the directory if
        needed.

        Args:
            data: JSON body with ``name`` (str) and ``content`` (str) fields.

        Returns:
            JSON with ``status: saved`` and the prompt ``name`` on success.
            HTTP 400 if ``name`` is absent or invalid.
        """
        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response

        name_raw = data.get("name")
        if not isinstance(name_raw, str):
            return litestar_error_response("'name' must be a string", 400)
        if len(name_raw) > 255:
            return litestar_error_response("'name' must not exceed 255 characters", 400)
        name = name_raw.strip()
        prompt_content_raw = data.get("content", "")
        if not isinstance(prompt_content_raw, str):
            return litestar_error_response("'content' must be a string", 422)
        prompt_content = prompt_content_raw

        if not name:
            return litestar_error_response("name is required", 400)

        # Reject traversal payloads — names must be plain identifiers.
        import re as _re

        if not _re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name):
            return litestar_error_response(
                "Invalid prompt name — use letters, digits, underscores, or hyphens only", 400
            )

        prompts_dir = shared.PROJECT_ROOT / "system_prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize name to prevent path traversal
        safe_name = _secure_filename(name)
        if not safe_name:
            return litestar_error_response("Invalid prompt name", 400)

        prompt_file = prompts_dir / f"{safe_name}.txt"
        prompt_file.write_text(prompt_content, encoding="utf-8")

        return Response(
            content={"status": "saved", "name": safe_name},
            status_code=200,
            media_type=MediaType.JSON,
        )

    @delete("/api/v1/system-prompts/{name:str}", media_type=MediaType.JSON, status_code=200, guards=[admin_guard])
    async def api_delete_system_prompt(name: str) -> Response:
        """Delete a named system prompt template from disk.

        These are user-managed prompt templates stored in ``system_prompts/*.txt``.
        They serve as a library of reusable prompts for the UI, but are NOT
        the runtime inference prompt source. Runtime prompts are built by
        PromptAssembler (``vetinari/prompts/assembler.py``).

        Removes ``<project_root>/system_prompts/<name>.txt`` if it exists.
        Succeeds silently when the file is already absent.

        Args:
            name: Stem name of the system prompt file (without ``.txt`` extension).

        Returns:
            JSON with ``status: deleted`` on success.
            HTTP 400 when the name contains unsafe path characters.
        """
        import re as _re

        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response

        # Reject traversal payloads before sanitisation — names must be plain
        # identifiers (letters, digits, underscores, hyphens).  Dots, slashes,
        # encoded separators, and control characters are all disallowed.
        if not _re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name):
            return litestar_error_response(
                "Invalid prompt name — use letters, digits, underscores, or hyphens only", 400
            )

        safe_name = _secure_filename(name)
        if not safe_name:
            return litestar_error_response("Invalid prompt name", 400)

        prompt_file = shared.PROJECT_ROOT / "system_prompts" / f"{safe_name}.txt"
        if prompt_file.exists():
            prompt_file.unlink()
        return Response(
            content={"status": "deleted"},
            status_code=200,
            media_type=MediaType.JSON,
        )

    @get("/api/v1/preferences", media_type=MediaType.JSON)
    async def api_get_preferences() -> dict[str, Any] | Response:
        """Retrieve all user preferences with defaults applied.

        Returns:
            JSON object with a ``preferences`` key containing the full
            preference map (explicit saved values merged over DEFAULTS).
            HTTP 503 if the preferences manager cannot be initialised.
        """
        from vetinari.web.preferences import get_preferences_manager
        from vetinari.web.responses import litestar_error_response

        try:
            mgr = get_preferences_manager()
            return {"preferences": mgr.get_all()}
        except Exception:
            logger.warning(
                "Preferences manager unavailable — could not retrieve preferences",
                exc_info=True,
            )
            return litestar_error_response("Preferences manager unavailable", 503)

    @put("/api/v1/preferences", media_type=MediaType.JSON)
    async def api_set_preferences(data: dict[str, Any]) -> dict[str, Any] | Response:
        """Update one or more user preferences from the JSON request body.

        Args:
            data: Mapping of preference keys to their new values.

        Returns:
            JSON object with ``preferences`` (full updated map) and optionally
            ``rejected_keys`` (list of keys that were not in the whitelist).
            Returns HTTP 400 if the request body is not a JSON object and
            HTTP 422 if the object does not contain any preference updates.
        """
        from vetinari.web.responses import litestar_error_response

        if not isinstance(data, dict):
            logger.warning(
                "PUT /api/v1/preferences rejected non-object body — got %s, returning 400",
                type(data).__name__,
            )
            return litestar_error_response(
                f"Request body must be a JSON object, not {type(data).__name__}",
                400,
            )

        if not data:
            return litestar_error_response(
                "Request body must not be empty - provide at least one preference key",
                422,
            )

        from vetinari.web.preferences import get_preferences_manager

        mgr = get_preferences_manager()
        results = mgr.set_many(data)
        rejected = [k for k, v in results.items() if not v]
        # Reject the request when every submitted key was unrecognised — a body
        # with only unknown keys is effectively malformed input for this endpoint.
        if rejected and len(rejected) == len(data):
            return litestar_error_response(
                f"None of the provided keys are recognised preferences: {sorted(rejected)}",
                400,
            )
        resp: dict[str, Any] = {"preferences": mgr.get_all()}
        if rejected:
            resp["rejected_keys"] = rejected
        return resp

    @get("/api/v1/settings", media_type=MediaType.JSON)
    async def api_get_settings() -> dict[str, Any] | Response:
        """Return current application settings including hardware profile and inference config.

        Returns:
            JSON with inference settings, detected hardware, API key availability,
            and per-agent timeout overrides.
            HTTP 503 if settings or hardware detection subsystem is unavailable.
        """
        from vetinari.web.responses import litestar_error_response

        try:
            from vetinari.config.settings import get_settings
            from vetinari.system.hardware_detect import detect_hardware

            settings = get_settings()
            hw = detect_hardware()
        except Exception:
            logger.warning(
                "Settings retrieval failed — settings or hardware detection subsystem unavailable",
                exc_info=True,
            )
            return litestar_error_response(
                "Settings retrieval failed — settings or hardware detection subsystem unavailable",
                503,
            )

        return {
            "inference": {
                "gpu_layers": settings.local_gpu_layers,
                "context_length": settings.local_context_length,
                "batch_size": settings.local_batch_size,
                "flash_attn": settings.local_flash_attn,
                "n_threads": settings.local_n_threads,
                "inference_timeout": settings.local_inference_timeout,
                "max_agent_retries": settings.max_agent_retries,
            },
            "hardware": hw.to_dict(),
            "api_keys": settings.detect_api_keys(),
            "agent_timeouts": settings.agent_timeouts,
            "log_level": settings.log_level,
            "observability_enabled": settings.enable_observability,
            "settings_source": {
                "runtime": "environment",
                "user_config": "~/.vetinari/config.yaml is persisted for operator/backend preferences",
            },
        }

    @put("/api/v1/settings", media_type=MediaType.JSON)
    async def api_update_settings(data: dict[str, Any]) -> Response:
        """Update application settings from JSON request body.

        Accepts partial updates — only provided fields are changed. Settings
        are persisted to the user config file at ``~/.vetinari/config.yaml``.

        Args:
            data: Partial settings update dict. Supports ``inference`` (dict),
                ``agent_timeouts`` (dict), and ``log_level`` (str) keys.

        Returns:
            JSON with the updated settings map, or HTTP 400 when body is not a
            JSON object or is empty.
        """
        from vetinari.web.responses import litestar_error_response

        if not isinstance(data, dict):
            logger.warning(
                "PUT /api/v1/settings rejected non-object body — got %s, returning 400",
                type(data).__name__,
            )
            return litestar_error_response(
                f"Request body must be a JSON object, not {type(data).__name__}",
                400,
            )

        if not data:
            return litestar_error_response("No settings provided in request body", 400)

        # Reject bodies that contain no recognised settings keys — prevents
        # malformed payloads (deeply-nested junk, unicode bombs, etc.) from
        # silently succeeding with a no-op write.
        if not (set(data.keys()) & _SETTINGS_VALID_KEYS):
            return litestar_error_response(
                "Request body must contain at least one recognised key: inference, agent_timeouts, or log_level",
                400,
            )

        # Validate that each recognised key, when present, carries a correctly-typed value.
        # inference and agent_timeouts must be dicts; log_level must be a non-empty string.
        # Reject the entire request if any supplied recognised key has an invalid type —
        # a partial-write that silently discards nulls/wrong-typed values would be confusing.
        _type_errors: list[str] = []
        if "inference" in data and not isinstance(data["inference"], dict):
            _type_errors.append("'inference' must be a JSON object (dict)")
        if "agent_timeouts" in data and not isinstance(data["agent_timeouts"], dict):
            _type_errors.append("'agent_timeouts' must be a JSON object (dict)")
        if "log_level" in data and (not isinstance(data["log_level"], str) or not data["log_level"].strip()):
            _type_errors.append("'log_level' must be a non-empty string")
        if _type_errors:
            return litestar_error_response("; ".join(_type_errors), 400)

        value_errors = _validate_settings_update(data)
        if value_errors:
            return litestar_error_response("; ".join(value_errors), 400)

        from vetinari.constants import get_user_dir

        config_path = get_user_dir() / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        existing: dict[str, Any] = {}
        if config_path.exists():
            try:
                loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning(
                    "Existing settings config is malformed or unreadable; refusing to overwrite %s",
                    config_path,
                    exc_info=True,
                )
                return litestar_error_response(
                    "Existing settings config is malformed or unreadable; refusing to overwrite it",
                    409,
                )
            if loaded is None:
                existing = {}
            elif isinstance(loaded, dict):
                existing = loaded
            else:
                return litestar_error_response(
                    "Existing settings config must be a YAML object; refusing to overwrite it",
                    409,
                )

        if "inference" in data and isinstance(data["inference"], dict):
            existing.setdefault("inference", {}).update(data["inference"])
        if "agent_timeouts" in data and isinstance(data["agent_timeouts"], dict):
            existing["agent_timeouts"] = data["agent_timeouts"]
        if "log_level" in data:
            existing["log_level"] = data["log_level"]

        try:
            _atomic_write_text(
                config_path,
                yaml.safe_dump(existing, default_flow_style=False, sort_keys=False),
            )
        except OSError:
            logger.warning("Failed to persist settings config to %s", config_path, exc_info=True)
            return litestar_error_response("Settings could not be persisted", 500)

        return Response(
            content={
                "status": "saved",
                "config_path": str(config_path),
                "runtime_effect": (
                    "saved to operator user config; VetinariSettings runtime values remain environment-driven "
                    "until that runtime loader consumes user config"
                ),
                "persisted": {key: existing[key] for key in _SETTINGS_VALID_KEYS if key in existing},
            },
            status_code=200,
            media_type=MediaType.JSON,
        )

    @get("/api/v1/variant", media_type=MediaType.JSON)
    async def api_get_variant() -> dict[str, Any] | Response:
        """Return the current processing depth variant level and its configuration.

        Returns:
            JSON with the current level name, description, and all available levels.
            HTTP 503 if the variant manager cannot be initialised.
        """
        from vetinari.web.responses import litestar_error_response

        try:
            from vetinari.web.variant_system import get_variant_manager

            mgr = get_variant_manager()
            cfg = mgr.get_config()
        except Exception:
            logger.warning(
                "Variant manager unavailable — could not retrieve variant configuration",
                exc_info=True,
            )
            return litestar_error_response("Variant manager unavailable", 503)

        return {
            "level": mgr.current_level,
            "description": cfg.description,
            "max_context_tokens": cfg.max_context_tokens,
            "max_planning_depth": cfg.max_planning_depth,
            "enable_verification": cfg.enable_verification,
            "enable_self_improvement": cfg.enable_self_improvement,
            "available_levels": mgr.get_all_levels(),
        }

    @put("/api/v1/variant", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_set_variant(data: dict[str, Any]) -> Response:
        """Switch the processing depth variant level.

        Accepts a JSON body with a ``level`` key (``"low"``, ``"medium"``, or
        ``"high"``).  Updates both the shared VariantManager singleton and the
        active orchestrator so all subsequent requests use the new limits.

        Args:
            data: JSON body with ``level`` field (``"low"``, ``"medium"``, ``"high"``).

        Returns:
            JSON with the new level name and its configuration on success.
            HTTP 400 if ``level`` is missing or unrecognised.
        """
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.variant_system import get_variant_manager

        level = (data.get("level") or "").strip().lower()
        if not level:
            return litestar_error_response("level is required", 400)

        mgr = get_variant_manager()
        try:
            cfg = mgr.set_level(level)
        except ValueError:
            logger.warning("Invalid autonomy level %r in request — returning 400", level)
            return litestar_error_response(f"Unknown level '{level}'. Use low, medium, or high.", 400)

        try:
            from vetinari.web.shared import get_orchestrator

            orb = get_orchestrator()
            if hasattr(orb, "set_variant_level"):
                orb.set_variant_level(level)
        except Exception:
            logger.warning("Orchestrator unavailable for variant propagation", exc_info=True)

        return Response(
            content={
                "level": mgr.current_level,
                "description": cfg.description,
                "max_context_tokens": cfg.max_context_tokens,
                "max_planning_depth": cfg.max_planning_depth,
                "enable_verification": cfg.enable_verification,
                "enable_self_improvement": cfg.enable_self_improvement,
            },
            status_code=200,
            media_type=MediaType.JSON,
        )

    @get("/api/v1/download")
    async def api_download(
        project_id: str | None = Parameter(query="project_id", default=None),
        task_id: str | None = Parameter(query="task_id", default=None),
        filename: str | None = Parameter(query="filename", default=None),
    ) -> Response:
        """Download a generated file by project_id, task_id, and filename.

        The route intentionally does not declare a JSON media_type because the
        successful response is a file attachment streamed by Litestar's File
        response — which sets its own Content-Type.  Error responses are
        returned as explicit JSON Response objects.

        Args:
            project_id: The project identifier (URL query parameter).
            task_id: The task identifier (URL query parameter).
            filename: The filename to download (URL query parameter).

        Returns:
            A file attachment response on success.
            HTTP 400 when any required query parameter is missing, invalid, or
            contains unsafe characters that the sanitiser would normalise away.
            HTTP 403 on path traversal attempt.
            HTTP 404 when the file does not exist.
        """
        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response

        if not project_id or not task_id or not filename:
            return litestar_error_response("project_id, task_id, and filename are required", 400)

        safe_project = _secure_filename(project_id)
        safe_task = _secure_filename(task_id)
        safe_name = _secure_filename(filename)

        # Reject identifiers that contained unsafe characters rather than silently
        # normalising them — a normalised name may resolve to an unrelated file.
        if safe_project != project_id or safe_task != task_id or safe_name != filename:
            logger.warning(
                "Download rejected — identifier contained unsafe characters: project=%r task=%r filename=%r",
                project_id,
                task_id,
                filename,
            )
            return litestar_error_response(
                "Invalid project_id, task_id, or filename — contains unsafe characters",
                400,
            )

        if not all([safe_project, safe_task, safe_name]):
            return litestar_error_response("Invalid path parameter", 400)

        projects_dir = shared.PROJECT_ROOT / "projects"
        file_path = (projects_dir / safe_project / "outputs" / safe_task / "generated" / safe_name).resolve()

        # Guard against path traversal — resolved path must stay inside projects_dir
        try:
            file_path.relative_to(projects_dir.resolve())
        except ValueError:
            logger.warning("Path traversal attempt blocked — resolved path escapes projects_dir: %s", file_path)
            return litestar_error_response("Access denied", 403)

        if not file_path.exists() or not file_path.is_file():
            return litestar_error_response("File not found", 404)

        try:
            return File(path=file_path, filename=safe_name)  # type: ignore[return-value]
        except Exception:
            logger.warning(
                "Could not construct file response for %s — file may have been removed after existence check",
                safe_name,
                exc_info=True,
            )
            return litestar_error_response("File could not be served", 500)

    @get("/api/v1/artifacts", media_type=MediaType.JSON)
    async def api_artifacts() -> dict[str, Any]:
        """List files in the build artifacts directory.

        Returns:
            JSON object with an ``artifacts`` key containing a list of
            ``{"name": str, "size": int, "path": str}`` dicts for each
            file found under ``build/artifacts/``.  The ``path`` field is
            the filename only — absolute host paths are never exposed.
        """
        from vetinari.web import shared

        build_dir = shared.PROJECT_ROOT / "build" / "artifacts"
        artifacts: list[dict[str, Any]] = []
        if build_dir.exists():
            try:
                # Use filename only — never expose absolute host filesystem paths
                artifacts.extend(
                    {"name": f.name, "size": f.stat().st_size, "path": f.name}
                    for f in build_dir.iterdir()
                    if f.is_file()
                )
            except OSError:
                logger.warning(
                    "Build artifacts directory %s could not be read — returning empty artifact list",
                    build_dir,
                )
                return {"artifacts": [], "error": "Build artifacts directory unreadable"}
        return {"artifacts": artifacts}

    return [
        api_workflow,
        api_system_prompts,
        api_save_system_prompt,
        api_delete_system_prompt,
        api_get_preferences,
        api_set_preferences,
        api_get_settings,
        api_update_settings,
        api_get_variant,
        api_set_variant,
        api_download,
        api_artifacts,
    ]
