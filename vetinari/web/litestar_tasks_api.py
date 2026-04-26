"""Task execution and output retrieval handlers.

Native Litestar equivalents of the routes previously registered by
``vetinari.web.tasks_api``. Part of Flask->Litestar migration (ADR-0066).
URL paths are identical to the Flask originals.
"""

from __future__ import annotations

import logging
import threading
import unicodedata
import uuid
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, get, post

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False

# Default system prompt used when the caller does not supply one.
_DEFAULT_SYSTEM_PROMPT: str = (
    "You are Vetinari, an AI orchestration assistant. "
    "Provide structured, actionable responses with clear reasoning. "
    "For code: follow PEP 8, include type hints and docstrings. "
    "For plans: break into concrete steps with dependencies. "
    "Always return valid JSON when structured output is expected."
)


def create_tasks_api_handlers() -> list[Any]:
    """Create Litestar handlers for task execution and output retrieval.

    Replicates the eight routes from ``vetinari.web.tasks_api``:
    ``GET /api/v1/tasks``, ``POST /api/v1/run-task``,
    ``POST /api/v1/run-all``, ``POST /api/v1/run-prompt``,
    ``GET /api/v1/output/{task_id}``, ``GET /api/v1/all-tasks``,
    ``GET /api/v1/project/{project_id}/task/{task_id}/output``, and
    ``POST /api/v1/project/{project_id}/task/{task_id}/override``.

    Returns an empty list when Litestar is not installed, so the factory is
    safe to call in Flask-only environments.

    Returns:
        List of Litestar route handler objects ready to register on a Router
        or Application.  Empty when Litestar is unavailable.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web import shared
    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.responses import litestar_error_response, success_response
    from vetinari.web.shared import PROJECT_ROOT, current_config, get_orchestrator, validate_path_param

    @get("/api/v1/tasks", media_type=MediaType.JSON)
    async def api_tasks() -> dict[str, Any]:
        """Return the task list from the manifest YAML.

        Returns:
            JSON object with a ``tasks`` key containing the list of tasks from
            the configured project manifest, or an empty list when no manifest
            exists.
        """
        config_path = Path(shared.PROJECT_ROOT) / current_config.config_path
        if config_path.is_file():
            with Path(config_path).open(encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            tasks = config.get("tasks", [])
            return success_response({"tasks": tasks})
        return success_response({"tasks": []})

    @post("/api/v1/run-task", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_run_task(data: dict[str, Any]) -> Any:
        """Run a single manifest task in a background thread.

        Request JSON must contain ``task_id``.  Admin-only.

        Args:
            data: Request body with required ``task_id`` string.

        Returns:
            JSON with ``status`` and ``task_id`` on success, or an error with
            HTTP 400 when ``task_id`` is missing.
        """
        task_id = data.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            return litestar_error_response("'task_id' must be a non-empty string", 400)

        orb = get_orchestrator()

        def _run_task() -> None:
            """Execute the requested task via the orchestrator."""
            orb.run_task(task_id)

        thread = threading.Thread(target=_run_task, daemon=True)
        thread.start()
        return {"status": "started", "task_id": task_id}

    @post("/api/v1/run-all", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_run_all(data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run all manifest tasks in a background thread.

        Admin-only.  Accepts no request body; any non-empty body is rejected
        with 422.

        Args:
            data: Request body — must be empty (``{}``).

        Returns:
            JSON with ``status`` confirming tasks were started.
        """
        if data is not None:
            return litestar_error_response("This endpoint takes no request body parameters", code=422)
        orb = get_orchestrator()

        def _run_all() -> None:
            """Execute all tasks via the orchestrator."""
            orb.run_all()

        thread = threading.Thread(target=_run_all, daemon=True)
        thread.start()
        return {"status": "started"}

    @post("/api/v1/run-prompt", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_run_prompt(data: dict[str, Any]) -> Any:
        """Run a free-form prompt through the adapter with guardrail checks.

        Request JSON fields:
            prompt (str): The user prompt.  Required.
            model (str): Model name override.  Optional.
            system_prompt (str): System prompt override.  Optional.

        Admin-only.

        Args:
            data: Request body with required ``prompt`` and optional ``model``
                and ``system_prompt`` keys.

        Returns:
            JSON with ``status``, ``task_id``, ``response``, ``model``, and
            ``latency_ms`` on success, or an error with HTTP 400 when the
            prompt is missing or blocked, HTTP 503 when model inference fails.
        """
        prompt_raw = data.get("prompt", "")
        if not isinstance(prompt_raw, str):
            return litestar_error_response("'prompt' must be a string", 422)
        prompt = unicodedata.normalize("NFC", prompt_raw)
        model_raw = data.get("model", "")
        if not isinstance(model_raw, str):
            return litestar_error_response("'model' must be a string", 422)
        model: str = model_raw
        system_prompt_raw = data.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)
        if not isinstance(system_prompt_raw, str):
            return litestar_error_response("'system_prompt' must be a string", 422)
        system_prompt: str = system_prompt_raw

        if not prompt:
            return litestar_error_response("prompt is required", 400)

        # Guardrails: check user input at trust boundary
        try:
            from vetinari.safety.guardrails import get_guardrails

            gr_result = get_guardrails().check_input(prompt)
            if not gr_result.allowed:
                return litestar_error_response(
                    "Input blocked by safety guardrails",
                    400,
                    details={"violations": [v.to_dict() for v in gr_result.violations]},
                )
        except Exception as exc:
            logger.warning("Guardrails input check failed (non-blocking) — proceeding without guardrails: %s", exc)

        orb = get_orchestrator()

        if not model:
            if orb.model_pool.models:
                model = orb.model_pool.models[0].get("name", "")
            else:
                return litestar_error_response("No models available", 400)

        try:
            result = orb.adapter.chat(model, system_prompt, prompt)
        except Exception as exc:
            logger.warning("Prompt execution failed for model %s — ensure a local model is loaded: %s", model, exc)
            return litestar_error_response(
                "Model inference failed — ensure a local model is loaded and running",
                503,
                details={"model": model},
            )

        output_text: str = result.get("output", "")

        # Guardrails: check output at trust boundary
        try:
            from vetinari.safety.guardrails import get_guardrails

            gr_out = get_guardrails().check_output(output_text)
            if not gr_out.allowed:
                output_text = gr_out.content  # filtered content
        except Exception as exc:
            logger.warning("Guardrails output check failed (non-blocking) — returning unfiltered output: %s", exc)

        task_id = "custom_" + uuid.uuid4().hex[:12]
        output_path = PROJECT_ROOT / "outputs" / task_id
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "output.txt").write_text(output_text, encoding="utf-8")

        from vetinari.types import StatusEnum

        return {
            "status": StatusEnum.COMPLETED.value,
            "task_id": task_id,
            "response": output_text,
            "model": model,
            "latency_ms": result.get("latency_ms", 0),
        }

    @get("/api/v1/output/{task_id:str}", media_type=MediaType.JSON)
    async def api_output(task_id: str) -> dict[str, Any]:
        """Return the text output for a task, searching all projects first.

        Args:
            task_id: The identifier of the task whose output to retrieve.

        Returns:
            JSON with ``output``, ``task_id``, and optionally ``project_id``
            when the task was found inside a project directory.

        Raises:
            Nothing — returns a 400 error response on invalid task_id, empty
            dict on missing output.
        """
        if not validate_path_param(task_id):
            return litestar_error_response("Invalid task ID", 400)

        projects_dir = PROJECT_ROOT / "projects"
        if projects_dir.exists():
            for p in projects_dir.iterdir():
                if p.is_dir():
                    output_path = p / "outputs" / task_id / "output.txt"
                    if output_path.exists():
                        content = output_path.read_text(encoding="utf-8")
                        return {"output": content, "task_id": task_id, "project_id": p.name}

        output_path = PROJECT_ROOT / "outputs" / task_id / "output.txt"
        if output_path.exists():
            content = output_path.read_text(encoding="utf-8")
            return {"output": content, "task_id": task_id}
        return {"output": "", "task_id": task_id}

    @get("/api/v1/all-tasks", media_type=MediaType.JSON)
    async def api_all_tasks() -> dict[str, Any]:
        """Return a flat list of all tasks across all projects.

        Tasks are sorted by project modification time, newest project first.

        Returns:
            JSON with a ``tasks`` key containing task descriptors from every
            project that has a ``project.yaml`` manifest.
        """
        projects_dir = PROJECT_ROOT / "projects"
        all_tasks: list[dict[str, Any]] = []

        if projects_dir.exists():
            for p in sorted(
                projects_dir.iterdir(),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            ):
                if p.is_dir():
                    config_file = p / "project.yaml"
                    if config_file.exists():
                        with Path(config_file).open(encoding="utf-8") as f:
                            config = yaml.safe_load(f) or {}
                        all_tasks.extend(
                            {
                                "project_id": p.name,
                                "project_name": config.get("project_name", p.name),
                                "task_id": t.get("id", ""),
                                "description": t.get("description", ""),
                                "assigned_model": t.get("assigned_model_id", ""),
                            }
                            for t in config.get("tasks", [])
                        )

        return {"tasks": all_tasks}

    @get(
        "/api/v1/project/{project_id:str}/task/{task_id:str}/output",
        media_type=MediaType.JSON,
    )
    async def api_task_output(project_id: str, task_id: str) -> Any:
        """Return the output text and generated file list for a specific task.

        Args:
            project_id: The project identifier.
            task_id: The task identifier within the project.

        Returns:
            JSON with ``project_id``, ``task_id``, ``output``, and ``files``,
            or HTTP 404 when the project directory does not exist.
        """
        if not validate_path_param(project_id) or not validate_path_param(task_id):
            return litestar_error_response("Invalid parameters", 400)
        project_dir = PROJECT_ROOT / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        output_path = project_dir / "outputs" / task_id / "output.txt"
        output = ""
        if output_path.exists():
            output = output_path.read_text(encoding="utf-8")

        generated_dir = project_dir / "outputs" / task_id / "generated"
        files: list[dict[str, str]] = []
        if generated_dir.exists():
            files.extend({"name": f.name, "path": str(f)} for f in generated_dir.iterdir() if f.is_file())

        return {"project_id": project_id, "task_id": task_id, "output": output, "files": files}

    @post(
        "/api/v1/project/{project_id:str}/task/{task_id:str}/override",
        media_type=MediaType.JSON,
        guards=[admin_guard],
    )
    async def api_task_override(project_id: str, task_id: str, data: dict[str, Any]) -> Any:
        """Override the model assigned to a specific task.

        Request JSON fields:
            model_id (str): The model identifier to assign.

        Admin-only.

        Args:
            project_id: The project identifier.
            task_id: The task identifier within the project.
            data: Request body with optional ``model_id`` key.

        Returns:
            JSON with ``status``, ``task_id``, and ``model_override`` on
            success, or HTTP 404 when the project config does not exist.
        """
        model_id: str = data.get("model_id", "")

        if not validate_path_param(project_id) or not validate_path_param(task_id):
            return litestar_error_response("Invalid parameters", 400)

        project_dir = PROJECT_ROOT / "projects" / project_id
        config_file = project_dir / "project.yaml"

        if not config_file.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        with Path(config_file).open(encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        for task in config.get("tasks", []):
            if task.get("id") == task_id:
                task["model_override"] = model_id
                break

        with Path(config_file).open("w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        return {"status": "ok", "task_id": task_id, "model_override": model_id}

    return [
        api_tasks,
        api_run_task,
        api_run_all,
        api_run_prompt,
        api_output,
        api_all_tasks,
        api_task_output,
        api_task_override,
    ]
