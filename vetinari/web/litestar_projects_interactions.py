"""Project interaction and execution Litestar handlers.

Covers project chat messages, single-task execution, and goal verification
routes extracted from ``litestar_projects_api.py``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Response, post

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False

# Per-project conversation file locks.
# Who writes: api_project_message. Who reads: api_project_message.
# Lock protects concurrent read-modify-write on conversation.json.
_conv_locks: dict[str, Any] = {}
_conv_locks_guard: threading.Lock = threading.Lock()


def _create_projects_interaction_handlers(*, default_system_prompt: str) -> list[Any]:
    """Create Litestar handlers for project interaction routes.

    Args:
        default_system_prompt: Prompt used when project config omits one.

    Returns:
        List of Litestar route handler objects, or empty list when Litestar
        is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard

    @post("/api/project/{project_id:str}/message", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_project_message(project_id: str, data: dict[str, Any]) -> Response:
        """Append a user message to a project conversation and get an AI reply.

        Runs guardrail checks on the message, appends it to the conversation,
        calls the project's assigned model, and persists the updated conversation.

        Args:
            project_id: The project whose conversation is being updated.
            data: JSON body with ``message`` and optional ``attachments``.

        Returns:
            JSON with ``status``, ``response``, and the full ``conversation`` list.
        """
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import validate_path_param

        if not validate_path_param(project_id):
            return litestar_error_response("Invalid project ID", 400)

        import json as _json
        import pathlib
        import threading
        import unicodedata

        import yaml

        from vetinari.types import AgentType
        from vetinari.web import shared
        from vetinari.web.shared import get_orchestrator

        message = unicodedata.normalize("NFC", data.get("message", ""))
        attachments: list[dict[str, Any]] = data.get("attachments", [])

        if not message and not attachments:
            return litestar_error_response("message or attachments required", 400)

        try:
            from vetinari.safety.guardrails import RailContext, get_guardrails

            _guardrails = get_guardrails()
            _rail_result = _guardrails.check_input(
                message, RailContext(source="chat", agent_type=AgentType.WORKER.value)
            )
            if not _rail_result.passed:
                logger.warning("Guardrail blocked chat message: %s", _rail_result.reason)
                return litestar_error_response(f"Message blocked by safety filter: {_rail_result.reason}", 400)
        except Exception:
            logger.warning(
                "Guardrails unavailable for chat message check in project %s — proceeding",
                project_id,
            )

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        orb = get_orchestrator()

        with _conv_locks_guard:
            if project_id not in _conv_locks:
                _conv_locks[project_id] = threading.Lock()
            conv_lock = _conv_locks[project_id]

        with conv_lock:
            conv_file = project_dir / "conversation.json"
            conversation: list[dict[str, Any]] = []
            if conv_file.exists():
                with pathlib.Path(conv_file).open(encoding="utf-8") as f:
                    conversation = _json.load(f)

            config_file = project_dir / "project.yaml"
            model = "auto"
            config: dict[str, Any] = {}
            if config_file.exists():
                with pathlib.Path(config_file).open(encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                    model = config.get("model", model)

            user_msg: dict[str, Any] = {"role": "user", "content": message}
            if attachments:
                user_msg["attachments"] = [
                    {
                        "id": att.get("id", ""),
                        "filename": att.get("filename", ""),
                        "type": att.get("type", ""),
                        "size": att.get("size", 0),
                    }
                    for att in attachments
                    if att.get("id")
                ]
            conversation.append(user_msg)

            # Truncate context to avoid exceeding the model's context window.
            # ~12 000 chars (~3 000 tokens) leaves room for the response.
            _max_context_chars = 12000
            recent_msgs = conversation[-10:]
            context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_msgs])
            if len(context_str) > _max_context_chars:
                context_str = context_str[-_max_context_chars:]

            system_prompt = (
                config.get("system_prompt", default_system_prompt) if config_file.exists() else default_system_prompt
            )
            result = orb.adapter.chat(model, system_prompt, context_str)
            response = result.get("output", "")
            conversation.append({"role": "assistant", "content": response})

            with pathlib.Path(conv_file).open("w", encoding="utf-8") as f:
                _json.dump(conversation, f, indent=2)

        return Response(
            content={"status": "ok", "response": response, "conversation": conversation},
            status_code=200,
            media_type=MediaType.JSON,
        )

    @post("/api/project/{project_id:str}/execute", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_project_execute(project_id: str, data: dict[str, Any]) -> Response:
        """Execute a specific task within a project via TwoLayerOrchestrator.

        Classifies the task goal through RequestIntake at the web front door
        to determine the pipeline tier before dispatching.  Express-tier tasks
        bypass planning and route directly to the Worker agent.

        Args:
            project_id: Project containing the task to execute.
            data: JSON body with required ``task_id`` and optional ``model``.

        Returns:
            JSON with ``status``, ``task_id``, ``project_id``, and
            ``intake_tier``.
        """
        import pathlib
        import threading

        import yaml

        from vetinari.types import StatusEnum
        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import _push_sse_event, get_orchestrator, validate_path_param

        if not validate_path_param(project_id):
            return litestar_error_response("Invalid project ID", 400)

        task_id = data.get("task_id")
        if not task_id:
            return litestar_error_response("task_id is required", 400)

        proj_dir = shared.PROJECT_ROOT / "projects" / project_id
        if not proj_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        config_path = proj_dir / "project.yaml"
        if not config_path.exists():
            return litestar_error_response("Project config not found", 404)

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

        orb = get_orchestrator()
        if not orb:
            return litestar_error_response("Orchestrator not available", 503)

        task_goal = target_task.get("description", "")
        intake_tier_value: str = "standard"
        try:
            from vetinari.orchestration.intake import get_request_intake

            _intake = get_request_intake()
            _intake_context: dict[str, Any] = {
                "file_count": len(project_config.get("expected_outputs", [])),
            }
            _tier, _features = _intake.classify_with_features(task_goal, _intake_context)
            intake_tier_value = _tier.value
            logger.info(
                "[Intake] project=%s task=%s classified as %s (confidence=%.2f, words=%d)",
                project_id,
                task_id,
                _tier.value,
                _features.confidence,
                _features.word_count,
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[Intake] project=%s task=%s features: ambiguous=%s, cross_cutting=%d",
                    project_id,
                    task_id,
                    _features.has_ambiguous_words,
                    _features.cross_cutting_keywords,
                )
        except Exception:
            logger.warning(
                "Intake classification failed for project=%s task=%s — falling through to full pipeline",
                project_id,
                task_id,
            )

        def _run_via_tlo() -> None:
            """Run a single task through the TLO pipeline in a background thread."""
            try:
                from vetinari.orchestration.two_layer import get_two_layer_orchestrator

                _tlo = get_two_layer_orchestrator()

                model_override = data.get("model") or target_task.get("model_override") or None
                _pipeline_context: dict[str, Any] = {
                    "project_id": project_id,
                    "task_id": task_id,
                    "category": project_config.get("category", ""),
                    "tech_stack": project_config.get("tech_stack", ""),
                    "priority": project_config.get("priority", "quality"),
                    "required_features": project_config.get("required_features", []),
                    "things_to_avoid": project_config.get("things_to_avoid", []),
                    "expected_outputs": project_config.get("expected_outputs", []),
                    "intake_tier": intake_tier_value,
                }

                pipeline_result = _tlo.generate_and_execute(
                    task_goal,
                    constraints=_pipeline_context,
                    context=_pipeline_context,
                    project_id=project_id,
                    model_id=model_override,
                )

                _final_output = pipeline_result.get("final_output", "")
                _success = pipeline_result.get(StatusEnum.FAILED.value, 0) == 0

                target_task["status"] = StatusEnum.COMPLETED.value if _success else StatusEnum.FAILED.value
                target_task["output"] = str(_final_output)[:10000] if _final_output else ""
                with pathlib.Path(config_path).open("w", encoding="utf-8") as wf:
                    yaml.dump(project_config, wf, allow_unicode=True)

                _push_sse_event(
                    project_id,
                    "task_complete",
                    {
                        "project_id": project_id,
                        "task_id": task_id,
                        "status": StatusEnum.COMPLETED.value if _success else StatusEnum.FAILED.value,
                        "output_length": len(str(_final_output)),
                    },
                )
            except Exception as exc:
                logger.error(
                    "Task %s execution via TLO failed in project %s: %s",
                    task_id,
                    project_id,
                    exc,
                )
                _push_sse_event(
                    project_id,
                    "task_error",
                    {
                        "project_id": project_id,
                        "task_id": task_id,
                        "error": "Task execution failed. Check server logs for details.",
                    },
                )

        thread = threading.Thread(target=_run_via_tlo, daemon=True)
        thread.start()

        return Response(
            content={
                "status": "started",
                "task_id": task_id,
                "project_id": project_id,
                "intake_tier": intake_tier_value,
            },
            status_code=200,
            media_type=MediaType.JSON,
        )

    @post("/api/project/{project_id:str}/verify-goal", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_verify_goal(project_id: str, data: dict[str, Any]) -> Response:
        """Verify the final deliverable against the original project goal.

        Loads the goal from project YAML when not provided in the request body.
        Delegates to the goal verifier and returns a compliance report.

        Args:
            project_id: The project whose goal is being verified.
            data: JSON body with optional ``goal``, ``final_output``,
                ``required_features``, ``things_to_avoid``, ``task_outputs``,
                and ``expected_outputs``.

        Returns:
            JSON with a ``report`` dict and ``corrective_tasks`` list.
        """
        import pathlib

        import yaml

        from vetinari.validation import get_goal_verifier
        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import validate_path_param

        if not validate_path_param(project_id):
            return litestar_error_response("Invalid project ID", 400)

        # Validate field types before use — int/null values must not reach the verifier.
        for _str_field in ("goal", "final_output"):
            _val = data.get(_str_field)
            if _val is not None and not isinstance(_val, str):
                return litestar_error_response(f"'{_str_field}' must be a string", 400)

        goal: str = data.get("goal", "")
        final_output: str = data.get("final_output", "")
        required_features: list[str] = data.get("required_features", [])
        things_to_avoid: list[str] = data.get("things_to_avoid", [])
        task_outputs: list[Any] = data.get("task_outputs", [])
        expected_outputs: list[str] = data.get("expected_outputs", [])

        if not goal:
            proj_dir = shared.PROJECT_ROOT / "projects" / project_id
            config_path = proj_dir / "project.yaml"
            if config_path.exists():
                with pathlib.Path(config_path).open(encoding="utf-8") as f:
                    proj_config: dict[str, Any] = yaml.safe_load(f) or {}
                goal = proj_config.get("goal", proj_config.get("description", ""))
                required_features = required_features or proj_config.get("required_features", [])
                things_to_avoid = things_to_avoid or proj_config.get("things_to_avoid", [])

        if not goal:
            return litestar_error_response("goal is required", 400)

        verifier = get_goal_verifier()
        report = verifier.verify(
            project_id=project_id,
            goal=goal,
            final_output=final_output,
            required_features=required_features,
            things_to_avoid=things_to_avoid,
            task_outputs=task_outputs,
            expected_outputs=expected_outputs,
        )

        return Response(
            content={
                "report": report.to_dict(),
                "corrective_tasks": report.get_corrective_tasks(),
            },
            status_code=200,
            media_type=MediaType.JSON,
        )

    return [
        api_project_message,
        api_project_execute,
        api_verify_goal,
    ]
