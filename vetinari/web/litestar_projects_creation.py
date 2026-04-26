"""Project creation Litestar handlers.

Owns the new-project planning route and its background auto-run helper,
extracted from ``litestar_projects_api.py`` without changing API behavior.
"""

from __future__ import annotations

import logging
from typing import Any

from vetinari.web.litestar_projects_formatters import (
    build_agent_system_prompt as _build_agent_system_prompt,
)
from vetinari.web.litestar_projects_formatters import (
    build_fallback_task_plan as _build_fallback_task_plan,
)
from vetinari.web.litestar_projects_formatters import (
    enrich_goal_with_metadata as _enrich_goal_with_metadata,
)
from vetinari.web.litestar_projects_formatters import (
    format_foreman_response as _format_foreman_response,
)
from vetinari.web.litestar_projects_formatters import normalize_task as _normalize_task

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Response, post

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def _create_projects_creation_handlers(*, default_system_prompt: str) -> list[Any]:
    """Create Litestar handlers for project creation routes.

    Args:
        default_system_prompt: Prompt used when requests omit a system prompt.

    Returns:
        List of Litestar route handler objects, or empty list when Litestar
        is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard

    def _run_tasks_background(
        proj_id: str,
        project_dir: Any,
        config_path: Any,
        project_config: dict,
        tasks: list,
        agent_system_prompt: str,
        planning_model_id: str,
        goal: str,
        category: str = "",
        tech_stack: str = "",
        priority: str = "quality",
        platforms: list | None = None,
        required_features: list | None = None,
        things_to_avoid: list | None = None,
        expected_outputs: list | None = None,
        conversation: list | None = None,
        conv_file: Any = None,
    ) -> None:
        """Run all project tasks in a background thread via the orchestrator.

        Iterates through each task, calls the adapter, persists output to disk,
        updates the project YAML status, and emits SSE events.  Exits early if
        the project's cancel flag is set.

        Args:
            proj_id: The project directory name (used for SSE and cancel flag lookup).
            project_dir: ``pathlib.Path`` to the project directory.
            config_path: ``pathlib.Path`` to ``project.yaml``.
            project_config: Parsed project config dict (mutated in place).
            tasks: Normalized task list to execute sequentially.
            agent_system_prompt: System prompt to pass to each agent call.
            planning_model_id: Model ID selected during planning.
            goal: High-level project goal string.
            category: Project category label.
            tech_stack: Technology stack description.
            priority: Execution priority hint.
            platforms: Deployment targets.
            required_features: Mandatory features list.
            things_to_avoid: Excluded patterns list.
            expected_outputs: Expected deliverables list.
            conversation: In-memory conversation list to append task outputs to.
            conv_file: ``pathlib.Path`` to ``conversation.json`` for persistence.
        """
        import json as _bg_json
        import pathlib as _bg_pathlib

        import yaml as _bg_yaml

        from vetinari.types import StatusEnum as _StatusEnum
        from vetinari.web.shared import (
            _cancel_flags,
            _cleanup_project_state,
            _push_sse_event,
            get_orchestrator,
        )

        if platforms is None:
            platforms = []
        if required_features is None:
            required_features = []
        if things_to_avoid is None:
            things_to_avoid = []
        if expected_outputs is None:
            expected_outputs = []
        if conversation is None:
            conversation = []

        orb = get_orchestrator()
        _push_sse_event(proj_id, "execution_start", {"project_id": proj_id, "task_count": len(tasks)})

        for i, task in enumerate(tasks):
            # Check cancel flag before each task
            if proj_id in _cancel_flags and _cancel_flags[proj_id].is_set():
                logger.info("Background execution cancelled for project %s at task %d", proj_id, i)
                break

            tid = task.get("id", f"t{i + 1}")
            task_desc = task.get("description", "")
            _push_sse_event(proj_id, "task_start", {"project_id": proj_id, "task_id": tid, "index": i})

            try:
                result = orb.adapter.chat(planning_model_id, agent_system_prompt, task_desc)
                output = result.get("output", "")
            except Exception:
                logger.exception(
                    "Task %s execution failed in background run for project %s",
                    tid,
                    proj_id,
                )
                output = ""

            output_dir = project_dir / "outputs" / tid
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "output.txt").write_text(output, encoding="utf-8")

            task["status"] = _StatusEnum.COMPLETED.value if output else _StatusEnum.FAILED.value
            task["output"] = output[:500] if output else ""

            _push_sse_event(
                proj_id,
                "task_complete",
                {"project_id": proj_id, "task_id": tid, "status": task["status"]},
            )

        project_config["status"] = _StatusEnum.COMPLETED.value
        try:
            with _bg_pathlib.Path(config_path).open("w", encoding="utf-8") as _f:
                _bg_yaml.dump(project_config, _f, allow_unicode=True)
        except Exception:
            logger.exception("Failed to persist final status for project %s", proj_id)

        if conv_file and conversation:
            try:
                with _bg_pathlib.Path(conv_file).open("w", encoding="utf-8") as _f:
                    _bg_json.dump(conversation, _f, indent=2)
            except Exception:
                logger.exception("Failed to persist conversation for project %s", proj_id)

        _push_sse_event(proj_id, "execution_complete", {"project_id": proj_id})
        _cleanup_project_state(proj_id)

    @post("/api/new-project", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_new_project(data: dict[str, Any]) -> Response:
        """Create a new project from a high-level goal.

        Generates a plan via ``PlanModeEngine``, persists the project directory
        with YAML config and initial conversation, and optionally launches
        background task execution when ``auto_run`` is ``true``.

        Args:
            data: JSON body containing at minimum ``goal``.  Accepts
                ``model``, ``system_prompt``, ``auto_run``, ``project_name``,
                ``project_rules``, ``required_features``, ``things_to_avoid``,
                ``expected_outputs``, ``tech_stack``, ``platforms``,
                ``priority``, and ``category``.

        Returns:
            JSON with ``status``, ``project_id``, ``tasks``, ``warnings``,
            ``conversation``, and planning metadata, or HTTP 400/429/500 on
            failure.

        Raises:
            RuntimeError: If writing the project config YAML to disk fails.
                The partial project directory is cleaned up before raising.
        """
        import json as _json
        import pathlib
        import threading
        import unicodedata
        import uuid

        import yaml

        from vetinari.types import AgentType
        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import (
            _cancel_flags,
            _cancel_flags_lock,
            _get_sse_queue,
            _push_sse_event,
            _register_project_task,
            current_config,
            get_orchestrator,
        )

        # Enforce concurrent project limit
        _max_concurrent = current_config.max_concurrent_tasks
        with _cancel_flags_lock:
            _running_count = len(_cancel_flags)
        if _running_count >= _max_concurrent:
            return litestar_error_response(
                f"Too many concurrent projects ({_running_count}/{_max_concurrent}). "
                "Wait for a running project to complete or cancel one first.",
                429,
            )

        # Validate string fields before use — .get() returns None when key is
        # present with a null value, which would crash unicodedata.normalize().
        _raw_goal = data.get("goal", "")
        _raw_project_name = data.get("project_name", "")
        if not isinstance(_raw_goal, str):
            return litestar_error_response("'goal' must be a string", 400)
        if not isinstance(_raw_project_name, str):
            return litestar_error_response("'project_name' must be a string", 400)

        goal = unicodedata.normalize("NFC", _raw_goal)
        model = data.get("model", "")
        system_prompt = data.get("system_prompt", default_system_prompt)
        auto_run = data.get("auto_run", False)
        project_name = unicodedata.normalize("NFC", _raw_project_name)
        project_rules = data.get("project_rules", "")
        required_features = data.get("required_features", [])
        things_to_avoid = data.get("things_to_avoid", [])
        expected_outputs = data.get("expected_outputs", [])
        tech_stack = data.get("tech_stack", "")
        platforms = data.get("platforms", [])
        priority = data.get("priority", "quality")
        category = data.get("category", "")

        if not goal:
            return litestar_error_response("goal is required", 400)

        try:
            from vetinari.adapter_manager import get_adapter_manager

            _am = get_adapter_manager()
            _am.discover_models()
        except Exception:
            logger.warning("Could not verify model availability for new project — proceeding anyway")

        orb = get_orchestrator()
        if hasattr(orb.model_pool, "discover_models"):
            orb.model_pool.discover_models()
        available_models = orb.model_pool.models

        from vetinari.planning.plan_mode import PlanModeEngine
        from vetinari.planning.plan_types import PlanGenerationRequest

        engine = PlanModeEngine()
        _enriched_goal = _enrich_goal_with_metadata(
            goal=goal,
            category=category,
            tech_stack=tech_stack,
            priority=priority,
            platforms=platforms,
            required_features=required_features,
            things_to_avoid=things_to_avoid,
            expected_outputs=expected_outputs,
        )

        _temp_proj_id = f"planning_{uuid.uuid4().hex[:8]}"
        _push_sse_event(
            _temp_proj_id,
            "planning_start",
            {"goal": goal[:200], "category": category, "model": model},
        )
        try:
            plan = engine.generate_plan(PlanGenerationRequest(goal=_enriched_goal, constraints=system_prompt))
        except Exception:
            logger.exception("Plan generation failed for goal: %s", goal[:200])
            return litestar_error_response("Plan generation failed — check server logs for details", 500)

        raw_tasks = [s.to_dict() for s in plan.subtasks]
        tasks = [_normalize_task(t, i, goal) for i, t in enumerate(raw_tasks)]

        plan_notes = getattr(plan, "notes", None) or getattr(plan, "policy_notes", None) or ""
        planning_model = (
            plan_notes.split(": ")[-1]
            if plan_notes
            else (model or (available_models[0].get("name", "") if available_models else ""))
        )
        project_config: dict[str, Any] = {
            "project_name": (project_name or goal.split("\n")[0])
            .removeprefix("Project: ")
            .replace("\n", " ")
            .strip()[:60],
            "description": goal,
            "high_level_goal": goal,
            "goal": goal,
            "tasks": tasks,
            "model": planning_model,
            "active_model_id": model or planning_model,
            "plan_notes": getattr(plan, "notes", "") or "",
            "warnings": getattr(plan, "warnings", []),
            "system_prompt": system_prompt,
            "project_rules": project_rules,
            "required_features": required_features,
            "things_to_avoid": things_to_avoid,
            "expected_outputs": expected_outputs,
            "tech_stack": tech_stack,
            "platforms": platforms,
            "priority": priority,
            "category": category,
            "status": "planned" if not auto_run else "running",
            "archived": False,
        }

        # Serialize before touching the filesystem — a YAML error must never
        # leave a half-created directory on disk.
        project_config_yaml = yaml.dump(project_config, allow_unicode=True)

        project_dir = shared.PROJECT_ROOT / "projects" / f"project_{uuid.uuid4().hex[:12]}"
        project_dir.mkdir(parents=True, exist_ok=True)
        try:
            config_path = project_dir / "project.yaml"
            with config_path.open("w", encoding="utf-8") as f:
                f.write(project_config_yaml)
        except Exception as _write_exc:
            try:
                # VET142-excluded: lifecycle-fenced deletion via RecycleStore.retire
                from vetinari.safety.recycle import RecycleStore

                RecycleStore().retire(
                    project_dir,
                    reason="new_project: rollback of partial directory after config write failure",
                )
            except Exception as _rm_exc:
                logger.warning(
                    "new_project: could not clean up partial directory %s after write failure: %s",
                    project_dir,
                    _rm_exc,
                )
            raise RuntimeError("Failed to write project config") from _write_exc

        if project_rules:
            try:
                from vetinari.rules_manager import get_rules_manager

                rm = get_rules_manager()
                rules_list = [r.strip() for r in project_rules.splitlines() if r.strip()]
                rm.set_project_rules(project_dir.name, rules_list)
            except Exception as _rules_err:
                logger.warning(
                    "Could not save project rules for %s: %s",
                    project_dir.name,
                    _rules_err,
                )

        agent_system_prompt = _build_agent_system_prompt(
            system_prompt=system_prompt,
            category=category,
            tech_stack=tech_stack,
            priority=priority,
            platforms=platforms,
            required_features=required_features,
            things_to_avoid=things_to_avoid,
            expected_outputs=expected_outputs,
        )

        _plan_notes = getattr(plan, "notes", None) or ""
        planning_model_id = model or (
            _plan_notes.split(": ")[-1]
            if ": " in _plan_notes
            else (available_models[0].get("name", "") if available_models else "")
        )

        model_response = ""
        try:
            try:
                from vetinari.agents.contracts import AgentTask
                from vetinari.orchestration.two_layer import get_two_layer_orchestrator

                _tlo = get_two_layer_orchestrator()
                _project_context: dict[str, Any] = {
                    "system_prompt": agent_system_prompt,
                    "model_id": planning_model_id,
                    "category": category,
                    "tech_stack": tech_stack,
                    "priority": priority,
                    "platforms": platforms,
                    "required_features": required_features,
                    "things_to_avoid": things_to_avoid,
                    "expected_outputs": expected_outputs,
                    "project_name": project_config.get("project_name", ""),
                }
                _plan_task = AgentTask(
                    task_id=f"plan_{project_dir.name}",
                    agent_type=AgentType.FOREMAN,
                    description="Generate initial plan response",
                    prompt=goal,
                    context=_project_context,
                )
                _plan_agent = _tlo._get_agent(AgentType.FOREMAN.value)
                if _plan_agent is not None:
                    _plan_result = _plan_agent.execute(_plan_task)
                    if _plan_result.success and _plan_result.output:
                        model_response = _format_foreman_response(_plan_result.output, tasks)
            except Exception as _orch_err:
                logger.warning(
                    "Orchestrator pipeline unavailable for planning (%s) — falling back to adapter",
                    _orch_err,
                )

            if not model_response:
                result = orb.adapter.chat(planning_model_id, agent_system_prompt, goal)
                model_response = result.get("output", "")
            logger.info("Model response received: %d chars", len(model_response))
        except Exception as e:
            logger.error("Error getting model response for new project: %s", e)
            model_response = ""

        if model_response and len(model_response) > 10:
            task_plan = model_response
        else:
            task_plan = _build_fallback_task_plan(tasks, getattr(plan, "warnings", []))

        conversation: list[dict[str, Any]] = [
            {"role": "user", "content": goal},
            {"role": "assistant", "content": task_plan},
        ]

        conv_file = project_dir / "conversation.json"
        with pathlib.Path(conv_file).open("w", encoding="utf-8") as f:
            _json.dump(conversation, f, indent=2)

        if auto_run:
            _proj_id = project_dir.name
            _register_project_task(_proj_id)
            _get_sse_queue(_proj_id)

            thread = threading.Thread(
                target=_run_tasks_background,
                kwargs={
                    "proj_id": _proj_id,
                    "project_dir": project_dir,
                    "config_path": config_path,
                    "project_config": project_config,
                    "tasks": tasks,
                    "agent_system_prompt": agent_system_prompt,
                    "planning_model_id": planning_model_id,
                    "goal": goal,
                    "category": category,
                    "tech_stack": tech_stack,
                    "priority": priority,
                    "platforms": platforms,
                    "required_features": required_features,
                    "things_to_avoid": things_to_avoid,
                    "expected_outputs": expected_outputs,
                    "conversation": conversation,
                    "conv_file": conv_file,
                },
                daemon=True,
            )
            thread.start()

        return Response(
            content={
                "status": "planned" if not auto_run else "started",
                "project_id": project_dir.name,
                "project_path": str(project_dir),
                "tasks": tasks,
                "results": [],
                "model": project_config["model"],
                "active_model_id": project_config.get("active_model_id", ""),
                "warnings": getattr(plan, "warnings", []),
                "conversation": conversation,
                "needs_context": getattr(plan, "needs_context", False),
                "follow_up_question": getattr(plan, "follow_up_question", ""),
            },
            status_code=200,
            media_type=MediaType.JSON,
        )

    return [api_new_project]
