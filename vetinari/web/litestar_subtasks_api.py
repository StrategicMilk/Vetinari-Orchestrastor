"""Subtask, assignment, template, and goal verification handlers.

Native Litestar equivalents of the routes previously registered by
``vetinari.web.subtasks_api``. Part of Flask->Litestar migration (ADR-0066).
URL paths are identical to the Flask originals.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, get, post, put
    from litestar.params import Parameter

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_subtasks_api_handlers() -> list[Any]:
    """Create Litestar handlers for subtasks, assignments, templates, and goal verification.

    Replicates the eleven routes from ``vetinari.web.subtasks_api``:
    subtask CRUD (``GET``/``POST`` ``/api/v1/subtasks/{plan_id}``,
    ``PUT /api/v1/subtasks/{plan_id}/{subtask_id}``,
    ``GET /api/v1/subtasks/{plan_id}/tree``), assignment management
    (``POST /api/v1/assignments/execute-pass``,
    ``GET /api/v1/assignments/{plan_id}``,
    ``PUT /api/v1/assignments/{plan_id}/{subtask_id}``),
    template listing and migration
    (``GET /api/v1/templates/versions``, ``GET /api/v1/templates``,
    ``POST /api/v1/plans/{plan_id}/migrate_templates``), and goal
    verification (``POST /api/v1/project/{project_id}/verify-goal``).

    Returns an empty list when Litestar is not installed, so the factory is
    safe to call in Flask-only environments.

    Returns:
        List of Litestar route handler objects ready to register on a Router
        or Application.  Empty when Litestar is unavailable.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.responses import litestar_error_response
    from vetinari.web.shared import PROJECT_ROOT, validate_path_param

    # ── Subtask routes ────────────────────────────────────────────────────────

    @get("/api/v1/subtasks/{plan_id:str}", media_type=MediaType.JSON)
    async def api_get_subtasks(
        plan_id: str,
        parent_id: str | None = Parameter(query="parent_id", default=None),
    ) -> dict[str, Any]:
        """Return the subtasks for a plan, optionally filtered by parent.

        When ``parent_id`` is omitted, only root-level subtasks are returned.

        Args:
            plan_id: The plan identifier.
            parent_id: Optional parent subtask identifier for filtering.

        Returns:
            JSON with ``plan_id``, ``subtasks``, and ``total``.
        """
        if not validate_path_param(plan_id):
            return litestar_error_response(f"Invalid plan_id: {plan_id!r}", 400)

        from vetinari.planning.subtask_tree import subtask_tree

        if parent_id:
            subtasks = subtask_tree.get_subtasks_by_parent(plan_id, parent_id)
        else:
            subtasks = subtask_tree.get_root_subtasks(plan_id)

        return {
            "plan_id": plan_id,
            "subtasks": [s.to_dict() for s in subtasks],
            "total": len(subtasks),
        }

    @post("/api/v1/subtasks/{plan_id:str}", media_type=MediaType.JSON)
    async def api_create_subtask(plan_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Create a new subtask within a plan.

        Args:
            plan_id: The plan identifier.
            data: Request body.  Supported fields:
                parent_id (str): Parent subtask ID (default ``"root"``).
                depth (int): Nesting depth (default 0).
                description (str): Human-readable task description.
                prompt (str): Execution prompt for the assigned agent.
                agent_type (str): Agent type string (default ``"builder"``).
                max_depth (int): Maximum allowed decomposition depth (default 14).
                max_depth_override (int): Per-subtask depth override (default 0).
                dod_level (str): Definition-of-Done level (default ``"Standard"``).
                dor_level (str): Definition-of-Ready level (default ``"Standard"``).
                estimated_effort (float): Effort estimate in arbitrary units (default 1.0).
                inputs (list): List of input artefacts.
                outputs (list): List of expected output artefacts.
                decomposition_seed (str): Optional seed string to guide decomposition.

        Returns:
            Subtask dict on success.
        """
        if not validate_path_param(plan_id):
            return litestar_error_response(f"Invalid plan_id: {plan_id!r}", 400)

        # Guard against orphan subtasks: verify the plan exists before creating
        # a subtask entry.  SubtaskTree creates entries unconditionally, so we
        # must check here.
        from vetinari.planning import get_plan_manager

        _plan_manager = get_plan_manager()
        _plan = _plan_manager.get_plan(plan_id)
        if not _plan:
            from vetinari.planning.plan_mode import get_plan_engine

            _plan = get_plan_engine().get_plan(plan_id)
        if not _plan:
            return litestar_error_response("Plan not found", 404)

        from vetinari.planning.subtask_tree import subtask_tree

        subtask = subtask_tree.create_subtask(
            plan_id=plan_id,
            parent_id=data.get("parent_id", "root"),
            depth=data.get("depth", 0),
            description=data.get("description", ""),
            prompt=data.get("prompt", ""),
            agent_type=data.get("agent_type", "builder"),
            max_depth=data.get("max_depth", 14),
            max_depth_override=data.get("max_depth_override", 0),
            dod_level=data.get("dod_level", "Standard"),
            dor_level=data.get("dor_level", "Standard"),
            estimated_effort=data.get("estimated_effort", 1.0),
            inputs=data.get("inputs", []),
            outputs=data.get("outputs", []),
            decomposition_seed=data.get("decomposition_seed", ""),
        )
        return subtask.to_dict()

    @put("/api/v1/subtasks/{plan_id:str}/{subtask_id:str}", media_type=MediaType.JSON)
    async def api_update_subtask(
        plan_id: str, subtask_id: str, data: dict[str, Any]
    ) -> Any:
        """Update fields on an existing subtask.

        Args:
            plan_id: The plan identifier.
            subtask_id: The subtask identifier.
            data: Request body containing any subtask fields to update.

        Returns:
            Updated subtask dict on success, or HTTP 404 when not found.
        """
        if not validate_path_param(plan_id):
            return litestar_error_response(f"Invalid plan_id: {plan_id!r}", 400)
        if not validate_path_param(subtask_id):
            return litestar_error_response(f"Invalid subtask_id: {subtask_id!r}", 400)

        from vetinari.planning.subtask_tree import subtask_tree

        subtask = subtask_tree.update_subtask(plan_id, subtask_id, data)
        if not subtask:
            return litestar_error_response("Subtask not found", 404)
        return subtask.to_dict()

    @get("/api/v1/subtasks/{plan_id:str}/tree", media_type=MediaType.JSON)
    async def api_get_subtask_tree(plan_id: str) -> dict[str, Any]:
        """Return the complete flattened subtask tree for a plan.

        Args:
            plan_id: The plan identifier.

        Returns:
            JSON with ``plan_id``, ``subtasks``, ``total``, and ``depth``.
        """
        if not validate_path_param(plan_id):
            return litestar_error_response(f"Invalid plan_id: {plan_id!r}", 400)

        from vetinari.planning.subtask_tree import subtask_tree

        all_subtasks = subtask_tree.get_all_subtasks(plan_id)
        tree_depth = subtask_tree.get_tree_depth(plan_id)

        return {
            "plan_id": plan_id,
            "subtasks": [s.to_dict() for s in all_subtasks],
            "total": len(all_subtasks),
            "depth": tree_depth,
        }

    # ── Assignment routes ─────────────────────────────────────────────────────

    @post("/api/v1/assignments/execute-pass", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_assignment_execute_pass(data: dict[str, Any]) -> Any:
        """Execute an assignment pass over a plan's subtasks.

        Args:
            data: Request body with required ``plan_id`` (str) and optional
                ``auto_assign`` (bool, default ``True``).

        Returns:
            JSON result dict from ``execute_assignment_pass`` on success, or
            HTTP 400 when ``plan_id`` is missing.
        """
        from vetinari.planning.assignment_pass import execute_assignment_pass

        plan_id = data.get("plan_id")
        if not plan_id:
            return litestar_error_response("plan_id required", 400)

        # Guard: verify the plan exists before running an assignment pass.
        # execute_assignment_pass silently returns zero-count for nonexistent
        # plans because get_all_subtasks([]) returns [] without error.
        from vetinari.planning import get_plan_manager

        _pm = get_plan_manager()
        _plan = _pm.get_plan(plan_id)
        if not _plan:
            from vetinari.planning.plan_mode import get_plan_engine

            _plan = get_plan_engine().get_plan(plan_id)
        if not _plan:
            return litestar_error_response("Plan not found", 404)

        auto_assign: bool = data.get("auto_assign", True)
        return execute_assignment_pass(plan_id, auto_assign)

    @get("/api/v1/assignments/{plan_id:str}", media_type=MediaType.JSON)
    async def api_get_assignments(plan_id: str) -> dict[str, Any]:
        """Return all agent assignments for a plan's subtasks.

        Args:
            plan_id: The plan identifier.

        Returns:
            JSON with ``plan_id``, ``assignments``, and ``total``.
        """
        if not validate_path_param(plan_id):
            return litestar_error_response(f"Invalid plan_id: {plan_id!r}", 400)

        from vetinari.planning.subtask_tree import subtask_tree

        all_subtasks = subtask_tree.get_all_subtasks(plan_id)

        assignments = [
            {
                "subtask_id": st.subtask_id,
                "description": st.description,
                "agent_type": st.agent_type,
                "assigned_agent": st.assigned_agent,
                "status": st.status,
                "depth": st.depth,
            }
            for st in all_subtasks
        ]

        return {"plan_id": plan_id, "assignments": assignments, "total": len(assignments)}

    @put(
        "/api/v1/assignments/{plan_id:str}/{subtask_id:str}",
        media_type=MediaType.JSON,
        guards=[admin_guard],
    )
    async def api_override_assignment(
        plan_id: str, subtask_id: str, data: dict[str, Any]
    ) -> Any:
        """Override the agent assigned to a specific subtask.

        Args:
            plan_id: The plan identifier.
            subtask_id: The subtask identifier.
            data: Request body with required ``assigned_agent`` (str) key.

        Returns:
            Updated subtask dict on success, or HTTP 400 when
            ``assigned_agent`` is missing, or HTTP 404 when not found.
        """
        if not validate_path_param(plan_id):
            return litestar_error_response(f"Invalid plan_id: {plan_id!r}", 400)
        if not validate_path_param(subtask_id):
            return litestar_error_response(f"Invalid subtask_id: {subtask_id!r}", 400)

        from vetinari.planning.subtask_tree import subtask_tree
        from vetinari.types import StatusEnum

        assigned_agent = data.get("assigned_agent")
        if not assigned_agent:
            return litestar_error_response("assigned_agent required", 400)

        subtask = subtask_tree.update_subtask(
            plan_id,
            subtask_id,
            {"assigned_agent": assigned_agent, "status": StatusEnum.ASSIGNED.value},
        )
        if not subtask:
            return litestar_error_response("Subtask not found", 404)
        return subtask.to_dict()

    # ── Template routes ───────────────────────────────────────────────────────

    @get("/api/v1/templates/versions", media_type=MediaType.JSON)
    async def api_template_versions() -> dict[str, Any]:
        """Return available template versions and the current default.

        Returns:
            JSON with ``versions`` list and ``default`` version string.
        """
        try:
            from vetinari.template_loader import template_loader

            versions = template_loader.list_versions()
            default = template_loader.default_version()
            return {"versions": versions, "default": default}
        except Exception:
            logger.warning("Template loader unavailable — cannot serve template versions")
            return litestar_error_response("Template subsystem unavailable", 503)

    @get("/api/v1/templates", media_type=MediaType.JSON)
    async def api_templates(
        version: str | None = Parameter(query="version", default=None),
        agent_type: str | None = Parameter(query="agent_type", default=None),
    ) -> dict[str, Any]:
        """Return templates, optionally filtered by version and agent type.

        Args:
            version: Optional template version to load.
            agent_type: Optional agent type filter.

        Returns:
            JSON with ``templates``, ``total``, and ``version`` keys.
        """
        try:
            from vetinari.template_loader import template_loader

            templates = template_loader.load_templates(version=version, agent_type=agent_type)
            return {
                "templates": templates,
                "total": len(templates),
                "version": version or template_loader.default_version(),
            }
        except Exception:
            logger.warning("Template loader unavailable — cannot serve templates")
            return litestar_error_response("Template subsystem unavailable", 503)

    @post(
        "/api/v1/plans/{plan_id:str}/migrate_templates",
        media_type=MediaType.JSON,
        guards=[admin_guard],
    )
    async def api_migrate_templates(plan_id: str, data: dict[str, Any]) -> Any:
        """Migrate or preview migration of a plan to a target template version.

        Args:
            plan_id: The plan identifier.
            data: Request body with required ``target_version`` (str) and
                optional ``dry_run`` (bool, default ``True``).  When
                ``dry_run`` is ``True``, returns a diff preview without
                applying changes.  When ``False``, applies the migration.

        Returns:
            For dry runs: JSON with ``plan_id``, ``from_version``,
            ``to_version``, ``dry_run``, ``differences``, and
            ``recommendation``.  For live runs: JSON with ``plan_id``,
            ``from_version``, ``to_version``, ``dry_run``, ``status``, and
            ``message``.  HTTP 400 when ``target_version`` is missing or
            invalid; HTTP 404 when the plan does not exist.
        """
        from vetinari.planning import get_plan_manager
        from vetinari.template_loader import template_loader

        target_version = data.get("target_version")
        dry_run: bool = data.get("dry_run", True)

        if not target_version:
            return litestar_error_response("target_version required", 400)

        available_versions = template_loader.list_versions()
        if target_version not in available_versions:
            return litestar_error_response(f"Invalid target version. Available: {available_versions}", 400)

        # Resolve plan from either store — PlanManager owns plan_* IDs,
        # PlanModeEngine owns pmode-* IDs.  Try PlanManager first, then fall
        # back to PlanModeEngine so both ID namespaces work.
        plan_manager = get_plan_manager()
        plan = plan_manager.get_plan(plan_id)
        use_engine_store = False
        if not plan:
            from vetinari.planning.plan_mode import get_plan_engine

            engine = get_plan_engine()
            plan = engine.get_plan(plan_id)
            use_engine_store = True
        if not plan:
            return litestar_error_response("Plan not found", 404)

        from_version = plan.template_version

        if dry_run:
            target_templates = template_loader.load_templates(version=target_version)
            current_templates = (
                template_loader.load_templates(version=from_version) if from_version != target_version else []
            )

            current_ids = {t["template_id"] for t in current_templates}
            target_ids = {t["template_id"] for t in target_templates}

            differences: list[dict[str, Any]] = []
            added = list(target_ids - current_ids)
            removed = list(current_ids - target_ids)
            if added:
                differences.append({"type": "added", "template_ids": added})
            if removed:
                differences.append({"type": "removed", "template_ids": removed})

            recommendation = "re-decompose" if differences else "map-in-place"
            return {
                "plan_id": plan_id,
                "from_version": from_version,
                "to_version": target_version,
                "dry_run": dry_run,
                "differences": differences,
                "recommendation": recommendation,
            }

        plan.template_version = target_version
        if use_engine_store:
            from vetinari.planning.plan_mode import get_plan_engine

            get_plan_engine()._persist_plan(plan)
        else:
            plan_manager._save_plan(plan)
        return {
            "plan_id": plan_id,
            "from_version": from_version,
            "to_version": target_version,
            "dry_run": dry_run,
            "status": "migrated",
            "message": f"Plan migrated from {from_version} to {target_version}",
        }

    # ── Goal verification routes ──────────────────────────────────────────────

    @post(
        "/api/v1/project/{project_id:str}/verify-goal",
        media_type=MediaType.JSON,
    )
    async def api_verify_goal(project_id: str, data: dict[str, Any]) -> Any:
        """Verify the final deliverable against the original project goal.

        Args:
            project_id: The project identifier.
            data: Request body.  Supported fields:
                goal (str): Goal text.  When omitted, loaded from the project YAML.
                final_output (str): Assembled final output text.
                required_features (list): Features the output must include.
                things_to_avoid (list): Anti-patterns to detect.
                task_outputs (list): Individual task output snippets.
                expected_outputs (list): Expected output artefacts.

        Returns:
            JSON with ``report`` dict and ``corrective_tasks`` list on
            success, or HTTP 400 when no goal can be determined.
        """
        if not validate_path_param(project_id):
            return litestar_error_response(f"Invalid project_id: {project_id!r}", 400)

        import yaml as _yaml

        from vetinari.validation import get_goal_verifier

        goal: str = data.get("goal", "")
        final_output: str = data.get("final_output", "")
        required_features: list[Any] = data.get("required_features", [])
        things_to_avoid: list[Any] = data.get("things_to_avoid", [])
        task_outputs: list[Any] = data.get("task_outputs", [])
        expected_outputs: list[Any] = data.get("expected_outputs", [])

        if not goal:
            # Try to load from project file
            config_path = PROJECT_ROOT / "projects" / project_id / "project.yaml"
            if config_path.exists():
                from pathlib import Path

                with Path(config_path).open(encoding="utf-8") as f:
                    proj_config = _yaml.safe_load(f) or {}
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

        return {
            "report": report.to_dict(),
            "corrective_tasks": report.get_corrective_tasks(),
        }

    return [
        api_get_subtasks,
        api_create_subtask,
        api_update_subtask,
        api_get_subtask_tree,
        api_assignment_execute_pass,
        api_get_assignments,
        api_override_assignment,
        api_template_versions,
        api_templates,
        api_migrate_templates,
        api_verify_goal,
    ]
