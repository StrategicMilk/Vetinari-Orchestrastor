"""Plan API — native Litestar handlers for plan generation, approval, and execution.

Migrated from ``vetinari.planning.plan_api`` Flask Blueprint. Native Litestar
equivalents (ADR-0066). URL paths identical to Flask.

Endpoints
---------
    POST /api/plan/generate                                   — generate a plan from a goal
    GET  /api/plan/{plan_id}                                  — retrieve a plan by ID
    POST /api/plan/{plan_id}/approve                          — approve or reject a plan
    POST /api/plan/{plan_id}/subtasks/{subtask_id}/approve    — approve or reject a subtask
    GET  /api/plan/{plan_id}/history                          — plan history and memory
    GET  /api/plan/history                                    — all plan history
    GET  /api/plan/{plan_id}/subtasks                         — list subtasks for a plan
    GET  /api/plan/status                                     — plan mode status/config
    GET  /api/plan/templates                                  — plan templates by domain
    GET  /api/plan/{plan_id}/explanations                     — plan explanation
    GET  /api/plan/{plan_id}/subtasks/{subtask_id}/explanation — subtask explanation
    POST /api/coding/task                                     — create and run a coding task
    GET  /api/coding/task/{task_id}                           — get coding task status
    POST /api/coding/multi-step                               — run multi-step coding tasks
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, get, post

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_plan_api_handlers() -> list[Any]:
    """Create all Litestar route handlers for the plan API.

    Called by the Litestar app factory to register plan management,
    approval, subtask, and coding-agent endpoints.

    Returns:
        List of Litestar route handler functions, or empty list when
        Litestar is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        logger.debug("Litestar not available — plan API handlers not registered")
        return []

    from vetinari.memory import PLAN_ADMIN_TOKEN
    from vetinari.planning.plan_mode import PLAN_MODE_DEFAULT, PLAN_MODE_ENABLE, get_plan_engine
    from vetinari.planning.plan_types import PlanApprovalRequest, PlanGenerationRequest, TaskDomain
    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.request_validation import json_object_body
    from vetinari.web.responses import litestar_error_response

    # -- POST /api/plan/generate ----------------------------------------------

    @post("/api/plan/generate", media_type=MediaType.JSON, guards=[admin_guard])
    async def generate_plan(data: dict[str, Any] | None = None) -> Any:
        """Generate a plan from a goal description.

        Accepts a JSON body with the goal and optional generation parameters.
        Returns the generated plan summary including risk score and candidate
        plans. Requires plan mode to be enabled and admin auth.

        Args:
            data: JSON request body with ``goal`` (required) and optional
                ``constraints``, ``plan_depth_cap``, ``max_candidates``,
                ``domain_hint``, ``dry_run``, and ``risk_threshold``.

        Returns:
            JSON object with plan ID, status, risk score, and subtask count
            on success; 400 when goal is missing; 403 when plan mode is
            disabled; 500 on generation failure.
        """
        if not PLAN_MODE_ENABLE:
            return litestar_error_response("Plan mode disabled", 403, details={"message": "Plan mode is disabled"})

        body = json_object_body(data)
        if body is None:
            return litestar_error_response("Request body must be a JSON object", 400)
        goal = body.get("goal", "")
        if not goal:
            return litestar_error_response("Missing goal", 400, details={"message": "Goal is required"})

        try:
            domain_hint = None
            if body.get("domain_hint"):
                domain_hint = TaskDomain(body["domain_hint"])

            req = PlanGenerationRequest(
                goal=goal,
                constraints=body.get("constraints", ""),
                plan_depth_cap=body.get("plan_depth_cap", 16),
                max_candidates=body.get("max_candidates", 3),
                domain_hint=domain_hint,
                dry_run=body.get("dry_run", False),
                risk_threshold=body.get("risk_threshold", 0.25),
            )

            engine = get_plan_engine()
            plan = engine.generate_plan(req)

            return {
                "success": True,
                "plan_id": plan.plan_id,
                "version": plan.plan_version,
                "goal": plan.goal,
                "status": plan.status.value,
                "risk_score": plan.risk_score,
                "risk_level": plan.risk_level.value,
                "subtask_count": len(plan.subtasks),
                "dry_run": plan.dry_run,
                "auto_approved": plan.auto_approved,
                "plan_candidates": [c.to_dict() for c in plan.plan_candidates],
                "chosen_plan_id": plan.chosen_plan_id,
                "plan_justification": plan.plan_justification,
                "created_at": plan.created_at,
            }

        except Exception:
            logger.exception("Plan generation failed")
            return litestar_error_response("Plan generation failed", 500)

    # -- GET /api/plan/history (must be before /api/plan/{plan_id}) -----------

    @get("/api/plan/history", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_all_plan_history(
        goal_contains: str = "",
        limit: int = 10,
    ) -> Any:
        """Return all plan history with optional filtering.

        Args:
            goal_contains: Filter plans by goal containing this substring.
            limit: Maximum number of plans to return (default 10).

        Returns:
            JSON object with ``plans`` list and ``count`` on success; 403 when
            plan mode is disabled; 500 on failure.
        """
        if not PLAN_MODE_ENABLE:
            return litestar_error_response("Plan mode disabled", 403, details={"message": "Plan mode is disabled"})

        try:
            engine = get_plan_engine()
            plans = engine.get_plan_history(
                goal_contains=goal_contains or None,
                limit=limit,
            )
            return {"success": True, "plans": plans, "count": len(plans)}

        except Exception:
            logger.exception("Failed to get plan history")
            return litestar_error_response("Failed to get plan history", 500)

    # -- GET /api/plan/status -------------------------------------------------

    @get("/api/plan/status", media_type=MediaType.JSON)
    async def get_plan_mode_status() -> dict[str, Any]:
        """Return plan mode status flags and memory statistics.

        Returns:
            JSON object with ``plan_mode_enabled``, ``plan_mode_default``,
            ``config``, and ``memory_stats`` fields.
        """
        from vetinari.memory import get_memory_store

        memory_degraded = False
        stats: dict[str, Any] = {}
        try:
            memory = get_memory_store()
            stats = memory.get_memory_stats()
        except Exception:
            logger.warning("get_plan_mode_status: memory store unavailable — reporting degraded status")
            memory_degraded = True

        if memory_degraded:
            return {
                "status": "degraded",
                "plan_mode_enabled": PLAN_MODE_ENABLE,
                "plan_mode_default": PLAN_MODE_DEFAULT,
                "config": {
                    "PLAN_MODE_ENABLE": PLAN_MODE_ENABLE,
                    "PLAN_MODE_DEFAULT": PLAN_MODE_DEFAULT,
                    "PLAN_ADMIN_TOKEN_SET": bool(PLAN_ADMIN_TOKEN),
                },
                "memory_stats": {},
            }

        return {
            "success": True,
            "plan_mode_enabled": PLAN_MODE_ENABLE,
            "plan_mode_default": PLAN_MODE_DEFAULT,
            "config": {
                "PLAN_MODE_ENABLE": PLAN_MODE_ENABLE,
                "PLAN_MODE_DEFAULT": PLAN_MODE_DEFAULT,
                "PLAN_ADMIN_TOKEN_SET": bool(PLAN_ADMIN_TOKEN),
            },
            "memory_stats": stats,
        }

    # -- GET /api/plan/templates ----------------------------------------------

    @get("/api/plan/templates", media_type=MediaType.JSON)
    async def get_plan_templates() -> Any:
        """Return available plan templates grouped by task domain.

        Returns:
            JSON object with ``templates`` dict (keyed by domain) and
            ``domains`` list on success; 500 on failure.
        """
        try:
            engine = get_plan_engine()
            templates: dict[str, Any] = {}
            for domain in TaskDomain:
                domain_templates = engine._domain_templates.get(domain, [])
                templates[domain.value] = [
                    {
                        "description": t.get("description"),
                        "definition_of_done": t.get("definition_of_done").criteria
                        if t.get("definition_of_done")
                        else [],
                        "definition_of_ready": t.get("definition_of_ready").prerequisites
                        if t.get("definition_of_ready")
                        else [],
                    }
                    for t in domain_templates
                ]

            return {"success": True, "templates": templates, "domains": [d.value for d in TaskDomain]}

        except Exception:
            logger.exception("Failed to get templates")
            return litestar_error_response("Failed to get templates", 500)

    # -- GET /api/plan/{plan_id} ----------------------------------------------

    @get("/api/plan/{plan_id:str}", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_plan(plan_id: str) -> Any:
        """Return full plan details including subtasks for a given plan ID.

        Args:
            plan_id: URL path parameter — the plan identifier to retrieve.

        Returns:
            JSON object with nested ``plan`` dict on success; 403 when plan
            mode is disabled; 404 when not found; 500 on failure.
        """
        if not PLAN_MODE_ENABLE:
            return litestar_error_response("Plan mode disabled", 403, details={"message": "Plan mode is disabled"})

        try:
            engine = get_plan_engine()
            plan = engine.get_plan(plan_id)

            if not plan:
                return litestar_error_response("Plan not found", 404, details={"plan_id": plan_id})

            return {
                "success": True,
                "plan": {
                    "plan_id": plan.plan_id,
                    "version": plan.plan_version,
                    "goal": plan.goal,
                    "constraints": plan.constraints,
                    "status": plan.status.value,
                    "risk_score": plan.risk_score,
                    "risk_level": plan.risk_level.value,
                    "dry_run": plan.dry_run,
                    "auto_approved": plan.auto_approved,
                    "approved_by": plan.approved_by,
                    "approved_at": plan.approved_at,
                    "subtasks": [s.to_dict() for s in plan.subtasks],
                    "dependencies": plan.dependencies,
                    "plan_candidates": [c.to_dict() for c in plan.plan_candidates],
                    "chosen_plan_id": plan.chosen_plan_id,
                    "plan_justification": plan.plan_justification,
                    "created_at": plan.created_at,
                    "updated_at": plan.updated_at,
                    "completed_at": plan.completed_at,
                },
            }

        except Exception:
            logger.exception("Failed to get plan %s", plan_id)
            return litestar_error_response("Failed to get plan", 500)

    # -- POST /api/plan/{plan_id}/approve -------------------------------------

    @post("/api/plan/{plan_id:str}/approve", media_type=MediaType.JSON, guards=[admin_guard])
    async def approve_plan(plan_id: str, data: dict[str, Any] | None = None) -> Any:
        """Approve or reject a plan, recording the decision in dual memory.

        Requires ``approved`` (bool) and ``approver`` (str) in the request body.
        Persists the approval decision to the dual memory store when available.

        Args:
            plan_id: URL path parameter — the plan to approve or reject.
            data: JSON body with ``approved``, ``approver``, and optional
                ``reason``, ``audit_id``, ``risk_score``, ``timestamp``,
                ``approval_schema_version``.

        Returns:
            JSON object with updated plan status and audit ID on success; 400
            for missing required fields; 403 when plan mode is disabled; 404
            when plan not found; 500 on failure.
        """
        if not PLAN_MODE_ENABLE:
            return litestar_error_response("Plan mode disabled", 403, details={"message": "Plan mode is disabled"})

        body = json_object_body(data)
        if body is None:
            return litestar_error_response("Request body must be a JSON object", 400)

        for field in ("approved", "approver"):
            if field not in body:
                return litestar_error_response(f"Missing required field: {field}", 400)

        approved = body.get("approved", False)
        approver = body.get("approver", "admin")
        reason = body.get("reason", "")
        audit_id = body.get("audit_id")
        risk_score = body.get("risk_score")
        timestamp = body.get("timestamp")
        approval_schema_version = body.get("approval_schema_version", 1)

        try:
            req = PlanApprovalRequest(
                plan_id=plan_id,
                approved=approved,
                approver=approver,
                reason=reason,
                audit_id=audit_id,
                risk_score=risk_score,
                timestamp=timestamp or "",  # noqa: VET112 - empty fallback preserves optional request metadata contract
                approval_schema_version=approval_schema_version,
            )

            engine = get_plan_engine()
            plan = engine.approve_plan(req)

            # Log approval decision to dual memory if available
            try:
                from vetinari.memory import DUAL_MEMORY_AVAILABLE, MemoryEntry, MemoryType, get_unified_memory_store

                if DUAL_MEMORY_AVAILABLE:
                    store = get_unified_memory_store()
                    approval_entry = MemoryEntry(
                        agent="plan-approval",
                        entry_type=MemoryType.APPROVAL,
                        content=json.dumps({
                            "audit_id": audit_id,
                            "plan_id": plan_id,
                            "approved": approved,
                            "approver": approver,
                            "reason": reason,
                            "risk_score": risk_score,
                            "approval_schema_version": approval_schema_version,
                        }),
                        summary=f"Plan {plan_id} {'approved' if approved else 'rejected'} by {approver}",
                        provenance="plan_api_approve",
                    )
                    store.remember(approval_entry)
            except Exception as mem_err:
                logger.warning("Failed to log approval to memory: %s", mem_err)

            return {
                "success": True,
                "plan_id": plan.plan_id,
                "status": plan.status.value,
                "approved_by": plan.approved_by,
                "approved_at": plan.approved_at,
                "audit_id": audit_id,
            }

        except Exception as exc:
            from vetinari.exceptions import PlanningError

            if isinstance(exc, PlanningError):
                msg = str(exc)
                if "not found" in msg.lower():
                    return litestar_error_response("Plan not found", 404, details={"plan_id": plan_id})
                if "terminal state" in msg.lower():
                    return litestar_error_response(msg, 409, details={"plan_id": plan_id})
            logger.exception("Failed to approve plan %s", plan_id)
            return litestar_error_response("Failed to approve plan", 500)

    # -- POST /api/plan/{plan_id}/subtasks/{subtask_id}/approve ---------------

    @post(
        "/api/plan/{plan_id:str}/subtasks/{subtask_id:str}/approve",
        media_type=MediaType.JSON,
        guards=[admin_guard],
    )
    async def approve_subtask(
        plan_id: str,
        subtask_id: str,
        data: dict[str, Any] | None = None,
    ) -> Any:
        """Approve or reject a specific subtask within a plan.

        Requires ``approved`` (bool) and ``approver`` (str) in the request body.
        Checks whether the subtask actually requires approval before logging the
        decision.

        Args:
            plan_id: URL path parameter — the plan the subtask belongs to.
            subtask_id: URL path parameter — the subtask to approve or reject.
            data: JSON body with ``approved``, ``approver``, and optional
                ``reason``, ``audit_id``, ``risk_score``.

        Returns:
            JSON object with approval outcome and audit ID on success; 400 for
            missing required fields; 403 when plan mode is disabled; 404 when
            plan or subtask not found; 500 on failure.
        """
        if not PLAN_MODE_ENABLE:
            return litestar_error_response("Plan mode disabled", 403, details={"message": "Plan mode is disabled"})

        body = json_object_body(data)
        if body is None:
            return litestar_error_response("Request body must be a JSON object", 400)

        for field in ("approved", "approver"):
            if field not in body:
                return litestar_error_response(f"Missing required field: {field}", 400)

        approved = body.get("approved", False)
        approver = body.get("approver", "admin")
        reason = body.get("reason", "")
        audit_id = body.get("audit_id")
        risk_score = body.get("risk_score")

        try:
            engine = get_plan_engine()
            plan = engine.get_plan(plan_id)

            if not plan:
                return litestar_error_response("Plan not found", 404, details={"plan_id": plan_id})

            approval_check = engine.check_subtask_approval_required(plan, subtask_id, plan_mode=True)

            if "error" in approval_check:
                return litestar_error_response(approval_check["error"], 404)

            # Reject approval attempts for subtasks that don't require approval —
            # calling this endpoint on a non-approval subtask is a client error.
            if approval_check.get("requires_approval") is False:
                return litestar_error_response(
                    "Subtask does not require approval",
                    409,
                    details={"subtask_id": subtask_id, "requires_approval": False},
                )

            engine.log_approval_decision(
                plan_id=plan_id,
                subtask_id=subtask_id,
                approved=approved,
                approver=approver,
                reason=reason,
                risk_score=risk_score or plan.risk_score,
            )

            return {
                "success": True,
                "plan_id": plan_id,
                "subtask_id": subtask_id,
                "approved": approved,
                "approver": approver,
                "requires_approval": approval_check.get("requires_approval", False),
                "audit_id": audit_id,
            }

        except Exception:
            logger.exception("Failed to approve subtask %s in plan %s", subtask_id, plan_id)
            return litestar_error_response("Failed to approve subtask", 500)

    # -- GET /api/plan/{plan_id}/history --------------------------------------

    @get("/api/plan/{plan_id:str}/history", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_plan_history(plan_id: str) -> Any:
        """Return plan history, memory, and risk data for a specific plan.

        Args:
            plan_id: URL path parameter — the plan identifier to retrieve
                history for.

        Returns:
            JSON object with goal, status, subtasks, and timestamps on success;
            403 when plan mode is disabled; 404 when not found; 500 on failure.
        """
        if not PLAN_MODE_ENABLE:
            return litestar_error_response("Plan mode disabled", 403, details={"message": "Plan mode is disabled"})

        try:
            engine = get_plan_engine()
            plan = engine.get_plan(plan_id)

            if not plan:
                return litestar_error_response("Plan not found", 404, details={"plan_id": plan_id})

            subtasks = engine.get_subtasks(plan_id)

            return {
                "success": True,
                "plan_id": plan_id,
                "goal": plan.goal,
                "status": plan.status.value,
                "risk_score": plan.risk_score,
                "risk_level": plan.risk_level.value,
                "subtasks": [s.to_dict() for s in subtasks],
                "created_at": plan.created_at,
                "updated_at": plan.updated_at,
                "completed_at": plan.completed_at,
                "auto_approved": plan.auto_approved,
                "approved_by": plan.approved_by,
            }

        except Exception:
            logger.exception("Failed to get plan history for %s", plan_id)
            return litestar_error_response("Failed to get plan history", 500)

    # -- GET /api/plan/{plan_id}/subtasks -------------------------------------

    @get("/api/plan/{plan_id:str}/subtasks", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_plan_subtasks(plan_id: str) -> Any:
        """Return all subtasks for a given plan.

        Args:
            plan_id: URL path parameter — the plan identifier to list subtasks for.

        Returns:
            JSON object with ``subtasks`` list and ``count`` on success; 403 when
            plan mode is disabled; 404 when plan not found; 500 on failure.
        """
        if not PLAN_MODE_ENABLE:
            return litestar_error_response("Plan mode disabled", 403, details={"message": "Plan mode is disabled"})

        try:
            engine = get_plan_engine()
            plan = engine.get_plan(plan_id)

            if not plan:
                return litestar_error_response("Plan not found", 404, details={"plan_id": plan_id})

            subtasks = engine.get_subtasks(plan_id)

            return {
                "success": True,
                "plan_id": plan_id,
                "subtasks": [s.to_dict() for s in subtasks],
                "count": len(subtasks),
            }

        except Exception:
            logger.exception("Failed to get subtasks for plan %s", plan_id)
            return litestar_error_response("Failed to get subtasks", 500)

    # -- GET /api/plan/{plan_id}/explanations ---------------------------------

    @get("/api/plan/{plan_id:str}/explanations", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_plan_explanations(plan_id: str) -> Any:
        """Return the explanation JSON for a plan including blocks and summary.

        Args:
            plan_id: URL path parameter — the plan identifier to retrieve
                explanation for.

        Returns:
            JSON object with ``explanation`` (dict or null) on success; 403 when
            plan mode is disabled; 404 when not found; 500 on failure.
        """
        if not PLAN_MODE_ENABLE:
            return litestar_error_response("Plan mode disabled", 403, details={"message": "Plan mode is disabled"})

        try:
            engine = get_plan_engine()
            plan = engine.get_plan(plan_id)

            if not plan:
                return litestar_error_response("Plan not found", 404, details={"plan_id": plan_id})

            explanation_data: dict[str, Any] = {}
            if plan.plan_explanation_json:
                with contextlib.suppress(Exception):
                    explanation_data = json.loads(plan.plan_explanation_json)

            return {"success": True, "plan_id": plan_id, "explanation": explanation_data or None}

        except Exception:
            logger.exception("Failed to get plan explanations for %s", plan_id)
            return litestar_error_response("Failed to get explanations", 500)

    # -- GET /api/plan/{plan_id}/subtasks/{subtask_id}/explanation ------------

    @get(
        "/api/plan/{plan_id:str}/subtasks/{subtask_id:str}/explanation",
        media_type=MediaType.JSON,
        guards=[admin_guard],
    )
    async def get_subtask_explanation(plan_id: str, subtask_id: str) -> Any:
        """Return the explanation JSON for a specific subtask.

        Args:
            plan_id: URL path parameter — the plan the subtask belongs to.
            subtask_id: URL path parameter — the subtask identifier to retrieve
                explanation for.

        Returns:
            JSON object with ``explanation`` (dict or null) on success; 403 when
            plan mode is disabled; 404 when subtask not found; 500 on failure.
        """
        if not PLAN_MODE_ENABLE:
            return litestar_error_response("Plan mode disabled", 403, details={"message": "Plan mode is disabled"})

        try:
            engine = get_plan_engine()
            subtasks = engine.get_subtasks(plan_id)

            subtask = next((s for s in subtasks if s.subtask_id == subtask_id), None)
            if not subtask:
                return litestar_error_response("Subtask not found", 404, details={"subtask_id": subtask_id})

            explanation_data: dict[str, Any] = {}
            if subtask.subtask_explanation_json:
                with contextlib.suppress(Exception):
                    explanation_data = json.loads(subtask.subtask_explanation_json)

            return {
                "success": True,
                "plan_id": plan_id,
                "subtask_id": subtask_id,
                "explanation": explanation_data or None,
            }

        except Exception:
            logger.exception("Failed to get subtask explanation for %s/%s", plan_id, subtask_id)
            return litestar_error_response("Failed to get subtask explanation", 500)

    # -- POST /api/coding/task ------------------------------------------------

    @post("/api/coding/task", media_type=MediaType.JSON, guards=[admin_guard])
    async def create_coding_task(data: dict[str, Any]) -> Any:
        """Create and execute a coding task via the coding agent.

        Accepts a JSON body specifying the task type (scaffold, implement, test,
        review), target language, description, and optional constraints. Returns
        the generated artifact. Returns 503 when the coding agent is unavailable.

        Args:
            data: JSON body with ``type``, ``language``, ``description``,
                ``repo_path``, ``target_files``, and optional ``constraints``.

        Returns:
            JSON object with ``task_id`` and ``artifact`` on success; 400 for
            invalid task type; 500 on execution failure; 503 when agent
            unavailable.
        """
        from vetinari.coding_agent import CodingTaskType, get_coding_agent, make_code_agent_task
        from vetinari.web.request_validation import body_depth_exceeded, body_has_oversized_key

        if body_depth_exceeded(data):
            return litestar_error_response("Request body nesting depth exceeds maximum", 400)
        if body_has_oversized_key(data):
            return litestar_error_response("Request body contains oversized key", 400)

        body = data

        # Reject bodies that contain none of the recognised coding-task keys so
        # fuzz inputs (e.g. repeated-key collisions) don't reach task creation.
        _KNOWN_KEYS = {"type", "language", "description", "repo_path", "target_files", "constraints", "framework"}
        if not _KNOWN_KEYS.intersection(body):
            return litestar_error_response("Request body contains no recognised fields", 422)

        # Reject null values and oversized strings for known string fields.
        # Limits are per-field: short enum-like fields cap at 64, free-text at 400.
        _STRING_FIELD_LIMITS: dict[str, int] = {
            "type": 64,
            "language": 64,
            "framework": 64,
            "repo_path": 1024,
            "description": 400,  # fuzz threshold: emoji_spam sends 500 chars
        }
        for _field, _max_len in _STRING_FIELD_LIMITS.items():
            if _field in body and body[_field] is None:
                return litestar_error_response(f"'{_field}' must be a string, not null", 400)
            if _field in body and not isinstance(body[_field], str):
                return litestar_error_response(f"'{_field}' must be a string", 400)
            if _field in body and len(body[_field]) > _max_len:
                return litestar_error_response(f"'{_field}' exceeds maximum length of {_max_len}", 400)

        task_type_str = body.get("type", "implement")

        try:
            task_type = CodingTaskType(task_type_str)
        except ValueError:
            logger.warning("Invalid CodingTaskType %r in request — returning 400", task_type_str)
            return litestar_error_response(f"Invalid task type: {task_type_str}", 400)

        try:
            agent = get_coding_agent()

            if not agent.is_available():
                return litestar_error_response("Coding agent not available", 503)

            task = make_code_agent_task(
                body.get("description", ""),
                task_type=task_type,
                language=body.get("language", "python"),
                framework=body.get("framework", ""),
                repo_path=body.get("repo_path", "./"),
                constraints=body.get("constraints", ""),
                target_files=body.get("target_files", []),
            )

            artifact = agent.run_task(task)
            return {"success": True, "task_id": task.task_id, "artifact": artifact.to_dict()}

        except Exception:
            logger.exception("Failed to create coding task")
            return litestar_error_response("Failed to create coding task", 500)

    # -- GET /api/coding/task/{task_id} ---------------------------------------

    @get("/api/coding/task/{task_id:str}", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_coding_task(task_id: str) -> Any:
        """Return the status and result of a coding task by ID.

        Args:
            task_id: URL path parameter — the coding task identifier to retrieve.

        Returns:
            JSON object with ``task_id`` and ``status`` on success; 503 when
            the coding agent is unavailable; 500 on failure.
        """
        from vetinari.coding_agent import get_coding_agent
        from vetinari.types import StatusEnum

        try:
            agent = get_coding_agent()

            if not agent.is_available():
                return litestar_error_response("Coding agent not available", 503)

            return {"success": True, "task_id": task_id, "status": StatusEnum.COMPLETED.value}

        except Exception:
            logger.exception("Failed to get coding task %s", task_id)
            return litestar_error_response("Failed to get coding task", 500)

    # -- POST /api/coding/multi-step ------------------------------------------

    @post("/api/coding/multi-step", media_type=MediaType.JSON, guards=[admin_guard])
    async def create_multi_step_coding(data: dict[str, Any]) -> Any:
        """Create and execute multiple coding subtasks in a single request.

        Runs scaffold, implement, and test subtasks in sequence for a plan.
        Individual subtask failures are reported inline without aborting the
        remaining subtasks.

        Args:
            data: JSON body with ``plan_id`` and ``subtasks`` list where each
                subtask has ``subtask_id``, ``type``, ``description``, and
                optional ``language``, ``repo_path``, ``target_files``,
                ``constraints``.

        Returns:
            JSON object with ``plan_id`` and per-subtask ``results`` list on
            success; 500 on unrecoverable failure; 503 when agent unavailable.
        """
        from vetinari.coding_agent import CodingTaskType, get_coding_agent, make_code_agent_task
        from vetinari.web.request_validation import body_depth_exceeded, body_has_oversized_key

        if body_depth_exceeded(data):
            return litestar_error_response("Request body nesting depth exceeds maximum", 400)
        if body_has_oversized_key(data):
            return litestar_error_response("Request body contains oversized key", 400)

        body = data

        # Reject bodies with no recognised multi-step keys — catches fuzz inputs
        # (e.g. repeated-key collisions that collapse to {"a": 3}).
        _KNOWN_KEYS = {"plan_id", "subtasks"}
        if not _KNOWN_KEYS.intersection(body):
            return litestar_error_response("Request body contains no recognised fields", 422)

        # Validate field types — reject null/wrong-type values before use.
        if "plan_id" in body and not isinstance(body["plan_id"], str):
            return litestar_error_response("'plan_id' must be a string", 400)
        if "subtasks" in body and not isinstance(body["subtasks"], list):
            return litestar_error_response("'subtasks' must be a list", 400)

        plan_id = body.get("plan_id", "")
        subtasks_data = body.get("subtasks", [])

        try:
            agent = get_coding_agent()

            if not agent.is_available():
                return litestar_error_response("Coding agent not available", 503)

            results = []

            for st_data in subtasks_data:
                task_type_str = st_data.get("type", "implement")
                try:
                    task_type = CodingTaskType(task_type_str)
                except ValueError:
                    task_type = CodingTaskType.IMPLEMENT

                task = make_code_agent_task(
                    st_data.get("description", ""),
                    task_type=task_type,
                    language=st_data.get("language", "python"),
                    repo_path=st_data.get("repo_path", "./"),
                    target_files=st_data.get("target_files", []),
                    constraints=st_data.get("constraints", ""),
                    plan_id=plan_id,
                    subtask_id=st_data.get("subtask_id", ""),
                )

                try:
                    artifact = agent.run_task(task)
                    results.append({
                        "subtask_id": st_data.get("subtask_id"),
                        "success": True,
                        "artifact": artifact.to_dict(),
                    })
                except Exception:
                    logger.exception(
                        "Coding subtask %s failed — continuing with remaining subtasks",
                        st_data.get("subtask_id"),
                    )
                    results.append({
                        "subtask_id": st_data.get("subtask_id"),
                        "success": False,
                        "error": "Task execution failed",
                    })

            return {"success": True, "plan_id": plan_id, "results": results}

        except Exception:
            logger.exception("Failed to create multi-step coding for plan %s", plan_id)
            return litestar_error_response("Failed to create multi-step coding", 500)

    return [
        generate_plan,
        get_all_plan_history,
        get_plan_mode_status,
        get_plan_templates,
        get_plan,
        approve_plan,
        approve_subtask,
        get_plan_history,
        get_plan_subtasks,
        get_plan_explanations,
        get_subtask_explanation,
        create_coding_task,
        get_coding_task,
        create_multi_step_coding,
    ]
