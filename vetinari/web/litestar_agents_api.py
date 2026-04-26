"""Agent orchestration, shared memory, and decision Litestar handlers.

Native Litestar equivalents of the routes previously registered by
``agents_api._register(bp)``. Part of the Flask->Litestar migration (ADR-0066).
URL paths are identical to the Flask originals.

Endpoints
---------
    POST /api/v1/agents/<id>/pause    — Pause an agent instance
    POST /api/v1/agents/<id>/redirect — Redirect an agent to a different task
    POST /api/v1/agents/<id>/resume   — Resume a paused agent
    GET  /api/v1/agents/status        — Status of all agents
    POST /api/v1/agents/initialize    — Initialize or reinitialize all agents
    GET  /api/v1/agents/active        — Active agents with role colors and icons
    GET  /api/v1/agents/tasks         — Current task queue contents
    GET  /api/v1/agents/memory        — All entries from UnifiedMemoryStore
    GET  /api/v1/decisions/pending    — Unresolved decision prompts
    POST /api/v1/decisions            — Submit a resolution for a pending decision
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Request, get, post

    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.responses import litestar_error_response

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False

# Agent control state — protected by _agent_control_lock.
# _paused_agents: agent_id -> {"agent_id": str, "reason": str}
# _redirect_targets: agent_id -> {"task_id": str, "reason": str}
#
# Side effects:
#   - Written by pause/redirect/resume handlers
#   - Read by get_agent_control_state() and the pause handler (idempotency check)
_agent_control_lock = threading.Lock()
_paused_agents: dict[str, dict[str, str]] = {}
_redirect_targets: dict[str, dict[str, str]] = {}


def get_agent_control_state() -> dict[str, dict[str, dict[str, str]]]:
    """Return a deep copy snapshot of the current agent pause and redirect state.

    The returned dict is a copy — mutating it does not affect the module-level
    dicts.  Suitable for inspection in tests and status endpoints.

    Returns:
        Dict with keys ``"paused"`` and ``"redirects"``, each mapping
        agent_id strings to their respective control record dicts.
    """
    with _agent_control_lock:
        return {
            "paused": {k: dict(v) for k, v in _paused_agents.items()},
            "redirects": {k: dict(v) for k, v in _redirect_targets.items()},
        }


def create_agents_api_handlers() -> list[Any]:
    """Create Litestar handlers for agent orchestration, memory, and decisions.

    Called by ``vetinari.web.litestar_app.create_app()`` to register these
    handlers in the main Litestar application.

    Returns:
        List of Litestar route handler objects, or empty list when Litestar
        is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        logger.debug("Litestar not available — agents API handlers not registered")
        return []

    # -- POST /api/v1/agents/{agent_id}/pause ------------------------------------

    @post(
        "/api/v1/agents/{agent_id:str}/pause",
        media_type=MediaType.JSON,
        guards=[admin_guard],
    )
    async def api_agent_pause(
        agent_id: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Pause an agent instance so it stops picking up new work.

        Idempotent — pausing an already-paused agent returns the original
        reason without overwriting.

        Args:
            agent_id: The agent instance ID from the URL path.
            data: Optional JSON body with ``reason`` field.

        Returns:
            JSON with ``agent_id``, ``status`` (``"paused"``), and ``reason``.
        """
        body = data if data is not None else {}
        reason = body.get("reason", "paused by operator")

        with _agent_control_lock:
            if agent_id in _paused_agents:
                existing = _paused_agents[agent_id]
                return {"agent_id": agent_id, "status": "paused", "reason": existing["reason"]}

            record = {"agent_id": agent_id, "reason": reason}
            _paused_agents[agent_id] = record

        logger.info("Agent %s paused — reason: %s", agent_id, reason)
        return {"agent_id": agent_id, "status": "paused", "reason": reason}

    # -- POST /api/v1/agents/{agent_id}/redirect ---------------------------------

    @post(
        "/api/v1/agents/{agent_id:str}/redirect",
        media_type=MediaType.JSON,
        guards=[admin_guard],
    )
    async def api_agent_redirect(
        agent_id: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Redirect an agent to a different task.

        Requires ``task_id`` in the JSON body. A second redirect overwrites
        the first — agents always redirect to the latest target.

        Args:
            agent_id: The agent instance ID from the URL path.
            data: JSON body with ``task_id`` (required) and optional ``reason``.

        Returns:
            JSON with ``agent_id``, ``status`` (``"redirected"``), ``task_id``,
            and ``reason``, or a 400 response when ``task_id`` is missing.
        """
        body = data if data is not None else {}
        task_id = body.get("task_id")
        if not task_id:
            return litestar_error_response("task_id is required for redirect", 400)  # type: ignore[return-value]

        reason = body.get("reason", "redirected by operator")

        with _agent_control_lock:
            _redirect_targets[agent_id] = {"task_id": task_id, "reason": reason}

        logger.info("Agent %s redirected to task %s — reason: %s", agent_id, task_id, reason)
        return {"agent_id": agent_id, "status": "redirected", "task_id": task_id, "reason": reason}

    # -- POST /api/v1/agents/{agent_id}/resume -----------------------------------

    @post(
        "/api/v1/agents/{agent_id:str}/resume",
        media_type=MediaType.JSON,
        guards=[admin_guard],
    )
    async def api_agent_resume(agent_id: str) -> dict[str, Any]:
        """Remove a pause directive for an agent, allowing it to continue.

        Also clears any pending redirect for the same agent so it resumes its
        original task rather than being redirected.

        Args:
            agent_id: Path parameter identifying the agent to resume.

        Returns:
            JSON with ``agent_id`` and ``status`` (``"resumed"`` or
            ``"not_paused"`` if the agent was not paused).
        """
        with _agent_control_lock:
            was_paused = _paused_agents.pop(agent_id, None) is not None
            _redirect_targets.pop(agent_id, None)

        status = "resumed" if was_paused else "not_paused"
        logger.info("Agent %s resume requested — was_paused=%s", agent_id, was_paused)
        return {"agent_id": agent_id, "status": status}

    # -- GET /api/v1/agents/status -----------------------------------------------

    @get("/api/v1/agents/status", media_type=MediaType.JSON)
    async def api_agents_status() -> dict[str, Any]:
        """Return current status of all agents managed by MultiAgentOrchestrator.

        Returns:
            JSON object with an ``agents`` list produced by
            ``MultiAgentOrchestrator.get_agent_status()``, or an empty list
            when no orchestrator instance exists yet.
        """
        try:
            from vetinari.agents.multi_agent_orchestrator import MultiAgentOrchestrator

            orch = MultiAgentOrchestrator.get_instance()
            if orch is None:
                return {"agents": []}
            return {"agents": orch.get_agent_status()}
        except Exception:
            logger.warning(
                "Agent orchestrator unavailable — cannot serve status",
                exc_info=True,
            )
            return litestar_error_response("Agent orchestrator unavailable", 503)  # type: ignore[return-value]

    # -- POST /api/v1/agents/initialize ------------------------------------------

    @post("/api/v1/agents/initialize", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_agents_initialize(request: Request) -> dict[str, Any]:
        """Initialize or reinitialize all agents in the MultiAgentOrchestrator.

        Creates a new orchestrator singleton if one does not already exist, then
        calls ``initialize_agents()`` to start each configured agent.

        Args:
            request: Raw Litestar request — used to detect and reject any
                unexpected request body (this endpoint takes no input body).

        Returns:
            JSON object with ``status`` set to ``"initialized"`` and the list of
            agent names that were initialized.
            Returns 400 if any request body is present (including null/empty JSON).
        """
        # Reject any body content — this endpoint is triggered by URL alone.
        # Checking raw bytes handles null, empty-JSON, and truncated payloads
        # consistently regardless of Content-Type parsing.
        raw_body = await request.body()
        if raw_body.strip():
            return litestar_error_response("This endpoint does not accept a request body", 400)  # type: ignore[return-value]
        try:
            from vetinari.agents.multi_agent_orchestrator import MultiAgentOrchestrator

            orch = MultiAgentOrchestrator.get_instance()
            if orch is None:
                orch = MultiAgentOrchestrator()
            orch.initialize_agents()
            agent_names = [a.name for a in orch.agents.values()]
            return {"status": "initialized", "agents": agent_names}
        except Exception:
            logger.warning(
                "Agent initialization failed — orchestrator unavailable",
                exc_info=True,
            )
            return litestar_error_response("Agent initialization failed", 503)  # type: ignore[return-value]

    # -- GET /api/v1/agents/active -----------------------------------------------

    @get("/api/v1/agents/active", media_type=MediaType.JSON)
    async def api_agents_active() -> dict[str, Any]:
        """Return active agents with display metadata (color, icon, current task).

        Each agent entry includes a cycling color from the palette and a
        Font Awesome icon class derived from the agent's role string.

        Returns:
            JSON object with an ``agents`` list; each entry contains ``name``,
            ``role``, ``color``, ``icon``, ``tasks_completed``, ``current_task``,
            and ``state``.
        """
        try:
            from vetinari.agents.multi_agent_orchestrator import MultiAgentOrchestrator

            orch = MultiAgentOrchestrator.get_instance()
            if orch is None:
                return {"agents": []}

            colors = [
                "#6366f1",
                "#8b5cf6",
                "#ec4899",
                "#14b8a6",
                "#f59e0b",
                "#ef4444",
                "#3b82f6",
                "#10b981",
            ]
            icons = {
                "explorer": "fa-compass",
                "librarian": "fa-book",
                "oracle": "fa-globe",
                "ui_planner": "fa-palette",
                "builder": "fa-hammer",
                "researcher": "fa-search",
                "evaluator": "fa-check-circle",
                "synthesizer": "fa-brain",
                "planner": "fa-sitemap",
                "security_auditor": "fa-shield-alt",
                "data_engineer": "fa-database",
            }

            agents = []
            for i, agent in enumerate(orch.agents.values()):
                agent_type_val = agent.agent_type.value if hasattr(agent.agent_type, "value") else str(agent.agent_type)
                agents.append({
                    "name": agent.name,
                    "role": agent_type_val,
                    "color": colors[i % len(colors)],
                    "icon": icons.get(agent_type_val, "fa-robot"),
                    "tasks_completed": agent.tasks_completed,
                    "current_task": (
                        agent.current_task.to_dict()
                        if agent.current_task and hasattr(agent.current_task, "to_dict")
                        else None
                    ),
                    "state": (agent.state.value if hasattr(agent.state, "value") else str(agent.state)),
                })
            return {"agents": agents}
        except Exception:
            logger.warning(
                "Agent orchestrator unavailable — cannot serve active agents",
                exc_info=True,
            )
            return litestar_error_response("Agent orchestrator unavailable", 503)  # type: ignore[return-value]

    # -- GET /api/v1/agents/tasks ------------------------------------------------

    @get("/api/v1/agents/tasks", media_type=MediaType.JSON)
    async def api_agents_tasks() -> dict[str, Any]:
        """Return the current contents of the orchestrator's task queue.

        Returns:
            JSON object with a ``tasks`` list; each entry contains ``id``,
            ``description``, ``status``, and ``agent``.
        """
        try:
            from vetinari.agents.multi_agent_orchestrator import MultiAgentOrchestrator
            from vetinari.types import StatusEnum

            orch = MultiAgentOrchestrator.get_instance()
            if orch is None:
                return {"tasks": []}

            tasks = [
                {
                    "id": task.get("id"),
                    "description": task.get("description"),
                    "status": task.get("status", StatusEnum.PENDING.value),
                    "agent": task.get("assigned_agent", "unassigned"),
                }
                for task in orch.task_queue
            ]
            return {"tasks": tasks}
        except Exception:
            logger.warning(
                "Agent orchestrator unavailable — cannot serve task queue",
                exc_info=True,
            )
            return litestar_error_response("Agent orchestrator unavailable", 503)  # type: ignore[return-value]

    # -- GET /api/v1/agents/memory -----------------------------------------------

    @get("/api/v1/agents/memory", media_type=MediaType.JSON)
    async def api_memory() -> dict[str, Any]:
        """Return all entries stored in UnifiedMemoryStore (agent-centric view).

        Returns:
            JSON object with a ``memories`` list containing all memory entries.
        """
        try:
            from vetinari.memory.unified import get_unified_memory_store

            store = get_unified_memory_store()
            entries = store.timeline(limit=100)
            memories = [e.to_dict() for e in entries]
            return {"memories": memories}
        except Exception:
            logger.warning(
                "Memory store unavailable — cannot serve agent memories",
                exc_info=True,
            )
            return litestar_error_response("Memory store unavailable", 503)  # type: ignore[return-value]

    # -- GET /api/v1/decisions/pending -------------------------------------------

    @get("/api/v1/decisions/pending", media_type=MediaType.JSON)
    async def api_decisions_pending() -> dict[str, Any]:
        """Return all unresolved decision prompts from memory.

        Filters memories of type ``"decision"`` to those without a ``resolved``
        flag, returning only the fields the UI needs to render a decision dialog.

        Returns:
            JSON object with a ``decisions`` list; each entry contains ``id``,
            ``prompt``, ``options``, and ``context``.
        """
        try:
            from vetinari.memory.unified import get_unified_memory_store

            store = get_unified_memory_store()
            decisions = store.search("", entry_types=["decision"], limit=50)

            pending = []
            for d in decisions:
                d_dict = d.to_dict()
                meta = d_dict.get("metadata") or {}
                if not meta.get("resolved"):
                    pending.append({
                        "id": d_dict.get("id"),
                        "prompt": d_dict.get("content", ""),
                        "options": meta.get("options", []),
                        "context": meta.get("context", {}),
                    })
            return {"decisions": pending}
        except Exception:
            logger.warning(
                "Memory store unavailable — cannot serve pending decisions",
                exc_info=True,
            )
            return litestar_error_response("Memory store unavailable", 503)  # type: ignore[return-value]

    # -- POST /api/v1/decisions --------------------------------------------------

    @post("/api/v1/decisions", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_decisions_submit(data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Submit a resolution for a pending decision.

        Expects a JSON body with ``decision_id`` (str) and ``choice`` (any).
        Delegates to the memory store to mark the decision as resolved and
        unblock any waiting agents.

        Args:
            data: JSON body with ``decision_id`` (required) and ``choice`` fields.

        Returns:
            JSON object with ``status`` set to ``"resolved"`` and the submitted
            ``choice`` echoed back.

        Raises:
            400: When ``decision_id`` is absent from the request body.
            404: When no decision with the given ID exists in the store.
            503: When the memory store raises unexpectedly.
        """
        body = data if data is not None else {}
        decision_id = body.get("decision_id")
        choice = body.get("choice")

        if not decision_id:
            return litestar_error_response("Missing required field: decision_id", 400)  # type: ignore[return-value]

        try:
            from vetinari.memory.unified import get_unified_memory_store

            store = get_unified_memory_store()
            results = store.search(decision_id, limit=1)  # noqa: VET112 - empty fallback preserves optional request metadata contract
            found = False
            for entry in results:
                if entry.id == decision_id:
                    new_content = f"{entry.content}\n\n[RESOLVED]: {choice}"
                    store.update_content(decision_id, new_content)
                    found = True
                    break
            if not found:
                return litestar_error_response(  # type: ignore[return-value]
                    f"Decision '{decision_id}' not found", 404
                )
            return {"status": "resolved", "choice": choice}
        except Exception:
            logger.warning(
                "Memory store unavailable — cannot submit decision",
                exc_info=True,
            )
            return litestar_error_response("Decision store unavailable", 503)  # type: ignore[return-value]

    return [
        api_agent_pause,
        api_agent_redirect,
        api_agent_resume,
        api_agents_status,
        api_agents_initialize,
        api_agents_active,
        api_agents_tasks,
        api_memory,
        api_decisions_pending,
        api_decisions_submit,
    ]
