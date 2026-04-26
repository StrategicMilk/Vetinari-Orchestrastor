"""Audit trail Litestar handlers for context manifest history and task execution.

Native Litestar equivalents of the routes previously registered by
``audit_api._register(bp)``. Part of the Flask->Litestar migration (ADR-0066).
URL paths are identical to the Flask originals.

Endpoints
---------
    GET /api/v1/audit/manifests          — Query manifest audit trail with filtering
    GET /api/v1/audit/decisions          — Chronological decision log with filtering
    GET /api/v1/audit/tasks/{task_id}    — Full audit trail for a specific task
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, get

    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.responses import litestar_error_response

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_audit_api_handlers() -> list[Any]:
    """Create Litestar handlers for the audit trail API.

    Called by ``vetinari.web.litestar_app.create_app()`` to register these
    handlers in the main Litestar application.

    Returns:
        List of Litestar route handler objects, or empty list when Litestar
        is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        logger.debug("Litestar not available — audit API handlers not registered")
        return []

    # -- GET /api/v1/audit/manifests ---------------------------------------------

    @get("/api/v1/audit/manifests", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_audit_manifests(
        agent: str = "",
        mode: str = "",
        limit: int = 50,
    ) -> dict[str, Any]:
        """Query manifest audit trail with optional agent/mode filtering.

        Args:
            agent: Filter by agent type (e.g., ``"WORKER"``). Case-insensitive.
            mode: Filter by execution mode (e.g., ``"build"``).
            limit: Maximum number of results (default 50, max 200).

        Returns:
            JSON object with a ``manifests`` list of manifest audit records,
            or a 503 response when the memory store is unavailable.
        """
        try:
            from vetinari.memory.unified import get_unified_memory_store

            # Clamp limit to the same bounds as the Flask original
            clamped_limit = max(1, min(limit, 200))
            agent_filter = agent.upper()
            mode_filter = mode

            store = get_unified_memory_store()
            episodes = store.recall_episodes("", k=clamped_limit)
            results = []
            for ep in episodes:
                ep_dict = ep.to_dict() if hasattr(ep, "to_dict") else {}
                if agent_filter and ep_dict.get("agent_type", "") != agent_filter:
                    continue
                if mode_filter and ep_dict.get("task_type", "") != mode_filter:
                    continue
                results.append({
                    "episode_id": ep_dict.get("episode_id", ""),
                    "timestamp": ep_dict.get("timestamp", ""),
                    "agent_type": ep_dict.get("agent_type", ""),
                    "task_type": ep_dict.get("task_type", ""),
                    "manifest_hash": ep_dict.get("manifest_hash", ""),
                    "manifest_summary": ep_dict.get("manifest_summary", {}),
                    "quality_score": ep_dict.get("quality_score", 0.0),
                    "success": ep_dict.get("success", False),
                })
            return {"manifests": results[:clamped_limit]}
        except Exception:
            logger.warning("Manifest audit store unavailable — cannot serve audit manifests, returning 503")
            return litestar_error_response(  # type: ignore[return-value]
                "Audit manifests subsystem unavailable", 503
            )

    # -- GET /api/v1/audit/decisions ---------------------------------------------

    @get("/api/v1/audit/decisions", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_audit_decisions(
        decision_type: str = "",
        limit: int = 50,
    ) -> dict[str, Any]:
        """Return the chronological decision log with optional filtering.

        Decisions are recorded by ``AuditLogger.log_decision()`` and capture
        what Vetinari chose during execution and why — for example, which model
        was selected, how a task was tier-classified, or why a retry was triggered.

        Args:
            decision_type: Filter by decision category (e.g., ``"model_selection"``,
                ``"tier_classification"``, ``"retry_action"``).
            limit: Maximum number of results (default 50, max 200).

        Returns:
            JSON object with a ``decisions`` list, each entry containing
            ``timestamp``, ``decision_type``, ``choice``, ``reasoning``,
            ``alternatives``, and ``context``.
            Returns 503 when the audit logger is unavailable.
        """
        try:
            from vetinari.audit import get_audit_logger

            clamped_limit = max(1, min(limit, 200))
            audit = get_audit_logger()
            decisions = audit.read_decisions(decision_type=decision_type, limit=clamped_limit)
            return {"decisions": decisions}
        except Exception:
            logger.warning("Audit logger unavailable — cannot serve audit decisions, returning 503")
            return litestar_error_response(  # type: ignore[return-value]
                "Audit decisions subsystem unavailable", 503
            )

    # -- GET /api/v1/audit/tasks/{task_id} ---------------------------------------

    @get("/api/v1/audit/tasks/{task_id:str}", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_audit_task(task_id: str) -> dict[str, Any]:
        """Retrieve the full audit trail for a specific task.

        Returns all memory entries and episodes linked to the given task ID,
        providing a complete history of what happened during task execution.

        Args:
            task_id: The task identifier to query.

        Returns:
            JSON object with ``task_id``, ``memories`` list, and ``episodes`` list.
            Returns 503 when the memory store is unavailable.
        """
        try:
            from vetinari.memory.unified import get_unified_memory_store

            store = get_unified_memory_store()
            memories = store.search(task_id, limit=50)
            memory_dicts = [m.to_dict() for m in memories]

            episodes = store.recall_episodes(task_id, k=20)
            episode_dicts = [ep.to_dict() if hasattr(ep, "to_dict") else {} for ep in episodes]

            return {
                "task_id": task_id,
                "memories": memory_dicts,
                "episodes": episode_dicts,
            }
        except Exception:
            logger.warning(
                "Memory store unavailable for task %s — cannot serve task audit trail, returning 503",
                task_id,
            )
            return litestar_error_response(  # type: ignore[return-value]
                "Audit task subsystem unavailable", 503
            )

    return [get_audit_manifests, get_audit_decisions, get_audit_task]
