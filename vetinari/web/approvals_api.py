"""Approvals and trust API — native Litestar handlers for the autonomy subsystem.

Exposes the approval queue and governor trust engine over HTTP so the dashboard
and operators can review pending actions, approve or reject them, and inspect
the full decision audit log.

Endpoints
---------
    GET    /api/v1/approvals/pending                 — pending actions awaiting human decision
    POST   /api/v1/approvals/{action_id}/approve     — approve a pending action
    POST   /api/v1/approvals/{action_id}/reject      — reject a pending action with a reason
    GET    /api/v1/decisions/log                     — decision audit log with optional filters
    GET    /api/v1/decisions/history                 — unified decision history from all stores
    GET    /api/v1/autonomy/trust-status             — per-action-type trust metrics
    GET    /api/v1/autonomy/promotions               — actions eligible for level promotion
    POST   /api/v1/autonomy/promote/{action_type}    — apply a human-confirmed promotion
    POST   /api/v1/autonomy/veto/{action_type}       — veto promotion for an action type
    DELETE /api/v1/autonomy/veto/{action_type}       — clear a promotion veto
    GET    /api/v1/autonomy/vetoes                   — list currently vetoed action types
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Optional Litestar imports — graceful degradation when not installed
try:
    from litestar import MediaType, delete, get, post

    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.responses import litestar_error_response

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_approvals_handlers() -> list[Any]:
    """Create all Litestar route handlers for the approvals and trust API.

    Called by ``vetinari.web.litestar_app.create_app()`` to register these
    handlers in the main Litestar application.

    Returns:
        List of Litestar route handler functions, or empty list when
        Litestar is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        logger.debug("Litestar not available — approvals API handlers not registered")
        return []

    # -- GET /api/v1/approvals/pending ----------------------------------------

    @get("/api/v1/approvals/pending", media_type=MediaType.JSON, guards=[admin_guard])
    async def list_pending_approvals() -> list[dict[str, Any]]:
        """Return all pending approval requests awaiting human decision.

        Expired entries are automatically pruned before the result is returned.
        Each entry includes action_id, action_type, details, confidence,
        status, and created_at.

        Returns:
            JSON array of pending action objects.
        """
        from vetinari.autonomy.approval_queue import get_approval_queue

        pending = get_approval_queue().get_pending()
        return [
            {
                "action_id": p.action_id,
                "action_type": p.action_type,
                "details": p.details,
                "confidence": p.confidence,
                "status": p.status,
                "created_at": p.created_at,
            }
            for p in pending
        ]

    # -- POST /api/v1/approvals/{action_id}/approve ---------------------------

    @post("/api/v1/approvals/{action_id:str}/approve", media_type=MediaType.JSON, guards=[admin_guard])
    async def approve_action(action_id: str) -> dict[str, Any]:
        """Approve a pending action so it can proceed autonomously.

        Marks the action as approved in the approval queue. Returns success
        False (with HTTP 404) when the action is not found or not in pending
        status.

        Args:
            action_id: URL path parameter — the action to approve.

        Returns:
            JSON with success flag and action_id.
        """
        from vetinari.autonomy.approval_queue import get_approval_queue

        success = get_approval_queue().approve(action_id)
        if not success:
            return litestar_error_response(  # type: ignore[return-value]
                f"Action {action_id!r} not found or not in pending status",
                code=404,
            )
        logger.info("Action %s approved via API", action_id)
        return {"success": True, "action_id": action_id}

    # -- POST /api/v1/approvals/{action_id}/reject ----------------------------

    @post("/api/v1/approvals/{action_id:str}/reject", media_type=MediaType.JSON, guards=[admin_guard])
    async def reject_action(action_id: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reject a pending action with an optional reason.

        Marks the action as rejected in the approval queue. The ``reason``
        field from the request body is stored in the decision log.

        Args:
            action_id: URL path parameter — the action to reject.
            data: Optional request body with a ``reason`` string field.

        Returns:
            JSON with success flag and action_id.
        """
        from vetinari.autonomy.approval_queue import get_approval_queue

        reason = ""
        if data and isinstance(data.get("reason"), str):
            reason = data["reason"]

        success = get_approval_queue().reject(action_id, reason=reason)
        if not success:
            return litestar_error_response(  # type: ignore[return-value]
                f"Action {action_id!r} not found or not in pending status",
                code=404,
            )
        logger.info("Action %s rejected via API (reason=%s)", action_id, reason or "none")
        return {"success": True, "action_id": action_id}

    # -- GET /api/v1/decisions/log --------------------------------------------

    @get("/api/v1/decisions/log", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_decision_log(
        action_type: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Return the decision audit log with optional filters.

        All autonomous decisions at all levels are recorded here, not just
        deferred actions. Use this to audit agent behaviour over time.

        Args:
            action_type: Optional filter — returns only entries for this action type.
            limit: Maximum number of entries to return (most recent first). Clamped to 1-500.

        Returns:
            JSON with entries list and total count.
        """
        limit = max(1, min(limit, 500))
        from vetinari.autonomy.approval_queue import get_approval_queue

        entries = get_approval_queue().get_decision_log(action_type=action_type, limit=limit)
        return {
            "total": len(entries),
            "entries": [
                {
                    "action_id": e.action_id,
                    "action_type": e.action_type,
                    "autonomy_level": e.autonomy_level,
                    "decision": e.decision,
                    "confidence": e.confidence,
                    "outcome": e.outcome,
                    "timestamp": e.timestamp,
                }
                for e in entries
            ],
        }

    # -- GET /api/v1/autonomy/trust-status ------------------------------------

    @get("/api/v1/autonomy/trust-status", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_trust_status() -> dict[str, Any]:
        """Return per-action-type trust metrics from the progressive trust engine.

        Reports success rates, consecutive failures, and promotion eligibility
        for every action type that has been seen by the governor.

        Returns:
            JSON with action-type-keyed trust metric dicts.
        """
        from vetinari.autonomy.governor import get_governor

        status = get_governor().get_trust_status()
        return {"trust_status": status, "action_count": len(status)}

    # -- GET /api/v1/autonomy/promotions --------------------------------------

    @get("/api/v1/autonomy/promotions", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_promotions() -> dict[str, Any]:
        """Return action types currently eligible for a level promotion.

        Promotion requires 95%+ success rate over at least 50 actions.
        These are suggestions — they require human confirmation via the
        POST /api/v1/autonomy/promote/{action_type} endpoint.

        Returns:
            JSON with list of promotion suggestion dicts.
        """
        from vetinari.autonomy.governor import get_governor

        suggestions = get_governor().suggest_promotions()
        return {
            "count": len(suggestions),
            "promotions": [
                {
                    "action_type": s.action_type,
                    "current_level": s.current_level.value,
                    "suggested_level": s.suggested_level.value,
                    "success_rate": s.success_rate,
                    "total_actions": s.total_actions,
                }
                for s in suggestions
            ],
        }

    # -- POST /api/v1/autonomy/promote/{action_type} --------------------------

    @post("/api/v1/autonomy/promote/{action_type:str}", media_type=MediaType.JSON, guards=[admin_guard])
    async def apply_promotion(action_type: str) -> dict[str, Any]:
        """Apply a human-confirmed promotion to a higher autonomy level.

        This is the human sign-off step for promotions surfaced by
        GET /api/v1/autonomy/promotions. The action type must currently meet
        promotion eligibility (95%+ success over 50+ actions).

        Args:
            action_type: URL path parameter — the action type to promote.

        Returns:
            JSON with success flag and the action_type.
        """
        from vetinari.autonomy.governor import get_governor

        applied = get_governor().apply_promotion(action_type)
        if not applied:
            return litestar_error_response(  # type: ignore[return-value]
                f"Action type {action_type!r} is not eligible for promotion or is already at the maximum level",
                code=422,
            )
        logger.info("Promotion applied for action_type=%s via API", action_type)
        return {"success": True, "action_type": action_type}

    # -- GET /api/v1/decisions/history ----------------------------------------

    @get("/api/v1/decisions/history", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_decision_history(
        decision_type: str | None = None,
        confidence_level: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Return unified decision history from both autonomy and pipeline stores.

        Merges the approval queue's decision_log with the DecisionJournal,
        sorted by timestamp (most recent first). Each entry includes a ``source``
        field indicating where the decision was recorded: ``"autonomy"`` for the
        approval-queue decision_log, ``"pipeline"`` for the DecisionJournal.

        Args:
            decision_type: Optional filter by decision/action type.
            confidence_level: Optional filter by confidence level (e.g. ``"high"``).
            limit: Maximum entries to return across both sources combined.
                Clamped to 1-500.

        Returns:
            JSON with unified entries list and total count.
        """
        # Bound limit to prevent dumping unbounded history on negative or huge values
        limit = max(1, min(limit, 500))

        try:
            from vetinari.autonomy.approval_queue import get_approval_queue
            from vetinari.awareness.confidence import classify_confidence_score
            from vetinari.observability.decision_journal import get_decision_journal
            from vetinari.types import ConfidenceLevel as CL
            from vetinari.types import DecisionType

            entries: list[dict[str, Any]] = []

            # -- Validate enum filters up-front; reject unknown values with 400 --
            dt_filter: DecisionType | None = None
            cl_filter: CL | None = None
            if decision_type is not None:
                try:
                    dt_filter = DecisionType(decision_type)
                except ValueError:
                    logger.warning("Invalid decision_type '%s' in approval filter — rejecting with 400", decision_type)
                    return litestar_error_response(  # type: ignore[return-value]
                        f"Unknown decision_type '{decision_type}'",
                        400,
                    )
            if confidence_level is not None:
                try:
                    cl_filter = CL(confidence_level)
                except ValueError:
                    logger.warning(
                        "Invalid confidence_level '%s' in approval filter — rejecting with 400", confidence_level
                    )
                    return litestar_error_response(  # type: ignore[return-value]
                        f"Unknown confidence_level '{confidence_level}'",
                        400,
                    )

            # -- Autonomy decisions (approval queue decision_log) --
            aq = get_approval_queue()
            aq_entries = aq.get_decision_log(action_type=decision_type, limit=limit)
            for e in aq_entries:
                classified_level = classify_confidence_score(e.confidence)
                if cl_filter is not None and classified_level != cl_filter:
                    continue
                entries.append({
                    "id": e.action_id,
                    "type": e.action_type,
                    "source": "autonomy",
                    "decision": e.decision,
                    "confidence_score": e.confidence,
                    "confidence_level": classified_level.value,
                    "is_fallback": e.confidence == 0.0,
                    "outcome": e.outcome,
                    "autonomy_level": e.autonomy_level,
                    "timestamp": e.timestamp,
                })

            # -- Pipeline decisions (DecisionJournal) --
            journal = get_decision_journal()
            journal_entries = journal.get_decisions(
                decision_type=dt_filter,
                confidence_level=cl_filter,
                limit=limit,
            )
            entries.extend(
                {
                    "id": d.decision_id,
                    "type": d.decision_type.value,
                    "source": "pipeline",
                    "decision": d.action_taken,
                    "confidence_score": d.confidence_score,
                    "confidence_level": d.confidence_level.value,
                    "is_fallback": d.confidence_score == 0.0 and not d.confidence_factors,
                    "description": d.description,
                    "outcome": d.outcome,
                    "timestamp": d.timestamp,
                }
                for d in journal_entries
            )

            # Sort by timestamp descending, then apply limit across the merged set
            entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            entries = entries[:limit]

            return {"total": len(entries), "entries": entries}
        except Exception:
            logger.warning(
                "Decision history unavailable — approval queue or journal not accessible",
                exc_info=True,
            )
            return litestar_error_response("Decision history unavailable", 503)  # type: ignore[return-value]

    # -- POST /api/v1/autonomy/veto/{action_type} -----------------------------

    @post("/api/v1/autonomy/veto/{action_type:str}", media_type=MediaType.JSON, guards=[admin_guard])
    async def veto_promotion(action_type: str) -> dict[str, Any]:
        """Veto promotion for an action type, preventing automatic escalation.

        A vetoed action type will not appear in promotion suggestions and
        cannot be promoted until the veto is cleared via the DELETE endpoint.

        Args:
            action_type: URL path parameter — the action type to veto.
                Must contain only alphanumeric characters, underscores, and hyphens.

        Returns:
            JSON with success flag, action_type, and vetoed=True.
            Returns 400 when action_type contains unsafe characters.
        """
        import re as _re

        if not _re.fullmatch(r"[A-Za-z0-9_-]{1,128}", action_type):
            return litestar_error_response(
                "action_type must contain only letters, digits, underscores, or hyphens (max 128 chars)",
                code=400,
            )

        from vetinari.autonomy.governor import get_governor

        get_governor().veto_promotion(action_type)
        logger.info("Promotion veto set for action_type=%s via API", action_type)
        return {"success": True, "action_type": action_type, "vetoed": True}

    # -- DELETE /api/v1/autonomy/veto/{action_type} ---------------------------

    @delete("/api/v1/autonomy/veto/{action_type:str}", status_code=200, media_type=MediaType.JSON, guards=[admin_guard])
    async def clear_veto(action_type: str) -> dict[str, Any]:
        """Clear a promotion veto for an action type.

        After clearing, the action type will again appear in promotion suggestions
        if it meets the eligibility criteria.

        Args:
            action_type: URL path parameter — the action type to un-veto.
                Must contain only alphanumeric characters, underscores, and hyphens.

        Returns:
            JSON with success flag and action_type, or 400 for unsafe characters,
            or 404 if no veto existed.
        """
        import re as _re

        if not _re.fullmatch(r"[A-Za-z0-9_-]{1,128}", action_type):
            return litestar_error_response(  # type: ignore[return-value]
                "action_type must contain only letters, digits, underscores, or hyphens (max 128 chars)",
                code=400,
            )

        from vetinari.autonomy.governor import get_governor

        cleared = get_governor().clear_veto(action_type)
        if not cleared:
            return litestar_error_response(  # type: ignore[return-value]
                f"No veto found for action_type {action_type!r}",
                code=404,
            )
        logger.info("Promotion veto cleared for action_type=%s via API", action_type)
        return {"success": True, "action_type": action_type, "vetoed": False}

    # -- GET /api/v1/autonomy/vetoes ------------------------------------------

    @get("/api/v1/autonomy/vetoes", media_type=MediaType.JSON, guards=[admin_guard])
    async def list_vetoes() -> dict[str, Any]:
        """Return the list of action types currently vetoed from promotion.

        Returns:
            JSON with count and sorted list of vetoed action type strings.
        """
        from vetinari.autonomy.governor import get_governor

        vetoed = get_governor().get_vetoed_actions()
        return {"count": len(vetoed), "vetoed_action_types": sorted(vetoed)}

    return [
        list_pending_approvals,
        approve_action,
        reject_action,
        get_decision_log,
        get_decision_history,
        get_trust_status,
        get_promotions,
        apply_promotion,
        veto_promotion,
        clear_veto,
        list_vetoes,
    ]
