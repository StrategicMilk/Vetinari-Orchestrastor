"""Milestones approval API — native Litestar handlers for the milestone checkpoint system.

Exposes the milestone approval flow over HTTP so the dashboard and operators
can submit approval decisions (approve, revise, skip_remaining, abort) and
inspect milestone history during active plan execution.

This is step 4 of the pipeline: Intake → Planning → Execution → **Milestone Gate** → Assembly.

Endpoints
---------
    POST   /api/v1/milestones/approve   — submit an approval decision for the current milestone
    GET    /api/v1/milestones/status    — inspect milestone history and current policy
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# -- Module-level pending approval slot (thread-safe) -------------------------
#
# The execution engine polls has_pending_milestone_approval() / get_pending_milestone_approval()
# to pick up decisions submitted via the HTTP API.  A threading.Lock ensures the read-and-clear
# is atomic so the engine never processes the same approval twice.
#
# Who writes: submit_milestone_approval handler (below)
# Who reads:  vetinari.orchestration.graph_executor (via get_pending_milestone_approval)
# Lifecycle:  set on POST, cleared on first read by execution engine
# Lock:       _pending_lock guards both the read-and-clear and the write
_pending_approval: Any | None = None  # MilestoneApproval | None at runtime
_pending_lock = threading.Lock()

# Optional Litestar imports — graceful degradation when not installed
try:
    from litestar import MediaType, get, post

    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.responses import litestar_error_response

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def get_pending_milestone_approval() -> Any | None:
    """Atomically read and clear the pending milestone approval.

    Called by the execution engine to retrieve a user decision that was
    submitted via POST /api/v1/milestones/approve.  The slot is cleared on
    read so the same approval is never processed twice.

    Returns:
        The pending MilestoneApproval, or None if no decision is waiting.
    """
    global _pending_approval
    with _pending_lock:
        approval = _pending_approval
        _pending_approval = None
        return approval


def has_pending_milestone_approval() -> bool:
    """Return True when a milestone approval is waiting to be consumed.

    The execution engine uses this as a non-destructive check before calling
    get_pending_milestone_approval().

    Returns:
        True if a pending approval exists, False otherwise.
    """
    with _pending_lock:
        return _pending_approval is not None


def create_milestones_handlers() -> list[Any]:
    """Create all Litestar route handlers for the milestones approval API.

    Called by ``vetinari.web.litestar_app.create_app()`` to register these
    handlers in the main Litestar application.

    Returns:
        List of Litestar route handler functions, or empty list when
        Litestar is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        logger.debug("Litestar not available — milestones API handlers not registered")
        return []

    # -- POST /api/v1/milestones/approve --------------------------------------

    @post("/api/v1/milestones/approve", media_type=MediaType.JSON, guards=[admin_guard])
    async def submit_milestone_approval(data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Submit a user decision for the currently pending milestone checkpoint.

        Accepts an action and optional feedback string.  The decision is stored
        in a thread-safe module-level slot so the execution engine can pick it
        up on its next poll cycle without any direct coupling to the web layer.

        Args:
            data: JSON request body with ``action`` (required) and
                ``feedback`` (optional) fields.

        Returns:
            JSON with ``success=True`` and the accepted ``action`` string.

        Raises:
            400: When the body is missing, not a dict, missing the ``action``
                key, or ``action`` is not one of the valid MilestoneAction values.
            503: When the milestone approval subsystem (MilestoneAction/MilestoneApproval)
                cannot be imported.
        """
        global _pending_approval

        try:
            from vetinari.orchestration.milestones import MilestoneAction, MilestoneApproval
        except Exception:
            logger.warning(
                "Milestone approval subsystem unavailable — cannot process approval request",
                exc_info=True,
            )
            return litestar_error_response("Milestone approval subsystem unavailable", 503)  # type: ignore[return-value]

        if not isinstance(data, dict):
            return litestar_error_response("Request body must be a JSON object with an 'action' field")  # type: ignore[return-value]

        if "action" not in data:
            return litestar_error_response("Missing required field: 'action'")  # type: ignore[return-value]

        action_str = data["action"]
        valid_actions = {a.value for a in MilestoneAction}
        if action_str not in valid_actions:
            return litestar_error_response(  # type: ignore[return-value]
                f"Invalid action '{action_str}'. Must be one of: {', '.join(sorted(valid_actions))}"
            )

        feedback = data.get("feedback", "")
        if not isinstance(feedback, str):
            feedback = ""

        approval = MilestoneApproval(action=MilestoneAction(action_str), feedback=feedback)

        with _pending_lock:
            _pending_approval = approval

        # Also deliver directly to the MilestoneManager's event-wait path so
        # check_and_wait() unblocks immediately rather than waiting for the
        # executor's next poll cycle.  This is best-effort: if the executor
        # is not initialised, the _pending_approval slot above is sufficient
        # and check_and_wait() will pick it up on timeout → auto-approve.
        try:
            from vetinari.orchestration.graph_executor import get_graph_executor

            executor = get_graph_executor()
            milestone_manager = getattr(executor, "_milestone_manager", None)
            if milestone_manager is not None:
                milestone_manager.submit_approval(approval)
        except Exception:  # noqa: VET023 — best-effort direct delivery; pending slot is the primary path
            logger.debug("Milestone manager not available — approval stored in pending slot only")

        logger.info("Milestone approval submitted via API: action=%s", action_str)
        return {"success": True, "action": action_str}

    # -- GET /api/v1/milestones/status ----------------------------------------

    @get("/api/v1/milestones/status", media_type=MediaType.JSON)
    async def get_milestone_status() -> dict[str, Any]:
        """Return current milestone state: history and active policy.

        Queries the graph executor's MilestoneManager if one is active.
        Returns 503 when the executor or MilestoneManager is unavailable so
        callers can distinguish "subsystem down" from "no milestones active".

        Returns:
            JSON with ``has_active_milestone``, ``pending_approval``, and
            ``history`` fields when the milestone subsystem is available.

        Raises:
            503: When the graph executor or MilestoneManager cannot be reached.
        """
        has_pending = has_pending_milestone_approval()

        try:
            from vetinari.orchestration.graph_executor import get_graph_executor

            executor = get_graph_executor()
            milestone_manager = getattr(executor, "milestone_manager", None)
            if milestone_manager is None:
                # Executor found but carries no MilestoneManager — subsystem not
                # active for this execution context.
                logger.warning("Milestone manager not initialised on executor — cannot serve status")
                return litestar_error_response("Milestone manager unavailable", 503)  # type: ignore[return-value]
            history: list[dict[str, Any]] = milestone_manager.get_history()
            has_active = getattr(milestone_manager, "_pending", None) is not None
        except Exception:
            logger.warning("Milestone subsystem unavailable — cannot serve status", exc_info=True)
            return litestar_error_response("Milestone manager unavailable", 503)  # type: ignore[return-value]

        return {
            "has_active_milestone": has_active,
            "pending_approval": has_pending,
            "history": history,
        }

    return [submit_milestone_approval, get_milestone_status]
