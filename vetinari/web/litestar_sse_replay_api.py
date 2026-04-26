"""SSE event replay API — enables reconnecting clients to recover missed events.

Clients that disconnect mid-stream can replay events from the sse_event_log
SQLite table by calling GET /api/v1/projects/{project_id}/events/replay.
This is step 2 of the SSE delivery pipeline:
  Live Queue (shared.py) -> Persist (sse_event_log) -> **Replay (this module)**.

Endpoints
---------
    GET /api/v1/projects/{project_id}/events/replay
        Return persisted SSE events in chronological order.  Supports
        ``after_sequence`` to fetch only events missed since the client's
        last ``Last-Event-ID``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import Controller, MediaType, get
    from litestar.exceptions import ValidationException
    from litestar.params import Parameter

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_sse_replay_handlers() -> list[Any]:
    """Create Litestar handlers for the SSE event replay endpoint.

    Called by ``vetinari.web.litestar_app.create_app()`` to register the
    replay handler in the main Litestar application.

    Returns:
        List containing the ``SSEReplayController`` class, or an empty list
        when Litestar is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        logger.debug("Litestar not available — SSE replay handler not registered")
        return []

    return [SSEReplayController]


# ---------------------------------------------------------------------------
# Controller — defined at module level so Litestar's metaclass can resolve it
# ---------------------------------------------------------------------------


if _LITESTAR_AVAILABLE:

    class SSEReplayController(Controller):
        """Litestar controller for SSE event replay endpoints.

        Mounted at ``/api/v1/projects/{project_id}/events`` so that the
        replay endpoint is a sub-path of the project's event namespace.
        """

        path = "/api/v1/projects/{project_id:str}/events"
        tags = ["sse"]

        @get("/replay", media_type=MediaType.JSON)
        async def replay_events(
            self,
            project_id: str,
            after_sequence: int | None = Parameter(default=None, ge=0),
            limit: int = Parameter(default=100, ge=1, le=1000),
        ) -> Any:
            """Replay persisted SSE events for a project from the sse_event_log.

            Clients that reconnect after a disconnect can use ``after_sequence``
            to retrieve only the events they missed since their last received
            ``Last-Event-ID``.  Events are returned in chronological order
            (ascending by ``id``).

            Args:
                project_id: Project identifier (alphanumeric, hyphens, underscores
                    only — validated against path-traversal patterns).
                after_sequence: Only return events with ``sequence_num`` strictly
                    greater than this value.  Omit to return all available events
                    up to ``limit``.
                limit: Maximum number of events to return.  Clamped to 1-1000,
                    default 100.

            Returns:
                JSON array of event dicts in chronological order.  Each dict has
                keys: ``id`` (int), ``event_type`` (str), ``sequence`` (int),
                ``data`` (parsed JSON dict), ``emitted_at`` (str ISO timestamp).

            Raises:
                ValidationException: If ``project_id`` contains unsafe characters
                    (path traversal, injection sequences, etc.).
            """
            from vetinari.database import get_connection
            from vetinari.web.responses import litestar_error_response
            from vetinari.web.shared import validate_path_param

            safe_id = validate_path_param(project_id)
            if safe_id is None:
                raise ValidationException("Invalid project_id")

            try:
                conn = get_connection()
                if after_sequence is not None:
                    rows = conn.execute(
                        "SELECT id, event_type, payload_json, sequence_num, emitted_at"
                        " FROM sse_event_log"
                        " WHERE project_id = ? AND sequence_num > ?"
                        " ORDER BY id ASC LIMIT ?",
                        (safe_id, after_sequence, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT id, event_type, payload_json, sequence_num, emitted_at"
                        " FROM sse_event_log"
                        " WHERE project_id = ?"
                        " ORDER BY id ASC LIMIT ?",
                        (safe_id, limit),
                    ).fetchall()
            except ValidationException:
                raise
            except Exception:
                logger.warning("Database unavailable for SSE replay — project_id=%s", project_id)
                return litestar_error_response("SSE replay subsystem unavailable", 503)

            results: list[dict[str, Any]] = []
            for row in rows:
                try:
                    data = json.loads(row["payload_json"])
                except (json.JSONDecodeError, TypeError):
                    # Preserve undecodable payloads verbatim so callers can inspect them
                    data = {"_raw": row["payload_json"]}
                results.append({
                    "id": row["id"],
                    "event_type": row["event_type"],
                    "sequence": row["sequence_num"],
                    "data": data,
                    "emitted_at": row["emitted_at"],
                })
            return results

else:
    # Placeholder so the module-level name always exists, even without Litestar
    SSEReplayController = None  # type: ignore[assignment,misc]
