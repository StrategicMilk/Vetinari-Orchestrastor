"""SSE streaming and project state-control Litestar handlers.

Native Litestar equivalents of the streaming and execution-control routes
previously registered via Flask blueprints in ``projects_streaming.py``
(ADR-0066).  URL paths are identical to the Flask originals.

This module owns the SSE connection lifecycle: queue creation, heartbeat
keepalive, and queue cleanup on client disconnect.  It also owns the
cancel/pause/resume controls and the event-replay endpoint.

Split boundary: non-streaming CRUD and operational routes live in
``litestar_projects_api.py``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Response, get, post
    from litestar.params import Parameter
    from litestar.response import Stream

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_projects_execution_handlers() -> list[Any]:
    """Create Litestar handlers for project streaming and execution-control routes.

    Covers SSE streaming (``/api/project/{id}/stream``), cancel/pause/resume
    controls, and the event-replay endpoint
    (``/api/v1/projects/{id}/events``).

    Returns:
        List of Litestar route handler objects, or empty list when Litestar
        is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard

    # -- SSE stream -----------------------------------------------------------

    @get("/api/project/{project_id:str}/stream", media_type="text/event-stream")
    async def api_project_stream(project_id: str) -> Stream:
        """Subscribe to real-time SSE events for a project.

        Opens a persistent HTTP connection and yields server-sent events as
        the project's background thread emits status changes, task
        completions, and errors.  Sends a heartbeat every
        ``SSE_MESSAGE_TIMEOUT`` seconds when idle.  Always cleans up the
        SSE queue in a ``finally`` block to prevent queue leaks on client
        disconnect.

        Args:
            project_id: The project to subscribe to.

        Returns:
            A ``Stream`` response with ``text/event-stream`` media type.
        """
        import json as _json
        import queue as _queue
        from collections.abc import AsyncGenerator

        from vetinari.constants import SSE_MESSAGE_TIMEOUT
        from vetinari.web.shared import (
            _cleanup_project_state,
            _get_sse_queue,
            validate_path_param,
        )

        if not validate_path_param(project_id):
            # Return an error stream event rather than crashing the generator
            async def _error_stream() -> AsyncGenerator[bytes, None]:
                yield b'data: {"error": "Invalid project ID"}\n\n'

            return Stream(_error_stream(), media_type="text/event-stream")

        from vetinari.web import shared as _shared

        _project_dir = _shared.PROJECT_ROOT / "projects" / project_id
        if not _project_dir.exists():

            async def _not_found_stream() -> AsyncGenerator[bytes, None]:
                yield b'data: {"error": "Project not found"}\n\n'

            return Stream(_not_found_stream(), media_type="text/event-stream")

        async def event_generator() -> AsyncGenerator[bytes, None]:
            """Yield SSE-formatted bytes from the project event queue.

            Raises:
                asyncio.CancelledError: Propagates client disconnect cancellation.
            """
            import asyncio

            q = _get_sse_queue(project_id)
            connected_event = _json.dumps({"type": "connected", "project_id": project_id})
            yield f"data: {connected_event}\n\n".encode()
            try:
                while True:
                    try:
                        loop = asyncio.get_running_loop()
                        deadline = loop.time() + SSE_MESSAGE_TIMEOUT
                        while True:
                            try:
                                msg = q.get_nowait()
                                break
                            except _queue.Empty:
                                remaining = deadline - loop.time()
                                if remaining <= 0:
                                    raise
                                await asyncio.sleep(min(0.1, remaining))
                        if msg is None:
                            # Sentinel: execution finished, close the stream
                            done_event = _json.dumps({"type": "done"})
                            yield f"data: {done_event}\n\n".encode()
                            break
                        # Emit id: field so browsers honour Last-Event-ID for reconnect replay
                        event_id = msg.get("id", "")
                        if event_id:
                            yield f"id: {event_id}\nevent: {msg['event']}\ndata: {msg['data']}\n\n".encode()
                        else:
                            yield f"event: {msg['event']}\ndata: {msg['data']}\n\n".encode()
                    except _queue.Empty:
                        # Heartbeat to keep the TCP connection alive during idle
                        heartbeat = _json.dumps({"type": "heartbeat"})
                        yield f"data: {heartbeat}\n\n".encode()
            except GeneratorExit:
                # Client disconnected — cleanup runs in finally below.
                logger.warning(
                    "SSE client disconnected from project %s stream — cleaning up SSE queue",
                    project_id,
                )
            finally:
                # Always clean up queue and cancel flags on stream end, whether
                # the client disconnected, the sentinel was received, or an
                # unexpected error occurred.  Without this every completed or
                # disconnected project leaks a Queue in memory.
                _cleanup_project_state(project_id)
                logger.debug("SSE stream cleanup completed for project %s", project_id)

        return Stream(event_generator(), media_type="text/event-stream")

    # -- State controls -------------------------------------------------------

    @post("/api/project/{project_id:str}/cancel", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_cancel_project(project_id: str) -> Response:
        """Cancel a running project execution.

        Sets the project's cancel flag so the background execution loop exits
        after the current task, updates the project YAML status to
        ``"cancelled"``, emits a cancellation SSE event, and cleans up the
        project's SSE queue and cancel flag.

        Args:
            project_id: The project to cancel.

        Returns:
            JSON with ``status`` (``"cancelled"`` or ``"not_found"``) and
            ``project_id``.  HTTP 400 if the project ID is invalid.
        """
        import pathlib

        import yaml

        from vetinari.types import StatusEnum
        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import (
            _cancel_project_task,
            _cleanup_project_state,
            _push_sse_event,
            validate_path_param,
        )

        if not validate_path_param(project_id):
            return litestar_error_response("Invalid project ID", 400)

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        cancelled = _cancel_project_task(project_id)
        if not cancelled:
            # Project exists but is not currently running — no-op, return immediately
            return Response(
                content={"status": "not_running", "project_id": project_id},
                status_code=200,
                media_type=MediaType.JSON,
            )

        _push_sse_event(
            project_id,
            "cancelled",
            {
                "project_id": project_id,
                "status": StatusEnum.CANCELLED.value,
                "message": "Cancelled by user",
            },
        )

        # Persist the cancelled status so the UI shows it after reload
        try:
            config_path = project_dir / "project.yaml"
            if config_path.exists():
                with pathlib.Path(config_path).open(encoding="utf-8") as f:
                    project_config = yaml.safe_load(f) or {}
                project_config["status"] = "cancelled"
                with pathlib.Path(config_path).open("w", encoding="utf-8") as f:
                    yaml.dump(project_config, f, allow_unicode=True)
        except Exception:
            logger.warning(
                "Failed to update project %s config status to cancelled — file may be locked or missing",
                project_id,
                exc_info=True,
            )

        # Release SSE queues and cancel flags to prevent memory leaks
        _cleanup_project_state(project_id)

        return Response(
            content={"status": "cancelled" if cancelled else "not_found", "project_id": project_id},
            status_code=200,
            media_type=MediaType.JSON,
        )

    @post("/api/project/{project_id:str}/pause", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_pause_project(project_id: str) -> Response:
        """Pause a running project by writing a pause sentinel file.

        The execution loop checks for this file between tasks and enters a
        wait state until resumed.  Tasks already in progress finish before
        pausing.

        Args:
            project_id: The project to pause.

        Returns:
            JSON with ``status`` (``"paused"`` or ``"not_found"``) and
            ``project_id``.  HTTP 400 if the project ID is invalid.
        """
        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import _is_project_actually_running, _push_sse_event, validate_path_param

        if not validate_path_param(project_id):
            return litestar_error_response("Invalid project ID", 400)

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        # 409 Conflict when the project is idle — pausing a non-running project
        # would write a stale sentinel that blocks future execution silently.
        if not _is_project_actually_running(project_id):
            return litestar_error_response(
                f"Project {project_id!r} is not currently running — cannot pause an idle project",
                409,
            )

        pause_file = project_dir / ".paused"
        pause_file.write_text("paused", encoding="utf-8")
        logger.info("Project %s paused by user", project_id)

        _push_sse_event(project_id, "paused", {"project_id": project_id, "status": "paused"})

        return Response(
            content={"status": "paused", "project_id": project_id},
            status_code=200,
            media_type=MediaType.JSON,
        )

    @post("/api/project/{project_id:str}/resume", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_resume_project(project_id: str) -> Response:
        """Resume a paused project by removing the pause sentinel file.

        Args:
            project_id: The project to resume.

        Returns:
            JSON with ``status`` (``"resumed"`` or ``"not_paused"``) and
            ``project_id``.  HTTP 400 if the project ID is invalid.
        """
        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import _push_sse_event, validate_path_param

        if not validate_path_param(project_id):
            return litestar_error_response("Invalid project ID", 400)

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        pause_file = project_dir / ".paused"
        if pause_file.exists():
            pause_file.unlink()
            logger.info("Project %s resumed by user", project_id)
            _push_sse_event(project_id, "resumed", {"project_id": project_id, "status": "resumed"})
            return Response(
                content={"status": "resumed", "project_id": project_id},
                status_code=200,
                media_type=MediaType.JSON,
            )

        return Response(
            content={"status": "not_paused", "project_id": project_id},
            status_code=200,
            media_type=MediaType.JSON,
        )

    # -- Event replay ---------------------------------------------------------

    @get("/api/v1/projects/{project_id:str}/events", media_type=MediaType.JSON)
    async def api_project_events(
        project_id: str,
        limit: int | None = Parameter(query="limit", default=100),
        since: str | None = Parameter(query="since", default=None),
    ) -> Response:
        """Return persisted SSE events for a project, enabling client replay.

        Reads from the SSE event log written by ``_push_sse_event`` so the
        client can catch up on events missed while the stream was disconnected.
        Results are returned in ascending chronological order.

        Args:
            project_id: The project whose events to retrieve.
            limit: Maximum number of events to return (default 100).
            since: Optional ISO timestamp; only events after this time are
                returned.

        Returns:
            JSON with an ``events`` list and a ``count`` integer.
        """
        from datetime import datetime

        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import PROJECT_ROOT, validate_path_param
        from vetinari.web.sse_events import get_recent_sse_events

        safe_id = validate_path_param(project_id)
        if safe_id is None:
            return litestar_error_response("Invalid project ID", 400)

        project_dir = PROJECT_ROOT / "projects" / safe_id
        if not project_dir.exists():
            return litestar_error_response(f"Project {safe_id!r} not found", 404)

        safe_limit = limit if limit is not None else 100
        if safe_limit < 0:
            return litestar_error_response("'limit' must be a non-negative integer", 422)

        # Validate 'since' is a parseable ISO 8601 timestamp when provided
        if since is not None:
            try:
                datetime.fromisoformat(since.replace("Z", "+00:00"))
            except ValueError:
                logger.warning("Invalid ISO 8601 timestamp in 'since' parameter — %r, returning 422", since)
                return litestar_error_response(
                    f"'since' must be an ISO 8601 timestamp (e.g. '2024-01-01T00:00:00Z'), got: {since!r}",
                    422,
                )

        events = get_recent_sse_events(safe_id, limit=safe_limit, since=since)
        return Response(
            content={"events": events, "count": len(events)},
            status_code=200,
            media_type=MediaType.JSON,
        )

    return [
        api_project_stream,
        api_cancel_project,
        api_pause_project,
        api_resume_project,
        api_project_events,
    ]
