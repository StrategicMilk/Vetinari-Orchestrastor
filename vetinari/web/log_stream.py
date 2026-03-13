"""Flask Blueprint for SSE (Server-Sent Events) log streaming.

Provides two endpoints:

    GET /api/logs/stream   -- An SSE endpoint that pushes log records in
                              real-time as ``data: <json>`` frames.  A
                              keepalive comment is emitted every 30 s to
                              prevent proxies from closing the connection.

    GET /api/logs/recent   -- Returns the most recent buffered log records
                              as a JSON array (useful for initial page load
                              before the SSE connection is established).

Usage::

    from vetinari.web.log_stream import log_stream_bp
    app.register_blueprint(log_stream_bp)
"""

from __future__ import annotations

import logging
import queue

from flask import Blueprint, Response, jsonify

from vetinari.dashboard.log_aggregator import get_sse_backend

logger = logging.getLogger(__name__)

log_stream_bp = Blueprint("log_stream", __name__)


@log_stream_bp.route("/api/logs/stream")
def stream_logs():
    """SSE endpoint for real-time log streaming.

    Each log record is delivered as a ``data:`` frame containing a JSON
    object.  A keepalive comment (``:``) is sent every 30 seconds when no
    logs are available.

    The response uses ``text/event-stream`` MIME type and disables all
    caching / buffering so that intermediary proxies forward events
    immediately.
    """
    backend = get_sse_backend()
    q: queue.Queue = queue.Queue(maxsize=1000)
    backend.add_client(q)

    def generate():
        try:
            while True:
                try:
                    data = q.get(timeout=30)
                    yield f"data: {data}\n\n"
                except queue.Empty:
                    # Keepalive comment to prevent proxy / browser timeout
                    yield ": keepalive\n\n"
        except GeneratorExit:  # noqa: VET022
            pass
        finally:
            backend.remove_client(q)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@log_stream_bp.route("/api/logs/recent")
def recent_logs():
    """Return the most recent buffered log records as JSON.

    Response format::

        { "logs": ["<json-string>", ...] }
    """
    backend = get_sse_backend()
    return jsonify({"logs": backend.get_buffer()})
