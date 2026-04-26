"""Litestar SSE handler for real-time log streaming.

Replaces the Flask ``log_stream`` blueprint with async-native Litestar handlers.
This is the SSE delivery layer in the dashboard pipeline:

    Log record produced -> SSE backend queues -> **this handler streams** -> browser UI

Provides two endpoints:

    GET /api/logs/stream  -- SSE endpoint that pushes log records in real-time.
                            A keepalive comment is emitted every 30 s when idle.

    GET /api/logs/recent  -- Returns the most recent buffered log records as
                            a JSON array (useful for initial page load before
                            the SSE connection is established).

Side effects:
    - Module-level ``_sse_lock`` serialises connection-count updates.
    - ``_sse_connection_count`` is incremented on connect and decremented in the
      generator ``finally`` block, so cleanup is guaranteed even on abrupt close.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import AsyncGenerator
from typing import Any

from vetinari.constants import LOG_STREAM_TIMEOUT, SSE_LOG_STREAM_QUEUE_SIZE
from vetinari.dashboard.log_backends import get_sse_backend

logger = logging.getLogger(__name__)

# -- Optional Litestar imports -----------------------------------------------
try:
    from litestar import get
    from litestar.response import Response, ServerSentEvent

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False

# -- Connection limiter -------------------------------------------------------
# Hard cap prevents resource exhaustion from too many simultaneous SSE clients.
_MAX_SSE_CONNECTIONS = 50  # tune via load testing if needed
_MAX_SSE_IDLE_CYCLES = 120  # 120 keepalives x 30 s ~ 60 min before forced close
_sse_connection_count = 0
_sse_lock = threading.Lock()


def _decrement_sse_count() -> None:
    """Decrement the active SSE connection counter on stream close."""
    global _sse_connection_count
    with _sse_lock:
        _sse_connection_count -= 1


async def _queue_get_with_timeout(q: queue.Queue, timeout: float) -> str:
    """Poll a standard queue without blocking the event loop."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        try:
            return q.get_nowait()
        except queue.Empty:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise
            await asyncio.sleep(min(0.1, remaining))


async def _generate_log_events(q: queue.Queue) -> AsyncGenerator[dict[str, Any] | str, None]:
    """Yield SSE frames from the per-client queue until idle timeout or client disconnect.

    Polls the per-client queue without blocking the event loop. When the queue
    is empty for ``LOG_STREAM_TIMEOUT`` seconds a keepalive comment is emitted
    instead of a data frame.

    Args:
        q: Per-client queue populated by the SSE backend with JSON-encoded log
            record strings.

    Yields:
        Dicts with ``data`` key for log records, or ``comment`` key for
        keepalives — both formats are understood by Litestar's SSE encoder.
    """
    backend = get_sse_backend()
    idle_cycles = 0

    try:
        while True:
            try:
                data = await _queue_get_with_timeout(q, LOG_STREAM_TIMEOUT)
                yield {"data": data}
                idle_cycles = 0  # reset on any data
            except queue.Empty:
                idle_cycles += 1
                if idle_cycles >= _MAX_SSE_IDLE_CYCLES:
                    logger.info(
                        "SSE client idle for %d cycles (~%d min) — disconnecting",
                        idle_cycles,
                        idle_cycles * LOG_STREAM_TIMEOUT // 60,
                    )
                    break
                # Keepalive comment prevents proxy / browser connection timeout
                yield {"comment": "keepalive"}
    except GeneratorExit:  # noqa: VET022 — expected on client disconnect
        pass
    finally:
        backend.remove_client(q)
        _decrement_sse_count()


def create_log_stream_handlers() -> list[Any]:
    """Return Litestar route handlers for the log streaming endpoints.

    Builds and returns the two log-stream handlers when Litestar is available.
    Returns an empty list when Litestar is not installed so the caller can
    safely extend its handler list without a hard dependency on Litestar.

    Returns:
        List of Litestar route handler callables: ``[stream_logs, recent_logs]``,
        or an empty list when Litestar is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    @get("/api/logs/stream", sync_to_thread=False)
    def stream_logs() -> Response | ServerSentEvent:
        """SSE endpoint for real-time log streaming.

        Each log record is delivered as a ``data:`` frame containing a JSON
        object.  A keepalive comment is sent every 30 seconds when no logs
        are available.  Connections beyond the 50-client cap receive a 429.

        Returns:
            ``ServerSentEvent`` streaming response, or a 429 ``Response`` when
            the connection limit is reached, or a 503 when the SSE backend
            is unavailable.
        """
        from vetinari.web.responses import litestar_error_response

        global _sse_connection_count

        try:
            backend = get_sse_backend()
        except Exception as exc:
            logger.warning("SSE log backend unavailable — returning 503: %s", exc)
            return litestar_error_response("Log stream unavailable", 503)

        q: queue.Queue = queue.Queue(maxsize=SSE_LOG_STREAM_QUEUE_SIZE)

        with _sse_lock:
            if _sse_connection_count >= _MAX_SSE_CONNECTIONS:
                logger.warning(
                    "SSE connection limit (%d) reached — rejecting new client",
                    _MAX_SSE_CONNECTIONS,
                )
                return Response(
                    content={"error": "Too many SSE connections"},
                    status_code=429,
                    headers={"Cache-Control": "no-cache"},
                )
            _sse_connection_count += 1

        try:
            backend.add_client(q)
        except Exception as exc:
            with _sse_lock:
                _sse_connection_count -= 1
            logger.warning("Could not register SSE log client — returning 503: %s", exc)
            return litestar_error_response("Log stream registration failed", 503)

        return ServerSentEvent(
            _generate_log_events(q),
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @get("/api/logs/recent", sync_to_thread=False)
    def recent_logs() -> dict[str, Any] | Response:
        """Return the most recent buffered log records as JSON.

        Fetches up to 50 records from the SSE backend's ring buffer.  Intended
        for initial page load before the SSE connection is established.

        Returns:
            Dict with a ``logs`` key containing a list of JSON-encoded log
            record strings, or a 503 when the backend is unavailable.
        """
        from vetinari.web.responses import litestar_error_response

        try:
            backend = get_sse_backend()
            records = backend.get_recent(limit=50)
        except Exception as exc:
            logger.warning("Could not load recent logs — returning 503: %s", exc)
            return litestar_error_response("Recent logs unavailable", 503)

        return {"logs": [r.to_json() for r in records]}

    return [stream_logs, recent_logs]
