"""SSE parity tests for the native Litestar log stream handlers.

Verifies that ``vetinari.web.litestar_log_stream`` provides the same
contracts as the Flask ``log_stream`` blueprint it replaced:

- Factory returns exactly 2 handlers at the expected paths
- Neither handler carries admin_guard (public endpoints)
- ``_generate_log_events`` emits ``data`` frames, keepalive comments,
  honours the idle-cycle limit, and cleans up the backend queue on close
- ``stream_logs`` increments the connection counter and rejects new
  connections with 429 when the cap is reached
- ``recent_logs`` serialises records using ``.to_json()`` and passes
  ``limit=50`` to the backend
"""

from __future__ import annotations

import asyncio
import queue
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Skip the entire module when Litestar is not installed


# -- helpers -------------------------------------------------------------------


def _paths(handlers: list) -> set[str]:
    """Extract the declared route paths from a list of Litestar handlers.

    Args:
        handlers: List of Litestar route handler objects.

    Returns:
        Set of path strings, one per handler registration.
    """
    result: set[str] = set()
    for h in handlers:
        paths_attr = getattr(h, "paths", None)
        if paths_attr:
            result.update(paths_attr)
    return result


def _has_admin_guard(handler: object) -> bool:
    """Return True when admin_guard appears in a handler's guards list.

    Args:
        handler: A Litestar route handler object.

    Returns:
        True if admin_guard is in handler.guards, False otherwise.
    """
    from vetinari.web.litestar_guards import admin_guard

    guards = getattr(handler, "guards", None) or []
    return admin_guard in guards


def _handler_name(handler: object) -> str:
    """Return a human-readable name for a handler for assertion messages.

    Args:
        handler: A Litestar route handler object.

    Returns:
        The handler function name, or a repr fallback.
    """
    fn = getattr(handler, "fn", None)
    if fn is not None:
        return getattr(fn, "__name__", repr(handler))
    return repr(handler)


def _run_async(coro):
    """Run a coroutine in a fresh event loop and return the result.

    Uses ``asyncio.run()`` so this works on Python 3.10+ regardless of
    whether there is an existing loop in the current thread.

    Args:
        coro: An awaitable or coroutine object.

    Returns:
        The return value of the coroutine.
    """
    return asyncio.run(coro)


async def _collect_n(gen, n: int) -> list:
    """Consume exactly *n* items from an async generator and return them.

    Args:
        gen: An async generator to iterate.
        n: Number of items to collect.

    Returns:
        List of the first *n* items yielded by *gen*.
    """
    items = []
    async for item in gen:
        items.append(item)
        if len(items) >= n:
            break
    return items


# -- TestLogStreamHandlerFactory -----------------------------------------------


class TestLogStreamHandlerFactory:
    """Factory-level structural tests for create_log_stream_handlers()."""

    def test_handler_count(self) -> None:
        """Factory must return exactly 2 handlers (stream and recent)."""
        from vetinari.web.litestar_log_stream import create_log_stream_handlers

        handlers = create_log_stream_handlers()
        assert len(handlers) == 2, f"Expected 2 log-stream handlers, got {len(handlers)}"

    def test_expected_paths_present(self) -> None:
        """Both expected route paths must be registered on the returned handlers."""
        from vetinari.web.litestar_log_stream import create_log_stream_handlers

        handlers = create_log_stream_handlers()
        paths = _paths(handlers)
        expected = {"/api/logs/stream", "/api/logs/recent"}
        missing = expected - paths
        assert not missing, f"Log-stream routes missing: {missing}"

    def test_no_admin_guard(self) -> None:
        """Neither log-stream handler may carry admin_guard (public endpoints)."""
        from vetinari.web.litestar_log_stream import create_log_stream_handlers

        handlers = create_log_stream_handlers()
        for h in handlers:
            assert not _has_admin_guard(h), f"Unexpected admin_guard on public log-stream handler {_handler_name(h)}"

    def test_returns_empty_list_when_litestar_unavailable(self) -> None:
        """Factory must return [] when _LITESTAR_AVAILABLE is False.

        This ensures callers can safely extend their handler list even when
        Litestar is not installed in the environment.
        """
        import vetinari.web.litestar_log_stream as mod

        with patch.object(mod, "_LITESTAR_AVAILABLE", False):
            result = mod.create_log_stream_handlers()

        assert result == [], f"Expected [] when Litestar unavailable, got {result!r}"


# -- TestGenerateLogEvents -----------------------------------------------------


class TestGenerateLogEvents:
    """Unit tests for the _generate_log_events async generator."""

    @pytest.fixture(autouse=True)
    def _mock_backend(self) -> None:
        """Patch get_sse_backend so tests do not touch the real SSE backend."""
        self.backend = MagicMock()
        patcher = patch(
            "vetinari.web.litestar_log_stream.get_sse_backend",
            return_value=self.backend,
        )
        patcher.start()
        self.patcher = patcher
        yield
        patcher.stop()

    @pytest.fixture(autouse=True)
    def _reset_connection_count(self) -> None:
        """Reset the module-level SSE connection counter before and after each test."""
        import vetinari.web.litestar_log_stream as mod

        mod._sse_connection_count = 0
        yield
        mod._sse_connection_count = 0

    def test_yields_data_frame_from_queue(self) -> None:
        """Generator must yield {"data": <value>} when the queue has an item."""
        from vetinari.web.litestar_log_stream import _generate_log_events

        q: queue.Queue = queue.Queue()
        q.put('{"level":"INFO","message":"hello"}')

        items = _run_async(_collect_n(_generate_log_events(q), 1))

        assert len(items) == 1, "Expected exactly one item"
        assert items[0] == {"data": '{"level":"INFO","message":"hello"}'}, f"Unexpected data frame: {items[0]!r}"

    def test_yields_keepalive_on_empty_queue(self) -> None:
        """Generator must yield {"comment": "keepalive"} when queue is empty past timeout.

        The queue is left empty so the blocking get() times out. We patch the
        timeout to 0.01 s so the test completes in milliseconds.
        """
        from vetinari.web.litestar_log_stream import _generate_log_events

        q: queue.Queue = queue.Queue()

        with patch("vetinari.web.litestar_log_stream.LOG_STREAM_TIMEOUT", 0.01):
            items = _run_async(_collect_n(_generate_log_events(q), 1))

        assert len(items) == 1, "Expected exactly one keepalive"
        assert items[0] == {"comment": "keepalive"}, f"Unexpected keepalive frame: {items[0]!r}"

    def test_stops_after_max_idle_cycles(self) -> None:
        """Generator must exhaust after _MAX_SSE_IDLE_CYCLES consecutive timeouts.

        The idle-cycle limit is patched to 2.  The generator increments
        ``idle_cycles`` on each timeout and breaks when
        ``idle_cycles >= _MAX_SSE_IDLE_CYCLES``.  With limit=2 the sequence
        is: timeout -> idle_cycles=1 -> yield keepalive -> timeout ->
        idle_cycles=2 -> break (no second yield).  So exactly 1 keepalive
        frame is emitted before the generator stops.
        """
        from vetinari.web.litestar_log_stream import _generate_log_events

        q: queue.Queue = queue.Queue()

        async def _drain():
            items = []
            with (
                patch("vetinari.web.litestar_log_stream.LOG_STREAM_TIMEOUT", 0.01),
                patch("vetinari.web.litestar_log_stream._MAX_SSE_IDLE_CYCLES", 2),
            ):
                async for item in _generate_log_events(q):
                    items.append(item)
            return items

        items = _run_async(_drain())

        # With limit=2: first timeout yields 1 keepalive, second timeout hits
        # the >= check and breaks before yielding  -  so exactly 1 frame total.
        assert len(items) == 1, f"Expected 1 keepalive before idle disconnect (limit=2), got {len(items)}: {items}"
        assert items[0] == {"comment": "keepalive"}, f"Unexpected frame from idle generator: {items[0]!r}"

    def test_cleanup_on_generator_close(self) -> None:
        """Generator finally block must call remove_client and decrement the count.

        Closes the generator after one yield to trigger the finally block,
        then verifies both cleanup actions occurred.
        """
        import vetinari.web.litestar_log_stream as mod
        from vetinari.web.litestar_log_stream import _generate_log_events

        # Artificially raise the counter so we can confirm it decrements
        mod._sse_connection_count = 1
        q: queue.Queue = queue.Queue()
        q.put('{"msg":"cleanup-test"}')

        async def _start_then_close():
            gen = _generate_log_events(q)
            # Consume one item so the generator is running
            await gen.__anext__()
            # Close mid-stream  -  triggers GeneratorExit -> finally
            await gen.aclose()

        _run_async(_start_then_close())

        self.backend.remove_client.assert_called_once_with(q)
        assert mod._sse_connection_count == 0, f"Expected count=0 after cleanup, got {mod._sse_connection_count}"


# -- TestStreamLogsHandler -----------------------------------------------------


class TestStreamLogsHandler:
    """Unit tests for the stream_logs handler function."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Patch get_sse_backend and reset the connection counter before each test."""
        import vetinari.web.litestar_log_stream as mod

        self.mod = mod
        self.backend = MagicMock()
        mod._sse_connection_count = 0

        patcher = patch(
            "vetinari.web.litestar_log_stream.get_sse_backend",
            return_value=self.backend,
        )
        patcher.start()
        self.patcher = patcher
        yield
        patcher.stop()
        mod._sse_connection_count = 0

    def _get_stream_fn(self):
        """Return the unwrapped stream_logs handler function.

        Returns:
            The underlying callable from the Litestar-decorated stream_logs handler.
        """
        from vetinari.web.litestar_log_stream import create_log_stream_handlers

        handlers = create_log_stream_handlers()
        stream_handler = handlers[0]
        # Litestar stores the original function on .fn
        return stream_handler.fn

    def test_returns_server_sent_event_under_limit(self) -> None:
        """Handler must return a ServerSentEvent when below the connection cap."""
        from litestar.response import ServerSentEvent

        fn = self._get_stream_fn()
        result = fn()

        assert isinstance(result, ServerSentEvent), (
            f"Expected ServerSentEvent under the connection limit, got {type(result).__name__}"
        )

    def test_returns_429_at_connection_limit(self) -> None:
        """Handler must return a 429 Response when the connection cap is reached."""
        from litestar.response import Response

        self.mod._sse_connection_count = self.mod._MAX_SSE_CONNECTIONS
        fn = self._get_stream_fn()
        result = fn()

        assert isinstance(result, Response), f"Expected Response at connection limit, got {type(result).__name__}"
        assert result.status_code == 429, f"Expected status_code=429, got {result.status_code}"

    def test_increments_connection_count(self) -> None:
        """Accepting a new SSE connection must increment the module counter by 1."""
        assert self.mod._sse_connection_count == 0

        fn = self._get_stream_fn()
        fn()

        assert self.mod._sse_connection_count == 1, (
            f"Expected count=1 after one accepted connection, got {self.mod._sse_connection_count}"
        )

    def test_rejected_connection_does_not_increment_count(self) -> None:
        """A 429-rejected connection must not increment the counter.

        When the cap is already reached, turning the client away must leave
        the counter at exactly _MAX_SSE_CONNECTIONS, not above it.
        """
        self.mod._sse_connection_count = self.mod._MAX_SSE_CONNECTIONS
        fn = self._get_stream_fn()
        fn()

        assert self.mod._sse_connection_count == self.mod._MAX_SSE_CONNECTIONS, (
            f"Counter must not exceed cap after rejection, got {self.mod._sse_connection_count}"
        )


# -- TestRecentLogsHandler -----------------------------------------------------


class TestRecentLogsHandler:
    """Unit tests for the recent_logs handler function."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Patch get_sse_backend before each test."""
        self.backend = MagicMock()
        patcher = patch(
            "vetinari.web.litestar_log_stream.get_sse_backend",
            return_value=self.backend,
        )
        patcher.start()
        self.patcher = patcher
        yield
        patcher.stop()

    def _get_recent_fn(self):
        """Return the unwrapped recent_logs handler function.

        Returns:
            The underlying callable from the Litestar-decorated recent_logs handler.
        """
        from vetinari.web.litestar_log_stream import create_log_stream_handlers

        handlers = create_log_stream_handlers()
        recent_handler = handlers[1]
        return recent_handler.fn

    def test_returns_recent_records(self) -> None:
        """Handler must return {"logs": [<serialised records>]} with correct values."""
        record_a = SimpleNamespace(to_json=lambda: '{"message":"alpha","level":"INFO"}')
        record_b = SimpleNamespace(to_json=lambda: '{"message":"beta","level":"WARNING"}')
        self.backend.get_recent.return_value = [record_a, record_b]

        fn = self._get_recent_fn()
        result = fn()

        assert result == {
            "logs": [
                '{"message":"alpha","level":"INFO"}',
                '{"message":"beta","level":"WARNING"}',
            ]
        }, f"Unexpected recent_logs result: {result!r}"

    def test_calls_backend_with_limit_50(self) -> None:
        """Handler must request exactly 50 records from the backend ring buffer."""
        self.backend.get_recent.return_value = []

        fn = self._get_recent_fn()
        fn()

        self.backend.get_recent.assert_called_once_with(limit=50)

    def test_returns_empty_logs_when_backend_has_none(self) -> None:
        """Handler must return {"logs": []} when the backend buffer is empty.

        This covers the initial page load before any logs have been produced.
        """
        self.backend.get_recent.return_value = []

        fn = self._get_recent_fn()
        result = fn()

        assert result == {"logs": []}, f"Expected empty logs list, got {result!r}"
