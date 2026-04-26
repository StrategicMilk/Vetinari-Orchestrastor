"""Tests for vetinari.shutdown — graceful shutdown handlers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.shutdown import (
    register_callback,
    register_shutdown_handlers,
    reset,
    shutdown,
)


class TestShutdown:
    """Test shutdown callback execution."""

    def setup_method(self) -> None:
        """Reset shutdown state before each test."""
        reset()

    def test_shutdown_executes_callbacks(self) -> None:
        """shutdown() invokes all registered callbacks."""
        events: list[str] = []
        register_callback("test-callback", lambda: events.append("test-callback"))
        shutdown()
        assert events == ["test-callback"]

    def test_shutdown_executes_in_lifo_order(self) -> None:
        """Callbacks execute in LIFO order (last registered first)."""
        call_order: list[str] = []
        register_callback("first", lambda: call_order.append("first"))
        register_callback("second", lambda: call_order.append("second"))
        register_callback("third", lambda: call_order.append("third"))
        shutdown()
        assert call_order == ["third", "second", "first"]

    def test_shutdown_is_idempotent(self) -> None:
        """Calling shutdown() multiple times only executes callbacks once."""
        events: list[str] = []
        register_callback("test", lambda: events.append("test"))
        shutdown()
        shutdown()  # Second call should be a no-op
        assert events == ["test"]

    def test_shutdown_continues_on_callback_error(self) -> None:
        """If one callback raises, remaining callbacks still run."""
        events: list[str] = []

        def _failing() -> None:
            events.append("failing")
            raise RuntimeError("boom")

        register_callback("after", lambda: events.append("after"))
        register_callback("failing", _failing)
        shutdown()
        # "after" was registered first, runs second (LIFO), but "failing" error
        # should not prevent "after" from running
        assert events == ["failing", "after"]


class TestRegisterCallback:
    """Test callback registration."""

    def setup_method(self) -> None:
        reset()

    def test_register_callback_stores_callback(self) -> None:
        """Registered callback is invoked on shutdown."""
        called = []
        register_callback("test", lambda: called.append(True))
        shutdown()
        assert called == [True]

    def test_multiple_callbacks_all_run(self) -> None:
        """All registered callbacks are executed."""
        results: list[str] = []
        register_callback("a", lambda: results.append("a"))
        register_callback("b", lambda: results.append("b"))
        shutdown()
        assert set(results) == {"a", "b"}

    def test_register_callback_replaces_duplicate_name(self) -> None:
        """Registering the same callback name twice does not duplicate shutdown work."""
        first = MagicMock()
        second = MagicMock()
        register_callback("same-name", first)
        register_callback("same-name", second)
        shutdown()
        first.assert_not_called()
        second.assert_called_once()


class TestRegisterShutdownHandlers:
    """Test full handler registration."""

    def setup_method(self) -> None:
        reset()

    @patch("vetinari.shutdown.signal")  # noqa: VET242 - test patches process hooks to isolate shutdown side effects
    @patch("vetinari.shutdown.atexit")  # noqa: VET242 - test patches process hooks to isolate shutdown side effects
    def test_register_shutdown_handlers_registers_atexit(self, mock_atexit: MagicMock, mock_signal: MagicMock) -> None:
        """register_shutdown_handlers() registers an atexit callback."""
        register_shutdown_handlers()
        mock_atexit.register.assert_called_once_with(shutdown)

    def test_register_shutdown_handlers_registers_http_cleanup(self) -> None:
        """register_shutdown_handlers() registers HTTP session cleanup."""
        register_shutdown_handlers()
        # Verify HTTP cleanup is registered by checking callbacks run close_all
        from vetinari.http import close_all, create_session, session_count

        create_session()
        assert session_count() == 1
        shutdown()
        assert session_count() == 0

    @patch("vetinari.shutdown.signal")  # noqa: VET242 - test patches process hooks to isolate shutdown side effects
    @patch("vetinari.shutdown.atexit")  # noqa: VET242 - test patches process hooks to isolate shutdown side effects
    def test_register_shutdown_handlers_is_idempotent(
        self, mock_atexit: MagicMock, mock_signal: MagicMock
    ) -> None:
        """Repeated registration must not duplicate atexit or cleanup callbacks."""
        with patch("vetinari.http.close_all") as mock_close_all:
            register_shutdown_handlers()
            register_shutdown_handlers()
            shutdown()

        mock_atexit.register.assert_called_once_with(shutdown)
        mock_close_all.assert_called_once()

    def test_main_thread_call_installs_signals_after_background_registration(self) -> None:
        """A first off-main call must not permanently suppress signal setup."""
        import vetinari.shutdown as shutdown_mod

        background_thread = object()
        main_thread = object()
        shutdown_mod.reset()

        with (
            patch.object(shutdown_mod, "atexit") as mock_atexit,
            patch.object(shutdown_mod, "signal") as mock_signal,
        ):
            with (
                patch.object(shutdown_mod.threading, "main_thread", return_value=main_thread),
                patch.object(shutdown_mod.threading, "current_thread", return_value=background_thread),
            ):
                shutdown_mod.register_shutdown_handlers()

            assert mock_signal.signal.call_count == 0

            with (
                patch.object(shutdown_mod.threading, "main_thread", return_value=main_thread),
                patch.object(shutdown_mod.threading, "current_thread", return_value=main_thread),
            ):
                shutdown_mod.register_shutdown_handlers()

            mock_atexit.register.assert_called_once_with(shutdown_mod.shutdown)
            assert mock_signal.signal.call_count == 2


class TestReset:
    """Test state reset for testing."""

    def setup_method(self) -> None:
        reset()

    def test_reset_clears_shutdown_state(self) -> None:
        """reset() allows shutdown to run again after previous completion."""
        calls: list[str] = []
        register_callback("first", lambda: calls.append("first"))
        shutdown()
        assert calls == ["first"]

        # After reset, shutdown should be callable again with new callbacks
        reset()
        register_callback("second", lambda: calls.append("second"))
        shutdown()
        assert calls == ["first", "second"]
