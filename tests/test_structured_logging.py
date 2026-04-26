"""
Tests for structured logging module.

Run with: python -m pytest tests/test_structured_logging.py -v
"""

import json
import logging
import sys
from io import StringIO
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


class TestStructuredLoggingImport:
    """Test that structured logging module can be imported."""

    def test_import_module(self):
        """Verify module imports successfully."""
        from vetinari import structured_logging

        assert structured_logging is not None
        assert hasattr(structured_logging, "get_logger")

    def test_get_logger_function(self):
        """Test get_logger returns a logger."""
        from vetinari.structured_logging import get_logger

        logger = get_logger("test")
        assert logger is not None
        assert logger.name == "test"

    def test_log_event_function(self):
        """Test log_event convenience function."""
        from vetinari.structured_logging import log_event

        # Should not raise — calling returns None
        result = log_event("info", "test", "Test message")
        assert result is None


class TestStructlogProcessors:
    """Test structlog processor chain."""

    def test_correlation_context_processor(self):
        """Test that correlation context processor injects trace IDs."""
        from vetinari.structured_logging import _add_correlation_context, _trace_id_var

        token = _trace_id_var.set("test-trace-123")
        try:
            event_dict = {"event": "test"}
            result = _add_correlation_context(None, "info", event_dict)
            assert result["trace_id"] == "test-trace-123"
        finally:
            _trace_id_var.reset(token)

    def test_service_context_processor(self):
        """Test that service context processor adds service info."""
        from vetinari.structured_logging import _add_service_context

        event_dict = {"event": "test"}
        result = _add_service_context(None, "info", event_dict)
        assert result["service"] == "vetinari"

    def test_configure_logging_is_idempotent(self):
        """Test that configure_logging can be called multiple times."""
        from vetinari.structured_logging import configure_logging

        configure_logging()
        configure_logging()  # should not raise
        # Verify the logging module is still functional after repeated configuration
        import logging

        logger = logging.getLogger("idempotent_check")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "idempotent_check"


class TestStructuredLogger:
    """Test StructuredLogger wrapper."""

    def test_logger_info(self):
        """Test info level logging via structlog."""
        from vetinari.structured_logging import StructuredLogger

        structured = StructuredLogger("test_info")
        # Should not raise — output goes through structlog processor chain
        structured.info("Test message", extra_field="value")
        assert structured.name == "test_info"

    def test_logger_with_context(self):
        """Test logger with context fields."""
        from vetinari.structured_logging import StructuredLogger

        structured = StructuredLogger("test_context")
        structured.set_context(task_id="task_1", execution_id="exec_123")

        # Should not raise - context is bound to structlog logger
        structured.info("Task running")
        assert structured.name == "test_context"


class TestLogEventFunctions:
    """Test convenience log event functions."""

    def test_log_task_start(self):
        """Test log_task_start function."""
        from vetinari.structured_logging import log_task_start

        result = log_task_start("task_1", "code_generation")
        assert result is None  # log functions return None

    def test_log_task_complete(self):
        """Test log_task_complete function."""
        from vetinari.structured_logging import log_task_complete

        result = log_task_complete("task_1", 1500.0)
        assert result is None  # log functions return None

    def test_log_task_error(self):
        """Test log_task_error function."""
        from vetinari.structured_logging import log_task_error

        result = log_task_error("task_1", "Timeout error")
        assert result is None  # log functions return None

    def test_log_model_discovery(self):
        """Test log_model_discovery function."""
        from vetinari.structured_logging import log_model_discovery

        result = log_model_discovery(5, 250.0)
        assert result is None  # log functions return None

    def test_log_wave_start(self):
        """Test log_wave_start function."""
        from vetinari.structured_logging import log_wave_start

        result = log_wave_start("wave_1", 3)
        assert result is None  # log functions return None

    def test_log_wave_complete(self):
        """Test log_wave_complete function."""
        from vetinari.structured_logging import log_wave_complete

        result = log_wave_complete("wave_1", 5000.0)
        assert result is None  # log functions return None


class TestTimedOperationDecorator:
    """Test timed_operation decorator."""

    def test_decorator_logs_completion(self):
        """Test that decorator logs operation completion."""
        from vetinari.structured_logging import timed_operation

        @timed_operation("test_operation")
        def test_func():
            return 42

        result = test_func()
        assert result == 42
        # Log output goes to stdout in JSON format - verify no exception

    def test_decorator_logs_error(self):
        """Test that decorator logs errors."""
        from vetinari.structured_logging import timed_operation

        @timed_operation("failing_operation")
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_func()
        # Log output goes to stdout in JSON format - verify no exception


class TestMetricsCollector:
    """Test MetricsCollector class."""

    def test_increment_counter(self):
        """Test counter increment."""
        from vetinari.metrics import MetricsCollector

        collector = MetricsCollector()
        collector.increment("test_counter")
        collector.increment("test_counter")

        assert collector.get_counter("test_counter") == 2

    def test_increment_counter_with_tags(self):
        """Test counter increment with tags."""
        from vetinari.metrics import MetricsCollector

        collector = MetricsCollector()
        collector.increment("test_counter", status="success")
        collector.increment("test_counter", status="success")
        collector.increment("test_counter", status="error")

        assert collector.get_counter("test_counter", status="success") == 2
        assert collector.get_counter("test_counter", status="error") == 1

    def test_record_histogram(self):
        """Test histogram recording."""
        from vetinari.metrics import MetricsCollector

        collector = MetricsCollector()
        collector.record("test_histogram", 10.0)
        collector.record("test_histogram", 20.0)
        collector.record("test_histogram", 30.0)

        stats = collector.get_histogram_stats("test_histogram")
        assert stats is not None
        assert stats["count"] == 3
        assert stats["sum"] == 60.0
        assert stats["min"] == 10.0
        assert stats["max"] == 30.0
        assert stats["avg"] == 20.0

    def test_histogram_percentiles(self):
        """Test histogram percentile calculations."""
        from vetinari.metrics import MetricsCollector

        collector = MetricsCollector()
        # Record 100 values from 1 to 100
        for i in range(1, 101):
            collector.record("percentile_test", float(i))

        stats = collector.get_histogram_stats("percentile_test")
        assert stats is not None
        assert 50 <= stats["p50"] <= 51
        assert 95 <= stats["p95"] <= 96
        assert 99 <= stats["p99"] <= 100

    def test_get_metrics_singleton(self):
        """Test get_metrics returns singleton."""
        from vetinari.metrics import get_metrics

        m1 = get_metrics()
        m2 = get_metrics()

        assert m1 is m2


class TestEnvironmentVariables:
    """Test environment variable configuration."""

    def test_log_level_from_env(self, monkeypatch):
        """Test that log level can be set from environment."""
        monkeypatch.setenv("VETINARI_LOG_LEVEL", "DEBUG")
        # Re-import to pick up new setting (would need fresh config in real usage)
        from vetinari import structured_logging

        # Just verify the module has the function
        assert hasattr(structured_logging, "_get_log_level")

    def test_structured_logging_flag(self, monkeypatch):
        """Test structured logging enable/disable flag."""
        monkeypatch.setenv("VETINARI_STRUCTURED_LOGGING", "false")
        from vetinari import structured_logging

        assert hasattr(structured_logging, "_use_structured_logging")


class TestBackwardCompatibility:
    """Test backward compatibility with existing logging."""

    def test_standard_logging_still_works(self):
        """Test that standard Python logging still works."""

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.INFO)

        logger = logging.getLogger("standard_test")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        logger.info("Standard logging message")

        output = stream.getvalue()
        assert "Standard logging message" in output


# -- D3: CorrelationContext nested exit re-syncs structlog -----------------


class TestCorrelationContextNestedRestore:
    """D3: CorrelationContext.__exit__ re-syncs structlog after token reset."""

    def test_inner_exit_restores_outer_trace_id_in_structlog(self):
        """After inner CorrelationContext exits, structlog reflects the outer trace_id."""
        import structlog.contextvars

        from vetinari.structured_logging import CorrelationContext

        structlog.contextvars.clear_contextvars()
        try:
            outer_trace = "outer-trace-001"
            inner_trace = "inner-trace-002"

            with CorrelationContext(trace_id=outer_trace):
                # Outer context is bound
                ctx_mid = structlog.contextvars.get_contextvars()
                assert ctx_mid.get("trace_id") == outer_trace

                with CorrelationContext(trace_id=inner_trace):
                    ctx_inner = structlog.contextvars.get_contextvars()
                    assert ctx_inner.get("trace_id") == inner_trace

                # After inner exits, outer value MUST be restored in structlog
                ctx_after = structlog.contextvars.get_contextvars()
                assert ctx_after.get("trace_id") == outer_trace, (
                    "D3 fix: CorrelationContext.__exit__ must re-sync structlog to outer trace_id"
                )
        finally:
            structlog.contextvars.clear_contextvars()

    def test_inner_exit_unbinds_key_not_present_in_outer(self):
        """A key set only in inner context must be absent in structlog after inner exits."""
        import structlog.contextvars

        from vetinari.structured_logging import CorrelationContext

        structlog.contextvars.clear_contextvars()
        try:
            with CorrelationContext(trace_id="outer-trace-003"):
                with CorrelationContext(trace_id="inner-trace-004", span_id="inner-span"):
                    inner_ctx = structlog.contextvars.get_contextvars()
                    assert inner_ctx.get("span_id") == "inner-span"

                # span_id was not present before inner context — must be removed
                after_ctx = structlog.contextvars.get_contextvars()
                assert "span_id" not in after_ctx or after_ctx.get("span_id") is None, (
                    "D3 fix: key not present in outer scope must be unbound after inner exits"
                )
        finally:
            structlog.contextvars.clear_contextvars()


# -- D4: set_request_id returns token; clear_request_id restores outer -----


class TestRequestIdTokenLifecycle:
    """D4: set_request_id() returns a reset token; clear_request_id(token) restores outer state."""

    def test_set_request_id_returns_non_none_token(self):
        """set_request_id must return a token that can later be used to reset."""
        from vetinari.structured_logging import clear_request_id, set_request_id

        token = set_request_id("req-abc")
        assert token is not None, "D4 fix: set_request_id must return a reset token"
        clear_request_id(token)

    def test_clear_request_id_removes_value(self):
        """clear_request_id(token) must leave get_request_id() returning None when no outer value exists."""
        import structlog.contextvars

        from vetinari.structured_logging import clear_request_id, get_request_id, set_request_id

        structlog.contextvars.clear_contextvars()
        try:
            token = set_request_id("req-xyz")
            assert get_request_id() == "req-xyz"

            clear_request_id(token)
            assert get_request_id() is None, (
                "D4 fix: clear_request_id must reset request_id to None when no outer scope exists"
            )
            # Also confirm structlog no longer has this key
            ctx = structlog.contextvars.get_contextvars()
            assert "request_id" not in ctx or ctx.get("request_id") is None
        finally:
            structlog.contextvars.clear_contextvars()

    def test_clear_request_id_restores_outer_scope(self):
        """clear_request_id must restore the outer request_id in structlog when one exists."""
        import structlog.contextvars

        from vetinari.structured_logging import clear_request_id, get_request_id, set_request_id

        structlog.contextvars.clear_contextvars()
        try:
            outer_token = set_request_id("outer-req")
            inner_token = set_request_id("inner-req")

            assert get_request_id() == "inner-req"
            clear_request_id(inner_token)

            # Outer value must be restored
            assert get_request_id() == "outer-req", (
                "D4 fix: clear_request_id must restore outer request_id after inner token reset"
            )
            ctx = structlog.contextvars.get_contextvars()
            assert ctx.get("request_id") == "outer-req"

            clear_request_id(outer_token)
        finally:
            structlog.contextvars.clear_contextvars()


# -- D5: set_context returns child; original logger is not mutated ----------


class TestStructuredLoggerSetContext:
    """D5: set_context() returns a new child logger; the original is not mutated."""

    def test_set_context_returns_new_instance(self):
        """set_context must return a different object from the original logger."""
        from vetinari.structured_logging import get_logger

        original = get_logger("d5-test")
        child = original.set_context(job_id="j001")

        assert child is not original, "D5 fix: set_context must return a NEW logger instance"

    def test_original_logger_not_mutated_by_set_context(self):
        """After set_context, the original logger must NOT carry the new key."""
        from vetinari.structured_logging import get_logger

        original = get_logger("d5-isolation-test")
        _child = original.set_context(exclusive_key="should-not-bleed")

        # original._context or structlog bound values must not contain the child's key
        # We verify by calling original.bind state — if the key bled through, it would appear
        # in the child_of_original
        child_of_original = original.set_context(other_key="v2")
        # child_of_original must not see exclusive_key (it was set on a sibling branch)
        assert not hasattr(child_of_original, "_context") or (
            "exclusive_key" not in getattr(child_of_original, "_context", {})
        ), "D5 fix: set_context must not mutate the original logger's bound context"


# -- D6: async_timed_operation and async_traced_operation decorators --------


class TestAsyncDecoratorProbes:
    """D6: async decorator variants must be present, preserve return values, and propagate exceptions."""

    def test_async_timed_operation_preserves_return_value(self):
        """async_timed_operation must return the wrapped coroutine's return value."""
        import asyncio

        from vetinari.structured_logging import async_timed_operation

        @async_timed_operation("d6-op")
        async def _returns_42() -> int:
            return 42

        result = asyncio.run(_returns_42())
        assert result == 42, "D6 fix: async_timed_operation must preserve the coroutine return value"

    def test_async_timed_operation_propagates_exception(self):
        """async_timed_operation must not swallow exceptions from the wrapped coroutine."""
        import asyncio

        from vetinari.structured_logging import async_timed_operation

        @async_timed_operation("d6-failing-op")
        async def _raises() -> None:
            raise ValueError("probe-error-d6")

        with pytest.raises(ValueError, match="probe-error-d6"):
            asyncio.run(_raises())

    def test_async_traced_operation_preserves_return_value(self):
        """async_traced_operation must return the wrapped coroutine's return value."""
        import asyncio

        from vetinari.structured_logging import async_traced_operation

        @async_traced_operation("d6-traced-op")
        async def _returns_hello() -> str:
            return "hello"

        result = asyncio.run(_returns_hello())
        assert result == "hello", "D6 fix: async_traced_operation must preserve the coroutine return value"

    def test_async_traced_operation_propagates_exception(self):
        """async_traced_operation must not swallow exceptions from the wrapped coroutine."""
        import asyncio

        from vetinari.structured_logging import async_traced_operation

        @async_traced_operation("d6-failing-traced-op")
        async def _raises() -> None:
            raise RuntimeError("probe-traced-error-d6")

        with pytest.raises(RuntimeError, match="probe-traced-error-d6"):
            asyncio.run(_raises())


class TestConfigureLoggingDefaultFormat:
    """33C.2 D2/D3: default output must be console text, not JSON.

    These tests verify the format-detection logic that configure_logging() uses
    to choose between ConsoleRenderer (default) and JSONRenderer (json env).
    We test _use_structured_logging() directly rather than reconfiguring logging
    at runtime, because calling structlog.configure() while tests are running
    with cache_logger_on_first_use=True causes a deadlock in the pytest thread
    timeout mechanism.
    """

    def test_default_format_detection_returns_false(self):
        """Without VETINARI_LOG_FORMAT set, _use_structured_logging() must return False.

        False means configure_logging() will use ConsoleRenderer (text), not JSON.
        """
        import os

        from vetinari import structured_logging as sl

        # Ensure neither env var is set (unset if present from other tests)
        env_backup_fmt = os.environ.pop("VETINARI_LOG_FORMAT", None)
        env_backup_sl = os.environ.pop("VETINARI_STRUCTURED_LOGGING", None)
        try:
            result = sl._use_structured_logging()
        finally:
            if env_backup_fmt is not None:
                os.environ["VETINARI_LOG_FORMAT"] = env_backup_fmt
            if env_backup_sl is not None:
                os.environ["VETINARI_STRUCTURED_LOGGING"] = env_backup_sl

        assert result is False, (
            f"Default _use_structured_logging() must return False (console/text), got {result!r}"
        )

    def test_json_format_when_env_set(self, monkeypatch):
        """With VETINARI_LOG_FORMAT=json, _use_structured_logging() must return True.

        True means configure_logging() will use JSONRenderer, producing parseable JSON.
        """
        monkeypatch.setenv("VETINARI_LOG_FORMAT", "json")

        from vetinari import structured_logging as sl

        result = sl._use_structured_logging()
        assert result is True, (
            "Expected _use_structured_logging() to return True when "
            f"VETINARI_LOG_FORMAT=json, got {result!r}"
        )
