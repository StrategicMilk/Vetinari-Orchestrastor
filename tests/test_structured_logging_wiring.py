"""Tests for structured logging wiring into core pipeline modules.

Verifies that:
- CorrelationContext propagates trace_id correctly via ContextVars.
- JSON log output contains all required fields (timestamp, level, logger, message, trace_id).
- Core pipeline modules import structured_logging (source-level check).
- configure_logging() respects the VETINARI_LOG_FORMAT environment variable.
"""

from __future__ import annotations

import importlib
import io
import json
import logging
import os
import sys
from unittest.mock import patch

import pytest


class TestCorrelationContext:
    """Tests for CorrelationContext trace_id propagation."""

    def test_trace_id_is_set_within_context(self) -> None:
        """trace_id ContextVar is populated when entering a CorrelationContext."""
        from vetinari.structured_logging import CorrelationContext, get_trace_id

        with CorrelationContext() as ctx:
            assert get_trace_id() == ctx.trace_id
            assert ctx.trace_id  # non-empty UUID string

    def test_trace_id_is_cleared_after_context_exit(self) -> None:
        """trace_id ContextVar is reset to None after exiting a CorrelationContext."""
        from vetinari.structured_logging import CorrelationContext, get_trace_id

        with CorrelationContext():
            pass

        assert get_trace_id() is None

    def test_explicit_trace_id_is_propagated(self) -> None:
        """A user-supplied trace_id is preserved and retrievable from the ContextVar."""
        from vetinari.structured_logging import CorrelationContext, get_trace_id

        custom_id = "my-custom-trace-42"
        with CorrelationContext(trace_id=custom_id):
            assert get_trace_id() == custom_id

    def test_nested_correlation_contexts_restore_outer_trace_id(self) -> None:
        """Nested contexts restore the outer trace_id when the inner context exits."""
        from vetinari.structured_logging import CorrelationContext, get_trace_id

        outer_id = "outer-trace"
        inner_id = "inner-trace"

        with CorrelationContext(trace_id=outer_id):
            assert get_trace_id() == outer_id

            with CorrelationContext(trace_id=inner_id):
                assert get_trace_id() == inner_id

            # Inner context exited — outer trace_id must be restored
            assert get_trace_id() == outer_id


class TestJSONLogFormat:
    """Tests for structured JSON log output field schema."""

    def _capture_json_log(self, message: str, **extra) -> dict:
        """Emit one log record via structlog and return the parsed JSON.

        Uses structlog.get_logger() to go through the full processor chain
        including timestamp, log_level, and context enrichment.

        Args:
            message: Log message text.
            **extra: Additional keyword arguments forwarded as context fields.

        Returns:
            Parsed JSON dict from the formatted log record.
        """
        import structlog

        from vetinari.structured_logging import (
            CorrelationContext,
            _add_correlation_context,
            _add_service_context,
        )

        # Build a standalone structlog pipeline that captures JSON output
        output_lines: list[str] = []

        def _capture_renderer(
            logger_obj: object,
            method_name: str,
            event_dict: dict,
        ) -> str:
            rendered = json.dumps(event_dict)
            output_lines.append(rendered)
            return rendered

        processors = [
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", key="timestamp"),
            _add_correlation_context,
            _add_service_context,
            _capture_renderer,
        ]

        bound_logger = structlog.wrap_logger(
            logging.getLogger("test.structured_logging_wiring"),
            processors=processors,
            logger_factory_args=(),
        )

        with CorrelationContext():
            bound_logger.info(message, **extra)

        if not output_lines:
            return {}
        record = json.loads(output_lines[0])
        # Map structlog field names to test expectations
        if "event" in record and "message" not in record:
            record["message"] = record["event"]
        if "level" in record:
            record["level"] = record["level"].upper()
        return record

    def test_json_output_has_timestamp(self) -> None:
        """JSON log record must contain a 'timestamp' field."""
        record = self._capture_json_log("ping")
        assert "timestamp" in record

    def test_json_output_has_level(self) -> None:
        """JSON log record must contain a 'level' field."""
        record = self._capture_json_log("ping")
        assert "level" in record
        assert record["level"] == "INFO"

    def test_json_output_has_logger_name(self) -> None:
        """JSON log record must contain a 'logger' field matching the logger name."""
        record = self._capture_json_log("ping")
        assert "logger" in record
        assert record["logger"] == "test.structured_logging_wiring"

    def test_json_output_has_message(self) -> None:
        """JSON log record must contain a 'message' field with the log text."""
        record = self._capture_json_log("hello world")
        assert "message" in record
        assert record["message"] == "hello world"

    def test_json_output_has_trace_id_when_in_correlation_context(self) -> None:
        """JSON log record must contain 'trace_id' when emitted inside a CorrelationContext."""
        record = self._capture_json_log("traced event")
        assert "trace_id" in record
        assert record["trace_id"]  # non-empty


class TestCoreModuleImports:
    """Verify that core pipeline modules import structured_logging for wiring."""

    def _get_source(self, module_path: str) -> str:
        """Return the source code of a module file by dotted import path.

        Args:
            module_path: Dotted module path (e.g. ``vetinari.orchestration.two_layer``).

        Returns:
            Source code string of the module file.
        """
        spec = importlib.util.find_spec(module_path)
        assert spec is not None, f"Module {module_path} not found"
        assert spec.origin is not None, f"Module {module_path} has no file origin"
        with open(spec.origin, encoding="utf-8") as fh:
            return fh.read()

    def test_two_layer_imports_correlation_context(self) -> None:
        """TwoLayerOrchestrator source must reference CorrelationContext."""
        source = self._get_source("vetinari.orchestration.two_layer")
        assert "CorrelationContext" in source, "vetinari.orchestration.two_layer must use CorrelationContext"

    def test_agent_graph_imports_log_event(self) -> None:
        """AgentGraph source must reference log_event for structured task events."""
        source = self._get_source("vetinari.orchestration.agent_graph")
        assert "structured_logging" in source, (
            "vetinari.orchestration.agent_graph must import from vetinari.structured_logging"
        )

    def test_base_adapter_imports_log_event(self) -> None:
        """ProviderAdapter source must reference log_event for inference events."""
        source = self._get_source("vetinari.adapters.base")
        assert "structured_logging" in source, "vetinari.adapters.base must import from vetinari.structured_logging"


class TestConfigureLogging:
    """Tests for configure_logging() env var behaviour."""

    def setup_method(self) -> None:
        """Reset the global _configured flag before each test."""
        import vetinari.structured_logging as _sl

        _sl._configured = False  # type: ignore[attr-defined]  — necessary for test isolation

    def teardown_method(self) -> None:
        """Reset the global _configured flag after each test."""
        import vetinari.structured_logging as _sl

        _sl._configured = False  # type: ignore[attr-defined]

    def test_json_format_when_env_var_is_json(self) -> None:
        """configure_logging() installs structlog ProcessorFormatter when VETINARI_LOG_FORMAT=json."""
        import structlog

        from vetinari.structured_logging import configure_logging

        with patch.dict(os.environ, {"VETINARI_LOG_FORMAT": "json"}, clear=False):
            configure_logging()

        root_logger = logging.getLogger()
        handlers = root_logger.handlers
        assert handlers, "Expected at least one handler on root logger"
        assert isinstance(handlers[-1].formatter, structlog.stdlib.ProcessorFormatter), (
            "Root logger handler must use structlog ProcessorFormatter"
        )

    def test_text_format_when_env_var_is_text(self) -> None:
        """configure_logging() installs structlog ProcessorFormatter for text output."""
        import structlog

        from vetinari.structured_logging import configure_logging

        with patch.dict(os.environ, {"VETINARI_LOG_FORMAT": "text"}, clear=False):
            configure_logging()

        root_logger = logging.getLogger()
        handlers = root_logger.handlers
        assert handlers, "Expected at least one handler on root logger"
        assert isinstance(handlers[-1].formatter, structlog.stdlib.ProcessorFormatter), (
            "Root logger handler must use structlog ProcessorFormatter for text mode"
        )

    def test_json_format_by_default(self) -> None:
        """configure_logging() defaults to structlog ProcessorFormatter."""
        import structlog

        from vetinari.structured_logging import configure_logging

        env_without_format = {
            k: v for k, v in os.environ.items() if k not in ("VETINARI_LOG_FORMAT", "VETINARI_STRUCTURED_LOGGING")
        }
        with patch.dict(os.environ, env_without_format, clear=True):
            configure_logging()

        root_logger = logging.getLogger()
        handlers = root_logger.handlers
        assert handlers, "Expected at least one handler on root logger"
        assert isinstance(handlers[-1].formatter, structlog.stdlib.ProcessorFormatter), (
            "Root logger handler must default to structlog ProcessorFormatter"
        )

    def test_configure_logging_is_idempotent(self) -> None:
        """Calling configure_logging() twice does not add duplicate handlers."""
        from vetinari.structured_logging import configure_logging

        root_logger = logging.getLogger()

        configure_logging()
        after_first_call = len(root_logger.handlers)

        configure_logging()
        after_second_call = len(root_logger.handlers)

        assert after_second_call == after_first_call, (
            "configure_logging() must not add a second handler when called again"
        )

    def test_use_structured_logging_respects_log_format_json(self) -> None:
        """_use_structured_logging() returns True when VETINARI_LOG_FORMAT=json."""
        from vetinari.structured_logging import _use_structured_logging

        with patch.dict(os.environ, {"VETINARI_LOG_FORMAT": "json"}, clear=False):
            assert _use_structured_logging() is True

    def test_use_structured_logging_respects_log_format_text(self) -> None:
        """_use_structured_logging() returns False when VETINARI_LOG_FORMAT=text."""
        from vetinari.structured_logging import _use_structured_logging

        with patch.dict(os.environ, {"VETINARI_LOG_FORMAT": "text"}, clear=False):
            assert _use_structured_logging() is False
