"""Tests for vetinari.web.error_handlers — error message humanization."""

from __future__ import annotations

import pytest

from vetinari.web.error_handlers import humanize_error, humanize_error_message


class TestHumanizeError:
    """Tests for humanize_error() exception-to-message translation."""

    def test_timeout_error_produces_friendly_message(self):
        exc = TimeoutError("operation timed out after 30s")
        result = humanize_error(exc)
        assert "took too long" in result
        assert "30s" in result

    def test_connection_refused_produces_friendly_message(self):
        exc = ConnectionRefusedError("[Errno 111] Connection refused")
        result = humanize_error(exc)
        assert "model server" in result.lower() or "connect" in result.lower()

    def test_memory_error_produces_friendly_message(self):
        exc = MemoryError("out of memory")
        result = humanize_error(exc)
        assert "memory" in result.lower()

    def test_key_error_names_missing_field(self):
        exc = KeyError("model_id")
        result = humanize_error(exc)
        assert "model_id" in result
        assert "Missing" in result or "missing" in result

    def test_oom_pattern_match(self):
        exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        result = humanize_error(exc)
        assert "GPU memory" in result

    def test_unknown_error_returns_generic_message(self):
        exc = ValueError("some internal error xyz123")
        result = humanize_error(exc)
        assert "unexpected error" in result.lower()
        # Must NOT contain the raw exception text
        assert "xyz123" not in result

    def test_connection_error_pattern(self):
        exc = OSError("connection refused on port 8080")
        result = humanize_error(exc)
        assert "connect" in result.lower() or "model server" in result.lower()

    def test_file_not_found_error_produces_friendly_message(self):
        exc = FileNotFoundError("No such file: /path/to/model.gguf")
        result = humanize_error(exc)
        assert "not found" in result.lower() or "file" in result.lower()

    def test_permission_error_produces_friendly_message(self):
        exc = PermissionError("Permission denied: /var/log/vetinari.log")
        result = humanize_error(exc)
        assert "permission" in result.lower() or "insufficient" in result.lower()

    def test_result_is_never_empty(self):
        """humanize_error must always return a non-empty string."""
        for exc in [ValueError("x"), RuntimeError("y"), Exception("z")]:
            result = humanize_error(exc)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_rate_limit_pattern_match(self):
        exc = RuntimeError("rate limit exceeded, retry after 60s")
        result = humanize_error(exc)
        assert "wait" in result.lower() or "too many" in result.lower()

    def test_context_length_pattern_match(self):
        exc = RuntimeError("context length exceeded: 8192 tokens max")
        result = humanize_error(exc)
        assert "too long" in result.lower() or "context" in result.lower()


class TestHumanizeErrorMessage:
    """Tests for humanize_error_message() string-based translation."""

    def test_oom_string_pattern(self):
        result = humanize_error_message("CUDA out of memory. Tried to allocate 4GB")
        assert "GPU memory" in result

    def test_timeout_pattern(self):
        result = humanize_error_message("Request timed out after 60 seconds")
        assert "timed out" in result.lower() or "timeout" in result.lower()

    def test_unknown_message_returns_generic(self):
        result = humanize_error_message("frobnicator failed to grok the quux")
        assert "unexpected error" in result.lower()
        assert "frobnicator" not in result

    def test_connection_refused_pattern(self):
        result = humanize_error_message("connection refused on 127.0.0.1:11434")
        assert "connect" in result.lower() or "model server" in result.lower()

    def test_model_not_found_pattern(self):
        result = humanize_error_message("model not found: llama-3-8b-instruct.gguf")
        assert "model" in result.lower()

    def test_result_is_never_empty(self):
        """humanize_error_message must always return a non-empty string."""
        for msg in ["", "unknown", "xyz"]:
            result = humanize_error_message(msg)
            assert isinstance(result, str)
            assert len(result) > 0
