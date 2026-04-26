"""Tests for OTel backend configuration in vetinari.observability.otel_genai.

Covers:
- configure_backend() with valid and invalid inputs
- get_active_backend() reflects changes
- flush_file_backend() is a no-op unless backend is 'file'
- Environment variable initialisation (_init_backend_from_env)
- VETINARI_OTEL_BACKEND / VETINARI_OTEL_ENDPOINT env vars
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def reload_module():
    """Reload otel_genai so env-var init runs fresh."""
    import vetinari.observability.otel_genai as mod

    importlib.reload(mod)
    return mod


# ---------------------------------------------------------------------------
# configure_backend() validation
# ---------------------------------------------------------------------------


def test_configure_backend_noop(monkeypatch):
    import vetinari.observability.otel_genai as mod

    mod.configure_backend("noop")
    assert mod.get_active_backend() == "noop"


def test_configure_backend_file(monkeypatch):
    import vetinari.observability.otel_genai as mod

    mod.configure_backend("file")
    assert mod.get_active_backend() == "file"
    # Restore to noop so other tests are unaffected
    mod.configure_backend("noop")


def test_configure_backend_invalid_raises():
    import vetinari.observability.otel_genai as mod

    with pytest.raises(ValueError, match="Invalid OTel backend"):
        mod.configure_backend("prometheus")


def test_configure_backend_jaeger_no_sdk(monkeypatch):
    """configure_backend('jaeger') warns and reports noop when SDK is absent."""
    import vetinari.observability.otel_genai as mod

    # Simulate SDK absent by patching _OTEL_AVAILABLE
    monkeypatch.setattr(mod, "_OTEL_AVAILABLE", False)

    # Should not raise even though SDK is unavailable
    mod.configure_backend("jaeger", "http://localhost:4317")
    # get_active_backend() reports the actual export mode, not the requested backend.
    assert mod.get_active_backend() == "noop"

    # Restore
    mod.configure_backend("noop")


# ---------------------------------------------------------------------------
# flush_file_backend()
# ---------------------------------------------------------------------------


def test_flush_file_backend_noop_when_not_file(tmp_path):
    import vetinari.observability.otel_genai as mod

    mod.configure_backend("noop")
    result = mod.flush_file_backend()
    assert result == 0


def test_flush_file_backend_exports_spans(tmp_path, monkeypatch):
    """flush_file_backend() should export spans to outputs/traces/ when backend=file."""
    import vetinari.observability.otel_genai as mod

    # Point outputs/ into tmp_path so we don't pollute the repo
    fake_outputs = tmp_path / "outputs"
    monkeypatch.chdir(tmp_path)

    mod.configure_backend("file")

    # Record a test span so there's something to export
    tracer = mod.get_genai_tracer()
    tracer.reset()
    span = tracer.start_agent_span("test-agent", "chat")
    tracer.end_agent_span(span, status="ok", tokens_used=10)

    count = mod.flush_file_backend()
    assert count >= 1

    # At least one file must have been written
    trace_files = list((fake_outputs / "traces").glob("traces_*.json"))
    assert len(trace_files) >= 1

    mod.configure_backend("noop")


# ---------------------------------------------------------------------------
# Environment variable initialisation
# ---------------------------------------------------------------------------


def test_env_var_noop_is_default(monkeypatch):
    """When no env var is set, backend defaults to noop."""
    monkeypatch.delenv("VETINARI_OTEL_BACKEND", raising=False)
    monkeypatch.delenv("VETINARI_OTEL_ENDPOINT", raising=False)

    import vetinari.observability.otel_genai as mod

    mod._init_backend_from_env()
    assert mod.get_active_backend() == "noop"


def test_env_var_file_sets_file_backend(monkeypatch, tmp_path):
    """VETINARI_OTEL_BACKEND=file selects file backend."""
    monkeypatch.setenv("VETINARI_OTEL_BACKEND", "file")
    monkeypatch.delenv("VETINARI_OTEL_ENDPOINT", raising=False)
    monkeypatch.chdir(tmp_path)

    import vetinari.observability.otel_genai as mod

    mod._init_backend_from_env()
    assert mod.get_active_backend() == "file"

    # Clean up
    mod.configure_backend("noop")


def test_env_var_invalid_falls_back_to_noop(monkeypatch):
    """An unrecognised VETINARI_OTEL_BACKEND value logs a warning and defaults to noop."""
    monkeypatch.setenv("VETINARI_OTEL_BACKEND", "splunk")
    monkeypatch.delenv("VETINARI_OTEL_ENDPOINT", raising=False)

    import vetinari.observability.otel_genai as mod

    # Should not raise; should log a warning and set noop
    mod._init_backend_from_env()
    assert mod.get_active_backend() == "noop"
