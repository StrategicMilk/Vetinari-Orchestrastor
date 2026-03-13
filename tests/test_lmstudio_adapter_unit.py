"""Tests for vetinari.adapters.lmstudio_adapter — LM Studio adapter unit tests."""

from __future__ import annotations

from unittest.mock import patch

from vetinari.adapters.lmstudio_adapter import (
    get_lmstudio_headers,
    resolve_lmstudio_model,
)


class TestGetLmstudioHeaders:
    """Tests for the get_lmstudio_headers function."""

    def test_returns_content_type(self):
        headers = get_lmstudio_headers()
        assert headers["Content-Type"] == "application/json"

    def test_no_auth_without_token(self):
        with patch.dict("os.environ", {}, clear=True):
            headers = get_lmstudio_headers()
            assert "Authorization" not in headers or headers.get("Authorization") == ""

    def test_auth_with_token(self):
        with patch.dict("os.environ", {"LM_STUDIO_API_TOKEN": "test-token-123"}):
            headers = get_lmstudio_headers()
            assert headers.get("Authorization") == "Bearer test-token-123"


class TestResolveLmstudioModel:
    """Tests for the resolve_lmstudio_model function."""

    def test_real_model_name_returned_as_is(self):
        result = resolve_lmstudio_model("qwen2.5-coder-32b")
        assert result == "qwen2.5-coder-32b"

    def test_empty_string_triggers_resolution(self):
        with patch.dict("os.environ", {"VETINARI_DEFAULT_MODEL": "my-model"}):
            result = resolve_lmstudio_model("")
            assert result == "my-model"

    def test_default_keyword_triggers_resolution(self):
        with patch.dict("os.environ", {"VETINARI_DEFAULT_MODEL": "env-model"}):
            result = resolve_lmstudio_model("default")
            assert result == "env-model"
