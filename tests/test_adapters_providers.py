"""Tests for provider adapters (LlamaCpp, LiteLLM) — Phase 7B.

All cloud providers now route through LiteLLMAdapter (ADR-0062).
Only LOCAL retains its dedicated LlamaCppProviderAdapter.
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import TEST_API_KEY, TEST_ENDPOINT
from vetinari.adapters.base import (
    InferenceRequest,
    ModelInfo,
    ProviderConfig,
    ProviderType,
)
from vetinari.exceptions import ConfigurationError


def _cfg(pt, endpoint=TEST_ENDPOINT, api_key=TEST_API_KEY):
    """Create a test ProviderConfig."""
    return ProviderConfig(provider_type=pt, name="test", endpoint=endpoint, api_key=api_key)


class TestLlamaCppProviderAdapter:
    """Tests for the local GGUF inference adapter."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from vetinari.adapters.llama_cpp_adapter import LlamaCppProviderAdapter

        self.AdapterClass = LlamaCppProviderAdapter

    def test_wrong_provider_type_raises(self):
        """Constructing with a non-LOCAL provider type raises ConfigurationError."""
        cfg = _cfg(ProviderType.OPENAI)
        with pytest.raises(ConfigurationError):
            self.AdapterClass(cfg)

    def test_discover_models_returns_list(self):
        """discover_models returns a list (may be empty if no models dir)."""
        adapter = self.AdapterClass(_cfg(ProviderType.LOCAL))
        models = adapter.discover_models()
        assert isinstance(models, list)

    def test_health_check_returns_dict(self):
        """health_check always returns a dict with a 'healthy' key."""
        adapter = self.AdapterClass(_cfg(ProviderType.LOCAL))
        health = adapter.health_check()
        assert "healthy" in health

    def test_get_capabilities_returns_dict(self):
        """get_capabilities returns a dict."""
        adapter = self.AdapterClass(_cfg(ProviderType.LOCAL))
        caps = adapter.get_capabilities()
        assert isinstance(caps, dict)


class TestLiteLLMAdapter:
    """Tests for the unified LiteLLM adapter covering all cloud providers."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from vetinari.adapters.litellm_adapter import LiteLLMAdapter

        self.AdapterClass = LiteLLMAdapter

    def test_construct_with_any_cloud_provider_type(self):
        """LiteLLMAdapter accepts any ProviderType (it's universal)."""
        for pt in (
            ProviderType.OPENAI,
            ProviderType.ANTHROPIC,
            ProviderType.GEMINI,
            ProviderType.COHERE,
            ProviderType.LITELLM,
        ):
            adapter = self.AdapterClass(_cfg(pt))
            assert isinstance(adapter, self.AdapterClass)

    @pytest.mark.parametrize(
        ("input_id", "expected"),
        [
            ("gpt-4o", "openai/gpt-4o"),
            ("openai/gpt-4o", "openai/gpt-4o"),
            ("gpt-4", "openai/gpt-4"),
        ],
    )
    def test_model_id_formatting(self, input_id, expected):
        """Model IDs are prefixed with provider name unless already prefixed."""
        adapter = self.AdapterClass(_cfg(ProviderType.OPENAI))
        assert adapter._model_id(input_id) == expected

    def test_health_check_cloud(self):
        """Health check returns a dict with healthy key for cloud providers."""
        adapter = self.AdapterClass(_cfg(ProviderType.OPENAI))
        health = adapter.health_check()
        assert "healthy" in health
        assert "timestamp" in health

    def test_health_check_local_endpoint(self):
        """Health check for local endpoint returns unhealthy dict when unreachable."""
        adapter = self.AdapterClass(_cfg(ProviderType.LOCAL, endpoint="http://localhost:9999"))
        with patch("vetinari.http.create_session") as mock_session_fn:
            mock_session = MagicMock()
            mock_session.__enter__ = lambda s: s
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.get.side_effect = ConnectionRefusedError("Connection refused")
            mock_session_fn.return_value = mock_session

            health = adapter.health_check()
        assert "healthy" in health
        assert health["healthy"] is False

    def test_get_capabilities_empty_initially(self):
        """get_capabilities returns empty dict before discovery."""
        adapter = self.AdapterClass(_cfg(ProviderType.OPENAI))
        caps = adapter.get_capabilities()
        assert isinstance(caps, dict)
        assert len(caps) == 0

    @patch("vetinari.adapters.litellm_adapter._litellm")
    def test_infer_success(self, mock_litellm_fn):
        """infer returns output on successful litellm.completion call."""
        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 42
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 32
        mock_litellm.completion.return_value = mock_response
        mock_litellm_fn.return_value = mock_litellm

        adapter = self.AdapterClass(_cfg(ProviderType.OPENAI))
        result = adapter.infer(InferenceRequest(model_id="gpt-4", prompt="hello"))

        assert result.status == "ok"
        assert result.output == "test response"
        assert result.tokens_used == 42

    @patch("vetinari.adapters.litellm_adapter._litellm")
    def test_infer_error(self, mock_litellm_fn):
        """infer returns status='error' when litellm raises."""
        mock_litellm = MagicMock()
        mock_litellm.completion.side_effect = Exception("API error")
        mock_litellm_fn.return_value = mock_litellm

        adapter = self.AdapterClass(_cfg(ProviderType.OPENAI))
        result = adapter.infer(InferenceRequest(model_id="gpt-4", prompt="test"))

        assert result.status == "error"
        assert result.error  # Generic error message (real details in server logs)

    @patch("vetinari.adapters.litellm_adapter._litellm")
    def test_infer_passes_cache_control_for_anthropic(self, mock_litellm_fn):
        """Anthropic requests include cache_control on system messages."""
        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 5
        mock_response.usage.prompt_tokens = 2
        mock_response.usage.completion_tokens = 3
        mock_litellm.completion.return_value = mock_response
        mock_litellm_fn.return_value = mock_litellm

        adapter = self.AdapterClass(_cfg(ProviderType.ANTHROPIC))
        request = InferenceRequest(
            model_id="claude-3",
            prompt="test",
            system_prompt="system",
            metadata={"cache_control": {"type": "ephemeral"}},
        )
        adapter.infer(request)

        # Verify cache_control was added to system message
        call_kwargs = mock_litellm.completion.call_args
        messages = call_kwargs.kwargs["messages"]
        system_msg = messages[0]
        assert system_msg["cache_control"] == {"type": "ephemeral"}


class TestAdapterRegistry:
    """Tests for the adapter registry with LiteLLM routing."""

    def test_all_cloud_types_route_to_litellm(self):
        """All cloud ProviderType values map to LiteLLMAdapter."""
        from vetinari.adapters.litellm_adapter import LiteLLMAdapter
        from vetinari.adapters.registry import AdapterRegistry

        for pt in (
            ProviderType.OPENAI,
            ProviderType.ANTHROPIC,
            ProviderType.GEMINI,
            ProviderType.COHERE,
            ProviderType.LITELLM,
        ):
            adapter = AdapterRegistry.create_adapter(_cfg(pt))
            assert isinstance(adapter, LiteLLMAdapter)

    def test_local_type_routes_to_llama_cpp(self):
        """LOCAL ProviderType maps to LlamaCppProviderAdapter."""
        from vetinari.adapters.llama_cpp_adapter import LlamaCppProviderAdapter
        from vetinari.adapters.registry import AdapterRegistry

        adapter = AdapterRegistry.create_adapter(_cfg(ProviderType.LOCAL))
        assert isinstance(adapter, LlamaCppProviderAdapter)

    def test_litellm_in_supported_providers(self):
        """LITELLM is listed as a supported provider."""
        from vetinari.adapters.registry import AdapterRegistry

        supported = AdapterRegistry.list_supported_providers()
        assert ProviderType.LITELLM in supported
