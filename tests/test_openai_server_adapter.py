"""Tests for OpenAIServerAdapter — vLLM / NVIDIA NIMs adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.adapters.base import InferenceRequest, InferenceResponse, ProviderConfig, ProviderType
from vetinari.adapters.openai_server_adapter import OpenAIServerAdapter, _semantic_cache_identity


class FakeSemanticCache:
    """Small semantic-cache test double that records get/put identity material."""

    def __init__(self, hit: str | None = None) -> None:
        self.hit = hit
        self.get_calls: list[dict[str, object]] = []
        self.put_calls: list[dict[str, object]] = []

    def get(
        self,
        query: str,
        similarity_threshold: float = 0.0,
        task_type: str = "",
        model_id: str = "",
        system_prompt: str = "",
    ) -> str | None:
        self.get_calls.append({
            "query": query,
            "similarity_threshold": similarity_threshold,
            "task_type": task_type,
            "model_id": model_id,
            "system_prompt": system_prompt,
        })
        return self.hit

    def put(self, query: str, response: str, model_id: str = "", system_prompt: str = "") -> None:
        self.put_calls.append({
            "query": query,
            "response": response,
            "model_id": model_id,
            "system_prompt": system_prompt,
        })


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def vllm_config() -> ProviderConfig:
    """Config for a vLLM backend."""
    return ProviderConfig(
        provider_type=ProviderType.VLLM,
        name="vllm-test",
        endpoint="http://localhost:8000",
        extra_config={"gpu_only": True, "semantic_cache_enabled": False},
    )


@pytest.fixture
def nim_config() -> ProviderConfig:
    """Config for a NIM backend."""
    return ProviderConfig(
        provider_type=ProviderType.NIM,
        name="nim-test",
        endpoint="http://localhost:8001",
        extra_config={"gpu_only": True, "semantic_cache_enabled": False},
    )


@pytest.fixture
def adapter(vllm_config: ProviderConfig) -> OpenAIServerAdapter:
    """Adapter instance with vLLM config."""
    return OpenAIServerAdapter(vllm_config)


# ── Initialization Tests ─────────────────────────────────────────────────────


class TestAdapterInit:
    def test_vllm_config(self, vllm_config: ProviderConfig) -> None:
        adapter = OpenAIServerAdapter(vllm_config)
        assert adapter.provider_type == ProviderType.VLLM
        assert adapter.name == "vllm-test"
        assert adapter._api_base == "http://localhost:8000"
        assert adapter.gpu_only is True

    def test_nim_config(self, nim_config: ProviderConfig) -> None:
        adapter = OpenAIServerAdapter(nim_config)
        assert adapter.provider_type == ProviderType.NIM
        assert adapter.name == "nim-test"
        assert adapter._api_base == "http://localhost:8001"

    def test_gpu_only_default_true(self) -> None:
        config = ProviderConfig(
            provider_type=ProviderType.VLLM,
            name="test",
            endpoint="http://localhost:8000",
        )
        adapter = OpenAIServerAdapter(config)
        assert adapter.gpu_only is True

    def test_trailing_slash_stripped(self) -> None:
        config = ProviderConfig(
            provider_type=ProviderType.VLLM,
            name="test",
            endpoint="http://localhost:8000/",
        )
        adapter = OpenAIServerAdapter(config)
        assert adapter._api_base == "http://localhost:8000"

    def test_no_endpoint(self) -> None:
        config = ProviderConfig(
            provider_type=ProviderType.VLLM,
            name="test",
            endpoint="",
        )
        adapter = OpenAIServerAdapter(config)
        assert adapter._api_base == ""


# ── Health Check Tests ────────────────────────────────────────────────────────


class TestHealthCheck:
    def test_no_endpoint_returns_unhealthy(self) -> None:
        config = ProviderConfig(
            provider_type=ProviderType.VLLM,
            name="test",
            endpoint="",
        )
        adapter = OpenAIServerAdapter(config)
        result = adapter.health_check()
        assert result["healthy"] is False
        assert "No endpoint" in result["reason"]

    @patch("vetinari.adapters.openai_server_adapter._httpx")
    def test_healthy_server(self, mock_httpx: MagicMock, adapter: OpenAIServerAdapter) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"id": "model-1"}, {"id": "model-2"}]}
        mock_resp.raise_for_status = MagicMock()
        mock_httpx.return_value.get.return_value = mock_resp

        result = adapter.health_check()
        assert result["healthy"] is True
        assert "2 model(s)" in result["reason"]

    @patch("vetinari.adapters.openai_server_adapter._httpx")
    def test_unreachable_server(self, mock_httpx: MagicMock, adapter: OpenAIServerAdapter) -> None:
        mock_httpx.return_value.get.side_effect = ConnectionError("refused")
        result = adapter.health_check()
        assert result["healthy"] is False
        assert "Cannot reach" in result["reason"]


# ── vLLM Engine Version Gate (PRE-03 Task 1.3) ──────────────────────────────


class TestVLLMEngineVersionGate:
    """The adapter must hold itself unhealthy when vLLM reports a known-bad version.

    Closes PRE-03 Task 1.3: known-bad vLLM versions (e.g. 0.18.1 on SM120) MUST
    be rejected at adapter init / health-check time so the router skips this
    backend instead of dispatching inference to a regressed engine.
    """

    @staticmethod
    def _mock_responses(models_payload: dict, version_payload: dict | None, version_status: int = 200):
        """Build a fake httpx.get that distinguishes /v1/models from /version."""
        models_resp = MagicMock()
        models_resp.json.return_value = models_payload
        models_resp.raise_for_status = MagicMock()
        version_resp = MagicMock()
        version_resp.status_code = version_status
        if version_payload is not None:
            version_resp.json.return_value = version_payload

        def _get(url: str, *args, **kwargs):
            if url.endswith("/version"):
                return version_resp
            return models_resp

        return _get

    @patch("vetinari.adapters.openai_server_adapter._httpx")
    @patch("vetinari.adapters.openai_server_adapter.validate_runtime_version", create=True)
    def test_known_bad_version_marks_adapter_unhealthy(
        self,
        mock_validate: MagicMock,
        mock_httpx: MagicMock,
        adapter: OpenAIServerAdapter,
    ) -> None:
        from vetinari.runtime.runtime_doctor import RuntimeCheckResult

        mock_httpx.return_value.get.side_effect = self._mock_responses(
            models_payload={"data": [{"id": "m"}]},
            version_payload={"version": "0.18.1"},
        )
        # Patch validate_runtime_version inside the adapter module's import path.
        # The adapter imports it lazily inside _check_vllm_engine_version_once,
        # so we patch the runtime_doctor source instead.
        with patch(
            "vetinari.runtime.runtime_doctor.validate_runtime_version",
            return_value=RuntimeCheckResult(
                component="vllm",
                passed=False,
                detected_version="0.18.1",
                reason="vllm 0.18.1 falls inside known-bad range '==0.18.1'.",
                matrix_sources=("https://example.test/vllm",),
                is_blocker=True,
            ),
        ):
            result = adapter.health_check()

        assert result["healthy"] is False
        assert "0.18.1" in result["reason"]
        assert "supported matrix" in result["reason"].lower() or "known-bad" in result["reason"].lower()
        assert "remediation" in result["reason"].lower()

    @patch("vetinari.adapters.openai_server_adapter._httpx")
    def test_safe_version_keeps_adapter_healthy(
        self,
        mock_httpx: MagicMock,
        adapter: OpenAIServerAdapter,
    ) -> None:
        from vetinari.runtime.runtime_doctor import RuntimeCheckResult

        mock_httpx.return_value.get.side_effect = self._mock_responses(
            models_payload={"data": [{"id": "m"}]},
            version_payload={"version": "0.19.0"},
        )
        with patch(
            "vetinari.runtime.runtime_doctor.validate_runtime_version",
            return_value=RuntimeCheckResult(
                component="vllm",
                passed=True,
                detected_version="0.19.0",
                reason="vllm 0.19.0 satisfies the matrix.",
                matrix_sources=(),
                is_blocker=True,
            ),
        ):
            result = adapter.health_check()

        assert result["healthy"] is True

    @patch("vetinari.adapters.openai_server_adapter._httpx")
    def test_missing_version_endpoint_does_not_block(
        self,
        mock_httpx: MagicMock,
        adapter: OpenAIServerAdapter,
    ) -> None:
        # /version endpoint returns 404: treat as unknown version, do not block.
        mock_httpx.return_value.get.side_effect = self._mock_responses(
            models_payload={"data": [{"id": "m"}]},
            version_payload=None,
            version_status=404,
        )
        result = adapter.health_check()
        assert result["healthy"] is True

    def test_nim_provider_skips_vllm_version_check(
        self,
        nim_config: ProviderConfig,
    ) -> None:
        # NIM adapters must not invoke the vLLM gate even if /version exists.
        nim_adapter = OpenAIServerAdapter(nim_config)
        with patch("vetinari.adapters.openai_server_adapter._httpx") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"data": [{"id": "m"}]}
            mock_resp.raise_for_status = MagicMock()
            mock_httpx.return_value.get.return_value = mock_resp

            result = nim_adapter.health_check()
            assert result["healthy"] is True
            # Only /v1/models should have been called — never /version.
            called_urls = [call.args[0] for call in mock_httpx.return_value.get.call_args_list]
            assert all("/version" not in url for url in called_urls), called_urls


# ── Model Discovery Tests ────────────────────────────────────────────────────


class TestDiscoverModels:
    def test_no_endpoint_returns_empty(self) -> None:
        config = ProviderConfig(
            provider_type=ProviderType.VLLM,
            name="test",
            endpoint="",
        )
        adapter = OpenAIServerAdapter(config)
        assert adapter.discover_models() == []

    @patch("vetinari.adapters.openai_server_adapter._httpx")
    def test_discovers_models(self, mock_httpx: MagicMock, adapter: OpenAIServerAdapter) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"id": "qwen2.5-coder-7b", "owned_by": "qwen"},
                {"id": "llama-3.1-8b", "owned_by": "meta"},
            ],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_httpx.return_value.get.return_value = mock_resp

        models = adapter.discover_models()
        assert len(models) == 2
        assert models[0].id == "qwen2.5-coder-7b"
        assert models[0].provider == "vllm"
        assert "coding" in models[0].capabilities
        assert models[1].id == "llama-3.1-8b"
        assert models[1].free_tier is True  # Local server = no cost
        assert adapter._raw_model_entries["qwen2.5-coder-7b"]["owned_by"] == "qwen"


# ── Inference Tests ───────────────────────────────────────────────────────────


class TestInfer:
    def test_no_endpoint_returns_error(self) -> None:
        config = ProviderConfig(
            provider_type=ProviderType.VLLM,
            name="test",
            endpoint="",
        )
        adapter = OpenAIServerAdapter(config)
        request = InferenceRequest(model_id="test", prompt="hello")
        response = adapter.infer(request)
        assert response.status == "error"
        assert "No endpoint" in response.error

    @patch("vetinari.adapters.openai_server_adapter._httpx")
    def test_successful_inference(self, mock_httpx: MagicMock, adapter: OpenAIServerAdapter) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "model": "qwen2.5-coder-7b",
            "choices": [{"message": {"content": "Hello world!"}}],
            "usage": {"total_tokens": 42, "prompt_tokens": 10, "completion_tokens": 32},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_httpx.return_value.post.return_value = mock_resp

        request = InferenceRequest(
            model_id="qwen2.5-coder-7b",
            prompt="Say hello",
            system_prompt="You are helpful.",
            temperature=0.5,
        )
        response = adapter.infer(request)

        assert response.status == "ok"
        assert response.output == "Hello world!"
        assert response.tokens_used == 42
        assert response.model_id == "qwen2.5-coder-7b"
        assert response.metadata["gpu_only"] is True

        # Verify the request payload
        call_args = mock_httpx.return_value.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["model"] == "qwen2.5-coder-7b"
        assert payload["temperature"] == 0.5
        assert len(payload["messages"]) == 2  # system + user
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"

    @patch("vetinari.adapters.openai_server_adapter._httpx")
    def test_inference_failure(self, mock_httpx: MagicMock, adapter: OpenAIServerAdapter) -> None:
        mock_httpx.return_value.post.side_effect = ConnectionError("server down")

        request = InferenceRequest(model_id="test", prompt="hello")
        response = adapter.infer(request)
        assert response.status == "error"
        assert "failed" in response.error.lower()


class TestServerCacheParity:
    def test_semantic_cache_identity_isolates_server_and_sampler_without_raw_prompt_or_salt(self) -> None:
        request = InferenceRequest(
            model_id="qwen",
            prompt="raw prompt text",
            system_prompt="stable system prompt",
            temperature=0.1,
        )
        payload = {
            "model": "qwen",
            "messages": [
                {"role": "system", "content": "stable system prompt"},
                {"role": "user", "content": "raw prompt text"},
            ],
            "temperature": 0.1,
            "cache_salt": "super-secret",
        }

        key_a, context_a, identity_a = _semantic_cache_identity(
            request=request,
            provider_type="vllm",
            provider_name="vllm-test",
            provider_label="vllm",
            api_base="http://localhost:8000",
            gpu_only=True,
            payload=payload,
            raw_model_entry={"id": "qwen", "owned_by": "rev-a"},
            extra_config={"cache_namespace": "team-a", "vllm_engine_args_hash": "engine-a"},
        )
        key_b, context_b, _identity_b = _semantic_cache_identity(
            request=request,
            provider_type="vllm",
            provider_name="vllm-test",
            provider_label="vllm",
            api_base="http://localhost:9000",
            gpu_only=True,
            payload=payload,
            raw_model_entry={"id": "qwen", "owned_by": "rev-a"},
            extra_config={"cache_namespace": "team-a", "vllm_engine_args_hash": "engine-a"},
        )
        payload_with_different_sampler = dict(payload, temperature=0.2)
        _key_c, context_c, _identity_c = _semantic_cache_identity(
            request=request,
            provider_type="vllm",
            provider_name="vllm-test",
            provider_label="vllm",
            api_base="http://localhost:8000",
            gpu_only=True,
            payload=payload_with_different_sampler,
            raw_model_entry={"id": "qwen", "owned_by": "rev-a"},
            extra_config={"cache_namespace": "team-a", "vllm_engine_args_hash": "engine-a"},
        )

        assert key_a != key_b
        assert context_a != context_b
        assert context_a != context_c
        assert identity_a["payload"]["messages"][1]["content"] == "<query>"
        assert "raw prompt text" not in context_a
        assert "super-secret" not in context_a

    @patch("vetinari.adapters.openai_server_adapter._httpx")
    @patch("vetinari.adapters.openai_server_adapter._get_semantic_cache")
    def test_vllm_semantic_cache_hit_skips_http_post(
        self,
        mock_get_cache: MagicMock,
        mock_httpx: MagicMock,
    ) -> None:
        fake_cache = FakeSemanticCache(hit="cached answer")
        mock_get_cache.return_value = fake_cache
        config = ProviderConfig(
            provider_type=ProviderType.VLLM,
            name="vllm-cache",
            endpoint="http://localhost:8000",
            extra_config={
                "cache_salt": "super-secret",
                "gpu_only": True,
                "semantic_cache_enabled": True,
                "vllm_engine_args_hash": "engine-a",
            },
        )
        adapter = OpenAIServerAdapter(config)
        adapter._raw_model_entries["qwen"] = {"id": "qwen", "owned_by": "rev-a"}

        response = adapter.infer(InferenceRequest(model_id="qwen", prompt="hello"))

        assert response.status == "ok"
        assert response.output == "cached answer"
        assert response.metadata["cache_hit"] is True
        assert fake_cache.get_calls
        assert fake_cache.get_calls[0]["model_id"].startswith("vllm:vllm-cache:")
        assert "super-secret" not in str(fake_cache.get_calls[0]["system_prompt"])
        mock_httpx.return_value.post.assert_not_called()

    @patch("vetinari.adapters.openai_server_adapter._httpx")
    @patch("vetinari.adapters.openai_server_adapter._get_semantic_cache")
    def test_vllm_cache_miss_sends_cache_salt_and_stores_result(
        self,
        mock_get_cache: MagicMock,
        mock_httpx: MagicMock,
    ) -> None:
        fake_cache = FakeSemanticCache()
        mock_get_cache.return_value = fake_cache
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "model": "qwen",
            "system_fingerprint": "fp-test",
            "choices": [{"message": {"content": "fresh answer"}}],
            "usage": {"total_tokens": 8, "prompt_tokens": 3, "completion_tokens": 5},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_httpx.return_value.post.return_value = mock_resp

        config = ProviderConfig(
            provider_type=ProviderType.VLLM,
            name="vllm-cache",
            endpoint="http://localhost:8000",
            extra_config={
                "cache_salt": "super-secret",
                "gpu_only": True,
                "semantic_cache_enabled": True,
                "vllm_engine_args_hash": "engine-a",
            },
        )
        adapter = OpenAIServerAdapter(config)
        adapter._raw_model_entries["qwen"] = {"id": "qwen", "owned_by": "rev-a"}

        response = adapter.infer(InferenceRequest(model_id="qwen", prompt="hello"))

        payload = mock_httpx.return_value.post.call_args.kwargs["json"]
        assert payload["cache_salt"] == "super-secret"
        assert response.status == "ok"
        assert response.metadata["cache_hit"] is False
        assert response.metadata["system_fingerprint"] == "fp-test"
        assert response.metadata["server_cache_controls"]["cache_salt"]["sha256"]
        assert fake_cache.put_calls
        assert "super-secret" not in str(fake_cache.put_calls[0]["system_prompt"])

    @patch("vetinari.adapters.openai_server_adapter._httpx")
    @patch("vetinari.adapters.openai_server_adapter._get_semantic_cache")
    def test_nim_tracks_server_cache_provenance_without_sending_vllm_cache_fields(
        self,
        mock_get_cache: MagicMock,
        mock_httpx: MagicMock,
    ) -> None:
        fake_cache = FakeSemanticCache()
        mock_get_cache.return_value = fake_cache
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "model": "nim-model",
            "choices": [{"message": {"content": "nim answer"}}],
            "usage": {"total_tokens": 4, "prompt_tokens": 2, "completion_tokens": 2},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_httpx.return_value.post.return_value = mock_resp

        config = ProviderConfig(
            provider_type=ProviderType.NIM,
            name="nim-cache",
            endpoint="http://localhost:8001",
            extra_config={
                "cache_salt": "ignored-for-nim",
                "gpu_only": True,
                "kv_cache_reuse_enabled": True,
                "semantic_cache_enabled": True,
            },
        )
        adapter = OpenAIServerAdapter(config)

        response = adapter.infer(InferenceRequest(model_id="nim-model", prompt="hello"))

        payload = mock_httpx.return_value.post.call_args.kwargs["json"]
        assert "cache_salt" not in payload
        assert response.status == "ok"
        assert response.metadata["provider_label"] == "nim"
        assert response.metadata["server_provenance_sha256"]
        assert fake_cache.put_calls
        assert "kv_cache_reuse_enabled" in str(fake_cache.put_calls[0]["system_prompt"])


# ── Capabilities Tests ────────────────────────────────────────────────────────


class TestCapabilities:
    @patch("vetinari.adapters.openai_server_adapter._httpx")
    def test_get_capabilities(self, mock_httpx: MagicMock, adapter: OpenAIServerAdapter) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"id": "qwen2.5-coder-7b", "owned_by": "qwen"}],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_httpx.return_value.get.return_value = mock_resp

        caps = adapter.get_capabilities()
        assert "qwen2.5-coder-7b" in caps
        assert "coding" in caps["qwen2.5-coder-7b"]


# ── Registry Integration Tests ───────────────────────────────────────────────


class TestRegistryIntegration:
    def test_vllm_registered(self) -> None:
        from vetinari.adapters.registry import AdapterRegistry

        assert ProviderType.VLLM in AdapterRegistry._adapter_classes
        assert AdapterRegistry._adapter_classes[ProviderType.VLLM] is OpenAIServerAdapter

    def test_nim_registered(self) -> None:
        from vetinari.adapters.registry import AdapterRegistry

        assert ProviderType.NIM in AdapterRegistry._adapter_classes
        assert AdapterRegistry._adapter_classes[ProviderType.NIM] is OpenAIServerAdapter

    def test_enum_values_exist(self) -> None:
        from vetinari.types import ModelProvider

        assert ModelProvider.VLLM.value == "vllm"
        assert ModelProvider.NIM.value == "nim"
