"""Tests for the llama-cpp-python local inference adapter.

Covers LlamaCppProviderAdapter and LocalInferenceAdapter functionality
with mocked llama_cpp to avoid requiring actual model files.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vetinari.adapters.base import InferenceRequest, ProviderConfig, ProviderType
from vetinari.adapters.llama_cpp_adapter import (
    SYSTEM_PROMPT_BOUNDARY,
    LlamaCppProviderAdapter,
    _estimate_memory_gb,
    _infer_capabilities,
    _infer_context_window,
    _model_id_from_path,
)
from vetinari.adapters.llama_cpp_local_adapter import LocalInferenceAdapter
from vetinari.exceptions import ConfigurationError


@pytest.fixture(autouse=True)
def _disable_semantic_cache_side_effects():
    """Keep adapter tests local by disabling semantic-cache embedder imports."""
    with patch("vetinari.adapters.llama_cpp_adapter._get_semantic_cache_fn", return_value=lambda: None):
        yield


def _make_config(
    models_dir: str = "./models",
    gpu_layers: int = -1,
    context_length: int = 8192,
    memory_budget_gb: int = 32,
) -> ProviderConfig:
    """Build a LOCAL ProviderConfig using this module's ProviderType binding.

    Uses the file-level ``ProviderType`` import rather than the factory in
    ``tests.factories`` so that sys.modules isolation (which swaps the
    vetinari module objects between test files) does not cause enum identity
    mismatches inside ``LlamaCppProviderAdapter.__init__``.

    Args:
        models_dir: Path to the directory containing GGUF model files.
        gpu_layers: Number of layers to offload to GPU (-1 = all).
        context_length: Maximum context window size in tokens.
        memory_budget_gb: VRAM budget in gigabytes.

    Returns:
        A ProviderConfig configured for local llama-cpp-python inference.
    """
    return ProviderConfig(
        name="test-local",
        provider_type=ProviderType.LOCAL,
        endpoint="local",
        memory_budget_gb=memory_budget_gb,
        extra_config={
            "models_dir": models_dir,
            "gpu_layers": str(gpu_layers),
            "context_length": str(context_length),
        },
    )


# ── Utility function tests ──────────────────────────────────────────────────


class TestCapabilityInference:
    """Test capability inference from model names."""

    @pytest.mark.parametrize(
        ("model_name", "expected_caps"),
        [
            ("qwen2.5-coder-7b", ["coding", "general"]),
            ("qwen3-vl-32b", ["vision"]),
            ("qwen3-30b", ["reasoning"]),
            ("heretic-uncensored-thinking", ["uncensored", "reasoning"]),
            ("unknown-model-xyz", ["general"]),
        ],
    )
    def test_infer_capabilities(self, model_name, expected_caps):
        caps = _infer_capabilities(model_name)
        for cap in expected_caps:
            assert cap in caps


class TestContextWindowInference:
    """Test context window inference from model names."""

    def test_llama3_base_is_8k(self):
        # Plain Llama 3 (8B, 70B) — 8k context window, not 128k
        assert _infer_context_window("llama-3-8b") == 8192
        assert _infer_context_window("llama3-70b") == 8192

    def test_llama31_is_128k(self):
        # Llama 3.1 supports 128k
        assert _infer_context_window("llama-3.1-8b") == 131072
        assert _infer_context_window("llama-3.1-70b") == 131072

    def test_llama32_is_128k(self):
        # Llama 3.2 supports 128k
        assert _infer_context_window("llama-3.2-1b") == 131072
        assert _infer_context_window("llama-3.2-3b") == 131072

    def test_llama33_is_128k(self):
        # Llama 3.3 supports 128k
        assert _infer_context_window("llama-3.3-70b") == 131072

    def test_llama3_explicit_128k_variant(self):
        # Any Llama 3 model with "128k" in its name gets 128k context
        assert _infer_context_window("llama3-128k") == 131072

    def test_qwen25_context(self):
        assert _infer_context_window("qwen2.5-72b") == 32768

    def test_unknown_model_default(self):
        # Should return the default context length
        ctx = _infer_context_window("totally-unknown-model")
        assert ctx > 0


class TestMemoryEstimation:
    """Test VRAM estimation from file size."""

    def test_estimate_from_file_size(self, tmp_path):
        # Create a fake 4GB GGUF file
        fake_model = tmp_path / "model.gguf"
        fake_model.write_bytes(b"\x00" * (4 * 1024 * 1024 * 1024))
        estimate = _estimate_memory_gb(fake_model)
        assert 4.0 < estimate < 5.0  # 4GB * 1.1 overhead factor

    def test_small_model(self, tmp_path):
        fake_model = tmp_path / "tiny.gguf"
        fake_model.write_bytes(b"\x00" * (512 * 1024 * 1024))  # 512 MB
        estimate = _estimate_memory_gb(fake_model)
        assert estimate < 1.0


class TestModelIdFromPath:
    """Test model ID extraction from file paths."""

    def test_simple_name(self):
        assert _model_id_from_path(Path("qwen2.5-72b.gguf")) == "qwen2.5-72b"

    def test_nested_path(self):
        assert _model_id_from_path(Path("/models/large/llama-3.3-70b.gguf")) == "llama-3.3-70b"


# ── LlamaCppProviderAdapter tests ────────────────────────────────────────────


class TestLlamaCppProviderAdapter:
    """Test the main provider adapter."""

    def test_init_wrong_provider_type(self):
        cfg = ProviderConfig(
            name="bad",
            provider_type=ProviderType.OPENAI,
            endpoint="local",
        )
        with pytest.raises(ConfigurationError, match=r"ProviderType\.LOCAL"):
            LlamaCppProviderAdapter(cfg)

    def test_init_correct_provider_type(self):
        cfg = _make_config()
        adapter = LlamaCppProviderAdapter(cfg)
        assert adapter.provider_type == ProviderType.LOCAL

    def test_discover_models_empty_dir(self, tmp_path):
        cfg = _make_config(models_dir=str(tmp_path))
        adapter = LlamaCppProviderAdapter(cfg)
        models = adapter.discover_models()
        assert models == []

    def test_discover_models_with_gguf_files(self, tmp_path):
        # Create fake GGUF files
        (tmp_path / "qwen2.5-7b.gguf").write_bytes(b"\x00" * 1024)
        (tmp_path / "llama-3-8b.gguf").write_bytes(b"\x00" * 2048)

        cfg = _make_config(models_dir=str(tmp_path))
        adapter = LlamaCppProviderAdapter(cfg)
        models = adapter.discover_models()

        assert len(models) == 2
        ids = {m.id for m in models}
        assert "qwen2.5-7b" in ids
        assert "llama-3-8b" in ids

        # All models should have local provider
        for m in models:
            assert m.provider == "local"
            assert "general" in m.capabilities
            assert m.free_tier is True

    def test_discover_models_nonexistent_dir(self):
        cfg = _make_config(models_dir="/nonexistent/path")
        adapter = LlamaCppProviderAdapter(cfg)
        models = adapter.discover_models()
        assert models == []

    @patch("vetinari.adapters.llama_cpp_adapter._LLAMA_CPP_AVAILABLE", True)
    @patch("vetinari.adapters.llama_cpp_model_cache.llama_cpp")
    @patch("vetinari.adapters.llama_cpp_adapter.llama_cpp")
    def test_health_check_healthy(self, mock_llama_cpp, mock_cache_lc, tmp_path):
        (tmp_path / "model.gguf").write_bytes(b"\x00" * 1024)
        mock_llama_cpp.llama_supports_gpu_offload.return_value = True

        cfg = _make_config(models_dir=str(tmp_path))
        adapter = LlamaCppProviderAdapter(cfg)
        health = adapter.health_check()

        assert health["healthy"] is True
        assert health["gpu_offload"] is True

    @patch("vetinari.adapters.llama_cpp_adapter._LLAMA_CPP_AVAILABLE", False)
    def test_health_check_no_llama_cpp(self):
        cfg = _make_config()
        adapter = LlamaCppProviderAdapter(cfg)
        health = adapter.health_check()

        assert health["healthy"] is False
        assert "not installed" in health["reason"]

    @patch("vetinari.adapters.llama_cpp_adapter._LLAMA_CPP_AVAILABLE", False)
    def test_infer_no_llama_cpp(self):
        cfg = _make_config()
        adapter = LlamaCppProviderAdapter(cfg)
        req = InferenceRequest(model_id="test", prompt="hello")
        resp = adapter.infer(req)

        assert resp.status == "error"
        assert "not installed" in resp.error

    @patch("vetinari.adapters.llama_cpp_adapter._LLAMA_CPP_AVAILABLE", True)
    @patch("vetinari.adapters.llama_cpp_model_cache.llama_cpp")
    @patch("vetinari.adapters.llama_cpp_adapter.llama_cpp")
    def test_infer_success(self, mock_llama_cpp, mock_cache_lc, tmp_path):
        # Setup fake model file
        model_path = tmp_path / "test-model.gguf"
        model_path.write_bytes(b"\x00" * 1024)

        # Mock Llama instance — wire both adapter and model cache mocks
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hello, world!"}}],
            "usage": {"total_tokens": 42},
        }
        mock_llama_cpp.Llama.return_value = mock_llm
        mock_cache_lc.Llama.return_value = mock_llm

        cfg = _make_config(models_dir=str(tmp_path))
        adapter = LlamaCppProviderAdapter(cfg)
        adapter.discover_models()

        req = InferenceRequest(
            model_id="test-model",
            prompt="Say hello",
            system_prompt="You are helpful",
            max_tokens=100,
            temperature=0.5,
        )
        resp = adapter.infer(req)

        assert resp.status == "ok"
        assert resp.output == "Hello, world!"
        assert resp.tokens_used == 42
        assert resp.latency_ms >= 0

    @patch("vetinari.adapters.llama_cpp_adapter._LLAMA_CPP_AVAILABLE", True)
    @patch("vetinari.adapters.llama_cpp_model_cache.llama_cpp")
    @patch("vetinari.adapters.llama_cpp_adapter.llama_cpp")
    def test_infer_rejects_untrusted_embedded_chat_template(self, mock_llama_cpp, mock_cache_lc, tmp_path):
        model_path = tmp_path / "bad-template-model.gguf"
        model_path.write_bytes(b"\x00" * 1024)

        mock_llm = MagicMock()
        mock_llm.metadata = {
            "tokenizer.chat_template": "{{ cycler.__init__.__globals__.os.popen('id').read() }}",
        }
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "should not run"}}],
            "usage": {"total_tokens": 1},
        }
        mock_llama_cpp.Llama.return_value = mock_llm
        mock_cache_lc.Llama.return_value = mock_llm

        cfg = _make_config(models_dir=str(tmp_path))
        adapter = LlamaCppProviderAdapter(cfg)
        adapter.discover_models()

        resp = adapter.infer(InferenceRequest(model_id="bad-template-model", prompt="hello"))

        assert resp.status == "error"
        assert "Untrusted GGUF chat template" in resp.error
        assert mock_llm.create_chat_completion.call_count == 0

    @patch("vetinari.adapters.llama_cpp_adapter._LLAMA_CPP_AVAILABLE", True)
    @patch("vetinari.adapters.llama_cpp_model_cache.llama_cpp")
    @patch("vetinari.adapters.llama_cpp_adapter.llama_cpp")
    def test_model_caching(self, mock_llama_cpp, mock_cache_lc, tmp_path):
        model_path = tmp_path / "cached-model.gguf"
        model_path.write_bytes(b"\x00" * 1024)

        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"total_tokens": 1},
        }
        mock_llama_cpp.Llama.return_value = mock_llm
        mock_cache_lc.Llama = mock_llama_cpp.Llama

        cfg = _make_config(models_dir=str(tmp_path))
        adapter = LlamaCppProviderAdapter(cfg)
        adapter.discover_models()

        req = InferenceRequest(model_id="cached-model", prompt="test")

        # First call loads the model
        adapter.infer(req)
        assert mock_llama_cpp.Llama.call_count == 1

        # Second call should reuse cached model
        adapter.infer(req)
        assert mock_llama_cpp.Llama.call_count == 1  # Not called again

    @patch("vetinari.adapters.llama_cpp_adapter._LLAMA_CPP_AVAILABLE", True)
    @patch("vetinari.adapters.llama_cpp_model_cache.llama_cpp")
    @patch("vetinari.adapters.llama_cpp_adapter.llama_cpp")
    def test_vram_budget_eviction(self, mock_llama_cpp, mock_cache_lc, tmp_path):
        # Create two fake GGUF files (tiny, but _estimate_memory_gb is patched
        # to report 20 GB each so eviction logic is exercised)
        (tmp_path / "model-a.gguf").write_bytes(b"\x00" * 1024)
        (tmp_path / "model-b.gguf").write_bytes(b"\x00" * 1024)

        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"total_tokens": 1},
        }
        mock_llama_cpp.Llama.return_value = mock_llm
        mock_cache_lc.Llama = mock_llama_cpp.Llama

        with (
            # Disable semantic cache — when embedder is available, semantically
            # similar keys ("model-a:test" vs "model-b:test") cause cross-model
            # cache hits, bypassing model loading entirely.
            patch(
                "vetinari.adapters.llama_cpp_adapter._get_semantic_cache_fn",
                return_value=lambda: None,
            ),
            # Patch memory estimation so fake 1KB GGUF files report 20 GB each,
            # exercising the eviction path that depends on realistic VRAM usage.
            patch(
                "vetinari.adapters.llama_cpp_model_cache._estimate_memory_gb",
                return_value=20.0,
            ),
        ):
            # Budget of 32 GB, cpu_offload disabled so eviction is purely
            # VRAM-based. With cpu_offload enabled (default), partial GPU offload
            # reduces effective VRAM so both models fit without eviction.
            cfg = _make_config(models_dir=str(tmp_path), memory_budget_gb=32)
            adapter = LlamaCppProviderAdapter(cfg)
            adapter._cpu_offload_enabled = False
            adapter.discover_models()

            # Load first model (20 GB estimated by patched _estimate_memory_gb)
            adapter.infer(InferenceRequest(model_id="model-a", prompt="test"))
            assert "model-a" in adapter.get_loaded_models()

            # Load second model — 20+20=40 > 32 budget, should evict first
            adapter.infer(InferenceRequest(model_id="model-b", prompt="test"))
            assert "model-b" in adapter.get_loaded_models()
            assert "model-a" not in adapter.get_loaded_models(), "model-a should have been evicted"

    def test_unload_model(self, tmp_path):
        cfg = _make_config(models_dir=str(tmp_path))
        adapter = LlamaCppProviderAdapter(cfg)

        # Not loaded — returns False
        assert adapter.unload_model("nonexistent") is False

    def test_get_capabilities_triggers_discovery(self, tmp_path):
        (tmp_path / "test.gguf").write_bytes(b"\x00" * 1024)
        cfg = _make_config(models_dir=str(tmp_path))
        adapter = LlamaCppProviderAdapter(cfg)

        caps = adapter.get_capabilities()
        assert "test" in caps
        assert "general" in caps["test"]

    def test_get_vram_usage_empty(self):
        cfg = _make_config()
        adapter = LlamaCppProviderAdapter(cfg)
        assert adapter.get_vram_usage() == 0.0


# ── LocalInferenceAdapter tests ──────────────────────────────────────────────


class TestLocalInferenceAdapter:
    """Test the convenience wrapper."""

    @patch("vetinari.adapters.llama_cpp_adapter._LLAMA_CPP_AVAILABLE", True)
    @patch("vetinari.adapters.llama_cpp_model_cache.llama_cpp")
    @patch("vetinari.adapters.llama_cpp_adapter.llama_cpp")
    def test_chat(self, mock_llama_cpp, mock_cache_lc, tmp_path):
        model_path = tmp_path / "chat-model.gguf"
        model_path.write_bytes(b"\x00" * 1024)

        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hi there!"}}],
            "usage": {"total_tokens": 10},
        }
        mock_llama_cpp.Llama.return_value = mock_llm
        mock_cache_lc.Llama.return_value = mock_llm

        adapter = LocalInferenceAdapter(models_dir=str(tmp_path))
        result = adapter.chat("chat-model", "You are helpful", "Hello")

        assert result["output"] == "Hi there!"
        assert result["status"] == "ok"
        assert result["tokens_used"] == 10

    @patch("vetinari.adapters.llama_cpp_adapter._LLAMA_CPP_AVAILABLE", False)
    def test_is_healthy_no_llama_cpp(self, tmp_path):
        adapter = LocalInferenceAdapter(models_dir=str(tmp_path))
        assert adapter.is_healthy() is False

    @patch("vetinari.adapters.llama_cpp_adapter._LLAMA_CPP_AVAILABLE", True)
    @patch("vetinari.adapters.llama_cpp_model_cache.llama_cpp")
    @patch("vetinari.adapters.llama_cpp_adapter.llama_cpp")
    def test_list_loaded_models(self, mock_llama_cpp, mock_cache_lc, tmp_path):
        (tmp_path / "list-model.gguf").write_bytes(b"\x00" * 1024)
        mock_llama_cpp.llama_supports_gpu_offload.return_value = True

        adapter = LocalInferenceAdapter(models_dir=str(tmp_path))
        models = adapter.list_loaded_models()
        assert len(models) == 1
        assert models[0]["id"] == "list-model"

    def test_provider_property(self, tmp_path):
        adapter = LocalInferenceAdapter(models_dir=str(tmp_path))
        assert adapter.provider is not None
        assert adapter.provider.config.extra_config["models_dir"] == str(tmp_path)


# ── Session 15: KV cache quant config ──────────────────────────────────────


class TestKVCacheTypeConfig:
    """KV cache quantization type is read from settings and can be overridden."""

    @patch("vetinari.adapters.llama_cpp_adapter._LLAMA_CPP_AVAILABLE", True)
    @patch("vetinari.adapters.llama_cpp_adapter.get_settings")
    def test_defaults_from_settings(self, mock_get_settings, tmp_path):
        """Adapter reads cache_type_k/v from global settings when not in extra_config."""
        mock_settings = MagicMock()
        mock_settings.local_cache_type_k = "f16"
        mock_settings.local_cache_type_v = "f16"
        mock_get_settings.return_value = mock_settings

        cfg = _make_config(models_dir=str(tmp_path))
        adapter = LlamaCppProviderAdapter(cfg)

        assert adapter._cache_type_k == "f16"
        assert adapter._cache_type_v == "f16"

    @patch("vetinari.adapters.llama_cpp_adapter._LLAMA_CPP_AVAILABLE", True)
    @patch("vetinari.adapters.llama_cpp_adapter.get_settings")
    def test_extra_config_overrides_settings(self, mock_get_settings, tmp_path):
        """extra_config cache_type_k/v take precedence over global settings."""
        mock_settings = MagicMock()
        mock_settings.local_cache_type_k = "f16"
        mock_settings.local_cache_type_v = "f16"
        mock_get_settings.return_value = mock_settings

        cfg = ProviderConfig(
            name="test-local",
            provider_type=ProviderType.LOCAL,
            endpoint="local",
            memory_budget_gb=32,
            extra_config={
                "models_dir": str(tmp_path),
                "gpu_layers": "-1",
                "context_length": "8192",
                "cache_type_k": "q4_0",
                "cache_type_v": "q8_0",
            },
        )
        adapter = LlamaCppProviderAdapter(cfg)

        assert adapter._cache_type_k == "q4_0"
        assert adapter._cache_type_v == "q8_0"

    @patch("vetinari.adapters.llama_cpp_adapter._LLAMA_CPP_AVAILABLE", True)
    @patch("vetinari.adapters.llama_cpp_model_cache.llama_cpp")
    @patch("vetinari.adapters.llama_cpp_adapter.llama_cpp")
    @patch("vetinari.adapters.llama_cpp_adapter.get_settings")
    def test_kv_quant_passed_to_llama_constructor(self, mock_get_settings, mock_llama_cpp, mock_cache_lc, tmp_path):
        """The resolved GGML type constants are forwarded to the Llama constructor."""
        mock_settings = MagicMock()
        mock_settings.local_cache_type_k = "q8_0"
        mock_settings.local_cache_type_v = "q8_0"
        mock_get_settings.return_value = mock_settings

        # Simulate llama_cpp exposing the GGML type constants
        mock_cache_lc.GGML_TYPE_Q8_0 = "GGML_Q8_0_SENTINEL"
        mock_cache_lc.GGML_TYPE_F16 = "GGML_F16_SENTINEL"

        model_path = tmp_path / "kv-model.gguf"
        model_path.write_bytes(b"\x00" * 1024)

        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "warm"}}],
            "usage": {"total_tokens": 1},
        }
        mock_llama_cpp.Llama.return_value = mock_llm
        mock_cache_lc.Llama.return_value = mock_llm

        cfg = _make_config(models_dir=str(tmp_path))
        adapter = LlamaCppProviderAdapter(cfg)
        adapter._get_or_load_model("kv-model", tmp_path / "kv-model.gguf")

        call_kwargs = mock_cache_lc.Llama.call_args[1]
        assert call_kwargs.get("type_k") == "GGML_Q8_0_SENTINEL"
        assert call_kwargs.get("type_v") == "GGML_Q8_0_SENTINEL"


# ── Session 15: System prompt KV cache boundary ────────────────────────────


class TestSystemPromptBoundary:
    """SYSTEM_PROMPT_BOUNDARY splits stable prefix from dynamic context for KV reuse."""

    def test_constant_value(self):
        """The marker string is the agreed sentinel value."""
        assert SYSTEM_PROMPT_BOUNDARY == "<<<CONTEXT_BOUNDARY>>>"

    def test_prefix_extracted_before_boundary(self):
        """Text before the boundary is the stable KV cache prefix."""
        system_prompt = "You are a helpful agent." + SYSTEM_PROMPT_BOUNDARY + "\nUser context here."
        prefix = system_prompt.split(SYSTEM_PROMPT_BOUNDARY, 1)[0]
        assert prefix == "You are a helpful agent."

    def test_no_boundary_uses_full_prompt(self):
        """When no boundary marker is present, the full prompt is the cache key."""
        system_prompt = "You are a helpful agent."
        if SYSTEM_PROMPT_BOUNDARY in system_prompt:
            prefix = system_prompt.split(SYSTEM_PROMPT_BOUNDARY, 1)[0]
        else:
            prefix = system_prompt
        assert prefix == system_prompt

    def test_empty_prefix_when_boundary_at_start(self):
        """A boundary at the start yields an empty stable prefix."""
        system_prompt = SYSTEM_PROMPT_BOUNDARY + "dynamic only"
        prefix = system_prompt.split(SYSTEM_PROMPT_BOUNDARY, 1)[0]
        assert prefix == ""


# ── Session 15: Model warm-up ──────────────────────────────────────────────


class TestWarmUpModel:
    """Warm-up inference runs after model load and is non-fatal on failure."""

    @patch("vetinari.adapters.llama_cpp_adapter._LLAMA_CPP_AVAILABLE", True)
    @patch("vetinari.adapters.llama_cpp_model_cache.llama_cpp")
    @patch("vetinari.adapters.llama_cpp_adapter.llama_cpp")
    @patch("vetinari.adapters.llama_cpp_adapter.get_settings")
    def test_warm_up_called_after_load(self, mock_get_settings, mock_llama_cpp, mock_cache_lc, tmp_path):
        """create_chat_completion is called once with max_tokens=1 during warm-up."""
        mock_settings = MagicMock()
        mock_settings.local_cache_type_k = "f16"
        mock_settings.local_cache_type_v = "f16"
        mock_get_settings.return_value = mock_settings

        model_path = tmp_path / "warmup-model.gguf"
        model_path.write_bytes(b"\x00" * 1024)

        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"total_tokens": 1},
        }
        mock_llama_cpp.Llama.return_value = mock_llm
        mock_cache_lc.Llama.return_value = mock_llm

        cfg = _make_config(models_dir=str(tmp_path))
        adapter = LlamaCppProviderAdapter(cfg)
        adapter._get_or_load_model("warmup-model", tmp_path / "warmup-model.gguf")

        # warm-up fires create_chat_completion exactly once with max_tokens=1
        calls = mock_llm.create_chat_completion.call_args_list
        warmup_calls = [c for c in calls if c[1].get("max_tokens") == 1]
        assert len(warmup_calls) == 1
        assert warmup_calls[0][1].get("temperature") == 0.0

    @patch("vetinari.adapters.llama_cpp_adapter._LLAMA_CPP_AVAILABLE", True)
    @patch("vetinari.adapters.llama_cpp_model_cache.llama_cpp")
    @patch("vetinari.adapters.llama_cpp_adapter.llama_cpp")
    @patch("vetinari.adapters.llama_cpp_adapter.get_settings")
    def test_warm_up_failure_is_non_fatal(self, mock_get_settings, mock_llama_cpp, mock_cache_lc, tmp_path):
        """A warm-up exception does not prevent the model from loading."""
        mock_settings = MagicMock()
        mock_settings.local_cache_type_k = "f16"
        mock_settings.local_cache_type_v = "f16"
        mock_get_settings.return_value = mock_settings

        model_path = tmp_path / "fail-warmup.gguf"
        model_path.write_bytes(b"\x00" * 1024)

        mock_llm = MagicMock()
        mock_llm.create_chat_completion.side_effect = RuntimeError("GPU OOM")
        mock_llama_cpp.Llama.return_value = mock_llm
        mock_cache_lc.Llama.return_value = mock_llm

        cfg = _make_config(models_dir=str(tmp_path))
        adapter = LlamaCppProviderAdapter(cfg)
        # Should not raise despite warm-up failure — model is still registered
        adapter._get_or_load_model("fail-warmup", model_path)
        assert "fail-warmup" in adapter.get_loaded_models()
