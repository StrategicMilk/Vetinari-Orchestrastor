"""Native local inference adapter using llama-cpp-python.

Provides in-process GGUF model loading and inference via llama.cpp,
eliminating the need for an external inference server. This is the
primary local inference backend for Vetinari.

Features:
- Auto-discovery of .gguf files from a configurable directory
- In-process model loading with GPU offload
- LRU model cache with VRAM budget enforcement (via LlamaCppModelCacheMixin)
- Capability inference from model filenames
- Streaming and non-streaming inference
- Semantic caching and KV state caching for repeated prompts

Split layout:
    llama_cpp_model_info.py      — constants, helpers, _LoadedModel dataclass
    llama_cpp_model_cache.py     — LlamaCppModelCacheMixin (VRAM budget, LRU, loading)
    llama_cpp_local_adapter.py   — LocalInferenceAdapter convenience wrapper
    llama_cpp_lazy_imports.py    — lazy-import getters for vetinari sub-packages
    llama_cpp_adapter.py         — LlamaCppProviderAdapter (this file) + re-exports
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from vetinari.config.settings import get_settings
from vetinari.constants import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_GPU_LAYERS,
    INFERENCE_STATUS_OK,
)
from vetinari.exceptions import ConfigurationError
from vetinari.utils.lazy_import import lazy_import

from . import llama_cpp_adapter_helpers as _adapter_helpers
from .base import InferenceRequest, InferenceResponse, ModelInfo, ProviderAdapter, ProviderConfig, ProviderType
from .llama_cpp_adapter_discovery import _LlamaCppDiscoveryMixin
from .llama_cpp_lazy_imports import (
    _get_draft_pair_resolver_fn,
    _get_hash_system_prompt_fn,
    _get_kv_state_cache_fn,
    _get_model_selector_fn,
    _get_semantic_cache_fn,
)
from .llama_cpp_model_cache import LlamaCppModelCacheMixin
from .llama_cpp_model_info import (
    _EMPTY_CHOICES_FALLBACK,
    _EMPTY_DELTA_FALLBACK,
    DEFAULT_RAM_BUDGET_GB,
    _estimate_memory_gb,
    _infer_capabilities,
    _infer_context_window,
    _model_id_from_path,
)
from .speculative_decoding import SpeculativeDecodingConfig, detect_speculative_capability

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_BOUNDARY = _adapter_helpers.SYSTEM_PROMPT_BOUNDARY
ADAPTER_CACHE_VERSION = _adapter_helpers.ADAPTER_CACHE_VERSION
_artifact_sha256 = _adapter_helpers._artifact_sha256
_build_completion_kwargs = _adapter_helpers._build_completion_kwargs
_chat_template_validation_metadata = _adapter_helpers._chat_template_validation_metadata
_extract_embedded_chat_template = _adapter_helpers._extract_embedded_chat_template
_semantic_cache_identity = _adapter_helpers._semantic_cache_identity
_validate_loaded_chat_template = _adapter_helpers._validate_loaded_chat_template

# ── Optional llama-cpp-python import ─────────────────────────────────────────
llama_cpp, _LLAMA_CPP_AVAILABLE = lazy_import("llama_cpp")  # type: ignore[assignment]

# ── Lazy module-level imports for hot-path dependencies (avoids per-call import overhead) ──
_model_profiler_data, _ = lazy_import("vetinari.models.model_profiler_data")
_vram_manager_mod, _ = lazy_import("vetinari.models.vram_manager")

# VRAM cost of KV cache per token for each quantization type (bytes per token).
# Matches the values in vram_manager._KV_BYTES_PER_TOKEN — kept here so the
# inference hot-path does not need to import vram_manager for a simple dict lookup.
_KV_BYTES_PER_TOKEN: dict[str, int] = {
    "f16": 2048,  # 2 bytes/element — llama.cpp default
    "q8_0": 1024,  # 1 byte/element
    "q4_0": 512,  # 0.5 bytes/element
}


class LlamaCppProviderAdapter(_LlamaCppDiscoveryMixin, LlamaCppModelCacheMixin, ProviderAdapter):
    """Native local inference adapter using llama-cpp-python.

    Loads GGUF models directly into GPU/CPU memory and runs inference
    in-process via llama.cpp. No external server required.

    Inherits ``LlamaCppModelCacheMixin`` for VRAM-budget-aware model
    loading, per-model locking, LRU eviction, and background calibration.

    Configuration via ``ProviderConfig.extra_config``:
        - ``models_dir``: Directory to scan for .gguf files (default: ./models)
        - ``gpu_layers``: Number of layers to offload to GPU (-1 = all)
        - ``context_length``: Default context window size
        - ``ram_budget_gb``: CPU RAM available for offloaded layers (default: 30)
        - ``cpu_offload_enabled``: Allow partial GPU+CPU split loading (default: true)
    """

    def __init__(self, config: ProviderConfig):
        """Validate that this adapter is used only with LOCAL providers, then parse GPU and model-dir settings.

        Args:
            config: Provider configuration. Must have provider_type LOCAL.

        Raises:
            ConfigurationError: If provider_type is not LOCAL.
        """
        if config.provider_type != ProviderType.LOCAL:
            msg = f"LlamaCppProviderAdapter requires ProviderType.LOCAL, got {config.provider_type}"
            raise ConfigurationError(msg)
        super().__init__(config)

        extra = config.extra_config or {}
        from vetinari.constants import OPERATOR_MODELS_CACHE_DIR

        self._models_dir = Path(extra.get("models_dir", OPERATOR_MODELS_CACHE_DIR))
        self._gpu_layers = int(extra.get("gpu_layers", DEFAULT_GPU_LAYERS))
        self._default_context_length = int(extra.get("context_length", DEFAULT_CONTEXT_LENGTH))
        self._memory_budget_gb = config.memory_budget_gb
        # CPU RAM budget for layers offloaded from GPU; 0 disables CPU offload
        self._ram_budget_gb = float(extra.get("ram_budget_gb", DEFAULT_RAM_BUDGET_GB))
        # Whether to allow partial GPU + CPU split-loading for oversized models
        _cpu_offload_raw = extra.get("cpu_offload_enabled", "true")
        self._cpu_offload_enabled = str(_cpu_offload_raw).lower() not in ("false", "0", "no")

        # KV cache quantization — read from settings, allow per-adapter override via extra_config
        _settings = get_settings()
        self._cache_type_k: str = str(extra.get("cache_type_k", _settings.local_cache_type_k))
        self._cache_type_v: str = str(extra.get("cache_type_v", _settings.local_cache_type_v))

        # Shared lock protects the _loaded_models dict and _model_locks dict
        self._loaded_models: dict[str, Any] = {}
        self._registry_lock = threading.Lock()
        # Per-model locks allow concurrent loading of different models (5-10x throughput)
        self._model_locks: dict[str, threading.Lock] = {}
        self._discovered_models: list[ModelInfo] = []

        # Background calibration pool — max 1 concurrent calibration to avoid VRAM contention
        self._calibration_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="calibrate")

    def _get_speculative_config(self) -> SpeculativeDecodingConfig:
        """Read speculative decoding settings from application config.

        Consults ``VetinariSettings`` for the three speculative decoding fields
        (``speculative_decoding_enabled``, ``speculative_draft_model_id``,
        ``speculative_draft_n_tokens``) and returns a typed config object.

        Returns:
            ``SpeculativeDecodingConfig`` populated from settings.  All fields
            have safe defaults so the result is always valid.
        """
        _settings = get_settings()
        return SpeculativeDecodingConfig(
            enabled=_settings.speculative_decoding_enabled,
            draft_model_id=_settings.speculative_draft_model_id,
            draft_n_tokens=_settings.speculative_draft_n_tokens,
            use_prompt_lookup_fallback=True,  # Always allow PromptLookup fallback
        )

    def health_check(self) -> dict[str, Any]:
        """Check local inference health.

        Verifies llama-cpp-python is installed, GPU offload is available,
        and the models directory contains .gguf files.

        Returns:
            Health status dict with keys: healthy, reason, timestamp,
            and optionally gpu_offload and model_count.
        """
        timestamp = str(time.time())

        if not _LLAMA_CPP_AVAILABLE:
            return {
                "healthy": False,
                "reason": "llama-cpp-python is not installed",
                "timestamp": timestamp,
            }

        gpu_available = False
        try:
            if llama_cpp is not None:
                gpu_available = bool(llama_cpp.llama_supports_gpu_offload())
        except (ImportError, OSError, RuntimeError):
            logger.warning("GPU offload check failed", exc_info=True)

        if not self._models_dir.exists():
            return {
                "healthy": False,
                "reason": f"Models directory does not exist: {self._models_dir}",
                "timestamp": timestamp,
            }

        gguf_count = len(list(self._models_dir.rglob("*.gguf")))
        if gguf_count == 0:
            return {
                "healthy": False,
                "reason": f"No .gguf files found in {self._models_dir}",
                "timestamp": timestamp,
            }

        return {
            "healthy": True,
            "reason": f"OK: {gguf_count} models available, GPU offload={gpu_available}",
            "timestamp": timestamp,
            "gpu_offload": gpu_available,
            "model_count": gguf_count,
        }

    def _build_completion_kwargs(
        self,
        request: InferenceRequest,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build kwargs dict for llama_cpp create_chat_completion."""
        return _build_completion_kwargs(request, messages, stream=stream)

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference on a local GGUF model.

        Loads the model if not already cached. Enforces VRAM budget by
        evicting least-recently-used models when necessary.

        Args:
            request: Inference request with model_id, prompt, and parameters.

        Returns:
            InferenceResponse with output text, latency, and token usage.

        Raises:
            RuntimeError: Propagated only when an unexpected internal adapter
                error occurs before an ``InferenceResponse`` can be built.
        """
        self._emit_inference_started(request)
        start_time = time.time()

        if not _LLAMA_CPP_AVAILABLE:
            return InferenceResponse(
                model_id=request.model_id,
                output="",
                latency_ms=0,
                tokens_used=0,
                status="error",
                error="llama-cpp-python is not installed",
            )

        if not self._discovered_models:
            try:
                self.discover_models()
            except Exception:
                logger.warning("Lazy model discovery failed — no models may be available", exc_info=True)

        model_path, resolution_outcome = self._resolve_model_path_with_outcome(request.model_id)
        if model_path is None:
            return InferenceResponse(
                model_id=request.model_id,
                output="",
                latency_ms=0,
                tokens_used=0,
                status="error",
                error=f"Model not found: {request.model_id}",
            )

        model_id = _model_id_from_path(model_path)

        # Use model-family temperature matrix when the caller is using the
        # default (0.7) sentinel. This must happen before semantic-cache lookup
        # because temperature is output-affecting identity.
        if request.temperature == 0.7:
            try:
                _task_type = request.metadata.get("task_type", "general")
                if _model_profiler_data is None:
                    raise RuntimeError("model_profiler_data unavailable")
                _family = _model_profiler_data.detect_family(model_id)
                _quant = model_path.stem.lower()
                _profiled_temp = _model_profiler_data.get_temperature(_family, _task_type, _quant)
                request = dataclasses.replace(request, temperature=_profiled_temp)
                logger.debug(
                    "Temperature set from model profile: %s family=%s task=%s quant=%s -> %.3f",
                    model_id,
                    _family,
                    _task_type,
                    _quant,
                    _profiled_temp,
                )
            except Exception:
                logger.warning(
                    "get_temperature failed for %s — using request default temperature %.2f",
                    model_id,
                    request.temperature,
                )

        try:
            _cache = _get_semantic_cache_fn()()
            if _cache is not None:
                _cache_model_id, _cache_context = _semantic_cache_identity(
                    request,
                    model_path=model_path,
                    resolved_model_id=model_id,
                    resolution_outcome=resolution_outcome,
                )
                _cached = _cache.get(
                    request.prompt,
                    task_type=request.task_type or "",
                    model_id=_cache_model_id,
                    system_prompt=_cache_context,
                )
                if _cached is not None:
                    logger.debug("Semantic cache hit for resolved model %s", model_id)
                    return InferenceResponse(
                        model_id=model_id,
                        output=_cached,
                        latency_ms=0,
                        tokens_used=0,
                        status=INFERENCE_STATUS_OK,
                    )
        except Exception:
            logger.warning("Semantic cache unavailable — proceeding without cache", exc_info=True)

        try:
            llm = self._get_or_load_model(model_id, model_path)
        except Exception:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.exception("Failed to load model %s", model_id)
            return InferenceResponse(
                model_id=model_id,
                output="",
                latency_ms=latency_ms,
                tokens_used=0,
                status="error",
                error="Model load failed — check server logs for details",
            )

        template_validation = _validate_loaded_chat_template(model_id, llm)
        if template_validation is not None and (
            not template_validation.is_trusted or template_validation.fallback_used
        ):
            latency_ms = int((time.time() - start_time) * 1000)
            logger.warning(
                "Rejecting inference for %s due to untrusted embedded chat template",
                model_id,
            )
            return InferenceResponse(
                model_id=model_id,
                output="",
                latency_ms=latency_ms,
                tokens_used=0,
                status="error",
                error="Untrusted GGUF chat template rejected",
                metadata={"chat_template_validation": _chat_template_validation_metadata(template_validation)},
            )

        _kv_restored = False
        if request.system_prompt:
            try:
                _kv_cache = _get_kv_state_cache_fn()()
                # Hash only the stable prefix (agent role + tools) so the second call to the
                # same model reuses the KV state even when the dynamic context differs.
                _kv_prefix = (
                    request.system_prompt.split(SYSTEM_PROMPT_BOUNDARY, 1)[0]
                    if SYSTEM_PROMPT_BOUNDARY in request.system_prompt
                    else request.system_prompt
                )
                _prompt_hash = _get_hash_system_prompt_fn()(_kv_prefix)
                _saved_state = _kv_cache.get(model_id, _prompt_hash)
                if _saved_state is not None and hasattr(llm, "load_state"):
                    llm.load_state(_saved_state)
                    _kv_restored = True
                    logger.debug("KV state restored for %s (hash=%s)", model_id, _prompt_hash[:8])
            except Exception:
                logger.warning(
                    "KV state cache restore failed for %s — running without cached state", model_id, exc_info=True
                )

        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        # Use model-family temperature matrix when the caller is using the
        # default (0.7) sentinel — prevents hardcoded temperatures for every
        # model family and task type. Decision wiring uses ModelProfiler's
        # per-family temperature table. (Prior ADR-0066 citation here was
        # drift; ADR-0066 is the Flask-to-Litestar migration. No dedicated
        # ADR exists for this temperature-matrix wiring; flagged during
        # 2026-04-24 governance audit.)
        if request.temperature == 0.7:
            try:
                _task_type = request.metadata.get("task_type", "general")
                if _model_profiler_data is None:
                    raise RuntimeError("model_profiler_data unavailable")
                _family = _model_profiler_data.detect_family(model_id)
                _quant = model_path.stem.lower() if model_path else ""
                _profiled_temp = _model_profiler_data.get_temperature(_family, _task_type, _quant)
                # Mutate a copy so the original request is unchanged
                request = dataclasses.replace(request, temperature=_profiled_temp)
                logger.debug(
                    "Temperature set from model profile: %s family=%s task=%s quant=%s -> %.3f",
                    model_id,
                    _family,
                    _task_type,
                    _quant,
                    _profiled_temp,
                )
            except Exception:
                logger.warning(
                    "get_temperature failed for %s — using request default temperature %.2f",
                    model_id,
                    request.temperature,
                )

        # Speculative decoding capability gate: log whether draft acceleration is
        # active for this model.  The actual draft_model kwarg is injected at model-
        # load time (see LlamaCppModelCacheMixin._get_or_load_model).  Here we only
        # observe what is in use and optionally warn when config requested it but the
        # llama_cpp version cannot support it.
        _spec_cfg = self._get_speculative_config()
        if _spec_cfg.enabled:
            _cap = detect_speculative_capability(draft_model_id=_spec_cfg.draft_model_id)
            if not _cap.supported:
                logger.debug(
                    "Speculative decoding requested for %s but not supported by"
                    " this llama_cpp build (detection_method=%s) — using standard inference",
                    model_id,
                    _cap.detection_method,
                )
            elif _cap.has_draft_model:
                logger.debug(
                    "Speculative decoding active for %s with draft model %s",
                    model_id,
                    _cap.draft_model_id,
                )
            else:
                logger.debug(
                    "Speculative decoding enabled for %s; no draft model configured"
                    " — PromptLookupDecoding fallback will be used if available",
                    model_id,
                )
        else:
            logger.debug(
                "Speculative decoding disabled for %s — standard inference",
                model_id,
            )

        _completion_kwargs = self._build_completion_kwargs(request, messages)

        try:
            result = llm.create_chat_completion(**_completion_kwargs)

            choices = result.get("choices") or []
            output = choices[0]["message"]["content"] if choices else ""
            usage = result.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)

            # Extract logprob variance for inference confidence scoring
            _response_metadata: dict[str, Any] = {}
            if template_validation is not None:
                _response_metadata["chat_template_validation"] = _chat_template_validation_metadata(template_validation)
            if choices:
                _logprobs_data = choices[0].get("logprobs")
                if _logprobs_data and isinstance(_logprobs_data, dict):
                    _token_logprobs = _logprobs_data.get("token_logprobs")
                    if _token_logprobs and isinstance(_token_logprobs, list):
                        _valid_lps = [lp for lp in _token_logprobs if lp is not None]
                        if len(_valid_lps) >= 2:
                            _mean = sum(_valid_lps) / len(_valid_lps)
                            _variance = sum((lp - _mean) ** 2 for lp in _valid_lps) / len(_valid_lps)
                            # High variance = low confidence; convert to 0-1 scale
                            # Logprob variance of ~2.0+ means very uncertain
                            _confidence = max(0.0, min(1.0, 1.0 - (_variance / 3.0)))
                            _response_metadata["logprob_variance"] = round(_variance, 4)
                            _response_metadata["inference_confidence"] = round(_confidence, 4)
                            _response_metadata["token_logprobs"] = _valid_lps

            latency_ms = int((time.time() - start_time) * 1000)
            response = InferenceResponse(
                model_id=model_id,
                output=output,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                status=INFERENCE_STATUS_OK,
                metadata=_response_metadata,
            )

            # Update KV cache tracking in VRAMManager so VRAM budget stays accurate.
            # KV cache size = tokens_used * ~1 KB per token (q8_0 default), capped at
            # context_length.  This is an estimate; the true size depends on quant type
            # and architecture, but it is close enough for budget decisions.
            if tokens_used > 0:
                try:
                    _kv_bpt = _KV_BYTES_PER_TOKEN.get(self._cache_type_k, 2048)
                    _kv_gb = tokens_used * _kv_bpt / (1024**3)
                    if _vram_manager_mod is not None:
                        _vram_manager_mod.get_vram_manager().update_kv_cache(model_id, _kv_gb)
                except Exception:
                    logger.warning("update_kv_cache failed for %s — VRAM KV tracking may be stale", model_id)

            if request.system_prompt and not _kv_restored and hasattr(llm, "save_state"):
                try:
                    _kv_cache = _get_kv_state_cache_fn()()
                    # Use the same stable prefix hash as the restore path so the saved state
                    # can be reused on the next call regardless of dynamic context changes.
                    _save_prefix = (
                        request.system_prompt.split(SYSTEM_PROMPT_BOUNDARY, 1)[0]
                        if SYSTEM_PROMPT_BOUNDARY in request.system_prompt
                        else request.system_prompt
                    )
                    _save_hash = _get_hash_system_prompt_fn()(_save_prefix)
                    _kv_cache.put(model_id, _save_hash, llm.save_state())
                except Exception:
                    logger.warning("KV state save failed for %s — cache miss on next request", model_id, exc_info=True)

        except Exception:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.exception("Inference failed for model %s", model_id)
            response = InferenceResponse(
                model_id=model_id,
                output="",
                latency_ms=latency_ms,
                tokens_used=0,
                status="error",
                error="Inference failed — check server logs for details",
            )

        self._record_telemetry(request, response)

        # Record draft model acceptance for Thompson learning
        if response.status == INFERENCE_STATUS_OK:
            self._record_draft_acceptance(request, response, model_id)

        if response.status == INFERENCE_STATUS_OK and response.output:
            try:
                _cache = _get_semantic_cache_fn()()
                if _cache is not None:
                    _cache_model_id, _cache_context = _semantic_cache_identity(
                        request,
                        model_path=model_path,
                        resolved_model_id=model_id,
                        resolution_outcome=resolution_outcome,
                    )
                    _cache.put(request.prompt, response.output, model_id=_cache_model_id, system_prompt=_cache_context)
            except Exception:
                logger.warning("Semantic cache store failed — result will not be cached", exc_info=True)

        return response

    def _record_draft_acceptance(
        self,
        request: InferenceRequest,
        response: InferenceResponse,
        model_id: str,
    ) -> None:
        """Record speculative-decoding draft token acceptance for Thompson Sampling.

        Reads draft acceptance timings from the loaded model's last_timings dict and
        forwards per-token accept/reject events to the draft pair resolver and model
        selector for reward tracking.

        Args:
            request: The original inference request (used for task_type metadata).
            response: The completed inference response (unused; present for symmetry).
            model_id: The model ID whose timings should be read.
        """
        try:
            resolver = _get_draft_pair_resolver_fn()()
            with self._registry_lock:
                loaded = self._loaded_models.get(model_id)
            if loaded and hasattr(loaded.model, "_draft_model"):
                _timings = getattr(loaded.model, "last_timings", None)
                if _timings and "n_draft_accepted" in _timings:
                    _accepted = _timings["n_draft_accepted"]
                    _total = _timings.get("n_draft_total", 1)
                    for _ in range(min(_accepted, _total)):
                        resolver.record_acceptance(model_id, "draft", True)
                    for _ in range(_total - _accepted):
                        resolver.record_acceptance(model_id, "draft", False)

                    _acceptance_rate = _accepted / max(_total, 1)
                    _task_type = request.metadata.get("task_type", "general")
                    try:
                        selector = _get_model_selector_fn()()
                        if hasattr(selector, "record_reward"):
                            selector.record_reward(
                                f"{model_id}:draft:{_task_type}",
                                _acceptance_rate,
                            )
                    except Exception:
                        logger.warning(
                            "Thompson draft reward recording failed — acceptance rate not updated", exc_info=True
                        )
        except Exception:
            logger.warning(
                "Draft acceptance recording failed for %s — Thompson priors unchanged", model_id, exc_info=True
            )

    def infer_stream(self, request: InferenceRequest) -> Iterator[str]:
        """Stream inference tokens from a local GGUF model.

        Args:
            request: Inference request with model_id, prompt, and parameters.

        Yields:
            Token strings as they are generated.
        """
        if not _LLAMA_CPP_AVAILABLE:
            return

        model_path = self._resolve_model_path(request.model_id)
        if model_path is None:
            return

        model_id = _model_id_from_path(model_path)

        try:
            llm = self._get_or_load_model(model_id, model_path)
        except Exception as exc:
            logger.error("Failed to load model %s for streaming: %s", model_id, exc)
            return

        template_validation = _validate_loaded_chat_template(model_id, llm)
        if template_validation is not None and (
            not template_validation.is_trusted or template_validation.fallback_used
        ):
            logger.warning(
                "Rejecting streaming inference for %s due to untrusted embedded chat template",
                model_id,
            )
            return

        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        try:
            _stream_kwargs = self._build_completion_kwargs(request, messages, stream=True)
            stream = llm.create_chat_completion(**_stream_kwargs)

            for chunk in stream:
                delta = chunk.get("choices", _EMPTY_CHOICES_FALLBACK)[0].get("delta", _EMPTY_DELTA_FALLBACK)
                text = delta.get("content", "")
                if text:
                    yield text

        except Exception as exc:
            logger.error("Streaming failed for model %s: %s", model_id, exc)


__all__ = [
    "LlamaCppProviderAdapter",
    "_estimate_memory_gb",
    "_infer_capabilities",
    "_infer_context_window",
    "_model_id_from_path",
]
