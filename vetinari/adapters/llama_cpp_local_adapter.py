"""LocalInferenceAdapter — high-level convenience wrapper for local GGUF inference.

Provides a simple chat/infer/stream interface that closely mirrors the removed
LMStudioAdapter so existing callers can switch with minimal code changes.

Pipeline role:
    API route → **LocalInferenceAdapter** → LlamaCppProviderAdapter → llama.cpp
    This is the entry point agents use; it hides ProviderConfig construction,
    model discovery, and continuous-batching bookkeeping.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from vetinari.constants import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_GPU_LAYERS,
    OPERATOR_MODELS_CACHE_DIR,
)

from .base import InferenceRequest, ProviderConfig, ProviderType
from .llama_cpp_model_info import DEFAULT_MEMORY_BUDGET_GB, DEFAULT_RAM_BUDGET_GB

logger = logging.getLogger(__name__)

# ── Module-level lazy getters ─────────────────────────────────────────────────
# These callables live in sub-packages that would create circular imports at
# module load time. Each is imported once and cached under a threading.Lock.

_current_config_fn = None
_current_config_lock = threading.Lock()


def _get_current_config() -> Any:
    """Return the ``current_config`` dict from ``vetinari.web.shared``, importing once.

    Returns:
        The live configuration dict shared across the web layer.
    """
    global _current_config_fn
    if _current_config_fn is None:
        with _current_config_lock:
            if _current_config_fn is None:
                from vetinari.web.shared import current_config

                _current_config_fn = current_config
    return _current_config_fn


_inference_config_fn = None
_inference_config_lock = threading.Lock()


def _get_inference_config_fn() -> Any:
    """Return the ``get_inference_config`` callable, importing it once on first call.

    Returns:
        The ``get_inference_config`` factory function.
    """
    global _inference_config_fn
    if _inference_config_fn is None:
        with _inference_config_lock:
            if _inference_config_fn is None:
                from vetinari.config.inference_config import get_inference_config

                _inference_config_fn = get_inference_config
    return _inference_config_fn


_inference_batcher_fn = None
_batch_request_cls = None
_inference_batcher_lock = threading.Lock()


def _get_inference_batcher_fns() -> tuple[Any, Any]:
    """Return ``(get_inference_batcher, BatchRequest)`` from ``vetinari.inference``, importing once.

    Returns:
        Tuple of (get_inference_batcher callable, BatchRequest class).
    """
    global _inference_batcher_fn, _batch_request_cls
    if _inference_batcher_fn is None:
        with _inference_batcher_lock:
            if _inference_batcher_fn is None:
                from vetinari.inference import BatchRequest, get_inference_batcher

                _inference_batcher_fn = get_inference_batcher
                _batch_request_cls = BatchRequest
    return _inference_batcher_fn, _batch_request_cls


# ── LocalInferenceAdapter ─────────────────────────────────────────────────────


class LocalInferenceAdapter:
    """High-level convenience wrapper for local GGUF inference.

    Provides a simple interface matching the removed LMStudioAdapter
    so that callers throughout the codebase can switch with minimal
    code changes.

    Usage::

        adapter = LocalInferenceAdapter()
        result  = adapter.chat(model_id, system_prompt, input_text)
        # result = {"output": "...", "latency_ms": 42, ...}
    """

    def __init__(
        self,
        models_dir: str | Path | None = None,
        gpu_layers: int | None = None,
        context_length: int | None = None,
        memory_budget_gb: int = DEFAULT_MEMORY_BUDGET_GB,
        ram_budget_gb: float = DEFAULT_RAM_BUDGET_GB,
        cpu_offload_enabled: bool = True,
    ):
        """Build a ProviderConfig and discover available GGUF models on disk.

        Reads live config from ``vetinari.web.shared.current_config`` when the
        caller does not supply explicit overrides, so ``config/models.yaml``
        and environment variables reach the inference engine automatically.

        Args:
            models_dir: Directory containing .gguf files.
            gpu_layers: Number of layers to offload to GPU (-1 = all).
            context_length: Default context window size.
            memory_budget_gb: Maximum VRAM budget in GB.
            ram_budget_gb: CPU RAM available for offloaded model layers in GB.
                Set to 0 to disable CPU offload.
            cpu_offload_enabled: When True, models too large for VRAM are
                split across GPU and CPU instead of triggering full eviction.
        """
        try:
            _cfg = _get_current_config()
            _models_dir = str(models_dir or _cfg.get("models_dir", OPERATOR_MODELS_CACHE_DIR))  # noqa: VET306 — config default, runtime override via parameter
            _gpu_layers = (
                gpu_layers
                if gpu_layers is not None
                else int(_cfg.get("gpu_layers", _cfg.get("local_gpu_layers", DEFAULT_GPU_LAYERS)))
            )
            _context_length = (
                context_length
                if context_length is not None
                else int(_cfg.get("local_context_length", DEFAULT_CONTEXT_LENGTH))
            )
            memory_budget_gb = (
                memory_budget_gb
                if memory_budget_gb != DEFAULT_MEMORY_BUDGET_GB
                else int(_cfg.get("memory_budget_gb", DEFAULT_MEMORY_BUDGET_GB))
            )
        except Exception:
            _models_dir = str(models_dir or OPERATOR_MODELS_CACHE_DIR)  # noqa: VET306 — config default, runtime override via parameter
            _gpu_layers = gpu_layers if gpu_layers is not None else DEFAULT_GPU_LAYERS
            _context_length = context_length if context_length is not None else DEFAULT_CONTEXT_LENGTH

        cfg = ProviderConfig(
            name="local-inference",
            provider_type=ProviderType.LOCAL,
            endpoint="local",
            memory_budget_gb=memory_budget_gb,
            extra_config={
                "models_dir": _models_dir,
                "gpu_layers": str(_gpu_layers),
                "context_length": str(_context_length),
                "ram_budget_gb": str(ram_budget_gb),
                "cpu_offload_enabled": str(cpu_offload_enabled),
            },
        )

        # Lazy import to avoid circular dependency within the adapters package
        from vetinari.adapters.llama_cpp_adapter import LlamaCppProviderAdapter

        self.provider = LlamaCppProviderAdapter(cfg)
        self.provider.discover_models()

    @staticmethod
    def _resolve_inference_params(task_type: str = "general", model_id: str = "") -> dict[str, Any]:
        """Resolve inference parameters from InferenceConfigManager.

        Falls back to conservative defaults when the config manager is not
        loaded or the profile is missing.

        Args:
            task_type: Task profile key (e.g. ``"coding"``, ``"general"``).
            model_id: Model identifier for model-specific adjustments.

        Returns:
            Dict with ``temperature``, ``max_tokens``, ``top_p``, ``top_k``,
            and other inference parameters from the active profile.
        """
        try:
            cfg = _get_inference_config_fn()()
            return cast(dict[str, Any], cfg.get_effective_params(task_type, model_id))
        except Exception:
            logger.warning("InferenceConfigManager unavailable; using defaults")
            return {"temperature": 0.3, "max_tokens": 2048, "top_p": 0.9, "top_k": 40}

    def chat(
        self,
        model_id: str,
        system_prompt: str,
        input_text: str,
        timeout: int = 120,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Run a chat completion and return a result dict.

        Attempts the continuous-batching path first; falls back to
        direct ``LlamaCppProviderAdapter.infer()`` when unavailable.

        Args:
            model_id: Model to use (or ``"default"`` for first available).
            system_prompt: System prompt text.
            input_text: User message text.
            timeout: Unused — in-process inference has no timeout. Wrap in
                a ``ThreadPoolExecutor`` future when a deadline is needed.
            temperature: Optional temperature override. When provided,
                overrides the value from the inference profile. Use this
                for generation tasks that require a specific temperature
                (e.g. Magpie instruction generation at 0.9).

        Returns:
            Dict with keys: ``output``, ``latency_ms``, ``tokens_used``,
            ``status``, ``error``.
        """
        # Continuous batching path (Story 13)
        try:
            get_inference_batcher, BatchRequest = _get_inference_batcher_fns()
            _batcher = get_inference_batcher()
            if _batcher.enabled:
                _profile = self._resolve_inference_params("general")
                _effective_temp = temperature if temperature is not None else _profile.get("temperature", 0.3)
                _breq = BatchRequest(
                    request_id=uuid.uuid4().hex[:12],
                    model_id=model_id,
                    prompt=input_text,
                    system_prompt=system_prompt or "",  # noqa: VET112 - empty fallback preserves optional request metadata contract
                    max_tokens=_profile.get("max_tokens", 2048),
                    temperature=_effective_temp,
                    event=threading.Event(),
                )
                _output = _batcher.submit(_breq)
                return {
                    "output": _output,
                    "latency_ms": 0.0,
                    "tokens_used": len(_output.split()),
                    "status": "ok",
                    "error": None,
                }
        except Exception as _batch_err:
            logger.warning("Continuous batching unavailable: %s", _batch_err)

        # Standard inference path
        _params = self._resolve_inference_params("general", model_id)
        _effective_temp = temperature if temperature is not None else _params.get("temperature", 0.3)
        req = InferenceRequest(
            model_id=model_id,
            prompt=input_text,
            system_prompt=system_prompt or "",  # noqa: VET112 - empty fallback preserves optional request metadata contract
            max_tokens=_params.get("max_tokens", 2048),
            temperature=_effective_temp,
            top_p=_params.get("top_p", 0.9),
            top_k=_params.get("top_k", 40),
        )
        resp = self.provider.infer(req)
        return {
            "output": resp.output,
            "latency_ms": resp.latency_ms,
            "tokens_used": resp.tokens_used,
            "status": resp.status,
            "error": resp.error,
        }

    def infer(self, model_id: str, prompt: str, timeout: int = 120) -> dict[str, Any]:
        """Run a simple prompt inference without a system prompt.

        Args:
            model_id: Model to use.
            prompt: The prompt text.
            timeout: Unused; see ``chat()`` for details.

        Returns:
            Dict with keys: ``output``, ``latency_ms``, ``tokens_used``,
            ``status``, ``error``.
        """
        return self.chat(model_id, "", prompt, timeout=timeout)

    def chat_stream(
        self,
        model_id: str,
        system_prompt: str,
        input_text: str,
        timeout: int = 180,
    ) -> Iterator[str]:
        """Stream chat completion tokens.

        Args:
            model_id: Model to use.
            system_prompt: System prompt text.
            input_text: User message text.
            timeout: Unused; see ``chat()`` for details.

        Yields:
            Token strings as they are generated.
        """
        _params = self._resolve_inference_params("general", model_id)
        req = InferenceRequest(
            model_id=model_id,
            prompt=input_text,
            system_prompt=system_prompt or "",  # noqa: VET112 - empty fallback preserves optional request metadata contract
            max_tokens=_params.get("max_tokens", 2048),
            temperature=_params.get("temperature", 0.3),
            top_p=_params.get("top_p", 0.9),
            top_k=_params.get("top_k", 40),
        )
        yield from self.provider.infer_stream(req)

    def list_loaded_models(self) -> list[dict[str, Any]]:
        """Return discovered model information as serialisable dicts.

        Returns:
            List of dicts with keys: ``id``, ``name``, ``capabilities``,
            ``memory_gb``, ``context_len``.
        """
        models = self.provider.discover_models()
        return [
            {
                "id": m.id,
                "name": m.name,
                "capabilities": m.capabilities,
                "memory_gb": m.memory_gb,
                "context_len": m.context_len,
            }
            for m in models
        ]

    def is_healthy(self) -> bool:
        """Return True if local inference is available and models exist.

        Returns:
            True if llama-cpp-python is installed and .gguf models are found.
        """
        health = self.provider.health_check()
        return bool(health.get("healthy", False))
