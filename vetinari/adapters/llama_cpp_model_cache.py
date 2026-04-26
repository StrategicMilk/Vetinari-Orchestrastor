"""llama.cpp model cache mixin — VRAM budget management and model lifecycle.

Contains ``LlamaCppModelCacheMixin``, which encapsulates the per-model locking,
LRU eviction, speculative-decoding attachment, and background calibration logic
shared by ``LlamaCppProviderAdapter``.

Concrete subclasses must initialise the following instance attributes in
their own ``__init__`` before calling any mixin method:

    - ``_models_dir: Path`` — directory containing .gguf files
    - ``_gpu_layers: int`` — configured GPU layer count (-1 = all)
    - ``_default_context_length: int`` — fallback context window size
    - ``_memory_budget_gb: float`` — total VRAM budget in GB
    - ``_ram_budget_gb: float`` — CPU RAM budget for offloaded layers
    - ``_cpu_offload_enabled: bool`` — whether partial GPU+CPU split is allowed
    - ``_loaded_models: dict[str, _LoadedModel]`` — live model registry
    - ``_registry_lock: threading.Lock`` — guards _loaded_models and _model_locks
    - ``_model_locks: dict[str, threading.Lock]`` — per-model loading locks
    - ``_calibration_pool: ThreadPoolExecutor`` — background calibration executor

Separated from ``llama_cpp_adapter.py`` so the cache mechanics can be tested
and maintained independently of the inference dispatch logic.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from vetinari.utils.lazy_import import lazy_import

from .llama_cpp_model_info import (
    _estimate_memory_gb,
    _infer_context_window,
    _LoadedModel,
)

logger = logging.getLogger(__name__)

# Optional llama-cpp-python — same pattern as the parent adapter
llama_cpp, _LLAMA_CPP_AVAILABLE = lazy_import("llama_cpp")  # type: ignore[assignment]

# Maps config string (e.g. "q4_0") to the llama_cpp module attribute name for that GGML type.
# Used to resolve the correct constant at runtime when the config type is known.
_KV_QUANT_ATTR: dict[str, str] = {
    "f16": "GGML_TYPE_F16",
    "q8_0": "GGML_TYPE_Q8_0",
    "q4_0": "GGML_TYPE_Q4_0",
}


def _resolve_kv_quant_type(type_name: str, llama_cpp_mod: Any) -> Any:
    """Resolve a KV cache quantization config string to a llama_cpp GGML type constant.

    Falls back to the string name if the constant is not present in the installed
    llama_cpp version, allowing forward compatibility.

    Args:
        type_name: Quantization name from config (e.g. "f16", "q8_0", "q4_0").
        llama_cpp_mod: The imported llama_cpp module (or lazy proxy).

    Returns:
        The GGML type constant from llama_cpp, or the raw string as a fallback.
    """
    attr = _KV_QUANT_ATTR.get(type_name, "GGML_TYPE_F16")
    if hasattr(llama_cpp_mod, attr):
        return getattr(llama_cpp_mod, attr)
    # Older llama-cpp-python builds may not expose all type constants; pass string
    return type_name


def _extract_embedded_chat_template(llm: Any) -> str | None:
    for attr_name in ("metadata", "model_metadata", "_metadata"):
        metadata = getattr(llm, attr_name, None)
        if callable(metadata):
            try:
                metadata = metadata()
            except Exception as exc:
                logger.warning("Could not read llama.cpp metadata attribute %s: %s", attr_name, exc)
                continue
        if not isinstance(metadata, dict):
            continue
        for key in (
            "tokenizer.chat_template",
            "tokenizer.ggml.chat_template",
            "chat_template",
            "llama.chat_template",
        ):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _has_trusted_chat_template(model_id: str, llm: Any) -> bool:
    embedded_template = _extract_embedded_chat_template(llm)
    if embedded_template is None:
        return True
    from vetinari.models.chat_templates import validate_template

    _template, validation = validate_template(model_id, embedded_template)
    return validation.is_trusted and not validation.fallback_used


# ── Module-level lazy getters ─────────────────────────────────────────────────
# These modules live in sub-packages that would create circular imports at
# load time. Each is imported once and cached under a threading.Lock.


_model_profiler_fn: Any = None
_model_profiler_lock = threading.Lock()


def _get_model_profiler_fn() -> Any:
    """Return the ``get_model_profiler`` callable, importing it once on first call.

    Returns:
        The ``get_model_profiler`` factory function.
    """
    global _model_profiler_fn
    if _model_profiler_fn is None:
        with _model_profiler_lock:
            if _model_profiler_fn is None:
                from vetinari.models.model_profiler import get_model_profiler

                _model_profiler_fn = get_model_profiler
    return _model_profiler_fn


_draft_pair_resolver_fn: Any = None
_draft_pair_resolver_lock = threading.Lock()


def _get_draft_pair_resolver_fn() -> Any:
    """Return the ``get_draft_pair_resolver`` callable, importing it once on first call.

    Returns:
        The ``get_draft_pair_resolver`` factory function.
    """
    global _draft_pair_resolver_fn
    if _draft_pair_resolver_fn is None:
        with _draft_pair_resolver_lock:
            if _draft_pair_resolver_fn is None:
                from vetinari.models.draft_pair_resolver import get_draft_pair_resolver

                _draft_pair_resolver_fn = get_draft_pair_resolver
    return _draft_pair_resolver_fn


_calibrate_model_fn: Any = None
_seed_thompson_priors_fn: Any = None
_load_cached_profile_fn: Any = None
_save_profile_fn: Any = None
_calibration_lock = threading.Lock()


def _get_calibration_fns() -> tuple[Any, Any, Any, Any]:
    """Return calibration callables, importing from two modules once on first call.

    Returns:
        Tuple of ``(calibrate_model, seed_thompson_priors,
        _load_cached_profile, _save_profile)``.
    """
    global _calibrate_model_fn, _seed_thompson_priors_fn, _load_cached_profile_fn, _save_profile_fn
    if _calibrate_model_fn is None:
        with _calibration_lock:
            if _calibrate_model_fn is None:
                from vetinari.models.calibration import calibrate_model, seed_thompson_priors
                from vetinari.models.model_profiler import _load_cached_profile, _save_profile

                _calibrate_model_fn = calibrate_model
                _seed_thompson_priors_fn = seed_thompson_priors
                _load_cached_profile_fn = _load_cached_profile
                _save_profile_fn = _save_profile
    return _calibrate_model_fn, _seed_thompson_priors_fn, _load_cached_profile_fn, _save_profile_fn


# ── LlamaCppModelCacheMixin ───────────────────────────────────────────────────


class LlamaCppModelCacheMixin:
    """Mixin providing VRAM-budgeted LRU model caching for llama.cpp adapters.

    Encapsulates per-model locking, LRU eviction, speculative-decoding
    attachment, and background calibration. The concrete subclass
    (``LlamaCppProviderAdapter``) initialises all required instance attributes
    before any mixin method is invoked.

    Required subclass attributes (set in ``__init__`` before use):
        _models_dir: Directory containing .gguf files.
        _gpu_layers: Configured GPU layer count (-1 = all layers).
        _default_context_length: Fallback context window size in tokens.
        _memory_budget_gb: Total VRAM budget in GB.
        _ram_budget_gb: CPU RAM available for offloaded layers in GB.
        _cpu_offload_enabled: Whether partial GPU+CPU split loading is allowed.
        _loaded_models: Live model registry dict (model_id → _LoadedModel).
        _registry_lock: threading.Lock guarding _loaded_models and _model_locks.
        _model_locks: Per-model threading.Lock dict for concurrent loading.
        _calibration_pool: ThreadPoolExecutor for background calibration tasks.
        _cache_type_k: KV cache key quantization type string (e.g. "f16", "q8_0").
        _cache_type_v: KV cache value quantization type string (e.g. "f16", "q4_0").
    """

    # Declared here to satisfy type checkers; set by the concrete subclass.
    _models_dir: Path
    _gpu_layers: int
    _default_context_length: int
    _memory_budget_gb: float
    _ram_budget_gb: float
    _cache_type_k: str
    _cache_type_v: str
    _cpu_offload_enabled: bool
    _loaded_models: dict[str, _LoadedModel]
    _registry_lock: threading.Lock
    _model_locks: dict[str, threading.Lock]
    _calibration_pool: Any

    def _get_model_lock(self, model_id: str) -> threading.Lock:
        """Get or create a per-model lock for concurrent model loading.

        Multiple models can load simultaneously because each has its own lock.
        Only lock creation is synchronised via the registry lock.

        Args:
            model_id: Model identifier to get a lock for.

        Returns:
            A ``threading.Lock`` dedicated to this model.
        """
        with self._registry_lock:
            if model_id not in self._model_locks:
                self._model_locks[model_id] = threading.Lock()
            return self._model_locks[model_id]

    def _compute_gpu_layers(self, model_id: str, model_path: Path, memory_needed: float) -> int:
        """Determine how many layers to place on GPU, using partial offload when needed.

        When a model's estimated VRAM requirement exceeds the available budget
        and CPU offload is enabled, calculates the fraction of layers that fit
        in VRAM and lets llama.cpp run the rest on CPU.

        Args:
            model_id: Model identifier, used only for log messages.
            model_path: Path to the .gguf file (reserved for future per-model probing).
            memory_needed: Estimated full-GPU memory requirement in GB.

        Returns:
            Number of layers to offload to GPU (>=0), or -1 for all layers.
        """
        current_usage = sum(m.memory_gb for m in self._loaded_models.values())
        vram_available = self._memory_budget_gb - current_usage

        if memory_needed <= vram_available:
            return self._gpu_layers

        if not self._cpu_offload_enabled or self._ram_budget_gb <= 0:
            return self._gpu_layers

        cpu_overflow = memory_needed - vram_available
        if cpu_overflow > self._ram_budget_gb:
            logger.warning(
                "Model %s (%.1f GB) exceeds VRAM (%.1f GB available) + RAM budget (%.1f GB); "
                "attempting partial offload anyway",
                model_id,
                memory_needed,
                vram_available,
                self._ram_budget_gb,
            )

        # Estimate GPU layer count proportional to VRAM fraction.
        # 1000 is a safe upper bound — llama.cpp clamps to actual layer count.
        estimated_total_layers = 1000
        gpu_fraction = vram_available / memory_needed if memory_needed > 0 else 0.0
        gpu_fraction = max(0.0, min(1.0, gpu_fraction))
        partial_layers = int(estimated_total_layers * gpu_fraction)

        logger.info(
            "Loading model %s with partial GPU offload: %.1f GB available / %.1f GB needed — "
            "using ~%d/%d layers on GPU, remainder on CPU",
            model_id,
            vram_available,
            memory_needed,
            partial_layers,
            estimated_total_layers,
        )
        return partial_layers

    def _ensure_vram_budget(self, memory_needed: float, gpu_layers: int | None = None) -> None:
        """Evict LRU models until there is enough VRAM budget for the new model.

        When CPU offload is enabled and ``gpu_layers`` is a partial count (not
        -1), only the GPU fraction of the model's memory counts against budget.

        Args:
            memory_needed: Full memory requirement in GB for the new model.
            gpu_layers: Resolved GPU layer count from ``_compute_gpu_layers``.
        """
        current_usage = sum(m.memory_gb for m in self._loaded_models.values())

        if gpu_layers is not None and gpu_layers >= 0 and self._cpu_offload_enabled:
            gpu_fraction = gpu_layers / 1000.0
            effective_vram = memory_needed * gpu_fraction
        else:
            effective_vram = memory_needed

        while current_usage + effective_vram > self._memory_budget_gb and self._loaded_models:
            lru_id = min(self._loaded_models, key=lambda k: self._loaded_models[k].last_used)
            self._unload_model(lru_id)
            current_usage = sum(m.memory_gb for m in self._loaded_models.values())

    def _unload_model(self, model_id: str) -> bool:
        """Unload a model without acquiring the lock (caller must hold ``_registry_lock``).

        Also notifies VRAMManager so its tracking stays in sync with the actual
        load state.  Caller is responsible for holding ``_registry_lock`` before
        calling this method.

        Args:
            model_id: Model identifier to unload.

        Returns:
            ``True`` if the model was unloaded, ``False`` if it was not loaded.
        """
        if model_id not in self._loaded_models:
            return False
        loaded = self._loaded_models.pop(model_id)
        logger.info("Unloading model %s (%.1f GB freed)", model_id, loaded.memory_gb)
        del loaded.model  # Release the llama.cpp model and its VRAM
        # Notify VRAMManager so its budget accounting stays correct
        try:
            from vetinari.models.vram_manager import get_vram_manager

            get_vram_manager().register_unload(model_id)
        except Exception:
            logger.warning("VRAMManager could not be notified of unload for %s — VRAM budget may drift", model_id)
        return True

    def unload_model(self, model_id: str) -> bool:
        """Unload a model and free its VRAM.

        Args:
            model_id: Model identifier to unload.

        Returns:
            ``True`` if the model was unloaded, ``False`` if it was not loaded.
        """
        with self._registry_lock:
            return self._unload_model(model_id)

    def unload_all(self) -> int:
        """Unload all cached models and free all VRAM.

        Returns:
            Number of models unloaded.
        """
        with self._registry_lock:
            count = len(self._loaded_models)
            for model_id in list(self._loaded_models.keys()):
                self._unload_model(model_id)
            return count

    def get_loaded_models(self) -> list[str]:
        """Return list of currently loaded model IDs.

        Returns:
            List of model identifier strings.
        """
        return list(self._loaded_models.keys())

    def get_vram_usage(self) -> float:
        """Return estimated total VRAM usage of all loaded models in GB.

        Returns:
            Total estimated VRAM usage in GB.
        """
        return sum(m.memory_gb for m in self._loaded_models.values())

    def _get_or_load_model(self, model_id: str, model_path: Path) -> Any:
        """Get a cached model or load it from disk, enforcing the VRAM budget.

        Uses per-model locking so that loading model A does not block inference
        on already-loaded model B (5-10x throughput improvement when multiple
        models are active simultaneously).

        Args:
            model_id: Model identifier.
            model_path: Path to the .gguf file.

        Returns:
            A ``llama_cpp.Llama`` instance ready for inference.

        Raises:
            RuntimeError: If the model cannot be loaded after eviction attempts.
        """
        # Fast path: check cache under the lightweight registry lock
        with self._registry_lock:
            if model_id in self._loaded_models:
                loaded = self._loaded_models[model_id]
                loaded.last_used = time.time()
                return loaded.model

        # Slow path: acquire per-model lock to load without blocking other models
        model_lock = self._get_model_lock(model_id)
        with model_lock:
            # Double-check: another thread may have loaded while we waited
            with self._registry_lock:
                if model_id in self._loaded_models:
                    loaded = self._loaded_models[model_id]
                    loaded.last_used = time.time()
                    return loaded.model

            # Acquire a concurrent-load slot to prevent VRAM oversubscription when
            # multiple threads race to load different models simultaneously.
            _vram_mgr = None
            _slot_acquired = False
            try:
                from vetinari.models.vram_manager import get_vram_manager

                _vram_mgr = get_vram_manager()
                _slot_acquired = _vram_mgr.acquire_load_slot()
                if not _slot_acquired:
                    logger.warning(
                        "Load slot timeout for %s — concurrent model loads at limit, proceeding anyway",
                        model_id,
                    )
            except Exception:
                logger.warning(
                    "VRAMManager load slot unavailable for %s — proceeding without concurrency control",
                    model_id,
                )

            _reservation_placed = False
            try:
                with self._registry_lock:
                    memory_needed = _estimate_memory_gb(model_path)
                    gpu_layers = self._compute_gpu_layers(model_id, model_path, memory_needed)
                    self._ensure_vram_budget(memory_needed, gpu_layers)

                # Atomically reserve VRAM before loading starts, preventing TOCTOU
                # races when multiple threads load different models concurrently.
                # The reservation is cleared by register_load() on success or by
                # release_reservation() in the finally block on failure.
                if _vram_mgr is not None:
                    try:
                        _reserved = _vram_mgr.reserve(model_id, memory_needed)
                        _reservation_placed = True
                        if not _reserved:
                            logger.warning(
                                "VRAM reservation for %s denied (%.1f GB requested) — "
                                "proceeding anyway; budget may be exceeded",
                                model_id,
                                memory_needed,
                            )
                    except Exception:
                        logger.warning(
                            "VRAM reservation call failed for %s — proceeding without reservation",
                            model_id,
                        )

                # Profile the model via ModelProfiler for optimal constructor params
                try:
                    profiler = _get_model_profiler_fn()()
                    profile = profiler.profile_model(model_path)
                    llama_kwargs = profile.get_llama_kwargs()
                    llama_kwargs["n_gpu_layers"] = gpu_layers
                    context_length = llama_kwargs.get("n_ctx", self._default_context_length)
                    logger.info(
                        "Loading model %s via ModelProfiler (family=%s, ctx=%d, gpu_layers=%d, batch=%d)",
                        model_id,
                        profile.family,
                        context_length,
                        gpu_layers,
                        llama_kwargs.get("n_batch", 512),
                    )
                except Exception:
                    logger.debug("ModelProfiler unavailable; using default loading for %s", model_id, exc_info=True)
                    context_length = _infer_context_window(model_id) or self._default_context_length
                    llama_kwargs = {
                        "n_gpu_layers": gpu_layers,
                        "n_ctx": context_length,
                        "flash_attn": True,
                        "verbose": False,
                    }
                    logger.info(
                        "Loading model %s from %s (est. %.1f GB, ctx=%d, gpu_layers=%d)",
                        model_id,
                        model_path,
                        memory_needed,
                        context_length,
                        gpu_layers,
                    )

                # Apply KV quantization type from config — resolves "f16"/"q8_0"/"q4_0"
                # to the llama_cpp GGML_TYPE_* constant (or falls back to the string).
                llama_kwargs["type_k"] = _resolve_kv_quant_type(self._cache_type_k, llama_cpp)
                llama_kwargs["type_v"] = _resolve_kv_quant_type(self._cache_type_v, llama_cpp)

                # Speculative decoding: try draft model first, fall back to PromptLookupDecoding
                _draft_attached = False
                try:
                    resolver = _get_draft_pair_resolver_fn()()
                    available_gguf = sorted(self._models_dir.rglob("*.gguf")) if self._models_dir.exists() else []
                    pair = resolver.find_pair(model_path, available_gguf)
                    if pair is not None:
                        draft_model = pair.to_llama_draft_model()
                        if draft_model is not None:
                            llama_kwargs["draft_model"] = draft_model
                            _draft_attached = True
                            logger.info(
                                "Speculative decoding (draft model): %s -> %s (%.1fx)",
                                model_id,
                                pair.draft_model_path.stem,
                                pair.size_ratio,
                            )
                except Exception:
                    logger.warning("Draft pair resolution failed for %s", model_id, exc_info=True)

                # PromptLookupDecoding fallback: no extra VRAM, 1.3-1.8x speedup on structured output
                if llama_cpp is not None and not _draft_attached and hasattr(llama_cpp, "LlamaPromptLookupDecoding"):
                    try:
                        llama_kwargs["draft_model"] = llama_cpp.LlamaPromptLookupDecoding(num_pred_tokens=10)
                        logger.info("Speculative decoding (PromptLookup): %s with num_pred_tokens=10", model_id)
                    except Exception:
                        logger.warning(
                            "PromptLookupDecoding not available for %s — speculative decoding disabled", model_id
                        )

                if llama_cpp is None:
                    msg = "llama-cpp-python is not available"
                    raise RuntimeError(msg)
                llm = llama_cpp.Llama(model_path=str(model_path), **llama_kwargs)

                with self._registry_lock:
                    self._loaded_models[model_id] = _LoadedModel(
                        model=llm,
                        model_id=model_id,
                        file_path=model_path,
                        memory_gb=memory_needed,
                        context_length=context_length,
                    )

                # Notify VRAMManager that loading succeeded so its budget
                # accounting is accurate and the pending reservation is cleared.
                if _vram_mgr is not None:
                    try:
                        _vram_mgr.register_load(model_id, memory_needed)
                        _reservation_placed = False  # reservation cleared by register_load
                    except Exception:
                        logger.warning(
                            "VRAMManager register_load failed for %s — VRAM budget may be stale",
                            model_id,
                        )

                logger.info("Model %s loaded successfully", model_id)
                template_is_trusted = _has_trusted_chat_template(model_id, llm)

                # Warm up the model: forces weight dequantization and KV cache allocation
                # so the first real request does not pay the dequant penalty.
                if template_is_trusted:
                    self._warm_up_model(model_id, llm)

                # Run calibration in background — never block inference
                def _background_calibrate(cal_model_id: str, cal_llm: Any) -> None:
                    try:
                        calibrate_model, seed_thompson_priors, _load_cached_profile, _save_profile = (
                            _get_calibration_fns()
                        )
                        cached = _load_cached_profile(cal_model_id)
                        if cached is None or "calibration" not in cached:
                            cal_result = calibrate_model(cal_model_id, cal_llm)
                            seed_thompson_priors(cal_model_id, cal_result)
                            profile_data = cached if cached is not None else {}
                            profile_data["calibration"] = cal_result.to_dict()
                            _save_profile(cal_model_id, profile_data)
                            logger.info("Background calibration completed for %s", cal_model_id)
                    except Exception:
                        logger.warning("Background calibration failed for %s", cal_model_id, exc_info=True)

                if template_is_trusted:
                    self._calibration_pool.submit(_background_calibrate, model_id, llm)
                else:
                    logger.warning(
                        "Skipping background calibration for %s because embedded chat template is untrusted",
                        model_id,
                    )

                return llm
            finally:
                # Release any pending VRAM reservation if loading failed.
                # On success, register_load() already cleared it so this is a no-op.
                if _reservation_placed and _vram_mgr is not None:
                    try:
                        _vram_mgr.release_reservation(model_id)
                    except Exception:
                        logger.warning(
                            "Could not release VRAM reservation for %s after load failure — "
                            "VRAM budget may show %s as still reserved",
                            model_id,
                            model_id,
                        )
                # Always release the load slot, whether loading succeeded or failed
                if _slot_acquired and _vram_mgr is not None:
                    _vram_mgr.release_load_slot()

    def _warm_up_model(self, model_id: str, llm: Any) -> None:
        """Run a single dummy inference to prime weight dequantization and KV cache allocation.

        llama.cpp defers weight dequantization until first use; the KV cache allocator
        also runs on the first forward pass.  Without warm-up, the first real request
        pays a 200-800 ms setup penalty.  This method absorbs that cost at load time
        so it is invisible to the user.

        Args:
            model_id: Model identifier (used only for log messages).
            llm: Loaded Llama instance to warm up.
        """
        if not _has_trusted_chat_template(model_id, llm):
            logger.warning("Skipping warm-up for %s because embedded chat template is untrusted", model_id)
            return

        try:
            _start = time.time()
            llm.create_chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,  # noqa: VET129 — 1-token warmup probe (deterministic)
                temperature=0.0,  # noqa: VET129 — deterministic warmup
            )
            _ms = int((time.time() - _start) * 1000)
            logger.info("Model %s warmed up in %d ms (dequant + KV alloc complete)", model_id, _ms)
        except Exception:
            # Warm-up failure is non-fatal — model is still usable, first request will be slower
            logger.warning(
                "Warm-up inference failed for %s — first real request may be slower due to deferred dequantization",
                model_id,
            )
