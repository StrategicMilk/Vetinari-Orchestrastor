"""Model recommendation engine for native and GGUF model selection.

This is step 2 of the setup pipeline: Hardware Detection →
**Model Recommendation** → Init Wizard → Configuration.

Maps detected hardware capabilities to backend-aware model choices,
considering VRAM tiers, quantization levels, and use-case fitness.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from vetinari.knowledge import get_quant_recommendation
from vetinari.setup.model_recommender_catalog import (
    _CPU_OFFLOAD_MODELS,
    _VLLM_MODEL_TIERS,
    _VRAM_TIERS,
)
from vetinari.setup.model_recommender_types import SetupModelRecommendation
from vetinari.system.hardware_detect import GpuVendor, HardwareProfile

logger = logging.getLogger(__name__)

_NATIVE_BACKENDS = ("nim", "vllm")
_LLAMA_CPP_ALIASES = {"llama_cpp", "llama-cpp", "llama", "local"}


def _normalize_recommendation_backend(backend: str) -> str:
    normalized = backend.strip().lower().replace("-", "_")
    if normalized in _LLAMA_CPP_ALIASES:
        return "llama_cpp"
    return normalized


def _estimate_model_budget_gb(hardware: HardwareProfile) -> float:
    vram = hardware.effective_vram_gb
    if hardware.gpu_vendor == GpuVendor.APPLE and vram == 0:
        vram = round(hardware.ram_gb * 0.75 * 0.9, 1)
    if vram == 0 and hardware.ram_gb > 0:
        vram = round(hardware.ram_gb * 0.4, 1)
    return vram


def _default_backend_order(hardware: HardwareProfile) -> list[str]:
    if not hardware.has_gpu:
        return ["llama_cpp"]
    if hardware.gpu_vendor == GpuVendor.NVIDIA and hardware.cuda_available:
        return ["nim", "vllm", "llama_cpp"]
    return ["vllm", "llama_cpp"]


def _resolve_recommendation_backends(
    hardware: HardwareProfile,
    available_backends: list[str] | None,
) -> list[str]:
    raw_backends = _default_backend_order(hardware) if available_backends is None else available_backends
    resolved: list[str] = []
    for backend in raw_backends:
        normalized = _normalize_recommendation_backend(backend)
        if normalized in {"nim", "vllm", "llama_cpp"} and normalized not in resolved:
            resolved.append(normalized)
    return resolved or ["llama_cpp"]


def _backend_label(backend: str) -> str:
    if backend == "nim":
        return "NIM"
    if backend == "vllm":
        return "vLLM"
    return "llama.cpp"


def _native_model_for_backend(
    model: SetupModelRecommendation,
    backend: str,
    *,
    primary_backend: str | None = None,
) -> SetupModelRecommendation:
    label = _backend_label(backend)
    reason = model.reason.replace("vLLM", label).replace("vllm", backend)
    return replace(
        model,
        backend=backend,
        reason=reason,
        is_primary=model.is_primary and (primary_backend is None or backend == primary_backend),
    )


class ModelRecommender:
    """Recommends models based on detected hardware and intended use cases.

    Uses a VRAM-to-model matrix to select models that will fit in available
    GPU memory with headroom for the OS and inference overhead.

    The factory pipeline runs multiple agents in parallel — so users benefit
    from a *portfolio* of models: small efficient ones for grunt work
    (classification, routing, extraction) and larger ones for complex reasoning.
    """

    def __init__(self, vram_tiers: list[dict[str, Any]] | None = None) -> None:
        self._tiers = vram_tiers or _VRAM_TIERS

    def recommend_models(self, hardware: HardwareProfile) -> list[SetupModelRecommendation]:
        """Return recommended models for the given hardware profile.

        Selects the VRAM tier matching the hardware's effective VRAM, then
        returns all models in that tier.  For Apple Silicon with unified
        memory, uses the estimated VRAM share.

        Args:
            hardware: Detected hardware profile.

        Returns:
            List of ModelRecommendation objects, primary first.
        """
        vram = hardware.effective_vram_gb

        # Apple Silicon with no discrete GPU uses unified memory estimate
        if hardware.gpu_vendor == GpuVendor.APPLE and vram == 0:
            vram = round(hardware.ram_gb * 0.75 * 0.9, 1)

        # CPU-only fallback: use RAM / 2 as rough VRAM equivalent
        if vram == 0 and hardware.ram_gb > 0:
            vram = round(hardware.ram_gb * 0.4, 1)

        for tier in self._tiers:
            if tier["min_vram_gb"] <= vram < tier["max_vram_gb"]:
                models: list[SetupModelRecommendation] = list(tier["models"])
                logger.info(
                    "Recommended %d models for %.1f GB effective VRAM (tier: %s)",
                    len(models),
                    vram,
                    tier["label"],
                )
                return models

        # Fallback to smallest tier
        return list(self._tiers[0]["models"])

    def get_tier_label(self, hardware: HardwareProfile) -> str:
        """Return the human-readable label for the matched VRAM tier.

        Args:
            hardware: Detected hardware profile.

        Returns:
            Tier label string (e.g. "8-16 GB VRAM").
        """
        vram = hardware.effective_vram_gb
        if vram == 0 and hardware.ram_gb > 0:
            vram = round(hardware.ram_gb * 0.4, 1)

        for tier in self._tiers:
            if tier["min_vram_gb"] <= vram < tier["max_vram_gb"]:
                return str(tier["label"])
        return str(self._tiers[0]["label"])

    def recommend_for_task(
        self,
        hardware: HardwareProfile,
        task_type: str,
    ) -> list[SetupModelRecommendation]:
        """Recommend models optimised for a specific task type.

        Enriches the standard VRAM-tier recommendations with task-specific
        quantization advice from quantization.yaml.  When the preferred quant
        for the task matches a recommendation, the reason string is annotated
        so the user understands why that model was selected.

        Args:
            hardware: Detected hardware profile.
            task_type: Task type string (e.g., ``"coding"``, ``"reasoning"``).

        Returns:
            List of ModelRecommendation objects, primary first, with
            task-aware reason annotations where applicable.
        """
        base = self.recommend_models(hardware)
        rec = get_quant_recommendation(task_type, hardware.effective_vram_gb or None)

        if not rec:
            return base

        task_rec = rec.get("task_recommendation", {})
        preferred_quant = task_rec.get("preferred", "").lower()
        notes = task_rec.get("notes", "")

        if not preferred_quant:
            return base

        enriched: list[SetupModelRecommendation] = []
        for model in base:
            quant_lower = model.quantization.lower()
            if quant_lower == preferred_quant:
                annotation = f" (recommended quant for {task_type}"
                if notes:
                    annotation += f": {notes}"
                annotation += ")"
                enriched.append(replace(model, reason=model.reason + annotation))
            else:
                enriched.append(model)

        logger.info(
            "Task-aware recommendation for %s: preferred quant=%s, %d models annotated",
            task_type,
            preferred_quant,
            sum(1 for m in enriched if "(recommended quant" in m.reason),
        )
        return enriched

    def recommend_portfolio(
        self,
        hardware: HardwareProfile,
        available_backends: list[str] | None = None,
    ) -> dict[str, list[SetupModelRecommendation]]:
        """Recommend a complete model portfolio organized by use case.

        Vetinari's factory pipeline runs multiple agents in parallel, so users
        benefit from having models at different size tiers:

        - **grunt**: Small, fast models (1-3B) for classification, routing,
          extraction — the bulk of pipeline operations.
        - **worker**: Medium models (7-14B) for coding, review, documentation —
          the main workhorse models.
        - **thinker**: Large models (32-72B+) for deep reasoning, planning,
          architecture — complex tasks that benefit from scale.

        When NIM/vLLM is available, native SafeTensors/AWQ/GPTQ formats are
        preferred for models that fit in VRAM. GGUF remains available for
        llama.cpp sidecars, fallback, and CPU-offloaded large models.

        Args:
            hardware: Detected hardware profile.
            available_backends: List of available backends.

        Returns:
            Dict mapping use-case role to recommended models.
        """
        backends = _resolve_recommendation_backends(hardware, available_backends)
        vram = _estimate_model_budget_gb(hardware)
        native_backend = next((backend for backend in backends if backend in _NATIVE_BACKENDS), None)
        has_native_backend = native_backend is not None

        portfolio: dict[str, list[SetupModelRecommendation]] = {
            "grunt": [],
            "worker": [],
            "thinker": [],
        }

        # Grunt models: small, low-overhead llama.cpp sidecars are still useful.
        portfolio["grunt"] = [
            SetupModelRecommendation(
                name="Qwen 2.5 1.5B Q4_K_M",
                repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
                size_gb=1.1,
                quantization="Q4_K_M",
                parameter_count="1.5B",
                reason="Fast classification, routing, and extraction on a low-overhead llama.cpp sidecar",
                is_primary=True,
                best_for=("classification", "extraction", "general"),
            ),
        ]

        # Worker models: main coding/review workhorses
        worker_recs: list[SetupModelRecommendation] = []
        if native_backend and vram >= 8:
            worker_recs.append(
                SetupModelRecommendation(
                    name="Qwen 2.5 Coder 7B AWQ",
                    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
                    filename="",
                    size_gb=4.5,
                    quantization="AWQ",
                    parameter_count="7B",
                    reason=f"Coding workhorse - AWQ on {_backend_label(native_backend)} for fast parallel throughput",
                    is_primary=True,
                    model_format="awq",
                    backend=native_backend,
                    gpu_only=True,
                    best_for=("coding", "review", "documentation"),
                ),
            )
        if "llama_cpp" in backends and vram >= 4:
            worker_recs.append(
                SetupModelRecommendation(
                    name="Qwen 2.5 Coder 7B Q4_K_M",
                    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                    filename="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
                    size_gb=4.4,
                    quantization="Q4_K_M",
                    parameter_count="7B",
                    reason="Coding workhorse fallback - GGUF for llama.cpp",
                    is_primary=not has_native_backend,
                    best_for=("coding", "review", "documentation"),
                ),
            )
        portfolio["worker"] = worker_recs

        # Thinker models: reasoning and planning
        thinker_recs: list[SetupModelRecommendation] = []
        if native_backend and vram >= 24:
            thinker_recs.append(
                SetupModelRecommendation(
                    name="Qwen 2.5 32B AWQ",
                    repo_id="Qwen/Qwen2.5-32B-Instruct-AWQ",
                    filename="",
                    size_gb=18.0,
                    quantization="AWQ",
                    parameter_count="32B",
                    reason=f"Deep reasoning - AWQ on {_backend_label(native_backend)} for high-throughput planning",
                    is_primary=True,
                    model_format="awq",
                    backend=native_backend,
                    gpu_only=True,
                    best_for=("reasoning", "planning", "research", "creative"),
                ),
            )
        if native_backend and 16 <= vram < 24:
            thinker_recs.append(
                SetupModelRecommendation(
                    name="Qwen 2.5 14B AWQ",
                    repo_id="Qwen/Qwen2.5-14B-Instruct-AWQ",
                    filename="",
                    size_gb=9.0,
                    quantization="AWQ",
                    parameter_count="14B",
                    reason=f"Reasoning and planning - AWQ on {_backend_label(native_backend)} for parallel throughput",
                    is_primary=True,
                    model_format="awq",
                    backend=native_backend,
                    gpu_only=True,
                    best_for=("reasoning", "planning", "research"),
                ),
            )
        if "llama_cpp" in backends and vram >= 8:
            thinker_recs.append(
                SetupModelRecommendation(
                    name="Qwen 2.5 14B Q4_K_M",
                    repo_id="Qwen/Qwen2.5-14B-Instruct-GGUF",
                    filename="qwen2.5-14b-instruct-q4_k_m.gguf",
                    size_gb=8.7,
                    quantization="Q4_K_M",
                    parameter_count="14B",
                    reason="Reasoning and planning fallback - GGUF for llama.cpp",
                    is_primary=not has_native_backend,
                    best_for=("reasoning", "planning", "research"),
                ),
            )
        # CPU offload options for very capable reasoning
        if "llama_cpp" in backends and hardware.ram_gb >= 32 and vram >= 8:
            thinker_recs.append(
                SetupModelRecommendation(
                    name="Qwen 2.5 72B Q4_K_M (CPU offload)",
                    repo_id="bartowski/Qwen2.5-72B-Instruct-GGUF",
                    filename="Qwen2.5-72B-Instruct-Q4_K_M.gguf",
                    size_gb=42.0,
                    quantization="Q4_K_M",
                    parameter_count="72B",
                    reason="Top-tier reasoning — uses VRAM+RAM via CPU offload, slower but best quality",
                    model_format="gguf",
                    backend="llama_cpp",
                    gpu_only=False,
                    best_for=("reasoning", "planning", "research", "security", "creative"),
                ),
            )
        portfolio["thinker"] = thinker_recs

        logger.info(
            "Portfolio recommendation: grunt=%d, worker=%d, thinker=%d (vram=%.1fGB, backends=%s)",
            len(portfolio["grunt"]),
            len(portfolio["worker"]),
            len(portfolio["thinker"]),
            vram,
            backends,
        )
        return portfolio

    def recommend_models_multi_format(
        self,
        hardware: HardwareProfile,
        available_backends: list[str] | None = None,
    ) -> list[SetupModelRecommendation]:
        """Recommend models across all formats based on available backends.

        When vLLM/NIM is available, AWQ/GPTQ models that fit in VRAM are
        promoted as the primary recommendation — they provide significantly
        better throughput than GGUF on dedicated GPU inference.  GGUF models
        remain available as alternatives and for CPU-offloaded large models.

        Ordering priority:
        1. vLLM AWQ/GPTQ models that fit in VRAM (best throughput)
        2. GGUF models for the VRAM tier (versatile, support CPU offload)
        3. CPU-offload GGUF models (larger than VRAM, slower but more capable)

        Args:
            hardware: Detected hardware profile.
            available_backends: List of available backends (e.g. ``["llama_cpp",
                "vllm"]``). Defaults to hardware-capable native backends first
                when not specified.

        Returns:
            List of ModelRecommendation objects across all formats.
        """
        backends = _resolve_recommendation_backends(hardware, available_backends)
        gguf_recs = list(self.recommend_models(hardware)) if "llama_cpp" in backends else []

        has_gpu_backend = any(backend in backends for backend in _NATIVE_BACKENDS)

        if not has_gpu_backend:
            # No vLLM/NIM — just add CPU-offload options if enough RAM
            result = list(gguf_recs)
            if "llama_cpp" in backends and hardware.ram_gb >= 32:
                vram = _estimate_model_budget_gb(hardware)
                result.extend(m for m in _CPU_OFFLOAD_MODELS if m.size_gb <= (vram + hardware.ram_gb * 0.5))
            return result

        vram = _estimate_model_budget_gb(hardware)

        # Collect native models that fit in VRAM; these get top priority.
        gpu_recs: list[SetupModelRecommendation] = []
        first_native_backend = next((backend for backend in backends if backend in _NATIVE_BACKENDS), None)
        for backend in (backend for backend in backends if backend in _NATIVE_BACKENDS):
            for entry in _VLLM_MODEL_TIERS:
                if entry["min_vram_gb"] <= vram < entry["max_vram_gb"]:
                    gpu_recs.extend(
                        _native_model_for_backend(m, backend, primary_backend=first_native_backend)
                        for m in entry["models"]
                        if m.size_gb <= vram
                    )
                    break

        # Demote GGUF primary flags when native options exist.
        if gpu_recs:
            gguf_recs = [replace(r, is_primary=False) for r in gguf_recs]

        # Build ordered result: GPU-optimized first, then GGUF, then offload
        result: list[SetupModelRecommendation] = []
        result.extend(gpu_recs)
        result.extend(gguf_recs)

        if "llama_cpp" in backends and hardware.ram_gb >= 32:
            for model in _CPU_OFFLOAD_MODELS:
                if model.size_gb <= (vram + hardware.ram_gb * 0.5):
                    result.append(model)

        logger.info(
            "Multi-format recommendations: %d total (%d vLLM/NIM, %d GGUF, backends=%s)",
            len(result),
            len(gpu_recs),
            len(gguf_recs),
            backends,
        )
        return result

    def suggest_kv_cache_quant(
        self,
        hardware: HardwareProfile,
        context_length: int = 8192,
    ) -> str:
        """Suggest a KV cache quantization type based on available VRAM and context length.

        Uses a simple heuristic: estimate how much VRAM the KV cache would consume
        at f16 precision, then downgrade to q8_0 or q4_0 when the budget is tight.
        This is a lightweight recommendation for the setup wizard — production inference
        uses ``VRAMManager.recommend_kv_quant_for_context`` which reads live VRAM state.

        Args:
            hardware: Detected hardware profile with VRAM information.
            context_length: Desired context window in tokens (default: 8192).

        Returns:
            One of "f16", "q8_0", or "q4_0".
        """
        # Bytes per token for f16 KV cache (worst case estimate)
        _F16_BYTES_PER_TOKEN = 2048
        kv_f16_gb = context_length * _F16_BYTES_PER_TOKEN / (1024**3)

        vram = hardware.effective_vram_gb
        if hardware.gpu_vendor == GpuVendor.APPLE and vram == 0:
            vram = round(hardware.ram_gb * 0.75 * 0.9, 1)
        if vram == 0 and hardware.ram_gb > 0:
            vram = round(hardware.ram_gb * 0.4, 1)

        # Assume ~80% of VRAM is already committed to model weights; the rest is for KV cache
        available_for_kv = vram * 0.20

        if kv_f16_gb > available_for_kv * 0.75:
            return "q4_0"
        if kv_f16_gb > available_for_kv * 0.50:
            return "q8_0"
        return "f16"


ModelRecommendation = SetupModelRecommendation
