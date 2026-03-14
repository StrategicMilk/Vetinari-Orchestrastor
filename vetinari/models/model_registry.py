"""Vetinari Unified Model Registry.

================================
Single source of truth for all model state in the system.

All three routing paths (DynamicModelRouter, ModelRelay, PonderEngine) read
from this registry instead of maintaining separate model lists.  The registry:

1. Queries LM Studio ``/v1/models`` for currently-loaded models
2. Enriches auto-discovered models by inferring capabilities from their name
3. Merges with ``config/models.yaml`` for models that have explicit metadata
4. Tracks runtime state: is_loaded, last_seen, inferred context window, VRAM
5. Handles arbitrary/custom model names (e.g.
   ``qwen3-vl-32b-gemini-heretic-uncensored-thinking``)

Usage::

    from vetinari.model_registry import get_model_registry

    registry = get_model_registry()
    registry.refresh()                           # poll LM Studio
    models = registry.get_available_models()     # all loaded + configured
    info   = registry.get_model_info("my-model") # dict with capabilities etc.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model capability inference helpers
# ---------------------------------------------------------------------------

# Ordered: more specific patterns first
_CAPABILITY_PATTERNS: list[tuple] = [
    # Vision-language
    (r"\bvl\b|vision.?language|multimodal|visual", ["vision", "coding", "reasoning"]),
    # Uncensored / heretic variants
    (r"uncensored|heretic|unfiltered|abliterated", ["uncensored", "reasoning"]),
    # Thinking / extended CoT
    (r"thinking|cot|chain.?of.?thought", ["reasoning", "analysis"]),
    # Coding specialists
    (r"coder|codestral|deepseek.?coder|starcoder|code.?llama|codegen", ["coding", "fast"]),
    # Math / science
    (r"math|science|stem|minerva", ["reasoning", "analysis"]),
    # Instruction-following / chat
    (r"instruct|chat|assistant", ["coding", "reasoning"]),
    # General catch-alls by family
    (r"qwen\d|qwen3", ["reasoning", "coding"]),
    (r"llama.?3", ["reasoning", "coding"]),
    (r"mistral|mixtral", ["reasoning", "coding"]),
    (r"phi.?\d", ["reasoning", "fast"]),
    (r"gemma", ["reasoning"]),
    (r"yi.?\d", ["reasoning"]),
]

_CONTEXT_PATTERNS: list[tuple] = [
    (r"llama.?3", 131072),  # Llama 3.x default
    (r"qwen3", 32768),
    (r"qwen2\.5", 32768),
    (r"gemma.?2", 8192),
    (r"phi.?3", 4096),
    (r"mistral", 32768),
    (r"mixtral", 32768),
    (r"yi", 4096),
]


def _infer_capabilities(model_id: str) -> list[str]:
    """Infer capability tags from a model ID string."""
    lower = model_id.lower()
    caps: set = set()
    for pattern, tags in _CAPABILITY_PATTERNS:
        if re.search(pattern, lower):
            caps.update(tags)
    # Every model can do basic instruction following / general tasks
    caps.add("general")
    return sorted(caps)


def _infer_context_window(model_id: str) -> int:
    """Infer context window size from model family patterns."""
    lower = model_id.lower()
    for pattern, ctx in _CONTEXT_PATTERNS:
        if re.search(pattern, lower):
            return ctx
    # Conservative default for unknown models
    return 8192


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ModelInfo:
    """Runtime record for a single model."""

    model_id: str
    display_name: str
    provider: str  # "lmstudio" | "openai" | "claude" | …
    capabilities: list[str] = field(default_factory=list)
    context_window: int = 8192
    memory_requirements_gb: int = 8
    quantization: str = "unknown"
    latency_hint: str = "medium"  # "fast" | "medium" | "slow"
    privacy_level: str = "local"  # "local" | "public"
    cost_per_1k_tokens: float = 0.0
    requires_cpu_offload: bool = False
    preferred_for: list[str] = field(default_factory=list)
    is_loaded: bool = False  # currently hot in LM Studio
    last_seen: float = 0.0  # epoch timestamp
    endpoint: str = ""
    source: str = "discovered"  # "config" | "discovered" | "merged"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "model_id": self.model_id,
            "name": self.display_name,
            "provider": self.provider,
            "capabilities": self.capabilities,
            "context_window": self.context_window,
            "context_len": self.context_window,  # compat alias
            "context_length": self.context_window,  # compat alias
            "memory_requirements_gb": self.memory_requirements_gb,
            "quantization": self.quantization,
            "latency_hint": self.latency_hint,
            "privacy_level": self.privacy_level,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "requires_cpu_offload": self.requires_cpu_offload,
            "preferred_for": self.preferred_for,
            "is_loaded": self.is_loaded,
            "last_seen": self.last_seen,
            "endpoint": self.endpoint,
            "source": self.source,
            "tags": self.capabilities,  # alias for ponder.py compatibility
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Unified model registry with LM Studio discovery + static config merging."""

    _instance: ModelRegistry | None = None
    _lock = threading.Lock()

    # How often to auto-refresh from LM Studio (seconds)
    REFRESH_INTERVAL = 30

    def __init__(self):
        self._models: dict[str, ModelInfo] = {}
        self._last_refresh: float = 0.0
        self._refresh_lock = threading.Lock()
        self._lmstudio_host = os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")  # noqa: VET041
        self._config_path = Path(__file__).parent.parent / "config" / "models.yaml"

        # Load static config immediately (no network call)
        self._load_static_config()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> ModelRegistry:
        """Get instance.

        Returns:
            The result string.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_static_config(self) -> None:
        """Load models from config/models.yaml without network I/O."""
        if not self._config_path.exists():
            logger.debug("[ModelRegistry] config/models.yaml not found; skipping static load")
            return
        try:
            from vetinari.utils import load_yaml

            cfg = load_yaml(str(self._config_path))
        except Exception as e:
            logger.warning("[ModelRegistry] Failed to load models.yaml: %s", e)
            return

        all_model_configs = cfg.get("models", []) + cfg.get("cloud_models", [])
        for m in all_model_configs:
            mid = m.get("model_id", "")
            if not mid:
                continue
            info = ModelInfo(
                model_id=mid,
                display_name=m.get("display_name", mid),
                provider=m.get("provider", "lmstudio"),
                capabilities=m.get("capabilities", _infer_capabilities(mid)),
                context_window=m.get("context_window", _infer_context_window(mid)),
                memory_requirements_gb=m.get("memory_requirements_gb", 8),
                quantization=m.get("quantization", "unknown"),
                latency_hint=m.get("latency_hint", "medium"),
                privacy_level=m.get("privacy_level", "local"),
                cost_per_1k_tokens=m.get("cost_per_1k_tokens", 0.0),
                requires_cpu_offload=m.get("requires_cpu_offload", False),
                preferred_for=m.get("preferred_for", []),
                endpoint=m.get("endpoint", ""),
                source="config",
            )
            self._models[mid] = info

        logger.debug("[ModelRegistry] Loaded %s models from config", len(self._models))

    # ------------------------------------------------------------------
    # LM Studio discovery
    # ------------------------------------------------------------------

    def refresh(self, force: bool = False) -> None:
        """Poll LM Studio for currently-loaded models and merge results."""
        now = time.time()
        if not force and (now - self._last_refresh) < self.REFRESH_INTERVAL:
            return

        with self._refresh_lock:
            # Double-check after acquiring lock
            if not force and (now - self._last_refresh) < self.REFRESH_INTERVAL:
                return
            self._lmstudio_host = os.environ.get("LM_STUDIO_HOST", self._lmstudio_host)
            self._do_refresh()
            self._last_refresh = time.time()

    def _do_refresh(self) -> None:
        """Internal: query LM Studio and update registry."""
        try:
            import requests as _req

            resp = _req.get(
                f"{self._lmstudio_host}/v1/models",
                timeout=5,
                headers=self._build_auth_headers(),
            )
            if resp.status_code != 200:
                logger.debug("[ModelRegistry] /v1/models returned %s", resp.status_code)
                return
            data = resp.json()
            raw_models = data.get("data", data) if isinstance(data, dict) else data
            if not isinstance(raw_models, list):
                return
        except Exception as e:
            logger.debug("[ModelRegistry] LM Studio discovery failed: %s", e)
            return

        # Mark all previously-loaded lmstudio models as unloaded
        for info in self._models.values():
            if info.provider == "lmstudio":
                info.is_loaded = False

        loaded_ids = set()
        for raw in raw_models:
            mid = raw.get("id", raw.get("model", ""))
            if not mid:
                continue
            loaded_ids.add(mid)
            if mid in self._models:
                # Update existing entry
                existing = self._models[mid]
                existing.is_loaded = True
                existing.last_seen = time.time()
                # Only overwrite inferred fields if the config source is "discovered"
                if existing.source == "discovered":
                    existing.capabilities = _infer_capabilities(mid)
                    existing.context_window = _infer_context_window(mid)
            else:
                # New model not in config — create inferred entry
                from vetinari.utils import estimate_model_memory_gb

                self._models[mid] = ModelInfo(
                    model_id=mid,
                    display_name=raw.get("id", mid),
                    provider="lmstudio",
                    capabilities=_infer_capabilities(mid),
                    context_window=_infer_context_window(mid),
                    memory_requirements_gb=estimate_model_memory_gb(mid),
                    quantization=self._infer_quantization(mid),
                    latency_hint=self._infer_latency(mid),
                    privacy_level="local",
                    cost_per_1k_tokens=0.0,
                    is_loaded=True,
                    last_seen=time.time(),
                    endpoint=f"{self._lmstudio_host}/v1/chat/completions",
                    source="discovered",
                )

        logger.debug(
            f"[ModelRegistry] Refreshed: {len(loaded_ids)} loaded in LM Studio, {len(self._models)} total in registry"
        )

    # ------------------------------------------------------------------
    # Capability / latency inference helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_quantization(model_id: str) -> str:
        lower = model_id.lower()
        for q in ["q8", "q6", "q5_k_m", "q5_k_s", "q5", "q4_k_m", "q4_k_s", "q4", "q3", "f16", "f32"]:
            if q in lower:
                return q
        return "q4_k_m"  # reasonable default for most GGUF models

    @staticmethod
    def _infer_latency(model_id: str) -> str:
        lower = model_id.lower()
        # Large models are slow regardless of hardware
        if re.search(r"70b|72b|65b", lower):
            return "slow"
        if re.search(r"30b|32b|33b|34b", lower):
            return "medium"
        if re.search(r"7b|8b|14b|13b", lower):
            return "fast"
        if re.search(r"1b|2b|3b|4b", lower):
            return "fast"
        return "medium"

    @staticmethod
    def _build_auth_headers() -> dict[str, str]:
        token = os.environ.get("LM_STUDIO_API_TOKEN", "")
        return {"Authorization": f"Bearer {token}"} if token else {}

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_available_models(
        self,
        provider: str | None = None,
        loaded_only: bool = False,
        capability: str | None = None,
    ) -> list[ModelInfo]:
        """Return models matching optional filters.

        Args:
            provider:    Filter to a specific provider ("lmstudio", "openai", …)
            loaded_only: Only return models currently loaded in LM Studio
            capability:  Only return models with this capability tag

        Returns:
            List of results.
        """
        self.refresh()
        results = []
        for info in self._models.values():
            if provider and info.provider != provider:
                continue
            if loaded_only and not info.is_loaded:
                continue
            if capability and capability not in info.capabilities:
                continue
            results.append(info)
        return results

    def get_loaded_local_models(self) -> list[ModelInfo]:
        """Return all models currently loaded in LM Studio."""
        return self.get_available_models(provider="lmstudio", loaded_only=True)

    def get_model_info(self, model_id: str) -> ModelInfo | None:
        """Return info for a specific model ID, refreshing if needed.

        Returns:
            The ModelInfo | None result.
        """
        self.refresh()
        return self._models.get(model_id)

    def get_all_as_dicts(self) -> list[dict[str, Any]]:
        """Return all models as plain dicts (compatible with legacy routing code).

        Returns:
            The result string.
        """
        self.refresh()
        return [m.to_dict() for m in self._models.values()]

    def get_loaded_as_dicts(self) -> list[dict[str, Any]]:
        """Return currently-loaded LM Studio models as plain dicts."""
        return [m.to_dict() for m in self.get_loaded_local_models()]

    def list_loaded_models(self) -> list[dict[str, Any]]:
        """Compat shim for ponder.py which calls adapter.list_loaded_models()."""
        return self.get_loaded_as_dicts()

    def register_model(self, info: ModelInfo) -> None:
        """Manually register or override a model entry."""
        self._models[info.model_id] = info

    def get_registry_stats(self) -> dict[str, Any]:
        """Return a summary of registry state.

        Returns:
            The result string.
        """
        all_models = list(self._models.values())
        loaded = [m for m in all_models if m.is_loaded]
        return {
            "total": len(all_models),
            "loaded": len(loaded),
            "local": len([m for m in all_models if m.provider == "lmstudio"]),
            "cloud": len([m for m in all_models if m.provider != "lmstudio"]),
            "loaded_ids": [m.model_id for m in loaded],
            "last_refresh": self._last_refresh,
        }


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_registry: ModelRegistry | None = None
_registry_lock = threading.Lock()


def get_model_registry() -> ModelRegistry:
    """Return the global ModelRegistry singleton (created lazily).

    Returns:
        The result string.
    """
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ModelRegistry.get_instance()
    return _registry
