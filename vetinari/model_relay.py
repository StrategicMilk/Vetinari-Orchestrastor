"""vetinari.model_relay — DEPRECATED thin shim.

This module is retained for backward compatibility only.
New code should use vetinari.dynamic_model_router directly.

Migration guide
---------------
Old import                                    New import
--------------------------------------------  ----------------------------------------------------
from vetinari.model_relay import model_relay  from vetinari.model_relay import model_relay  (still works)
from vetinari.model_relay import ModelRelay   from vetinari.model_relay import ModelRelay   (still works)
from vetinari.model_relay import RoutingPolicy  from vetinari.model_relay import RoutingPolicy  (still works)
get_model_relay()                             get_model_relay()  (still works)
                                              -- or --
                                              from vetinari.dynamic_model_router import get_model_router

Unique functionality still in this module (not yet in dynamic_model_router):
- ModelEntry / RoutingPolicy / ModelStatus: yaml-backed static catalog types
- ModelRelay: YAML file-based model catalog with routing policy persistence
  (dynamic_model_router is runtime-only and does not persist config)
- pick_model_for_task(): simple capability-filtered scoring (no performance history)
  (dynamic_model_router.select_model() uses live performance metrics + Thompson Sampling)
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "vetinari.model_relay is deprecated. Use vetinari.dynamic_model_router instead.",
    DeprecationWarning,
    stacklevel=2,
)

# ---------------------------------------------------------------------------
# Standard library imports needed by the classes below
# ---------------------------------------------------------------------------
import logging  # noqa: E402
import os  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from enum import Enum  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Re-export canonical router symbols so callers can migrate incrementally.
# These are the canonical equivalents of the classes defined below.
# ---------------------------------------------------------------------------
from vetinari.dynamic_model_router import (  # noqa: F401, E402  (re-export)
    DynamicModelRouter,
    ModelCapabilities,
    ModelInfo,
    TaskType,
    get_model_router,
    infer_task_type,
    # NOTE: dynamic_model_router also defines a ModelSelection, but its shape
    # differs from the one defined in this module (it uses ModelInfo, not flat
    # scalar fields).  We intentionally do NOT re-export it here to avoid
    # shadowing the local ModelSelection that existing callers depend on.
    init_model_router,
)

# ---------------------------------------------------------------------------
# Local types — unique to this module (yaml-backed catalog)
# ---------------------------------------------------------------------------


class ModelStatus(Enum):
    AVAILABLE = "available"
    LOADING = "loading"
    UNAVAILABLE = "unavailable"


@dataclass
class ModelEntry:
    """A single entry in the YAML-backed model catalog."""

    model_id: str
    provider: str
    display_name: str
    capabilities: list[str] = field(default_factory=list)
    context_window: int = 4096
    latency_hint: str = "medium"
    privacy_level: str = "local"
    memory_requirements_gb: float = 0.0
    cost_per_1k_tokens: float = 0.0
    status: str = ModelStatus.AVAILABLE.value
    endpoint: str = ""
    current_load: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "display_name": self.display_name,
            "capabilities": self.capabilities,
            "context_window": self.context_window,
            "latency_hint": self.latency_hint,
            "privacy_level": self.privacy_level,
            "memory_requirements_gb": self.memory_requirements_gb,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "status": self.status,
            "endpoint": self.endpoint,
            "current_load": self.current_load,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ModelEntry:
        return cls(
            model_id=data.get("model_id", ""),
            provider=data.get("provider", "local"),
            display_name=data.get("display_name", data.get("model_id", "")),
            capabilities=data.get("capabilities", []),
            context_window=data.get("context_window", 4096),
            latency_hint=data.get("latency_hint", "medium"),
            privacy_level=data.get("privacy_level", "local"),
            memory_requirements_gb=data.get("memory_requirements_gb", 0.0),
            cost_per_1k_tokens=data.get("cost_per_1k_tokens", 0.0),
            status=data.get("status", ModelStatus.AVAILABLE.value),
            endpoint=data.get("endpoint", ""),
            current_load=data.get("current_load", 0.0),
        )


@dataclass
class RoutingPolicy:
    """Routing policy stored in the YAML catalog (persisted to disk)."""

    local_first: bool = True
    privacy_weight: float = 1.0
    latency_weight: float = 0.5
    cost_weight: float = 0.3
    max_cost_per_1k_tokens: float = 0.0
    preferred_providers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "local_first": self.local_first,
            "privacy_weight": self.privacy_weight,
            "latency_weight": self.latency_weight,
            "cost_weight": self.cost_weight,
            "max_cost_per_1k_tokens": self.max_cost_per_1k_tokens,
            "preferred_providers": self.preferred_providers,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RoutingPolicy:
        return cls(
            local_first=data.get("local_first", True),
            privacy_weight=data.get("privacy_weight", 1.0),
            latency_weight=data.get("latency_weight", 0.5),
            cost_weight=data.get("cost_weight", 0.3),
            max_cost_per_1k_tokens=data.get("max_cost_per_1k_tokens", 0.0),
            preferred_providers=data.get("preferred_providers", []),
        )


@dataclass
class ModelSelection:
    """Flat model-selection result returned by ModelRelay.pick_model_for_task().

    NOTE: dynamic_model_router also defines a ModelSelection but with a
    different shape (it embeds a full ModelInfo object).  Keep this flat
    version here for backward compatibility with web_ui.py callers that
    serialize it directly to JSON.
    """

    model_id: str
    provider: str
    endpoint: str
    reasoning: str
    confidence: float
    latency_estimate: str

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "endpoint": self.endpoint,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "latency_estimate": self.latency_estimate,
        }


# ---------------------------------------------------------------------------
# ModelRelay — YAML-backed model catalog (unique functionality)
# ---------------------------------------------------------------------------


class ModelRelay:
    """YAML-backed model catalog with static routing policy.

    Unique vs DynamicModelRouter:
    - Persists the model catalog and policy to a YAML config file
    - Policy weights are user-configurable and saved to disk
    - Does not track live performance metrics or use Thompson Sampling
    - pick_model_for_task() uses a simple weighted score on static attributes

    For runtime performance-aware routing, use DynamicModelRouter /
    get_model_router() from vetinari.dynamic_model_router.
    """

    _instance: ModelRelay | None = None

    @classmethod
    def get_instance(cls, config_path: str | None = None) -> ModelRelay:
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    def __init__(self, config_path: str | None = None):
        if config_path is None:
            env_path = os.environ.get("VETINARI_MODELS_CONFIG", "")
            if env_path:
                config_path = Path(env_path)
            else:
                pkg_root = Path(__file__).parent.parent
                config_path = pkg_root / "config" / "models.yaml"

        self.config_path = Path(config_path)
        self.models: dict[str, ModelEntry] = {}
        self.policy = RoutingPolicy()
        self._load_config()

    def _load_config(self):
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    data = yaml.safe_load(f)
                    if data:
                        for model_data in data.get("models", []):
                            model = ModelEntry.from_dict(model_data)
                            self.models[model.model_id] = model

                        if "policy" in data:
                            self.policy = RoutingPolicy.from_dict(data["policy"])
            except Exception as e:
                logger.error("Error loading model config: %s", e)

        if not self.models:
            self._load_default_models()

    def _load_default_models(self):
        default_models = [
            ModelEntry(
                model_id="qwen2.5-coder-7b",
                provider="lmstudio",
                display_name="Qwen 2.5 Coder 7B",
                capabilities=["coding", "fast"],
                context_window=32768,
                latency_hint="fast",
                privacy_level="local",
                memory_requirements_gb=8,
                endpoint=f"{os.environ.get('LM_STUDIO_HOST', 'http://localhost:1234')}/v1/chat/completions",
            ),
            ModelEntry(
                model_id="qwen2.5-72b",
                provider="lmstudio",
                display_name="Qwen 2.5 72B",
                capabilities=["reasoning", "coding"],
                context_window=32768,
                latency_hint="medium",
                privacy_level="local",
                memory_requirements_gb=48,
                endpoint=f"{os.environ.get('LM_STUDIO_HOST', 'http://localhost:1234')}/v1/chat/completions",
            ),
            ModelEntry(
                model_id="llama-3.3-70b",
                provider="lmstudio",
                display_name="Llama 3.3 70B",
                capabilities=["reasoning", "coding"],
                context_window=32768,
                latency_hint="medium",
                privacy_level="local",
                memory_requirements_gb=48,
                endpoint=f"{os.environ.get('LM_STUDIO_HOST', 'http://localhost:1234')}/v1/chat/completions",
            ),
            ModelEntry(
                model_id="gpt-4o",
                provider="openai",
                display_name="GPT-4o",
                capabilities=["reasoning", "vision", "coding"],
                context_window=128000,
                latency_hint="medium",
                privacy_level="public",
                cost_per_1k_tokens=0.005,
                endpoint="https://api.openai.com/v1/chat/completions",
            ),
        ]

        for model in default_models:
            self.models[model.model_id] = model

    def reload_catalog(self):
        self.models.clear()
        self._load_config()

    def get_available_models(self) -> list[ModelEntry]:
        return [m for m in self.models.values() if m.status == ModelStatus.AVAILABLE.value]

    def get_model(self, model_id: str) -> ModelEntry | None:
        return self.models.get(model_id)

    def get_all_models(self) -> list[ModelEntry]:
        return list(self.models.values())

    def get_policy(self) -> RoutingPolicy:
        return self.policy

    def set_policy(self, policy: RoutingPolicy):
        self.policy = policy
        self._save_config()

    def _save_config(self):
        data = {"models": [m.to_dict() for m in self.models.values()], "policy": self.policy.to_dict()}
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            yaml.dump(data, f)

    def pick_model_for_task(self, task_type: str | None = None, context: dict | None = None) -> ModelSelection:
        available = self.get_available_models()

        if not available:
            return ModelSelection(
                model_id="",
                provider="",
                endpoint="",
                reasoning="No available models",
                confidence=0.0,
                latency_estimate="unknown",
            )

        required_caps = []
        if task_type:
            if task_type == "coding":
                required_caps = ["coding"]
            elif task_type == "reasoning":
                required_caps = ["reasoning"]
            elif task_type == "vision":
                required_caps = ["vision"]

        candidates = available
        if required_caps:
            candidates = [m for m in candidates if any(cap in m.capabilities for cap in required_caps)]

        if not candidates:
            candidates = available

        scored = []
        for model in candidates:
            score = self._score_model(model)
            scored.append((model, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[0][0] if scored else None

        if not best:
            return ModelSelection(
                model_id="",
                provider="",
                endpoint="",
                reasoning="No suitable model found",
                confidence=0.0,
                latency_estimate="unknown",
            )

        return ModelSelection(
            model_id=best.model_id,
            provider=best.provider,
            endpoint=best.endpoint,
            reasoning=self._get_selection_reason(best, task_type),
            confidence=0.9 if best.privacy_level == "local" else 0.7,
            latency_estimate=best.latency_hint,
        )

    def _score_model(self, model: ModelEntry) -> float:
        privacy_scores = {"local": 1.0, "private": 0.7, "public": 0.3}
        latency_scores = {"fast": 1.0, "medium": 0.6, "slow": 0.3}

        privacy = privacy_scores.get(model.privacy_level, 0.5)
        latency = latency_scores.get(model.latency_hint, 0.5)

        cost = 1.0
        if model.cost_per_1k_tokens > 0:
            cost = max(0, 1.0 - (model.cost_per_1k_tokens * 100))

        score = (
            privacy * self.policy.privacy_weight + latency * self.policy.latency_weight + cost * self.policy.cost_weight
        )

        return score

    def _get_selection_reason(self, model: ModelEntry, task_type: str | None = None) -> str:
        reasons = []
        if model.privacy_level == "local":
            reasons.append("local model selected")
        if self.policy.local_first:
            reasons.append("local_first policy")
        if task_type:
            reasons.append(f"supports {task_type}")
        return ", ".join(reasons) if reasons else "best available model"

    def update_model_status(self, model_id: str, status: str):
        if model_id in self.models:
            self.models[model_id].status = status

    def add_model(self, model: ModelEntry):
        self.models[model.model_id] = model
        self._save_config()

    def remove_model(self, model_id: str):
        if model_id in self.models:
            del self.models[model_id]
            self._save_config()


# ---------------------------------------------------------------------------
# Public API — singletons and helpers
# ---------------------------------------------------------------------------


def get_model_relay() -> ModelRelay:
    """Return the singleton ModelRelay (yaml-backed catalog).

    Deprecated: new code should use get_model_router() from
    vetinari.dynamic_model_router for runtime performance-aware routing.
    """
    return ModelRelay.get_instance()


# Backward-compatible lazy proxy — no config I/O on import.
class _LazyModelRelay:
    """Proxy that resolves the ModelRelay singleton on first attribute access."""

    def __getattr__(self, name):
        return getattr(ModelRelay.get_instance(), name)

    def __repr__(self):
        return repr(ModelRelay.get_instance())


model_relay = _LazyModelRelay()
