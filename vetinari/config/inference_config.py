"""Inference Configuration Manager — vetinari.config.inference_config.

Loads per-task inference profiles from external JSON config, applies
model-size adjustments and per-model overrides, and clamps values.

Usage
-----
    from vetinari.config.inference_config import get_inference_config

    cfg = get_inference_config()
    params = cfg.get_effective_params("coding", "qwen2.5-coder-7b")
    # -> {"temperature": 0.05, "top_p": 0.89, "top_k": 35, "max_tokens": 4096, ...}
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclass for a resolved profile
# ---------------------------------------------------------------------------


@dataclass
class InferenceProfile:
    """Resolved inference parameters for a task type."""

    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    stop_sequences: list[str] = field(default_factory=list)
    prefer_json: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "stop_sequences": self.stop_sequences,
            "prefer_json": self.prefer_json,
        }


# ---------------------------------------------------------------------------
# Model size classification
# ---------------------------------------------------------------------------


def _classify_model_size(model_id: str) -> str:
    """Classify a model into a size tier based on its ID heuristics."""
    mid = model_id.lower()
    # Extract size numbers from model name (e.g., "qwen-7b", "llama-70b")

    matches = re.findall(r"(\d+)[bB]", mid)
    if matches:
        size_b = max(int(m) for m in matches)
        if size_b <= 10:
            return "small"
        elif size_b <= 40:
            return "medium"
        elif size_b <= 80:
            return "large"
        else:
            return "xlarge"
    # Fallback heuristics
    if any(k in mid for k in ("tiny", "mini", "small", "1b", "3b")):
        return "small"
    if any(k in mid for k in ("xl", "xxl", "ultra", "large")):
        return "xlarge"
    return "medium"  # safe default


# ---------------------------------------------------------------------------
# Config Manager
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = str(
    pathlib.Path(__file__).resolve().parent.parent.parent / "config" / "task_inference_profiles.json"
)


class InferenceConfigManager:
    """Manages per-task inference profiles loaded from external JSON config.

    Singleton — use ``get_inference_config()`` to get the shared instance.
    """

    _instance: InferenceConfigManager | None = None
    _class_lock = threading.Lock()

    def __new__(cls) -> InferenceConfigManager:
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._setup()
        return cls._instance

    def _setup(self) -> None:
        self._lock = threading.RLock()
        self._profiles: dict[str, dict[str, Any]] = {}
        self._model_size_adjustments: dict[str, dict[str, Any]] = {}
        self._model_overrides: dict[str, dict[str, Any]] = {}
        self._loaded = False
        self._config_path: str | None = None
        self._load_config()

    def _load_config(self, path: str | None = None) -> bool:
        """Load profiles from JSON config. Returns True on success."""
        config_path = path or self._config_path or _DEFAULT_CONFIG_PATH
        self._config_path = config_path

        try:
            with open(config_path, encoding="utf-8") as f:
                data = json.load(f)

            with self._lock:
                self._profiles = data.get("profiles", {})
                self._model_size_adjustments = data.get("model_size_adjustments", {})
                self._model_overrides = data.get("model_overrides", {})
                self._loaded = True

            logger.info("Loaded %d inference profiles from %s", len(self._profiles), config_path)
            return True
        except FileNotFoundError:
            logger.warning("Inference profiles not found at %s, using fallback", config_path)
            self._loaded = False
            return False
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to parse inference profiles: %s", e)
            self._loaded = False
            return False

    def reload(self, path: str | None = None) -> bool:
        """Hot-reload config without restart."""
        return self._load_config(path)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Profile lookup
    # ------------------------------------------------------------------

    def get_profile(self, task_type: str) -> InferenceProfile:
        """Get base profile for a task type (no model adjustments)."""
        with self._lock:
            raw = self._profiles.get(task_type)
            if raw is None:
                raw = self._profiles.get("general", {})

        return InferenceProfile(
            temperature=raw.get("temperature", 0.3),
            top_p=raw.get("top_p", 0.9),
            top_k=raw.get("top_k", 40),
            max_tokens=raw.get("max_tokens", 2048),
            stop_sequences=raw.get("stop_sequences", []),
            prefer_json=raw.get("prefer_json", False),
        )

    def get_effective_params(self, task_type: str, model_id: str = "") -> dict[str, Any]:
        """Resolve profile for (task_type, model_id) with size + override adjustments.

        Returns a dict with: temperature, top_p, top_k, max_tokens,
        stop_sequences, prefer_json.
        """
        profile = self.get_profile(task_type)

        if not model_id:
            return profile.to_dict()

        # Apply model-size adjustments
        size_tier = _classify_model_size(model_id)
        with self._lock:
            size_adj = self._model_size_adjustments.get(size_tier, {})

        temp_offset = size_adj.get("temperature_offset", 0.0)
        top_p_offset = size_adj.get("top_p_offset", 0.0)
        top_k_offset = size_adj.get("top_k_offset", 0)

        # Apply model-specific overrides
        with self._lock:
            model_ovr = self._model_overrides.get(model_id, {})

        temp_offset += model_ovr.get("temperature_offset", 0.0)
        top_p_offset += model_ovr.get("top_p_offset", 0.0)
        top_k_offset += model_ovr.get("top_k_offset", 0)

        # Apply offsets and clamp
        temperature = _clamp(profile.temperature + temp_offset, 0.0, 1.5)
        top_p = _clamp(profile.top_p + top_p_offset, 0.0, 1.0)
        top_k = int(_clamp(profile.top_k + top_k_offset, 1, 100))

        return {
            "temperature": round(temperature, 3),
            "top_p": round(top_p, 3),
            "top_k": top_k,
            "max_tokens": profile.max_tokens,
            "stop_sequences": profile.stop_sequences,
            "prefer_json": profile.prefer_json,
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_profiles(self) -> list[str]:
        with self._lock:
            return list(self._profiles.keys())

    def get_all_profiles(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._profiles)

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "loaded": self._loaded,
                "config_path": self._config_path,
                "profile_count": len(self._profiles),
                "model_size_tiers": list(self._model_size_adjustments.keys()),
                "model_overrides": list(self._model_overrides.keys()),
            }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


def get_inference_config() -> InferenceConfigManager:
    return InferenceConfigManager()


def reset_inference_config() -> None:
    with InferenceConfigManager._class_lock:
        InferenceConfigManager._instance = None
