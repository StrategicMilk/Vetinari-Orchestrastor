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
from functools import lru_cache
from typing import Any

from vetinari.config.model_config import get_task_default_model
from vetinari.constants import _PROJECT_ROOT
from vetinari.utils.serialization import dataclass_to_dict

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
    max_tokens: int = 8192
    stop_sequences: list[str] = field(default_factory=list)
    prefer_json: bool = False

    def __repr__(self) -> str:
        return f"InferenceProfile(temperature={self.temperature!r}, max_tokens={self.max_tokens!r}, prefer_json={self.prefer_json!r})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return dataclass_to_dict(self)


# ---------------------------------------------------------------------------
# Model size classification
# ---------------------------------------------------------------------------


@lru_cache(maxsize=256)
def _classify_model_size(model_id: str) -> str:
    """Classify a model into a size tier based on its ID heuristics."""
    mid = model_id.lower()
    # Extract size numbers from model name (e.g., "qwen-7b", "llama-70b")

    matches = re.findall(r"(\d+)[bB]", mid)
    if matches:
        size_b = max(int(m) for m in matches)
        if size_b <= 10:
            return "small"
        if size_b <= 40:
            return "medium"
        if size_b <= 80:
            return "large"
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

_DEFAULT_CONFIG_PATH = str(_PROJECT_ROOT / "config" / "task_inference_profiles.json")


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
        self.is_loaded = False
        self._config_path: str | None = None
        self._load_config()

    def _load_config(self, path: str | None = None) -> bool:
        """Load profiles from JSON config. Returns True on success."""
        config_path = path or self._config_path or _DEFAULT_CONFIG_PATH
        self._config_path = config_path

        try:
            with pathlib.Path(config_path).open(encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                logger.error(
                    "Inference profiles at %s has unexpected root type %s (expected dict)"
                    " — all profiles cleared, inference will use built-in fallbacks",
                    config_path,
                    type(data).__name__,
                )
                with self._lock:
                    self._profiles = {}
                    self._model_size_adjustments = {}
                    self._model_overrides = {}
                    self.is_loaded = False
                return False

            with self._lock:
                self._profiles = data.get("profiles", {})
                self._model_size_adjustments = data.get("model_size_adjustments", {})
                self._model_overrides = data.get("model_overrides", {})
                self.is_loaded = True

            logger.info("Loaded %d inference profiles from %s", len(self._profiles), config_path)
            return True
        except FileNotFoundError:
            logger.warning(
                "Inference profiles not found at %s — using built-in fallbacks",
                config_path,
            )
            self.is_loaded = False
            return False
        except json.JSONDecodeError as e:
            logger.error(
                "Inference profiles at %s is not valid JSON (line %d, col %d: %s)"
                " — all profiles cleared, inference will use built-in fallbacks",
                config_path,
                e.lineno,
                e.colno,
                e.msg,
            )
            with self._lock:
                self._profiles = {}
                self._model_size_adjustments = {}
                self._model_overrides = {}
                self.is_loaded = False
            return False
        except Exception:
            logger.exception(
                "Unexpected error loading inference profiles from %s"
                " — all profiles cleared, inference will use built-in fallbacks",
                config_path,
            )
            with self._lock:
                self._profiles = {}
                self._model_size_adjustments = {}
                self._model_overrides = {}
                self.is_loaded = False
            return False

    def reload(self, path: str | None = None) -> bool:
        """Hot-reload config without restart."""
        return self._load_config(path)

    # ------------------------------------------------------------------
    # Profile lookup
    # ------------------------------------------------------------------

    def get_profile(self, task_type: str) -> InferenceProfile:
        """Get the base inference profile for a task type without model-size adjustments.

        Resolution order:
        1. JSON config profile for ``task_type`` (most specific, operator-tuned)
        2. Knowledge YAML parameter guide for ``task_type`` (data-driven fallback)
        3. JSON config ``general`` profile (safe default)

        Args:
            task_type: Task profile key (e.g. ``"coding"``, ``"reasoning"``, ``"general"``).

        Returns:
            InferenceProfile with temperature, top_p, top_k, max_tokens, and stop_sequences.
        """
        with self._lock:
            raw = self._profiles.get(task_type)

        if raw is not None:
            return InferenceProfile(
                temperature=raw.get("temperature", 0.3),
                top_p=raw.get("top_p", 0.9),
                top_k=raw.get("top_k", 40),
                max_tokens=raw.get("max_tokens", 2048),
                stop_sequences=raw.get("stop_sequences", []),
                prefer_json=raw.get("prefer_json", False),
            )

        # JSON config has no entry for this task type — try knowledge YAML
        knowledge_profile = self._get_knowledge_profile(task_type)
        if knowledge_profile is not None:
            return knowledge_profile

        # Final fallback: general profile from JSON config
        with self._lock:
            raw = self._profiles.get("general", {})

        return InferenceProfile(
            temperature=raw.get("temperature", 0.3),
            top_p=raw.get("top_p", 0.9),
            top_k=raw.get("top_k", 40),
            max_tokens=raw.get("max_tokens", 2048),
            stop_sequences=raw.get("stop_sequences", []),
            prefer_json=raw.get("prefer_json", False),
        )

    def _get_knowledge_profile(self, task_type: str) -> InferenceProfile | None:
        """Build an InferenceProfile from knowledge YAML parameter recommendations.

        Handles both preset format (``{"preset": "code", "temperature": 0.05, ...}``)
        and per-parameter format (``{"temperature": {"recommended": 0.05, ...}, ...}``).

        Args:
            task_type: Task type to look up in parameters.yaml.

        Returns:
            InferenceProfile derived from knowledge data, or None if no useful
            knowledge exists for this task type.
        """
        try:
            from vetinari.knowledge import get_parameter_guide
        except ImportError:
            logger.warning(
                "Knowledge module unavailable — skipping knowledge-based profile for %s",
                task_type,
            )
            return None

        guide = get_parameter_guide(task_type)
        if not guide:
            return None

        # Preset format: {"preset": "code", "temperature": 0.05, "top_p": 0.89, ...}
        if "preset" in guide:
            temperature = guide.get("temperature")
            top_p = guide.get("top_p")
            top_k = guide.get("top_k")
            if temperature is not None:
                return InferenceProfile(
                    temperature=float(temperature),
                    top_p=float(top_p) if top_p is not None else 0.9,
                    top_k=int(top_k) if top_k is not None else 40,
                    max_tokens=int(guide.get("max_tokens", 8192)),
                    stop_sequences=guide.get("stop_sequences", []),
                    prefer_json=bool(guide.get("prefer_json", False)),
                )

        # Per-parameter format: {"temperature": {"recommended": 0.05, ...}, ...}
        temperature = guide.get("temperature", {}).get("recommended")
        top_p = guide.get("top_p", {}).get("recommended")
        top_k = guide.get("top_k", {}).get("recommended")
        if temperature is not None:
            return InferenceProfile(
                temperature=float(temperature),
                top_p=float(top_p) if top_p is not None else 0.9,
                top_k=int(top_k) if top_k is not None else 40,
                max_tokens=int(guide.get("max_tokens", {}).get("recommended", 8192)),
                stop_sequences=[],
                prefer_json=False,
            )

        return None

    def get_effective_params(self, task_type: str, model_id: str = "") -> dict[str, Any]:
        """Resolve inference params for a (task_type, model_id) pair with all adjustments applied.

        Applies size-tier offsets (small/medium/large/xlarge) and per-model overrides
        on top of the base profile, then clamps each value to a valid range.

        Args:
            task_type: Task profile key (e.g. ``"coding"``, ``"reasoning"``).
            model_id: Model identifier used to derive the size tier and per-model overrides.
                When empty, returns the base profile without any model adjustments.

        Returns:
            Dictionary with keys: temperature, top_p, top_k, max_tokens,
            stop_sequences, prefer_json — all clamped to valid ranges.
        """
        profile = self.get_profile(task_type)

        if not model_id:
            # Auto-select the best available model for this task type so that
            # model-size adjustments (temperature offsets, top_k tweaks) can
            # still be applied even when the caller didn't specify a model.
            try:
                model_id = get_task_default_model(task_type)
            except Exception:
                logger.warning(
                    "get_task_default_model unavailable for %s — using base profile without model-specific tuning",
                    task_type,
                )
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
        """Return the names of all loaded task-type inference profiles.

        Returns:
            List of profile key strings (e.g. ``["general", "coding", "reasoning"]``).
        """
        with self._lock:
            return list(self._profiles.keys())

    def get_all_profiles(self) -> dict[str, dict[str, Any]]:
        """Return all loaded inference profiles as a copy of the raw config data.

        Returns:
            Mapping from task-type name to its raw parameter dictionary as loaded
            from the JSON config file.
        """
        with self._lock:
            return dict(self._profiles)

    def get_stats(self) -> dict[str, Any]:
        """Return diagnostic information about the current config manager state.

        Returns:
            Dictionary with keys: loaded (bool), config_path, profile_count,
            model_size_tiers (list of tier names), and model_overrides (list of model IDs
            with explicit overrides).
        """
        with self._lock:
            return {
                "loaded": self.is_loaded,
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
    """Return the singleton InferenceConfigManager, creating it if necessary.

    Returns:
        The shared InferenceConfigManager used for per-task inference profile resolution.
    """
    return InferenceConfigManager()


def reset_inference_config() -> None:
    """Reset inference config."""
    with InferenceConfigManager._class_lock:
        InferenceConfigManager._instance = None
