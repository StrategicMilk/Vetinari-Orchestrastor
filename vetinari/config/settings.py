"""Pydantic Settings models for Vetinari configuration.

Provides validated, type-safe configuration models that replace raw JSON
loading.  Models can be populated from JSON config files, environment
variables (prefixed ``VETINARI_``), or programmatic defaults.

Usage::

    from vetinari.config.settings import VetinariSettings, get_settings

    settings = get_settings()
    profile = settings.get_inference_profile("coding")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# ── Project root detection ────────────────────────────────────────────────────

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


# ── Inference profile model ───────────────────────────────────────────────────


class InferenceProfileModel(BaseModel):
    """Validated inference parameters for a single task type.

    Attributes:
        temperature: Sampling temperature (0.0-1.5).
        top_p: Nucleus sampling probability ceiling (0.0-1.0).
        top_k: Top-k token count for sampling (1-100).
        max_tokens: Maximum tokens to generate.
        stop_sequences: Sequences that halt generation.
        prefer_json: Whether to request JSON output format.
        description: Human-readable description of the profile purpose.
    """

    temperature: float = Field(default=0.3, ge=0.0, le=1.5)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1, le=100)
    max_tokens: int = Field(default=2048, ge=1, le=65536)
    stop_sequences: list[str] = Field(default_factory=list)
    prefer_json: bool = False
    description: str = ""


class ModelSizeAdjustment(BaseModel):
    """Offset adjustments applied based on model parameter count.

    Attributes:
        note: Human-readable size tier description.
        temperature_offset: Added to base temperature.
        top_p_offset: Added to base top_p.
        top_k_offset: Added to base top_k.
    """

    note: str = ""
    temperature_offset: float = 0.0
    top_p_offset: float = 0.0
    top_k_offset: int = 0


class ModelOverride(BaseModel):
    """Per-model parameter overrides.

    Attributes:
        note: Reason for the override.
        temperature_offset: Added to base temperature.
        top_p_offset: Added to base top_p.
        top_k_offset: Added to base top_k.
    """

    note: str = ""
    temperature_offset: float = 0.0
    top_p_offset: float = 0.0
    top_k_offset: int = 0


# ── Inference config aggregate ────────────────────────────────────────────────


class InferenceConfig(BaseModel):
    """Complete inference configuration loaded from task_inference_profiles.json.

    Attributes:
        version: Schema version string.
        profiles: Mapping of task type to inference profile.
        model_size_adjustments: Offsets keyed by size tier.
        model_overrides: Per-model-id offsets.
    """

    version: str = "1.0"
    profiles: dict[str, InferenceProfileModel] = Field(default_factory=dict)
    model_size_adjustments: dict[str, ModelSizeAdjustment] = Field(default_factory=dict)
    model_overrides: dict[str, ModelOverride] = Field(default_factory=dict)

    @classmethod
    def from_json_file(cls, path: Path) -> InferenceConfig:
        """Load and validate inference config from a JSON file.

        Args:
            path: Path to the JSON configuration file.

        Returns:
            Validated InferenceConfig instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the JSON is malformed or fails validation.
        """
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
        return cls.model_validate(raw)


# ── Top-level application settings ────────────────────────────────────────────


class VetinariSettings(BaseSettings):
    """Top-level application settings populated from env vars and config files.

    Environment variables are prefixed with ``VETINARI_`` and use double
    underscores for nesting (e.g. ``VETINARI_LOG_LEVEL=DEBUG``).

    Attributes:
        log_level: Root logging level.
        inference_config_path: Path to task_inference_profiles.json.
        llm_guard_config_path: Path to llm_guard.yaml.
        lm_studio_base_url: Base URL for LM Studio API.
        lm_studio_timeout: Request timeout in seconds for LM Studio calls.
        max_agent_retries: Default retry limit for agent invocations.
        enable_observability: Whether OpenTelemetry tracing is active.
    """

    model_config = SettingsConfigDict(
        env_prefix="VETINARI_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    log_level: str = "INFO"
    inference_config_path: Path = _CONFIG_DIR / "task_inference_profiles.json"
    llm_guard_config_path: Path = _CONFIG_DIR / "llm_guard.yaml"
    lm_studio_base_url: str = "http://127.0.0.1:1234"  # noqa: VET041
    lm_studio_timeout: float = 120.0  # seconds
    max_agent_retries: int = 3
    enable_observability: bool = False

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        """Ensure log_level is a recognized Python logging level name."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
        return upper

    def load_inference_config(self) -> InferenceConfig:
        """Load and validate the inference config from the configured path.

        Returns:
            Validated InferenceConfig, or empty defaults if file is missing.
        """
        path = self.inference_config_path
        if not path.exists():
            logger.warning("Inference config not found at %s, using defaults", path)
            return InferenceConfig()
        try:
            cfg = InferenceConfig.from_json_file(path)
            logger.info("Loaded %d inference profiles from %s", len(cfg.profiles), path)
            return cfg
        except (ValueError, OSError) as exc:
            logger.error("Failed to load inference config: %s", exc)
            return InferenceConfig()

    def get_inference_profile(self, task_type: str) -> InferenceProfileModel:
        """Convenience method to load config and retrieve a single profile.

        Args:
            task_type: The task type key (e.g. ``"coding"``, ``"planning"``).

        Returns:
            The matching profile, or the ``"general"`` fallback, or defaults.
        """
        cfg = self.load_inference_config()
        if task_type in cfg.profiles:
            return cfg.profiles[task_type]
        if "general" in cfg.profiles:
            return cfg.profiles["general"]
        return InferenceProfileModel()


# ── Singleton accessor ────────────────────────────────────────────────────────

_settings_instance: VetinariSettings | None = None


def get_settings() -> VetinariSettings:
    """Return the process-global VetinariSettings instance.

    Returns:
        The singleton VetinariSettings.
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = VetinariSettings()
    return _settings_instance


def reset_settings() -> None:
    """Reset the singleton (intended for testing)."""
    global _settings_instance
    _settings_instance = None
