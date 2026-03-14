"""Centralized application configuration for Vetinari.

Replaces hardcoded config dicts scattered across web_ui.py, cli.py, and dashboard modules.
All values are loaded from environment variables with sensible defaults pulled from
those existing locations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class VetinariConfig:
    """Centralized configuration for the Vetinari system."""

    # Server
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False

    # LM Studio
    lm_studio_host: str = "http://localhost:1234"  # noqa: VET041
    api_token: str = ""

    # Model defaults (from current_config in web_ui.py)
    default_models: list[str] = field(
        default_factory=lambda: [
            "qwen3-coder-next",
            "qwen3-30b-a3b-gemini-pro-high-reasoning-2507-hi8",
        ]
    )
    fallback_models: list[str] = field(
        default_factory=lambda: [
            "llama-3.2-1b-instruct",
            "qwen2.5-0.5b-instruct",
            "devstral-small-2505-deepseek-v3.2-speciale-distill",
        ]
    )
    uncensored_fallback_models: list[str] = field(
        default_factory=lambda: [
            "qwen3-vl-32b-gemini-heretic-uncensored-thinking",
            "glm-4.7-flash-uncensored-heretic-neo-code-imatrix-max",
        ]
    )

    # Paths (from constants.py and cli.py defaults)
    config_path: str = "manifest/vetinari.yaml"
    project_dir: str = "projects"
    output_dir: str = "outputs"

    # Resource limits (from web_ui.py current_config and constants.py)
    memory_budget_gb: int = 48
    max_concurrent_tasks: int = 4

    # Timeouts (seconds)
    default_timeout: int = 120
    llm_timeout: int = 300

    # External discovery feature flag (from web_ui.py)
    enable_external_discovery: bool = True

    @classmethod
    def from_env(cls) -> VetinariConfig:
        """Load configuration from environment variables with defaults."""
        return cls(
            host=os.environ.get("VETINARI_WEB_HOST", "127.0.0.1"),
            port=int(os.environ.get("VETINARI_WEB_PORT", "5000")),
            debug=os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes"),
            lm_studio_host=os.environ.get("LM_STUDIO_HOST", "http://localhost:1234"),  # noqa: VET041
            api_token=os.environ.get("LM_STUDIO_API_TOKEN") or os.environ.get("VETINARI_API_TOKEN", ""),
            config_path=os.environ.get("VETINARI_CONFIG", "manifest/vetinari.yaml"),
            project_dir=os.environ.get("VETINARI_PROJECT_DIR", "projects"),
            output_dir=os.environ.get("VETINARI_OUTPUT_DIR", "outputs"),
            memory_budget_gb=int(os.environ.get("VETINARI_MEMORY_GB", "48")),
            max_concurrent_tasks=int(os.environ.get("VETINARI_MAX_CONCURRENT", "4")),
            default_timeout=int(os.environ.get("VETINARI_TIMEOUT", "120")),
            llm_timeout=int(os.environ.get("VETINARI_LLM_TIMEOUT", "300")),
            enable_external_discovery=os.environ.get("ENABLE_EXTERNAL_DISCOVERY", "true").lower()
            in ("1", "true", "yes"),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility with current_config consumers.

        Returns:
            Dictionary of results.
        """
        from dataclasses import asdict

        return asdict(self)


# Singleton — populated on first call to get_config()
_config: VetinariConfig | None = None


def get_config() -> VetinariConfig:
    """Get or create the global configuration singleton.

    Returns:
        The VetinariConfig result.
    """
    global _config
    if _config is None:
        _config = VetinariConfig.from_env()
    return _config
