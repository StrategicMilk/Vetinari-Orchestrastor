"""
Vetinari Central Configuration
================================
Single source of truth for all environment variables and derived paths.
Import from here instead of reading os.environ directly or hardcoding paths.

Usage:
    from vetinari.config import get_data_dir, get_lm_studio_host, VETINARI_CONFIG

Paths are derived in this priority order:
1. Explicit env var (e.g. VETINARI_DATA_DIR)
2. Project root (directory containing the vetinari/ package)
3. Never ~/.lmstudio/projects/Vetinari — that is machine-specific
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — always derived from the package location, never hardcoded
# ---------------------------------------------------------------------------

#: Absolute path to the vetinari/ package directory
PACKAGE_DIR: Path = Path(__file__).resolve().parent

#: Absolute path to the project root (one level above vetinari/)
PROJECT_ROOT: Path = PACKAGE_DIR.parent


# ---------------------------------------------------------------------------
# Data directory — where Vetinari stores all runtime data
# ---------------------------------------------------------------------------

def get_data_dir() -> Path:
    """Return the base data directory for all Vetinari runtime storage.

    Reads VETINARI_DATA_DIR env var; falls back to PROJECT_ROOT.
    Creates the directory if it doesn't exist.
    """
    data_dir_env = os.environ.get("VETINARI_DATA_DIR", "")
    if data_dir_env:
        path = Path(data_dir_env)
    else:
        path = PROJECT_ROOT
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_subdirectory(name: str) -> Path:
    """Return a named subdirectory under the data dir, creating it if needed."""
    sub = get_data_dir() / name
    sub.mkdir(parents=True, exist_ok=True)
    return sub


# ---------------------------------------------------------------------------
# LM Studio / model provider settings
# ---------------------------------------------------------------------------

def get_lm_studio_host() -> str:
    """Return the LM Studio host URL."""
    return os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")


def get_lm_studio_api_token() -> str:
    """Return the LM Studio API token (empty string if not set)."""
    return os.environ.get("LM_STUDIO_API_TOKEN", "") or os.environ.get("VETINARI_API_TOKEN", "")


# ---------------------------------------------------------------------------
# Plan mode / execution settings
# ---------------------------------------------------------------------------

def is_plan_mode_enabled() -> bool:
    return os.environ.get("PLAN_MODE_ENABLE", "true").lower() in ("1", "true", "yes")


def is_plan_mode_default() -> bool:
    return os.environ.get("PLAN_MODE_DEFAULT", "true").lower() in ("1", "true", "yes")


def get_plan_depth_cap() -> int:
    try:
        return int(os.environ.get("PLAN_DEPTH_CAP", "16"))
    except ValueError:
        return 16


def get_max_concurrent_tasks() -> int:
    try:
        return int(os.environ.get("MAX_CONCURRENT_TASKS", "4"))
    except ValueError:
        return 4


def get_execution_mode() -> str:
    return os.environ.get("EXECUTION_MODE", "execution").lower()


def get_verification_level() -> str:
    return os.environ.get("VERIFICATION_LEVEL", "standard").lower()


# ---------------------------------------------------------------------------
# External API keys
# ---------------------------------------------------------------------------

def get_openai_api_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "")


def get_anthropic_api_key() -> str:
    return os.environ.get("ANTHROPIC_API_KEY", "")


def get_google_api_key() -> str:
    return os.environ.get("GOOGLE_API_KEY", "")


def get_cohere_api_key() -> str:
    return os.environ.get("COHERE_API_KEY", "")


# ---------------------------------------------------------------------------
# Convenience bundle for components that need multiple settings at once
# ---------------------------------------------------------------------------

def get_vetinari_config() -> dict:
    """Return a dict with all core configuration values."""
    return {
        "host": get_lm_studio_host(),
        "api_token": get_lm_studio_api_token(),
        "data_dir": str(get_data_dir()),
        "project_root": str(PROJECT_ROOT),
        "plan_mode_enabled": is_plan_mode_enabled(),
        "plan_mode_default": is_plan_mode_default(),
        "plan_depth_cap": get_plan_depth_cap(),
        "max_concurrent_tasks": get_max_concurrent_tasks(),
        "execution_mode": get_execution_mode(),
        "verification_level": get_verification_level(),
    }
