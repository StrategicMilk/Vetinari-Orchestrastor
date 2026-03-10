"""Vetinari package - Comprehensive AI Orchestration System."""

__version__ = "0.3.0"

import os as _os
import pathlib as _pathlib


def _load_env_file() -> None:
    """Load .env file into os.environ (lightweight, no dependencies).

    Searches upward from this file's directory for a ``.env`` file and sets
    any ``KEY=VALUE`` pairs that are not already in the environment.
    """
    start = _pathlib.Path(__file__).resolve().parent.parent
    for candidate in (start, start.parent, _pathlib.Path.cwd()):
        env_path = candidate / ".env"
        if env_path.is_file():
            try:
                for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    # Don't overwrite values already in the environment
                    if key and key not in _os.environ:
                        _os.environ[key] = value
            except Exception:
                pass
            return  # Stop after first .env found


_load_env_file()

__all__ = [
    # Core
    "cli",
    "orchestrator",
    "lmstudio_adapter",
    "model_pool",
    "scheduler",
    "executor",
    "validator",
    "builder",
    "upgrader",
    "utils",
    # New components
    "dynamic_model_router",
    "model_discovery",
    "code_sandbox",
    "tool_interface",
    "token_optimizer",
]
