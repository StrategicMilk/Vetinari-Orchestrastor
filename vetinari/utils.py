"""Utils module."""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Singleton helper — replaces 18+ copy-pasted _instance patterns
# ---------------------------------------------------------------------------


class SingletonMeta(type):
    """Thread-safe singleton metaclass.

    Usage::

        class MyService(metaclass=SingletonMeta):
            def __init__(self, config=None):
                self.config = config or {}

        # First call creates the instance; subsequent calls return same object.
        svc = MyService(config={"key": "value"})
        assert MyService() is svc  # True

        # Reset for testing:
        MyService.reset_instance()
    """

    _instances: dict[type, Any] = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]

    def reset_instance(cls) -> None:
        """Remove the cached singleton instance (useful for tests)."""
        with cls._lock:
            cls._instances.pop(cls, None)


def setup_logging(level=logging.INFO, log_dir="logs"):
    """Set up logging.

    Args:
        level: The level.
        log_dir: The log dir.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/vetinari.log", mode="a"),
            logging.StreamHandler(),
        ],
    )


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand ${VAR} placeholders in YAML string values."""
    if isinstance(obj, str):
        return re.sub(
            r"\$\{([^}]+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            obj,
        )
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


def load_yaml(path: str) -> Any:
    """Load a YAML file and expand ${ENV_VAR} placeholders in string values.

    Returns:
        The Any result.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _expand_env_vars(data)


def load_config(path: str) -> Any:
    """Alias for load_yaml."""
    return load_yaml(path)


def estimate_model_memory_gb(model_id: str) -> int:
    """Estimate GPU memory requirement in GB from a model ID string.

    Shared utility used by model_search, live_model_search, ponder, and vram_manager.
    Returns a conservative estimate based on parameter count in the model name.

    Returns:
        The computed value.
    """
    model_lower = model_id.lower()

    # Extract parameter count patterns like 70b, 72b, 7b, 3.8b, 0.5b
    match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", model_lower)
    if match:
        params = float(match.group(1))
        # Q4 quantisation rule-of-thumb: ~0.55 GB per billion params
        # Add 2 GB overhead for KV cache + activations
        estimated = int(params * 0.55) + 2
        return max(2, estimated)

    # Explicit size keywords as fallback
    if any(x in model_lower for x in ["70b", "72b", "65b"]):
        return 40
    if any(x in model_lower for x in ["34b", "33b", "30b", "32b"]):
        return 20
    if any(x in model_lower for x in ["13b", "14b", "15b"]):
        return 10
    if any(x in model_lower for x in ["7b", "8b"]):
        return 6
    if any(x in model_lower for x in ["3b", "4b"]):
        return 3
    if any(x in model_lower for x in ["1b", "2b"]):
        return 2

    return 4  # conservative default for unknown sizes


# ---------------------------------------------------------------------------
# RFC 9457 Problem Details for HTTP APIs
# ---------------------------------------------------------------------------


def error_response(
    message: str,
    status: int = 400,
    detail: str | None = None,
    instance: str | None = None,
) -> tuple:
    """Create an RFC 9457-compliant error response for Flask.

    Args:
        message: Short human-readable summary (becomes 'title').
        status: HTTP status code.
        detail: Longer explanation (optional).
        instance: URI reference for the specific occurrence (optional).

    Returns:
        Tuple of (response_dict, status_code) suitable for Flask return.
    """
    body = {
        "type": "about:blank",
        "title": message,
        "status": status,
    }
    if detail:
        body["detail"] = detail
    if instance:
        body["instance"] = instance
    return body, status


def validate_required_fields(data: dict, fields: list) -> str | None:
    """Validate that all required fields are present in a request dict.

    Args:
        data: The request data dictionary.
        fields: List of required field names.

    Returns:
        Error message string if validation fails, None if all fields present.
    """
    if not data:
        return "Request body is required"
    missing = [f for f in fields if f not in data or data[f] is None]
    if missing:
        return f"Missing required fields: {', '.join(missing)}"
    return None
