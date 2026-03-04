import logging
import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml


def setup_logging(level=logging.INFO, log_dir="logs"):
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
    """Load a YAML file and expand ${ENV_VAR} placeholders in string values."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _expand_env_vars(data)


def load_config(path: str) -> Any:
    """Alias for load_yaml."""
    return load_yaml(path)


def estimate_model_memory_gb(model_id: str) -> int:
    """Estimate GPU memory requirement in GB from a model ID string.

    Shared utility used by model_search, live_model_search, ponder, and vram_manager.
    Returns a conservative estimate based on parameter count in the model name.
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
