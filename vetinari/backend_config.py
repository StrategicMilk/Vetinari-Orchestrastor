"""Shared backend configuration helpers for local, vLLM, and NIM runtimes.

Normalizes configuration from the project config, user config, and environment
variables so setup, health checks, and runtime registration agree on one shape.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from vetinari.config_paths import resolve_config_path
from vetinari.constants import (
    DEFAULT_NATIVE_MODELS_DIR,
    OPERATOR_MODELS_CACHE_DIR,
    get_user_dir,
)

logger = logging.getLogger(__name__)

_PROJECT_MODELS_CONFIG = resolve_config_path("models.yaml")

_DEFAULT_BACKEND_CONFIG: dict[str, Any] = {
    "selection_policy": "configured",
    "primary": "vllm",
    "fallback": "nim",
    "llama_cpp_use_cases": [
        "explicit_user_preference",
        "weak_or_no_server_setup",
        "gguf_only_models",
        "cpu_ram_vram_offload",
        "oversized_local_models",
        "recovery_fallback",
    ],
    "native_models_dir": DEFAULT_NATIVE_MODELS_DIR,
    "vllm": {
        "enabled": False,
        "endpoint": "http://localhost:8000",
        "gpu_only": True,
        "semantic_cache_enabled": True,
        "cache_namespace": "vetinari",
        "cache_salt": "",
        "prefix_caching_enabled": True,
        "prefix_caching_hash_algo": "sha256",
        "container_setup": {},
    },
    "nim": {
        "enabled": False,
        "endpoint": "http://localhost:8001",
        "gpu_only": True,
        "semantic_cache_enabled": True,
        "cache_namespace": "vetinari",
        "kv_cache_host_offload_enabled": None,
        "kv_cache_reuse_enabled": False,
        "supports_cache_salt": False,
    },
}

_DEFAULT_LOCAL_INFERENCE_CONFIG: dict[str, Any] = {
    "models_dir": OPERATOR_MODELS_CACHE_DIR,  # noqa: VET306 — config default, runtime overrides via settings
    "gpu_layers": -1,
    "context_length": 8192,
}


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk, returning an empty dict on failure."""
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore[import-untyped]

        with path.open(encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        return loaded if isinstance(loaded, dict) else {}
    except Exception as exc:
        logger.warning("Could not load YAML config from %s — using defaults: %s", path, exc)
        return {}


def _merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` and return a new dict."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce_env_bool(value: str) -> bool:
    """Coerce an environment flag string into a boolean."""
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _normalize_backend_name(value: Any) -> str:
    """Normalize backend aliases used by config and environment variables."""
    if value is None:
        return ""
    normalized_name = str(value).strip().lower().replace("-", "_")
    if normalized_name in {"local", "llama", "llamacpp", "llama_cpp"}:
        return "llama_cpp"
    if normalized_name in {"nims", "nvidia_nim", "nvidia_nims"}:
        return "nim"
    return normalized_name


def _backend_order_from_primary_fallback(backend: dict[str, Any]) -> list[str]:
    """Build an explicit order from legacy primary/fallback keys."""
    order: list[str] = []
    for backend_name in (backend.get("primary"), backend.get("fallback"), "llama_cpp"):
        if backend_name is None:
            continue
        normalized_name = _normalize_backend_name(backend_name)
        if normalized_name and normalized_name not in order:
            order.append(normalized_name)
    return order


def _normalize_user_config(config: dict[str, Any]) -> dict[str, Any]:
    """Translate legacy user-config keys into the runtime backend shape."""
    normalized: dict[str, Any] = {}

    if isinstance(config.get("models"), dict):
        normalized["models"] = dict(config["models"])

    inference = config.get("inference")
    if isinstance(inference, dict):
        local = {}
        if "models_dir" in inference:
            local["models_dir"] = inference["models_dir"]
            normalized.setdefault("models", {})["gguf_dir"] = inference["models_dir"]
        if "gpu_layers" in inference:
            local["gpu_layers"] = inference["gpu_layers"]
        if "context_length" in inference:
            local["context_length"] = inference["context_length"]
        if local:
            normalized["local_inference"] = local

    if isinstance(config.get("local_inference"), dict):
        normalized["local_inference"] = _merge_dict(
            normalized.get("local_inference", {}),
            config["local_inference"],
        )

    backend: dict[str, Any] = {}
    if isinstance(config.get("inference_backend"), dict):
        backend = _merge_dict(backend, config["inference_backend"])

    for backend_name in ("vllm", "nim"):
        legacy_cfg = config.get(backend_name)
        if isinstance(legacy_cfg, dict):
            backend[backend_name] = _merge_dict(backend.get(backend_name, {}), legacy_cfg)

    if ("primary" in backend or "fallback" in backend) and "fallback_order" not in backend:
        backend["fallback_order"] = _backend_order_from_primary_fallback(backend)

    if backend:
        normalized["inference_backend"] = backend

    return normalized


def load_backend_runtime_config(
    *,
    project_config_path: Path | None = None,
    user_config_path: Path | None = None,
) -> dict[str, Any]:
    """Return normalized backend config from project, user, and env sources.

    Args:
        project_config_path: Optional override for the project ``config/models.yaml`` path.
        user_config_path: Optional override for the user ``~/.vetinari/config.yaml`` path.

    Returns:
        Normalized runtime config covering backend enablement, endpoint URLs,
        and local/native model directories.
    """
    config: dict[str, Any] = {
        "inference_backend": dict(_DEFAULT_BACKEND_CONFIG),
        "local_inference": dict(_DEFAULT_LOCAL_INFERENCE_CONFIG),
        "models": {
            "gguf_dir": OPERATOR_MODELS_CACHE_DIR,  # noqa: VET306 — config default, runtime override via yaml
            "native_dir": DEFAULT_NATIVE_MODELS_DIR,
        },
    }

    project_loaded = _load_yaml_dict(project_config_path or _PROJECT_MODELS_CONFIG)
    if project_loaded:
        if isinstance(project_loaded.get("hardware"), dict):
            config["hardware"] = dict(project_loaded["hardware"])
        if isinstance(project_loaded.get("inference_backend"), dict):
            config["inference_backend"] = _merge_dict(config["inference_backend"], project_loaded["inference_backend"])
        if isinstance(project_loaded.get("local_inference"), dict):
            config["local_inference"] = _merge_dict(config["local_inference"], project_loaded["local_inference"])
        if isinstance(project_loaded.get("models"), dict):
            config["models"] = _merge_dict(config["models"], project_loaded["models"])

    user_loaded = _normalize_user_config(_load_yaml_dict(user_config_path or (get_user_dir() / "config.yaml")))
    if user_loaded:
        if isinstance(user_loaded.get("inference_backend"), dict):
            config["inference_backend"] = _merge_dict(config["inference_backend"], user_loaded["inference_backend"])
        if isinstance(user_loaded.get("local_inference"), dict):
            config["local_inference"] = _merge_dict(config["local_inference"], user_loaded["local_inference"])
        if isinstance(user_loaded.get("models"), dict):
            config["models"] = _merge_dict(config["models"], user_loaded["models"])

    gguf_dir = os.environ.get("VETINARI_MODELS_DIR")
    if gguf_dir:
        config["local_inference"]["models_dir"] = gguf_dir
        config["models"]["gguf_dir"] = gguf_dir

    native_dir = os.environ.get("VETINARI_NATIVE_MODELS_DIR")
    if native_dir:
        config["inference_backend"]["native_models_dir"] = native_dir
        config["models"]["native_dir"] = native_dir
    else:
        config["models"]["native_dir"] = config["inference_backend"].get(
            "native_models_dir",
            config["models"]["native_dir"],
        )

    vllm_endpoint = os.environ.get("VETINARI_VLLM_ENDPOINT")
    if vllm_endpoint:
        config["inference_backend"]["vllm"] = _merge_dict(
            config["inference_backend"].get("vllm", {}),
            {"enabled": True, "endpoint": vllm_endpoint},
        )

    vllm_cache_salt = os.environ.get("VETINARI_VLLM_CACHE_SALT")
    if vllm_cache_salt is not None:
        config["inference_backend"]["vllm"] = _merge_dict(
            config["inference_backend"].get("vllm", {}),
            {"cache_salt": vllm_cache_salt},
        )

    vllm_prefix_caching = os.environ.get("VETINARI_VLLM_PREFIX_CACHING_ENABLED")
    if vllm_prefix_caching is not None:
        config["inference_backend"]["vllm"] = _merge_dict(
            config["inference_backend"].get("vllm", {}),
            {"prefix_caching_enabled": _coerce_env_bool(vllm_prefix_caching)},
        )

    vllm_prefix_hash_algo = os.environ.get("VETINARI_VLLM_PREFIX_CACHING_HASH_ALGO")
    if vllm_prefix_hash_algo:
        config["inference_backend"]["vllm"] = _merge_dict(
            config["inference_backend"].get("vllm", {}),
            {"prefix_caching_hash_algo": vllm_prefix_hash_algo},
        )

    nim_endpoint = os.environ.get("VETINARI_NIM_ENDPOINT")
    if nim_endpoint:
        config["inference_backend"]["nim"] = _merge_dict(
            config["inference_backend"].get("nim", {}),
            {"enabled": True, "endpoint": nim_endpoint},
        )

    nim_kv_reuse = os.environ.get("NIM_ENABLE_KV_CACHE_REUSE")
    if nim_kv_reuse is not None:
        config["inference_backend"]["nim"] = _merge_dict(
            config["inference_backend"].get("nim", {}),
            {"kv_cache_reuse_enabled": _coerce_env_bool(nim_kv_reuse)},
        )

    nim_kv_host_offload = os.environ.get("NIM_ENABLE_KV_CACHE_HOST_OFFLOAD")
    if nim_kv_host_offload is not None:
        config["inference_backend"]["nim"] = _merge_dict(
            config["inference_backend"].get("nim", {}),
            {"kv_cache_host_offload_enabled": _coerce_env_bool(nim_kv_host_offload)},
        )

    preferred_backend = _normalize_backend_name(
        os.environ.get("VETINARI_PREFERRED_BACKEND") or os.environ.get("VETINARI_INFERENCE_BACKEND")
    )
    if preferred_backend:
        existing_order = config["inference_backend"].get("fallback_order")
        if not isinstance(existing_order, list):
            existing_order = _backend_order_from_primary_fallback(config["inference_backend"])
        fallback_order = [preferred_backend]
        for backend_name in existing_order:
            normalized_name = _normalize_backend_name(backend_name)
            if normalized_name and normalized_name not in fallback_order:
                fallback_order.append(normalized_name)
        if "llama_cpp" not in fallback_order:
            fallback_order.append("llama_cpp")
        config["inference_backend"]["primary"] = preferred_backend
        config["inference_backend"]["fallback"] = fallback_order[1] if len(fallback_order) > 1 else "llama_cpp"
        config["inference_backend"]["fallback_order"] = fallback_order

    return config


def resolve_provider_fallback_order(
    config: dict[str, Any],
    available_providers: set[str] | None = None,
) -> list[str]:
    """Return the preferred provider order for runtime fallback.

    Args:
        config: Runtime config dict produced by ``load_backend_runtime_config``.
        available_providers: Optional provider names to filter the final order by.

    Returns:
        Provider names in preferred fallback order, limited to configured and
        currently enabled backends.
    """
    backend_cfg = config.get("inference_backend", {})
    name_map = {
        "llama_cpp": "local",
        "local": "local",
        "vllm": "vllm",
        "nim": "nim",
        "litellm": "litellm",
    }

    order: list[str] = []

    def add_backend(backend_name: str | None) -> None:
        """Append a configured backend to the fallback order when it is usable."""
        if backend_name is None:
            return
        normalized_name = _normalize_backend_name(backend_name)
        if normalized_name in {"vllm", "nim"}:
            section = backend_cfg.get(normalized_name, {})
            if not (isinstance(section, dict) and section.get("enabled") and section.get("endpoint")):
                return
        provider_name = name_map.get(normalized_name)
        if provider_name and provider_name not in order:
            order.append(provider_name)

    configured_order = backend_cfg.get("fallback_order")
    if isinstance(configured_order, list):
        for backend_name in configured_order:
            add_backend(str(backend_name))
    else:
        add_backend(backend_cfg.get("primary"))
        add_backend(backend_cfg.get("fallback"))

    for backend_name in ("vllm", "nim"):
        section = backend_cfg.get(backend_name, {})
        if isinstance(section, dict) and section.get("enabled") and section.get("endpoint"):
            add_backend(backend_name)

    add_backend("llama_cpp")

    if available_providers is not None:
        order = [provider_name for provider_name in order if provider_name in available_providers]

    return order
