"""Tests for shared backend runtime configuration helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

from vetinari.backend_config import load_backend_runtime_config, resolve_provider_fallback_order


def test_user_config_legacy_backend_sections_are_normalized(tmp_path: Path, monkeypatch) -> None:
    """Legacy top-level vllm/nim sections should load into inference_backend."""
    project_config = tmp_path / "models.yaml"
    project_config.write_text("inference_backend:\n  primary: llama_cpp\n", encoding="utf-8")

    user_config = tmp_path / "config.yaml"
    user_config.write_text(
        yaml.safe_dump(
            {
                "vllm": {"enabled": True, "endpoint": "http://localhost:8000"},
                "nim": {"enabled": False, "endpoint": "http://localhost:8001"},
                "inference": {"models_dir": str(tmp_path / "gguf")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("VETINARI_VLLM_ENDPOINT", raising=False)
    monkeypatch.delenv("VETINARI_MODELS_DIR", raising=False)
    monkeypatch.delenv("VETINARI_NATIVE_MODELS_DIR", raising=False)
    cfg = load_backend_runtime_config(project_config_path=project_config, user_config_path=user_config)

    assert cfg["inference_backend"]["vllm"]["enabled"] is True
    assert cfg["inference_backend"]["vllm"]["endpoint"] == "http://localhost:8000"
    assert cfg["local_inference"]["models_dir"] == str(tmp_path / "gguf")
    assert cfg["models"]["gguf_dir"] == str(tmp_path / "gguf")


def test_env_endpoint_overrides_config_and_enables_backend(tmp_path: Path, monkeypatch) -> None:
    """Environment endpoint overrides should become the effective runtime config."""
    project_config = tmp_path / "models.yaml"
    project_config.write_text(
        yaml.safe_dump(
            {
                "inference_backend": {
                    "primary": "vllm",
                    "vllm": {"enabled": False, "endpoint": "http://localhost:8000"},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("VETINARI_VLLM_ENDPOINT", "http://127.0.0.1:9000")
    monkeypatch.delenv("VETINARI_MODELS_DIR", raising=False)
    monkeypatch.delenv("VETINARI_NATIVE_MODELS_DIR", raising=False)
    cfg = load_backend_runtime_config(project_config_path=project_config, user_config_path=tmp_path / "missing.yaml")

    assert cfg["inference_backend"]["vllm"]["enabled"] is True
    assert cfg["inference_backend"]["vllm"]["endpoint"] == "http://127.0.0.1:9000"


def test_cache_env_overrides_are_normalized(tmp_path: Path, monkeypatch) -> None:
    """Runtime cache environment flags should reach backend extra_config."""
    project_config = tmp_path / "models.yaml"
    project_config.write_text("inference_backend:\n  primary: vllm\n", encoding="utf-8")

    monkeypatch.setenv("VETINARI_VLLM_CACHE_SALT", "secret-salt")
    monkeypatch.setenv("VETINARI_VLLM_PREFIX_CACHING_ENABLED", "true")
    monkeypatch.setenv("VETINARI_VLLM_PREFIX_CACHING_HASH_ALGO", "sha256_cbor")
    monkeypatch.setenv("NIM_ENABLE_KV_CACHE_REUSE", "1")
    monkeypatch.setenv("NIM_ENABLE_KV_CACHE_HOST_OFFLOAD", "0")
    monkeypatch.delenv("VETINARI_VLLM_ENDPOINT", raising=False)
    monkeypatch.delenv("VETINARI_NIM_ENDPOINT", raising=False)
    monkeypatch.delenv("VETINARI_MODELS_DIR", raising=False)
    monkeypatch.delenv("VETINARI_NATIVE_MODELS_DIR", raising=False)

    cfg = load_backend_runtime_config(project_config_path=project_config, user_config_path=tmp_path / "missing.yaml")

    assert cfg["inference_backend"]["vllm"]["cache_salt"] == "secret-salt"
    assert cfg["inference_backend"]["vllm"]["prefix_caching_enabled"] is True
    assert cfg["inference_backend"]["vllm"]["prefix_caching_hash_algo"] == "sha256_cbor"
    assert cfg["inference_backend"]["nim"]["kv_cache_reuse_enabled"] is True
    assert cfg["inference_backend"]["nim"]["kv_cache_host_offload_enabled"] is False


def test_provider_order_prefers_native_backends_before_local() -> None:
    """Configured native backends should sort ahead of local GGUF fallback."""
    cfg = {
        "inference_backend": {
            "primary": "vllm",
            "fallback": "nim",
            "vllm": {"enabled": True, "endpoint": "http://localhost:8000"},
            "nim": {"enabled": True, "endpoint": "http://localhost:8001"},
        }
    }

    order = resolve_provider_fallback_order(cfg, available_providers={"local", "vllm", "nim"})

    assert order == ["vllm", "nim", "local"]


def test_provider_order_respects_setup_fallback_order() -> None:
    """Setup-generated fallback_order should drive runtime provider ordering."""
    cfg = {
        "inference_backend": {
            "fallback_order": ["nim", "vllm", "llama_cpp"],
            "vllm": {"enabled": True, "endpoint": "http://localhost:8000"},
            "nim": {"enabled": True, "endpoint": "http://localhost:8001"},
        }
    }

    order = resolve_provider_fallback_order(cfg, available_providers={"local", "vllm", "nim"})

    assert order == ["nim", "vllm", "local"]


def test_env_preferred_backend_reorders_runtime_config(tmp_path: Path, monkeypatch) -> None:
    """Explicit backend preference should be honored at runtime."""
    project_config = tmp_path / "models.yaml"
    project_config.write_text(
        yaml.safe_dump(
            {
                "inference_backend": {
                    "fallback_order": ["nim", "vllm", "llama_cpp"],
                    "vllm": {"enabled": True, "endpoint": "http://localhost:8000"},
                    "nim": {"enabled": True, "endpoint": "http://localhost:8001"},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("VETINARI_PREFERRED_BACKEND", "llama_cpp")
    monkeypatch.delenv("VETINARI_MODELS_DIR", raising=False)
    monkeypatch.delenv("VETINARI_NATIVE_MODELS_DIR", raising=False)
    cfg = load_backend_runtime_config(project_config_path=project_config, user_config_path=tmp_path / "missing.yaml")

    assert cfg["inference_backend"]["primary"] == "llama_cpp"
    assert cfg["inference_backend"]["fallback_order"] == ["llama_cpp", "nim", "vllm"]
