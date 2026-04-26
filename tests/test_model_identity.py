"""Regression tests for model artifact identity and cache isolation."""

from __future__ import annotations

import dataclasses
import hashlib
import threading
from pathlib import Path
from unittest.mock import patch

from vetinari.adapters.base import InferenceRequest, ProviderConfig, ProviderType
from vetinari.adapters.llama_cpp_adapter import LlamaCppProviderAdapter, _semantic_cache_identity
from vetinari.models.kv_state_cache import KVStateCache, hash_system_prompt
from vetinari.models.model_profiler_data import _config_path, _load_cached_profile, _save_profile
from vetinari.models.model_profiler_schemas import GGUFMetadata
from vetinari.models.model_registry import ModelRegistry
from vetinari.training.continual_learning import LoRAAdapterManager, ReplayBuffer


def _local_config(models_dir: Path) -> ProviderConfig:
    return ProviderConfig(
        name="test-local",
        provider_type=ProviderType.LOCAL,
        endpoint="local",
        extra_config={"models_dir": str(models_dir)},
    )


def test_kv_cache_invalidate_handles_colon_model_ids() -> None:
    cache = KVStateCache()
    prompt_hash = hash_system_prompt("system")

    cache.put("provider:model", prompt_hash, b"state")

    assert cache.invalidate("provider:model") == 1
    assert cache.get_stats()["size"] == 0


def test_llama_cpp_resolution_rejects_substring_and_fallback(tmp_path: Path) -> None:
    exact_model = tmp_path / "qwen2.5-coder.gguf"
    fallback_model = tmp_path / "tiny.gguf"
    exact_model.write_bytes(b"exact")
    fallback_model.write_bytes(b"fallback")

    adapter = LlamaCppProviderAdapter(_local_config(tmp_path))
    adapter.discover_models()

    assert adapter._resolve_model_path("qwen2.5-coder") == exact_model
    assert adapter._resolve_model_path("qwen2.5") is None
    assert adapter._resolve_model_path("missing-model") is None


def test_semantic_cache_identity_includes_artifact_and_sampler_params(tmp_path: Path) -> None:
    model_path = tmp_path / "local-model.gguf"
    model_path.write_bytes(b"artifact-v1")
    request = InferenceRequest(
        model_id="local-model",
        prompt="same prompt",
        system_prompt="system-a",
        temperature=0.2,
        top_p=0.8,
    )

    model_key_a, context_a = _semantic_cache_identity(
        request,
        model_path=model_path,
        resolved_model_id="local-model",
        resolution_outcome="exact",
    )
    model_key_b, context_b = _semantic_cache_identity(
        dataclasses.replace(request, temperature=0.3),
        model_path=model_path,
        resolved_model_id="local-model",
        resolution_outcome="exact",
    )

    digest = hashlib.sha256(b"artifact-v1").hexdigest()
    assert digest in model_key_a
    assert model_key_a == model_key_b
    assert context_a != context_b


def test_model_registry_records_artifact_provenance_and_header_quantization(tmp_path: Path) -> None:
    model_path = tmp_path / "artifact-model.gguf"
    model_bytes = b"gguf-bytes"
    model_path.write_bytes(model_bytes)

    with patch.object(ModelRegistry, "_load_static_config"):
        registry = ModelRegistry()
    registry._models_dir = str(tmp_path)

    metadata = GGUFMetadata(
        architecture="llama",
        block_count=32,
        context_length=8192,
        file_type=7,
        quantization="q8_0",
    )
    with (
        patch("vetinari.models.model_registry.read_metadata", return_value=metadata),
        patch("vetinari.utils.estimate_model_memory_gb", return_value=1),
    ):
        registry._do_refresh()

    entry = registry._models["artifact-model"]
    expected_hash = hashlib.sha256(model_bytes).hexdigest()
    assert entry.endpoint == str(model_path)
    assert entry.artifact_path == str(model_path.resolve())
    assert entry.artifact_sha256 == expected_hash
    assert entry.quantization == "q8_0"
    assert entry.gguf_metadata["file_type"] == 7
    assert entry.to_dict()["artifact_sha256"] == expected_hash


def test_model_profile_cache_is_atomic_and_corruption_visible(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VETINARI_USER_DIR", str(tmp_path))

    _save_profile("identity-model", {"model_id": "identity-model", "ok": True})
    assert _load_cached_profile("identity-model")["ok"] is True

    corrupt_path = _config_path("broken-model")
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text("{not-json", encoding="utf-8")

    assert _load_cached_profile("broken-model") is None
    assert not corrupt_path.exists()
    assert list(corrupt_path.parent.glob("broken-model.json.corrupt.*"))


def test_thompson_model_arms_keep_colon_model_ids_distinct() -> None:
    from vetinari.learning.model_selector import ThompsonSamplingSelector

    selector = ThompsonSamplingSelector.__new__(ThompsonSamplingSelector)
    selector._arms = {}
    selector._lock = threading.Lock()
    selector._update_count = 0
    selector._get_informed_prior = lambda _model_id, _task_type: (1.0, 1.0)

    selector.update("provider:model", "coding", quality_score=0.9, success=True)
    selector.update("provider", "model:coding", quality_score=0.1, success=False)

    rankings = selector.get_rankings("coding")
    assert len(rankings) == 1
    assert rankings[0][0] == "provider:model"
    assert len(selector._arms) == 2


def test_training_defaults_use_configured_user_dir_lazily(tmp_path: Path, monkeypatch) -> None:
    import vetinari.training.data_seeder as data_seeder

    monkeypatch.setenv("VETINARI_USER_DIR", str(tmp_path / "configured"))

    assert ReplayBuffer().buffer_path == tmp_path / "configured" / "replay_buffer.jsonl"
    assert LoRAAdapterManager().adapters_dir == tmp_path / "configured" / "adapters"
    assert data_seeder.TrainingDataSeeder().get_seed_status()["data_dir"] == str(
        tmp_path / "configured" / "training_data"
    )


def test_replay_and_adapter_registry_corruption_is_visible(tmp_path: Path) -> None:
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text('{"ok": true}\n{not-json}\n', encoding="utf-8")

    try:
        ReplayBuffer(buffer_path=replay_path)
    except ValueError as exc:
        assert "malformed JSONL" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("corrupt replay buffer must fail closed")

    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{not-json", encoding="utf-8")
    try:
        LoRAAdapterManager(adapters_dir=tmp_path)
    except ValueError as exc:
        assert "registry is corrupt" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("corrupt adapter registry must fail closed")

    assert list(tmp_path.glob("replay.jsonl.corrupt.*"))
    assert list(tmp_path.glob("registry.json.corrupt.*"))
