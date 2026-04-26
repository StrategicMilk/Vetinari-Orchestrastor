"""
Tests for model discovery retry logic and resilience.
Tests filesystem scan, fallback to static models, and health tracking.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Remove incomplete stubs left by earlier test files so real modules load
sys.modules.pop("vetinari.model_pool", None)

from tests.factories import make_mock_adapter_model_info as _make_mock_model_info
from vetinari.model_discovery import ModelDiscovery
from vetinari.models.model_pool import ModelPool


def _fake_huggingface_module(payload: bytes = b"GGUFpayload") -> object:
    digest = hashlib.sha256(payload).hexdigest()

    class _Lfs:
        size = len(payload)
        sha256 = digest

    class _Sibling:
        rfilename = "model.gguf"
        size = len(payload)
        lfs = _Lfs()

    class _Info:
        sha = "a" * 40
        siblings = [_Sibling()]

    class _HfApi:
        def model_info(self, **_kwargs):
            return _Info()

    def _download(*, filename: str, local_dir: str, **_kwargs) -> str:
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return str(target)

    return types.SimpleNamespace(HfApi=_HfApi, hf_hub_download=_download)


def _fake_native_huggingface_module() -> object:
    files = {
        "config.json": b'{"model_type":"qwen2"}',
        "tokenizer.json": b'{"tokenizer":"ok"}',
        "model.safetensors": b"native weights",
        "README.md": b"not downloaded",
    }
    weight_digest = hashlib.sha256(files["model.safetensors"]).hexdigest()

    class _Lfs:
        size = len(files["model.safetensors"])
        sha256 = weight_digest

    class _Sibling:
        def __init__(self, rfilename: str) -> None:
            self.rfilename = rfilename
            self.size = len(files[rfilename])
            self.lfs = _Lfs() if rfilename == "model.safetensors" else None

    class _Info:
        sha = "c" * 40
        siblings = [_Sibling(name) for name in files]

    class _HfApi:
        def model_info(self, **_kwargs):
            return _Info()

    def _snapshot_download(*, local_dir: str, allow_patterns: list[str], **_kwargs) -> str:
        root = Path(local_dir)
        for filename in allow_patterns:
            target = root / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(files[filename])
        return str(root)

    return types.SimpleNamespace(HfApi=_HfApi, snapshot_download=_snapshot_download)


class TestManagedModelDownloads:
    def test_download_model_writes_provenance_marker(self, tmp_path: Path) -> None:
        with patch.dict(sys.modules, {"huggingface_hub": _fake_huggingface_module()}):
            result = ModelDiscovery(cache_dir=str(tmp_path / "cache")).download_model(
                "owner/repo",
                "model.gguf",
                models_dir=tmp_path / "models",
            )

        model_path = Path(result["path"])
        marker_path = model_path.with_name(f"{model_path.name}.vetinari-download.json")
        assert model_path.read_bytes().startswith(b"GGUF")
        assert marker_path.exists()
        assert result["revision"] == "a" * 40
        assert result["sha256"] == hashlib.sha256(b"GGUFpayload").hexdigest()

    def test_download_model_rejects_invalid_existing_file(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "model.gguf").write_bytes(b"not-a-model")

        with patch.dict(sys.modules, {"huggingface_hub": _fake_huggingface_module()}):
            with pytest.raises(ValueError, match="not a valid GGUF"):
                ModelDiscovery(cache_dir=str(tmp_path / "cache")).download_model(
                    "owner/repo",
                    "model.gguf",
                    models_dir=models_dir,
                )

    def test_download_native_snapshot_writes_manifest(self, tmp_path: Path) -> None:
        with patch.dict(sys.modules, {"huggingface_hub": _fake_native_huggingface_module()}):
            result = ModelDiscovery(cache_dir=str(tmp_path / "cache")).download_model(
                "owner/Qwen2.5-Coder-7B",
                models_dir=tmp_path / "native",
                backend="vllm",
                model_format="safetensors",
            )

        snapshot_path = Path(result["path"])
        manifest_path = snapshot_path / ".vetinari-download.json"
        assert snapshot_path.is_dir()
        assert (snapshot_path / "config.json").exists()
        assert (snapshot_path / "model.safetensors").exists()
        assert not (snapshot_path / "README.md").exists()
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["repo_id"] == "owner/Qwen2.5-Coder-7B"
        assert manifest["backend"] == "vllm"
        assert manifest["format"] == "safetensors"
        assert manifest["revision"] == "c" * 40
        assert {file["filename"] for file in manifest["files"]} >= {"config.json", "model.safetensors"}

    def test_get_repo_files_filters_native_snapshot_files(self, tmp_path: Path) -> None:
        with patch.dict(sys.modules, {"huggingface_hub": _fake_native_huggingface_module()}):
            files = ModelDiscovery(cache_dir=str(tmp_path / "cache")).get_repo_files(
                "owner/Qwen2.5-Coder-7B",
                backend="vllm",
                model_format="awq",
                objective="coding",
                family="qwen",
                file_type="safetensors",
            )

        assert [file["filename"] for file in files] == ["model.safetensors"]
        assert files[0]["artifact_type"] == "snapshot_file"
        assert files[0]["backend"] == "vllm"
        assert files[0]["format"] == "awq"
        assert files[0]["quantization"] == "AWQ"

    def test_download_status_survives_restart_as_interrupted(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        state_path = cache_dir / "download_jobs.json"
        state_path.write_text(
            json.dumps({
                "version": 1,
                "jobs": {
                    "lost-job": {
                        "download_id": "lost-job",
                        "status": "running",
                        "repo_id": "owner/repo",
                        "filename": "model.gguf",
                        "revision": "a" * 40,
                    }
                },
            }),
            encoding="utf-8",
        )

        status = ModelDiscovery(cache_dir=str(cache_dir)).get_download_status("lost-job")

        assert status is not None
        assert status["status"] == "interrupted"
        assert "process exited" in status["error"]
        persisted = json.loads(state_path.read_text(encoding="utf-8"))
        assert persisted["jobs"]["lost-job"]["status"] == "interrupted"


class TestModelDiscoveryRetry:
    """Test model discovery with retry logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "models": [
                {
                    "id": "static-model-1",
                    "name": "Static Model 1",
                    "capabilities": ["code_gen"],
                    "memory_gb": 4,
                }
            ],
            "memory_budget_gb": 48,
        }

    def test_successful_discovery_first_attempt(self):
        """Verify successful discovery on first attempt."""
        pool = ModelPool(self.config)

        mock_models = [_make_mock_model_info("qwen-model", 8)]

        with patch("vetinari.adapters.llama_cpp_adapter.LlamaCppProviderAdapter") as MockAdapter:
            MockAdapter.return_value.discover_models.return_value = mock_models
            pool.discover_models()

        assert len(pool.models) >= 1
        assert not pool._discovery_failed
        assert pool._discovery_retry_count == 1

    def test_discovery_retry_on_failure(self):
        """Verify retry happens on exception and succeeds on last attempt."""
        pool = ModelPool(self.config)
        pool._max_discovery_retries = 3
        pool._discovery_retry_delay_base = 0.0  # No sleep in tests

        mock_models = [_make_mock_model_info("recovered-model", 6)]

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise OSError("Scan failed")
            mock = MagicMock()
            mock.discover_models.return_value = mock_models
            return mock

        with patch("vetinari.adapters.llama_cpp_adapter.LlamaCppProviderAdapter", side_effect=side_effect):
            pool.discover_models()

        # Should retry and eventually succeed on 3rd attempt
        assert call_count[0] == 3
        assert len(pool.models) >= 1
        assert not pool._discovery_failed

    def test_discovery_fallback_after_max_retries(self):
        """Verify fallback to static models after max retries."""
        pool = ModelPool(self.config)
        pool._discovery_retry_delay_base = 0.0

        with patch("vetinari.adapters.llama_cpp_adapter.LlamaCppProviderAdapter", side_effect=OSError("Scan failed")):
            pool.discover_models()

        # Should have fallen back to static models
        assert pool._discovery_failed
        assert pool._fallback_active
        assert len(pool.models) >= 1
        assert any(m["id"] == "static-model-1" for m in pool.models)


class TestModelDiscoveryFiltering:
    """Test model discovery filtering by memory budget."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "models": [],
            "memory_budget_gb": 16,
        }

    def test_memory_budget_filtering(self):
        """Verify models exceeding memory budget are filtered."""
        pool = ModelPool(self.config, memory_budget_gb=16)

        mock_models = [
            _make_mock_model_info("small-model", 8),
            _make_mock_model_info("large-model", 32),  # Exceeds budget
            _make_mock_model_info("medium-model", 12),
        ]

        with patch("vetinari.adapters.llama_cpp_adapter.LlamaCppProviderAdapter") as MockAdapter:
            MockAdapter.return_value.discover_models.return_value = mock_models
            pool.discover_models()

        model_ids = [m["id"] for m in pool.models]
        assert "small-model" in model_ids
        assert "medium-model" in model_ids
        assert "large-model" not in model_ids  # Filtered out

    def test_default_memory_when_missing(self):
        """Verify default memory assignment when memory_gb is falsy."""
        pool = ModelPool(self.config)

        mock_models = [_make_mock_model_info("model-no-memory-info", 0)]  # 0 = unknown

        with patch("vetinari.adapters.llama_cpp_adapter.LlamaCppProviderAdapter") as MockAdapter:
            MockAdapter.return_value.discover_models.return_value = mock_models
            pool.discover_models()

        discovered = [m for m in pool.models if m["id"] == "model-no-memory-info"]
        assert len(discovered) == 1
        # Default is max(2.0, budget * 0.25); with budget=16 that is 4.0
        assert discovered[0]["memory_gb"] == max(2.0, self.config["memory_budget_gb"] * 0.25)


class TestModelDiscoveryHealth:
    """Test discovery health tracking."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {"models": [], "memory_budget_gb": 32}

    def test_get_discovery_health_success(self):
        """Verify health reporting on success."""
        pool = ModelPool(self.config)

        mock_models = [_make_mock_model_info("test-model", 4)]

        with patch("vetinari.adapters.llama_cpp_adapter.LlamaCppProviderAdapter") as MockAdapter:
            MockAdapter.return_value.discover_models.return_value = mock_models
            pool.discover_models()

        health = pool.get_discovery_health()

        assert health["discovery_failed"] is False
        assert health["fallback_active"] is False
        assert health["models_available"] >= 1
        assert health["retry_count"] == 1
        assert health["last_error"] is None

    def test_get_discovery_health_failure(self):
        """Verify health reporting on failure."""
        pool = ModelPool(self.config)
        pool._discovery_retry_delay_base = 0.0

        with patch("vetinari.adapters.llama_cpp_adapter.LlamaCppProviderAdapter", side_effect=OSError("Scan failed")):
            pool.discover_models()

        health = pool.get_discovery_health()

        assert health["discovery_failed"] is True
        assert health["fallback_active"] is True
        assert health["last_error"] is not None
        assert health["retry_count"] > 0


class TestStaticModelInclusion:
    """Test that static models are always included."""

    def test_static_models_included_on_success(self):
        """Verify static models are included with discovered models."""
        config = {
            "models": [{"id": "static-1", "name": "Static 1", "memory_gb": 4}],
            "memory_budget_gb": 32,
        }
        pool = ModelPool(config)

        mock_models = [_make_mock_model_info("discovered-1", 4)]

        with patch("vetinari.adapters.llama_cpp_adapter.LlamaCppProviderAdapter") as MockAdapter:
            MockAdapter.return_value.discover_models.return_value = mock_models
            pool.discover_models()

        model_ids = [m["id"] for m in pool.models]
        assert "static-1" in model_ids
        assert "discovered-1" in model_ids

    def test_static_models_on_discovery_failure(self):
        """Verify static models are available when discovery fails."""
        config = {
            "models": [{"id": "static-fallback", "name": "Fallback", "memory_gb": 4}],
            "memory_budget_gb": 32,
        }
        pool = ModelPool(config)
        pool._discovery_retry_delay_base = 0.0

        with patch(
            "vetinari.adapters.llama_cpp_adapter.LlamaCppProviderAdapter", side_effect=Exception("Discovery failed")
        ):
            pool.discover_models()

        model_ids = [m["id"] for m in pool.models]
        assert "static-fallback" in model_ids
        assert pool._fallback_active


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
