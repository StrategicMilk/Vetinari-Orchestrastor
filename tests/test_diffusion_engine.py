"""Tests for the in-process diffusion engine.

Covers DiffusionEngine model discovery, generation, and VRAM management
with mocked diffusers to avoid requiring actual model files or GPU.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vetinari.image.diffusion_engine import (
    DiffusionEngine,
    _detect_model_type,
)

# ── Utility tests ────────────────────────────────────────────────────────────


class TestModelTypeDetection:
    """Test model type detection from directory names."""

    def test_sdxl_detection(self):
        assert _detect_model_type(Path("stable-diffusion-xl-base-1.0")) == "sdxl"

    def test_flux_detection(self):
        assert _detect_model_type(Path("flux-1-dev")) == "flux"

    def test_sd3_detection(self):
        assert _detect_model_type(Path("stable-diffusion-3-medium")) == "sd3"

    def test_sd15_detection(self):
        assert _detect_model_type(Path("stable-diffusion-v1-5")) == "sd15"

    def test_unknown_model(self):
        assert _detect_model_type(Path("totally-unknown-model")) == "auto"


# ── DiffusionEngine tests ───────────────────────────────────────────────────


class TestDiffusionEngine:
    """Test the diffusion engine."""

    def test_init_default(self):
        engine = DiffusionEngine()
        assert engine._output_dir is not None
        assert engine._pipeline is None

    @patch("vetinari.image.diffusion_engine._DIFFUSERS_AVAILABLE", False)
    def test_not_available_without_diffusers(self):
        engine = DiffusionEngine()
        assert engine.is_available() is False

    @patch("vetinari.image.diffusion_engine._PIL_AVAILABLE", False)
    def test_not_available_without_pillow(self):
        engine = DiffusionEngine()
        assert engine.is_available() is False

    def test_discover_models_empty_dir(self, tmp_path):
        engine = DiffusionEngine(models_dir=str(tmp_path))
        models = engine.discover_models()
        assert models == []

    def test_discover_models_nonexistent_dir(self):
        engine = DiffusionEngine(models_dir="/nonexistent/path")
        models = engine.discover_models()
        assert models == []

    def test_discover_hf_format_models(self, tmp_path):
        # Create fake HuggingFace-format model directory
        model_dir = tmp_path / "stable-diffusion-xl-base-1.0"
        model_dir.mkdir()
        (model_dir / "model_index.json").write_text("{}", encoding="utf-8")

        engine = DiffusionEngine(models_dir=str(tmp_path))
        models = engine.discover_models()

        assert len(models) == 1
        assert models[0]["id"] == "stable-diffusion-xl-base-1.0"
        assert models[0]["type"] == "sdxl"
        assert models[0]["format"] == "diffusers"

    def test_discover_safetensors_models(self, tmp_path):
        # Create fake safetensors file
        (tmp_path / "flux-1-dev.safetensors").write_bytes(b"\x00" * 100)

        engine = DiffusionEngine(models_dir=str(tmp_path))
        models = engine.discover_models()

        assert len(models) == 1
        assert models[0]["id"] == "flux-1-dev"
        assert models[0]["type"] == "flux"
        assert models[0]["format"] == "safetensors"

    @patch("vetinari.image.diffusion_engine._DIFFUSERS_AVAILABLE", False)
    def test_generate_without_diffusers(self):
        engine = DiffusionEngine()
        result = engine.generate(prompt="test")

        assert result["success"] is False
        assert "not installed" in result["error"]

    @patch("vetinari.image.diffusion_engine._DIFFUSERS_AVAILABLE", True)
    @patch("vetinari.image.diffusion_engine._TORCH_AVAILABLE", True)
    @patch("vetinari.image.diffusion_engine._PIL_AVAILABLE", True)
    @patch("vetinari.image.diffusion_engine.torch")
    @patch("vetinari.image.diffusion_engine.diffusers")
    def test_generate_success(self, mock_diffusers, mock_torch, tmp_path):
        # Setup mock model directory
        model_dir = tmp_path / "models" / "test-model"
        model_dir.mkdir(parents=True)
        (model_dir / "model_index.json").write_text("{}", encoding="utf-8")

        # Mock torch
        mock_torch.cuda.is_available.return_value = True
        mock_torch.float16 = "float16"
        mock_torch.Generator.return_value.manual_seed.return_value = MagicMock()

        # Mock pipeline
        mock_image = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.images = [mock_image]
        mock_diffusers.AutoPipelineForText2Image.from_pretrained.return_value = mock_pipeline

        output_dir = tmp_path / "output"
        engine = DiffusionEngine(
            models_dir=str(tmp_path / "models"),
            output_dir=str(output_dir),
        )
        result = engine.generate(prompt="a test image", seed=42)

        assert result["success"] is True
        assert result["filename"].endswith(".png")
        mock_image.save.assert_called_once()

    def test_unload_pipeline(self):
        engine = DiffusionEngine()
        engine._pipeline = MagicMock()
        engine._loaded_model_id = "test"

        engine._unload_pipeline()

        assert engine._pipeline is None
        assert engine._loaded_model_id is None

    def test_get_loaded_model_none(self):
        engine = DiffusionEngine()
        assert engine.get_loaded_model() is None

    def test_get_loaded_model_after_load(self):
        engine = DiffusionEngine()
        engine._loaded_model_id = "test-model"
        assert engine.get_loaded_model() == "test-model"


# ── New tests for output-dir writability, discovery correctness, has_models ──


class TestOutputDirWritability:
    """Test that an unwritable output dir is handled gracefully."""

    def test_unwritable_output_dir_does_not_raise(self, tmp_path: Path) -> None:
        """Constructor must not raise when output dir cannot be created."""
        unwritable = tmp_path / "no_write"
        with patch("vetinari.image.diffusion_engine.Path.mkdir", side_effect=PermissionError("denied")):
            engine = DiffusionEngine(output_dir=unwritable)
        assert engine._output_dir_writable is False

    def test_generate_fails_when_output_dir_unwritable(self, tmp_path: Path) -> None:
        """generate() must return success=False immediately when output dir is unwritable."""
        with patch("vetinari.image.diffusion_engine.Path.mkdir", side_effect=PermissionError("denied")):
            with patch("vetinari.image.diffusion_engine.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = True
                engine = DiffusionEngine(output_dir=tmp_path / "bad")

        result = engine.generate("a cat")
        assert result["success"] is False
        assert "not writable" in result["error"]


class TestDiscoverModelsCorrectness:
    """Test discovery edge cases fixed in the bug-fix pass."""

    def test_discover_hf_models_top_level_only(self, tmp_path: Path) -> None:
        """HF dirs must be discovered at depth=1, not recursively."""
        hf_model = tmp_path / "my_sd_model"
        hf_model.mkdir()
        (hf_model / "model_index.json").write_text("{}", encoding="utf-8")

        # Nested dir inside the HF model — must NOT be discovered as its own model
        nested = hf_model / "unet"
        nested.mkdir()
        (nested / "model_index.json").write_text("{}", encoding="utf-8")

        engine = DiffusionEngine(models_dir=tmp_path, output_dir=tmp_path / "out")
        models = engine.discover_models()
        ids = [m["id"] for m in models]
        assert "my_sd_model" in ids
        assert "unet" not in ids

    def test_discover_nested_safetensors(self, tmp_path: Path) -> None:
        """Safetensors files nested in subdirs (but not inside HF dirs) must be found."""
        sub = tmp_path / "checkpoints"
        sub.mkdir()
        ckpt = sub / "v1-5-pruned.safetensors"
        ckpt.write_bytes(b"")

        engine = DiffusionEngine(models_dir=tmp_path, output_dir=tmp_path / "out")
        models = engine.discover_models()
        assert any(m["id"] == "v1-5-pruned" for m in models)

    def test_safetensors_inside_hf_dir_not_double_counted(self, tmp_path: Path) -> None:
        """Component .safetensors inside an HF model dir must not appear as standalone models."""
        hf_model = tmp_path / "my_sd_model"
        hf_model.mkdir()
        (hf_model / "model_index.json").write_text("{}", encoding="utf-8")
        unet_dir = hf_model / "unet"
        unet_dir.mkdir()
        (unet_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"")

        engine = DiffusionEngine(models_dir=tmp_path, output_dir=tmp_path / "out")
        models = engine.discover_models()
        ids = [m["id"] for m in models]
        assert "diffusion_pytorch_model" not in ids
        assert "my_sd_model" in ids

    def test_cache_cleared_after_models_root_removed(self, tmp_path: Path) -> None:
        """discover_models() must return empty list when models root disappears."""
        import shutil

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        hf = models_dir / "mymodel"
        hf.mkdir()
        (hf / "model_index.json").write_text("{}", encoding="utf-8")

        engine = DiffusionEngine(models_dir=models_dir, output_dir=tmp_path / "out")
        assert len(engine.discover_models()) == 1

        shutil.rmtree(str(models_dir))
        assert engine.discover_models() == []

    def test_new_models_visible_after_rediscovery(self, tmp_path: Path) -> None:
        """Newly added models must appear on the next discover_models() call."""
        models_root = tmp_path / "models"
        models_root.mkdir()
        engine = DiffusionEngine(models_dir=models_root, output_dir=tmp_path / "out")
        assert engine.discover_models() == []

        hf = models_root / "new_model"
        hf.mkdir(parents=True)
        (hf / "model_index.json").write_text("{}", encoding="utf-8")
        models = engine.discover_models()
        assert any(m["id"] == "new_model" for m in models)


class TestHasModels:
    """Test has_models() convenience method."""

    def test_has_models_false_when_empty(self, tmp_path: Path) -> None:
        """has_models() returns False when models dir is empty."""
        engine = DiffusionEngine(models_dir=tmp_path / "empty", output_dir=tmp_path / "out")
        assert engine.has_models() is False

    def test_has_models_true_when_model_present(self, tmp_path: Path) -> None:
        """has_models() returns True when at least one model dir exists."""
        hf = tmp_path / "my_model"
        hf.mkdir()
        (hf / "model_index.json").write_text("{}", encoding="utf-8")
        engine = DiffusionEngine(models_dir=tmp_path, output_dir=tmp_path / "out")
        assert engine.has_models() is True
