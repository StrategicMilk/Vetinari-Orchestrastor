"""Tests for vetinari.setup.init_wizard — first-run setup wizard."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vetinari.setup.init_wizard import (
    WizardResult,
    _detect_available_backends,
    _scan_for_models,
    _select_backend_order,
    _smoke_test,
    _write_config,
    run_wizard,
)
from vetinari.setup.nim_container import plan_nim_container_setup
from vetinari.setup.vllm_container import plan_vllm_container_setup
from vetinari.system.hardware_detect import GpuInfo, GpuVendor, HardwareProfile


class TestScanForModels:
    """Tests for model file scanning."""

    def test_scan_empty_dir(self, tmp_path: Path) -> None:
        """Returns empty list when no model files exist."""
        found = _scan_for_models(extra_dirs=[tmp_path])
        # May find models in default dirs, but tmp_path has none
        # Just verify it returns a list without error
        assert isinstance(found, list)

    def test_scan_finds_gguf_files(self, tmp_path: Path) -> None:
        """Discovers .gguf files in scanned directories."""
        model_file = tmp_path / "test-model.gguf"
        model_file.write_bytes(b"fake gguf data")

        found = _scan_for_models(extra_dirs=[tmp_path])
        found_names = {p.name for p in found}
        assert "test-model.gguf" in found_names

    def test_scan_finds_awq_files(self, tmp_path: Path) -> None:
        """Discovers .awq files in scanned directories."""
        model_file = tmp_path / "test-model.awq"
        model_file.write_bytes(b"fake awq data")

        found = _scan_for_models(extra_dirs=[tmp_path])
        found_names = {p.name for p in found}
        assert "test-model.awq" in found_names

    def test_scan_deduplicates(self, tmp_path: Path) -> None:
        """Does not return duplicate paths."""
        model_file = tmp_path / "test-model.gguf"
        model_file.write_bytes(b"fake gguf data")

        # Pass the same dir twice
        found = _scan_for_models(extra_dirs=[tmp_path, tmp_path])
        gguf_found = [p for p in found if p.name == "test-model.gguf"]
        assert len(gguf_found) == 1

    def test_scan_respects_depth_limit(self, tmp_path: Path) -> None:
        """Bounded scans do not recurse beyond the configured depth."""
        deep = tmp_path / "a" / "b"
        deep.mkdir(parents=True)
        model_file = deep / "deep-model.gguf"
        model_file.write_bytes(b"fake gguf data")

        found = _scan_for_models(extra_dirs=[tmp_path], max_depth=1)
        assert model_file.resolve() not in found


class TestWriteConfig:
    """Tests for configuration file generation."""

    def test_writes_yaml_config(self, tmp_path: Path) -> None:
        """Generates a valid YAML config with hardware-based defaults."""
        config_path = tmp_path / "config.yaml"
        profile = HardwareProfile(cpu_count=8, ram_gb=32.0)

        result = _write_config(profile, config_path=config_path)
        assert result == config_path
        assert config_path.exists()

        import yaml

        content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert content["inference"]["n_gpu_layers"] == 0  # No GPU
        assert content["inference"]["n_ctx"] == 4096
        assert content["inference"]["n_threads"] == 4  # 8 // 2

    def test_gpu_enables_offloading(self, tmp_path: Path) -> None:
        """GPU profile sets n_gpu_layers=-1 for full offloading."""
        config_path = tmp_path / "config.yaml"
        gpu = GpuInfo(name="RTX 4090", vendor=GpuVendor.NVIDIA, vram_gb=24.0, cuda_available=True)
        profile = HardwareProfile(cpu_count=16, ram_gb=64.0, gpu=gpu)

        _write_config(profile, config_path=config_path)

        import yaml

        content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert content["inference"]["n_gpu_layers"] == -1
        assert content["inference"]["flash_attn"] is True

    def test_includes_model_path(self, tmp_path: Path) -> None:
        """Config includes model path when a model is selected."""
        config_path = tmp_path / "config.yaml"
        model_path = tmp_path / "test.gguf"
        model_path.write_bytes(b"fake")
        profile = HardwareProfile(cpu_count=4, ram_gb=8.0)

        _write_config(profile, model_path=model_path, config_path=config_path)

        import yaml

        content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert "models" in content
        assert content["models"]["default_model"] == str(model_path)
        assert content["models"]["gguf_dir"] == str(model_path.parent)
        assert "inference_backend" in content
        assert content["inference_backend"]["primary"] in {"llama_cpp", "nim", "vllm"}

    def test_writes_backend_sections(self, tmp_path: Path, monkeypatch) -> None:
        """Wizard config includes normalized backend and model-root sections."""
        config_path = tmp_path / "config.yaml"
        profile = HardwareProfile(cpu_count=8, ram_gb=32.0)
        monkeypatch.delenv("VETINARI_NIM_ENDPOINT", raising=False)
        monkeypatch.delenv("VETINARI_PREFERRED_BACKEND", raising=False)
        monkeypatch.delenv("VETINARI_INFERENCE_BACKEND", raising=False)

        _write_config(profile, config_path=config_path, available_backends=["llama_cpp", "vllm"])

        import yaml

        content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert content["local_inference"]["models_dir"]
        assert content["models"]["gguf_dir"]
        assert content["models"]["native_dir"]
        assert content["inference_backend"]["selection_policy"] == "hardware_aware"
        assert content["inference_backend"]["primary"] == "llama_cpp"
        assert content["inference_backend"]["fallback"] == "llama_cpp"
        assert content["inference_backend"]["fallback_order"] == ["llama_cpp"]
        assert "gguf_only_models" in content["inference_backend"]["llama_cpp_use_cases"]
        assert content["inference_backend"]["vllm"]["enabled"] is True
        assert content["inference_backend"]["vllm"]["semantic_cache_enabled"] is True
        assert content["inference_backend"]["vllm"]["prefix_caching_enabled"] is True
        assert content["inference_backend"]["vllm"]["container_setup"] == {}
        assert content["inference_backend"]["nim"]["enabled"] is False
        assert content["inference_backend"]["nim"]["kv_cache_reuse_enabled"] is False

    def test_nvidia_setup_prefers_nim_before_vllm(self, tmp_path: Path, monkeypatch) -> None:
        """NVIDIA/CUDA setup config prefers NIM, then vLLM, then llama.cpp."""
        config_path = tmp_path / "config.yaml"
        gpu = GpuInfo(name="RTX 4090", vendor=GpuVendor.NVIDIA, vram_gb=24.0, cuda_available=True)
        profile = HardwareProfile(cpu_count=16, ram_gb=64.0, gpu=gpu)
        monkeypatch.delenv("VETINARI_PREFERRED_BACKEND", raising=False)
        monkeypatch.delenv("VETINARI_INFERENCE_BACKEND", raising=False)
        nim_setup = plan_nim_container_setup(
            profile,
            env={"VETINARI_NIM_IMAGE": "nvcr.io/nim/example/model:latest", "NGC_API_KEY": "secret"},
            endpoint_ready=False,
            docker_available=True,
            nvidia_runtime_available=True,
        )

        _write_config(
            profile,
            config_path=config_path,
            available_backends=["llama_cpp", "vllm", "nim"],
            nim_setup=nim_setup,
        )

        import yaml

        content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert content["inference_backend"]["primary"] == "nim"
        assert content["inference_backend"]["fallback"] == "vllm"
        assert content["inference_backend"]["fallback_order"] == ["nim", "vllm", "llama_cpp"]
        assert content["inference_backend"]["nim"]["kv_cache_reuse_enabled"] is True
        setup = content["inference_backend"]["nim"]["container_setup"]
        assert setup["status"] == "guided_ready"
        assert setup["api_key_present"] is True
        assert "secret" not in setup["start_command"]

    def test_gpu_setup_records_vllm_container_plan(self, tmp_path: Path, monkeypatch) -> None:
        """Wizard config records a secret-free vLLM container setup plan."""
        config_path = tmp_path / "config.yaml"
        gpu = GpuInfo(name="RTX 4090", vendor=GpuVendor.NVIDIA, vram_gb=24.0, cuda_available=True)
        profile = HardwareProfile(cpu_count=16, ram_gb=64.0, gpu=gpu)
        monkeypatch.delenv("VETINARI_PREFERRED_BACKEND", raising=False)
        monkeypatch.delenv("VETINARI_INFERENCE_BACKEND", raising=False)
        vllm_setup = plan_vllm_container_setup(
            profile,
            env={"VETINARI_VLLM_MODEL": "Qwen/Qwen3-0.6B", "HF_TOKEN": "hf-secret"},
            endpoint_ready=False,
            docker_available=True,
            nvidia_runtime_available=True,
        )

        _write_config(
            profile,
            config_path=config_path,
            available_backends=["llama_cpp", "vllm"],
            vllm_setup=vllm_setup,
        )

        import yaml

        content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        setup = content["inference_backend"]["vllm"]["container_setup"]
        assert setup["status"] == "guided_ready"
        assert setup["hf_token_present"] is True
        assert setup["image"] == "vllm/vllm-openai:latest"
        assert "hf-secret" not in setup["start_command"]

    def test_explicit_backend_preference_is_respected(self, monkeypatch) -> None:
        """Explicit user backend preference should override hardware policy."""
        gpu = GpuInfo(name="RTX 4090", vendor=GpuVendor.NVIDIA, vram_gb=24.0, cuda_available=True)
        profile = HardwareProfile(cpu_count=16, ram_gb=64.0, gpu=gpu)
        monkeypatch.setenv("VETINARI_PREFERRED_BACKEND", "llama_cpp")

        order = _select_backend_order(profile, ["llama_cpp", "vllm", "nim"])

        assert order == ["llama_cpp", "nim", "vllm"]

    def test_detected_vllm_requires_reachable_endpoint(self, monkeypatch) -> None:
        """An installed vLLM package alone should not make vLLM a live backend."""
        gpu = GpuInfo(name="RTX 4090", vendor=GpuVendor.NVIDIA, vram_gb=24.0, cuda_available=True)
        profile = HardwareProfile(cpu_count=16, ram_gb=64.0, gpu=gpu)
        fake_httpx = MagicMock()
        fake_httpx.get.side_effect = RuntimeError("offline")
        monkeypatch.delenv("VETINARI_VLLM_ENDPOINT", raising=False)
        monkeypatch.delenv("VETINARI_NIM_ENDPOINT", raising=False)

        with (
            patch.dict("sys.modules", {"vllm": MagicMock(), "httpx": fake_httpx}),
            patch("vetinari.setup.init_wizard.is_vllm_endpoint_ready", return_value=False),
        ):
            backends = _detect_available_backends(profile)

        assert backends == ["llama_cpp"]

    def test_detected_vllm_endpoint_is_available(self, monkeypatch) -> None:
        """A reachable vLLM endpoint should be marked available."""
        gpu = GpuInfo(name="RTX 4090", vendor=GpuVendor.NVIDIA, vram_gb=24.0, cuda_available=True)
        profile = HardwareProfile(cpu_count=16, ram_gb=64.0, gpu=gpu)
        fake_httpx = MagicMock()
        fake_httpx.get.side_effect = RuntimeError("offline")
        monkeypatch.delenv("VETINARI_NIM_ENDPOINT", raising=False)

        with (
            patch.dict("sys.modules", {"httpx": fake_httpx}),
            patch("vetinari.setup.init_wizard.is_vllm_endpoint_ready", return_value=True),
        ):
            backends = _detect_available_backends(profile)

        assert "vllm" in backends


class TestSmokeTest:
    """Tests for the vetinari import smoke test."""

    def test_smoke_test_succeeds(self) -> None:
        """Smoke test passes when vetinari is importable."""
        assert _smoke_test() is True


class TestRunWizard:
    """Tests for the full wizard flow."""

    @patch("vetinari.setup.init_wizard._detect_available_backends", return_value=["llama_cpp"])
    @patch("vetinari.setup.init_wizard.plan_vllm_container_setup")
    @patch("vetinari.setup.init_wizard.plan_nim_container_setup")
    @patch("vetinari.setup.init_wizard.detect_hardware")
    @patch("vetinari.setup.init_wizard._scan_for_models", return_value=[])
    def test_non_interactive_completes(
        self,
        _mock_scan: MagicMock,
        mock_hw: MagicMock,
        mock_nim_plan: MagicMock,
        mock_vllm_plan: MagicMock,
        _mock_backends: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Non-interactive wizard completes without prompts."""
        mock_hw.return_value = HardwareProfile(cpu_count=4, ram_gb=16.0)
        plan = MagicMock()
        plan.can_auto_start = False
        plan.status = "unsupported_hardware"
        plan.hardware_eligible = False
        plan.to_config.return_value = {}
        mock_nim_plan.return_value = plan
        mock_vllm_plan.return_value = plan
        config_path = tmp_path / "config.yaml"

        result = run_wizard(
            skip_download=True,
            non_interactive=True,
            config_path=config_path,
        )

        assert isinstance(result, WizardResult)
        assert result.hardware is not None
        assert result.hardware.cpu_count == 4
        assert result.config_path == config_path
        assert config_path.exists()

    @patch("vetinari.setup.init_wizard._detect_available_backends", return_value=["llama_cpp"])
    @patch("vetinari.setup.init_wizard.plan_vllm_container_setup")
    @patch("vetinari.setup.init_wizard.plan_nim_container_setup")
    @patch("vetinari.setup.init_wizard.detect_hardware")
    @patch("vetinari.setup.init_wizard._scan_for_models")
    def test_uses_existing_models(
        self,
        mock_scan: MagicMock,
        mock_hw: MagicMock,
        mock_nim_plan: MagicMock,
        mock_vllm_plan: MagicMock,
        _mock_backends: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Wizard uses existing models when found on disk."""
        model_path = tmp_path / "existing.gguf"
        model_path.write_bytes(b"fake gguf")
        mock_scan.return_value = [model_path]
        mock_hw.return_value = HardwareProfile(cpu_count=4, ram_gb=16.0)
        plan = MagicMock()
        plan.can_auto_start = False
        plan.status = "unsupported_hardware"
        plan.hardware_eligible = False
        plan.to_config.return_value = {}
        mock_nim_plan.return_value = plan
        mock_vllm_plan.return_value = plan

        result = run_wizard(
            skip_download=False,
            non_interactive=True,
            config_path=tmp_path / "config.yaml",
        )

        assert result.models_found == [model_path]
        # Should not attempt download since models already exist
        assert result.model_downloaded is None
