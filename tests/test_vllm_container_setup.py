"""Tests for vLLM container setup planning."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vetinari.setup.vllm_container import plan_vllm_container_setup, start_vllm_container
from vetinari.system.hardware_detect import GpuInfo, GpuVendor, HardwareProfile


def _nvidia_profile() -> HardwareProfile:
    gpu = GpuInfo(name="RTX 4090", vendor=GpuVendor.NVIDIA, vram_gb=24.0, cuda_available=True)
    return HardwareProfile(cpu_count=16, ram_gb=64.0, gpu=gpu)


def _amd_profile() -> HardwareProfile:
    gpu = GpuInfo(name="RX 7900 XTX", vendor=GpuVendor.AMD, vram_gb=24.0)
    return HardwareProfile(cpu_count=16, ram_gb=64.0, gpu=gpu, os_name="Linux")


def test_plan_auto_start_for_cuda_vllm_when_inputs_are_present() -> None:
    """Auto mode is startable with CUDA hardware, Docker, runtime, and model."""
    env = {
        "VETINARI_VLLM_SETUP_MODE": "auto",
        "VETINARI_VLLM_MODEL": "Qwen/Qwen3-0.6B",
        "VETINARI_VLLM_CACHE_DIR": "/tmp/hf-cache",
        "VETINARI_VLLM_CACHE_SALT": "secret-salt",
        "HF_TOKEN": "hf-secret",
    }

    plan = plan_vllm_container_setup(
        _nvidia_profile(),
        env=env,
        endpoint_ready=False,
        docker_available=True,
        nvidia_runtime_available=True,
    )

    assert plan.status == "auto_ready"
    assert plan.device_backend == "cuda"
    assert plan.can_auto_start is True
    assert "--gpus all" in plan.start_command
    assert "--model Qwen/Qwen3-0.6B" in plan.start_command
    assert "--enable-prefix-caching" in plan.start_command
    assert "hf-secret" not in plan.start_command
    assert "secret-salt" not in plan.start_command


def test_plan_reports_missing_model_without_endpoint() -> None:
    """vLLM setup should explain why it cannot guide or auto-start."""
    plan = plan_vllm_container_setup(
        _nvidia_profile(),
        env={},
        endpoint_ready=False,
        docker_available=True,
        nvidia_runtime_available=True,
    )

    assert plan.status == "missing_prerequisites"
    assert "VETINARI_VLLM_MODEL" in plan.missing


def test_plan_uses_rocm_image_for_amd_gpu() -> None:
    """AMD GPUs should get the ROCm vLLM container path."""
    plan = plan_vllm_container_setup(
        _amd_profile(),
        env={"VETINARI_VLLM_MODEL": "Qwen/Qwen3-0.6B"},
        endpoint_ready=False,
        docker_available=True,
        nvidia_runtime_available=False,
    )

    assert plan.status == "guided_ready"
    assert plan.device_backend == "rocm"
    assert plan.image == "vllm/vllm-openai-rocm:latest"
    assert "--device /dev/kfd" in plan.start_command


def test_plan_rejects_local_container_setup_on_cpu_only_hardware() -> None:
    """CPU-only machines should not be guided into a vLLM server container."""
    plan = plan_vllm_container_setup(
        HardwareProfile(cpu_count=8, ram_gb=32.0),
        env={"VETINARI_VLLM_SETUP_MODE": "auto", "VETINARI_VLLM_MODEL": "Qwen/Qwen3-0.6B"},
        endpoint_ready=False,
        docker_available=True,
        nvidia_runtime_available=True,
    )

    assert plan.status == "unsupported_hardware"
    assert plan.can_auto_start is False
    assert "gpu_server_hardware" in plan.missing


def test_start_container_passes_token_by_env_name_only() -> None:
    """Auto-start should pass HF_TOKEN by name and never include the secret in argv."""
    env = {
        "VETINARI_VLLM_SETUP_MODE": "auto",
        "VETINARI_VLLM_MODEL": "Qwen/Qwen3-0.6B",
        "HF_TOKEN": "hf-secret",
    }
    plan = plan_vllm_container_setup(
        _nvidia_profile(),
        env=env,
        endpoint_ready=False,
        docker_available=True,
        nvidia_runtime_available=True,
    )

    with (
        patch("vetinari.setup.vllm_container.shutil.which", return_value="docker"),
        patch("vetinari.setup.vllm_container.subprocess.run", return_value=MagicMock(returncode=0)) as run,
    ):
        started = start_vllm_container(plan, env=env)

    assert started.started is True
    docker_args = run.call_args.args[0]
    assert docker_args[:3] == ["docker", "run", "-d"]
    assert "HF_TOKEN" in docker_args
    assert "hf-secret" not in docker_args
