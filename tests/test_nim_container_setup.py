"""Tests for NIM container setup planning."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vetinari.setup.nim_container import plan_nim_container_setup, start_nim_container
from vetinari.system.hardware_detect import GpuInfo, GpuVendor, HardwareProfile


def _nvidia_profile() -> HardwareProfile:
    gpu = GpuInfo(name="RTX 4090", vendor=GpuVendor.NVIDIA, vram_gb=24.0, cuda_available=True)
    return HardwareProfile(cpu_count=16, ram_gb=64.0, gpu=gpu)


def test_plan_auto_start_when_nim_inputs_are_present() -> None:
    """Auto mode is startable only with NVIDIA/CUDA, Docker, runtime, image, and API key."""
    env = {
        "VETINARI_NIM_SETUP_MODE": "auto",
        "VETINARI_NIM_IMAGE": "nvcr.io/nim/example/model:latest",
        "NGC_API_KEY": "super-secret",
        "VETINARI_NIM_CACHE_DIR": "/tmp/nim-cache",
    }

    plan = plan_nim_container_setup(
        _nvidia_profile(),
        env=env,
        endpoint_ready=False,
        docker_available=True,
        nvidia_runtime_available=True,
    )

    assert plan.status == "auto_ready"
    assert plan.can_auto_start is True
    assert plan.missing == ()
    assert "--gpus all" in plan.start_command
    assert "NIM_ENABLE_KV_CACHE_REUSE=1" in plan.start_command
    assert "super-secret" not in plan.start_command


def test_plan_reports_missing_prerequisites_without_endpoint() -> None:
    """NIM setup should explain why it cannot guide or auto-start."""
    plan = plan_nim_container_setup(
        _nvidia_profile(),
        env={},
        endpoint_ready=False,
        docker_available=False,
        nvidia_runtime_available=False,
    )

    assert plan.status == "missing_prerequisites"
    assert "docker" in plan.missing
    assert "VETINARI_NIM_IMAGE" in plan.missing
    assert "NGC_API_KEY" in plan.missing


def test_plan_rejects_local_container_setup_on_cpu_only_hardware() -> None:
    """CPU-only machines should not be guided into a local NIM container setup."""
    plan = plan_nim_container_setup(
        HardwareProfile(cpu_count=8, ram_gb=32.0),
        env={"VETINARI_NIM_SETUP_MODE": "auto", "VETINARI_NIM_IMAGE": "image", "NGC_API_KEY": "secret"},
        endpoint_ready=False,
        docker_available=True,
        nvidia_runtime_available=True,
    )

    assert plan.status == "unsupported_hardware"
    assert plan.can_auto_start is False
    assert "nvidia_cuda_gpu" in plan.missing


def test_start_container_logs_into_ngc_without_exposing_secret() -> None:
    """Auto-start should use stdin for docker login and then run the planned image."""
    env = {
        "VETINARI_NIM_SETUP_MODE": "auto",
        "VETINARI_NIM_IMAGE": "nvcr.io/nim/example/model:latest",
        "NGC_API_KEY": "super-secret",
    }
    plan = plan_nim_container_setup(
        _nvidia_profile(),
        env=env,
        endpoint_ready=False,
        docker_available=True,
        nvidia_runtime_available=True,
    )

    with (
        patch("vetinari.setup.nim_container.shutil.which", return_value="docker"),
        patch("vetinari.setup.nim_container.subprocess.run", return_value=MagicMock(returncode=0)) as run,
    ):
        started = start_nim_container(plan, env=env)

    assert started.started is True
    login_call, run_call = run.call_args_list
    assert login_call.kwargs["input"] == "super-secret"
    assert "super-secret" not in run_call.args[0]
    assert run_call.args[0][:3] == ["docker", "run", "-d"]
