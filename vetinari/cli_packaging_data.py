"""Packaging CLI data for init, hardware detection, and model recommendations.

This module provides the shared constants and helpers used by the packaging
CLI submodules. The ``vetinari init`` command itself delegates to
``vetinari.setup.init_wizard`` so config generation has a single authoritative
implementation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from vetinari.constants import OPERATOR_MODELS_CACHE_DIR

logger = logging.getLogger(__name__)

# CLI-local path constants honor the canonical operator model cache root.
DEFAULT_USER_MODELS_DIR: Path = Path(OPERATOR_MODELS_CACHE_DIR)

_MODEL_TIERS: list[dict[str, Any]] = [
    {
        "min_vram_gb": 0,
        "max_vram_gb": 4,
        "label": "< 4 GB VRAM (CPU / very low VRAM)",
        "models": [
            {
                "name": "TinyLlama 1.1B Q4_K_M",
                "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            },
            {
                "name": "Phi-2 Q4_K_M",
                "repo": "TheBloke/phi-2-GGUF",
                "filename": "phi-2.Q4_K_M.gguf",
                "url": "https://huggingface.co/TheBloke/phi-2-GGUF",
            },
        ],
    },
    {
        "min_vram_gb": 4,
        "max_vram_gb": 8,
        "label": "4-8 GB VRAM",
        "models": [
            {
                "name": "Mistral 7B Q4_K_M",
                "repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            },
            {
                "name": "Llama 3.1 8B Q4_K_M",
                "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            },
        ],
    },
    {
        "min_vram_gb": 8,
        "max_vram_gb": 16,
        "label": "8-16 GB VRAM",
        "models": [
            {
                "name": "Llama 3.1 8B Q6_K",
                "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                "filename": "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf",
                "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            },
            {
                "name": "Mistral 7B Q6_K",
                "repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                "filename": "mistral-7b-instruct-v0.2.Q6_K.gguf",
                "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            },
        ],
    },
    {
        "min_vram_gb": 16,
        "max_vram_gb": 999,
        "label": "16+ GB VRAM",
        "models": [
            {
                "name": "Llama 3.1 8B F16",
                "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                "filename": "Meta-Llama-3.1-8B-Instruct-f16.gguf",
                "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            },
            {
                "name": "Codestral 22B Q4_K_M",
                "repo": "bartowski/Codestral-22B-v0.1-GGUF",
                "filename": "Codestral-22B-v0.1-Q4_K_M.gguf",
                "url": "https://huggingface.co/bartowski/Codestral-22B-v0.1-GGUF",
            },
        ],
    },
]

try:
    from rich.console import Console as _Console

    _RICH_AVAILABLE = True
    _console = _Console()
except ImportError:
    _RICH_AVAILABLE = False
    _console = None  # type: ignore[assignment]

_CHECK_PASS = "PASS"
_CHECK_FAIL = "FAIL"
_CHECK_WARN = "WARN"
_CHECK_INFO = "INFO"


def _print_header(title: str) -> None:
    """Print a section header with consistent formatting."""
    if _RICH_AVAILABLE and _console is not None:
        _console.rule(f"[bold cyan]{title}[/bold cyan]")
    else:
        width = 60
        print(f"\n{'=' * width}")
        print(f"  {title}")
        print(f"{'=' * width}")


def _print_check(label: str, status: str, detail: str = "") -> None:
    """Print a single diagnostic check result with color when available."""
    colour_map = {
        _CHECK_PASS: "green",
        _CHECK_FAIL: "red",
        _CHECK_WARN: "yellow",
        _CHECK_INFO: "blue",
    }
    symbol_map = {
        _CHECK_PASS: "[OK]",
        _CHECK_FAIL: "[FAIL]",
        _CHECK_WARN: "[WARN]",
        _CHECK_INFO: "[INFO]",
    }
    symbol = symbol_map.get(status, "[?]")
    suffix = f"  {detail}" if detail else ""
    if _RICH_AVAILABLE and _console is not None:
        colour = colour_map.get(status, "white")
        _console.print(f"  [{colour}]{symbol:<8}[/{colour}] {label}{suffix}")
    else:
        print(f"  {symbol:<8} {label}{suffix}")


def _detect_hardware() -> dict[str, Any]:
    """Detect CPU, RAM, and GPU/VRAM hardware available on this machine."""
    hw: dict[str, Any] = {
        "cpu_count": os.cpu_count() or 1,
        "ram_gb": 0.0,
        "gpu_name": None,
        "vram_gb": 0.0,
        "cuda_available": False,
    }

    try:
        import psutil

        hw["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        logger.debug("psutil not available; RAM detection skipped")

    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        hw["gpu_name"] = pynvml.nvmlDeviceGetName(handle)
        if isinstance(hw["gpu_name"], bytes):
            hw["gpu_name"] = hw["gpu_name"].decode("utf-8", errors="replace")
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        hw["vram_gb"] = round(mem_info.total / (1024**3), 1)
        hw["cuda_available"] = True
        pynvml.nvmlShutdown()
    except Exception:
        logger.warning("pynvml GPU detection failed", exc_info=True)

    return hw


def _get_recommended_models(vram_gb: float) -> list[dict[str, Any]]:
    """Return backend-aware model recommendations for the detected VRAM."""
    try:
        from vetinari.setup.model_recommender import ModelRecommender
        from vetinari.system.hardware_detect import GpuInfo, GpuVendor, HardwareProfile

        has_gpu = vram_gb > 0
        hardware = HardwareProfile(
            cpu_count=os.cpu_count() or 1,
            ram_gb=max(32.0, vram_gb * 2) if has_gpu else 8.0,
            gpu=GpuInfo(
                name="Detected GPU" if has_gpu else "",
                vendor=GpuVendor.NVIDIA if has_gpu else GpuVendor.NONE,
                vram_gb=vram_gb,
                cuda_available=has_gpu,
            ),
        )
        recommendations = ModelRecommender().recommend_models_multi_format(hardware)
        return [
            {
                "name": rec.name,
                "repo": rec.repo_id,
                "filename": rec.filename,
                "url": f"https://huggingface.co/{rec.repo_id}",
                "backend": rec.backend,
                "format": rec.model_format,
                "quantization": rec.quantization,
                "size_gb": rec.size_gb,
                "parameter_count": rec.parameter_count,
            }
            for rec in recommendations
        ]
    except Exception:
        logger.warning("backend-aware model recommendations failed; falling back to legacy GGUF tiers", exc_info=True)

    for tier in _MODEL_TIERS:
        if tier["min_vram_gb"] <= vram_gb < tier["max_vram_gb"]:
            return list(tier["models"])  # type: ignore[return-value]
    return list(_MODEL_TIERS[0]["models"])  # type: ignore[return-value]


def cmd_init(args: Any) -> int:
    """Run the authoritative first-run setup wizard.

    Returns:
        Exit code from the setup wizard result.
    """
    from vetinari.setup.init_wizard import run_wizard

    result = run_wizard(skip_download=bool(getattr(args, "skip_download", False)))
    return 0 if result.success else 1
