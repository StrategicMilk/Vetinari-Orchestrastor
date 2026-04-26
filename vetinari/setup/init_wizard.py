"""First-run setup wizard — interactive onboarding for new Vetinari users.

This is the final step of the setup pipeline: Hardware Detection →
Model Recommendation → **Init Wizard** → Configuration.

Orchestrates the full first-run experience:
1. Detect hardware (CPU/RAM/GPU)
2. Scan for existing GGUF models on disk
3. Recommend models based on detected VRAM
4. Optionally download a recommended model
5. Write config with smart defaults
6. Run smoke test to verify setup

Entry point: ``run_wizard(skip_download=False, non_interactive=False)``
CLI binding: ``vetinari init`` (see ``cli_packaging.py``)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any

from rich.console import Console

from vetinari.constants import (
    DEFAULT_MODELS_DIR,  # noqa: VET306 - constants resolve to operator/user cache roots.
    DEFAULT_NATIVE_MODELS_DIR,  # noqa: VET306 - constants resolve to operator/user cache roots.
    OPERATOR_MODELS_CACHE_DIR,  # noqa: VET306 - constants resolve to operator/user cache roots.
    get_user_dir,
)
from vetinari.setup.model_recommender import ModelRecommender, SetupModelRecommendation
from vetinari.setup.nim_container import (
    NIMContainerPlan,
    plan_nim_container_setup,
    start_nim_container,
)
from vetinari.setup.vllm_container import (
    VLLMContainerPlan,
    plan_vllm_container_setup,
    start_vllm_container,
)
from vetinari.setup.vllm_container import is_openai_endpoint_ready as is_vllm_endpoint_ready
from vetinari.system.hardware_detect import GpuVendor, HardwareProfile, detect_hardware

logger = logging.getLogger(__name__)
console = Console()

# ── Constants ─────────────────────────────────────────────────────────────────

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DEFAULT_GGUF_MODELS_DIR: Path = Path(DEFAULT_MODELS_DIR)  # noqa: VET306 — user-scoped via get_user_dir()
DEFAULT_NATIVE_MODELS_PATH: Path = Path(DEFAULT_NATIVE_MODELS_DIR)  # noqa: VET306 — user-scoped via get_user_dir()
DOWNLOAD_GGUF_MODELS_DIR: Path = Path(OPERATOR_MODELS_CACHE_DIR)
DOWNLOAD_NATIVE_MODELS_DIR: Path = Path(
    os.environ.get("VETINARI_NATIVE_MODELS_DIR", str(DOWNLOAD_GGUF_MODELS_DIR / "native"))
)
DEFAULT_CONFIG_PATH: Path = get_user_dir() / "config.yaml"

# Common locations where users may have existing GGUF or native model assets
_COMMON_MODEL_DIRS = [
    _PROJECT_ROOT / "models",
    _PROJECT_ROOT / "models" / "native",
    Path.home() / ".cache" / "huggingface",
    Path.home() / ".ollama" / "models",
    Path.home() / "models",
    Path.home() / ".local" / "share" / "vetinari" / "models",
]

WIZARD_STEPS = 6  # Total steps shown to the user
_MODEL_SCAN_SUFFIXES = {".gguf", ".awq", ".safetensors", ".gptq"}
_MODEL_SCAN_MAX_FILES = 5000
_MODEL_SCAN_MAX_DEPTH = 8
_MODEL_SCAN_TIMEOUT_SECONDS = 5.0
_FALSE_ENV_VALUES = {"", "0", "false", "no", "off"}
_LLAMA_CPP_POLICY_USE_CASES = [
    "explicit_user_preference",
    "weak_or_no_server_setup",
    "gguf_only_models",
    "cpu_ram_vram_offload",
    "oversized_local_models",
    "recovery_fallback",
]


@dataclass
class WizardResult:
    """Outcome of the setup wizard run.

    Attributes:
        success: Whether the wizard completed successfully.
        hardware: Detected hardware profile.
        models_found: List of discovered model asset paths found on disk.
        model_downloaded: Path to downloaded model, if any.
        config_path: Path to the generated config file.
        errors: List of non-fatal errors encountered.
    """

    success: bool = False
    hardware: HardwareProfile | None = None
    models_found: list[Path] = field(default_factory=list)
    model_downloaded: Path | None = None
    config_path: Path | None = None
    vllm_setup: dict[str, Any] = field(default_factory=dict)
    nim_setup: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return "WizardResult(...)"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _scan_for_models(
    extra_dirs: list[Path] | None = None,
    *,
    max_files: int = _MODEL_SCAN_MAX_FILES,
    max_depth: int = _MODEL_SCAN_MAX_DEPTH,
    timeout_seconds: float = _MODEL_SCAN_TIMEOUT_SECONDS,
    cancel_event: Event | None = None,
) -> list[Path]:
    """Scan common directories for existing GGUF and AWQ model files.

    Args:
        extra_dirs: Additional directories to scan beyond the defaults.
        max_files: Maximum number of directory entries to inspect.
        max_depth: Maximum recursion depth under each scan root.
        timeout_seconds: Wall-clock scan budget in seconds.
        cancel_event: Optional event that stops scanning when set.

    Returns:
        Sorted list of discovered model file paths.
    """
    dirs_to_scan = [DEFAULT_GGUF_MODELS_DIR, DEFAULT_NATIVE_MODELS_PATH, *_COMMON_MODEL_DIRS]
    if extra_dirs:
        dirs_to_scan.extend(extra_dirs)

    # Also check VETINARI_MODELS_DIR env var
    env_dir = os.environ.get("VETINARI_MODELS_DIR", "")
    if env_dir:
        dirs_to_scan.append(Path(env_dir))

    found: list[Path] = []
    seen: set[Path] = set()
    scanned = 0
    started = time.monotonic()

    for scan_dir in dirs_to_scan:
        if cancel_event is not None and cancel_event.is_set():
            break
        if time.monotonic() - started > timeout_seconds:
            logger.warning("Model scan timed out after %.1fs", timeout_seconds)
            break
        if not scan_dir.is_dir():
            continue
        try:
            root = scan_dir.resolve()
            for dirpath, dirnames, filenames in os.walk(root):
                if cancel_event is not None and cancel_event.is_set():
                    break
                if time.monotonic() - started > timeout_seconds:
                    logger.warning("Model scan timed out after %.1fs", timeout_seconds)
                    break
                current = Path(dirpath)
                try:
                    depth = len(current.relative_to(root).parts)
                except ValueError:
                    logger.warning("Skipped model scan path outside root: %s", current)
                    continue
                if depth >= max_depth:
                    dirnames[:] = []
                else:
                    dirnames[:] = [name for name in dirnames if not name.startswith(".")]
                for filename in filenames:
                    scanned += 1
                    if scanned > max_files:
                        logger.warning("Model scan stopped after %d files", max_files)
                        found.sort(key=lambda p: p.name)
                        return found
                    model_path = current / filename
                    if model_path.suffix.lower() not in _MODEL_SCAN_SUFFIXES:
                        continue
                    resolved = model_path.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        found.append(resolved)
        except PermissionError:
            logger.warning("Permission denied scanning %s — skipping directory in model search", scan_dir)

    found.sort(key=lambda p: p.name)
    return found


def _coerce_env_bool(value: str | None, *, default: bool = False) -> bool:
    """Coerce setup environment flags into booleans."""
    if value is None:
        return default
    return value.strip().lower() not in _FALSE_ENV_VALUES


def _normalize_backend_name(value: str | None) -> str:
    """Normalize configured backend names to setup/runtime backend ids."""
    normalized = (value or "").strip().lower().replace("-", "_")
    if normalized in {"local", "llama", "llamacpp", "llama_cpp"}:
        return "llama_cpp"
    if normalized in {"nims", "nvidia_nim", "nvidia_nims"}:
        return "nim"
    return normalized


def _preferred_backend_from_env() -> str | None:
    """Return explicit user backend preference, if configured."""
    for env_name in ("VETINARI_PREFERRED_BACKEND", "VETINARI_INFERENCE_BACKEND"):
        preferred = _normalize_backend_name(os.environ.get(env_name))
        if preferred:
            return preferred
    return None


def _supports_vllm_setup(hardware: HardwareProfile | None) -> bool:
    """Return True when vLLM is a reasonable first-run backend candidate."""
    if hardware is None:
        return True
    if not hardware.has_gpu:
        return False
    return hardware.gpu_vendor in {GpuVendor.NVIDIA, GpuVendor.AMD, GpuVendor.INTEL}


def _select_backend_order(
    hardware: HardwareProfile,
    available_backends: list[str] | None = None,
    *,
    preferred_backend: str | None = None,
) -> list[str]:
    """Return hardware-aware backend order for generated setup config."""
    detected = {_normalize_backend_name(backend) for backend in (available_backends or ["llama_cpp"])}
    detected.discard("")
    detected.add("llama_cpp")

    order: list[str] = []

    def add(name: str) -> None:
        """Append a backend to the setup order once."""
        if name in detected and name not in order:
            order.append(name)

    preferred = _normalize_backend_name(preferred_backend) or _preferred_backend_from_env()
    if preferred:
        add(preferred)

    has_nvidia_cuda = hardware.gpu_vendor == GpuVendor.NVIDIA and hardware.cuda_available
    remote_nim_configured = bool(os.environ.get("VETINARI_NIM_ENDPOINT"))

    if has_nvidia_cuda or remote_nim_configured:
        add("nim")
    if _supports_vllm_setup(hardware):
        add("vllm")
    add("llama_cpp")
    return order


def _detect_available_backends(hardware: HardwareProfile | None = None) -> list[str]:
    """Detect which inference backends are available on this system.

    Checks reachability of vLLM and NIM endpoints.
    llama-cpp-python is always included as the default backend.

    Returns:
        List of available backend names (e.g. ``["llama_cpp", "vllm"]``).
    """
    backends = ["llama_cpp"]  # Always available for GGUF/offload/recovery fallback.

    # Check for a running vLLM server. Importability alone is not a live backend.
    vllm_endpoint = os.environ.get("VETINARI_VLLM_ENDPOINT", "http://localhost:8000")
    if hardware is not None or os.environ.get("VETINARI_VLLM_ENDPOINT"):
        if is_vllm_endpoint_ready(vllm_endpoint):
            backends.append("vllm")
            logger.info("vLLM detected at %s", vllm_endpoint)
        else:
            logger.warning("vLLM endpoint %s not reachable - vLLM backend will not be available", vllm_endpoint)

    # Check for NVIDIA NIM (look for NIM environment or running container)
    nim_endpoint = os.environ.get("VETINARI_NIM_ENDPOINT", "http://localhost:8001")
    if hardware is not None or os.environ.get("VETINARI_NIM_ENDPOINT"):
        try:
            import httpx

            resp = httpx.get(f"{nim_endpoint}/v1/models", timeout=5)
            if resp.status_code == 200:
                backends.append("nim")
                logger.info("NVIDIA NIM detected at %s", nim_endpoint)
        except Exception:
            logger.warning("NIM endpoint %s not reachable - NIM backend will not be available", nim_endpoint)

    return backends


def _write_config(
    hardware: HardwareProfile,
    model_path: Path | None = None,
    config_path: Path | None = None,
    available_backends: list[str] | None = None,
    vllm_setup: VLLMContainerPlan | None = None,
    nim_setup: NIMContainerPlan | None = None,
) -> Path:
    """Write a vetinari config YAML with detected hardware settings.

    Args:
        hardware: Detected hardware profile for smart defaults.
        model_path: Path to the selected/downloaded model file.
        config_path: Where to write the config (default: ~/.vetinari/config.yaml).
        available_backends: List of detected backends (e.g. ``["llama_cpp", "vllm"]``).
        vllm_setup: Optional vLLM container setup plan to record in config.
        nim_setup: Optional NIM container setup plan to record in config.

    Returns:
        Path to the written config file.
    """
    import yaml  # type: ignore[import-untyped]

    config_path = config_path or (get_user_dir() / "config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    backends = available_backends or ["llama_cpp"]

    # Smart defaults based on hardware
    n_gpu_layers = -1 if hardware.has_gpu else 0  # -1 = offload all layers to GPU
    n_ctx = 4096  # Safe default context window
    flash_attn = hardware.cuda_available  # Flash attention only with CUDA
    n_threads = max(1, hardware.cpu_count // 2)  # Half of logical cores
    n_batch = 512  # Default batch size

    backend_order = _select_backend_order(hardware, backends)
    primary_backend = backend_order[0]
    fallback_backend = backend_order[1] if len(backend_order) > 1 else "llama_cpp"

    config: dict[str, Any] = {
        "inference": {
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "n_batch": n_batch,
            "flash_attn": flash_attn,
            "n_threads": n_threads,
        },
        "local_inference": {
            "models_dir": str(model_path.parent if model_path else DEFAULT_GGUF_MODELS_DIR),
            "gpu_layers": n_gpu_layers,
            "context_length": 8192,
        },
        "models": {
            "gguf_dir": str(model_path.parent if model_path else DEFAULT_GGUF_MODELS_DIR),
            "native_dir": os.environ.get("VETINARI_NATIVE_MODELS_DIR", str(DEFAULT_NATIVE_MODELS_PATH)),
        },
        "inference_backend": {
            "selection_policy": "hardware_aware",
            "primary": primary_backend,
            "fallback": fallback_backend,
            "fallback_order": backend_order,
            "llama_cpp_use_cases": list(_LLAMA_CPP_POLICY_USE_CASES),
            "native_models_dir": os.environ.get("VETINARI_NATIVE_MODELS_DIR", str(DEFAULT_NATIVE_MODELS_PATH)),
        },
        "hardware": hardware.to_dict(),
    }

    if model_path and model_path.exists():
        config["models"] = {
            "default_model": str(model_path),
            "gguf_dir": str(model_path.parent),
            "native_dir": os.environ.get("VETINARI_NATIVE_MODELS_DIR", str(DEFAULT_NATIVE_MODELS_PATH)),
        }

    # vLLM/NIM backend configuration (ADR-0084)
    vllm_endpoint = os.environ.get("VETINARI_VLLM_ENDPOINT", "http://localhost:8000")
    nim_endpoint = os.environ.get("VETINARI_NIM_ENDPOINT", "http://localhost:8001")
    vllm_prefix_caching = os.environ.get("VETINARI_VLLM_PREFIX_CACHING_ENABLED", "true")
    nim_kv_reuse = os.environ.get("NIM_ENABLE_KV_CACHE_REUSE")
    nim_kv_reuse_enabled = _coerce_env_bool(
        nim_kv_reuse,
        default=bool(nim_setup and nim_setup.hardware_eligible),
    )
    nim_kv_host_offload = os.environ.get("NIM_ENABLE_KV_CACHE_HOST_OFFLOAD")
    config["inference_backend"]["vllm"] = {
        "enabled": "vllm" in backends,
        "endpoint": vllm_endpoint,
        "gpu_only": True,
        "semantic_cache_enabled": True,
        "cache_namespace": "vetinari",
        "cache_salt": os.environ.get("VETINARI_VLLM_CACHE_SALT", ""),
        "prefix_caching_enabled": vllm_prefix_caching.strip().lower() not in ("", "0", "false", "no", "off"),
        "prefix_caching_hash_algo": os.environ.get("VETINARI_VLLM_PREFIX_CACHING_HASH_ALGO", "sha256"),
        "container_setup": vllm_setup.to_config() if vllm_setup else {},
    }
    config["inference_backend"]["nim"] = {
        "enabled": "nim" in backends,
        "endpoint": nim_endpoint,
        "gpu_only": True,
        "semantic_cache_enabled": True,
        "cache_namespace": "vetinari",
        "kv_cache_reuse_enabled": nim_kv_reuse_enabled,
        "kv_cache_host_offload_enabled": (
            _coerce_env_bool(nim_kv_host_offload) if nim_kv_host_offload is not None else None
        ),
        "supports_cache_salt": False,
        "container_setup": nim_setup.to_config() if nim_setup else {},
    }

    rendered = yaml.safe_dump(config, default_flow_style=False, sort_keys=False)
    temp_path = config_path.with_name(f".{config_path.name}.tmp")
    temp_path.write_text(rendered, encoding="utf-8")
    temp_path.replace(config_path)
    return config_path


def _smoke_test() -> bool:
    """Run a quick import test to verify the vetinari package is functional.

    Returns:
        True if the package imports successfully.
    """
    try:
        import vetinari  # noqa: F401 - import intentionally probes or re-exports API surface

        return True
    except Exception as exc:
        logger.warning("Smoke test failed — vetinari import error: %s", exc)
        return False


# ── Main Wizard ──────────────────────────────────────────────────────────────


def run_wizard(
    skip_download: bool = False,
    non_interactive: bool = False,
    config_path: Path | None = None,
) -> WizardResult:
    """Run the first-time setup wizard.

    Performs hardware detection, model scanning, recommendation,
    optional download, configuration generation, and a smoke test.

    Args:
        skip_download: If True, skip the model download step.
        non_interactive: If True, accept all defaults without prompting
            (for CI/Docker environments, equivalent to ``--yes``).
        config_path: Override the config output path (default: ~/.vetinari/config.yaml).

    Returns:
        WizardResult with the outcome of each step.
    """
    result = WizardResult()

    # ── Step 1: Hardware detection ────────────────────────────────────────────
    console.print(f"\n[1/{WIZARD_STEPS}] Detecting hardware...")
    hardware = detect_hardware()
    result.hardware = hardware

    console.print(f"      CPU cores : {hardware.cpu_count}")
    console.print(f"      RAM       : {hardware.ram_gb:.1f} GB")
    if hardware.has_gpu:
        console.print(f"      GPU       : {hardware.gpu_name}")
        console.print(f"      VRAM      : {hardware.vram_gb:.1f} GB")
        vendor_info = []
        if hardware.cuda_available:
            vendor_info.append("CUDA")
        if hardware.metal_available:
            vendor_info.append("Metal")
        if vendor_info:
            console.print(f"      Accel     : {', '.join(vendor_info)}")
    else:
        console.print("      GPU       : not detected (CPU inference will be used)")

    vllm_setup = plan_vllm_container_setup(hardware)
    if vllm_setup.can_auto_start:
        console.print("      vLLM setup: auto mode requested; starting container...")
        vllm_setup = start_vllm_container(vllm_setup)
        if vllm_setup.started:
            console.print("      vLLM setup: container start requested successfully")
        else:
            result.errors.append(f"vLLM container start failed: {vllm_setup.error}")
            console.print("      vLLM setup: container start failed; see config for details")
    elif vllm_setup.status in {"guided_ready", "manual_required", "missing_prerequisites", "unsupported_hardware"}:
        logger.info("vLLM container setup status: %s", vllm_setup.status)
    result.vllm_setup = vllm_setup.to_config()

    nim_setup = plan_nim_container_setup(hardware)
    if nim_setup.can_auto_start:
        console.print("      NIM setup : auto mode requested; starting container...")
        nim_setup = start_nim_container(nim_setup)
        if nim_setup.started:
            console.print("      NIM setup : container start requested successfully")
        else:
            result.errors.append(f"NIM container start failed: {nim_setup.error}")
            console.print("      NIM setup : container start failed; see config for details")
    elif nim_setup.status in {"guided_ready", "manual_required", "missing_prerequisites", "unsupported_hardware"}:
        logger.info("NIM container setup status: %s", nim_setup.status)
    result.nim_setup = nim_setup.to_config()

    # ── Step 2: Scan for existing models and detect backends ────────────────
    console.print(f"\n[2/{WIZARD_STEPS}] Scanning for existing models and backends...")
    models_found = _scan_for_models()
    result.models_found = models_found
    detected_backends = _detect_available_backends(hardware)

    if models_found:
        console.print(f"      Found {len(models_found)} model file(s):")
        for mp in models_found[:10]:
            size_mb = mp.stat().st_size / (1024 * 1024) if mp.exists() else 0
            console.print(f"        - {mp.name} ({size_mb:.0f} MB)")
        if len(models_found) > 10:
            console.print(f"        ... and {len(models_found) - 10} more")
    else:
        console.print("      No existing models found.")

    console.print(f"      Backends   : {', '.join(detected_backends)}")
    if "vllm" in detected_backends:
        console.print("      vLLM       : detected (GPU-only, high throughput)")
    elif vllm_setup.hardware_eligible:
        console.print(f"      vLLM       : {vllm_setup.status.replace('_', ' ')}")
    if "nim" in detected_backends:
        console.print("      NVIDIA NIM : detected")
    elif nim_setup.hardware_eligible:
        console.print(f"      NVIDIA NIM : {nim_setup.status.replace('_', ' ')}")

    # ── Step 3: Model recommendation ─────────────────────────────────────────
    console.print(f"\n[3/{WIZARD_STEPS}] Recommending models for your hardware...")
    recommender = ModelRecommender()
    tier_label = recommender.get_tier_label(hardware)
    console.print(f"      Hardware tier: {tier_label}")

    # None means "all backends are considered" — we don't filter at wizard time
    # because the user may not have installed optional backends (vLLM, NIM) yet.
    available_backends: list[str] | None = None

    # Show portfolio recommendations organized by use case
    portfolio = recommender.recommend_portfolio(hardware, available_backends)
    _role_labels = {
        "grunt": "Classification & Routing (fast, small)",
        "worker": "Coding & Review (main workhorse)",
        "thinker": "Reasoning & Planning (complex tasks)",
    }
    for role in ("grunt", "worker", "thinker"):
        role_recs = portfolio.get(role, [])
        if not role_recs:
            continue
        console.print(f"\n      {_role_labels.get(role, role)}:")
        for rec in role_recs:
            marker = " (recommended)" if rec.is_primary else ""
            backend_tag = f" [{rec.backend}]" if rec.backend != "llama_cpp" else ""
            format_tag = f" ({rec.model_format.upper()})" if rec.model_format != "gguf" else ""
            offload_tag = (
                " [CPU offload]" if not rec.gpu_only and rec.size_gb > (hardware.effective_vram_gb or 0) else ""
            )
            console.print(f"        - {rec.name}{format_tag}{backend_tag}{offload_tag} ({rec.size_gb:.1f} GB){marker}")
            console.print(f"          {rec.reason}")
            if rec.best_for:
                console.print(f"          Best for: {', '.join(rec.best_for)}")

    # Flatten portfolio for download selection
    recommendations = recommender.recommend_models_multi_format(hardware, available_backends)

    # ── Step 4: Model download ───────────────────────────────────────────────
    console.print(f"\n[4/{WIZARD_STEPS}] Model download...")
    selected_model_path: Path | None = None

    if skip_download:
        console.print("      Skipping download (--skip-download).")
    elif models_found:
        console.print("      Using existing model(s) — download not needed.")
        selected_model_path = models_found[0]
    elif non_interactive:
        # Auto-select primary recommendation in CI mode
        primary = next((r for r in recommendations if r.is_primary), recommendations[0])
        console.print(f"      Auto-selecting: {primary.name}")
        selected_model_path = _download_model(primary)
        if selected_model_path:
            result.model_downloaded = selected_model_path
        else:
            result.errors.append(f"Download failed for {primary.name}")
            console.print("      Download failed — you can download manually later.")
    else:
        # Interactive: ask user
        choice = input("\n      Download a model? [Y/n] ").strip().lower()
        if choice in ("", "y", "yes"):
            primary = next((r for r in recommendations if r.is_primary), recommendations[0])
            selected_model_path = _download_model(primary)
            if selected_model_path:
                result.model_downloaded = selected_model_path
            else:
                result.errors.append(f"Download failed for {primary.name}")
        else:
            console.print("      Skipping download.")

    # ── Step 5: Write configuration ──────────────────────────────────────────
    console.print(f"\n[5/{WIZARD_STEPS}] Writing configuration...")
    try:
        cfg_path = _write_config(hardware, selected_model_path, config_path, detected_backends, vllm_setup, nim_setup)
        result.config_path = cfg_path
        console.print(f"      Config written to: {cfg_path}")
    except Exception as exc:
        result.errors.append(f"Config write failed: {exc}")
        console.print(f"      Config write failed: {exc}")

    # ── Step 6: Smoke test ───────────────────────────────────────────────────
    console.print(f"\n[6/{WIZARD_STEPS}] Running smoke test...")
    if _smoke_test():
        console.print("      Vetinari package imports successfully.")
        result.success = True
    else:
        result.errors.append("Smoke test failed — vetinari import error")
        console.print("      Smoke test failed — check installation.")

    # ── Summary ──────────────────────────────────────────────────────────────
    console.print("\n" + "=" * 50)
    if result.success:
        console.print("  Setup complete! Run 'vetinari serve' to start.")
    else:
        console.print("  Setup completed with warnings:")
        for err in result.errors:
            console.print(f"    - {err}")
    console.print("=" * 50)

    return result


def _download_model(recommendation: SetupModelRecommendation) -> Path | None:
    """Download a recommended model via huggingface_hub.

    Args:
        recommendation: The model recommendation to download.

    Returns:
        Path to the downloaded file, or None on failure.
    """
    try:
        import huggingface_hub  # type: ignore[import-untyped]  # noqa: F401 - import intentionally probes or re-exports API surface
    except ImportError:
        logger.warning(
            "huggingface_hub not installed — model download unavailable; install with: pip install huggingface-hub"  # noqa: VET301 — user guidance string
        )
        console.print("      huggingface_hub not installed. Install with:")
        console.print("        pip install huggingface-hub")  # noqa: VET301 — user guidance string
        return None

    target_dir = DOWNLOAD_NATIVE_MODELS_DIR if recommendation.backend in {"vllm", "nim"} else DOWNLOAD_GGUF_MODELS_DIR
    Path(target_dir).mkdir(parents=True, exist_ok=True)  # noqa: VET306 — user-scoped via get_user_dir()

    console.print(f"      Downloading {recommendation.name} ({recommendation.size_gb:.1f} GB)...")
    console.print(f"      From: {recommendation.repo_id}")

    try:
        from vetinari.model_discovery import ModelDiscovery

        downloaded = ModelDiscovery().download_model(
            repo_id=recommendation.repo_id,
            filename=recommendation.filename if recommendation.backend == "llama_cpp" else None,
            models_dir=target_dir,  # noqa: VET306 - constant read is config/test contract, not a write target
            backend=recommendation.backend,
            model_format=recommendation.model_format,
        )
        path = Path(str(downloaded["path"]))
        console.print(f"      Saved to: {path}")
        console.print(f"      Revision: {downloaded.get('revision')}")
        return path
    except Exception as exc:
        logger.warning("Model download failed for %s: %s", recommendation.repo_id, exc)
        console.print(f"      Download error: {exc}")
        return None
