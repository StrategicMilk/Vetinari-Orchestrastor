"""Vetinari VRAM Manager.

======================
Tracks GPU memory budget for llama-cpp-python local inference and advises on model
load/unload decisions.

Hardware target: RTX 5090 (32 GB VRAM) + 9950X3D + 64 GB DDR5.

Features
--------
- Real-time VRAM monitoring via ``pynvml`` (optional) with fallback estimates
- Tracks which models the registry believes are loaded + their VRAM footprint
- Checks whether a model fits before routing to it
- Recommends which model to evict (LRU + lowest priority) to make room
- Respects CPU-offload flag: models >32 GB can run with split GPU+RAM

Usage::

    from vetinari.models.vram_manager import get_vram_manager

    manager = get_vram_manager()
    manager.refresh()

    if manager.can_load("qwen3-vl-32b"):
        logger.debug("Fits in VRAM")
    else:
        evict = manager.recommend_eviction("qwen3-vl-32b")
        logger.debug("Unload %s first", evict)

    logger.debug(manager.status_summary())
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional nvidia-ml-py import — lazy init to avoid module-level I/O
# ---------------------------------------------------------------------------
_pynvml = None  # Lazy: initialized on first GPU query
_PYNVML_AVAILABLE: bool | None = None  # None = not yet checked


def _ensure_nvml() -> bool:
    """Lazily initialize NVML on first use, avoiding module-level I/O.

    Returns:
        True if pynvml is available and initialized.
    """
    global _pynvml, _PYNVML_AVAILABLE
    if _PYNVML_AVAILABLE is not None:
        return _PYNVML_AVAILABLE
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        _pynvml = pynvml
        _PYNVML_AVAILABLE = True
    except Exception:
        _PYNVML_AVAILABLE = False
    return _PYNVML_AVAILABLE


# Phase transition log message format (noqa: VET315 — not CSS, Python log format)
_PHASE_TRANSITION_LOG = "VRAM phase change: %s -> %s"

# Safety margin fraction — reserve 10% of VRAM for CUDA context + KV cache growth
_SAFETY_MARGIN = 0.10

# Thermal thresholds (Celsius)
_THERMAL_WARN_C = 80
_THERMAL_THROTTLE_C = 85

# Maximum concurrent model loads
_MAX_CONCURRENT_LOADS = 2

# VRAM cost of KV cache per token for each quantization type.
# Values are bytes per token; divide by 1024**3 to get GB.
# q4_0 uses 4x fewer bytes than f16 — allows 4x longer context for the same VRAM.
_KV_BYTES_PER_TOKEN: dict[str, int] = {
    "f16": 2048,  # 2 bytes/element — llama.cpp default
    "q8_0": 1024,  # 1 byte/element — half the VRAM vs f16
    "q4_0": 512,  # 0.5 bytes/element — quarter the VRAM vs f16
}


class ExecutionPhase:
    """Pipeline execution phase — determines which models should be loaded."""

    PLANNING = "planning"
    EXECUTION = "execution"
    REVIEW = "review"


@dataclass
class ModelLease:
    """Active claim on a loaded model — prevents eviction during inference.

    When an agent begins inference against a model, it acquires a lease.
    The VRAMManager refuses to evict models with active leases.  Leases
    auto-expire after ``max_duration_s`` to prevent leaked claims from
    permanently pinning models.
    """

    model_id: str
    holder_id: str  # e.g. "worker:task-42"
    acquired_at: float = field(default_factory=time.time)
    max_duration_s: float = 300.0  # 5-minute default expiry

    def __repr__(self) -> str:
        return f"ModelLease(model_id={self.model_id!r}, holder_id={self.holder_id!r}, expired={self.is_expired})"

    @property
    def is_expired(self) -> bool:
        """True when the lease has exceeded its max duration."""
        return (time.time() - self.acquired_at) > self.max_duration_s


class GPUArchitecture:
    """GPU architecture family detected via CUDA compute capability."""

    UNKNOWN = "unknown"
    ADA_LOVELACE = "ada_lovelace"  # sm_89 (RTX 40xx)
    BLACKWELL = "blackwell"  # sm_120 (RTX 50xx)


def detect_gpu_architecture(gpu_index: int = 0) -> str:
    """Detect GPU architecture via CUDA compute capability.

    Args:
        gpu_index: GPU device index.

    Returns:
        GPUArchitecture string value.
    """
    if not _ensure_nvml():
        return GPUArchitecture.UNKNOWN
    try:
        handle = _pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        major, minor = _pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        sm = major * 10 + minor
        if sm >= 120:
            return GPUArchitecture.BLACKWELL
        if sm >= 89:
            return GPUArchitecture.ADA_LOVELACE
        return GPUArchitecture.UNKNOWN
    except Exception as exc:
        logger.warning("GPU architecture detection failed: %s", exc)
        return GPUArchitecture.UNKNOWN


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VRAMSnapshot:
    """Point-in-time GPU memory reading."""

    gpu_index: int
    total_gb: float
    used_gb: float
    free_gb: float
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"VRAMSnapshot(gpu_index={self.gpu_index!r}, free_gb={self.free_gb!r}, used_gb={self.used_gb!r})"


@dataclass
class ModelVRAMEstimate:
    """VRAM + RAM usage estimate for a loaded model."""

    model_id: str
    gpu_gb: float  # VRAM used by this model's weights
    cpu_gb: float  # System RAM used (CPU offload)
    last_used: float  # epoch timestamp
    priority: int = 5  # 1 = highest (keep), 10 = lowest (evict first)
    kv_cache_gb: float = 0.0  # VRAM used by KV cache for this model
    is_pinned: bool = False  # True = never evict this model

    @property
    def total_gpu_gb(self) -> float:
        """Total GPU memory including KV cache."""
        return self.gpu_gb + self.kv_cache_gb

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return (
            f"ModelVRAMEstimate(model_id={self.model_id!r},"
            f" gpu_gb={self.gpu_gb!r}, kv={self.kv_cache_gb:.1f}, priority={self.priority!r})"
        )


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class VRAMManager:
    """GPU memory budget manager for local llama-cpp-python models."""

    _instance: VRAMManager | None = None
    _lock = threading.Lock()

    def __init__(self):
        self._gpu_total_gb: float = float(os.environ.get("VETINARI_GPU_VRAM_GB", "32"))
        self._cpu_offload_gb: float = float(os.environ.get("VETINARI_CPU_OFFLOAD_GB", "30"))
        self._overhead_gb: float = self._gpu_total_gb * _SAFETY_MARGIN  # 10% safety margin
        self._estimates: dict[str, ModelVRAMEstimate] = {}
        self._reservations: dict[str, float] = {}  # model_id -> reserved GB (pre-load claim)
        self._lock_rw = threading.RLock()
        self._last_snapshot: VRAMSnapshot | None = None
        self._current_phase: str = ExecutionPhase.EXECUTION
        self._gpu_arch: str | None = None  # Lazy detected
        self._load_semaphore = threading.Semaphore(_MAX_CONCURRENT_LOADS)
        self._leases: dict[str, ModelLease] = {}  # holder_id -> active lease

        # Try to load hardware config from models.yaml
        self._load_hardware_config()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> VRAMManager:
        """Get instance.

        Returns:
            The VRAMManager result.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _load_hardware_config(self) -> None:
        try:
            from pathlib import Path

            from vetinari.utils import load_yaml

            cfg_path = Path(__file__).parent.parent / "config" / "models.yaml"  # noqa: VET306 — config read, not artifact write
            if cfg_path.exists():
                cfg = load_yaml(str(cfg_path))
                hw = cfg.get("hardware", {})
                if hw.get("gpu_vram_gb"):
                    self._gpu_total_gb = float(hw["gpu_vram_gb"])
                    self._overhead_gb = self._gpu_total_gb * _SAFETY_MARGIN
                if hw.get("max_cpu_offload_gb"):
                    self._cpu_offload_gb = float(hw["max_cpu_offload_gb"])
        except Exception as e:
            logger.warning("[VRAMManager] Could not load hardware config: %s", e)

    # ------------------------------------------------------------------
    # Real-time VRAM reading
    # ------------------------------------------------------------------

    def get_gpu_snapshot(self, gpu_index: int = 0) -> VRAMSnapshot | None:
        """Return live GPU memory snapshot via nvidia-ml-py if available.

        Args:
            gpu_index: GPU device index (default: 0).

        Returns:
            VRAMSnapshot with current GPU memory readings, or None if unavailable.
        """
        if not _ensure_nvml():
            return None
        try:
            handle = _pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            mem = _pynvml.nvmlDeviceGetMemoryInfo(handle)
            snap = VRAMSnapshot(
                gpu_index=gpu_index,
                total_gb=mem.total / 1e9,
                used_gb=mem.used / 1e9,
                free_gb=mem.free / 1e9,
            )
            self._last_snapshot = snap
            return snap
        except Exception as e:
            logger.warning("VRAMManager GPU snapshot failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Thermal monitoring
    # ------------------------------------------------------------------

    def get_gpu_temperature(self, gpu_index: int = 0) -> int | None:
        """Read GPU core temperature in Celsius via nvidia-ml-py.

        Args:
            gpu_index: GPU device index.

        Returns:
            Temperature in Celsius, or None if unavailable.
        """
        if not _ensure_nvml():
            return None
        try:
            handle = _pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            temp = _pynvml.nvmlDeviceGetTemperature(handle, _pynvml.NVML_TEMPERATURE_GPU)
            if temp >= _THERMAL_WARN_C:
                logger.warning("GPU temperature %d\u00b0C \u2014 approaching thermal limit", temp)
            return temp
        except Exception as exc:
            logger.warning(
                "GPU temperature read failed for GPU %d: %s — temperature monitoring unavailable", gpu_index, exc
            )
            return None

    def is_thermal_throttled(self, gpu_index: int = 0) -> bool:
        """Return True when GPU temp is at or above throttle threshold (85C).

        Args:
            gpu_index: GPU device index.

        Returns:
            True if thermal throttling should be applied.
        """
        temp = self.get_gpu_temperature(gpu_index)
        return temp is not None and temp >= _THERMAL_THROTTLE_C

    def get_gpu_architecture(self, gpu_index: int = 0) -> str:
        """Return the detected GPU architecture (cached after first detection).

        Args:
            gpu_index: GPU device index.

        Returns:
            GPUArchitecture string value.
        """
        if self._gpu_arch is None:
            self._gpu_arch = detect_gpu_architecture(gpu_index)
        return self._gpu_arch

    def get_free_vram_gb(self) -> float:
        """Return estimated free VRAM in GB (accounts for reservations and KV cache).

        Returns:
            Estimated free VRAM considering loaded models, reservations, and KV caches.
        """
        snap = self.get_gpu_snapshot()
        reserved = sum(self._reservations.values())
        if snap:
            return max(0.0, snap.free_gb - reserved)

        # Fallback: estimate from tracked loads + reservations
        used = sum(e.total_gpu_gb for e in self._estimates.values())
        return max(0.0, self._gpu_total_gb - self._overhead_gb - used - reserved)

    def get_used_vram_gb(self) -> float:
        """Return estimated used VRAM in GB.

        Returns:
            The computed value.
        """
        snap = self.get_gpu_snapshot()
        if snap:
            return snap.used_gb
        used = sum(e.gpu_gb for e in self._estimates.values())
        return self._overhead_gb + used

    # ------------------------------------------------------------------
    # Model tracking
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Sync tracked model state with the ModelRegistry."""
        try:
            from vetinari.models.model_registry import get_model_registry

            registry = get_model_registry()
            loaded = registry.get_loaded_local_models()
            with self._lock_rw:
                # Remove entries no longer loaded
                loaded_ids = {m.model_id for m in loaded}
                stale = [mid for mid in self._estimates if mid not in loaded_ids]
                for mid in stale:
                    del self._estimates[mid]
                # Add new entries
                for m in loaded:
                    if m.model_id not in self._estimates:
                        self._estimates[m.model_id] = ModelVRAMEstimate(
                            model_id=m.model_id,
                            gpu_gb=min(m.memory_requirements_gb, self._gpu_total_gb),
                            cpu_gb=max(0.0, m.memory_requirements_gb - self._gpu_total_gb),
                            last_used=time.time(),
                        )
        except Exception as e:
            logger.warning("[VRAMManager] refresh() failed: %s", e)

    def mark_used(self, model_id: str) -> None:
        """Update last-used timestamp for a model (call before inference)."""
        with self._lock_rw:
            if model_id in self._estimates:
                estimate = self._estimates[model_id]
                now = time.time()
                estimate.last_used = now if now > estimate.last_used else math.nextafter(estimate.last_used, math.inf)

    def register_load(self, model_id: str, vram_gb: float, cpu_gb: float = 0.0) -> None:
        """Register that a model has been loaded, clearing any pending reservation.

        Args:
            model_id: Model identifier.
            vram_gb: VRAM consumed by model weights in GB.
            cpu_gb: System RAM consumed by CPU-offloaded layers in GB.
        """
        with self._lock_rw:
            self._reservations.pop(model_id, None)
            self._estimates[model_id] = ModelVRAMEstimate(
                model_id=model_id,
                gpu_gb=vram_gb,
                cpu_gb=cpu_gb,
                last_used=time.time(),
            )

    def register_unload(self, model_id: str) -> None:
        """Register that a model has been unloaded."""
        with self._lock_rw:
            self._estimates.pop(model_id, None)
            self._reservations.pop(model_id, None)

    # ------------------------------------------------------------------
    # Atomic reservations
    # ------------------------------------------------------------------

    def reserve(self, model_id: str, gpu_gb: float) -> bool:
        """Atomically reserve VRAM before loading starts, preventing TOCTOU races.

        Args:
            model_id: Model to reserve VRAM for.
            gpu_gb: VRAM to reserve in GB.

        Returns:
            True if reservation succeeded, False if insufficient VRAM.
        """
        with self._lock_rw:
            if model_id in self._reservations:
                return True
            free = self.get_free_vram_gb()
            if gpu_gb > free:
                return False
            self._reservations[model_id] = gpu_gb
            logger.debug("VRAM reserved %.1f GB for %s", gpu_gb, model_id)
            return True

    def release_reservation(self, model_id: str) -> None:
        """Release a VRAM reservation (e.g., if load fails).

        Args:
            model_id: Model whose reservation to release.
        """
        with self._lock_rw:
            removed = self._reservations.pop(model_id, None)
            if removed is not None:
                logger.debug("VRAM reservation released for %s (%.1f GB)", model_id, removed)

    def acquire_load_slot(self) -> bool:
        """Acquire a concurrent model load slot (blocks up to 30s).

        Returns:
            True if slot acquired, False on timeout.
        """
        return self._load_semaphore.acquire(timeout=30.0)

    def release_load_slot(self) -> None:
        """Release a concurrent model load slot."""
        self._load_semaphore.release()

    # ------------------------------------------------------------------
    # Pinned models
    # ------------------------------------------------------------------

    def pin(self, model_id: str) -> None:
        """Mark a model as pinned — never evicted by recommend_eviction.

        Args:
            model_id: Model to pin.
        """
        with self._lock_rw:
            if model_id in self._estimates:
                self._estimates[model_id].is_pinned = True
                logger.info("Model %s pinned", model_id)

    def unpin(self, model_id: str) -> None:
        """Remove the pin from a model, making it eligible for eviction.

        Args:
            model_id: Model to unpin.
        """
        with self._lock_rw:
            if model_id in self._estimates:
                self._estimates[model_id].is_pinned = False

    def update_kv_cache(self, model_id: str, kv_cache_gb: float) -> None:
        """Update KV cache size tracking for a loaded model.

        Args:
            model_id: Model identifier.
            kv_cache_gb: Current KV cache size in GB.
        """
        with self._lock_rw:
            if model_id in self._estimates:
                self._estimates[model_id].kv_cache_gb = kv_cache_gb

    # ------------------------------------------------------------------
    # Phase-based model management
    # ------------------------------------------------------------------

    def set_phase(self, phase: str) -> None:
        """Set the current execution phase for VRAM budgeting decisions.

        Args:
            phase: One of ExecutionPhase.PLANNING, .EXECUTION, .REVIEW.
        """
        old = self._current_phase
        self._current_phase = phase
        if old != phase:
            logger.info(_PHASE_TRANSITION_LOG, old, phase)

    def get_phase_recommendation(self) -> dict[str, Any]:
        """Return recommended model configuration for the current phase.

        Returns:
            Dict with phase name, recommended models to load/unload, and VRAM budget.
        """
        recs = {
            ExecutionPhase.PLANNING: {
                "load": ["foreman-32b", "embeddings", "draft-1.5b"],
                "unload": ["worker-moe", "sub-foreman-9b"],
                "estimated_vram_gb": 20.5,
            },
            ExecutionPhase.EXECUTION: {
                "load": ["worker-moe", "sub-foreman-9b", "embeddings", "draft-1.5b"],
                "unload": ["foreman-32b"],
                "estimated_vram_gb": 20.5,
            },
            ExecutionPhase.REVIEW: {
                "load": ["foreman-32b", "embeddings"],
                "unload": ["worker-moe", "sub-foreman-9b"],
                "estimated_vram_gb": 19.5,
            },
        }
        rec = recs.get(self._current_phase, {})
        return {"phase": self._current_phase, "gpu_total_gb": self._gpu_total_gb, **rec}

    # ------------------------------------------------------------------
    # Model leases — prevent eviction during active inference
    # ------------------------------------------------------------------

    def acquire_lease(self, model_id: str, holder_id: str, max_duration_s: float = 300.0) -> bool:
        """Claim a model for active inference, protecting it from eviction.

        Args:
            model_id: Model to lease.
            holder_id: Unique identifier for the holder (e.g. ``"worker:task-42"``).
            max_duration_s: Lease auto-expiry in seconds.

        Returns:
            True if the lease was granted (model is tracked as loaded or reserved).
        """
        with self._lock_rw:
            self._reap_expired_leases()
            is_known = model_id in self._estimates or model_id in self._reservations
            if not is_known:
                logger.warning("Lease denied for %s — model not tracked as loaded", model_id)
                return False
            self._leases[holder_id] = ModelLease(
                model_id=model_id,
                holder_id=holder_id,
                max_duration_s=max_duration_s,
            )
            logger.debug("Lease granted: %s -> %s (%.0fs)", holder_id, model_id, max_duration_s)
            return True

    def release_lease(self, holder_id: str) -> None:
        """Release a model lease after inference completes.

        Args:
            holder_id: The holder identifier used when acquiring.
        """
        with self._lock_rw:
            removed = self._leases.pop(holder_id, None)
            if removed:
                logger.debug("Lease released: %s -> %s", holder_id, removed.model_id)

    def is_leased(self, model_id: str) -> bool:
        """Return True if any non-expired lease exists for this model.

        Args:
            model_id: Model to check.

        Returns:
            True if at least one active lease protects this model.
        """
        with self._lock_rw:
            self._reap_expired_leases()
            return any(lease.model_id == model_id for lease in self._leases.values())

    def active_lease_count(self) -> int:
        """Return the number of active (non-expired) leases.

        Returns:
            Count of leases currently held, after expiring any that have exceeded their duration.
        """
        with self._lock_rw:
            self._reap_expired_leases()
            return len(self._leases)

    def get_leased_model_ids(self) -> set[str]:
        """Return model IDs that have at least one active lease.

        Returns:
            Set of model_id strings currently referenced by at least one non-expired lease.
        """
        with self._lock_rw:
            self._reap_expired_leases()
            return {lease.model_id for lease in self._leases.values()}

    def _reap_expired_leases(self) -> None:
        """Remove leases that have exceeded their max duration (caller holds lock)."""
        expired = [hid for hid, lease in self._leases.items() if lease.is_expired]
        for hid in expired:
            lease = self._leases.pop(hid)
            logger.warning("Lease expired: %s -> %s (held %.0fs)", hid, lease.model_id, lease.max_duration_s)

    # ------------------------------------------------------------------
    # Capacity decisions
    # ------------------------------------------------------------------

    def get_model_vram_requirement(self, model_id: str) -> float:
        """Return estimated VRAM requirement for a model in GB.

        Returns:
            The computed value.
        """
        try:
            from vetinari.models.model_registry import get_model_registry

            info = get_model_registry().get_model_info(model_id)
            if info:
                return float(info.memory_requirements_gb)
        except Exception:
            logger.warning("Model registry lookup failed for %s, using size estimator", model_id)
        from vetinari.utils import estimate_model_memory_gb

        return float(estimate_model_memory_gb(model_id))

    def can_load(self, model_id: str) -> bool:
        """Return True if the model fits within available VRAM + CPU offload budget.

        Returns:
            True if successful, False otherwise.
        """
        required = self.get_model_vram_requirement(model_id)
        free_vram = self.get_free_vram_gb()

        # Fits entirely in VRAM
        if required <= free_vram:
            return True

        # Check if CPU offload would cover the remainder
        gpu_portion = min(required, self._gpu_total_gb - self._overhead_gb)
        cpu_portion = required - gpu_portion
        cpu_used = sum(e.cpu_gb for e in self._estimates.values())
        cpu_free = self._cpu_offload_gb - cpu_used

        if cpu_portion <= cpu_free:
            logger.debug(
                f"[VRAMManager] {model_id} needs CPU offload: {gpu_portion:.1f} GB GPU + {cpu_portion:.1f} GB RAM"
            )
            return True

        return False

    def recommend_eviction(self, model_id: str) -> str | None:
        """Return the model_id that should be evicted to make room for ``model_id``.

        Prioritises: lowest priority first, then least-recently-used.
        Returns None if no loaded models exist or eviction would not help.

        Returns:
            The model_id of the best eviction candidate (lowest priority, then
            oldest last-used), or None if no eviction is needed because the
            requested model already fits in free VRAM.
        """
        required = self.get_model_vram_requirement(model_id)
        free_vram = self.get_free_vram_gb()

        if required <= free_vram:
            return None  # No eviction needed

        with self._lock_rw:
            self._reap_expired_leases()
            leased_ids = {lease.model_id for lease in self._leases.values()}
            # Skip pinned and leased models — they are never eviction candidates
            candidates = sorted(
                [e for e in self._estimates.values() if not e.is_pinned and e.model_id not in leased_ids],
                key=lambda e: (e.priority, e.last_used),  # lower priority + older = evict first
            )

        for candidate in candidates:
            if candidate.model_id == model_id:
                continue
            freed = candidate.gpu_gb
            if free_vram + freed >= required:
                return candidate.model_id

        # Nothing single-model eviction can help
        return candidates[0].model_id if candidates else None

    def get_evictable_vram_gb(self) -> float:
        """Return the total VRAM that could be freed by evicting non-pinned, non-leased models.

        Useful for upfront capacity planning: free VRAM + evictable VRAM = maximum
        VRAM available for new models.

        Returns:
            Total VRAM in GB that eviction could recover.
        """
        with self._lock_rw:
            self._reap_expired_leases()
            leased_ids = {lease.model_id for lease in self._leases.values()}
            return sum(
                e.total_gpu_gb for e in self._estimates.values() if not e.is_pinned and e.model_id not in leased_ids
            )

    def get_max_available_vram_gb(self) -> float:
        """Return maximum VRAM that could become available (free + evictable).

        This is the upper bound on what a new model could use, assuming all
        non-protected models are evicted first.

        Returns:
            Maximum available VRAM in GB.
        """
        return self.get_free_vram_gb() + self.get_evictable_vram_gb()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status_summary(self) -> dict[str, Any]:
        """Return a summary dict for dashboard / logging.

        Returns:
            A dict with GPU totals (``gpu_total_gb``, ``gpu_used_gb``,
            ``gpu_free_gb``), CPU offload budget and usage, whether pynvml
            is available, and a ``loaded_models`` sub-dict mapping each
            tracked model_id to its current ``gpu_gb`` and ``cpu_gb``
            allocations.
        """
        self.refresh()
        snap = self.get_gpu_snapshot()
        used_by_models = sum(e.gpu_gb for e in self._estimates.values())
        free_est = self.get_free_vram_gb()

        temp = self.get_gpu_temperature()
        return {
            "gpu_total_gb": snap.total_gb if snap else self._gpu_total_gb,
            "gpu_used_gb": snap.used_gb if snap else (used_by_models + self._overhead_gb),
            "gpu_free_gb": snap.free_gb if snap else free_est,
            "cpu_offload_budget_gb": self._cpu_offload_gb,
            "cpu_offload_used_gb": sum(e.cpu_gb for e in self._estimates.values()),
            "pynvml_available": _ensure_nvml(),
            "gpu_architecture": self.get_gpu_architecture(),
            "gpu_temperature_c": temp,
            "thermal_throttled": temp is not None and temp >= _THERMAL_THROTTLE_C,
            "current_phase": self._current_phase,
            "pending_reservations": dict(self._reservations),
            "active_leases": {
                hid: {"model_id": lease.model_id, "age_s": round(time.time() - lease.acquired_at, 1)}
                for hid, lease in self._leases.items()
                if not lease.is_expired
            },
            "loaded_models": {
                mid: {"gpu_gb": e.gpu_gb, "cpu_gb": e.cpu_gb, "kv_cache_gb": e.kv_cache_gb, "pinned": e.is_pinned}
                for mid, e in self._estimates.items()
            },
        }

    def recommend_kv_quant_for_context(
        self,
        context_length: int,
        model_vram_gb: float = 0.0,
    ) -> str:
        """Recommend a KV cache quantization type given context length and VRAM constraints.

        Estimates the VRAM cost of the KV cache at f16 precision and downgrades
        to q8_0 or q4_0 when the budget is tight.  The thresholds are:
          - < 2 GB free after model load → q4_0 (4x cheaper than f16)
          - < 4 GB free after model load → q8_0 (2x cheaper than f16)
          - Otherwise → f16 (best quality, no quantization artefacts)

        Args:
            context_length: Desired context window in tokens.
            model_vram_gb: VRAM already committed to model weights (GB).

        Returns:
            One of "f16", "q8_0", or "q4_0".
        """
        with self._lock_rw:
            used_gb = sum(e.total_gpu_gb for e in self._estimates.values())
            free_gb = max(0.0, self._gpu_total_gb - self._overhead_gb - used_gb - model_vram_gb)

        # Estimate KV cache cost at f16 (worst case) to decide if we need to quantize
        kv_f16_gb = context_length * _KV_BYTES_PER_TOKEN["f16"] / (1024**3)

        if kv_f16_gb > free_gb * 0.75:
            # Very tight — use q4_0 to quarter the KV cache footprint
            return "q4_0"
        if kv_f16_gb > free_gb * 0.50:
            # Moderate pressure — q8_0 halves the cost with minimal quality loss
            return "q8_0"
        return "f16"


# ---------------------------------------------------------------------------
# Module-level accessor
# ---------------------------------------------------------------------------

_vram_manager: VRAMManager | None = None
_vram_lock = threading.Lock()


def get_vram_manager() -> VRAMManager:
    """Return the global VRAMManager singleton (created lazily).

    Returns:
        The VRAMManager result.
    """
    global _vram_manager
    if _vram_manager is None:
        with _vram_lock:
            if _vram_manager is None:
                _vram_manager = VRAMManager.get_instance()
    return _vram_manager
