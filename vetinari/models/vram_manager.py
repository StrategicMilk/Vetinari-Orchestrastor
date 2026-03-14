"""Vetinari VRAM Manager.

======================
Tracks GPU memory budget for a local LM Studio setup and advises on model
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

    from vetinari.vram_manager import get_vram_manager

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
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional pynvml import — graceful degradation if not installed
# ---------------------------------------------------------------------------
try:
    import pynvml  # type: ignore

    pynvml.nvmlInit()
    _PYNVML_AVAILABLE = True
except Exception:
    _PYNVML_AVAILABLE = False


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


@dataclass
class ModelVRAMEstimate:
    """VRAM + RAM usage estimate for a loaded model."""

    model_id: str
    gpu_gb: float  # VRAM used by this model
    cpu_gb: float  # System RAM used (CPU offload)
    last_used: float  # epoch timestamp
    priority: int = 5  # 1 = highest (keep), 10 = lowest (evict first)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class VRAMManager:
    """GPU memory budget manager for local LM Studio models."""

    _instance: VRAMManager | None = None
    _lock = threading.Lock()

    def __init__(self):
        self._gpu_total_gb: float = float(os.environ.get("VETINARI_GPU_VRAM_GB", "32"))
        self._cpu_offload_gb: float = float(os.environ.get("VETINARI_CPU_OFFLOAD_GB", "30"))
        self._overhead_gb: float = 2.0  # OS + CUDA context overhead
        self._estimates: dict[str, ModelVRAMEstimate] = {}
        self._lock_rw = threading.RLock()
        self._last_snapshot: VRAMSnapshot | None = None

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

            cfg_path = Path(__file__).parent.parent / "config" / "models.yaml"
            if cfg_path.exists():
                cfg = load_yaml(str(cfg_path))
                hw = cfg.get("hardware", {})
                if hw.get("gpu_vram_gb"):
                    self._gpu_total_gb = float(hw["gpu_vram_gb"])
                if hw.get("max_cpu_offload_gb"):
                    self._cpu_offload_gb = float(hw["max_cpu_offload_gb"])
        except Exception as e:
            logger.debug("[VRAMManager] Could not load hardware config: %s", e)

    # ------------------------------------------------------------------
    # Real-time VRAM reading
    # ------------------------------------------------------------------

    def get_gpu_snapshot(self, gpu_index: int = 0) -> VRAMSnapshot | None:
        """Return live GPU memory snapshot via pynvml if available.

        Returns:
            The VRAMSnapshot | None result.
        """
        if not _PYNVML_AVAILABLE:
            return None
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            snap = VRAMSnapshot(
                gpu_index=gpu_index,
                total_gb=mem.total / 1e9,
                used_gb=mem.used / 1e9,
                free_gb=mem.free / 1e9,
            )
            self._last_snapshot = snap
            return snap
        except Exception as e:
            logger.debug("[VRAMManager] pynvml snapshot failed: %s", e)
            return None

    def get_free_vram_gb(self) -> float:
        """Return estimated free VRAM in GB.

        Returns:
            The computed value.
        """
        snap = self.get_gpu_snapshot()
        if snap:
            return snap.free_gb

        # Fallback: estimate from tracked loads
        used = sum(e.gpu_gb for e in self._estimates.values())
        return max(0.0, self._gpu_total_gb - self._overhead_gb - used)

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
            logger.debug("[VRAMManager] refresh() failed: %s", e)

    def mark_used(self, model_id: str) -> None:
        """Update last-used timestamp for a model (call before inference)."""
        with self._lock_rw:
            if model_id in self._estimates:
                self._estimates[model_id].last_used = time.time()

    def register_load(self, model_id: str, vram_gb: float, cpu_gb: float = 0.0) -> None:
        """Manually register that a model has been loaded.

        Args:
            model_id: The model id.
            vram_gb: The vram gb.
            cpu_gb: The cpu gb.
        """
        with self._lock_rw:
            self._estimates[model_id] = ModelVRAMEstimate(
                model_id=model_id,
                gpu_gb=vram_gb,
                cpu_gb=cpu_gb,
                last_used=time.time(),
            )

    def register_unload(self, model_id: str) -> None:
        """Manually register that a model has been unloaded."""
        with self._lock_rw:
            self._estimates.pop(model_id, None)

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
        except Exception:  # noqa: S110, VET022
            pass
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
            The result string.
        """
        required = self.get_model_vram_requirement(model_id)
        free_vram = self.get_free_vram_gb()

        if required <= free_vram:
            return None  # No eviction needed

        with self._lock_rw:
            candidates = sorted(
                self._estimates.values(),
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

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status_summary(self) -> dict[str, Any]:
        """Return a summary dict for dashboard / logging.

        Returns:
            The result string.
        """
        self.refresh()
        snap = self.get_gpu_snapshot()
        used_by_models = sum(e.gpu_gb for e in self._estimates.values())
        free_est = self.get_free_vram_gb()

        return {
            "gpu_total_gb": snap.total_gb if snap else self._gpu_total_gb,
            "gpu_used_gb": snap.used_gb if snap else (used_by_models + self._overhead_gb),
            "gpu_free_gb": snap.free_gb if snap else free_est,
            "cpu_offload_budget_gb": self._cpu_offload_gb,
            "cpu_offload_used_gb": sum(e.cpu_gb for e in self._estimates.values()),
            "pynvml_available": _PYNVML_AVAILABLE,
            "loaded_models": {mid: {"gpu_gb": e.gpu_gb, "cpu_gb": e.cpu_gb} for mid, e in self._estimates.items()},
        }


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
