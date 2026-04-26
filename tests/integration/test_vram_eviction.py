"""Integration tests for VRAM eviction under load.

Verifies that model eviction triggered during inference does not crash
the system — a critical production safety requirement.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from vetinari.models.vram_manager import ModelVRAMEstimate, VRAMManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_vram_manager() -> VRAMManager:
    """Provide a fresh VRAMManager for every test, bypassing the singleton.

    Creating the instance directly avoids cross-test contamination from
    the module-level singleton and from real GPU queries.
    """
    # Bypass hardware config loading and pynvml to stay fully offline
    with (
        patch("vetinari.models.vram_manager._ensure_nvml", return_value=False),
        patch("vetinari.models.vram_manager.VRAMManager._load_hardware_config"),
    ):
        manager = VRAMManager.__new__(VRAMManager)
        manager._gpu_total_gb = 32.0
        manager._cpu_offload_gb = 30.0
        manager._overhead_gb = 3.2  # 10% of 32 GB
        manager._estimates = {}
        manager._reservations = {}
        manager._leases = {}
        import threading

        manager._lock_rw = threading.RLock()
        manager._last_snapshot = None
        manager._current_phase = "execution"
        manager._gpu_arch = None
        import threading as _t

        manager._load_semaphore = _t.Semaphore(2)
    return manager


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _add_model(manager: VRAMManager, model_id: str, gpu_gb: float, priority: int = 5) -> None:
    """Register a model as loaded with a known VRAM footprint."""
    estimate = ModelVRAMEstimate(
        model_id=model_id,
        gpu_gb=gpu_gb,
        cpu_gb=0.0,
        last_used=time.time(),
        priority=priority,
    )
    manager._estimates[model_id] = estimate


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvictIdleModel:
    """Evicting a model that is not in use should succeed cleanly."""

    def test_evict_idle_model_removes_entry(self, isolated_vram_manager: VRAMManager) -> None:
        """register_unload for an idle model removes it from tracked estimates."""
        mgr = isolated_vram_manager
        _add_model(mgr, "model-idle-7b", gpu_gb=7.0)

        assert "model-idle-7b" in mgr._estimates
        mgr.register_unload("model-idle-7b")

        assert "model-idle-7b" not in mgr._estimates, "Model should be removed from estimates after register_unload"

    def test_evict_unknown_model_does_not_raise(self, isolated_vram_manager: VRAMManager) -> None:
        """Unloading a model that was never registered must not raise."""
        mgr = isolated_vram_manager
        # Must complete without exception
        mgr.register_unload("model-never-loaded")
        assert "model-never-loaded" not in mgr._estimates

    def test_evict_idle_model_frees_vram(self, isolated_vram_manager: VRAMManager) -> None:
        """Unloading a model removes its VRAM estimate, reducing tracked usage."""
        mgr = isolated_vram_manager
        _add_model(mgr, "model-large-14b", gpu_gb=14.0)

        gpu_before = sum(e.gpu_gb for e in mgr._estimates.values())
        mgr.register_unload("model-large-14b")
        gpu_after = sum(e.gpu_gb for e in mgr._estimates.values())

        assert gpu_after < gpu_before, (
            f"Tracked GPU usage should decrease after eviction: before={gpu_before:.1f}, after={gpu_after:.1f}"
        )
        assert "model-large-14b" not in mgr._estimates


class TestEvictDuringInference:
    """Eviction while inference is ongoing must degrade gracefully."""

    def test_evict_during_inference_does_not_crash(self, isolated_vram_manager: VRAMManager) -> None:
        """Calling register_unload while mark_used is active must not raise."""
        mgr = isolated_vram_manager
        _add_model(mgr, "model-active-8b", gpu_gb=8.0)

        # Simulate active inference by calling mark_used then immediately evicting
        mgr.mark_used("model-active-8b")
        # Eviction during inference should not raise; the registry removes the entry
        mgr.register_unload("model-active-8b")

        assert "model-active-8b" not in mgr._estimates, (
            "Model must be removed from estimates even when evicted during active mark_used"
        )

    def test_can_load_returns_false_after_eviction(self, isolated_vram_manager: VRAMManager) -> None:
        """A huge model that needs CPU offload is rejected when the CPU budget is consumed.

        can_load() checks CPU offload for models that exceed VRAM.  A model requiring
        50 GB total needs 28.8 GB of GPU + 21.2 GB of CPU offload.  When the CPU offload
        budget (30 GB) is already full, the model cannot be loaded.  After clearing all
        models it fits.
        """
        mgr = isolated_vram_manager

        # Fill CPU offload capacity so the 50 GB model cannot spill to RAM
        # 50 GB model needs: gpu_portion = 28.8 GB, cpu_portion = 21.2 GB
        # If cpu_used = 30 GB then cpu_free = 0 < 21.2 → cannot load
        mgr._estimates["model-cpu-hog"] = ModelVRAMEstimate(
            model_id="model-cpu-hog",
            gpu_gb=0.0,
            cpu_gb=30.0,  # fills the entire CPU offload budget
            last_used=time.time(),
        )

        # 50 GB model won't fit: VRAM free = 32-3.2 = 28.8 GB (no gpu usage),
        # needs cpu_portion = 50-28.8 = 21.2 GB but cpu_free = 30-30 = 0
        with patch.object(mgr, "get_model_vram_requirement", return_value=50.0):
            cannot_load = not mgr.can_load("model-huge-50b")

        assert cannot_load, "50 GB model should not fit when the CPU offload budget is fully consumed"

        # After evicting the CPU hog, the 50 GB model's offload needs (21.2 GB) fit in the freed 30 GB
        mgr.register_unload("model-cpu-hog")
        with patch.object(mgr, "get_model_vram_requirement", return_value=50.0):
            can_load_now = mgr.can_load("model-huge-50b")

        assert can_load_now, "50 GB model should fit after the CPU offload consumer is evicted"


class TestEvictLRUOrder:
    """recommend_eviction must pick the LRU + lowest-priority model first."""

    def test_evict_lru_order_picks_oldest(self, isolated_vram_manager: VRAMManager) -> None:
        """With equal priority, recommend_eviction picks the least-recently-used model."""
        mgr = isolated_vram_manager
        now = time.time()

        old_estimate = ModelVRAMEstimate(
            model_id="model-old-8b",
            gpu_gb=8.0,
            cpu_gb=0.0,
            last_used=now - 3600,  # 1 hour ago
            priority=5,
        )
        new_estimate = ModelVRAMEstimate(
            model_id="model-new-8b",
            gpu_gb=8.0,
            cpu_gb=0.0,
            last_used=now,  # just used
            priority=5,
        )
        mgr._estimates["model-old-8b"] = old_estimate
        mgr._estimates["model-new-8b"] = new_estimate

        # Need 20 GB but only ~12.8 GB free — must evict something.
        # Patch get_free_vram_gb to return a controlled value so the test
        # isn't affected by GPU snapshot or estimate arithmetic changes.
        with (
            patch.object(mgr, "get_model_vram_requirement", return_value=20.0),
            patch.object(mgr, "get_free_vram_gb", return_value=0.0),
        ):
            candidate = mgr.recommend_eviction("model-target-20b")

        assert candidate == "model-old-8b", (
            f"Expected LRU model 'model-old-8b' to be eviction candidate, got {candidate!r}"
        )

    def test_evict_priority_over_recency(self, isolated_vram_manager: VRAMManager) -> None:
        """A model with the smallest priority integer is evicted before one with a larger integer.

        The sort key is ``(priority, last_used)`` ascending — priority=2 sorts
        before priority=8, so the model with priority=2 is the first eviction
        candidate regardless of recency.
        """
        mgr = isolated_vram_manager
        now = time.time()

        # priority=2: sorts first → evicted first (even though recently used)
        mgr._estimates["model-priority-2"] = ModelVRAMEstimate(
            model_id="model-priority-2",
            gpu_gb=10.0,
            cpu_gb=0.0,
            last_used=now,
            priority=2,
        )
        # priority=8: sorts later → evicted last (even though old)
        mgr._estimates["model-priority-8"] = ModelVRAMEstimate(
            model_id="model-priority-8",
            gpu_gb=10.0,
            cpu_gb=0.0,
            last_used=now - 7200,
            priority=8,
        )

        with (
            patch.object(mgr, "get_model_vram_requirement", return_value=15.0),
            patch.object(mgr, "get_free_vram_gb", return_value=0.0),
        ):
            candidate = mgr.recommend_eviction("model-target-15b")

        assert candidate == "model-priority-2", (
            f"Expected model with priority=2 (sorted first) to be eviction candidate, got {candidate!r}"
        )

    def test_pinned_model_never_evicted(self, isolated_vram_manager: VRAMManager) -> None:
        """A pinned model is never returned as an eviction candidate."""
        mgr = isolated_vram_manager
        now = time.time()

        mgr._estimates["model-pinned"] = ModelVRAMEstimate(
            model_id="model-pinned",
            gpu_gb=15.0,
            cpu_gb=0.0,
            last_used=now - 7200,
            priority=10,
            is_pinned=True,
        )
        mgr._estimates["model-unpinned"] = ModelVRAMEstimate(
            model_id="model-unpinned",
            gpu_gb=5.0,
            cpu_gb=0.0,
            last_used=now - 100,
            priority=5,
            is_pinned=False,
        )

        with (
            patch.object(mgr, "get_model_vram_requirement", return_value=25.0),
            patch.object(mgr, "get_free_vram_gb", return_value=0.0),
        ):
            candidate = mgr.recommend_eviction("model-target-25b")

        assert candidate != "model-pinned", "Pinned model must never be returned as an eviction candidate"
        assert candidate == "model-unpinned", f"Expected unpinned model as candidate, got {candidate!r}"


class TestEvictAllModelsDegradation:
    """Evicting all models leaves the system in a valid (empty) state."""

    def test_evict_all_models_status_summary_still_works(self, isolated_vram_manager: VRAMManager) -> None:
        """After all models are evicted, status_summary returns a valid dict with zero used."""
        mgr = isolated_vram_manager
        _add_model(mgr, "model-a-7b", gpu_gb=7.0)
        _add_model(mgr, "model-b-14b", gpu_gb=14.0)

        mgr.register_unload("model-a-7b")
        mgr.register_unload("model-b-14b")

        # status_summary calls refresh() internally; patch to avoid registry access
        with patch.object(mgr, "refresh"):
            summary = mgr.status_summary()

        assert isinstance(summary, dict), "status_summary must return a dict"
        assert "loaded_models" in summary, "status_summary must include loaded_models key"
        assert len(summary["loaded_models"]) == 0, (
            f"Expected no loaded models after full eviction, got: {summary['loaded_models']}"
        )

    def test_can_load_returns_true_after_full_eviction(self, isolated_vram_manager: VRAMManager) -> None:
        """After evicting the CPU offload consumer, a large model can be accommodated."""
        mgr = isolated_vram_manager
        # Fill the entire CPU offload budget so a 40 GB model (needs 11.2 GB CPU) cannot load
        mgr._estimates["model-cpu-fill"] = ModelVRAMEstimate(
            model_id="model-cpu-fill",
            gpu_gb=0.0,
            cpu_gb=30.0,
            last_used=time.time(),
        )

        # 40 GB model: gpu_portion=28.8, cpu_portion=11.2 — cannot fit, cpu_free=0
        with patch.object(mgr, "get_model_vram_requirement", return_value=40.0):
            assert not mgr.can_load("model-40b"), "40 GB model should not fit before eviction"

        mgr.register_unload("model-cpu-fill")

        # After eviction, cpu_free=30 >= 11.2, so it fits
        with patch.object(mgr, "get_model_vram_requirement", return_value=40.0):
            assert mgr.can_load("model-40b"), "40 GB model should fit after full CPU offload eviction"
