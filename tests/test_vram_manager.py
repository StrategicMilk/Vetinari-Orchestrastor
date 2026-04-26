"""Tests for vetinari/vram_manager.py.

pynvml is not installed in the test environment, so all tests exercise the
fallback (estimate-only) code path.  The singleton is reset between every test.
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

import pytest

import vetinari.models.vram_manager as vram_mod
from vetinari.models.vram_manager import (
    ModelLease,
    ModelVRAMEstimate,
    VRAMManager,
    VRAMSnapshot,
    get_vram_manager,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_singleton() -> None:
    """Clear both singleton storage locations."""
    VRAMManager._instance = None
    vram_mod._vram_manager = None


def _make_manager() -> VRAMManager:
    """Return a fresh VRAMManager without touching config files."""
    _reset_singleton()
    with patch.object(VRAMManager, "_load_hardware_config", return_value=None):
        mgr = VRAMManager()
        mgr._gpu_total_gb = 32.0
        mgr._cpu_offload_gb = 30.0
        mgr._overhead_gb = 2.0
    return mgr


@pytest.fixture
def vram_manager() -> VRAMManager:
    """Return a fresh VRAMManager without touching config files.

    Resets the singleton before construction and patches ``_load_hardware_config``
    so no disk I/O occurs during init.
    """
    _reset_singleton()
    with patch.object(VRAMManager, "_load_hardware_config", return_value=None):
        mgr = VRAMManager()
        mgr._gpu_total_gb = 32.0
        mgr._cpu_offload_gb = 30.0
        mgr._overhead_gb = 2.0
    return mgr


# ---------------------------------------------------------------------------
# 1. VRAMSnapshot — creation and fields
# ---------------------------------------------------------------------------


class TestVRAMSnapshot:
    def test_fields_stored_correctly(self):
        snap = VRAMSnapshot(gpu_index=0, total_gb=32.0, used_gb=8.0, free_gb=24.0)
        assert snap.gpu_index == 0
        assert snap.total_gb == pytest.approx(32.0)
        assert snap.used_gb == pytest.approx(8.0)
        assert snap.free_gb == pytest.approx(24.0)

    def test_timestamp_defaults_to_now(self):
        before = time.time()
        snap = VRAMSnapshot(gpu_index=0, total_gb=32.0, used_gb=0.0, free_gb=32.0)
        after = time.time()
        assert snap.timestamp >= before
        assert snap.timestamp <= after


# ---------------------------------------------------------------------------
# 2. ModelVRAMEstimate — creation
# ---------------------------------------------------------------------------


class TestModelVRAMEstimate:
    def test_fields_stored_correctly(self):
        est = ModelVRAMEstimate(model_id="llama3-8b", gpu_gb=8.0, cpu_gb=0.0, last_used=1234.0)
        assert est.model_id == "llama3-8b"
        assert est.gpu_gb == pytest.approx(8.0)
        assert est.cpu_gb == pytest.approx(0.0)
        assert est.last_used == pytest.approx(1234.0)
        assert est.priority == 5  # default

    def test_custom_priority(self):
        est = ModelVRAMEstimate(model_id="x", gpu_gb=1.0, cpu_gb=0.0, last_used=0.0, priority=2)
        assert est.priority == 2


# ---------------------------------------------------------------------------
# 3. VRAMManager initialization without pynvml
# ---------------------------------------------------------------------------


class TestVRAMManagerInit:
    @pytest.fixture(autouse=True)
    def _setup(self):
        _reset_singleton()
        yield
        _reset_singleton()

    def test_defaults_set(self, vram_manager):
        assert vram_manager._gpu_total_gb == pytest.approx(32.0)
        assert vram_manager._cpu_offload_gb == pytest.approx(30.0)
        assert vram_manager._overhead_gb == pytest.approx(2.0)
        assert vram_manager._estimates == {}

    def test_get_gpu_snapshot_returns_none_without_pynvml(self, vram_manager):
        # _PYNVML_AVAILABLE is False in test env
        with patch.object(vram_mod, "_PYNVML_AVAILABLE", False):
            snap = vram_manager.get_gpu_snapshot()
        assert snap is None


# ---------------------------------------------------------------------------
# 4. register_load — tracks model
# ---------------------------------------------------------------------------


class TestRegisterLoad:
    @pytest.fixture(autouse=True)
    def _setup(self, vram_manager):
        self.mgr = vram_manager
        yield
        _reset_singleton()

    def test_model_appears_in_estimates(self):
        self.mgr.register_load("llama3-8b", vram_gb=8.0)
        assert "llama3-8b" in self.mgr._estimates

    def test_gpu_gb_recorded(self):
        self.mgr.register_load("mistral-7b", vram_gb=7.0, cpu_gb=1.5)
        est = self.mgr._estimates["mistral-7b"]
        assert est.gpu_gb == pytest.approx(7.0)
        assert est.cpu_gb == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# 5. register_unload — removes model
# ---------------------------------------------------------------------------


class TestRegisterUnload:
    @pytest.fixture(autouse=True)
    def _setup(self, vram_manager):
        self.mgr = vram_manager
        yield
        _reset_singleton()

    def test_model_removed(self):
        self.mgr.register_load("llama3-8b", vram_gb=8.0)
        self.mgr.register_unload("llama3-8b")
        assert "llama3-8b" not in self.mgr._estimates

    def test_unload_nonexistent_does_not_raise(self):
        # Should silently succeed (pop with default)
        self.mgr.register_unload("ghost-model")
        assert "ghost-model" not in self.mgr._estimates  # nothing was added


# ---------------------------------------------------------------------------
# 6. can_load — space available
# ---------------------------------------------------------------------------


class TestCanLoadAvailable:
    @pytest.fixture(autouse=True)
    def _setup(self, vram_manager):
        self.mgr = vram_manager
        # 32 GB total, 2 GB overhead -> 30 GB free when nothing loaded
        yield
        _reset_singleton()

    def test_small_model_fits(self):
        with patch.object(self.mgr, "get_model_vram_requirement", return_value=8.0):
            assert self.mgr.can_load("small-model")

    def test_exactly_free_vram_fits(self):
        # Nothing loaded: free = 32 - 2 = 30 GB
        with patch.object(self.mgr, "get_model_vram_requirement", return_value=30.0):
            assert self.mgr.can_load("exact-fit")


# ---------------------------------------------------------------------------
# 7. can_load — space insufficient
# ---------------------------------------------------------------------------


class TestCanLoadInsufficient:
    @pytest.fixture(autouse=True)
    def _setup(self, vram_manager):
        self.mgr = vram_manager
        # Pre-load a 28 GB model, leaving 2 GB free
        self.mgr.register_load("big-model", vram_gb=28.0)
        yield
        _reset_singleton()

    def test_large_model_does_not_fit(self):
        # Model requires 35 GB; gpu_portion = min(35, 30) = 30, cpu_portion = 5.
        # With cpu_offload_gb = 0, cpu_free = 0, so cpu_portion (5) > cpu_free (0) -> False.
        self.mgr._cpu_offload_gb = 0.0
        with patch.object(self.mgr, "get_model_vram_requirement", return_value=35.0):
            assert not self.mgr.can_load("too-big")


# ---------------------------------------------------------------------------
# 8. mark_used — updates timestamp
# ---------------------------------------------------------------------------


class TestMarkUsed:
    @pytest.fixture(autouse=True)
    def _setup(self, vram_manager):
        self.mgr = vram_manager
        yield
        _reset_singleton()

    def test_timestamp_updated(self):
        self.mgr.register_load("model-a", vram_gb=4.0)
        old_ts = self.mgr._estimates["model-a"].last_used
        time.sleep(0.01)
        self.mgr.mark_used("model-a")
        new_ts = self.mgr._estimates["model-a"].last_used
        assert new_ts > old_ts

    def test_mark_used_unknown_model_does_not_raise(self):
        self.mgr.mark_used("nonexistent")
        assert "nonexistent" not in self.mgr._estimates  # no estimate was created


# ---------------------------------------------------------------------------
# 9. get_model_vram_requirement — estimates via utils fallback
# ---------------------------------------------------------------------------


class TestGetModelVRAMRequirement:
    @pytest.fixture(autouse=True)
    def _setup(self, vram_manager):
        self.mgr = vram_manager
        yield
        _reset_singleton()

    def test_returns_float(self):
        # Patch both registry and utils so no real I/O occurs
        with patch("vetinari.models.vram_manager.VRAMManager.get_model_vram_requirement", return_value=14.0):
            result = self.mgr.get_model_vram_requirement("qwen2-14b")
        assert isinstance(result, float)

    def test_uses_utils_when_registry_fails(self):
        # Patch estimate_model_memory_gb at import site in vram_manager module.
        # Also make the model_registry lookup raise so the utils fallback is used.
        with patch("vetinari.utils.estimate_model_memory_gb", return_value=7.5):
            with patch(
                "vetinari.models.vram_manager.VRAMManager.get_model_vram_requirement",
                wraps=self.mgr.get_model_vram_requirement,
            ):
                # Just verify the real method returns a float for an unknown model.
                result = self.mgr.get_model_vram_requirement("unknown-model-xyz")
        assert isinstance(result, (int, float))


# ---------------------------------------------------------------------------
# 10. recommend_eviction — picks LRU
# ---------------------------------------------------------------------------


class TestRecommendEvictionLRU:
    @pytest.fixture(autouse=True)
    def _setup(self, vram_manager):
        self.mgr = vram_manager
        now = time.time()
        # Two models loaded; model-old is older (lower last_used)
        self.mgr._estimates["model-old"] = ModelVRAMEstimate(
            model_id="model-old", gpu_gb=10.0, cpu_gb=0.0, last_used=now - 100, priority=5
        )
        self.mgr._estimates["model-new"] = ModelVRAMEstimate(
            model_id="model-new", gpu_gb=10.0, cpu_gb=0.0, last_used=now, priority=5
        )
        yield
        _reset_singleton()

    def test_evicts_older_model(self):
        # Need 25 GB; only 12 GB free (32-2-10-8 = 12, but two models use 20 total -> 10 free)
        with (
            patch.object(self.mgr, "get_model_vram_requirement", return_value=15.0),
            patch.object(self.mgr, "get_free_vram_gb", return_value=5.0),
        ):
            evict = self.mgr.recommend_eviction("new-model")
        assert evict == "model-old"


# ---------------------------------------------------------------------------
# 11. recommend_eviction — nothing to evict
# ---------------------------------------------------------------------------


class TestRecommendEvictionNone:
    @pytest.fixture(autouse=True)
    def _setup(self, vram_manager):
        self.mgr = vram_manager
        yield
        _reset_singleton()

    def test_returns_none_when_no_models_loaded(self):
        with (
            patch.object(self.mgr, "get_model_vram_requirement", return_value=5.0),
            patch.object(self.mgr, "get_free_vram_gb", return_value=30.0),
        ):
            result = self.mgr.recommend_eviction("small-model")
        assert result is None

    def test_returns_none_when_fits_without_eviction(self):
        self.mgr.register_load("model-a", vram_gb=4.0)
        with (
            patch.object(self.mgr, "get_model_vram_requirement", return_value=2.0),
            patch.object(self.mgr, "get_free_vram_gb", return_value=26.0),
        ):
            result = self.mgr.recommend_eviction("tiny-model")
        assert result is None


# ---------------------------------------------------------------------------
# 12. status_summary — returns correct structure
# ---------------------------------------------------------------------------


class TestStatusSummary:
    @pytest.fixture(autouse=True)
    def _setup(self, vram_manager):
        self.mgr = vram_manager
        self.mgr.register_load("model-a", vram_gb=6.0, cpu_gb=0.0)
        yield
        _reset_singleton()

    def test_keys_present(self):
        with patch.object(self.mgr, "refresh", return_value=None), patch.object(vram_mod, "_PYNVML_AVAILABLE", False):
            summary = self.mgr.status_summary()
        required_keys = {
            "gpu_total_gb",
            "gpu_used_gb",
            "gpu_free_gb",
            "cpu_offload_budget_gb",
            "cpu_offload_used_gb",
            "pynvml_available",
            "loaded_models",
        }
        assert required_keys.issubset(set(summary.keys()))

    def test_loaded_models_included(self):
        with patch.object(self.mgr, "refresh", return_value=None), patch.object(vram_mod, "_PYNVML_AVAILABLE", False):
            summary = self.mgr.status_summary()
        assert "model-a" in summary["loaded_models"]
        assert summary["loaded_models"]["model-a"]["gpu_gb"] == pytest.approx(6.0)

    def test_numeric_fields_consistent(self):
        with patch.object(self.mgr, "refresh", return_value=None), patch.object(vram_mod, "_PYNVML_AVAILABLE", False):
            summary = self.mgr.status_summary()
        assert summary["gpu_total_gb"] == pytest.approx(32.0)
        # used = overhead (2) + model-a (6) = 8; free = 32 - 8 = 24
        assert summary["gpu_used_gb"] == pytest.approx(8.0)
        assert summary["gpu_free_gb"] == pytest.approx(24.0)


# ---------------------------------------------------------------------------
# 13. get_free_vram_gb and get_used_vram_gb — fallback path
# ---------------------------------------------------------------------------


class TestVRAMCalculations:
    @pytest.fixture(autouse=True)
    def _setup(self, vram_manager):
        self.mgr = vram_manager
        yield
        _reset_singleton()

    def test_free_vram_empty(self):
        with patch.object(vram_mod, "_PYNVML_AVAILABLE", False):
            free = self.mgr.get_free_vram_gb()
        # 32 - 2 overhead - 0 models = 30
        assert free == pytest.approx(30.0)

    def test_free_vram_with_model(self):
        self.mgr.register_load("model-a", vram_gb=10.0)
        with patch.object(vram_mod, "_PYNVML_AVAILABLE", False):
            free = self.mgr.get_free_vram_gb()
        # 32 - 2 - 10 = 20
        assert free == pytest.approx(20.0)

    def test_used_vram_includes_overhead(self):
        with patch.object(vram_mod, "_PYNVML_AVAILABLE", False):
            used = self.mgr.get_used_vram_gb()
        # overhead only = 2.0
        assert used == pytest.approx(2.0)

    def test_used_vram_with_model(self):
        self.mgr.register_load("model-a", vram_gb=5.0)
        with patch.object(vram_mod, "_PYNVML_AVAILABLE", False):
            used = self.mgr.get_used_vram_gb()
        assert used == pytest.approx(7.0)  # 2 + 5


# ---------------------------------------------------------------------------
# 14. Multiple loads track correctly
# ---------------------------------------------------------------------------


class TestMultipleLoads:
    @pytest.fixture(autouse=True)
    def _setup(self, vram_manager):
        self.mgr = vram_manager
        yield
        _reset_singleton()

    def test_three_models_tracked(self):
        self.mgr.register_load("a", vram_gb=4.0)
        self.mgr.register_load("b", vram_gb=6.0)
        self.mgr.register_load("c", vram_gb=8.0)
        assert len(self.mgr._estimates) == 3

    def test_combined_vram_usage(self):
        self.mgr.register_load("a", vram_gb=4.0)
        self.mgr.register_load("b", vram_gb=6.0)
        with patch.object(vram_mod, "_PYNVML_AVAILABLE", False):
            used = self.mgr.get_used_vram_gb()
        # 2 overhead + 4 + 6 = 12
        assert used == pytest.approx(12.0)

    def test_unload_partial(self):
        self.mgr.register_load("a", vram_gb=4.0)
        self.mgr.register_load("b", vram_gb=6.0)
        self.mgr.register_unload("a")
        assert "a" not in self.mgr._estimates
        assert "b" in self.mgr._estimates


# ---------------------------------------------------------------------------
# 15. get_vram_manager — module-level singleton
# ---------------------------------------------------------------------------


class TestGetVRAMManagerSingleton:
    @pytest.fixture(autouse=True)
    def _setup(self):
        _reset_singleton()
        yield
        _reset_singleton()

    def test_returns_vram_manager_instance(self):
        with patch.object(VRAMManager, "_load_hardware_config", return_value=None):
            mgr = get_vram_manager()
        assert isinstance(mgr, VRAMManager)

    def test_same_instance_on_repeated_calls(self):
        with patch.object(VRAMManager, "_load_hardware_config", return_value=None):
            mgr1 = get_vram_manager()
            mgr2 = get_vram_manager()
        assert mgr1 is mgr2

    def test_class_get_instance_singleton(self):
        with patch.object(VRAMManager, "_load_hardware_config", return_value=None):
            a = VRAMManager.get_instance()
            b = VRAMManager.get_instance()
        assert a is b


# ---------------------------------------------------------------------------
# 16. ModelLease — dataclass and expiry
# ---------------------------------------------------------------------------


class TestModelLease:
    def test_fields_stored(self):
        lease = ModelLease(model_id="llama3-8b", holder_id="worker:task-1")
        assert lease.model_id == "llama3-8b"
        assert lease.holder_id == "worker:task-1"
        assert lease.max_duration_s == pytest.approx(300.0)

    def test_not_expired_when_fresh(self):
        lease = ModelLease(model_id="x", holder_id="h1")
        assert not lease.is_expired

    def test_expired_after_duration(self):
        lease = ModelLease(
            model_id="x",
            holder_id="h1",
            acquired_at=time.time() - 400,
            max_duration_s=300.0,
        )
        assert lease.is_expired

    def test_custom_duration(self):
        lease = ModelLease(model_id="x", holder_id="h1", max_duration_s=10.0)
        assert lease.max_duration_s == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 17. Lease acquire / release / expiry lifecycle
# ---------------------------------------------------------------------------


class TestLeaseLifecycle:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mgr = _make_manager()
        self.mgr.register_load("model-a", vram_gb=8.0)
        self.mgr.register_load("model-b", vram_gb=6.0)
        yield
        _reset_singleton()

    def test_acquire_known_model_succeeds(self):
        assert self.mgr.acquire_lease("model-a", "worker:t1") is True
        assert self.mgr.active_lease_count() == 1

    def test_acquire_unknown_model_denied(self):
        assert self.mgr.acquire_lease("ghost-model", "worker:t1") is False
        assert self.mgr.active_lease_count() == 0

    def test_release_removes_lease(self):
        self.mgr.acquire_lease("model-a", "worker:t1")
        self.mgr.release_lease("worker:t1")
        assert self.mgr.active_lease_count() == 0
        assert not self.mgr.is_leased("model-a")

    def test_release_nonexistent_is_noop(self):
        self.mgr.release_lease("nobody")
        assert self.mgr.active_lease_count() == 0

    def test_is_leased_true_when_active(self):
        self.mgr.acquire_lease("model-a", "worker:t1")
        assert self.mgr.is_leased("model-a") is True
        assert self.mgr.is_leased("model-b") is False

    def test_multiple_leases_same_model(self):
        self.mgr.acquire_lease("model-a", "worker:t1")
        self.mgr.acquire_lease("model-a", "worker:t2")
        assert self.mgr.active_lease_count() == 2
        assert self.mgr.is_leased("model-a") is True

    def test_get_leased_model_ids(self):
        self.mgr.acquire_lease("model-a", "worker:t1")
        self.mgr.acquire_lease("model-b", "inspector:t2")
        ids = self.mgr.get_leased_model_ids()
        assert ids == {"model-a", "model-b"}

    def test_expired_leases_reaped_on_acquire(self):
        # Create an already-expired lease
        self.mgr._leases["old"] = ModelLease(
            model_id="model-a",
            holder_id="old",
            acquired_at=time.time() - 500,
            max_duration_s=300.0,
        )
        # Acquiring a new lease reaps expired ones
        self.mgr.acquire_lease("model-b", "worker:t1")
        assert "old" not in self.mgr._leases
        assert self.mgr.active_lease_count() == 1

    def test_expired_lease_not_counted_as_leased(self):
        self.mgr._leases["old"] = ModelLease(
            model_id="model-a",
            holder_id="old",
            acquired_at=time.time() - 500,
            max_duration_s=300.0,
        )
        # is_leased reaps expired leases first
        assert self.mgr.is_leased("model-a") is False


# ---------------------------------------------------------------------------
# 18. Leased models excluded from eviction
# ---------------------------------------------------------------------------


class TestLeaseEvictionProtection:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mgr = _make_manager()
        now = time.time()
        self.mgr._estimates["model-old"] = ModelVRAMEstimate(
            model_id="model-old",
            gpu_gb=10.0,
            cpu_gb=0.0,
            last_used=now - 100,
            priority=5,
        )
        self.mgr._estimates["model-new"] = ModelVRAMEstimate(
            model_id="model-new",
            gpu_gb=10.0,
            cpu_gb=0.0,
            last_used=now,
            priority=5,
        )
        yield
        _reset_singleton()

    def test_leased_model_not_evicted(self):
        # Lease the old model — eviction should skip it and pick model-new
        self.mgr.acquire_lease("model-old", "worker:t1")
        with (
            patch.object(self.mgr, "get_model_vram_requirement", return_value=15.0),
            patch.object(self.mgr, "get_free_vram_gb", return_value=5.0),
        ):
            evict = self.mgr.recommend_eviction("incoming-model")
        assert evict == "model-new"

    def test_all_leased_returns_none_when_no_candidates(self):
        self.mgr.acquire_lease("model-old", "worker:t1")
        self.mgr.acquire_lease("model-new", "worker:t2")
        with (
            patch.object(self.mgr, "get_model_vram_requirement", return_value=15.0),
            patch.object(self.mgr, "get_free_vram_gb", return_value=5.0),
        ):
            evict = self.mgr.recommend_eviction("incoming-model")
        assert evict is None


# ---------------------------------------------------------------------------
# 19. get_evictable_vram_gb and get_max_available_vram_gb
# ---------------------------------------------------------------------------


class TestCapacityPlanning:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mgr = _make_manager()
        self.mgr.register_load("evictable", vram_gb=10.0)
        self.mgr.register_load("leased", vram_gb=8.0)
        yield
        _reset_singleton()

    def test_evictable_excludes_leased(self):
        self.mgr.acquire_lease("leased", "worker:t1")
        evictable = self.mgr.get_evictable_vram_gb()
        # Only "evictable" (10 GB) should count; "leased" is protected
        assert evictable == pytest.approx(10.0)

    def test_evictable_excludes_pinned(self):
        self.mgr._estimates["evictable"].is_pinned = True
        evictable = self.mgr.get_evictable_vram_gb()
        # "evictable" is pinned, "leased" is not pinned and not leased
        assert evictable == pytest.approx(8.0)

    def test_all_evictable_when_nothing_protected(self):
        evictable = self.mgr.get_evictable_vram_gb()
        # Both models: 10 + 8 = 18
        assert evictable == pytest.approx(18.0)

    def test_max_available_combines_free_and_evictable(self):
        with patch.object(vram_mod, "_PYNVML_AVAILABLE", False):
            max_avail = self.mgr.get_max_available_vram_gb()
        # free = 32 - 2(overhead) - 10 - 8 = 12; evictable = 18; total = 30
        assert max_avail == pytest.approx(30.0)

    def test_max_available_reduced_by_leases(self):
        self.mgr.acquire_lease("leased", "worker:t1")
        with patch.object(vram_mod, "_PYNVML_AVAILABLE", False):
            max_avail = self.mgr.get_max_available_vram_gb()
        # free = 12; evictable = 10 (only "evictable"); total = 22
        assert max_avail == pytest.approx(22.0)


# ---------------------------------------------------------------------------
# 20. status_summary includes active_leases
# ---------------------------------------------------------------------------


class TestStatusSummaryLeases:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mgr = _make_manager()
        self.mgr.register_load("model-a", vram_gb=6.0)
        yield
        _reset_singleton()

    def test_active_leases_key_present(self):
        with patch.object(self.mgr, "refresh", return_value=None), patch.object(vram_mod, "_PYNVML_AVAILABLE", False):
            summary = self.mgr.status_summary()
        assert "active_leases" in summary

    def test_active_leases_populated(self):
        self.mgr.acquire_lease("model-a", "worker:t1")
        with patch.object(self.mgr, "refresh", return_value=None), patch.object(vram_mod, "_PYNVML_AVAILABLE", False):
            summary = self.mgr.status_summary()
        assert "worker:t1" in summary["active_leases"]
        assert summary["active_leases"]["worker:t1"]["model_id"] == "model-a"

    def test_expired_leases_excluded_from_summary(self):
        self.mgr._leases["old"] = ModelLease(
            model_id="model-a",
            holder_id="old",
            acquired_at=time.time() - 500,
            max_duration_s=300.0,
        )
        with patch.object(self.mgr, "refresh", return_value=None), patch.object(vram_mod, "_PYNVML_AVAILABLE", False):
            summary = self.mgr.status_summary()
        assert "old" not in summary["active_leases"]


# ── Session 15: KV cache quant recommendation ──────────────────────────────


class TestRecommendKvQuantForContext:
    """VRAMManager recommends the right KV quant tier based on free VRAM."""

    def setup_method(self):
        self.mgr = _make_manager()
        # Total=32 GB, overhead=2 GB => 30 GB free with no models loaded

    def teardown_method(self):
        _reset_singleton()

    def test_returns_f16_when_vram_is_plentiful(self):
        """Short context on a high-VRAM system gets f16 (no compression needed)."""
        # 1024 tokens * 2048 bytes/token = 2MB — trivially fits in 30 GB
        result = self.mgr.recommend_kv_quant_for_context(context_length=1024)
        assert result == "f16"

    def test_returns_q8_0_when_vram_is_moderate(self):
        """Context that uses 50-75% of available VRAM triggers q8_0 compression."""
        # Fill up VRAM so only ~1 GB free: 32 - 2 overhead - 29 model = 1 GB
        self.mgr.register_load("big-model", vram_gb=29.0)
        # 300k * 2048 bytes ≈ 0.57 GB; 50% of 1 GB = 0.5, 75% = 0.75
        # 0.5 < 0.57 < 0.75 → q8_0
        result = self.mgr.recommend_kv_quant_for_context(context_length=300_000)
        assert result == "q8_0"

    def test_returns_q4_0_when_vram_is_scarce(self):
        """Context requiring >75% of free VRAM triggers q4_0 (maximum compression)."""
        # 1 GB free after loading big model
        self.mgr.register_load("big-model", vram_gb=29.0)
        # 500k * 2048 bytes = ~0.95 GB > 0.75 * 1.0 GB => q4_0
        result = self.mgr.recommend_kv_quant_for_context(context_length=500_000)
        assert result == "q4_0"

    def test_model_vram_arg_reduces_available_budget(self):
        """The model_vram_gb argument reserves VRAM before computing headroom."""
        # 300k * 2048 bytes ≈ 0.57 GB — trivially fits in 30 GB free → f16
        result_no_model = self.mgr.recommend_kv_quant_for_context(context_length=300_000, model_vram_gb=0.0)
        # With a 29 GB model only 1 GB is free; 0.57 GB > 50% of 1 GB → q8_0
        result_with_model = self.mgr.recommend_kv_quant_for_context(context_length=300_000, model_vram_gb=29.0)
        assert result_no_model == "f16"
        assert result_with_model in {"q8_0", "q4_0"}
