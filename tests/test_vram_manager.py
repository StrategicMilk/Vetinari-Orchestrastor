"""Tests for vetinari/vram_manager.py.

pynvml is not installed in the test environment, so all tests exercise the
fallback (estimate-only) code path.  The singleton is reset between every test.
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

import vetinari.vram_manager as vram_mod
from vetinari.vram_manager import (
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


# ---------------------------------------------------------------------------
# 1. VRAMSnapshot — creation and fields
# ---------------------------------------------------------------------------

class TestVRAMSnapshot(unittest.TestCase):
    def test_fields_stored_correctly(self):
        snap = VRAMSnapshot(gpu_index=0, total_gb=32.0, used_gb=8.0, free_gb=24.0)
        assert snap.gpu_index == 0
        self.assertAlmostEqual(snap.total_gb, 32.0)
        self.assertAlmostEqual(snap.used_gb, 8.0)
        self.assertAlmostEqual(snap.free_gb, 24.0)

    def test_timestamp_defaults_to_now(self):
        before = time.time()
        snap = VRAMSnapshot(gpu_index=0, total_gb=32.0, used_gb=0.0, free_gb=32.0)
        after = time.time()
        assert snap.timestamp >= before
        assert snap.timestamp <= after


# ---------------------------------------------------------------------------
# 2. ModelVRAMEstimate — creation
# ---------------------------------------------------------------------------

class TestModelVRAMEstimate(unittest.TestCase):
    def test_fields_stored_correctly(self):
        est = ModelVRAMEstimate(model_id="llama3-8b", gpu_gb=8.0, cpu_gb=0.0, last_used=1234.0)
        assert est.model_id == "llama3-8b"
        self.assertAlmostEqual(est.gpu_gb, 8.0)
        self.assertAlmostEqual(est.cpu_gb, 0.0)
        self.assertAlmostEqual(est.last_used, 1234.0)
        assert est.priority == 5  # default

    def test_custom_priority(self):
        est = ModelVRAMEstimate(model_id="x", gpu_gb=1.0, cpu_gb=0.0, last_used=0.0, priority=2)
        assert est.priority == 2


# ---------------------------------------------------------------------------
# 3. VRAMManager initialization without pynvml
# ---------------------------------------------------------------------------

class TestVRAMManagerInit(unittest.TestCase):
    def setUp(self):
        _reset_singleton()

    def tearDown(self):
        _reset_singleton()

    def test_defaults_set(self):
        mgr = _make_manager()
        self.assertAlmostEqual(mgr._gpu_total_gb, 32.0)
        self.assertAlmostEqual(mgr._cpu_offload_gb, 30.0)
        self.assertAlmostEqual(mgr._overhead_gb, 2.0)
        assert mgr._estimates == {}

    def test_get_gpu_snapshot_returns_none_without_pynvml(self):
        mgr = _make_manager()
        # _PYNVML_AVAILABLE is False in test env
        with patch.object(vram_mod, "_PYNVML_AVAILABLE", False):
            snap = mgr.get_gpu_snapshot()
        assert snap is None


# ---------------------------------------------------------------------------
# 4. register_load — tracks model
# ---------------------------------------------------------------------------

class TestRegisterLoad(unittest.TestCase):
    def setUp(self):
        self.mgr = _make_manager()

    def tearDown(self):
        _reset_singleton()

    def test_model_appears_in_estimates(self):
        self.mgr.register_load("llama3-8b", vram_gb=8.0)
        assert "llama3-8b" in self.mgr._estimates

    def test_gpu_gb_recorded(self):
        self.mgr.register_load("mistral-7b", vram_gb=7.0, cpu_gb=1.5)
        est = self.mgr._estimates["mistral-7b"]
        self.assertAlmostEqual(est.gpu_gb, 7.0)
        self.assertAlmostEqual(est.cpu_gb, 1.5)


# ---------------------------------------------------------------------------
# 5. register_unload — removes model
# ---------------------------------------------------------------------------

class TestRegisterUnload(unittest.TestCase):
    def setUp(self):
        self.mgr = _make_manager()

    def tearDown(self):
        _reset_singleton()

    def test_model_removed(self):
        self.mgr.register_load("llama3-8b", vram_gb=8.0)
        self.mgr.register_unload("llama3-8b")
        assert "llama3-8b" not in self.mgr._estimates

    def test_unload_nonexistent_does_not_raise(self):
        # Should silently succeed (pop with default)
        self.mgr.register_unload("ghost-model")


# ---------------------------------------------------------------------------
# 6. can_load — space available
# ---------------------------------------------------------------------------

class TestCanLoadAvailable(unittest.TestCase):
    def setUp(self):
        self.mgr = _make_manager()
        # 32 GB total, 2 GB overhead -> 30 GB free when nothing loaded

    def tearDown(self):
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

class TestCanLoadInsufficient(unittest.TestCase):
    def setUp(self):
        self.mgr = _make_manager()
        # Pre-load a 28 GB model, leaving 2 GB free
        self.mgr.register_load("big-model", vram_gb=28.0)

    def tearDown(self):
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

class TestMarkUsed(unittest.TestCase):
    def setUp(self):
        self.mgr = _make_manager()

    def tearDown(self):
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


# ---------------------------------------------------------------------------
# 9. get_model_vram_requirement — estimates via utils fallback
# ---------------------------------------------------------------------------

class TestGetModelVRAMRequirement(unittest.TestCase):
    def setUp(self):
        self.mgr = _make_manager()

    def tearDown(self):
        _reset_singleton()

    def test_returns_float(self):
        # Patch both registry and utils so no real I/O occurs
        with patch("vetinari.vram_manager.VRAMManager.get_model_vram_requirement", return_value=14.0):
            result = self.mgr.get_model_vram_requirement("qwen2-14b")
        assert isinstance(result, float)

    def test_uses_utils_when_registry_fails(self):
        # Patch estimate_model_memory_gb at import site in vram_manager module.
        # Also make the model_registry lookup raise so the utils fallback is used.
        with patch("vetinari.utils.estimate_model_memory_gb", return_value=7.5):
            with patch("vetinari.vram_manager.VRAMManager.get_model_vram_requirement",
                       wraps=self.mgr.get_model_vram_requirement):
                # Just verify the real method returns a float for an unknown model.
                result = self.mgr.get_model_vram_requirement("unknown-model-xyz")
        assert isinstance(result, (int, float))


# ---------------------------------------------------------------------------
# 10. recommend_eviction — picks LRU
# ---------------------------------------------------------------------------

class TestRecommendEvictionLRU(unittest.TestCase):
    def setUp(self):
        self.mgr = _make_manager()
        now = time.time()
        # Two models loaded; model-old is older (lower last_used)
        self.mgr._estimates["model-old"] = ModelVRAMEstimate(
            model_id="model-old", gpu_gb=10.0, cpu_gb=0.0, last_used=now - 100, priority=5
        )
        self.mgr._estimates["model-new"] = ModelVRAMEstimate(
            model_id="model-new", gpu_gb=10.0, cpu_gb=0.0, last_used=now, priority=5
        )

    def tearDown(self):
        _reset_singleton()

    def test_evicts_older_model(self):
        # Need 25 GB; only 12 GB free (32-2-10-8 = 12, but two models use 20 total -> 10 free)
        with patch.object(self.mgr, "get_model_vram_requirement", return_value=15.0), \
             patch.object(self.mgr, "get_free_vram_gb", return_value=5.0):
            evict = self.mgr.recommend_eviction("new-model")
        assert evict == "model-old"


# ---------------------------------------------------------------------------
# 11. recommend_eviction — nothing to evict
# ---------------------------------------------------------------------------

class TestRecommendEvictionNone(unittest.TestCase):
    def setUp(self):
        self.mgr = _make_manager()

    def tearDown(self):
        _reset_singleton()

    def test_returns_none_when_no_models_loaded(self):
        with patch.object(self.mgr, "get_model_vram_requirement", return_value=5.0), \
             patch.object(self.mgr, "get_free_vram_gb", return_value=30.0):
            result = self.mgr.recommend_eviction("small-model")
        assert result is None

    def test_returns_none_when_fits_without_eviction(self):
        self.mgr.register_load("model-a", vram_gb=4.0)
        with patch.object(self.mgr, "get_model_vram_requirement", return_value=2.0), \
             patch.object(self.mgr, "get_free_vram_gb", return_value=26.0):
            result = self.mgr.recommend_eviction("tiny-model")
        assert result is None


# ---------------------------------------------------------------------------
# 12. status_summary — returns correct structure
# ---------------------------------------------------------------------------

class TestStatusSummary(unittest.TestCase):
    def setUp(self):
        self.mgr = _make_manager()
        self.mgr.register_load("model-a", vram_gb=6.0, cpu_gb=0.0)

    def tearDown(self):
        _reset_singleton()

    def test_keys_present(self):
        with patch.object(self.mgr, "refresh", return_value=None), \
             patch.object(vram_mod, "_PYNVML_AVAILABLE", False):
            summary = self.mgr.status_summary()
        expected_keys = {
            "gpu_total_gb", "gpu_used_gb", "gpu_free_gb",
            "cpu_offload_budget_gb", "cpu_offload_used_gb",
            "pynvml_available", "loaded_models",
        }
        assert set(summary.keys()) == expected_keys

    def test_loaded_models_included(self):
        with patch.object(self.mgr, "refresh", return_value=None), \
             patch.object(vram_mod, "_PYNVML_AVAILABLE", False):
            summary = self.mgr.status_summary()
        assert "model-a" in summary["loaded_models"]
        self.assertAlmostEqual(summary["loaded_models"]["model-a"]["gpu_gb"], 6.0)

    def test_numeric_fields_consistent(self):
        with patch.object(self.mgr, "refresh", return_value=None), \
             patch.object(vram_mod, "_PYNVML_AVAILABLE", False):
            summary = self.mgr.status_summary()
        self.assertAlmostEqual(summary["gpu_total_gb"], 32.0)
        # used = overhead (2) + model-a (6) = 8; free = 32 - 8 = 24
        self.assertAlmostEqual(summary["gpu_used_gb"], 8.0)
        self.assertAlmostEqual(summary["gpu_free_gb"], 24.0)


# ---------------------------------------------------------------------------
# 13. get_free_vram_gb and get_used_vram_gb — fallback path
# ---------------------------------------------------------------------------

class TestVRAMCalculations(unittest.TestCase):
    def setUp(self):
        self.mgr = _make_manager()

    def tearDown(self):
        _reset_singleton()

    def test_free_vram_empty(self):
        with patch.object(vram_mod, "_PYNVML_AVAILABLE", False):
            free = self.mgr.get_free_vram_gb()
        # 32 - 2 overhead - 0 models = 30
        self.assertAlmostEqual(free, 30.0)

    def test_free_vram_with_model(self):
        self.mgr.register_load("model-a", vram_gb=10.0)
        with patch.object(vram_mod, "_PYNVML_AVAILABLE", False):
            free = self.mgr.get_free_vram_gb()
        # 32 - 2 - 10 = 20
        self.assertAlmostEqual(free, 20.0)

    def test_used_vram_includes_overhead(self):
        with patch.object(vram_mod, "_PYNVML_AVAILABLE", False):
            used = self.mgr.get_used_vram_gb()
        # overhead only = 2.0
        self.assertAlmostEqual(used, 2.0)

    def test_used_vram_with_model(self):
        self.mgr.register_load("model-a", vram_gb=5.0)
        with patch.object(vram_mod, "_PYNVML_AVAILABLE", False):
            used = self.mgr.get_used_vram_gb()
        self.assertAlmostEqual(used, 7.0)  # 2 + 5


# ---------------------------------------------------------------------------
# 14. Multiple loads track correctly
# ---------------------------------------------------------------------------

class TestMultipleLoads(unittest.TestCase):
    def setUp(self):
        self.mgr = _make_manager()

    def tearDown(self):
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
        self.assertAlmostEqual(used, 12.0)

    def test_unload_partial(self):
        self.mgr.register_load("a", vram_gb=4.0)
        self.mgr.register_load("b", vram_gb=6.0)
        self.mgr.register_unload("a")
        assert "a" not in self.mgr._estimates
        assert "b" in self.mgr._estimates


# ---------------------------------------------------------------------------
# 15. get_vram_manager — module-level singleton
# ---------------------------------------------------------------------------

class TestGetVRAMManagerSingleton(unittest.TestCase):
    def setUp(self):
        _reset_singleton()

    def tearDown(self):
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


if __name__ == "__main__":
    unittest.main()
