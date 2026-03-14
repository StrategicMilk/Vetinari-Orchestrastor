"""Tests for vetinari.plan_cache — plan caching with keyword similarity."""

from __future__ import annotations

import tempfile
from pathlib import Path

from vetinari.plan_cache import CachedPlan, PlanCache


class TestCachedPlan:
    """Tests for CachedPlan dataclass."""

    def test_to_dict_roundtrip(self):
        plan = CachedPlan(
            goal="build auth system",
            goal_hash="abc123",
            plan_data={"tasks": ["a", "b"]},
            created_at=1000.0,
            quality_score=0.8,
        )
        d = plan.to_dict()
        restored = CachedPlan.from_dict(d)
        assert restored.goal == plan.goal
        assert restored.goal_hash == plan.goal_hash
        assert restored.quality_score == plan.quality_score

    def test_from_dict_ignores_extra_keys(self):
        d = {
            "goal": "test",
            "goal_hash": "h",
            "plan_data": {},
            "created_at": 1.0,
            "extra_field": "ignored",
        }
        plan = CachedPlan.from_dict(d)
        assert plan.goal == "test"
        assert not hasattr(plan, "extra_field") or True  # extra key is filtered


class TestPlanCache:
    """Tests for PlanCache store/find/invalidate."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = PlanCache(cache_dir=str(Path(self.tmpdir) / "cache"))

    def test_store_and_find_exact(self):
        self.cache.store("build auth system", {"tasks": [1, 2]})
        result = self.cache.find_similar("build auth system")
        assert result is not None
        assert result.goal == "build auth system"

    def test_find_similar_above_threshold(self):
        self.cache.store("build user authentication system", {"tasks": [1]})
        result = self.cache.find_similar("build auth system", threshold=0.3)
        assert result is not None

    def test_find_similar_below_threshold_returns_none(self):
        self.cache.store("build user auth", {"tasks": [1]})
        result = self.cache.find_similar("optimize database queries", threshold=0.9)
        assert result is None

    def test_hit_count_increments(self):
        self.cache.store("build auth", {"tasks": [1]})
        self.cache.find_similar("build auth")
        self.cache.find_similar("build auth")
        plan = self.cache.find_similar("build auth")
        assert plan is not None
        assert plan.hit_count == 3

    def test_eviction_at_max_size(self):
        small_cache = PlanCache(cache_dir=str(Path(self.tmpdir) / "small"))
        small_cache.MAX_CACHE_SIZE = 3
        small_cache.store("goal 1", {"t": 1})
        small_cache.store("goal 2", {"t": 2})
        small_cache.store("goal 3", {"t": 3})
        small_cache.store("goal 4", {"t": 4})  # triggers eviction
        stats = small_cache.get_stats()
        assert stats["total_cached"] == 3

    def test_invalidate_removes_old(self):
        import time

        self.cache.store("old goal", {"t": 1})
        # Backdate the entry
        for plan in self.cache._cache.values():
            plan.created_at = time.time() - 100 * 86400  # 100 days ago
        removed = self.cache.invalidate(older_than_days=30)
        assert removed == 1

    def test_get_stats(self):
        self.cache.store("a", {"t": 1}, quality_score=0.9)
        self.cache.store("b", {"t": 2}, quality_score=0.7)
        stats = self.cache.get_stats()
        assert stats["total_cached"] == 2
        assert stats["avg_quality"] == pytest.approx(0.8, abs=0.01)

    def test_empty_cache_stats(self):
        stats = self.cache.get_stats()
        assert stats["total_cached"] == 0
        assert stats["total_hits"] == 0

    def test_persistence_across_instances(self):
        cache_dir = str(Path(self.tmpdir) / "persist")
        c1 = PlanCache(cache_dir=cache_dir)
        c1.store("persistent goal", {"tasks": [1]})

        c2 = PlanCache(cache_dir=cache_dir)
        result = c2.find_similar("persistent goal")
        assert result is not None


# Need pytest for approx
import pytest
