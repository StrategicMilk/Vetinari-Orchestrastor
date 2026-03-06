"""Tests for PlanCache and CachedPlan."""
import time
import pytest
from vetinari.plan_cache import PlanCache, CachedPlan


@pytest.fixture
def cache(tmp_path):
    """Return a PlanCache backed by a temp directory."""
    return PlanCache(cache_dir=str(tmp_path / "plan_cache"))


class TestStoreAndFind:
    def test_store_and_find_exact_match(self, cache):
        cache.store("build a web app", {"tasks": ["create UI", "create API"]})
        result = cache.find_similar("build a web app")
        assert result is not None
        assert result.goal == "build a web app"
        assert result.hit_count == 1

    def test_find_similar_with_keyword_overlap(self, cache):
        cache.store("build a web application with authentication", {"tasks": []})
        # High keyword overlap should match above default threshold (0.6)
        result = cache.find_similar("build a web application with authentication system", threshold=0.5)
        assert result is not None

    def test_find_similar_returns_none_below_threshold(self, cache):
        cache.store("build a rocket ship for space exploration", {"tasks": []})
        # Completely different goal
        result = cache.find_similar("analyze database performance metrics", threshold=0.6)
        assert result is None


class TestInvalidate:
    def test_invalidate_removes_old_entries(self, cache):
        # Store an entry then manually backdate it
        cache.store("old goal", {"tasks": []})
        # Force the created_at to be old
        for plan in cache._cache.values():
            plan.created_at = time.time() - (40 * 86400)  # 40 days ago
        removed = cache.invalidate(older_than_days=30)
        assert removed == 1
        assert len(cache._cache) == 0


class TestGetStats:
    def test_get_stats_returns_correct_counts(self, cache):
        cache.store("goal one", {"tasks": []}, quality_score=0.8)
        cache.store("goal two", {"tasks": []}, quality_score=0.6)
        cache.find_similar("goal one")
        stats = cache.get_stats()
        assert stats["total_cached"] == 2
        assert stats["total_hits"] == 1
        assert abs(stats["avg_quality"] - 0.7) < 0.01


class TestEviction:
    def test_cache_eviction_when_over_max_size(self, cache):
        cache.MAX_CACHE_SIZE = 3
        for i in range(5):
            cache.store(f"unique goal number {i} with extra words", {"tasks": []})
            time.sleep(0.01)  # ensure distinct last_hit ordering
        assert len(cache._cache) <= cache.MAX_CACHE_SIZE


class TestCachedPlanSerialization:
    def test_serialization_round_trip(self):
        plan = CachedPlan(
            goal="test goal",
            goal_hash="abc123",
            plan_data={"tasks": ["task1"]},
            created_at=1234567890.0,
            hit_count=3,
            last_hit=1234567900.0,
            quality_score=0.9,
        )
        d = plan.to_dict()
        restored = CachedPlan.from_dict(d)
        assert restored.goal == plan.goal
        assert restored.goal_hash == plan.goal_hash
        assert restored.plan_data == plan.plan_data
        assert restored.hit_count == plan.hit_count
        assert restored.quality_score == plan.quality_score


class TestSimilarity:
    def test_similarity_correct_jaccard_score(self, cache):
        # "apple banana cherry" vs "apple banana date"
        # intersection = {apple, banana}, union = {apple, banana, cherry, date} -> 2/4 = 0.5
        score = cache._similarity("apple banana cherry", "apple banana date")
        assert abs(score - 0.5) < 0.001

    def test_similarity_identical_goals(self, cache):
        score = cache._similarity("build a web app", "build a web app")
        assert score == 1.0

    def test_similarity_no_overlap(self, cache):
        score = cache._similarity("apple banana", "cherry date")
        assert score == 0.0
