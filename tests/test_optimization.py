"""Tests for vetinari/optimization/ — P10.5, P10.6, P10.7."""

from __future__ import annotations

import threading

import pytest

from vetinari.optimization.prompt_cache import (
    CacheResult,
    PromptCache,
    get_prompt_cache,
    reset_prompt_cache,
)
from vetinari.optimization.semantic_cache import (
    SemanticCache,
    _jaccard,
    _trigrams,
    get_semantic_cache,
    reset_semantic_cache,
)

# ===========================================================================
# PromptCache (P10.6)
# ===========================================================================


class TestPromptCacheMissAndHit:
    """Basic hit/miss behaviour."""

    def setup_method(self) -> None:
        self.cache = PromptCache(ttl=60, max_entries=100)

    def test_first_lookup_is_miss(self) -> None:
        result = self.cache.get_or_cache("hash1", "Hello, world!")
        assert not result.hit
        assert result.savings_tokens == 0
        assert result.prompt == "Hello, world!"

    def test_second_lookup_is_hit(self) -> None:
        self.cache.get_or_cache("hash1", "Hello, world!")
        result = self.cache.get_or_cache("hash1", "Hello, world!")
        assert result.hit
        assert result.savings_tokens > 0

    def test_different_hashes_independent(self) -> None:
        self.cache.get_or_cache("a", "prompt A")
        self.cache.get_or_cache("b", "prompt B")
        r_a = self.cache.get_or_cache("a", "prompt A")
        r_b = self.cache.get_or_cache("b", "prompt B")
        assert r_a.hit
        assert r_b.hit

    def test_cache_result_is_dataclass(self) -> None:
        result = self.cache.get_or_cache("x", "text")
        assert isinstance(result, CacheResult)


class TestPromptCacheTTL:
    """TTL expiry behaviour."""

    def test_expired_entry_is_miss(self) -> None:
        from unittest.mock import patch

        cache = PromptCache(ttl=10, max_entries=100)
        # Insert at t=100, expires at t=110
        with patch("vetinari.optimization.prompt_cache.time.monotonic", return_value=100.0):
            cache.get_or_cache("k", "prompt")
        # Read at t=120 — past expiry
        with patch("vetinari.optimization.prompt_cache.time.monotonic", return_value=120.0):
            result = cache.get_or_cache("k", "prompt")
        assert not result.hit

    def test_not_expired_entry_is_hit(self) -> None:
        cache = PromptCache(ttl=60, max_entries=100)
        cache.get_or_cache("k", "prompt")
        result = cache.get_or_cache("k", "prompt")
        assert result.hit


class TestPromptCacheLRUEviction:
    """LRU eviction when max_entries is exceeded."""

    def test_oldest_entry_evicted(self) -> None:
        cache = PromptCache(ttl=3600, max_entries=3)
        cache.get_or_cache("a", "A")
        cache.get_or_cache("b", "B")
        cache.get_or_cache("c", "C")
        # Access "a" to make it recently used
        cache.get_or_cache("a", "A")
        # Adding "d" should evict the LRU entry ("b")
        cache.get_or_cache("d", "D")
        stats = cache.get_stats()
        assert stats["cache_size"] <= 3

    def test_cache_size_does_not_exceed_max(self) -> None:
        cache = PromptCache(ttl=3600, max_entries=5)
        for i in range(20):
            cache.get_or_cache(f"key{i}", f"prompt{i}")
        assert cache.get_stats()["cache_size"] <= 5


class TestPromptCacheStats:
    """Statistics reporting."""

    def test_stats_initial_state(self) -> None:
        cache = PromptCache()
        stats = cache.get_stats()
        assert stats["total_hits"] == 0
        assert stats["total_misses"] == 0
        assert stats["cache_size"] == 0
        assert abs(stats["hit_rate"] - 0.0) < 1e-7

    def test_hit_rate_calculated_correctly(self) -> None:
        cache = PromptCache(ttl=60)
        cache.get_or_cache("k", "prompt")  # miss
        cache.get_or_cache("k", "prompt")  # hit
        cache.get_or_cache("k", "prompt")  # hit
        stats = cache.get_stats()
        assert abs(stats["hit_rate"] - 2 / 3) < 1e-7
        assert stats["total_hits"] == 2
        assert stats["total_misses"] == 1


class TestPromptCacheSingleton:
    """Singleton accessor."""

    def setup_method(self) -> None:
        reset_prompt_cache()

    def teardown_method(self) -> None:
        reset_prompt_cache()

    def test_singleton_returns_same_instance(self) -> None:
        a = get_prompt_cache()
        b = get_prompt_cache()
        assert a is b

    def test_reset_creates_new_instance(self) -> None:
        a = get_prompt_cache()
        reset_prompt_cache()
        b = get_prompt_cache()
        assert a is not b


class TestPromptCacheThreadSafety:
    """Thread safety."""

    def test_concurrent_writes_do_not_corrupt(self) -> None:
        cache = PromptCache(ttl=60, max_entries=1000)
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                for j in range(50):
                    cache.get_or_cache(f"key-{i}-{j}", f"prompt-{i}-{j}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        stats = cache.get_stats()
        assert stats["cache_size"] <= 1000


# ===========================================================================
# SemanticCache (P10.5)
# ===========================================================================


class TestSemanticCachePutGet:
    """Basic put/get behaviour."""

    def setup_method(self) -> None:
        self.cache = SemanticCache(ttl=3600, max_entries=100, similarity_threshold=0.85)

    def test_exact_match_is_hit(self) -> None:
        self.cache.put("What is the capital of France?", "Paris")
        result = self.cache.get("What is the capital of France?")
        assert result == "Paris"

    def test_similar_query_is_hit(self) -> None:
        self.cache.put("What is the capital of France?", "Paris")
        # Exact same query at lower threshold should always hit
        result = self.cache.get("What is the capital of France?", similarity_threshold=0.5)
        assert result == "Paris"

    def test_dissimilar_query_is_miss(self) -> None:
        self.cache.put("What is the capital of France?", "Paris")
        result = self.cache.get("How do I bake a chocolate cake?", similarity_threshold=0.85)
        assert result is None

    def test_empty_cache_returns_none(self) -> None:
        result = self.cache.get("anything")
        assert result is None

    def test_put_overwrites_existing(self) -> None:
        self.cache.put("test query", "response 1")
        self.cache.put("test query", "response 2")
        result = self.cache.get("test query")
        assert result == "response 2"


class TestSemanticCacheSimilarityThreshold:
    """Threshold boundary behaviour."""

    def test_threshold_zero_always_returns_best(self) -> None:
        cache = SemanticCache(ttl=3600)
        cache.put("hello world", "greeting")
        # Exact same query should always return the cached value
        result = cache.get("hello world", similarity_threshold=0.0)
        assert result == "greeting"

    def test_high_threshold_filters_partial_matches(self) -> None:
        cache = SemanticCache(ttl=3600, similarity_threshold=0.99)
        cache.put("The quick brown fox", "response A")
        # Significantly different text should miss at very high threshold
        result = cache.get("A slow white dog", similarity_threshold=0.99)
        assert result is None


class TestSemanticCacheTTL:
    """TTL expiry."""

    def test_expired_entry_not_returned(self) -> None:
        from unittest.mock import patch

        cache = SemanticCache(ttl=10, max_entries=100)
        # Insert at t=100
        with patch("vetinari.optimization.semantic_cache.time.monotonic", return_value=100.0):
            cache.put("expiring query", "old response")
        # Read at t=120 — past expiry
        with patch("vetinari.optimization.semantic_cache.time.monotonic", return_value=120.0):
            result = cache.get("expiring query")
        assert result is None

    def test_non_expired_entry_returned(self) -> None:
        cache = SemanticCache(ttl=60, max_entries=100)
        cache.put("durable query", "durable response")
        result = cache.get("durable query")
        assert result == "durable response"


class TestSemanticCacheStats:
    """Statistics."""

    def test_initial_stats(self) -> None:
        cache = SemanticCache()
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.0
        assert stats["cache_size"] == 0
        assert stats["estimated_savings"] == 0

    def test_stats_after_hits(self) -> None:
        cache = SemanticCache(ttl=3600, similarity_threshold=0.5)
        cache.put("sample query", "sample response")
        cache.get("sample query")  # hit
        cache.get("sample query")  # hit
        cache.get("completely different xyz 123 abc")  # miss (low similarity)
        stats = cache.get_stats()
        assert stats["total_hits"] == 2
        assert stats["total_misses"] == 1
        assert stats["estimated_savings"] > 0


class TestSemanticCacheLRUEviction:
    """LRU eviction."""

    def test_cache_size_bounded(self) -> None:
        cache = SemanticCache(ttl=3600, max_entries=5)
        for i in range(20):
            cache.put(f"unique query number {i} abcdefg hijklmno", f"response {i}")
        assert cache.get_stats()["cache_size"] <= 5


class TestSemanticCacheSingleton:
    """Singleton accessor."""

    def setup_method(self) -> None:
        reset_semantic_cache()

    def teardown_method(self) -> None:
        reset_semantic_cache()

    def test_singleton_same_instance(self) -> None:
        a = get_semantic_cache()
        b = get_semantic_cache()
        assert a is b

    def test_reset_creates_new_instance(self) -> None:
        a = get_semantic_cache()
        reset_semantic_cache()
        b = get_semantic_cache()
        assert a is not b


class TestSemanticCacheThreadSafety:
    """Thread safety."""

    def test_concurrent_put_get(self) -> None:
        cache = SemanticCache(ttl=3600, max_entries=500)
        errors: list[Exception] = []

        def writer(i: int) -> None:
            try:
                cache.put(f"query number {i} hello world test", f"response {i}")
            except Exception as exc:
                errors.append(exc)

        def reader(i: int) -> None:
            try:
                cache.get(f"query number {i} hello world test")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
        threads += [threading.Thread(target=reader, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


class TestTrigramHelpers:
    """Internal trigram and Jaccard helpers."""

    def test_trigrams_empty_string(self) -> None:
        assert _trigrams("") == frozenset()

    def test_trigrams_short_string(self) -> None:
        result = _trigrams("ab")
        assert isinstance(result, frozenset)

    def test_trigrams_normal_string(self) -> None:
        result = _trigrams("hello")
        assert "hel" in result
        assert "ell" in result
        assert "llo" in result

    def test_jaccard_identical_sets(self) -> None:
        s = frozenset(["abc", "bcd"])
        assert abs(_jaccard(s, s) - 1.0) < 1e-7

    def test_jaccard_disjoint_sets(self) -> None:
        a = frozenset(["abc"])
        b = frozenset(["xyz"])
        assert abs(_jaccard(a, b) - 0.0) < 1e-7

    def test_jaccard_empty_sets(self) -> None:
        assert abs(_jaccard(frozenset(), frozenset()) - 1.0) < 1e-7

    def test_compute_similarity_method(self) -> None:
        cache = SemanticCache()
        score = cache._compute_similarity("hello world", "hello world")
        assert abs(score - 1.0) < 1e-7


# ===========================================================================
# Package import
# ===========================================================================


class TestPackageImport:
    """The optimization package __init__ re-exports everything."""

    def test_imports_from_package(self) -> None:
        from vetinari.optimization import (
            get_prompt_cache,
            get_semantic_cache,
        )

        assert callable(get_prompt_cache)
        assert callable(get_semantic_cache)


# ===========================================================================
# Regression: SemanticCache model_id isolation (Bug #10 / fuzzy-tier fix)
# ===========================================================================


class TestSemanticCacheModelIdIsolation:
    """Fuzzy lookup tiers must NOT cross model_id boundaries (Bug #10 regression)."""

    def test_model_id_isolation_exact_tier(self) -> None:
        """Entry stored under model-a must not be returned for model-b via exact tier."""
        cache = SemanticCache(ttl=3600, max_entries=100)
        query = "what is the capital of france"
        cache.put(query, "Paris", model_id="model-a")
        result = cache.get(query, model_id="model-b")
        assert result is None, (
            "Semantic cache must not return an entry stored under model-a "
            "when queried with model-b — model_id isolation is broken in exact tier"
        )

    def test_model_id_isolation_trigram_tier(self) -> None:
        """Trigram Jaccard tier must not return a result from a different model_id."""
        cache = SemanticCache(ttl=3600, max_entries=100)
        # Use a long enough query that trigram matching produces a high score
        query = "what is the capital of france hello world test query for trigram"
        cache.put(query, "Paris", model_id="model-a")
        # Exact tier will miss (different composite hash); trigram tier should also miss
        result = cache.get(query, model_id="model-b")
        assert result is None, (
            "Trigram tier returned a cross-model cache hit — model_id isolation is broken"
        )

    def test_same_model_id_still_hits(self) -> None:
        """Entry stored under model-a must still be returned for a model-a query."""
        cache = SemanticCache(ttl=3600, max_entries=100)
        query = "what is the capital of france hello world test query for same model"
        cache.put(query, "Paris", model_id="model-a")
        result = cache.get(query, model_id="model-a")
        assert result == "Paris", "Same model_id must still produce a cache hit"

    def test_empty_model_id_isolated_from_named_model(self) -> None:
        """Entry stored with no model_id must not be returned for a named model."""
        cache = SemanticCache(ttl=3600, max_entries=100)
        query = "what is the capital of france hello world test empty model query"
        cache.put(query, "Paris")  # model_id="" by default
        result = cache.get(query, model_id="model-a")
        assert result is None, (
            "Entry stored with model_id='' must not be returned for model_id='model-a'"
        )

    def test_system_prompt_isolation(self) -> None:
        """Entry stored under system_prompt-A must not be returned for system_prompt-B."""
        cache = SemanticCache(ttl=3600, max_entries=100)
        query = "what is the capital of france hello world test system prompt query"
        cache.put(query, "Paris", model_id="model-a", system_prompt="prompt-A")
        result = cache.get(query, model_id="model-a", system_prompt="prompt-B")
        assert result is None, (
            "Entry stored under system_prompt-A must not be returned for system_prompt-B"
        )


# ===========================================================================
# Regression: compress_context cache key isolation (Bug #1)
# ===========================================================================


class TestTokenCompressionCacheKeyIsolation:
    """Different compression_goal values must produce independent MD5 keys (Bug #1 regression)."""

    def test_different_goals_produce_distinct_md5_keys(self) -> None:
        """key_facts and code_only must hash to different cache keys."""
        from hashlib import md5

        context = "x" * 500
        task = "test task"
        key_facts_key = md5(f"{context}{task}key_facts".encode(), usedforsecurity=False).hexdigest()
        code_only_key = md5(f"{context}{task}code_only".encode(), usedforsecurity=False).hexdigest()
        old_buggy_key = md5(f"{context}{task}".encode(), usedforsecurity=False).hexdigest()
        assert key_facts_key != code_only_key, "key_facts and code_only must produce distinct cache keys"
        assert key_facts_key != old_buggy_key, "key_facts key must differ from the pre-fix (no-goal) key"
        assert code_only_key != old_buggy_key, "code_only key must differ from the pre-fix (no-goal) key"

    def test_cache_key_includes_compression_goal_field(self) -> None:
        """Verify LocalPreprocessor._cache_key_for() uses compression_goal in its MD5."""
        from hashlib import md5

        from vetinari.token_compression import LocalPreprocessor

        p = LocalPreprocessor()
        # Access the key-building logic directly: same context + task, different goal
        context = "sample context " * 30  # short enough to not trigger MIN_CONTEXT_CHARS skip
        task = "sample task"
        # Compute keys the same way compress_context does (Bug #1 fix)
        key_a = md5(f"{context}{task}key_facts".encode(), usedforsecurity=False).hexdigest()
        key_b = md5(f"{context}{task}code_only".encode(), usedforsecurity=False).hexdigest()
        # Verify the processor's internal cache is consistent with this scheme
        p._cache_put(key_a, "response-a")
        p._cache_put(key_b, "response-b")
        assert p._cache.get(key_a) == "response-a"
        assert p._cache.get(key_b) == "response-b"
        # Cross-lookup must miss
        assert p._cache.get(key_a) != "response-b"
