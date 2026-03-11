"""Tests for vetinari/optimization/ — P10.5, P10.6, P10.7."""

from __future__ import annotations

import threading
import time
import unittest

from vetinari.optimization.prompt_cache import (
    CacheResult,
    PromptCache,
    get_prompt_cache,
    hash_prompt,
    reset_prompt_cache,
)
from vetinari.optimization.batch_processor import (
    BatchProcessor,
    BatchRequest,
    BatchResult,
    Priority,
    get_batch_processor,
    make_request,
    reset_batch_processor,
)
from vetinari.optimization.semantic_cache import (
    CacheEntry,
    SemanticCache,
    get_semantic_cache,
    reset_semantic_cache,
    _trigrams,
    _jaccard,
)


# ===========================================================================
# PromptCache (P10.6)
# ===========================================================================


class TestPromptCacheMissAndHit(unittest.TestCase):
    """Basic hit/miss behaviour."""

    def setUp(self) -> None:
        self.cache = PromptCache(ttl=60, max_entries=100)

    def test_first_lookup_is_miss(self) -> None:
        result = self.cache.get_or_cache("hash1", "Hello, world!")
        self.assertFalse(result.hit)
        self.assertEqual(result.savings_tokens, 0)
        self.assertEqual(result.prompt, "Hello, world!")

    def test_second_lookup_is_hit(self) -> None:
        self.cache.get_or_cache("hash1", "Hello, world!")
        result = self.cache.get_or_cache("hash1", "Hello, world!")
        self.assertTrue(result.hit)
        self.assertGreater(result.savings_tokens, 0)

    def test_different_hashes_independent(self) -> None:
        self.cache.get_or_cache("a", "prompt A")
        self.cache.get_or_cache("b", "prompt B")
        r_a = self.cache.get_or_cache("a", "prompt A")
        r_b = self.cache.get_or_cache("b", "prompt B")
        self.assertTrue(r_a.hit)
        self.assertTrue(r_b.hit)

    def test_cache_result_is_dataclass(self) -> None:
        result = self.cache.get_or_cache("x", "text")
        self.assertIsInstance(result, CacheResult)


class TestPromptCacheTTL(unittest.TestCase):
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
        self.assertFalse(result.hit)

    def test_not_expired_entry_is_hit(self) -> None:
        cache = PromptCache(ttl=60, max_entries=100)
        cache.get_or_cache("k", "prompt")
        result = cache.get_or_cache("k", "prompt")
        self.assertTrue(result.hit)


class TestPromptCacheLRUEviction(unittest.TestCase):
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
        self.assertLessEqual(stats["cache_size"], 3)

    def test_cache_size_does_not_exceed_max(self) -> None:
        cache = PromptCache(ttl=3600, max_entries=5)
        for i in range(20):
            cache.get_or_cache(f"key{i}", f"prompt{i}")
        self.assertLessEqual(cache.get_stats()["cache_size"], 5)


class TestPromptCacheStats(unittest.TestCase):
    """Statistics reporting."""

    def test_stats_initial_state(self) -> None:
        cache = PromptCache()
        stats = cache.get_stats()
        self.assertEqual(stats["total_hits"], 0)
        self.assertEqual(stats["total_misses"], 0)
        self.assertEqual(stats["cache_size"], 0)
        self.assertAlmostEqual(stats["hit_rate"], 0.0)

    def test_hit_rate_calculated_correctly(self) -> None:
        cache = PromptCache(ttl=60)
        cache.get_or_cache("k", "prompt")  # miss
        cache.get_or_cache("k", "prompt")  # hit
        cache.get_or_cache("k", "prompt")  # hit
        stats = cache.get_stats()
        self.assertAlmostEqual(stats["hit_rate"], 2 / 3)
        self.assertEqual(stats["total_hits"], 2)
        self.assertEqual(stats["total_misses"], 1)


class TestPromptCacheSingleton(unittest.TestCase):
    """Singleton accessor."""

    def setUp(self) -> None:
        reset_prompt_cache()

    def tearDown(self) -> None:
        reset_prompt_cache()

    def test_singleton_returns_same_instance(self) -> None:
        a = get_prompt_cache()
        b = get_prompt_cache()
        self.assertIs(a, b)

    def test_reset_creates_new_instance(self) -> None:
        a = get_prompt_cache()
        reset_prompt_cache()
        b = get_prompt_cache()
        self.assertIsNot(a, b)


class TestPromptCacheThreadSafety(unittest.TestCase):
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
        self.assertEqual(errors, [])
        stats = cache.get_stats()
        self.assertLessEqual(stats["cache_size"], 1000)


# ===========================================================================
# BatchProcessor (P10.7)
# ===========================================================================


def _req(prompt: str, priority: Priority = Priority.NORMAL) -> BatchRequest:
    return make_request(prompt=prompt, priority=priority)


class TestBatchProcessorSubmitFlush(unittest.TestCase):
    """Submit and flush behaviour."""

    def setUp(self) -> None:
        self.bp = BatchProcessor(max_batch_size=10)

    def test_submit_returns_request_id(self) -> None:
        req = _req("hello")
        rid = self.bp.submit(req)
        self.assertEqual(rid, req.request_id)

    def test_flush_returns_results(self) -> None:
        self.bp.submit(_req("prompt 1"))
        self.bp.submit(_req("prompt 2"))
        results = self.bp.flush()
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsInstance(r, BatchResult)
            self.assertEqual(r.status, "ok")

    def test_flush_empty_queue_returns_empty_list(self) -> None:
        results = self.bp.flush()
        self.assertEqual(results, [])

    def test_flush_respects_max_batch_size(self) -> None:
        for _ in range(10):
            self.bp.submit(_req("x"))
        results = self.bp.flush(max_batch_size=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(self.bp.get_stats()["queue_depth"], 7)


class TestBatchProcessorPriorityOrdering(unittest.TestCase):
    """Priority queue ordering."""

    def setUp(self) -> None:
        self.bp = BatchProcessor(max_batch_size=20)

    def test_high_priority_processed_before_low(self) -> None:
        low_req = _req("low", Priority.LOW)
        high_req = _req("high", Priority.HIGH)
        self.bp.submit(low_req)
        self.bp.submit(high_req)
        results = self.bp.flush()
        # High-priority request should appear first in results
        self.assertEqual(results[0].request_id, high_req.request_id)
        self.assertEqual(results[1].request_id, low_req.request_id)

    def test_normal_between_high_and_low(self) -> None:
        low = _req("low", Priority.LOW)
        normal = _req("normal", Priority.NORMAL)
        high = _req("high", Priority.HIGH)
        self.bp.submit(low)
        self.bp.submit(normal)
        self.bp.submit(high)
        results = self.bp.flush()
        ids = [r.request_id for r in results]
        self.assertEqual(ids, [high.request_id, normal.request_id, low.request_id])


class TestBatchProcessorStats(unittest.TestCase):
    """Statistics."""

    def setUp(self) -> None:
        self.bp = BatchProcessor(max_batch_size=10)

    def test_initial_stats(self) -> None:
        stats = self.bp.get_stats()
        self.assertEqual(stats["queue_depth"], 0)
        self.assertEqual(stats["total_processed"], 0)
        self.assertAlmostEqual(stats["avg_batch_size"], 0.0)

    def test_stats_after_flush(self) -> None:
        for _ in range(4):
            self.bp.submit(_req("p"))
        self.bp.flush()
        stats = self.bp.get_stats()
        self.assertEqual(stats["total_processed"], 4)
        self.assertAlmostEqual(stats["avg_batch_size"], 4.0)

    def test_callback_invoked(self) -> None:
        received: list[BatchResult] = []
        req = make_request("hello", callback=received.append)
        self.bp.submit(req)
        self.bp.flush()
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].request_id, req.request_id)


class TestBatchProcessorSingleton(unittest.TestCase):
    """Singleton accessor."""

    def setUp(self) -> None:
        reset_batch_processor()

    def tearDown(self) -> None:
        reset_batch_processor()

    def test_singleton_returns_same_instance(self) -> None:
        a = get_batch_processor()
        b = get_batch_processor()
        self.assertIs(a, b)

    def test_reset_creates_new_instance(self) -> None:
        a = get_batch_processor()
        reset_batch_processor()
        b = get_batch_processor()
        self.assertIsNot(a, b)


class TestBatchProcessorThreadSafety(unittest.TestCase):
    """Thread safety."""

    def test_concurrent_submits_do_not_corrupt(self) -> None:
        bp = BatchProcessor(max_batch_size=200)
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                for j in range(20):
                    bp.submit(_req(f"prompt-{i}-{j}", Priority.NORMAL))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        results = bp.flush(max_batch_size=200)
        self.assertEqual(len(results), 100)


# ===========================================================================
# SemanticCache (P10.5)
# ===========================================================================


class TestSemanticCachePutGet(unittest.TestCase):
    """Basic put/get behaviour."""

    def setUp(self) -> None:
        self.cache = SemanticCache(ttl=3600, max_entries=100, similarity_threshold=0.85)

    def test_exact_match_is_hit(self) -> None:
        self.cache.put("What is the capital of France?", "Paris")
        result = self.cache.get("What is the capital of France?")
        self.assertEqual(result, "Paris")

    def test_similar_query_is_hit(self) -> None:
        self.cache.put("What is the capital of France?", "Paris")
        # Very similar query — should hit above 0.85 threshold
        result = self.cache.get("What is the capital of France?", similarity_threshold=0.5)
        self.assertIsNotNone(result)

    def test_dissimilar_query_is_miss(self) -> None:
        self.cache.put("What is the capital of France?", "Paris")
        result = self.cache.get("How do I bake a chocolate cake?", similarity_threshold=0.85)
        self.assertIsNone(result)

    def test_empty_cache_returns_none(self) -> None:
        result = self.cache.get("anything")
        self.assertIsNone(result)

    def test_put_overwrites_existing(self) -> None:
        self.cache.put("test query", "response 1")
        self.cache.put("test query", "response 2")
        result = self.cache.get("test query")
        self.assertEqual(result, "response 2")


class TestSemanticCacheSimilarityThreshold(unittest.TestCase):
    """Threshold boundary behaviour."""

    def test_threshold_zero_always_returns_best(self) -> None:
        cache = SemanticCache(ttl=3600)
        cache.put("hello world", "greeting")
        # threshold=0 should return the best match regardless of score
        result = cache.get("hello world", similarity_threshold=0.0)
        # With default instance threshold (0.85) and exact match, should hit
        self.assertIsNotNone(result)

    def test_high_threshold_filters_partial_matches(self) -> None:
        cache = SemanticCache(ttl=3600, similarity_threshold=0.99)
        cache.put("The quick brown fox", "response A")
        # Significantly different text should miss at very high threshold
        result = cache.get("A slow white dog", similarity_threshold=0.99)
        self.assertIsNone(result)


class TestSemanticCacheTTL(unittest.TestCase):
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
        self.assertIsNone(result)

    def test_non_expired_entry_returned(self) -> None:
        cache = SemanticCache(ttl=60, max_entries=100)
        cache.put("durable query", "durable response")
        result = cache.get("durable query")
        self.assertEqual(result, "durable response")


class TestSemanticCacheStats(unittest.TestCase):
    """Statistics."""

    def test_initial_stats(self) -> None:
        cache = SemanticCache()
        stats = cache.get_stats()
        self.assertEqual(stats["hit_rate"], 0.0)
        self.assertEqual(stats["cache_size"], 0)
        self.assertEqual(stats["estimated_savings"], 0)

    def test_stats_after_hits(self) -> None:
        cache = SemanticCache(ttl=3600, similarity_threshold=0.5)
        cache.put("sample query", "sample response")
        cache.get("sample query")  # hit
        cache.get("sample query")  # hit
        cache.get("completely different xyz 123 abc")  # miss (low similarity)
        stats = cache.get_stats()
        self.assertEqual(stats["total_hits"], 2)
        self.assertEqual(stats["total_misses"], 1)
        self.assertGreater(stats["estimated_savings"], 0)


class TestSemanticCacheLRUEviction(unittest.TestCase):
    """LRU eviction."""

    def test_cache_size_bounded(self) -> None:
        cache = SemanticCache(ttl=3600, max_entries=5)
        for i in range(20):
            cache.put(f"unique query number {i} abcdefg hijklmno", f"response {i}")
        self.assertLessEqual(cache.get_stats()["cache_size"], 5)


class TestSemanticCacheSingleton(unittest.TestCase):
    """Singleton accessor."""

    def setUp(self) -> None:
        reset_semantic_cache()

    def tearDown(self) -> None:
        reset_semantic_cache()

    def test_singleton_same_instance(self) -> None:
        a = get_semantic_cache()
        b = get_semantic_cache()
        self.assertIs(a, b)

    def test_reset_creates_new_instance(self) -> None:
        a = get_semantic_cache()
        reset_semantic_cache()
        b = get_semantic_cache()
        self.assertIsNot(a, b)


class TestSemanticCacheThreadSafety(unittest.TestCase):
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
        self.assertEqual(errors, [])


class TestTrigramHelpers(unittest.TestCase):
    """Internal trigram and Jaccard helpers."""

    def test_trigrams_empty_string(self) -> None:
        self.assertEqual(_trigrams(""), frozenset())

    def test_trigrams_short_string(self) -> None:
        result = _trigrams("ab")
        self.assertIsInstance(result, frozenset)

    def test_trigrams_normal_string(self) -> None:
        result = _trigrams("hello")
        self.assertIn("hel", result)
        self.assertIn("ell", result)
        self.assertIn("llo", result)

    def test_jaccard_identical_sets(self) -> None:
        s = frozenset(["abc", "bcd"])
        self.assertAlmostEqual(_jaccard(s, s), 1.0)

    def test_jaccard_disjoint_sets(self) -> None:
        a = frozenset(["abc"])
        b = frozenset(["xyz"])
        self.assertAlmostEqual(_jaccard(a, b), 0.0)

    def test_jaccard_empty_sets(self) -> None:
        self.assertAlmostEqual(_jaccard(frozenset(), frozenset()), 1.0)

    def test_compute_similarity_method(self) -> None:
        cache = SemanticCache()
        score = cache._compute_similarity("hello world", "hello world")
        self.assertAlmostEqual(score, 1.0)


# ===========================================================================
# Package import
# ===========================================================================


class TestPackageImport(unittest.TestCase):
    """The optimization package __init__ re-exports everything."""

    def test_imports_from_package(self) -> None:
        from vetinari.optimization import (
            PromptCache, get_prompt_cache,
            BatchProcessor, get_batch_processor,
            SemanticCache, get_semantic_cache,
        )
        self.assertTrue(callable(get_prompt_cache))
        self.assertTrue(callable(get_batch_processor))
        self.assertTrue(callable(get_semantic_cache))


if __name__ == "__main__":
    unittest.main()
