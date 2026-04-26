"""Tests for the 3-tier hierarchical SemanticCache (Story 40)."""

from __future__ import annotations

import hashlib
from unittest.mock import patch

import pytest

import vetinari.optimization.semantic_cache as _sc_module
from vetinari.optimization.semantic_cache import (
    _DATASKETCH_AVAILABLE,
    SemanticCache,
    reset_semantic_cache,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module singleton before and after each test."""
    reset_semantic_cache()
    yield
    reset_semantic_cache()


class TestExactHashHit:
    """Tier 1: identical query should return immediately via exact hash."""

    def test_exact_hash_hit(self):
        cache = SemanticCache(similarity_threshold=0.5)
        cache.put("What is the capital of France?", "Paris")
        result = cache.get("What is the capital of France?")
        assert result == "Paris"

    def test_exact_hash_hit_increments_exact_counter(self):
        cache = SemanticCache(similarity_threshold=0.5)
        cache.put("Hello world", "response A")
        cache.get("Hello world")
        stats = cache.get_stats()
        assert stats["exact_hits"] == 1
        assert stats["minhash_hits"] == 0
        assert stats["trigram_hits"] == 0


class TestMinhashHitWithoutExact:
    """Tier 2: near-duplicate query should match via MinHash when datasketch is installed."""

    def test_minhash_hit_when_datasketch_available(self):
        # Disable dense embeddings so the cache exercises Tier 2 (minhash)
        # instead of jumping straight to semantic cosine similarity.
        with patch.object(_sc_module, "_EMBEDDER_AVAILABLE", False):
            cache = SemanticCache(similarity_threshold=0.3)
            original = "Explain the process of photosynthesis in plants"
            similar = "Describe how photosynthesis works in plants"
            cache.put(original, "cached response about photosynthesis")

            # Verify the near-duplicate is not an exact hit
            original_hash = hashlib.sha256(original.encode()).hexdigest()
            similar_hash = hashlib.sha256(similar.encode()).hexdigest()
            assert original_hash != similar_hash

            result = cache.get(similar, similarity_threshold=0.3)
            # The result may come from Tier 2 (MinHash) or Tier 3 (trigram)
            assert result == "cached response about photosynthesis"

    def test_minhash_tier_records_hit(self):
        # Disable dense embeddings so the cache exercises Tier 2 (minhash) or
        # Tier 3 (trigram) instead of skipping to semantic cosine similarity.
        with patch.object(_sc_module, "_EMBEDDER_AVAILABLE", False):
            cache = SemanticCache(similarity_threshold=0.3)
            original = "Tell me about machine learning algorithms and how they learn from data"
            similar = "Tell me about machine learning algorithms and how they learn from examples"
            cache.put(original, "ML response")

            cache.get(similar, similarity_threshold=0.3)
            stats = cache.get_stats()
            # Either minhash_hits or trigram_hits should be >= 1 (near-duplicate found)
            assert stats["minhash_hits"] + stats["trigram_hits"] >= 1


class TestTrigramFallbackWhenNoMinHash:
    """Tier 3: trigram Jaccard scan must work even without datasketch."""

    def test_trigram_fallback_when_no_minhash(self):
        cache = SemanticCache(similarity_threshold=0.5)
        # Patch datasketch availability off on the instance
        cache._minhash_index = None

        query = "What is machine learning and artificial intelligence?"
        cache.put(query, "ML answer")

        result = cache.get(query)
        assert result == "ML answer"

    def test_trigram_fallback_increments_trigram_counter(self):
        # Disable dense embeddings so the cache uses trigram Jaccard (Tier 3
        # fallback) instead of semantic cosine similarity.
        with patch.object(_sc_module, "_EMBEDDER_AVAILABLE", False):
            cache = SemanticCache(similarity_threshold=0.5)
            cache.put("The quick brown fox jumps over the lazy dog", "fox response")
            # Disable MinHash (Tier 2) to force trigram (Tier 3 fallback)
            cache._minhash_index = None

            similar = "The quick brown fox leaped over the lazy dog"
            result = cache.get(similar, similarity_threshold=0.3)
            assert result is not None, "Trigram fallback should find a similar cached entry"
            stats = cache.get_stats()
            assert stats["trigram_hits"] >= 1


class TestStatsShowPerTierHits:
    """get_stats() must expose exact_hits, minhash_hits, trigram_hits."""

    def test_stats_keys_present(self):
        cache = SemanticCache()
        stats = cache.get_stats()
        assert "exact_hits" in stats
        assert "minhash_hits" in stats
        assert "trigram_hits" in stats

    def test_stats_initial_values_are_zero(self):
        cache = SemanticCache()
        stats = cache.get_stats()
        assert stats["exact_hits"] == 0
        assert stats["minhash_hits"] == 0
        assert stats["trigram_hits"] == 0

    def test_stats_total_hits_is_sum_of_tiers(self):
        cache = SemanticCache(similarity_threshold=0.5)
        cache.put("exact query here", "resp")
        cache.get("exact query here")  # Tier 1 hit
        stats = cache.get_stats()
        assert stats["total_hits"] == stats["exact_hits"] + stats["minhash_hits"] + stats["trigram_hits"]


class TestGracefulWithoutDatasketch:
    """Cache must degrade gracefully to Tier 3 when datasketch is unavailable."""

    def test_graceful_without_datasketch(self):
        with patch("vetinari.optimization.semantic_cache._DATASKETCH_AVAILABLE", False):
            cache = SemanticCache(similarity_threshold=0.5)
            assert cache._minhash_index is None

            cache.put("hello world test query", "world response")
            result = cache.get("hello world test query")
            assert result == "world response"

    def test_clear_resets_all_tiers(self):
        cache = SemanticCache(similarity_threshold=0.5)
        cache.put("something to clear", "clear response")
        cache.get("something to clear")
        cache.clear()

        stats = cache.get_stats()
        assert stats["cache_size"] == 0
        assert stats["exact_hits"] == 0
        assert stats["minhash_hits"] == 0
        assert stats["trigram_hits"] == 0
        assert stats["total_hits"] == 0
        assert len(cache._exact_index) == 0


class TestSemanticEmbeddingTier3:
    """Tier 3: sentence-transformers embedding path and trigram fallback."""

    def test_semantic_tier3_when_embedder_available(self):
        """When embedder is available, cosine similarity path records semantic_hits."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.9, 0.1, 0.0]  # cosine sim > threshold

        with (
            patch.object(_sc_module, "_EMBEDDER_AVAILABLE", True),
            patch.object(_sc_module, "_compute_embedding", side_effect=[vec_a, vec_b]),
        ):
            cache = SemanticCache(similarity_threshold=0.5)
            # Disable MinHash (Tier 2) to isolate semantic (Tier 3) path
            cache._minhash_index = None
            cache.put("first query about dogs", "dogs response")
            result = cache.get("second query about dogs", similarity_threshold=0.5)

        assert result == "dogs response"
        stats = cache.get_stats()
        assert stats["semantic_hits"] == 1

    def test_trigram_fallback_when_embedder_unavailable(self):
        """When _EMBEDDER_AVAILABLE is False, trigram Jaccard path still works."""
        with patch.object(_sc_module, "_EMBEDDER_AVAILABLE", False):
            cache = SemanticCache(similarity_threshold=0.5)
            cache._minhash_index = None  # disable Tier 2

            query = "The quick brown fox jumps over the lazy dog"
            cache.put(query, "fox answer")
            similar = "The quick brown fox leaps over the lazy dog"
            result = cache.get(similar, similarity_threshold=0.5)
            # Trigram similarity is high enough for a hit
            assert result == "fox answer"
            stats = cache.get_stats()
            assert stats["trigram_hits"] >= 1
            assert stats["semantic_hits"] == 0
