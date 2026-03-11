"""Semantic caching for LLM responses (P10.5).

Stores query/response pairs and retrieves cached responses when a new query
is semantically similar to a previously cached one.  Similarity is computed
using character trigram Jaccard similarity — no external vector DB required.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS: int = 86400  # 24 hours
_DEFAULT_MAX_ENTRIES: int = 500
_DEFAULT_SIMILARITY_THRESHOLD: float = 0.85

_instance: Optional[SemanticCache] = None
_instance_lock: threading.Lock = threading.Lock()


@dataclass
class CacheEntry:
    """A single entry stored in the :class:`SemanticCache`.

    Attributes:
        query_hash: SHA-256 hex digest of the query text (used as cache key).
        query_text: Original query string.
        response: Cached LLM response string.
        embedding: Pre-computed trigram set for similarity lookups.
        timestamp: Monotonic insertion time.
        hit_count: Number of times this entry has been retrieved.
    """

    query_hash: str
    query_text: str
    response: str
    embedding: frozenset[str]
    timestamp: float
    hit_count: int = 0


class SemanticCache:
    """Thread-safe semantic cache for LLM query/response pairs.

    Uses character trigram Jaccard similarity to determine whether an incoming
    query is semantically close enough to a cached query to reuse its response.

    Args:
        ttl: Time-to-live in seconds.  Default 86400 (24 hours).
        max_entries: Maximum entries before LRU eviction.  Default 500.
        similarity_threshold: Default minimum Jaccard score for a cache hit.
    """

    def __init__(
        self,
        ttl: int = _DEFAULT_TTL_SECONDS,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self._ttl = ttl
        self._max_entries = max_entries
        self._default_threshold = similarity_threshold
        self._lock = threading.Lock()
        # Ordered by insertion / access time for LRU eviction
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0
        self._tokens_saved: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        query: str,
        similarity_threshold: float = 0.0,
    ) -> Optional[str]:
        """Return a cached response if a similar query exists.

        Scans all non-expired cache entries and returns the response from the
        most-similar entry whose Jaccard score meets ``similarity_threshold``.

        Args:
            query: The new query text to look up.
            similarity_threshold: Minimum Jaccard similarity score (0–1) for a
                                  hit.  Defaults to the instance-level threshold
                                  when 0.0 is supplied.

        Returns:
            Cached response string on hit, or ``None`` on miss.
        """
        threshold = similarity_threshold if similarity_threshold > 0.0 else self._default_threshold
        query_embedding = _trigrams(query)

        with self._lock:
            self._evict_expired()
            best_score = 0.0
            best_key: Optional[str] = None

            for key, entry in self._store.items():
                score = _jaccard(query_embedding, entry.embedding)
                if score > best_score:
                    best_score = score
                    best_key = key

            if best_key is not None and best_score >= threshold:
                entry = self._store[best_key]
                entry.hit_count += 1
                self._store.move_to_end(best_key)
                self._hits += 1
                saved = max(1, len(entry.response) // 4)
                self._tokens_saved += saved
                logger.debug(
                    "SemanticCache HIT (score=%.3f) for query '%.40s...'",
                    best_score,
                    query,
                )
                return entry.response

            self._misses += 1
            return None

    def put(self, query: str, response: str) -> None:
        """Store a query/response pair in the cache.

        If an identical query hash is already present the entry is refreshed
        with the new response and moved to the most-recently-used position.

        Args:
            query: The query text.
            response: The LLM response to cache.
        """
        import hashlib

        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        embedding = _trigrams(query)

        with self._lock:
            self._evict_expired()
            entry = CacheEntry(
                query_hash=query_hash,
                query_text=query,
                response=response,
                embedding=embedding,
                timestamp=time.monotonic(),
            )
            self._store[query_hash] = entry
            self._store.move_to_end(query_hash)
            self._evict_lru()

    def get_stats(self) -> dict:
        """Return cache statistics.

        Returns:
            Dictionary with keys: ``hit_rate``, ``cache_size``,
            ``estimated_savings`` (token count), ``total_hits``,
            ``total_misses``.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "hit_rate": hit_rate,
                "cache_size": len(self._store),
                "estimated_savings": self._tokens_saved,
                "total_hits": self._hits,
                "total_misses": self._misses,
            }

    def clear(self) -> None:
        """Remove all entries and reset statistics."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0
            self._tokens_saved = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_expired(self) -> None:
        """Remove expired entries (must be called under lock)."""
        now = time.monotonic()
        expired = [k for k, e in self._store.items() if now - e.timestamp >= self._ttl]
        for k in expired:
            del self._store[k]

    def _evict_lru(self) -> None:
        """Remove least-recently-used entries until within capacity (must be called under lock)."""
        while len(self._store) > self._max_entries:
            self._store.popitem(last=False)

    # ------------------------------------------------------------------
    # Similarity (exposed for testing)
    # ------------------------------------------------------------------

    def _compute_similarity(self, a: str, b: str) -> float:
        """Compute character trigram Jaccard similarity between two strings.

        Args:
            a: First string.
            b: Second string.

        Returns:
            Jaccard similarity score in [0, 1].
        """
        return _jaccard(_trigrams(a), _trigrams(b))


# ---------------------------------------------------------------------------
# Module-level similarity helpers
# ---------------------------------------------------------------------------


def _trigrams(text: str) -> frozenset[str]:
    """Extract character trigrams from *text*.

    Args:
        text: Input string.

    Returns:
        Frozenset of 3-character substrings.
    """
    t = text.lower()
    if len(t) < 3:
        return frozenset({t}) if t else frozenset()
    return frozenset(t[i : i + 3] for i in range(len(t) - 2))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Compute Jaccard similarity between two frozensets.

    Args:
        a: First set.
        b: Second set.

    Returns:
        |a ∩ b| / |a ∪ b|, or 1.0 if both sets are empty.
    """
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


def get_semantic_cache(
    ttl: int = _DEFAULT_TTL_SECONDS,
    max_entries: int = _DEFAULT_MAX_ENTRIES,
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
) -> SemanticCache:
    """Return the module-level singleton :class:`SemanticCache`.

    Args:
        ttl: TTL in seconds (used on first creation only).
        max_entries: Max entries (used on first creation only).
        similarity_threshold: Default similarity threshold (used on first creation only).

    Returns:
        The singleton :class:`SemanticCache` instance.
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = SemanticCache(
                    ttl=ttl,
                    max_entries=max_entries,
                    similarity_threshold=similarity_threshold,
                )
    return _instance


def reset_semantic_cache() -> None:
    """Destroy the singleton so the next call to ``get_semantic_cache`` creates a fresh one.

    Intended for use in tests only.
    """
    global _instance
    with _instance_lock:
        _instance = None
