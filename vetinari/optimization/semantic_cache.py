"""Semantic caching for LLM responses (P10.5).

Stores query/response pairs and retrieves cached responses when a new query
is semantically similar to a previously cached one.  Uses a 3-tier MinCache
pattern:

  Tier 1 — exact match via SHA-256 hash (O(1))
  Tier 2 — approximate match via MinHash LSH (O(1), requires datasketch)
  Tier 3 — sentence-transformers embedding cosine sim (requires sentence-transformers)
           or trigram Jaccard scan fallback (O(n), always available)
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from vetinari.constants import CACHE_MAX_ENTRIES_SEMANTIC
from vetinari.utils.lazy_import import lazy_import
from vetinari.utils.math_helpers import cosine_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional datasketch import
# ---------------------------------------------------------------------------

_datasketch, _DATASKETCH_AVAILABLE = lazy_import("datasketch")
MinHash = _datasketch.MinHash if _datasketch else None  # type: ignore[assignment]
MinHashLSH = _datasketch.MinHashLSH if _datasketch else None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Optional sentence-transformers import (lazy-loaded on first use)
# ---------------------------------------------------------------------------


_EMBEDDER_AVAILABLE: bool | None = None  # Lazy-initialized on first use


def _check_embedder_available() -> bool:
    """Probe whether the unified embedder is usable for dense similarity.

    Tries to import ``get_embedder`` from ``vetinari.embeddings`` and checks
    the ``.available`` attribute.  Returns ``False`` on any import error or
    if the embedder signals it is not ready — ensuring Tier 3 degrades
    gracefully to trigram Jaccard when sentence-transformers is absent.

    Returns:
        ``True`` when the embedder is importable and ready, ``False`` otherwise.
    """
    try:
        from vetinari.embeddings import get_embedder

        return bool(get_embedder().available)
    except Exception:
        logger.warning("Embedder unavailable for semantic cache — semantic deduplication disabled")
        return False


def _get_embedder_available() -> bool:
    """Lazy-check embedder availability on first access (not at import time)."""
    global _EMBEDDER_AVAILABLE
    if _EMBEDDER_AVAILABLE is None:
        _EMBEDDER_AVAILABLE = _check_embedder_available()
    return _EMBEDDER_AVAILABLE


def _compute_embedding(text: str) -> list[float] | None:
    """Compute a dense embedding using the unified embedder from vetinari.embeddings.

    Uses the singleton ``get_embedder()`` which has proper double-checked locking
    and falls back to n-gram hashing when sentence-transformers is unavailable.

    Args:
        text: Input string to embed.

    Returns:
        A list of floats representing the embedding.
    """
    from vetinari.embeddings import get_embedder

    try:
        return get_embedder().embed(text)
    except Exception:
        logger.warning("Embedding computation failed for semantic cache — skipping Tier 3")
        return None


_DEFAULT_TTL_SECONDS: int = 86400  # 24 hours
_DEFAULT_SIMILARITY_THRESHOLD: float = 0.85

# "semantic" is the backend label used in TelemetryCollector.memory_metrics for
# cache dedup accounting — distinct from 'oc' / 'mnemosyne' memory backends.
_TELEMETRY_BACKEND: str = "semantic"


def _report_cache_hit() -> None:
    """Increment the dedup-hit counter in TelemetryCollector for the semantic cache."""
    try:
        from vetinari.telemetry import get_telemetry_collector

        get_telemetry_collector().record_dedup_hit(_TELEMETRY_BACKEND)
    except Exception:
        # Telemetry is best-effort — never let a metrics failure break a cache lookup.
        logger.warning("SemanticCache could not report dedup hit to telemetry — hit counter may be underreported")


def _report_cache_miss() -> None:
    """Increment the dedup-miss counter in TelemetryCollector for the semantic cache."""
    try:
        from vetinari.telemetry import get_telemetry_collector

        get_telemetry_collector().record_dedup_miss(_TELEMETRY_BACKEND)
    except Exception:
        logger.warning("SemanticCache could not report dedup miss to telemetry — miss counter may be underreported")


# Task-aware similarity thresholds — creative/code tasks need stricter matching
# because small prompt differences yield very different outputs, while error
# handling tasks can reuse cached responses more aggressively.
TASK_TYPE_THRESHOLDS: dict[str, float] = {
    "coding": 0.95,
    "code": 0.95,
    "creative": 0.95,
    "creative_writing": 0.95,
    "docs": 0.85,
    "documentation": 0.85,
    "research": 0.85,
    "error": 0.75,
    "error_recovery": 0.75,
    "security": 0.90,
    "data": 0.85,
    "general": 0.85,
}


def get_threshold_for_task_type(task_type: str) -> float:
    """Return the similarity threshold for a given task type.

    Creative tasks need stricter thresholds (0.95) to avoid stale cache hits.
    Error recovery can be lenient (0.75) since similar errors need similar fixes.

    Args:
        task_type: The task type string (e.g. "coding", "docs", "error").

    Returns:
        Similarity threshold between 0.0 and 1.0.
    """
    return TASK_TYPE_THRESHOLDS.get(task_type, _DEFAULT_SIMILARITY_THRESHOLD)


_MINHASH_NUM_PERM: int = 128
_MINHASH_THRESHOLD: float = 0.5

_instance: SemanticCache | None = None
_instance_lock: threading.Lock = threading.Lock()


@dataclass
class CacheEntry:  # noqa: VET114 — hit_count incremented in-place by SemanticCache lookups
    r"""A single entry stored in the :class:`SemanticCache`.

    Attributes:
        query_hash: SHA-256 hex digest of the composite key
            ``query + "\\x00" + model_id + "\\x00" + system_prompt``.
        query_text: Original query string.
        response: Cached LLM response string.
        embedding: Pre-computed trigram set for similarity lookups.
        timestamp: Monotonic insertion time.
        model_id: The model this response was generated by.  Empty string
            means "any model" (backwards-compatible default).
        system_prompt: The system prompt in use when the response was cached.
            Empty string means "any system prompt".
        hit_count: Number of times this entry has been retrieved.
        dense_embedding: Sentence-transformer embedding vector, or ``None``
            when sentence-transformers is not available.
    """

    query_hash: str
    query_text: str
    response: str
    embedding: frozenset[str]
    timestamp: float
    model_id: str = ""  # model that produced this cached response
    system_prompt: str = ""  # system prompt active when response was cached
    hit_count: int = 0
    dense_embedding: list[float] | None = None  # sentence-transformer embedding

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"CacheEntry(query_hash={self.query_hash!r}, model_id={self.model_id!r}, hit_count={self.hit_count!r})"


class SemanticCache:
    """Thread-safe semantic cache for LLM query/response pairs.

    Uses a 3-tier lookup strategy:

    1. **Exact** — O(1) SHA-256 hash match.
    2. **MinHash LSH** — O(1) approximate-nearest-neighbour when
       ``datasketch`` is installed.
    3. **Trigram Jaccard scan** — O(n) linear scan, always available.

    Args:
        ttl: Time-to-live in seconds.  Default 86400 (24 hours).
        max_entries: Maximum entries before LRU eviction.  Default 500.
        similarity_threshold: Default minimum Jaccard score for a cache hit.
    """

    def __init__(
        self,
        ttl: int = _DEFAULT_TTL_SECONDS,
        max_entries: int = CACHE_MAX_ENTRIES_SEMANTIC,
        similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self._ttl = ttl
        self._max_entries = max_entries
        self._default_threshold = similarity_threshold
        self._lock = threading.Lock()
        # Ordered by insertion / access time for LRU eviction
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()

        # Tier 1: exact hash index — SHA-256 hex → cache key (same value, kept separate for clarity)
        self._exact_index: dict[str, str] = {}

        # Tier 2: MinHash LSH index (optional — None when datasketch unavailable)
        self._minhash_index: object | None = None
        if _DATASKETCH_AVAILABLE:
            self._minhash_index = MinHashLSH(threshold=_MINHASH_THRESHOLD, num_perm=_MINHASH_NUM_PERM)

        # Per-tier hit counters
        self._exact_hits: int = 0
        self._minhash_hits: int = 0
        self._semantic_hits: int = 0
        self._trigram_hits: int = 0

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
        task_type: str = "",
        model_id: str = "",
        system_prompt: str = "",
    ) -> str | None:
        """Return a cached response if a similar query exists.

        Runs a 3-tier lookup: exact hash → MinHash LSH → trigram Jaccard.
        Returns the first matching response.  When ``task_type`` is supplied the
        per-task-type threshold is used instead of the instance default, so
        creative tasks (0.95) are stricter than error-recovery tasks (0.75).
        Hit/miss outcomes are reported to ``TelemetryCollector`` under the
        ``"semantic"`` backend key.

        The ``model_id`` and ``system_prompt`` parameters are used for
        isolation: a response cached for model-a is NEVER returned for a
        query under model-b, even if the query text is identical.  All three
        tiers (exact, LSH, trigram) enforce this constraint.

        Args:
            query: The new query text to look up.
            similarity_threshold: Minimum Jaccard similarity score (0-1) for a
                                  hit.  Defaults to the instance-level threshold
                                  when 0.0 is supplied.
            task_type: Optional task type string used to look up a task-specific
                       similarity threshold via ``get_threshold_for_task_type``.
                       Ignored when ``similarity_threshold`` is non-zero.
            model_id: The model this request is targeting.  Entries stored
                      under a different model_id are not returned.
            system_prompt: The active system prompt.  Entries stored with a
                           different system_prompt are not returned.

        Returns:
            Cached response string on hit, or ``None`` on miss.
        """
        if similarity_threshold > 0.0:
            threshold = similarity_threshold
        elif task_type:
            threshold = get_threshold_for_task_type(task_type)
        else:
            threshold = self._default_threshold

        # Composite key includes model_id and system_prompt so the exact tier
        # is automatically isolated.  Fuzzy tiers check entry.model_id directly.
        key_material = f"{query}\x00{model_id}\x00{system_prompt}"
        composite_hash = hashlib.sha256(key_material.encode("utf-8")).hexdigest()

        with self._lock:
            self._evict_expired()

            # ── Tier 1: Exact hash match (composite key — inherently isolated) ──
            if composite_hash in self._exact_index:
                key = self._exact_index[composite_hash]
                if key in self._store:
                    entry = self._store[key]
                    entry.hit_count += 1
                    self._store.move_to_end(key)
                    self._hits += 1
                    self._exact_hits += 1
                    saved = max(1, len(entry.response) // 4)
                    self._tokens_saved += saved
                    logger.debug("SemanticCache EXACT HIT for query '%.40s...'", query)
                    _report_cache_hit()
                    return entry.response

            # ── Tier 2: MinHash LSH ───────────────────────────────────
            if _DATASKETCH_AVAILABLE and self._minhash_index is not None and len(self._store) > 0:
                minhash = _make_minhash(query)
                try:
                    candidates = self._minhash_index.query(minhash)
                except Exception:
                    candidates = []

                if candidates:
                    # Pick best Jaccard among LSH candidates — skip entries from
                    # a different model or system_prompt (model_id isolation).
                    query_embedding = _trigrams(query)
                    best_score = 0.0
                    best_key: str | None = None
                    for candidate_key in candidates:
                        if candidate_key in self._store:
                            cand_entry = self._store[candidate_key]
                            if cand_entry.model_id != model_id or cand_entry.system_prompt != system_prompt:
                                continue
                            score = _jaccard(query_embedding, cand_entry.embedding)
                            if score > best_score:
                                best_score = score
                                best_key = candidate_key

                    if best_key is not None and best_score >= threshold:
                        entry = self._store[best_key]
                        entry.hit_count += 1
                        self._store.move_to_end(best_key)
                        self._hits += 1
                        self._minhash_hits += 1
                        saved = max(1, len(entry.response) // 4)
                        self._tokens_saved += saved
                        logger.debug(
                            "SemanticCache MINHASH HIT (score=%.3f) for query '%.40s...'",
                            best_score,
                            query,
                        )
                        _report_cache_hit()
                        return entry.response

            # ── Tier 3a: Semantic embedding (when available) ──
            if _get_embedder_available():
                query_dense = _compute_embedding(query)
                if query_dense is not None:
                    best_score = 0.0
                    best_key = None
                    for key, entry in self._store.items():
                        if entry.model_id != model_id or entry.system_prompt != system_prompt:
                            continue
                        if entry.dense_embedding is not None:
                            score = cosine_similarity(query_dense, entry.dense_embedding)
                            if score > best_score:
                                best_score = score
                                best_key = key
                    if best_key is not None and best_score >= threshold:
                        entry = self._store[best_key]
                        entry.hit_count += 1
                        self._store.move_to_end(best_key)
                        self._hits += 1
                        self._semantic_hits += 1
                        saved = max(1, len(entry.response) // 4)
                        self._tokens_saved += saved
                        logger.debug(
                            "SemanticCache SEMANTIC HIT (score=%.3f) for query '%.40s...'",
                            best_score,
                            query,
                        )
                        _report_cache_hit()
                        return entry.response

            # ── Tier 3b: Trigram Jaccard — last resort, always available ──
            query_embedding = _trigrams(query)
            best_score = 0.0
            best_key = None

            for key, entry in self._store.items():
                # Enforce model_id / system_prompt isolation in the trigram tier.
                if entry.model_id != model_id or entry.system_prompt != system_prompt:
                    continue
                score = _jaccard(query_embedding, entry.embedding)
                if score > best_score:
                    best_score = score
                    best_key = key

            if best_key is not None and best_score >= threshold:
                entry = self._store[best_key]
                entry.hit_count += 1
                self._store.move_to_end(best_key)
                self._hits += 1
                self._trigram_hits += 1
                saved = max(1, len(entry.response) // 4)
                self._tokens_saved += saved
                logger.debug(
                    "SemanticCache TRIGRAM HIT (score=%.3f) for query '%.40s...'",
                    best_score,
                    query,
                )
                _report_cache_hit()
                return entry.response

            self._misses += 1
            _report_cache_miss()
            return None

    def put(
        self,
        query: str,
        response: str,
        model_id: str = "",
        system_prompt: str = "",
    ) -> None:
        r"""Store a query/response pair in the cache.

        Inserts into all three tier indices.  If an identical composite key is
        already present the entry is refreshed with the new response and
        moved to the most-recently-used position.

        The composite cache key is ``SHA-256(query + "\\x00" + model_id + "\\x00"
        + system_prompt)`` so entries for different models or system prompts
        are always stored separately and never collide.

        Args:
            query: The query text.
            response: The LLM response to cache.
            model_id: The model that generated ``response``.  Used to
                      isolate cache entries by model — a lookup with a
                      different ``model_id`` will never hit this entry.
            system_prompt: The system prompt active when ``response`` was
                           generated.  Same isolation contract as ``model_id``.
        """
        key_material = f"{query}\x00{model_id}\x00{system_prompt}"
        composite_hash = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
        embedding = _trigrams(query)

        dense_emb = _compute_embedding(query) if _get_embedder_available() else None

        with self._lock:
            self._evict_expired()
            entry = CacheEntry(
                query_hash=composite_hash,
                query_text=query,
                response=response,
                embedding=embedding,
                timestamp=time.monotonic(),
                model_id=model_id,
                system_prompt=system_prompt,
                dense_embedding=dense_emb,
            )
            self._store[composite_hash] = entry
            self._store.move_to_end(composite_hash)

            # Tier 1: exact index (composite_hash → composite_hash)
            self._exact_index[composite_hash] = composite_hash

            # Tier 2: MinHash LSH — keyed by composite_hash so different
            # model_id/system_prompt entries live at distinct LSH keys.
            if _DATASKETCH_AVAILABLE and self._minhash_index is not None:
                minhash = _make_minhash(query)
                try:
                    # Remove stale entry first (LSH insert is not idempotent)
                    if composite_hash in self._minhash_index:
                        self._minhash_index.remove(composite_hash)
                    self._minhash_index.insert(composite_hash, minhash)
                except Exception as exc:
                    logger.warning("MinHash insert failed for key %s: %s", composite_hash[:8], exc)

            self._evict_lru()

    def get_stats(self) -> dict:
        """Return cache statistics including per-tier hit counters.

        Returns:
            Dictionary with keys: ``hit_rate``, ``cache_size``,
            ``estimated_savings`` (token count), ``total_hits``,
            ``total_misses``, ``exact_hits``, ``minhash_hits``,
            ``semantic_hits``, ``trigram_hits``.
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
                "exact_hits": self._exact_hits,
                "minhash_hits": self._minhash_hits,
                "semantic_hits": self._semantic_hits,
                "trigram_hits": self._trigram_hits,
            }

    def clear(self) -> None:
        """Remove all entries and reset statistics."""
        with self._lock:
            self._store.clear()
            self._exact_index.clear()
            if _DATASKETCH_AVAILABLE and self._minhash_index is not None:
                # Re-create the LSH index (no bulk-clear API in datasketch)
                self._minhash_index = MinHashLSH(threshold=_MINHASH_THRESHOLD, num_perm=_MINHASH_NUM_PERM)
            self._hits = 0
            self._misses = 0
            self._tokens_saved = 0
            self._exact_hits = 0
            self._minhash_hits = 0
            self._semantic_hits = 0
            self._trigram_hits = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_expired(self) -> None:
        """Remove expired entries (must be called under lock)."""
        now = time.monotonic()
        expired = [k for k, e in self._store.items() if now - e.timestamp >= self._ttl]
        for k in expired:
            del self._store[k]
            self._exact_index.pop(k, None)
            if _DATASKETCH_AVAILABLE and self._minhash_index is not None:
                try:
                    if k in self._minhash_index:
                        self._minhash_index.remove(k)
                except Exception as exc:
                    logger.warning("MinHash evict-expired remove failed for key %s: %s", k[:8], exc)

    def _evict_lru(self) -> None:
        """Remove least-recently-used entries until within capacity (must be called under lock)."""
        while len(self._store) > self._max_entries:
            key, _ = self._store.popitem(last=False)
            self._exact_index.pop(key, None)
            if _DATASKETCH_AVAILABLE and self._minhash_index is not None:
                try:
                    if key in self._minhash_index:
                        self._minhash_index.remove(key)
                except Exception as exc:
                    logger.warning("MinHash evict-lru remove failed for key %s: %s", key[:8], exc)

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
        |a n b| / |a U b|, or 1.0 if both sets are empty.
    """
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _make_minhash(text: str) -> MinHash:  # type: ignore[name-defined]
    """Create a MinHash from character trigrams of *text*.

    Args:
        text: Input string.

    Returns:
        A ``datasketch.MinHash`` with trigrams as shingles.
    """
    mh = MinHash(num_perm=_MINHASH_NUM_PERM)
    for trigram in _trigrams(text):
        mh.update(trigram.encode("utf-8"))
    return mh


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


def get_semantic_cache(
    ttl: int = _DEFAULT_TTL_SECONDS,
    max_entries: int = CACHE_MAX_ENTRIES_SEMANTIC,
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
