"""Plan caching layer — reuse past plans for similar goals."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CachedPlan:
    """Cached plan."""
    goal: str
    goal_hash: str
    plan_data: dict[str, Any]
    created_at: float
    hit_count: int = 0
    last_hit: float = 0.0
    quality_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CachedPlan:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class PlanCache:
    """Cache past plans keyed by goal similarity.

    30s planning -> 300ms cache hit on similar goals.
    Uses keyword overlap for similarity matching (no embedding model needed).
    """

    DEFAULT_CACHE_DIR = ".vetinari/plan_cache"
    MAX_CACHE_SIZE = 100
    DEFAULT_THRESHOLD = 0.6
    DEFAULT_MAX_AGE_DAYS = 30

    def __init__(self, cache_dir: str | None = None):
        self._cache_dir = Path(cache_dir or self.DEFAULT_CACHE_DIR)
        self._cache: dict[str, CachedPlan] = {}
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self._load_cache()
            self._loaded = True

    def _load_cache(self):
        cache_file = self._cache_dir / "plans.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                for entry in data:
                    plan = CachedPlan.from_dict(entry)
                    self._cache[plan.goal_hash] = plan
            except Exception as e:
                logger.debug("Plan cache load error: %s", e)

    def _save_cache(self):
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self._cache_dir / "plans.json"
            data = [p.to_dict() for p in self._cache.values()]
            cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug("Plan cache save error: %s", e)

    def _goal_hash(self, goal: str) -> str:
        normalized = " ".join(goal.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _extract_keywords(self, text: str) -> set:
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "shall",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "and",
            "or",
            "but",
            "not",
            "no",
            "if",
            "then",
            "than",
            "that",
            "this",
            "it",
            "its",
            "my",
            "your",
            "we",
            "they",
            "i",
            "me",
        }
        words = set(text.lower().split())
        return words - stop_words

    def _similarity(self, goal_a: str, goal_b: str) -> float:
        kw_a = self._extract_keywords(goal_a)
        kw_b = self._extract_keywords(goal_b)
        if not kw_a or not kw_b:
            return 0.0
        intersection = kw_a & kw_b
        union = kw_a | kw_b
        return len(intersection) / len(union) if union else 0.0

    def find_similar(self, goal: str, threshold: float | None = None) -> CachedPlan | None:
        """Find a cached plan similar to the given goal.

        Args:
            goal: The goal.
            threshold: The threshold.

        Returns:
            The CachedPlan | None result.
        """
        self._ensure_loaded()
        threshold = threshold or self.DEFAULT_THRESHOLD

        # Exact match first
        goal_hash = self._goal_hash(goal)
        if goal_hash in self._cache:
            plan = self._cache[goal_hash]
            plan.hit_count += 1
            plan.last_hit = time.time()
            return plan

        # Similarity search
        best_plan = None
        best_score = 0.0

        for plan in self._cache.values():
            score = self._similarity(goal, plan.goal)
            if score > best_score and score >= threshold:
                best_score = score
                best_plan = plan

        if best_plan:
            best_plan.hit_count += 1
            best_plan.last_hit = time.time()
            logger.info("Plan cache hit (similarity=%.2f)", best_score)

        return best_plan

    def store(self, goal: str, plan_data: dict[str, Any], quality_score: float = 0.0) -> None:
        """Store a plan in the cache.

        Args:
            goal: The goal.
            plan_data: The plan data.
            quality_score: The quality score.
        """
        self._ensure_loaded()

        goal_hash = self._goal_hash(goal)
        self._cache[goal_hash] = CachedPlan(
            goal=goal,
            goal_hash=goal_hash,
            plan_data=plan_data,
            created_at=time.time(),
            quality_score=quality_score,
        )

        # Evict oldest if over limit
        if len(self._cache) > self.MAX_CACHE_SIZE:
            oldest_key = min(self._cache, key=lambda k: self._cache[k].last_hit or self._cache[k].created_at)
            del self._cache[oldest_key]

        self._save_cache()

    def invalidate(self, older_than_days: int | None = None) -> int:
        """Remove stale entries. Returns count of removed entries.

        Returns:
            The computed value.
        """
        self._ensure_loaded()
        older_than_days = older_than_days or self.DEFAULT_MAX_AGE_DAYS
        cutoff = time.time() - (older_than_days * 86400)

        to_remove = [k for k, v in self._cache.items() if v.created_at < cutoff]
        for k in to_remove:
            del self._cache[k]

        if to_remove:
            self._save_cache()

        return len(to_remove)

    def get_stats(self) -> dict[str, Any]:
        """Get stats.

        Returns:
            The result string.
        """
        self._ensure_loaded()
        return {
            "total_cached": len(self._cache),
            "total_hits": sum(p.hit_count for p in self._cache.values()),
            "avg_quality": (sum(p.quality_score for p in self._cache.values()) / max(len(self._cache), 1)),
        }


_plan_cache: PlanCache | None = None


def get_plan_cache(cache_dir: str | None = None) -> PlanCache:
    """Get plan cache.

    Returns:
        The PlanCache result.
    """
    global _plan_cache
    if _plan_cache is None:
        _plan_cache = PlanCache(cache_dir)
    return _plan_cache
