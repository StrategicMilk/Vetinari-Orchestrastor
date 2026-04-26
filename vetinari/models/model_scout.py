"""Model Scout — Online discovery and recommendation loop.

When Thompson Sampling identifies task types where all available models score
poorly, the scout searches for better models using existing ModelDiscovery
adapters (HuggingFace, Reddit, GitHub, PapersWithCode).

Factory analogy: equipment procurement — the factory actively scouts for
better machines when current equipment underperforms.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from datetime import datetime, timezone
from typing import Any

from vetinari.constants import get_user_dir

logger = logging.getLogger(__name__)


def _infer_candidate_format(candidate: Any) -> str:
    identity = f"{getattr(candidate, 'id', '')} {getattr(candidate, 'name', '')}".lower()
    metrics = getattr(candidate, "metrics", {}) or {}
    if isinstance(metrics, dict):
        identity = f"{identity} {' '.join(str(value) for value in metrics.values())}".lower()
    if "gguf" in identity:
        return "gguf"
    if "awq" in identity:
        return "awq"
    if "gptq" in identity:
        return "gptq"
    return "safetensors"


def _backend_for_format(model_format: str) -> str:
    return "llama_cpp" if model_format == "gguf" else "vllm"


@dataclass
class ModelRecommendation:
    """A recommended model from the scout.

    Args:
        model_name: The model name/identifier.
        source: Where it was found (huggingface, reddit, github, etc.).
        task_type: The task type this model excels at.
        estimated_quality: Estimated quality score based on benchmarks/sentiment.
        vram_estimate_gb: Estimated VRAM requirement in GB.
        reason: Why this model was recommended.
        recommended_backend: Backend to use for download/inference.
        recommended_format: Artifact format to download.
    """

    model_name: str = ""
    source: str = ""
    task_type: str = ""
    estimated_quality: float = 0.0
    vram_estimate_gb: float = 0.0
    reason: str = ""
    recommended_backend: str = "vllm"
    recommended_format: str = "safetensors"

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return (
            f"ModelRecommendation(model_name={self.model_name!r},"
            f" task_type={self.task_type!r},"
            f" estimated_quality={self.estimated_quality!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation of the recommendation.
        """
        return {
            "model_name": self.model_name,
            "source": self.source,
            "task_type": self.task_type,
            "estimated_quality": round(self.estimated_quality, 3),
            "vram_estimate_gb": round(self.vram_estimate_gb, 1),
            "reason": self.reason,
            "recommended_backend": self.recommended_backend,
            "recommended_format": self.recommended_format,
        }


class ModelScout:
    """Scouts for better models when current ones underperform.

    Integrates with Thompson Sampling to detect underperformance and
    with ModelDiscovery to search for alternatives.
    """

    UNDERPERFORMANCE_THRESHOLD = 0.5  # Beta mean below this triggers scouting
    MAX_RECOMMENDATIONS = 5

    # Map task types to focused search queries for ModelDiscovery adapters
    # Prefer native vLLM/NIM snapshots; include GGUF for llama.cpp fallbacks.
    SEARCH_QUERIES: dict[str, str] = {
        "coding": "best coding LLM vLLM NIM safetensors AWQ GPTQ local 2026",
        "reasoning": "best reasoning LLM vLLM NIM safetensors AWQ GPTQ local 2026",
        "general": "best general purpose LLM vLLM NIM safetensors AWQ GPTQ local 2026",
        "review": "best code review LLM vLLM NIM safetensors AWQ GPTQ local 2026",
        "architecture": "best architecture design LLM vLLM NIM safetensors AWQ GPTQ local 2026",
    }

    def __init__(self) -> None:
        self._cache: dict[str, list[ModelRecommendation]] = {}
        # Tracks when each task_type was last populated so we can warn on stale hits
        self._cache_timestamps: dict[str, float] = {}
        self._lock = threading.Lock()

    def is_underperforming(self, task_type: str) -> bool:
        """Check if all available models are underperforming for a task type.

        Returns True when all models tracked by Thompson Sampling have a Beta
        distribution mean below UNDERPERFORMANCE_THRESHOLD for the given task
        type.

        Args:
            task_type: The task type to check.

        Returns:
            True if all models are underperforming, False otherwise.
        """
        try:
            from vetinari.learning.model_selector import get_thompson_selector

            selector = get_thompson_selector()
            rankings = selector.get_rankings(task_type)
            if not rankings:
                return False
            return all(mean < self.UNDERPERFORMANCE_THRESHOLD for _, mean in rankings)
        except Exception:
            logger.warning("Cannot check model performance for task_type=%s", task_type, exc_info=True)
            return False

    def _cache_age_s(self, task_type: str) -> float:
        """Return how many seconds ago the cache for task_type was populated.

        Args:
            task_type: The task type key to look up.

        Returns:
            Age in seconds, or 0.0 if no timestamp recorded.
        """
        ts = self._cache_timestamps.get(task_type, 0.0)
        return time.time() - ts if ts > 0 else 0.0

    def scout_for_task(self, task_type: str) -> list[ModelRecommendation]:
        """Search for models that excel at the given task type.

        Uses ModelDiscovery adapters to search HuggingFace, Reddit, GitHub,
        and PapersWithCode. Results are cached for the lifetime of this
        instance (cleared explicitly via clear_cache).

        Args:
            task_type: The task type to search for (e.g., "coding", "reasoning").

        Returns:
            Ranked list of model recommendations, sorted by estimated quality
            descending.
        """
        with self._lock:
            if task_type in self._cache:
                age_s = self._cache_age_s(task_type)
                logger.warning(
                    "Returning stale model scout cache for task_type=%s — cached %.0fs ago",
                    task_type,
                    age_s,
                )
                return self._cache[task_type]

        recommendations: list[ModelRecommendation] = []
        query = self.SEARCH_QUERIES.get(
            task_type,
            f"best {task_type} LLM vLLM NIM safetensors AWQ GPTQ local",
        )

        try:
            from vetinari.model_discovery import ModelDiscovery
            from vetinari.resilience.wiring import call_with_breaker

            discovery = ModelDiscovery()
            candidates = call_with_breaker("model_scout", discovery.search, query)

            for candidate in candidates[: self.MAX_RECOMMENDATIONS]:
                recommended_format = _infer_candidate_format(candidate)
                rec = ModelRecommendation(
                    model_name=candidate.name or candidate.id,
                    source=candidate.source_type,
                    task_type=task_type,
                    estimated_quality=candidate.final_score,
                    vram_estimate_gb=float(candidate.memory_gb),
                    reason=candidate.short_rationale or f"Found via {candidate.source_type}",
                    recommended_backend=_backend_for_format(recommended_format),
                    recommended_format=recommended_format,
                )
                recommendations.append(rec)
        except Exception:
            logger.warning(
                "ModelDiscovery unavailable for task_type=%s, returning empty recommendations",
                task_type,
                exc_info=True,
            )

        # Sort by estimated quality descending before deduplication so the first
        # occurrence of each model_name is always the highest-quality entry.
        recommendations.sort(key=lambda r: r.estimated_quality, reverse=True)

        # Deduplicate by model_name — keep first (highest-quality) occurrence.
        seen: set[str] = set()
        deduplicated: list[ModelRecommendation] = []
        for rec in recommendations:
            if rec.model_name not in seen:
                seen.add(rec.model_name)
                deduplicated.append(rec)

        with self._lock:
            self._cache[task_type] = deduplicated
            self._cache_timestamps[task_type] = time.time()

        return deduplicated

    def get_recommendations(self, task_type: str) -> list[ModelRecommendation]:
        """Get ranked recommendations for a task type.

        Checks the in-memory cache first, then scouts via ModelDiscovery if
        no cached results exist. Returns a defensive copy of each recommendation
        so callers cannot mutate the internal cache state.

        Args:
            task_type: The task type to get recommendations for.

        Returns:
            Ranked list of new ModelRecommendation objects (copies, not references).
        """
        # Defensive copy: callers must not be able to mutate the cached list or
        # the individual recommendation objects — both are returned as fresh copies.
        return [dataclass_replace(r) for r in self.scout_for_task(task_type)]

    def get_status(self) -> dict[str, Any]:
        """Return scout status for health checks.

        Returns:
            Dictionary with ok flag, cached task types, and total recommendation
            count.
        """
        with self._lock:
            return {
                "ok": True,
                "cached_task_types": list(self._cache.keys()),
                "total_recommendations": sum(len(recs) for recs in self._cache.values()),
            }

    def clear_cache(self) -> None:
        """Clear the in-memory recommendation cache.

        Forces fresh ModelDiscovery searches on the next call to
        get_recommendations or scout_for_task.
        """
        with self._lock:
            self._cache.clear()


# ── Model Freshness Checker ──────────────────────────────────────────────────
# Periodic check for newer, better models that have been released since the
# user's current models were downloaded.  Compares against community benchmarks
# (Open LLM Leaderboard, LiveCodeBench) and user sentiment (HuggingFace likes,
# Reddit discussions) to surface genuinely better options.
#
# Runs weekly via the kaizen system or manually via `vetinari check-models`.


@dataclass
class ModelUpgradeCandidate:
    """A model that may be better than what the user currently has.

    Attributes:
        current_model_id: The user's current model that this would replace.
        candidate_name: Name of the potentially better model.
        candidate_repo_id: HuggingFace repo ID for download.
        benchmark_score: Aggregate benchmark score (0.0-1.0, higher is better).
        sentiment_score: Community sentiment score (0.0-1.0, higher is better).
        overall_score: Combined score weighting benchmarks (60%) and sentiment (40%).
        available_formats: Model formats available (gguf, awq, gptq, safetensors).
        recommended_backend: Backend to prefer for the upgrade download.
        recommended_format: Artifact format to prefer for the upgrade download.
        vram_estimate_gb: Estimated VRAM requirement.
        reason: Why this model is recommended as an upgrade.
    """

    current_model_id: str = ""
    candidate_name: str = ""
    candidate_repo_id: str = ""
    benchmark_score: float = 0.0
    sentiment_score: float = 0.0
    overall_score: float = 0.0
    available_formats: list[str] = field(default_factory=list)
    recommended_backend: str = "vllm"
    recommended_format: str = "safetensors"
    vram_estimate_gb: float = 0.0
    reason: str = ""

    def __repr__(self) -> str:
        return (
            f"ModelUpgradeCandidate(candidate={self.candidate_name!r},"
            f" replaces={self.current_model_id!r},"
            f" score={self.overall_score:.3f})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation of the upgrade candidate.
        """
        return {
            "current_model_id": self.current_model_id,
            "candidate_name": self.candidate_name,
            "candidate_repo_id": self.candidate_repo_id,
            "benchmark_score": round(self.benchmark_score, 3),
            "sentiment_score": round(self.sentiment_score, 3),
            "overall_score": round(self.overall_score, 3),
            "available_formats": self.available_formats,
            "recommended_backend": self.recommended_backend,
            "recommended_format": self.recommended_format,
            "vram_estimate_gb": round(self.vram_estimate_gb, 1),
            "reason": self.reason,
        }


# Benchmark weight (60%) vs sentiment weight (40%) for overall scoring
_BENCHMARK_WEIGHT = 0.6
_SENTIMENT_WEIGHT = 0.4

# How many days between automatic freshness checks
FRESHNESS_CHECK_INTERVAL_DAYS = 7


class ModelFreshnessChecker:
    """Periodically checks for newer, better-performing models.

    Compares the user's currently installed models against the latest
    community benchmarks and user sentiment signals.  When a significantly
    better model is found, it surfaces an upgrade suggestion.

    Benchmark sources: Open LLM Leaderboard, LiveCodeBench, HumanEval
    Sentiment sources: HuggingFace likes/downloads, Reddit mentions

    This is step 4 of the kaizen cycle:
    Monitor -> Detect -> **Suggest Upgrade** -> User Decision -> Install
    """

    # Only suggest upgrades with at least this much improvement
    MIN_IMPROVEMENT_THRESHOLD = 0.10  # 10% better than current model

    def __init__(self, vram_budget_gb: float = 32.0) -> None:
        self._vram_budget_gb = vram_budget_gb
        self._last_check_file = get_user_dir() / "last_model_check.json"
        self._lock = threading.Lock()

    def should_check(self) -> bool:
        """Whether it's time for a freshness check (weekly interval).

        Returns:
            True if more than FRESHNESS_CHECK_INTERVAL_DAYS have passed
            since the last check.
        """
        if not self._last_check_file.exists():
            return True

        try:
            data = json.loads(self._last_check_file.read_text(encoding="utf-8"))
            last_check = datetime.fromisoformat(data.get("last_check", "2000-01-01"))
            days_since = (datetime.now(timezone.utc) - last_check).days
            return days_since >= FRESHNESS_CHECK_INTERVAL_DAYS
        except Exception:  # noqa: VET024 — fail-safe: corrupt/missing check file means a fresh check is warranted
            logger.warning("Could not read last freshness check file — assuming check is due")
            return True

    def check_for_upgrades(self) -> list[ModelUpgradeCandidate]:
        """Check for better models than what the user currently has.

        Queries ModelDiscovery for the latest models, scores them against
        benchmarks and community sentiment, and returns upgrade candidates
        that meaningfully outperform the current models.

        Returns:
            List of ModelUpgradeCandidate objects, sorted by overall score
            descending.  Empty if no upgrades are available.
        """
        with self._lock:
            candidates: list[ModelUpgradeCandidate] = []

            # Get current models from the adapter
            current_models = self._get_current_models()
            if not current_models:
                logger.info("No models currently installed — skipping freshness check")
                return []

            # Search for latest models across all formats
            for task_type in ("coding", "reasoning", "general"):
                task_candidates = self._find_upgrades_for_task(task_type, current_models)
                candidates.extend(task_candidates)

            # Record this check and persist candidates
            self._record_check(len(candidates), candidates)

            candidates.sort(key=lambda c: c.overall_score, reverse=True)
            if candidates:
                logger.info(
                    "Model freshness check found %d potential upgrade(s)",
                    len(candidates),
                )
            return candidates

    def _get_current_models(self) -> dict[str, dict[str, Any]]:
        """Get dict of currently installed models with their performance data.

        Returns:
            Dict mapping model_id to performance metadata.
        """
        try:
            from vetinari.learning.model_selector import get_thompson_selector

            selector = get_thompson_selector()
            result: dict[str, dict[str, Any]] = {}

            for task_type in ("coding", "reasoning", "general"):
                rankings = selector.get_rankings(task_type)
                for model_id, mean_score in rankings:
                    if model_id not in result:
                        result[model_id] = {
                            "model_id": model_id,
                            "best_task": task_type,
                            "best_score": mean_score,
                            "tasks": {},
                        }
                    result[model_id]["tasks"][task_type] = mean_score
            return result
        except Exception:
            logger.warning(
                "Thompson selector unavailable for freshness check — current model scores not factored into upgrade comparison"
            )
            return {}

    def _find_upgrades_for_task(
        self,
        task_type: str,
        current_models: dict[str, dict[str, Any]],
    ) -> list[ModelUpgradeCandidate]:
        """Search for models that outperform current models for a task type.

        Uses ModelDiscovery to find candidates, then scores them against
        benchmarks and sentiment.  Only returns candidates that exceed
        the improvement threshold.

        Args:
            task_type: Task type to search for (e.g. "coding").
            current_models: Dict of current models with performance scores.

        Returns:
            List of upgrade candidates for this task type.
        """
        # Find the best current model for this task
        best_current_score = 0.0
        best_current_id = ""
        for model_id, info in current_models.items():
            task_score = info.get("tasks", {}).get(task_type, 0.0)
            if task_score > best_current_score:
                best_current_score = task_score
                best_current_id = model_id

        if not best_current_id:
            return []

        # Search for newer models
        query = f"best {task_type} LLM 2026 benchmark vLLM NIM safetensors AWQ GPTQ"
        try:
            from vetinari.model_discovery import ModelDiscovery
            from vetinari.resilience.wiring import call_with_breaker

            discovery = ModelDiscovery()
            raw_candidates = call_with_breaker("model_scout", discovery.search, query)
        except Exception:
            logger.warning("ModelDiscovery unavailable for freshness check — no upgrade candidates found")
            return []

        upgrades: list[ModelUpgradeCandidate] = []

        for candidate in raw_candidates[:10]:
            # Skip if it's already one of our models
            candidate_name = candidate.name or candidate.id
            if any(candidate_name.lower() in mid.lower() for mid in current_models):
                continue

            # Skip if too large for our VRAM budget (with offload headroom)
            vram_needed = float(candidate.memory_gb)
            if vram_needed > self._vram_budget_gb * 2:
                continue

            # Score against benchmarks and sentiment
            benchmark_score = self._estimate_benchmark_score(candidate)
            sentiment_score = self._estimate_sentiment_score(candidate)
            overall_score = benchmark_score * _BENCHMARK_WEIGHT + sentiment_score * _SENTIMENT_WEIGHT

            # Only suggest if meaningfully better
            improvement = overall_score - best_current_score
            if improvement < self.MIN_IMPROVEMENT_THRESHOLD:
                continue

            # Determine available formats
            formats = self._detect_available_formats(candidate)
            recommended_format = next(
                (fmt for fmt in ("awq", "gptq", "safetensors", "gguf") if fmt in formats),
                "safetensors",
            )

            upgrades.append(
                ModelUpgradeCandidate(
                    current_model_id=best_current_id,
                    candidate_name=candidate_name,
                    candidate_repo_id=getattr(candidate, "repo_id", ""),
                    benchmark_score=benchmark_score,
                    sentiment_score=sentiment_score,
                    overall_score=overall_score,
                    available_formats=formats,
                    recommended_backend=_backend_for_format(recommended_format),
                    recommended_format=recommended_format,
                    vram_estimate_gb=vram_needed,
                    reason=(
                        f"Scores {improvement:.0%} higher than {best_current_id} "
                        f"for {task_type} tasks (benchmark={benchmark_score:.2f}, "
                        f"sentiment={sentiment_score:.2f})"
                    ),
                )
            )

        return upgrades

    def _estimate_benchmark_score(self, candidate: Any) -> float:
        """Estimate benchmark quality from candidate metadata.

        Uses available signals: final_score from discovery, parameter count
        heuristics, and any benchmark data in the metadata.

        Args:
            candidate: A ModelCandidate from ModelDiscovery.

        Returns:
            Score between 0.0 and 1.0.
        """
        score = getattr(candidate, "final_score", 0.0)

        # Boost for known high-quality families
        name_lower = (candidate.name or candidate.id or "").lower()
        if any(fam in name_lower for fam in ("qwen3", "qwen2.5", "llama-3.3", "llama-3.1")):
            score = max(score, 0.6)
        if any(fam in name_lower for fam in ("deepseek-v3", "mistral-large")):
            score = max(score, 0.7)

        return min(1.0, score)

    def _estimate_sentiment_score(self, candidate: Any) -> float:
        """Estimate community sentiment from download counts and engagement.

        Args:
            candidate: A ModelCandidate from ModelDiscovery.

        Returns:
            Score between 0.0 and 1.0.
        """
        score = 0.3  # Base score — if we found it, someone uses it

        # Use likes/downloads if available
        likes = getattr(candidate, "likes", 0) or 0
        downloads = getattr(candidate, "downloads", 0) or 0

        if likes > 1000:
            score += 0.3
        elif likes > 100:
            score += 0.15
        elif likes > 10:
            score += 0.05

        if downloads > 100000:
            score += 0.3
        elif downloads > 10000:
            score += 0.15
        elif downloads > 1000:
            score += 0.05

        return min(1.0, score)

    def _detect_available_formats(self, candidate: Any) -> list[str]:
        """Detect which model formats are available for a candidate.

        Args:
            candidate: A ModelCandidate from ModelDiscovery.

        Returns:
            List of available format strings (e.g. ["gguf", "awq"]).
        """
        name_lower = (candidate.name or candidate.id or "").lower()
        metrics = getattr(candidate, "metrics", {}) or {}
        if isinstance(metrics, dict):
            name_lower = f"{name_lower} {' '.join(str(value) for value in metrics.values())}".lower()
        formats = []

        if "safetensors" in name_lower:
            formats.append("safetensors")
        if "gguf" in name_lower:
            formats.append("gguf")
        if "awq" in name_lower:
            formats.append("awq")
        if "gptq" in name_lower:
            formats.append("gptq")

        # Most popular models have GGUF and SafeTensors variants
        if not formats:
            formats = ["safetensors", "gguf"]

        return formats

    def get_cached_upgrades(self) -> list[ModelUpgradeCandidate]:
        """Load upgrade candidates from the last freshness check.

        Returns:
            List of cached upgrade candidates, or empty if no check has run.
        """
        upgrades_file = self._last_check_file.parent / "model_upgrades.json"
        if not upgrades_file.exists():
            return []
        try:
            data = json.loads(upgrades_file.read_text(encoding="utf-8"))
            return [ModelUpgradeCandidate(**entry) for entry in data.get("upgrades", [])]
        except Exception:
            logger.warning("Could not load cached upgrade candidates — returning empty list")
            return []

    def _record_check(self, candidates_found: int, upgrades: list[ModelUpgradeCandidate] | None = None) -> None:
        """Record the freshness check timestamp and any upgrade candidates.

        Writes both the check metadata and the upgrade candidates to disk
        so they can be surfaced by API endpoints or CLI commands.

        Args:
            candidates_found: Number of upgrade candidates found.
            upgrades: The actual upgrade candidates to persist.
        """
        try:
            self._last_check_file.parent.mkdir(parents=True, exist_ok=True)
            self._last_check_file.write_text(
                json.dumps({
                    "last_check": datetime.now(timezone.utc).isoformat(),
                    "candidates_found": candidates_found,
                }),
                encoding="utf-8",
            )

            # Persist upgrade candidates for API/CLI consumption
            if upgrades:
                upgrades_file = self._last_check_file.parent / "model_upgrades.json"
                upgrades_file.write_text(
                    json.dumps({
                        "checked_at": datetime.now(timezone.utc).isoformat(),
                        "upgrades": [u.to_dict() for u in upgrades],
                    }),
                    encoding="utf-8",
                )
        except Exception:
            logger.warning(
                "Could not write freshness check results — upgrade candidates will not persist between sessions"
            )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_model_scout: ModelScout | None = None
_scout_lock = threading.Lock()


def get_model_scout() -> ModelScout:
    """Return the singleton ModelScout instance (thread-safe, lazy init).

    Returns:
        The shared ModelScout instance.
    """
    global _model_scout
    if _model_scout is None:
        with _scout_lock:
            if _model_scout is None:
                _model_scout = ModelScout()
    return _model_scout


def reset_model_scout() -> None:
    """Reset the singleton ModelScout for testing.

    After calling this, the next call to get_model_scout creates a fresh
    instance. Only intended for use in test teardown.
    """
    global _model_scout
    with _scout_lock:
        _model_scout = None
