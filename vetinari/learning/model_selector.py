"""Thompson Sampling Model Selector - Vetinari Self-Improvement Subsystem.

Implements Bayesian bandit-style model selection that naturally balances
exploration (trying less-used models) with exploitation (using proven ones).

Each model+task_type pair maintains a Beta distribution:
  - alpha = successes (weighted by quality scores)
  - beta  = failures

When selecting a model, we sample from each distribution and pick the highest.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from vetinari.learning.thompson_arms import ThompsonBetaArm, ThompsonTaskContext
from vetinari.learning.thompson_persistence import prune_stale_arms

logger = logging.getLogger(__name__)

# Alias for backward compatibility with tests and old code
BetaArm = ThompsonBetaArm

__all__ = [
    "BetaArm",
    "ThompsonBetaArm",
    "ThompsonSamplingSelector",
    "ThompsonTaskContext",
    "get_model_selector",
    "reset_thompson_selector",
]

_MODEL_ARM_KEY_PREFIX = "model-json:"
_STRUCTURED_ARM_PREFIXES = ("strategy:", "mode_", "tier_", "ctx_")


def _make_arm_key(model_id: str, task_type: str) -> str:
    """Return a persistence key that does not alias colon-bearing model IDs."""
    if (":" in model_id or ":" in task_type) and not model_id.startswith(_STRUCTURED_ARM_PREFIXES):
        return _MODEL_ARM_KEY_PREFIX + json.dumps([model_id, task_type], separators=(",", ":"), ensure_ascii=True)
    return f"{model_id}:{task_type}"


def _parse_arm_key(key: str) -> tuple[str, str] | None:
    """Parse a persisted arm key, including the colon-safe JSON format."""
    if key.startswith(_MODEL_ARM_KEY_PREFIX):
        try:
            model_id, task_type = json.loads(key[len(_MODEL_ARM_KEY_PREFIX) :])
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Ignoring malformed Thompson arm key: %r", key)
            return None
        return str(model_id), str(task_type)

    if ":" not in key:
        return None
    model_id, task_type = key.rsplit(":", 1)
    return model_id, task_type


@dataclass
class TaskContext:
    """Features that inform model selection — the "context" in contextual bandit.

    Args:
        task_type: Type of task (code, research, architecture, review, etc.).
        estimated_complexity: Complexity rating 1-10 from intake.
        prompt_length: Token count in the task description.
        domain: Domain (python, javascript, infrastructure, etc.).
        requires_reasoning: Whether multi-step logic is needed.
        requires_creativity: Whether open-ended generation is needed.
        requires_precision: Whether exact syntax/structured output is needed.
        file_count: Number of files in scope.
    """

    task_type: str = "general"
    estimated_complexity: int = 5
    prompt_length: int = 0
    domain: str = "general"
    requires_reasoning: bool = False
    requires_creativity: bool = False
    requires_precision: bool = False
    file_count: int = 0

    def __repr__(self) -> str:
        return f"TaskContext(task_type={self.task_type!r}, estimated_complexity={self.estimated_complexity!r}, domain={self.domain!r})"

    def to_bucket(self) -> int:
        """Hash context features into a bucket for Thompson arm lookup.

        Returns ~50 buckets. Enough signal to differentiate "coding+simple"
        from "coding+complex" without making arms too sparse.  # noqa: VET070

        Returns:
            Bucket index (0-49).
        """
        complexity_bin = "lo" if self.estimated_complexity <= 3 else ("mid" if self.estimated_complexity <= 7 else "hi")
        key = f"{self.task_type}:{complexity_bin}:{self.domain}:{self.requires_reasoning}"
        # Use hashlib for deterministic hashing across process restarts.
        # Python's built-in hash() is randomised by PYTHONHASHSEED, so two
        # processes with the same context key would map to different buckets,
        # breaking cross-restart arm persistence.
        return int(hashlib.md5(key.encode(), usedforsecurity=False).hexdigest(), 16) % 50


class ThompsonSamplingSelector:
    """Thompson Sampling model selector for exploration-exploitation balance.

    Automatically:
    - Explores new/untested models (slightly skeptical Beta(2,2) prior)
    - Exploits well-performing models for their appropriate task types
    - Incorporates cost as a penalty on expected return
    - Persists arm states to survive restarts
    """

    COST_WEIGHT = 0.15  # How much to penalize expensive models
    MIN_ARMS = 1
    MAX_ARMS = 500  # Cap on total arms to prevent unbounded memory growth

    # Cold-start priors: empty dict because _get_or_create_arm() uses
    # BenchmarkSeeder._get_informed_prior() for real models on first observation.
    # Previous cloud API model names (claude-sonnet-4-20250514 etc.) never matched
    # local GGUF model IDs, creating phantom arms. (Decision: SESSION-2-M1 fix 14)
    BENCHMARK_PRIORS: dict[str, tuple[float, float]] = {}

    PERIODIC_SAVE_INTERVAL = 25  # Save state to disk every N observations (frequent saves prevent data loss on crash)

    def __init__(self):
        # Key: "model_id:task_type" for legacy/simple IDs, or a JSON tuple
        # for colon-bearing model IDs.
        self._arms: dict[str, BetaArm] = {}
        self._lock = threading.Lock()
        self._update_count: int = 0  # Observations since last periodic save
        self._load_state()
        with self._lock:
            pruned = prune_stale_arms(self._arms)
        if pruned:
            logger.debug("Pruned %d stale Thompson arms at startup", pruned)
        self._seed_from_benchmarks()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_model(
        self,
        task_type: str,
        candidate_models: list[str],
        cost_per_model: dict[str, float] | None = None,
    ) -> str:
        """Select the best model for a task type using Thompson Sampling.

        Args:
            task_type: Task type string.
            candidate_models: List of available model IDs.
            cost_per_model: Optional cost estimates per model (higher cost = penalty).

        Returns:
            Selected model ID, or "default" when candidates is empty.
        """
        with self._lock:
            if not candidate_models:
                return "default"

            cost_per_model = cost_per_model or {}  # noqa: VET112 - empty fallback preserves optional request metadata contract
            max_cost = max(cost_per_model.values(), default=1.0)
            if max_cost == 0.0:
                max_cost = 1.0  # Avoid division by zero in cost normalization

            best_model = candidate_models[0]
            best_score = -1.0

            for model_id in candidate_models:
                arm = self._get_or_create_arm(model_id, task_type)
                sampled = arm.sample()

                # Additive penalty preserves exploration properties (multiplicative distorts Beta)
                cost = cost_per_model.get(model_id, max_cost * 0.5)
                adjusted = sampled - self.COST_WEIGHT * (cost / max_cost)

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "[Thompson] %s/%s: sample=%.3f adjusted=%.3f (alpha=%.1f, beta=%.1f)",
                        model_id,
                        task_type,
                        sampled,
                        adjusted,
                        arm.alpha,
                        arm.beta,
                    )

                if adjusted > best_score:
                    best_score = adjusted
                    best_model = model_id

            logger.info("[Thompson] Selected %s for %s (score=%.3f)", best_model, task_type, best_score)
            if logger.isEnabledFor(logging.DEBUG):
                try:
                    from vetinari.optimization.semantic_cache import get_threshold_for_task_type

                    _threshold = get_threshold_for_task_type(task_type)
                    logger.debug(
                        "[Thompson] Semantic cache threshold for %s: %.2f",
                        task_type,
                        _threshold,
                    )
                except (ImportError, AttributeError, KeyError):
                    logger.debug("[Thompson] Could not load cache threshold for %s — skipping debug log", task_type)
            return best_model

    def update(
        self,
        model_id: str,
        task_type: str,
        quality_score: float,
        success: bool,
    ) -> None:
        """Update a model arm after observing an outcome; auto-saves every PERIODIC_SAVE_INTERVAL updates.

        Args:
            model_id: The model that was used.
            task_type: The task type.
            quality_score: Observed quality score 0.0-1.0.
            success: Whether the task succeeded.
        """
        with self._lock:
            arm = self._get_or_create_arm(model_id, task_type)
            arm.update(quality_score, success)
            self._update_count += 1
            if self._update_count % self.PERIODIC_SAVE_INTERVAL == 0:
                self._save_state()
                logger.info(
                    "[Thompson] Periodic state save after %d observations",
                    self._update_count,
                )

    # Benchmark results are 3x more reliable than single-task outcomes
    BENCHMARK_WEIGHT_MULTIPLIER = 3

    def update_from_benchmark(
        self,
        model_id: str,
        pass_rate: float,
        n_trials: int,
        task_type: str = "general",
    ) -> None:
        """Update arms from benchmark results (weighted 3x vs single-task observations).

        Args:
            model_id: The model that was benchmarked.
            pass_rate: Fraction of benchmark cases passed (0.0-1.0).
            n_trials: Number of benchmark trials run.
            task_type: Task type the benchmark covers.
        """
        pass_rate = max(0.0, min(1.0, pass_rate))  # Clamp to prevent Beta corruption

        with self._lock:
            arm = self._get_or_create_arm(model_id, task_type)
            w = self.BENCHMARK_WEIGHT_MULTIPLIER

            successes = pass_rate * n_trials
            failures = n_trials - successes

            arm.alpha += successes * w
            arm.beta += failures * w
            arm.total_pulls += n_trials
            arm.last_updated = datetime.now(timezone.utc).isoformat()

        logger.info(
            "[Thompson] Benchmark update for %s/%s: "
            "pass_rate=%.3f, n_trials=%d, "
            "weighted_successes=%.1f, weighted_failures=%.1f, "
            "new_mean=%.3f",
            model_id,
            task_type,
            pass_rate,
            n_trials,
            successes * w,
            failures * w,
            arm.mean,
        )
        self._save_state()

    def get_rankings(self, task_type: str) -> list[tuple[str, float]]:
        """Return (model_id, mean) tuples for all arms matching task_type, sorted by expected value.

        Args:
            task_type: The task type string to filter arms by.

        Returns:
            List sorted from highest to lowest expected value; empty when no arms exist.
        """
        arms = [(arm.model_id, arm.mean) for arm in self._arms.values() if arm.task_type == task_type]
        arms.sort(key=lambda x: x[1], reverse=True)
        return arms

    # ------------------------------------------------------------------
    # Tier selection for intake routing (Department 3)
    # ------------------------------------------------------------------

    TIER_MIN_PULLS = 10  # See model_selector_tiers.TIER_MIN_PULLS

    def has_sufficient_data(self, pattern_key: str) -> bool:
        """Check if enough tier data exists to override rule-based routing. See ``model_selector_tiers``.

        Returns:
            True when all arms for the pattern key have been pulled at least TIER_MIN_PULLS times.
        """
        from vetinari.learning.model_selector_tiers import has_sufficient_data as _has_sufficient_data

        return _has_sufficient_data(self, pattern_key)

    def select_tier(self, pattern_key: str) -> str:
        """Select best tier via Thompson Sampling. See ``model_selector_tiers``.

        Returns:
            The tier identifier (e.g. "fast", "balanced", "quality") with the highest sampled reward.
        """
        from vetinari.learning.model_selector_tiers import select_tier as _select_tier

        return _select_tier(self, pattern_key)

    def update_tier(self, pattern_key: str, tier_used: str, quality_score: float, rework_count: int = 0) -> None:
        """Update tier arm after task completion. See ``model_selector_tiers``.

        Args:
            pattern_key: The request pattern key identifying the tier bandit context.
            tier_used: The tier name that was actually used (e.g. "fast", "quality").
            quality_score: Observed quality score for the completed task, 0.0-1.0.
            rework_count: Number of rework iterations required; used to penalise the reward.
        """
        from vetinari.learning.model_selector_tiers import update_tier as _update_tier

        _update_tier(self, pattern_key, tier_used, quality_score, rework_count)

    def get_arm_state(self, model_id: str, task_type: str) -> dict[str, Any]:
        """Return current Beta distribution state for an arm. See ``model_selector_tiers``.

        Args:
            model_id: Identifier of the model whose arm state to retrieve.
            task_type: Task domain for the arm (e.g. "coding", "summarisation").

        Returns:
            Dict with alpha, beta, n_pulls, and mean_reward for the specified model/task arm.
        """
        from vetinari.learning.model_selector_tiers import get_arm_state as _get_arm_state

        return _get_arm_state(self, model_id, task_type)

    # ------------------------------------------------------------------
    # Mode selection for multi-mode agents (Department 6, connection #77)
    # ------------------------------------------------------------------

    def select_mode(
        self,
        agent_type: str,
        task_type: str,
        candidate_modes: list[str],
    ) -> str:
        """Select best agent mode via Thompson Sampling. See ``model_selector_contextual``.

        Args:
            agent_type: The agent performing the task (e.g. "worker", "inspector").
            task_type: Task domain used to scope the bandit arms.
            candidate_modes: List of mode names to choose among (e.g. ["fast", "deep"]).

        Returns:
            The mode name from candidate_modes with the highest sampled reward.
        """
        from vetinari.learning.model_selector_contextual import select_mode as _select_mode

        return _select_mode(self, agent_type, task_type, candidate_modes)

    def update_mode(
        self,
        agent_type: str,
        task_type: str,
        mode: str,
        quality_score: float,
        success: bool,
    ) -> None:
        """Update mode arm after observing an outcome. See ``model_selector_contextual``.

        Args:
            agent_type: The agent whose mode arm to update.
            task_type: Task domain used to scope the bandit arms.
            mode: The mode name that was actually used.
            quality_score: Observed quality score for the completed task, 0.0-1.0.
            success: Whether the task completed without rework or failure.
        """
        from vetinari.learning.model_selector_contextual import update_mode as _update_mode

        _update_mode(self, agent_type, task_type, mode, quality_score, success)

    def has_mode_data(self, agent_type: str, task_type: str) -> bool:
        """Check if sufficient mode data exists for Thompson override. See ``model_selector_contextual``.

        Args:
            agent_type: The agent type to check mode data for.
            task_type: Task domain to scope the arm lookup.

        Returns:
            True when all candidate mode arms for this agent/task pair have been pulled enough times.
        """
        from vetinari.learning.model_selector_contextual import has_mode_data as _has_mode_data

        return _has_mode_data(self, agent_type, task_type)

    def select_strategy(
        self,
        agent_type: str,
        mode: str,
        strategy_key: str,
    ) -> str | int | float:
        """Select best strategy value via Thompson Sampling. See ``model_selector_contextual``.

        Args:
            agent_type: The agent whose strategy to select.
            mode: Current execution mode scoping this strategy decision.
            strategy_key: The strategy parameter name (e.g. "temperature_bucket", "depth").

        Returns:
            The strategy value (string, int, or float) with the highest sampled reward for this key.
        """
        from vetinari.learning.model_selector_contextual import select_strategy as _select_strategy

        return _select_strategy(self, agent_type, mode, strategy_key)

    def update_strategy(
        self,
        agent_type: str,
        mode: str,
        strategy_key: str,
        value: str | float,
        quality_score: float,
    ) -> None:
        """Update a strategy arm after observing an outcome. See ``model_selector_contextual``.

        Args:
            agent_type: The agent whose strategy arm to update.
            mode: Execution mode scoping this strategy arm.
            strategy_key: The strategy parameter name being updated.
            value: The actual value that was used in this execution.
            quality_score: Observed quality score for the outcome, 0.0-1.0.
        """
        from vetinari.learning.model_selector_contextual import update_strategy as _update_strategy

        _update_strategy(self, agent_type, mode, strategy_key, value, quality_score)

    DECAY_FACTOR = 0.995  # See model_selector_contextual.DECAY_FACTOR

    def select_model_contextual(
        self,
        task_context: TaskContext,
        candidate_models: list[str],
        cost_per_model: dict[str, float] | None = None,
    ) -> str:
        """Select best model via context-aware Thompson Sampling. See ``model_selector_contextual``.

        Args:
            task_context: Structured context describing the current task (type, complexity, etc.).
            candidate_models: List of model_ids to select among.
            cost_per_model: Optional mapping of model_id to relative cost weight for penalisation.

        Returns:
            The model_id from candidate_models with the highest context-adjusted sampled reward.
        """
        from vetinari.learning.model_selector_contextual import select_model_contextual as _select_contextual

        return _select_contextual(self, task_context, candidate_models, cost_per_model)

    def update_contextual(
        self,
        task_context: TaskContext,
        model_id: str,
        quality_score: float,
        success: bool,
    ) -> None:
        """Update contextual arm with exponential decay. See ``model_selector_contextual``.

        Args:
            task_context: Structured context describing the completed task.
            model_id: The model_id that was actually used for this task.
            quality_score: Observed quality score for the outcome, 0.0-1.0.
            success: Whether the task completed without rework or failure.
        """
        from vetinari.learning.model_selector_contextual import update_contextual as _update_contextual

        _update_contextual(self, task_context, model_id, quality_score, success)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_create_arm(self, model_id: str, task_type: str) -> BetaArm:
        """Return existing arm or create one; evicts LRU when at MAX_ARMS capacity."""
        key = self._arm_key(model_id, task_type)
        if key not in self._arms:
            # Evict LRU arm if at capacity
            if len(self._arms) >= self.MAX_ARMS:
                self._evict_lru_arm()
            alpha, beta = self._get_informed_prior(model_id, task_type)
            self._arms[key] = BetaArm(
                model_id=model_id,
                task_type=task_type,
                alpha=alpha,
                beta=beta,
            )
        return self._arms[key]

    @staticmethod
    def _arm_key(model_id: str, task_type: str) -> str:
        """Return the stable key for a model/task arm."""
        return _make_arm_key(model_id, task_type)

    def _evict_lru_arm(self) -> None:
        """Remove the arm with the oldest last_updated timestamp."""
        if not self._arms:
            return
        lru_key = min(self._arms, key=lambda k: self._arms[k].last_updated)
        logger.debug("[Thompson] Evicting LRU arm %s to stay within MAX_ARMS=%d", lru_key, self.MAX_ARMS)
        del self._arms[lru_key]

    def _seed_from_benchmarks(self) -> None:
        """Seed arms from BENCHMARK_PRIORS on cold start; skipped if arms already exist."""
        if self._arms:
            return  # Already have state — don't overwrite with priors
        seeded = 0
        for key, (alpha, beta) in self.BENCHMARK_PRIORS.items():
            if key not in self._arms:
                parts = _parse_arm_key(key)
                if parts is not None:
                    self._arms[key] = BetaArm(
                        model_id=parts[0],
                        task_type=parts[1],
                        alpha=alpha,
                        beta=beta,
                    )
                    seeded += 1
        if seeded:
            logger.info("[Thompson] Cold-start: seeded %d arms from BENCHMARK_PRIORS", seeded)

    def _get_informed_prior(self, model_id: str, task_type: str) -> tuple:
        """Get informed prior from BenchmarkSeeder, fallback to Beta(1,1)."""
        try:
            from vetinari.learning.benchmark_seeder import get_benchmark_seeder

            return get_benchmark_seeder().get_prior(model_id, task_type)
        except Exception:
            logger.warning("BenchmarkSeeder unavailable for %s:%s, using uninformed prior", model_id, task_type)
            return (1.0, 1.0)

    def _get_state_dir(self) -> str:
        """Return .vetinari state dir path. See ``model_selector_persistence``."""
        from vetinari.learning.model_selector_persistence import get_state_dir

        return get_state_dir(self)

    def _load_state(self) -> None:
        """Load arm states from SQLite, falling back to JSON, then prune stale arms."""
        from vetinari.learning.model_selector_persistence import load_state
        from vetinari.learning.thompson_persistence import prune_stale_arms

        load_state(self)
        prune_stale_arms(self._arms)

    def _load_state_from_db(self) -> int:
        """Load arm states from SQLite table. See ``model_selector_persistence``."""
        from vetinari.learning.model_selector_persistence import load_state_from_db

        return load_state_from_db(self)

    def _migrate_from_json(self) -> None:
        """One-time migration from legacy JSON. See ``model_selector_persistence``."""
        from vetinari.learning.model_selector_persistence import migrate_from_json

        migrate_from_json(self)

    def _save_state(self) -> None:
        """Persist arm states to SQLite with JSON fallback. See ``model_selector_persistence``."""
        from vetinari.learning.model_selector_persistence import save_state

        save_state(self)


# Singleton
_thompson_selector: ThompsonSamplingSelector | None = None
_thompson_selector_lock = threading.Lock()
_thompson_selector_save_callback: Any | None = None


def get_thompson_selector() -> ThompsonSamplingSelector:
    """Return the singleton ThompsonSamplingSelector instance (thread-safe).

    Registers atexit handler and shutdown callback on first creation
    to persist Thompson state on process exit.

    Returns:
        The shared ThompsonSamplingSelector instance.
    """
    global _thompson_selector, _thompson_selector_save_callback
    if _thompson_selector is None:
        with _thompson_selector_lock:
            if _thompson_selector is None:
                _thompson_selector = ThompsonSamplingSelector()
                _thompson_selector_save_callback = _thompson_selector._save_state
                # Register atexit handler for graceful persistence
                import atexit

                atexit.register(_thompson_selector_save_callback)
                # Also register with shutdown.py callback system
                try:
                    from vetinari.shutdown import register_callback

                    register_callback("Thompson Sampling state", _thompson_selector_save_callback)
                except Exception:
                    logger.warning("shutdown.py unavailable, atexit-only Thompson persistence")
    return _thompson_selector


def get_model_selector() -> ThompsonSamplingSelector:
    """Backward-compatible alias for the Thompson selector singleton."""
    return get_thompson_selector()


def reset_thompson_selector(*, save: bool = False) -> None:
    """Reset the Thompson selector singleton and unregister shutdown callbacks.

    This is intended for tests and process-lifecycle cleanup that need to drop
    per-environment database paths without leaving an atexit callback pointing
    at stale state.
    """
    global _thompson_selector, _thompson_selector_save_callback
    with _thompson_selector_lock:
        selector = _thompson_selector
        callback = _thompson_selector_save_callback
        if save and selector is not None:
            with contextlib.suppress(Exception):
                selector._save_state()
        if callback is not None:
            import atexit

            with contextlib.suppress(Exception):
                atexit.unregister(callback)
            with contextlib.suppress(Exception):
                from vetinari.shutdown import unregister_callback

                unregister_callback("Thompson Sampling state", callback)
        _thompson_selector = None
        _thompson_selector_save_callback = None


# ── Module-level convenience function for curriculum integration ──────────


def get_skill_rankings() -> list[dict[str, Any]]:
    """Return task-type skill rankings for TrainingCurriculum integration.

    Returns:
        List of ``{"task_type": str, "score": float}`` dicts sorted by mean
        expected value. Empty list when no arms have been observed.
    """
    try:
        selector = get_thompson_selector()
    except Exception:
        logger.warning("Could not get ThompsonSamplingSelector for skill rankings")
        return []

    task_types = [
        "coding",
        "planning",
        "analysis",
        "research",
        "architecture",
        "review",
        "documentation",
        "testing",
        "refactoring",
        "general",
    ]
    rankings: list[dict[str, Any]] = []
    for tt in task_types:
        ranked = selector.get_rankings(tt)
        if ranked:
            best_score = ranked[0][1]  # (model_id, score) tuples
            rankings.append({"task_type": tt, "score": best_score})
    return rankings
