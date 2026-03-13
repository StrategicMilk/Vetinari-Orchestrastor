"""Thompson Sampling Model Selector - Vetinari Self-Improvement Subsystem.

Implements Bayesian bandit-style model selection that naturally balances
exploration (trying less-used models) with exploitation (using proven ones).

Each model+task_type pair maintains a Beta distribution:
  - alpha = successes (weighted by quality scores)
  - beta  = failures

When selecting a model, we sample from each distribution and pick the highest.
"""

from __future__ import annotations

import logging
import os
import random
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BetaArm:
    """Beta distribution arm for a model+task_type pair."""

    model_id: str
    task_type: str
    alpha: float = 2.0  # Successes (quality-weighted); Beta(2,2) slightly skeptical prior
    beta: float = 2.0  # Failures; accelerates convergence vs naive Beta(1,1)
    total_pulls: int = 0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def mean(self) -> float:
        """Expected value of the Beta distribution."""
        return self.alpha / (self.alpha + self.beta)

    def sample(self) -> float:
        """Sample from the Beta distribution using the ratio method."""
        # Approximate Beta sampling via the relationship to Gamma distributions
        # Uses Python's built-in random.betavariate
        try:
            return random.betavariate(self.alpha, self.beta)
        except ValueError:
            return self.mean

    def update(self, quality_score: float, success: bool) -> None:
        """Update the arm based on observed outcome."""
        if success:
            self.alpha += quality_score  # Weight by quality
        else:
            self.beta += 1.0 - quality_score
        self.total_pulls += 1
        self.last_updated = datetime.now().isoformat()


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

    def __init__(self):
        # Key: "model_id:task_type"
        self._arms: dict[str, BetaArm] = {}
        self._lock = threading.Lock()
        self._load_state()

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
            cost_per_model: Optional cost estimates per model.

        Returns:
            Selected model ID.
        """
        with self._lock:
            if not candidate_models:
                return "default"

            cost_per_model = cost_per_model or {}
            max_cost = max(cost_per_model.values(), default=1.0) or 1.0

            best_model = candidate_models[0]
            best_score = -1.0

            for model_id in candidate_models:
                arm = self._get_or_create_arm(model_id, task_type)
                sampled = arm.sample()

                # Apply cost penalty additively to preserve Thompson Sampling
                # exploration properties (multiplicative would distort the Beta distribution)
                cost = cost_per_model.get(model_id, max_cost * 0.5)  # default to median, not free
                cost_penalty = self.COST_WEIGHT * (cost / max_cost)
                adjusted = sampled - cost_penalty

                logger.debug(
                    f"[Thompson] {model_id}/{task_type}: sample={sampled:.3f} "
                    f"cost_penalty={cost_penalty:.3f} adjusted={adjusted:.3f} "
                    f"(alpha={arm.alpha:.1f}, beta={arm.beta:.1f})"
                )

                if adjusted > best_score:
                    best_score = adjusted
                    best_model = model_id

            logger.info("[Thompson] Selected %s for %s (score=%.3f)", best_model, task_type, best_score)
            return best_model

    def update(
        self,
        model_id: str,
        task_type: str,
        quality_score: float,
        success: bool,
    ) -> None:
        """Update the arm after observing an outcome.

        Args:
            model_id: The model that was used.
            task_type: The task type.
            quality_score: Observed quality score 0.0-1.0.
            success: Whether the task succeeded.
        """
        with self._lock:
            arm = self._get_or_create_arm(model_id, task_type)
            arm.update(quality_score, success)
            self._save_state()

    # Benchmark results are 3x more reliable than single-task outcomes
    BENCHMARK_WEIGHT_MULTIPLIER = 3

    def update_from_benchmark(
        self,
        model_id: str,
        pass_rate: float,
        n_trials: int,
        task_type: str = "general",
    ) -> None:
        """Update Thompson Sampling arms from benchmark results.

        Benchmark results are weighted 3x (BENCHMARK_WEIGHT_MULTIPLIER) because
        they are more reliable than single-task outcomes -- they aggregate over
        many standardised test cases.

        Args:
            model_id: The model that was benchmarked.
            pass_rate: Fraction of benchmark cases passed (0.0-1.0).
            n_trials: Number of benchmark trials run.
            task_type: Task type the benchmark covers.
        """
        arm = self._get_or_create_arm(model_id, task_type)
        w = self.BENCHMARK_WEIGHT_MULTIPLIER

        successes = pass_rate * n_trials
        failures = n_trials - successes

        arm.alpha += successes * w
        arm.beta += failures * w
        arm.total_pulls += n_trials
        arm.last_updated = datetime.now().isoformat()

        logger.info(
            f"[Thompson] Benchmark update for {model_id}/{task_type}: "
            f"pass_rate={pass_rate:.3f}, n_trials={n_trials}, "
            f"weighted_successes={successes * w:.1f}, weighted_failures={failures * w:.1f}, "
            f"new_mean={arm.mean:.3f}"
        )
        self._save_state()

    def get_rankings(self, task_type: str) -> list[tuple[str, float]]:
        """Get model rankings for a task type (by expected value)."""
        arms = [(k.split(":")[0], arm.mean) for k, arm in self._arms.items() if k.endswith(f":{task_type}")]
        arms.sort(key=lambda x: x[1], reverse=True)
        return arms

    def get_arm_state(self, model_id: str, task_type: str) -> dict[str, Any]:
        """Get the current state of a Beta arm."""
        arm = self._get_or_create_arm(model_id, task_type)
        return {
            "model_id": arm.model_id,
            "task_type": arm.task_type,
            "alpha": arm.alpha,
            "beta": arm.beta,
            "mean": arm.mean,
            "total_pulls": arm.total_pulls,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_create_arm(self, model_id: str, task_type: str) -> BetaArm:
        key = f"{model_id}:{task_type}"
        if key not in self._arms:
            alpha, beta = self._get_informed_prior(model_id, task_type)
            self._arms[key] = BetaArm(
                model_id=model_id,
                task_type=task_type,
                alpha=alpha,
                beta=beta,
            )
        return self._arms[key]

    def _get_informed_prior(self, model_id: str, task_type: str) -> tuple:
        """Get informed prior from BenchmarkSeeder, fallback to Beta(1,1)."""
        try:
            from vetinari.learning.benchmark_seeder import get_benchmark_seeder

            return get_benchmark_seeder().get_prior(model_id, task_type)
        except Exception:
            return (1.0, 1.0)

    def _get_state_dir(self) -> str:
        """Get the .vetinari state directory, using project root or env var."""
        import os

        # Allow override via env var
        state_dir = os.environ.get("VETINARI_STATE_DIR", "")
        if not state_dir:
            # Use the directory two levels above this file (project root/.vetinari)
            pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            state_dir = os.path.join(pkg_root, ".vetinari")
        return state_dir

    def _load_state(self) -> None:
        """Load persisted arm states from memory/disk."""
        try:
            import json

            state_file = os.path.join(self._get_state_dir(), "thompson_state.json")
            if os.path.exists(state_file):
                with open(state_file) as f:
                    data = json.load(f)
                for key, d in data.items():
                    self._arms[key] = BetaArm(**d)
                logger.debug("[Thompson] Loaded %s arm states", len(self._arms))
        except Exception as e:
            logger.debug("[Thompson] Could not load state: %s", e)

    def _save_state(self) -> None:
        """Persist arm states for continuity across restarts."""
        try:
            import json
            from dataclasses import asdict

            state_dir = self._get_state_dir()
            os.makedirs(state_dir, exist_ok=True)
            state_file = os.path.join(state_dir, "thompson_state.json")
            with open(state_file, "w") as f:
                json.dump({k: asdict(v) for k, v in self._arms.items()}, f, indent=2)
        except Exception as e:
            logger.debug("[Thompson] Could not save state: %s", e)


# Singleton
_thompson_selector: ThompsonSamplingSelector | None = None


def get_thompson_selector() -> ThompsonSamplingSelector:
    global _thompson_selector
    if _thompson_selector is None:
        _thompson_selector = ThompsonSamplingSelector()
    return _thompson_selector
