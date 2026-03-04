"""
Thompson Sampling Model Selector - Vetinari Self-Improvement Subsystem

Implements Bayesian bandit-style model selection that naturally balances
exploration (trying less-used models) with exploitation (using proven ones).

Each model+task_type pair maintains a Beta distribution:
  - alpha = successes (weighted by quality scores)
  - beta  = failures

When selecting a model, we sample from each distribution and pick the highest.
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BetaArm:
    """Beta distribution arm for a model+task_type pair."""
    model_id: str
    task_type: str
    alpha: float = 1.0      # Successes (quality-weighted)
    beta: float = 1.0       # Failures
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
            self.alpha += quality_score       # Weight by quality
        else:
            self.beta += (1.0 - quality_score)
        self.total_pulls += 1
        self.last_updated = datetime.now().isoformat()


class ThompsonSamplingSelector:
    """
    Thompson Sampling model selector for exploration-exploitation balance.

    Automatically:
    - Explores new/untested models (uninformed Beta(1,1) prior)
    - Exploits well-performing models for their appropriate task types
    - Incorporates cost as a penalty on expected return
    - Persists arm states to survive restarts
    """

    COST_WEIGHT = 0.15   # How much to penalize expensive models
    MIN_ARMS = 1

    def __init__(self):
        # Key: "model_id:task_type"
        self._arms: Dict[str, BetaArm] = {}
        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_model(
        self,
        task_type: str,
        candidate_models: List[str],
        cost_per_model: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Select the best model for a task type using Thompson Sampling.

        Args:
            task_type: Task type string.
            candidate_models: List of available model IDs.
            cost_per_model: Optional cost estimates per model.

        Returns:
            Selected model ID.
        """
        if not candidate_models:
            return "default"

        cost_per_model = cost_per_model or {}
        max_cost = max(cost_per_model.values(), default=1.0) or 1.0

        best_model = candidate_models[0]
        best_score = -1.0

        for model_id in candidate_models:
            arm = self._get_or_create_arm(model_id, task_type)
            sampled = arm.sample()

            # Apply cost penalty: more expensive → lower adjusted score
            cost = cost_per_model.get(model_id, 0.0)
            cost_penalty = self.COST_WEIGHT * (cost / max_cost)
            adjusted = sampled * (1.0 - cost_penalty)

            logger.debug(
                f"[Thompson] {model_id}/{task_type}: sample={sampled:.3f} "
                f"cost_penalty={cost_penalty:.3f} adjusted={adjusted:.3f} "
                f"(alpha={arm.alpha:.1f}, beta={arm.beta:.1f})"
            )

            if adjusted > best_score:
                best_score = adjusted
                best_model = model_id

        logger.info(f"[Thompson] Selected {best_model} for {task_type} (score={best_score:.3f})")
        return best_model

    def update(
        self,
        model_id: str,
        task_type: str,
        quality_score: float,
        success: bool,
    ) -> None:
        """
        Update the arm after observing an outcome.

        Args:
            model_id: The model that was used.
            task_type: The task type.
            quality_score: Observed quality score 0.0-1.0.
            success: Whether the task succeeded.
        """
        arm = self._get_or_create_arm(model_id, task_type)
        arm.update(quality_score, success)
        self._save_state()

    def get_rankings(self, task_type: str) -> List[Tuple[str, float]]:
        """Get model rankings for a task type (by expected value)."""
        arms = [(k.split(":")[0], arm.mean)
                for k, arm in self._arms.items()
                if k.endswith(f":{task_type}")]
        arms.sort(key=lambda x: x[1], reverse=True)
        return arms

    def get_arm_state(self, model_id: str, task_type: str) -> Dict[str, Any]:
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
            self._arms[key] = BetaArm(model_id=model_id, task_type=task_type)
        return self._arms[key]

    def _load_state(self) -> None:
        """Load persisted arm states from memory/disk."""
        try:
            import json, os
            state_file = os.path.join(
                os.path.expanduser("~"), ".lmstudio", "projects", "Vetinari",
                ".vetinari", "thompson_state.json"
            )
            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    data = json.load(f)
                for key, d in data.items():
                    self._arms[key] = BetaArm(**d)
                logger.debug(f"[Thompson] Loaded {len(self._arms)} arm states")
        except Exception as e:
            logger.debug(f"[Thompson] Could not load state: {e}")

    def _save_state(self) -> None:
        """Persist arm states for continuity across restarts."""
        try:
            import json, os
            from dataclasses import asdict
            state_dir = os.path.join(
                os.path.expanduser("~"), ".lmstudio", "projects", "Vetinari", ".vetinari"
            )
            os.makedirs(state_dir, exist_ok=True)
            state_file = os.path.join(state_dir, "thompson_state.json")
            with open(state_file, "w") as f:
                json.dump({k: asdict(v) for k, v in self._arms.items()}, f, indent=2)
        except Exception as e:
            logger.debug(f"[Thompson] Could not save state: {e}")


# Singleton
_thompson_selector: Optional[ThompsonSamplingSelector] = None


def get_thompson_selector() -> ThompsonSamplingSelector:
    global _thompson_selector
    if _thompson_selector is None:
        _thompson_selector = ThompsonSamplingSelector()
    return _thompson_selector
