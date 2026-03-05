"""
Cost Optimizer - Vetinari Self-Improvement Subsystem

Analyzes cost data to automatically route tasks to the cheapest
adequate model, respecting quality thresholds.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CostEfficiency:
    """Quality-per-dollar metric for a model+task_type pair."""
    model_id: str
    task_type: str
    avg_quality: float
    avg_cost_usd: float
    quality_per_dollar: float
    total_uses: int


class CostOptimizer:
    """
    Selects the cheapest model that meets a quality threshold.

    Integrates with the CostTracker analytics module to make
    evidence-based routing decisions.
    """

    DEFAULT_MIN_QUALITY = 0.65   # Minimum acceptable quality
    LOCAL_COST = 0.0              # Local models are free

    def __init__(self):
        self._cost_tracker = None

    def _get_cost_tracker(self):
        if self._cost_tracker is None:
            try:
                from vetinari.analytics.cost import get_cost_tracker
                self._cost_tracker = get_cost_tracker()
            except Exception:
                pass
        return self._cost_tracker

    def select_cheapest_adequate(
        self,
        task_type: str,
        candidate_models: List[str],
        min_quality: float = None,
        max_cost_usd: float = None,
    ) -> str:
        """
        Select the cheapest model that meets the quality threshold.

        Args:
            task_type: The task type.
            candidate_models: List of model IDs to consider.
            min_quality: Minimum acceptable quality (default 0.65).
            max_cost_usd: Maximum cost per call in USD.

        Returns:
            The selected model ID.
        """
        min_quality = min_quality or self.DEFAULT_MIN_QUALITY
        efficiencies = self._get_efficiencies(task_type, candidate_models)

        # Filter by quality threshold
        adequate = [e for e in efficiencies if e.avg_quality >= min_quality]

        # Further filter by max cost
        if max_cost_usd is not None:
            adequate = [e for e in adequate if e.avg_cost_usd <= max_cost_usd]

        if not adequate:
            # No model meets criteria -- return the highest quality one
            if efficiencies:
                return max(efficiencies, key=lambda e: e.avg_quality).model_id
            return candidate_models[0] if candidate_models else "default"

        # Among adequate models, pick the cheapest
        cheapest = min(adequate, key=lambda e: e.avg_cost_usd)
        logger.info(
            f"[CostOptimizer] Selected {cheapest.model_id} for {task_type}: "
            f"quality={cheapest.avg_quality:.2f} cost=${cheapest.avg_cost_usd:.4f}"
        )
        return cheapest.model_id

    def _get_efficiencies(self, task_type: str, models: List[str]) -> List[CostEfficiency]:
        """Compute quality-per-dollar for each model."""
        tracker = self._get_cost_tracker()
        efficiencies: List[CostEfficiency] = []

        for model_id in models:
            cost = 0.0
            quality = 0.7  # Prior

            # Get cost from analytics
            if tracker:
                try:
                    report = tracker.get_report()
                    for entry in report.get("entries", []):
                        if entry.get("model_id") == model_id:
                            cost = float(entry.get("cost_usd", 0.0))
                            break
                except Exception:
                    pass

            # Get quality from Thompson Sampling selector
            try:
                from vetinari.learning.model_selector import get_thompson_selector
                arm_state = get_thompson_selector().get_arm_state(model_id, task_type)
                quality = arm_state.get("mean", 0.7)
            except Exception:
                pass

            # Local models (cost=0) get a high but bounded score; use quality * 10 as proxy
            qpd = quality * 10.0 if cost == 0.0 else quality / cost
            efficiencies.append(CostEfficiency(
                model_id=model_id,
                task_type=task_type,
                avg_quality=quality,
                avg_cost_usd=cost,
                quality_per_dollar=qpd,
                total_uses=0,
            ))

        return efficiencies

    def get_budget_forecast(
        self,
        planned_tasks: int,
        task_types: List[str],
        models: List[str],
    ) -> Dict[str, Any]:
        """
        Estimate total cost for a planned set of tasks.

        Returns:
            Dict with estimated_cost_usd, breakdown_by_type, warnings.
        """
        total_cost = 0.0
        breakdown: Dict[str, float] = {}
        tracker = self._get_cost_tracker()

        for i, task_type in enumerate(task_types):
            if i >= planned_tasks:
                break
            efficiencies = self._get_efficiencies(task_type, models)
            if efficiencies:
                avg = sum(e.avg_cost_usd for e in efficiencies) / len(efficiencies)
            else:
                avg = 0.0
            breakdown[task_type] = avg
            total_cost += avg

        warnings: List[str] = []
        if total_cost > 1.0:
            warnings.append(f"Estimated cost ${total_cost:.2f} exceeds $1.00 threshold")

        return {
            "estimated_cost_usd": round(total_cost, 4),
            "breakdown_by_type": breakdown,
            "warnings": warnings,
        }


_cost_optimizer: Optional[CostOptimizer] = None


def get_cost_optimizer() -> CostOptimizer:
    global _cost_optimizer
    if _cost_optimizer is None:
        _cost_optimizer = CostOptimizer()
    return _cost_optimizer
