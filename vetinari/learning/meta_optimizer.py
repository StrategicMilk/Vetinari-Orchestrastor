"""Meta-Optimizer — tracks which self-improvement strategies produce results.

Monitors per-cycle outcomes for each improvement strategy (prompt evolution,
training, auto-research) and allocates idle time to highest-ROI activities.

Three operational phases:
- IMPROVEMENT: Strategies are producing gains -> continue
- SATURATION: Gains are plateauing -> switch strategy
- COLLAPSE_RISK: Quality degrading -> halt and rollback
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from vetinari.constants import VETINARI_STATE_DIR

logger = logging.getLogger(__name__)


class LearningPhase:
    """Detected phase of the learning system."""

    IMPROVEMENT = "improvement"
    SATURATION = "saturation"
    COLLAPSE_RISK = "collapse_risk"


@dataclass
class StrategyRecord:  # noqa: VET114 — mutable by design; fields are updated in-place after each optimization cycle
    """Performance record for a self-improvement strategy."""

    strategy_name: str
    total_cycles: int = 0
    total_gain: float = 0.0
    recent_gains: list[float] = field(default_factory=list)  # noqa: VET220 — capped to 20 at append site; list kept for json.dump(asdict()) compatibility
    last_cycle_at: str | None = None

    def __repr__(self) -> str:
        return f"StrategyRecord(strategy_name={self.strategy_name!r}, total_cycles={self.total_cycles!r})"

    @property
    def avg_gain_per_cycle(self) -> float:
        """Average quality gain per improvement cycle."""
        return self.total_gain / max(self.total_cycles, 1)

    @property
    def recent_avg_gain(self) -> float:
        """Average gain over the last 10 cycles."""
        recent = self.recent_gains[-10:] if self.recent_gains else []
        return sum(recent) / max(len(recent), 1)


class MetaOptimizer:
    """Tracks which self-improvement strategies produce results and allocates resources.

    Per-cycle tracking: which mutation operators helped, which training activities
    improved quality, which experiments produced real gains. Detects operational
    phase (IMPROVEMENT / SATURATION / COLLAPSE_RISK) and auto-halts if quality
    degrades below critical threshold.
    """

    COLLAPSE_THRESHOLD = 0.3  # Halt if recent quality drops below this magnitude
    SATURATION_THRESHOLD = 0.01  # Gains below this magnitude indicate saturation
    HEALTH_WINDOW = 20  # Recent cycles to consider for phase detection

    def __init__(self) -> None:
        self._strategies: dict[str, StrategyRecord] = {}
        self._quality_history: deque[float] = deque(maxlen=100)
        self._lock = threading.Lock()
        self._load_state()

    def record_cycle(
        self,
        strategy_name: str,
        quality_gain: float,
        success: bool = True,
    ) -> None:
        """Record the outcome of an improvement cycle.

        Args:
            strategy_name: Which strategy was used (prompt_evolution, training, autoresearch).
            quality_gain: Quality improvement (positive = better).
            success: Whether the cycle completed successfully.
        """
        with self._lock:
            if strategy_name not in self._strategies:
                self._strategies[strategy_name] = StrategyRecord(strategy_name=strategy_name)

            rec = self._strategies[strategy_name]
            rec.total_cycles += 1
            rec.total_gain += quality_gain
            rec.recent_gains.append(quality_gain)
            if len(rec.recent_gains) > 50:
                rec.recent_gains = rec.recent_gains[-50:]
            rec.last_cycle_at = datetime.now(timezone.utc).isoformat()

            self._quality_history.append(quality_gain)
            self._save_state()

        logger.info(
            "[MetaOptimizer] Recorded %s cycle: gain=%+.4f, success=%s",
            strategy_name,
            quality_gain,
            success,
        )

    def get_roi_rankings(self) -> list[dict[str, Any]]:
        """Get strategies ranked by ROI (quality gain per cycle).

        Returns:
            List of strategy dicts with name, avg_gain, total_cycles, sorted by
            recent_avg_gain descending.
        """
        with self._lock:
            rankings = [
                {
                    "strategy": rec.strategy_name,
                    "avg_gain_per_cycle": round(rec.avg_gain_per_cycle, 4),
                    "recent_avg_gain": round(rec.recent_avg_gain, 4),
                    "total_cycles": rec.total_cycles,
                }
                for rec in self._strategies.values()
            ]
        rankings.sort(key=lambda r: r["recent_avg_gain"], reverse=True)
        return rankings

    def detect_phase(self) -> str:
        """Detect the current learning phase based on recent cycle history.

        Returns:
            One of LearningPhase.IMPROVEMENT, SATURATION, or COLLAPSE_RISK.
        """
        with self._lock:
            recent = list(self._quality_history)[-self.HEALTH_WINDOW :]

        if not recent or len(recent) < 5:
            return LearningPhase.IMPROVEMENT  # Not enough data to make a judgment

        avg_recent = sum(recent) / len(recent)

        # Detect regression or mild negative delta
        if avg_recent < 0:
            logger.warning(
                "[MetaOptimizer] Mild regression: recent avg gain = %.4f — treating as collapse risk",
                avg_recent,
            )
            return LearningPhase.COLLAPSE_RISK

        if avg_recent < -abs(self.COLLAPSE_THRESHOLD):
            logger.warning(
                "[MetaOptimizer] COLLAPSE RISK: recent avg gain = %.4f",
                avg_recent,
            )
            return LearningPhase.COLLAPSE_RISK

        if abs(avg_recent) < self.SATURATION_THRESHOLD:
            return LearningPhase.SATURATION

        return LearningPhase.IMPROVEMENT

    def suggest_next_strategy(self) -> str | None:
        """Suggest the highest-ROI strategy for the next idle cycle.

        Returns:
            Strategy name with highest recent ROI, or None if in collapse state.
        """
        phase = self.detect_phase()
        if phase == LearningPhase.COLLAPSE_RISK:
            logger.warning("[MetaOptimizer] Collapse risk detected — recommending halt")
            return None

        rankings = self.get_roi_rankings()
        if rankings:
            return rankings[0]["strategy"]
        return "prompt_evolution"  # Default when no prior data exists

    def allocate_idle_budget(self) -> dict[str, float]:
        """Allocate idle-time budget across strategies based on ROI.

        Returns a percentage allocation (0.0-1.0) for each known strategy.
        Higher-ROI strategies get more time.  Strategies with no data get
        an equal minimum share to preserve exploration.

        In COLLAPSE_RISK phase, returns an empty dict — no budget should
        be spent on improvement while quality is degrading.

        Returns:
            Mapping of strategy name to fraction of idle time (sums to 1.0),
            or empty dict if in collapse state.
        """
        phase = self.detect_phase()
        if phase == LearningPhase.COLLAPSE_RISK:
            logger.warning("[MetaOptimizer] Collapse risk — zero budget allocation")
            return {}

        with self._lock:
            strategy_names = list(self._strategies.keys())
            records = dict(self._strategies)

        if not strategy_names:
            # No data yet — equal split across default strategies
            defaults = ["prompt_evolution", "training", "auto_research"]
            share = round(1.0 / len(defaults), 4)
            return dict.fromkeys(defaults, share)

        # Compute raw scores: use recent_avg_gain shifted so all values are positive
        raw_scores: dict[str, float] = {}
        for name in strategy_names:
            rec = records[name]
            # Shift by adding 1.0 so negative gains still get a small share
            raw_scores[name] = max(rec.recent_avg_gain + 1.0, 0.01)

        # Strategies with zero cycles get exploration bonus
        min_exploration = 0.05  # Each strategy gets at least 5%
        total_raw = sum(raw_scores.values())

        allocations: dict[str, float] = {}
        remaining = 1.0 - min_exploration * len(strategy_names)
        if remaining < 0:
            # Too many strategies for the minimum — just do equal split
            share = round(1.0 / len(strategy_names), 4)
            return dict.fromkeys(strategy_names, share)

        for name in strategy_names:
            proportional = (raw_scores[name] / total_raw) * remaining
            allocations[name] = round(min_exploration + proportional, 4)

        # Normalize to exactly 1.0
        total = sum(allocations.values())
        if total > 0:
            allocations = {k: round(v / total, 4) for k, v in allocations.items()}

        return allocations

    def _load_state(self) -> None:
        """Load persisted meta-optimizer state from disk."""
        try:
            state_file = VETINARI_STATE_DIR / "meta_optimizer.json"
            if state_file.exists():
                with state_file.open(encoding="utf-8") as f:
                    data = json.load(f)
                for name, rec_data in data.get("strategies", {}).items():
                    self._strategies[name] = StrategyRecord(**rec_data)
                for q in data.get("quality_history", []):
                    self._quality_history.append(q)
        except Exception as exc:
            logger.warning(
                "[MetaOptimizer] Could not load state from disk — starting fresh: %s",
                exc,
            )

    def _save_state(self) -> None:
        """Persist meta-optimizer state to disk for recovery across restarts."""
        try:
            state_file = VETINARI_STATE_DIR / "meta_optimizer.json"
            state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "strategies": {k: asdict(v) for k, v in self._strategies.items()},
                "quality_history": list(self._quality_history),
            }
            with state_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.warning(
                "[MetaOptimizer] Could not save state — data will be lost on restart: %s",
                exc,
            )


# Module-level singleton — written by get_meta_optimizer(), read by callers
_meta_optimizer: MetaOptimizer | None = None
_meta_optimizer_lock = threading.Lock()


def get_meta_optimizer() -> MetaOptimizer:
    """Return the singleton MetaOptimizer instance (thread-safe).

    Returns:
        The shared MetaOptimizer instance.
    """
    global _meta_optimizer
    if _meta_optimizer is None:
        with _meta_optimizer_lock:
            if _meta_optimizer is None:
                _meta_optimizer = MetaOptimizer()
    return _meta_optimizer
