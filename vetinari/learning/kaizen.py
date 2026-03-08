"""
Kaizen Engine -- vetinari.learning.kaizen

Continuous improvement using PDCA (Plan-Do-Check-Act) cycles.
Tracks and eliminates 7 categories of Muda (waste).

Usage::

    from vetinari.learning.kaizen import get_kaizen_engine

    engine = get_kaizen_engine()
    improvement = engine.plan_improvement(
        area="orchestration",
        observation="Tasks frequently retry due to model overload",
        category="waiting",
    )
    engine.execute_improvement(improvement["id"])
    engine.check_results(improvement["id"], metric_before=0.6, metric_after=0.8)
    engine.act_on_results(improvement["id"])
"""

from __future__ import annotations

import logging
import uuid
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Muda (waste) categories from lean manufacturing
# ---------------------------------------------------------------------------

class MudaCategory(Enum):
    """Seven categories of waste (Muda) adapted for AI orchestration."""
    OVERPROCESSING = "overprocessing"    # Doing more work than needed (e.g., over-thinking)
    WAITING = "waiting"                  # Idle time waiting for models/resources
    DEFECTS = "defects"                  # Failed tasks requiring rework
    INVENTORY = "inventory"              # Excessive context/data accumulation
    MOTION = "motion"                    # Unnecessary agent hand-offs
    TRANSPORT = "transport"              # Redundant data movement between stages
    OVERPRODUCTION = "overproduction"    # Generating unused outputs/artifacts


# ---------------------------------------------------------------------------
# PDCA cycle phases
# ---------------------------------------------------------------------------

class PDCAPhase(Enum):
    """Phases of the Plan-Do-Check-Act cycle."""
    PLAN = "plan"
    DO = "do"
    CHECK = "check"
    ACT = "act"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Improvement:
    """A single improvement tracked through the PDCA cycle."""
    id: str
    area: str                                # e.g., "orchestration", "model_routing"
    observation: str                         # What was observed
    category: MudaCategory                   # Waste category
    phase: PDCAPhase = PDCAPhase.PLAN
    hypothesis: str = ""                     # What we think will improve
    action_plan: str = ""                    # What to do
    metric_before: Optional[float] = None    # Baseline metric
    metric_after: Optional[float] = None     # Post-change metric
    improvement_pct: Optional[float] = None  # Percentage improvement
    outcome: str = ""                        # Result description
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "area": self.area,
            "observation": self.observation,
            "category": self.category.value,
            "phase": self.phase.value,
            "hypothesis": self.hypothesis,
            "action_plan": self.action_plan,
            "metric_before": self.metric_before,
            "metric_after": self.metric_after,
            "improvement_pct": self.improvement_pct,
            "outcome": self.outcome,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


@dataclass
class WasteEvent:
    """A recorded instance of waste."""
    category: MudaCategory
    area: str
    description: str
    estimated_cost_ms: float = 0.0  # Time wasted in milliseconds
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "area": self.area,
            "description": self.description,
            "estimated_cost_ms": self.estimated_cost_ms,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# KaizenEngine
# ---------------------------------------------------------------------------

class KaizenEngine:
    """
    Drives continuous improvement through PDCA cycles and waste tracking.

    Methods follow the Deming cycle:
    - ``plan_improvement()`` -- identify waste and formulate a hypothesis
    - ``execute_improvement()`` -- apply the change (DO phase)
    - ``check_results()`` -- compare before/after metrics (CHECK phase)
    - ``act_on_results()`` -- standardise or abandon (ACT phase)
    """

    def __init__(self):
        self._improvements: Dict[str, Improvement] = {}
        self._waste_log: List[WasteEvent] = []
        self._waste_totals: Dict[str, float] = {cat.value: 0.0 for cat in MudaCategory}

        logger.debug("[KaizenEngine] Initialized")

    # ------------------------------------------------------------------
    # Waste tracking
    # ------------------------------------------------------------------

    def record_waste(
        self,
        category: str,
        area: str,
        description: str,
        estimated_cost_ms: float = 0.0,
    ) -> WasteEvent:
        """
        Record an instance of waste.

        Args:
            category: One of the 7 Muda categories (string value).
            area: System area where waste occurred.
            description: What happened.
            estimated_cost_ms: Estimated time wasted.

        Returns:
            The recorded WasteEvent.
        """
        try:
            muda = MudaCategory(category)
        except ValueError:
            muda = MudaCategory.OVERPROCESSING  # default

        event = WasteEvent(
            category=muda,
            area=area,
            description=description,
            estimated_cost_ms=estimated_cost_ms,
        )
        self._waste_log.append(event)
        self._waste_totals[muda.value] += estimated_cost_ms

        logger.debug(
            "[KaizenEngine] Waste recorded: %s in %s (%.0fms)",
            muda.value, area, estimated_cost_ms,
        )
        return event

    def get_waste_summary(self) -> Dict[str, Any]:
        """Return waste totals by category and top areas."""
        area_totals: Dict[str, float] = {}
        for event in self._waste_log:
            area_totals[event.area] = area_totals.get(event.area, 0.0) + event.estimated_cost_ms

        return {
            "by_category": dict(self._waste_totals),
            "by_area": area_totals,
            "total_waste_ms": sum(self._waste_totals.values()),
            "total_events": len(self._waste_log),
        }

    # ------------------------------------------------------------------
    # PDCA: Plan
    # ------------------------------------------------------------------

    def plan_improvement(
        self,
        area: str,
        observation: str,
        category: str = "overprocessing",
        hypothesis: str = "",
        action_plan: str = "",
    ) -> Dict[str, Any]:
        """
        PLAN phase: Identify an improvement opportunity.

        Args:
            area: System area to improve.
            observation: What was observed (the problem).
            category: Muda waste category.
            hypothesis: What we think will help.
            action_plan: Steps to take.

        Returns:
            Dict representation of the new Improvement.
        """
        try:
            muda = MudaCategory(category)
        except ValueError:
            muda = MudaCategory.OVERPROCESSING

        improvement = Improvement(
            id=f"kaizen_{uuid.uuid4().hex[:8]}",
            area=area,
            observation=observation,
            category=muda,
            phase=PDCAPhase.PLAN,
            hypothesis=hypothesis or f"Reducing {muda.value} waste in {area} will improve throughput",
            action_plan=action_plan or f"Investigate and address {muda.value} in {area}",
        )
        self._improvements[improvement.id] = improvement

        logger.info(
            "[KaizenEngine] PLAN: %s -- %s (%s)",
            improvement.id, observation[:60], muda.value,
        )
        return improvement.to_dict()

    # ------------------------------------------------------------------
    # PDCA: Do
    # ------------------------------------------------------------------

    def execute_improvement(
        self,
        improvement_id: str,
        metric_before: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        DO phase: Mark the improvement as being executed.

        Args:
            improvement_id: The improvement to execute.
            metric_before: Baseline metric value before the change.

        Returns:
            Updated improvement dict.
        """
        imp = self._improvements.get(improvement_id)
        if imp is None:
            return {"error": f"Improvement {improvement_id} not found"}

        imp.phase = PDCAPhase.DO
        if metric_before is not None:
            imp.metric_before = metric_before

        logger.info("[KaizenEngine] DO: %s (baseline=%.3f)", improvement_id, metric_before or 0.0)
        return imp.to_dict()

    # ------------------------------------------------------------------
    # PDCA: Check
    # ------------------------------------------------------------------

    def check_results(
        self,
        improvement_id: str,
        metric_before: Optional[float] = None,
        metric_after: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        CHECK phase: Compare before/after metrics.

        Args:
            improvement_id: The improvement to check.
            metric_before: Override baseline if not set during DO.
            metric_after: Post-change metric value.

        Returns:
            Updated improvement dict with improvement percentage.
        """
        imp = self._improvements.get(improvement_id)
        if imp is None:
            return {"error": f"Improvement {improvement_id} not found"}

        imp.phase = PDCAPhase.CHECK
        if metric_before is not None:
            imp.metric_before = metric_before
        if metric_after is not None:
            imp.metric_after = metric_after

        # Calculate improvement percentage
        if imp.metric_before is not None and imp.metric_after is not None and imp.metric_before > 0:
            imp.improvement_pct = ((imp.metric_after - imp.metric_before) / imp.metric_before) * 100
        else:
            imp.improvement_pct = 0.0

        logger.info(
            "[KaizenEngine] CHECK: %s -- before=%.3f after=%.3f (%.1f%%)",
            improvement_id,
            imp.metric_before or 0.0,
            imp.metric_after or 0.0,
            imp.improvement_pct or 0.0,
        )
        return imp.to_dict()

    # ------------------------------------------------------------------
    # PDCA: Act
    # ------------------------------------------------------------------

    def act_on_results(
        self,
        improvement_id: str,
        adopt: bool = True,
        outcome: str = "",
    ) -> Dict[str, Any]:
        """
        ACT phase: Decide whether to standardise or abandon the change.

        Args:
            improvement_id: The improvement to finalise.
            adopt: True to standardise, False to abandon.
            outcome: Description of the decision.

        Returns:
            Final improvement dict.
        """
        imp = self._improvements.get(improvement_id)
        if imp is None:
            return {"error": f"Improvement {improvement_id} not found"}

        if adopt:
            imp.phase = PDCAPhase.COMPLETED
            imp.outcome = outcome or "Improvement adopted and standardised"
        else:
            imp.phase = PDCAPhase.ABANDONED
            imp.outcome = outcome or "Improvement abandoned -- no significant benefit"

        imp.completed_at = time.time()

        logger.info(
            "[KaizenEngine] ACT: %s -- %s (%s)",
            improvement_id, imp.phase.value, imp.outcome[:60],
        )
        return imp.to_dict()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_improvement(self, improvement_id: str) -> Optional[Dict[str, Any]]:
        """Get a single improvement by ID."""
        imp = self._improvements.get(improvement_id)
        return imp.to_dict() if imp else None

    def list_improvements(
        self,
        phase: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List improvements with optional filtering."""
        results = list(self._improvements.values())
        if phase:
            try:
                target_phase = PDCAPhase(phase)
                results = [i for i in results if i.phase == target_phase]
            except ValueError:
                pass
        if category:
            try:
                target_cat = MudaCategory(category)
                results = [i for i in results if i.category == target_cat]
            except ValueError:
                pass
        return [i.to_dict() for i in results]

    def get_active_improvements(self) -> List[Dict[str, Any]]:
        """Return improvements that are still in progress (PLAN, DO, or CHECK)."""
        active = [
            i for i in self._improvements.values()
            if i.phase in (PDCAPhase.PLAN, PDCAPhase.DO, PDCAPhase.CHECK)
        ]
        return [i.to_dict() for i in active]

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics."""
        phase_counts = {}
        for imp in self._improvements.values():
            phase_counts[imp.phase.value] = phase_counts.get(imp.phase.value, 0) + 1

        return {
            "total_improvements": len(self._improvements),
            "by_phase": phase_counts,
            "waste_summary": self.get_waste_summary(),
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_kaizen_engine: Optional[KaizenEngine] = None


def get_kaizen_engine() -> KaizenEngine:
    """Get or create the global KaizenEngine."""
    global _kaizen_engine
    if _kaizen_engine is None:
        _kaizen_engine = KaizenEngine()
    return _kaizen_engine


def reset_kaizen_engine() -> None:
    """Reset the singleton (useful for testing)."""
    global _kaizen_engine
    _kaizen_engine = None
