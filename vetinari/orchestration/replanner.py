"""
Dynamic Mid-Execution Re-Planning
====================================

Detects when the current plan is no longer viable and generates a new
plan that preserves completed work while re-planning only remaining tasks.

Triggers: task failure, budget exceeded, quality drops, new information.

Usage:
    from vetinari.orchestration.replanner import Replanner

    replanner = Replanner()
    if replanner.should_replan(execution_metrics):
        new_plan = replanner.replan(current_plan, completed_tasks, reason="task_3 failed")
"""

import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ReplanTrigger(Enum):
    """Reasons that trigger re-planning."""
    TASK_FAILURE = "task_failure"
    BUDGET_EXCEEDED = "budget_exceeded"
    QUALITY_DROP = "quality_drop"
    NEW_INFORMATION = "new_information"
    DEPENDENCY_BROKEN = "dependency_broken"
    TIMEOUT = "timeout"
    MANUAL = "manual"


@dataclass
class ReplanEvent:
    """Record of a re-planning event."""
    event_id: str = field(default_factory=lambda: f"rp_{uuid.uuid4().hex[:8]}")
    trigger: str = ""
    reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_task_ids: List[str] = field(default_factory=list)
    removed_task_ids: List[str] = field(default_factory=list)
    added_task_ids: List[str] = field(default_factory=list)
    preserved_task_ids: List[str] = field(default_factory=list)
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReplanResult:
    """Result of a re-planning operation."""
    success: bool = False
    new_plan: Any = None
    event: Optional[ReplanEvent] = None
    original_task_count: int = 0
    new_task_count: int = 0
    preserved_count: int = 0
    removed_count: int = 0
    added_count: int = 0
    duration_seconds: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.event:
            d["event"] = self.event.to_dict()
        return d


class Replanner:
    """
    Dynamic mid-execution re-planner.

    Monitors execution metrics and triggers re-planning when the current
    plan becomes non-viable. Preserves completed work and only re-plans
    the remaining tasks.

    Features:
    - Configurable trigger conditions
    - Preserves completed task outputs
    - Dependency-aware re-planning
    - History tracking of all re-plan events
    - Pluggable plan generation via callback
    """

    def __init__(
        self,
        plan_generator: Optional[Callable] = None,
        max_replans: int = 5,
        failure_threshold: int = 2,
        quality_threshold: float = 0.4,
        budget_threshold: float = 0.9,
    ):
        """
        Args:
            plan_generator: Callable(goal, completed_tasks, constraints) -> new_plan.
                            Used to generate a replacement plan for remaining work.
            max_replans: Maximum number of re-plans allowed per execution.
            failure_threshold: Number of task failures before triggering re-plan.
            quality_threshold: Quality score below which to trigger re-plan.
            budget_threshold: Budget usage ratio above which to trigger re-plan.
        """
        self._plan_generator = plan_generator or self._default_plan_generator
        self._max_replans = max_replans
        self._failure_threshold = failure_threshold
        self._quality_threshold = quality_threshold
        self._budget_threshold = budget_threshold

        self._replan_count = 0
        self._history: List[ReplanEvent] = []

        logger.info(
            "Replanner initialized: max_replans=%d failure_threshold=%d "
            "quality_threshold=%.2f budget_threshold=%.2f",
            max_replans, failure_threshold, quality_threshold, budget_threshold,
        )

    # ------------------------------------------------------------------
    # Trigger evaluation
    # ------------------------------------------------------------------

    def should_replan(self, execution_metrics: Dict[str, Any]) -> bool:
        """
        Evaluate execution metrics to determine if re-planning is needed.

        Args:
            execution_metrics: Dictionary with keys:
                - failed_tasks: List[str] - IDs of failed tasks
                - quality_scores: Dict[str, float] - task_id -> quality score
                - budget_usage_ratio: float - fraction of budget used
                - elapsed_seconds: float - time elapsed
                - timeout_seconds: float - maximum allowed time
                - new_information: Optional[str] - new info that changes the plan

        Returns:
            True if re-planning should be triggered.
        """
        if self._replan_count >= self._max_replans:
            logger.debug("Max replans reached (%d), not triggering", self._max_replans)
            return False

        trigger, reason = self._evaluate_triggers(execution_metrics)
        if trigger:
            logger.info("Replan trigger detected: %s - %s", trigger.value, reason)
            return True

        return False

    def get_replan_reason(self, execution_metrics: Dict[str, Any]) -> Tuple[Optional[ReplanTrigger], str]:
        """
        Get the specific trigger and reason for re-planning.

        Returns:
            Tuple of (trigger_type, reason_string) or (None, "") if no trigger.
        """
        return self._evaluate_triggers(execution_metrics)

    def _evaluate_triggers(self, metrics: Dict[str, Any]) -> Tuple[Optional[ReplanTrigger], str]:
        """Evaluate all trigger conditions against metrics."""
        # Task failure threshold
        failed = metrics.get("failed_tasks", [])
        if len(failed) >= self._failure_threshold:
            return (
                ReplanTrigger.TASK_FAILURE,
                f"{len(failed)} tasks failed (threshold: {self._failure_threshold}): {', '.join(failed[:5])}",
            )

        # Quality drop
        quality_scores = metrics.get("quality_scores", {})
        if quality_scores:
            avg_quality = sum(quality_scores.values()) / len(quality_scores)
            if avg_quality < self._quality_threshold:
                return (
                    ReplanTrigger.QUALITY_DROP,
                    f"Average quality {avg_quality:.2f} below threshold {self._quality_threshold:.2f}",
                )

        # Budget exceeded
        budget_ratio = metrics.get("budget_usage_ratio", 0.0)
        if budget_ratio >= self._budget_threshold:
            return (
                ReplanTrigger.BUDGET_EXCEEDED,
                f"Budget usage at {budget_ratio:.0%} (threshold: {self._budget_threshold:.0%})",
            )

        # Timeout approaching
        elapsed = metrics.get("elapsed_seconds", 0)
        timeout = metrics.get("timeout_seconds", 0)
        if timeout > 0 and elapsed > timeout * 0.9:
            return (
                ReplanTrigger.TIMEOUT,
                f"Elapsed {elapsed:.0f}s approaching timeout {timeout:.0f}s",
            )

        # New information
        new_info = metrics.get("new_information")
        if new_info:
            return (
                ReplanTrigger.NEW_INFORMATION,
                f"New information received: {str(new_info)[:200]}",
            )

        # Dependency broken
        broken_deps = metrics.get("broken_dependencies", [])
        if broken_deps:
            return (
                ReplanTrigger.DEPENDENCY_BROKEN,
                f"Broken dependencies: {', '.join(broken_deps[:5])}",
            )

        return None, ""

    # ------------------------------------------------------------------
    # Re-planning
    # ------------------------------------------------------------------

    def replan(
        self,
        current_plan: Any,
        completed_tasks: List[str],
        reason: str,
        execution_metrics: Optional[Dict[str, Any]] = None,
        completed_outputs: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> ReplanResult:
        """
        Generate a new plan preserving completed work.

        Args:
            current_plan: The current plan object (with .tasks or dict with "tasks" key).
            completed_tasks: List of task IDs that are already completed.
            reason: Human-readable reason for re-planning.
            execution_metrics: Current execution metrics snapshot.
            completed_outputs: Outputs from completed tasks (task_id -> output).
            constraints: Additional constraints for the new plan.

        Returns:
            ReplanResult with the new plan and event metadata.
        """
        start_time = time.time()

        if self._replan_count >= self._max_replans:
            logger.warning("Max replans (%d) reached, cannot replan", self._max_replans)
            return ReplanResult(
                success=False,
                error=f"Maximum replans ({self._max_replans}) exceeded",
            )

        logger.info("Re-planning triggered: %s (replan #%d)", reason, self._replan_count + 1)

        # Extract tasks from current plan
        all_tasks = self._extract_tasks(current_plan)
        completed_set = set(completed_tasks)

        # Separate completed from remaining
        preserved = {tid: info for tid, info in all_tasks.items() if tid in completed_set}
        remaining = {tid: info for tid, info in all_tasks.items() if tid not in completed_set}

        # Extract the goal from the plan
        goal = self._extract_goal(current_plan)

        # Determine trigger type
        trigger = ReplanTrigger.MANUAL
        if execution_metrics:
            detected_trigger, _ = self._evaluate_triggers(execution_metrics)
            if detected_trigger:
                trigger = detected_trigger

        # Generate new plan for remaining work
        try:
            plan_constraints = {
                "completed_tasks": list(preserved.keys()),
                "completed_outputs": completed_outputs or {},
                "failed_reason": reason,
                "remaining_tasks": list(remaining.keys()),
                **(constraints or {}),
            }

            new_plan = self._plan_generator(goal, list(preserved.keys()), plan_constraints)
        except Exception as exc:
            logger.error("Plan generation failed during replan: %s", exc)
            return ReplanResult(
                success=False,
                error=f"Plan generation failed: {exc}",
                duration_seconds=time.time() - start_time,
            )

        # Determine what changed
        new_tasks = self._extract_tasks(new_plan) if new_plan else {}
        new_task_ids = set(new_tasks.keys())
        old_remaining_ids = set(remaining.keys())
        added_ids = new_task_ids - old_remaining_ids - completed_set
        removed_ids = old_remaining_ids - new_task_ids

        # Record event
        event = ReplanEvent(
            trigger=trigger.value,
            reason=reason,
            completed_task_ids=list(completed_set),
            removed_task_ids=list(removed_ids),
            added_task_ids=list(added_ids),
            preserved_task_ids=list(preserved.keys()),
            metrics_snapshot=execution_metrics or {},
        )
        self._history.append(event)
        self._replan_count += 1

        result = ReplanResult(
            success=True,
            new_plan=new_plan,
            event=event,
            original_task_count=len(all_tasks),
            new_task_count=len(new_tasks) + len(preserved),
            preserved_count=len(preserved),
            removed_count=len(removed_ids),
            added_count=len(added_ids),
            duration_seconds=time.time() - start_time,
        )

        logger.info(
            "Re-plan complete: preserved=%d, removed=%d, added=%d, new_total=%d (%.1fs)",
            result.preserved_count, result.removed_count,
            result.added_count, result.new_task_count, result.duration_seconds,
        )
        return result

    # ------------------------------------------------------------------
    # History and state
    # ------------------------------------------------------------------

    def get_history(self) -> List[ReplanEvent]:
        """Get all re-planning events."""
        return list(self._history)

    @property
    def replan_count(self) -> int:
        """Number of re-plans performed."""
        return self._replan_count

    @property
    def replans_remaining(self) -> int:
        """Number of re-plans still allowed."""
        return max(0, self._max_replans - self._replan_count)

    def reset(self) -> None:
        """Reset replan state (for testing or new execution)."""
        self._replan_count = 0
        self._history.clear()
        logger.info("Replanner state reset")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_tasks(self, plan: Any) -> Dict[str, Dict[str, Any]]:
        """Extract task_id -> task_info from a plan object."""
        tasks: Dict[str, Dict[str, Any]] = {}

        if isinstance(plan, dict):
            for item in plan.get("tasks", []):
                tid = item.get("id", item.get("task_id", ""))
                if tid:
                    tasks[tid] = item
        elif isinstance(plan, list):
            for item in plan:
                if isinstance(item, dict):
                    tid = item.get("id", item.get("task_id", ""))
                    if tid:
                        tasks[tid] = item
        elif hasattr(plan, "tasks"):
            for task in plan.tasks:
                tid = getattr(task, "id", getattr(task, "task_id", ""))
                if tid:
                    tasks[tid] = {
                        "id": tid,
                        "description": getattr(task, "description", ""),
                        "task_type": getattr(task, "task_type", "general"),
                        "depends_on": getattr(task, "depends_on", []),
                    }

        return tasks

    def _extract_goal(self, plan: Any) -> str:
        """Extract the goal/objective from a plan."""
        if isinstance(plan, dict):
            return plan.get("goal", plan.get("objective", plan.get("description", "")))
        if hasattr(plan, "goal"):
            return plan.goal
        if hasattr(plan, "objective"):
            return plan.objective
        return ""

    @staticmethod
    def _default_plan_generator(
        goal: str,
        completed_tasks: List[str],
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Default plan generator that creates a minimal continuation plan.

        In production, this should be replaced with a real planner that
        uses LLM inference to generate an optimized plan.
        """
        remaining = constraints.get("remaining_tasks", [])
        logger.debug(
            "Default plan generator: goal='%s', completed=%d, remaining=%d",
            goal[:80], len(completed_tasks), len(remaining),
        )

        # Create a simple sequential plan from remaining tasks
        tasks = []
        for i, tid in enumerate(remaining):
            tasks.append({
                "id": tid,
                "description": f"Remaining task {tid}",
                "task_type": "general",
                "depends_on": [remaining[i - 1]] if i > 0 else [],
            })

        return {
            "goal": goal,
            "tasks": tasks,
            "replanned": True,
            "completed_tasks": completed_tasks,
        }
