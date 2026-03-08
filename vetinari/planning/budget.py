"""
Budget-Aware Planning
======================

Provides token budget allocation, tracking, and dynamic strategy adaptation.
Ensures plans stay within resource constraints and degrade gracefully when
budgets are exceeded.

Usage:
    from vetinari.planning.budget import BudgetPlanner

    planner = BudgetPlanner(total_tokens=100000)
    allocations = planner.allocate_budget(plan)
    planner.track_usage("task_1", 1500)
    remaining = planner.check_remaining("task_1")
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BudgetStrategy(Enum):
    """Token allocation strategy."""
    PROPORTIONAL = "proportional"    # Allocate proportional to task complexity
    EQUAL = "equal"                  # Equal allocation across tasks
    PRIORITY = "priority"            # Weighted by task priority
    ADAPTIVE = "adaptive"           # Start proportional, adapt based on usage


class DepthMode(Enum):
    """Reasoning depth when budget allows or constrains."""
    DEEP = "deep"           # Full chain-of-thought, high temperature
    STANDARD = "standard"   # Normal reasoning
    SHALLOW = "shallow"     # Quick heuristic, low token output
    MINIMAL = "minimal"     # Bare minimum to complete


@dataclass
class TaskBudget:
    """Budget allocation and tracking for a single task."""
    task_id: str
    allocated_tokens: int = 0
    used_tokens: int = 0
    reserved_tokens: int = 0       # Tokens reserved for verification/retry
    depth_mode: str = "standard"
    priority: int = 5              # 1=highest, 10=lowest
    complexity_score: float = 0.5  # 0.0 to 1.0
    started: bool = False
    completed: bool = False
    exceeded: bool = False

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.allocated_tokens - self.used_tokens)

    @property
    def usage_ratio(self) -> float:
        if self.allocated_tokens == 0:
            return 0.0
        return self.used_tokens / self.allocated_tokens

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["remaining_tokens"] = self.remaining_tokens
        d["usage_ratio"] = self.usage_ratio
        return d


@dataclass
class BudgetSnapshot:
    """Point-in-time snapshot of budget state."""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_tokens: int = 0
    allocated_tokens: int = 0
    used_tokens: int = 0
    remaining_tokens: int = 0
    tasks_completed: int = 0
    tasks_exceeded: int = 0
    replan_triggered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BudgetPlanner:
    """
    Budget-aware planning and resource allocation for multi-agent execution.

    Features:
    - Per-task token budget allocation with multiple strategies
    - Real-time usage tracking
    - Dynamic depth adaptation (deep vs shallow reasoning)
    - Graceful degradation when approaching limits
    - Automatic re-planning trigger when budgets are exceeded
    """

    def __init__(
        self,
        total_tokens: int = 100000,
        strategy: BudgetStrategy = BudgetStrategy.ADAPTIVE,
        reserve_ratio: float = 0.15,
        replan_threshold: float = 0.9,
        on_replan: Optional[Callable] = None,
    ):
        """
        Args:
            total_tokens: Total token budget for the entire plan.
            strategy: Budget allocation strategy.
            reserve_ratio: Fraction of each task's budget reserved for retries.
            replan_threshold: Trigger re-planning when this fraction of total budget is used.
            on_replan: Callback invoked when re-planning is triggered.
        """
        self._total_tokens = total_tokens
        self._strategy = strategy
        self._reserve_ratio = reserve_ratio
        self._replan_threshold = replan_threshold
        self._on_replan = on_replan
        self._lock = threading.Lock()

        self._task_budgets: Dict[str, TaskBudget] = {}
        self._used_total: int = 0
        self._snapshots: List[BudgetSnapshot] = []
        self._replan_triggered = False

        logger.info(
            "BudgetPlanner initialized: total=%d strategy=%s reserve=%.0f%%",
            total_tokens, strategy.value, reserve_ratio * 100,
        )

    # ------------------------------------------------------------------
    # Budget allocation
    # ------------------------------------------------------------------

    def allocate_budget(
        self,
        plan: Any,
        task_complexities: Optional[Dict[str, float]] = None,
        task_priorities: Optional[Dict[str, int]] = None,
    ) -> Dict[str, TaskBudget]:
        """
        Allocate token budgets to each task in the plan.

        Args:
            plan: A plan object with a .tasks attribute (list of task objects
                  with .id or task_id and .description attributes), or a list
                  of dicts with "id" and optional "complexity" / "priority" keys.
            task_complexities: Optional map of task_id -> complexity (0.0-1.0).
            task_priorities: Optional map of task_id -> priority (1-10).

        Returns:
            Dictionary of task_id -> TaskBudget with allocations.
        """
        with self._lock:
            tasks = self._extract_tasks(plan)
            if not tasks:
                logger.warning("No tasks found in plan for budget allocation")
                return {}

            complexities = task_complexities or {}
            priorities = task_priorities or {}

            # Estimate complexity from task descriptions if not provided
            for tid, desc in tasks.items():
                if tid not in complexities:
                    complexities[tid] = self._estimate_complexity(desc)
                if tid not in priorities:
                    priorities[tid] = 5

            # Allocate based on strategy
            allocations = self._compute_allocations(tasks, complexities, priorities)

            # Create TaskBudget objects
            for tid, tokens in allocations.items():
                reserved = int(tokens * self._reserve_ratio)
                depth = self._select_depth_mode(tokens, complexities.get(tid, 0.5))
                self._task_budgets[tid] = TaskBudget(
                    task_id=tid,
                    allocated_tokens=tokens,
                    reserved_tokens=reserved,
                    depth_mode=depth,
                    priority=priorities.get(tid, 5),
                    complexity_score=complexities.get(tid, 0.5),
                )

            self._take_snapshot()
            logger.info(
                "Budget allocated: %d tasks, %d total tokens, %d allocated",
                len(tasks), self._total_tokens, sum(allocations.values()),
            )
            return dict(self._task_budgets)

    def _compute_allocations(
        self,
        tasks: Dict[str, str],
        complexities: Dict[str, float],
        priorities: Dict[str, int],
    ) -> Dict[str, int]:
        """Compute token allocations based on strategy."""
        n = len(tasks)
        if n == 0:
            return {}

        available = self._total_tokens - self._used_total

        if self._strategy == BudgetStrategy.EQUAL:
            per_task = available // n
            return {tid: per_task for tid in tasks}

        elif self._strategy == BudgetStrategy.PRIORITY:
            # Invert priority (1=highest -> highest weight)
            weights = {tid: (11 - priorities.get(tid, 5)) for tid in tasks}
            total_weight = sum(weights.values()) or 1
            return {tid: int(available * (w / total_weight)) for tid, w in weights.items()}

        else:  # PROPORTIONAL or ADAPTIVE
            # Weight by complexity score
            weights = {tid: max(0.1, complexities.get(tid, 0.5)) for tid in tasks}
            total_weight = sum(weights.values()) or 1
            return {tid: int(available * (w / total_weight)) for tid, w in weights.items()}

    def _estimate_complexity(self, description: str) -> float:
        """Heuristic complexity estimation from task description."""
        desc_lower = description.lower()
        score = 0.5

        # Complexity indicators
        high_indicators = ["implement", "architect", "design", "refactor", "optimize", "integrate", "debug"]
        low_indicators = ["list", "check", "verify", "format", "rename", "update comment"]

        for word in high_indicators:
            if word in desc_lower:
                score += 0.1
        for word in low_indicators:
            if word in desc_lower:
                score -= 0.1

        # Length heuristic: longer descriptions often mean more complex tasks
        if len(description) > 200:
            score += 0.1
        elif len(description) < 50:
            score -= 0.1

        return max(0.1, min(1.0, score))

    def _select_depth_mode(self, allocated_tokens: int, complexity: float) -> str:
        """Select reasoning depth based on token budget and complexity."""
        if allocated_tokens > 20000:
            return DepthMode.DEEP.value
        elif allocated_tokens > 8000:
            return DepthMode.STANDARD.value
        elif allocated_tokens > 3000:
            return DepthMode.SHALLOW.value
        else:
            return DepthMode.MINIMAL.value

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    def track_usage(self, task_id: str, tokens_used: int) -> TaskBudget:
        """
        Record token usage for a task.

        Args:
            task_id: The task identifier.
            tokens_used: Number of tokens consumed.

        Returns:
            Updated TaskBudget for the task.

        Raises:
            KeyError: If task_id has no budget allocation.
        """
        with self._lock:
            if task_id not in self._task_budgets:
                # Auto-create a budget entry for untracked tasks
                self._task_budgets[task_id] = TaskBudget(
                    task_id=task_id,
                    allocated_tokens=max(1000, self._total_tokens // 20),
                )
                logger.warning("Auto-created budget for untracked task: %s", task_id)

            budget = self._task_budgets[task_id]
            budget.used_tokens += tokens_used
            budget.started = True
            self._used_total += tokens_used

            # Check if task exceeded its budget
            if budget.used_tokens > budget.allocated_tokens:
                budget.exceeded = True
                logger.warning(
                    "Task %s exceeded budget: %d/%d tokens",
                    task_id, budget.used_tokens, budget.allocated_tokens,
                )

            # Check global budget threshold for re-planning
            global_ratio = self._used_total / self._total_tokens if self._total_tokens > 0 else 0
            if global_ratio >= self._replan_threshold and not self._replan_triggered:
                self._trigger_replan(
                    f"Global budget usage at {global_ratio:.0%} "
                    f"(threshold: {self._replan_threshold:.0%})"
                )

            return budget

    def check_remaining(self, task_id: str) -> int:
        """
        Check remaining token budget for a task.

        Args:
            task_id: The task identifier.

        Returns:
            Number of remaining tokens, or 0 if task not found.
        """
        with self._lock:
            budget = self._task_budgets.get(task_id)
            if budget is None:
                return 0
            return budget.remaining_tokens

    def mark_completed(self, task_id: str) -> Optional[TaskBudget]:
        """Mark a task as completed and reclaim unused tokens."""
        with self._lock:
            budget = self._task_budgets.get(task_id)
            if budget is None:
                return None
            budget.completed = True
            # Reclaimed tokens could be redistributed
            reclaimed = budget.remaining_tokens
            if reclaimed > 0:
                logger.debug("Task %s completed: %d tokens reclaimed", task_id, reclaimed)
            return budget

    # ------------------------------------------------------------------
    # Dynamic strategy adaptation
    # ------------------------------------------------------------------

    def get_depth_recommendation(self, task_id: str) -> str:
        """
        Get the recommended reasoning depth for a task based on remaining budget.

        Adapts in real-time: if the task is consuming more than expected,
        the depth downgrades to conserve tokens.

        Args:
            task_id: The task identifier.

        Returns:
            Depth mode string: "deep", "standard", "shallow", or "minimal".
        """
        with self._lock:
            budget = self._task_budgets.get(task_id)
            if budget is None:
                return DepthMode.STANDARD.value

            remaining = budget.remaining_tokens
            original = budget.allocated_tokens

            if original == 0:
                return DepthMode.MINIMAL.value

            ratio = remaining / original

            if ratio > 0.7:
                return DepthMode.DEEP.value
            elif ratio > 0.4:
                return DepthMode.STANDARD.value
            elif ratio > 0.15:
                return DepthMode.SHALLOW.value
            else:
                return DepthMode.MINIMAL.value

    def redistribute_budget(self, from_task_id: str, to_task_id: str, tokens: int) -> bool:
        """
        Redistribute tokens from one task to another.

        Useful when a completed task has surplus or a critical task needs more.

        Returns:
            True if redistribution succeeded.
        """
        with self._lock:
            from_budget = self._task_budgets.get(from_task_id)
            to_budget = self._task_budgets.get(to_task_id)

            if from_budget is None or to_budget is None:
                logger.warning("Cannot redistribute: task not found")
                return False

            available = from_budget.remaining_tokens
            transfer = min(tokens, available)
            if transfer <= 0:
                logger.warning("No tokens available to redistribute from %s", from_task_id)
                return False

            from_budget.allocated_tokens -= transfer
            to_budget.allocated_tokens += transfer
            logger.info(
                "Redistributed %d tokens: %s -> %s",
                transfer, from_task_id, to_task_id,
            )
            return True

    # ------------------------------------------------------------------
    # Re-planning
    # ------------------------------------------------------------------

    def _trigger_replan(self, reason: str) -> None:
        """Trigger re-planning when budget thresholds are exceeded."""
        self._replan_triggered = True
        logger.warning("Re-planning triggered: %s", reason)
        self._take_snapshot(replan_triggered=True)

        if self._on_replan:
            try:
                self._on_replan(reason, self.get_summary())
            except Exception as exc:
                logger.error("Re-plan callback failed: %s", exc)

    def should_replan(self) -> Tuple[bool, str]:
        """
        Check whether re-planning should be triggered.

        Returns:
            Tuple of (should_replan: bool, reason: str).
        """
        with self._lock:
            if self._total_tokens == 0:
                return False, ""

            global_ratio = self._used_total / self._total_tokens
            if global_ratio >= self._replan_threshold:
                return True, f"Global budget at {global_ratio:.0%}"

            exceeded_tasks = [b for b in self._task_budgets.values() if b.exceeded]
            if len(exceeded_tasks) > len(self._task_budgets) * 0.3:
                return True, f"{len(exceeded_tasks)} tasks exceeded budget"

            return False, ""

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current budget state."""
        with self._lock:
            budgets = list(self._task_budgets.values())
            return {
                "total_tokens": self._total_tokens,
                "used_tokens": self._used_total,
                "remaining_tokens": self._total_tokens - self._used_total,
                "usage_ratio": self._used_total / self._total_tokens if self._total_tokens > 0 else 0,
                "strategy": self._strategy.value,
                "tasks_total": len(budgets),
                "tasks_started": sum(1 for b in budgets if b.started),
                "tasks_completed": sum(1 for b in budgets if b.completed),
                "tasks_exceeded": sum(1 for b in budgets if b.exceeded),
                "replan_triggered": self._replan_triggered,
                "task_budgets": {b.task_id: b.to_dict() for b in budgets},
            }

    def get_task_budget(self, task_id: str) -> Optional[TaskBudget]:
        """Get the budget for a specific task."""
        return self._task_budgets.get(task_id)

    def _take_snapshot(self, replan_triggered: bool = False) -> None:
        """Record a point-in-time budget snapshot."""
        budgets = list(self._task_budgets.values())
        snapshot = BudgetSnapshot(
            total_tokens=self._total_tokens,
            allocated_tokens=sum(b.allocated_tokens for b in budgets),
            used_tokens=self._used_total,
            remaining_tokens=self._total_tokens - self._used_total,
            tasks_completed=sum(1 for b in budgets if b.completed),
            tasks_exceeded=sum(1 for b in budgets if b.exceeded),
            replan_triggered=replan_triggered,
        )
        self._snapshots.append(snapshot)
        # Keep bounded
        if len(self._snapshots) > 500:
            self._snapshots = self._snapshots[-250:]

    def _extract_tasks(self, plan: Any) -> Dict[str, str]:
        """Extract task_id -> description mapping from a plan object or list."""
        tasks: Dict[str, str] = {}

        if isinstance(plan, dict):
            for item in plan.get("tasks", []):
                tid = item.get("id", item.get("task_id", str(uuid.uuid4().hex[:8])))
                desc = item.get("description", "")
                tasks[tid] = desc
        elif isinstance(plan, list):
            for item in plan:
                if isinstance(item, dict):
                    tid = item.get("id", item.get("task_id", str(uuid.uuid4().hex[:8])))
                    desc = item.get("description", "")
                    tasks[tid] = desc
                elif hasattr(item, "id"):
                    tasks[item.id] = getattr(item, "description", "")
                elif hasattr(item, "task_id"):
                    tasks[item.task_id] = getattr(item, "description", "")
        elif hasattr(plan, "tasks"):
            for task in plan.tasks:
                tid = getattr(task, "id", getattr(task, "task_id", str(uuid.uuid4().hex[:8])))
                desc = getattr(task, "description", "")
                tasks[tid] = desc

        return tasks
