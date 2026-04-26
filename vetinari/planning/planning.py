"""Planning module — LEGACY wave-based plan management.

.. deprecated::
    This module is deprecated. Use ``vetinari.plan_mode.PlanModeEngine``
    for plan generation and ``vetinari.orchestration`` for execution.

# CANONICAL: vetinari.planning.plan_mode
# This file is the DEPRECATED legacy implementation. New code must NOT import
# from here. The authoritative modules are:
#   - vetinari/planning/plan_mode.py    — plan generation (PlanModeEngine)
#   - vetinari/planning/plan_types.py   — planning domain types (Plan, Subtask, etc.)
#   - vetinari/planning/plan_api.py     — REST endpoints for plan operations
#
# This file is retained only for backward-compatibility of existing callers.
# It will be removed once all call sites have migrated.
"""

from __future__ import annotations

import json
import logging
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vetinari.constants import get_user_dir
from vetinari.planning.plan_types import Plan as CanonicalPlan
from vetinari.types import PlanStatus, StatusEnum  # canonical source
from vetinari.utils.serialization import dataclass_to_dict

warnings.warn(
    "vetinari.planning is deprecated. Its wave-based plan management "
    "will be migrated into vetinari.orchestration in a future release. "
    "Use vetinari.plan_mode.PlanModeEngine for plan generation.",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)


# Decision: WaveStatus consolidated into StatusEnum in vetinari.types (ADR-0075).
# WaveStatus values are a subset of StatusEnum: PENDING, RUNNING, COMPLETED, FAILED, BLOCKED.
WaveStatus = StatusEnum  # Backward-compat alias — use StatusEnum directly in new code


@dataclass
class PlanTask:
    """A unit of work within a plan or wave."""

    task_id: str
    agent_type: str  # AgentType.value string — use AgentType enum at call sites
    description: str
    prompt: str
    status: str = StatusEnum.PENDING.value
    dependencies: list[str] = field(default_factory=list)
    assigned_agent: str = ""
    result: Any = None
    error: str = ""
    planned_start: str = ""
    planned_end: str = ""
    actual_start: str = ""
    actual_end: str = ""
    retry_count: int = 0
    priority: int = 5
    estimated_effort: float = 1.0
    parent_id: str = ""
    depth: int = 0
    max_depth: int = 14
    max_depth_override: int = 0
    subtasks: list[PlanTask] = field(default_factory=list)
    decomposition_seed: str = ""
    dod_level: str = "Standard"
    dor_level: str = "Standard"
    wave_id: str = ""

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"PlanTask(task_id={self.task_id!r}, agent_type={self.agent_type!r}, status={self.status!r})"

    def to_dict(self) -> dict:
        """Serialize all fields to a JSON-serializable dict, with nested subtasks recursively expanded."""
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PlanTask:
        """From dict.

        Returns:
            The PlanTask result.
        """
        subtasks = [PlanTask.from_dict(t) for t in data.get("subtasks", [])]
        return cls(
            task_id=data.get("task_id", ""),
            agent_type=data.get("agent_type", "builder"),
            description=data.get("description", ""),
            prompt=data.get("prompt", ""),
            status=data.get("status", StatusEnum.PENDING.value),
            dependencies=data.get("dependencies", []),
            assigned_agent=data.get("assigned_agent", ""),
            result=data.get("result"),
            error=data.get("error", ""),
            planned_start=data.get("planned_start", ""),
            planned_end=data.get("planned_end", ""),
            actual_start=data.get("actual_start", ""),
            actual_end=data.get("actual_end", ""),
            retry_count=data.get("retry_count", 0),
            priority=data.get("priority", 5),
            estimated_effort=data.get("estimated_effort", 1.0),
            parent_id=data.get("parent_id", ""),
            depth=data.get("depth", 0),
            max_depth=data.get("max_depth", 14),
            max_depth_override=data.get("max_depth_override", 0),
            subtasks=subtasks,
            decomposition_seed=data.get("decomposition_seed", ""),
            dod_level=data.get("dod_level", "Standard"),
            dor_level=data.get("dor_level", "Standard"),
            wave_id=data.get("wave_id", ""),
        )


@dataclass
class Wave:
    """A group of tasks executed in parallel."""

    wave_id: str
    milestone: str
    description: str
    order: int
    status: str = WaveStatus.PENDING.value
    tasks: list[PlanTask] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"Wave(wave_id={self.wave_id!r}, status={self.status!r}, order={self.order!r})"

    def to_dict(self) -> dict:
        """Serialize all fields to a JSON-serializable dict, with nested tasks recursively expanded."""
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Wave:
        """From dict.

        Returns:
            The Wave result.
        """
        tasks = [PlanTask.from_dict(t) for t in data.get("tasks", [])]
        return cls(
            wave_id=data.get("wave_id", ""),
            milestone=data.get("milestone", ""),
            description=data.get("description", ""),
            order=data.get("order", 1),
            status=data.get("status", WaveStatus.PENDING.value),
            tasks=tasks,
            dependencies=data.get("dependencies", []),
        )

    @property
    def completed_count(self) -> int:
        """Count the number of tasks in this wave that have completed successfully.

        Returns:
            Number of tasks with a completed status.
        """
        return sum(1 for t in self.tasks if t.status == StatusEnum.COMPLETED.value)

    @property
    def total_count(self) -> int:
        """Return the total number of tasks assigned to this wave.

        Returns:
            Total task count in the wave.
        """
        return len(self.tasks)


@dataclass
class PlanningExecutionPlan:
    """An execution plan containing ordered waves of tasks."""

    plan_id: str
    title: str
    prompt: str
    created_by: str
    created_at: str
    updated_at: str
    status: str = PlanStatus.PENDING.value
    waves: list[Wave] = field(default_factory=list)
    max_depth_override: int = 0
    seed_mix: str = "50% Oracle, 25% Researcher, 25% Explorer"
    seed_rate: int = 2
    decomposed_depth: int = 0
    adr_history: list[dict] = field(default_factory=list)
    template_version: str = "v1"

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"ExecutionPlan(plan_id={self.plan_id!r}, status={self.status!r}, waves={len(self.waves)!r})"

    def to_dict(self) -> dict:
        """Serialize all dataclass fields plus computed progress metrics to a JSON-serializable dict.

        Returns:
            Dictionary of all persisted fields merged with ``total_tasks``,
            ``completed_tasks``, and ``progress_percent`` computed at call time.
        """
        data = dataclass_to_dict(self)
        data["total_tasks"] = self.total_tasks
        data["completed_tasks"] = self.completed_tasks
        data["progress_percent"] = self.progress_percent
        return data

    @classmethod
    def from_dict(cls, data: dict) -> PlanningExecutionPlan:
        """From dict.

        Returns:
            The ExecutionPlan result.
        """
        waves = [Wave.from_dict(w) for w in data.get("waves", [])]
        return cls(
            plan_id=data.get("plan_id", ""),
            title=data.get("title", ""),
            prompt=data.get("prompt", ""),
            created_by=data.get("created_by", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            status=data.get("status", PlanStatus.PENDING.value),
            waves=waves,
            max_depth_override=data.get("max_depth_override", 0),
            seed_mix=data.get("seed_mix", "50% Oracle, 25% Researcher, 25% Explorer"),
            seed_rate=data.get("seed_rate", 2),
            decomposed_depth=data.get("decomposed_depth", 0),
            adr_history=data.get("adr_history", []),
            template_version=data.get("template_version", "v1"),
        )

    @property
    def total_tasks(self) -> int:
        """Return the total number of tasks across all waves in the plan.

        Returns:
            Aggregate task count summed from every wave.
        """
        return sum(len(w.tasks) for w in self.waves)

    @property
    def completed_tasks(self) -> int:
        """Return the number of completed tasks across all waves in the plan.

        Returns:
            Aggregate count of tasks with a completed status.
        """
        return sum(w.completed_count for w in self.waves)

    @property
    def progress_percent(self) -> float:
        """Calculate overall plan completion as a percentage.

        Returns:
            Percentage of completed tasks rounded to one decimal place,
            or 0.0 if the plan has no tasks.
        """
        if self.total_tasks == 0:
            return 0.0
        return round((self.completed_tasks / self.total_tasks) * 100, 1)

    @property
    def current_wave(self) -> Wave | None:
        """Return the wave that is currently running, if any.

        Returns:
            The first wave with a running status, or None if no wave is active.
        """
        for wave in self.waves:
            if wave.status == WaveStatus.RUNNING.value:
                return wave
        return None

    @property
    def effective_max_depth(self) -> int:
        """Return the effective maximum decomposition depth for this plan.

        When a max_depth_override is set, it is clamped to the range 12-16.
        Otherwise the default depth of 14 is used.

        Returns:
            The clamped override depth or the default depth of 14.
        """
        if self.max_depth_override > 0:
            return max(12, min(16, self.max_depth_override))
        return 14

    def add_adr(self, adr_id: str, title: str, context: str, decision: str, status: str = "proposed") -> dict[str, str]:
        """Record an Architecture Decision Record reference in this plan's history.

        Args:
            adr_id: Unique ADR identifier (e.g. ``ADR-0042``).
            title: Short human-readable title of the decision.
            context: Problem statement and constraints that prompted the decision.
            decision: The chosen approach.
            status: ADR lifecycle status (default ``proposed``).

        Returns:
            The newly created ADR dictionary with all fields plus a timestamp.
        """
        adr = {
            "adr_id": adr_id,
            "title": title,
            "context": context,
            "decision": decision,
            "status": status,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.adr_history.append(adr)
        return adr


# Backward-compatible export for callers that historically imported
# ``vetinari.planning.planning.Plan`` after the canonical Plan migration.
Plan = CanonicalPlan


class PlanManager:
    """Plan manager."""

    _instance = None

    @classmethod
    def get_instance(cls, storage_path: str | None = None) -> PlanManager:
        """Get instance.

        Returns:
            The PlanManager result.
        """
        if cls._instance is None:
            cls._instance = cls(storage_path)
        return cls._instance

    def __init__(self, storage_path: str | None = None):
        if storage_path is None:
            storage_path = get_user_dir() / "plans"

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.plans: dict[str, PlanningExecutionPlan] = {}
        self._load_plans()

    def _load_plans(self):
        for file in self.storage_path.glob("*.json"):
            try:
                with Path(file).open(encoding="utf-8") as f:
                    data = json.load(f)
                    plan = PlanningExecutionPlan.from_dict(data)
                    self.plans[plan.plan_id] = plan
            except Exception as e:
                logger.error("Error loading plan %s: %s", file, e)

    def _save_plan(self, plan: PlanningExecutionPlan) -> None:
        """Persist a plan to its JSON file.

        Args:
            plan: The Plan to save.

        Raises:
            ValueError: If the plan ID contains path traversal sequences that
                would place the file outside the configured storage directory.
        """
        target = (self.storage_path / f"{plan.plan_id}.json").resolve()
        if not target.is_relative_to(self.storage_path.resolve()):
            raise ValueError(f"Plan ID contains path traversal: {plan.plan_id}")
        with target.open("w", encoding="utf-8") as f:
            json.dump(plan.to_dict(), f, indent=2)

    def create_plan(
        self,
        title: str,
        prompt: str,
        created_by: str = "user",
        waves_data: list[dict] | None = None,
    ) -> PlanningExecutionPlan:
        """Create plan.

        Args:
            title: The title.
            prompt: The prompt.
            created_by: The created by.
            waves_data: The waves data.

        Returns:
            The Plan result.
        """
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        waves = []
        if waves_data:
            for i, wave_data in enumerate(waves_data):
                wave_id = f"wave_{i + 1}"
                tasks = []
                for j, task_data in enumerate(wave_data.get("tasks", [])):
                    task_id = f"task_{i + 1}_{j + 1}"
                    task = PlanTask(
                        task_id=task_id,
                        agent_type=task_data.get("agent_type", "builder"),
                        description=task_data.get("description", ""),
                        prompt=task_data.get("prompt", ""),
                        dependencies=task_data.get("dependencies", []),
                        priority=task_data.get("priority", 5),
                    )
                    tasks.append(task)

                wave = Wave(
                    wave_id=wave_id,
                    milestone=wave_data.get("milestone", f"Wave {i + 1}"),
                    description=wave_data.get("description", ""),
                    order=i + 1,
                    tasks=tasks,
                    dependencies=wave_data.get("dependencies", []),
                )
                waves.append(wave)

        plan = PlanningExecutionPlan(
            plan_id=plan_id,
            title=title,
            prompt=prompt,
            created_by=created_by,
            created_at=now,
            updated_at=now,
            waves=waves,
        )

        self.plans[plan_id] = plan
        self._save_plan(plan)

        return plan

    def get_plan(self, plan_id: str) -> PlanningExecutionPlan | None:
        """Retrieve a plan by its unique identifier.

        Args:
            plan_id: The unique identifier of the plan to retrieve.

        Returns:
            The matching Plan instance, or None if no plan exists with that id.
        """
        return self.plans.get(plan_id)

    def list_plans(
        self, status: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[PlanningExecutionPlan]:
        """List plans.

        Args:
            status: The status.
            limit: The limit.
            offset: The offset.

        Returns:
            List of results.
        """
        plans = list(self.plans.values())
        if status:
            plans = [p for p in plans if p.status == status]
        plans.sort(key=lambda p: p.created_at, reverse=True)
        return plans[offset : offset + limit]

    def update_plan(self, plan_id: str, updates: dict) -> PlanningExecutionPlan | None:
        """Update plan.

        Args:
            plan_id: The plan id.
            updates: The updates.

        Returns:
            The Plan | None result.
        """
        plan = self.plans.get(plan_id)
        if not plan:
            return None

        if "title" in updates:
            plan.title = updates["title"]
        if "status" in updates:
            plan.status = updates["status"]

        plan.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_plan(plan)
        return plan

    def delete_plan(self, plan_id: str) -> bool:
        """Delete plan.

        Args:
            plan_id: The plan id to delete.

        Returns:
            True if successful, False otherwise.

        Raises:
            ValueError: If the plan ID contains path traversal sequences that
                would place the file outside the configured storage directory.
        """
        target = (self.storage_path / f"{plan_id}.json").resolve()
        if not target.is_relative_to(self.storage_path.resolve()):
            raise ValueError(f"Plan ID contains path traversal: {plan_id}")
        if plan_id in self.plans:
            del self.plans[plan_id]
            if target.exists():
                target.unlink()

            # Also clean up any associated subtask tree so orphaned subtask
            # JSON files don't accumulate on disk.
            try:
                from vetinari.planning.subtask_tree import subtask_tree

                subtask_tree.delete_tree(plan_id)
            except Exception:
                logger.warning(
                    "SubtaskTree.delete_tree skipped for plan %s — subtask data may need manual cleanup",
                    plan_id,
                )

            return True
        return False

    def start_plan(self, plan_id: str) -> PlanningExecutionPlan | None:
        """Start plan.

        Returns:
            The Plan | None result.
        """
        plan = self.plans.get(plan_id)
        if not plan:
            return None

        plan.status = PlanStatus.EXECUTING.value
        plan.updated_at = datetime.now(timezone.utc).isoformat()

        if plan.waves:
            plan.waves[0].status = WaveStatus.RUNNING.value
            for task in plan.waves[0].tasks:
                task.status = StatusEnum.PENDING.value

        self._save_plan(plan)
        return plan

    def pause_plan(self, plan_id: str) -> PlanningExecutionPlan | None:
        """Pause plan.

        Returns:
            The Plan | None result.
        """
        plan = self.plans.get(plan_id)
        if not plan or plan.status != PlanStatus.EXECUTING.value:
            return None

        plan.status = PlanStatus.PAUSED.value
        plan.updated_at = datetime.now(timezone.utc).isoformat()

        for wave in plan.waves:
            if wave.status == WaveStatus.RUNNING.value:
                wave.status = WaveStatus.BLOCKED.value

        self._save_plan(plan)
        return plan

    def resume_plan(self, plan_id: str) -> PlanningExecutionPlan | None:
        """Resume plan.

        Returns:
            The Plan | None result.
        """
        plan = self.plans.get(plan_id)
        if not plan or plan.status != PlanStatus.PAUSED.value:
            return None

        plan.status = PlanStatus.EXECUTING.value
        plan.updated_at = datetime.now(timezone.utc).isoformat()

        for wave in plan.waves:
            if wave.status == WaveStatus.BLOCKED.value:
                wave.status = WaveStatus.RUNNING.value

        self._save_plan(plan)
        return plan

    def cancel_plan(self, plan_id: str) -> PlanningExecutionPlan | None:
        """Cancel plan.

        Returns:
            The Plan | None result.
        """
        plan = self.plans.get(plan_id)
        if not plan:
            return None

        plan.status = PlanStatus.CANCELLED.value
        plan.updated_at = datetime.now(timezone.utc).isoformat()

        for wave in plan.waves:
            if wave.status in [WaveStatus.PENDING.value, WaveStatus.RUNNING.value]:
                wave.status = WaveStatus.BLOCKED.value
            for task in wave.tasks:
                if task.status in [StatusEnum.PENDING.value, StatusEnum.RUNNING.value]:
                    task.status = StatusEnum.BLOCKED.value

        self._save_plan(plan)
        return plan

    def update_task_status(
        self,
        plan_id: str,
        wave_id: str,
        task_id: str,
        status: str,
        result: Any = None,
        error: str = "",
    ) -> PlanningExecutionPlan | None:
        """Update task status.

        Args:
            plan_id: The plan id.
            wave_id: The wave id.
            task_id: The task id.
            status: The status.
            result: The result.
            error: The error.

        Returns:
            The Plan | None result.
        """
        plan = self.plans.get(plan_id)
        if not plan:
            return None

        for wave in plan.waves:
            if wave.wave_id == wave_id:
                for task in wave.tasks:
                    if task.task_id == task_id:
                        task.status = status
                        if status == StatusEnum.RUNNING.value:
                            task.actual_start = datetime.now(timezone.utc).isoformat()
                        elif status == StatusEnum.COMPLETED.value:
                            task.actual_end = datetime.now(timezone.utc).isoformat()
                            task.result = result
                        elif status == StatusEnum.FAILED.value:
                            task.actual_end = datetime.now(timezone.utc).isoformat()
                            task.error = error
                        break
                break

        self._check_wave_completion(plan, wave_id)
        self._save_plan(plan)
        return plan

    def _check_wave_completion(self, plan: PlanningExecutionPlan, completed_wave_id: str):
        for wave in plan.waves:
            if wave.wave_id == completed_wave_id:
                all_completed = all(t.status == StatusEnum.COMPLETED.value for t in wave.tasks)
                if all_completed:
                    wave.status = WaveStatus.COMPLETED.value

                    next_wave_idx = wave.order
                    if next_wave_idx < len(plan.waves):
                        next_wave = plan.waves[next_wave_idx]
                        if next_wave.status == WaveStatus.PENDING.value:
                            next_wave.status = WaveStatus.RUNNING.value
                            for task in next_wave.tasks:
                                if task.status == StatusEnum.PENDING.value:
                                    task.status = StatusEnum.PENDING.value
                    else:
                        plan.status = PlanStatus.COMPLETED.value

        plan.updated_at = datetime.now(timezone.utc).isoformat()


def get_plan_manager() -> PlanManager:
    """Lazily return the singleton PlanManager. Use this instead of module-level plan_manager."""
    return PlanManager.get_instance()


# Backward-compatible alias — resolved lazily so importing this module does NOT
# trigger filesystem I/O at import time.
class _LazyPlanManager:
    """Proxy that resolves the PlanManager singleton on first attribute access.

    Deprecated: Use ``vetinari.planning.plan_mode.PlanModeEngine`` directly.
    """

    def __getattr__(self, name):
        warnings.warn(
            "_LazyPlanManager is deprecated. Use vetinari.planning.plan_mode.PlanModeEngine directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(PlanManager.get_instance(), name)

    def __repr__(self):
        return repr(PlanManager.get_instance())


plan_manager = _LazyPlanManager()
