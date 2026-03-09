import json
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from vetinari.agents.contracts import Task, TaskStatus

logger = logging.getLogger(__name__)


class PlanStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WaveStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class AgentType(Enum):
    EXPLORER = "explorer"
    LIBRARIAN = "librarian"
    ORACLE = "oracle"
    UI_PLANNER = "ui_planner"
    BUILDER = "builder"
    RESEARCHER = "researcher"
    EVALUATOR = "evaluator"
    SYNTHESIZER = "synthesizer"
    PONDER = "ponder"


@dataclass
class Wave:
    wave_id: str
    milestone: str
    description: str
    order: int
    status: str = WaveStatus.PENDING.value
    tasks: List[Task] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'wave_id': self.wave_id,
            'milestone': self.milestone,
            'description': self.description,
            'order': self.order,
            'status': self.status,
            'tasks': [t.to_dict() for t in self.tasks],
            'dependencies': self.dependencies
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Wave':
        tasks = [Task.from_dict(t) for t in data.get('tasks', [])]
        return cls(
            wave_id=data.get('wave_id', ''),
            milestone=data.get('milestone', ''),
            description=data.get('description', ''),
            order=data.get('order', 1),
            status=data.get('status', WaveStatus.PENDING.value),
            tasks=tasks,
            dependencies=data.get('dependencies', [])
        )

    @property
    def completed_count(self) -> int:
        return sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)

    @property
    def total_count(self) -> int:
        return len(self.tasks)


@dataclass
class Plan:
    plan_id: str
    title: str
    prompt: str
    created_by: str
    created_at: str
    updated_at: str
    status: str = PlanStatus.PENDING.value
    waves: List[Wave] = field(default_factory=list)
    max_depth_override: int = 0
    seed_mix: str = "50% Oracle, 25% Researcher, 25% Explorer"
    seed_rate: int = 2
    decomposed_depth: int = 0
    adr_history: List[dict] = field(default_factory=list)
    template_version: str = "v1"

    def to_dict(self) -> dict:
        return {
            'plan_id': self.plan_id,
            'title': self.title,
            'prompt': self.prompt,
            'created_by': self.created_by,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'status': self.status,
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'progress_percent': self.progress_percent,
            'max_depth_override': self.max_depth_override,
            'seed_mix': self.seed_mix,
            'seed_rate': self.seed_rate,
            'decomposed_depth': self.decomposed_depth,
            'adr_history': self.adr_history,
            'template_version': self.template_version,
            'waves': [w.to_dict() for w in self.waves]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Plan':
        waves = [Wave.from_dict(w) for w in data.get('waves', [])]
        plan = cls(
            plan_id=data.get('plan_id', ''),
            title=data.get('title', ''),
            prompt=data.get('prompt', ''),
            created_by=data.get('created_by', ''),
            created_at=data.get('created_at', ''),
            updated_at=data.get('updated_at', ''),
            status=data.get('status', PlanStatus.PENDING.value),
            waves=waves,
            max_depth_override=data.get('max_depth_override', 0),
            seed_mix=data.get('seed_mix', "50% Oracle, 25% Researcher, 25% Explorer"),
            seed_rate=data.get('seed_rate', 2),
            decomposed_depth=data.get('decomposed_depth', 0),
            adr_history=data.get('adr_history', []),
            template_version=data.get('template_version', 'v1')
        )
        return plan

    @property
    def total_tasks(self) -> int:
        return sum(len(w.tasks) for w in self.waves)

    @property
    def completed_tasks(self) -> int:
        return sum(w.completed_count for w in self.waves)

    @property
    def progress_percent(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return round((self.completed_tasks / self.total_tasks) * 100, 1)

    @property
    def current_wave(self) -> Optional[Wave]:
        for wave in self.waves:
            if wave.status == WaveStatus.RUNNING.value:
                return wave
        return None

    @property
    def effective_max_depth(self) -> int:
        if self.max_depth_override > 0:
            return max(12, min(16, self.max_depth_override))
        return 14

    def add_adr(self, adr_id: str, title: str, context: str, decision: str, status: str = "proposed"):
        adr = {
            "adr_id": adr_id,
            "title": title,
            "context": context,
            "decision": decision,
            "status": status,
            "created_at": datetime.now().isoformat()
        }
        self.adr_history.append(adr)
        return adr


def _to_task_status(raw: str) -> TaskStatus:
    """Convert a raw status string to a TaskStatus enum value."""
    try:
        return TaskStatus(raw)
    except ValueError:
        return TaskStatus.PENDING


class PlanManager:
    _instance = None

    @classmethod
    def get_instance(cls, storage_path: str = None):
        if cls._instance is None:
            cls._instance = cls(storage_path)
        return cls._instance

    def __init__(self, storage_path: str = None):
        if storage_path is None:
            storage_path = Path.home() / ".lmstudio" / "projects" / "Vetinari" / "plans"

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.plans: Dict[str, Plan] = {}
        self._load_plans()

    def _load_plans(self):
        for file in self.storage_path.glob("*.json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    plan = Plan.from_dict(data)
                    self.plans[plan.plan_id] = plan
            except Exception as e:
                logger.error(f"Error loading plan {file}: {e}")

    def _save_plan(self, plan: Plan):
        file_path = self.storage_path / f"{plan.plan_id}.json"
        with open(file_path, 'w') as f:
            json.dump(plan.to_dict(), f, indent=2)

    def create_plan(self, title: str, prompt: str, created_by: str = "user", waves_data: List[dict] = None) -> Plan:
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        now = datetime.now().isoformat()

        waves = []
        if waves_data:
            for i, wave_data in enumerate(waves_data):
                wave_id = f"wave_{i+1}"
                tasks = []
                for j, task_data in enumerate(wave_data.get('tasks', [])):
                    tid = f"task_{i+1}_{j+1}"
                    task = Task(
                        id=tid,
                        description=task_data.get('description', ''),
                        prompt=task_data.get('prompt', ''),
                        dependencies=task_data.get('dependencies', []),
                        priority=task_data.get('priority', 5),
                    )
                    tasks.append(task)

                wave = Wave(
                    wave_id=wave_id,
                    milestone=wave_data.get('milestone', f'Wave {i+1}'),
                    description=wave_data.get('description', ''),
                    order=i+1,
                    tasks=tasks,
                    dependencies=wave_data.get('dependencies', [])
                )
                waves.append(wave)

        plan = Plan(
            plan_id=plan_id,
            title=title,
            prompt=prompt,
            created_by=created_by,
            created_at=now,
            updated_at=now,
            waves=waves
        )

        self.plans[plan_id] = plan
        self._save_plan(plan)

        return plan

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        return self.plans.get(plan_id)

    def list_plans(self, status: str = None, limit: int = 50, offset: int = 0) -> List[Plan]:
        plans = list(self.plans.values())
        if status:
            plans = [p for p in plans if p.status == status]
        plans.sort(key=lambda p: p.created_at, reverse=True)
        return plans[offset:offset+limit]

    def update_plan(self, plan_id: str, updates: dict) -> Optional[Plan]:
        plan = self.plans.get(plan_id)
        if not plan:
            return None

        if 'title' in updates:
            plan.title = updates['title']
        if 'status' in updates:
            plan.status = updates['status']

        plan.updated_at = datetime.now().isoformat()
        self._save_plan(plan)
        return plan

    def delete_plan(self, plan_id: str) -> bool:
        if plan_id in self.plans:
            del self.plans[plan_id]
            file_path = self.storage_path / f"{plan_id}.json"
            if file_path.exists():
                file_path.unlink()
            return True
        return False

    def start_plan(self, plan_id: str) -> Optional[Plan]:
        plan = self.plans.get(plan_id)
        if not plan:
            return None

        plan.status = PlanStatus.ACTIVE.value
        plan.updated_at = datetime.now().isoformat()

        if plan.waves:
            plan.waves[0].status = WaveStatus.RUNNING.value
            for task in plan.waves[0].tasks:
                task.status = TaskStatus.PENDING

        self._save_plan(plan)
        return plan

    def pause_plan(self, plan_id: str) -> Optional[Plan]:
        plan = self.plans.get(plan_id)
        if not plan or plan.status != PlanStatus.ACTIVE.value:
            return None

        plan.status = PlanStatus.PAUSED.value
        plan.updated_at = datetime.now().isoformat()

        for wave in plan.waves:
            if wave.status == WaveStatus.RUNNING.value:
                wave.status = WaveStatus.BLOCKED.value

        self._save_plan(plan)
        return plan

    def resume_plan(self, plan_id: str) -> Optional[Plan]:
        plan = self.plans.get(plan_id)
        if not plan or plan.status != PlanStatus.PAUSED.value:
            return None

        plan.status = PlanStatus.ACTIVE.value
        plan.updated_at = datetime.now().isoformat()

        for wave in plan.waves:
            if wave.status == WaveStatus.BLOCKED.value:
                wave.status = WaveStatus.RUNNING.value

        self._save_plan(plan)
        return plan

    def cancel_plan(self, plan_id: str) -> Optional[Plan]:
        plan = self.plans.get(plan_id)
        if not plan:
            return None

        plan.status = PlanStatus.CANCELLED.value
        plan.updated_at = datetime.now().isoformat()

        for wave in plan.waves:
            if wave.status in [WaveStatus.PENDING.value, WaveStatus.RUNNING.value]:
                wave.status = WaveStatus.BLOCKED.value
            for task in wave.tasks:
                if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                    task.status = TaskStatus.BLOCKED

        self._save_plan(plan)
        return plan

    def update_task_status(self, plan_id: str, wave_id: str, task_id: str, status: str, result: Any = None, error: str = "") -> Optional[Plan]:
        plan = self.plans.get(plan_id)
        if not plan:
            return None

        task_status = _to_task_status(status)

        for wave in plan.waves:
            if wave.wave_id == wave_id:
                for task in wave.tasks:
                    if task.task_id == task_id:
                        task.status = task_status
                        if task_status == TaskStatus.RUNNING:
                            task.actual_start = datetime.now().isoformat()
                        elif task_status == TaskStatus.COMPLETED:
                            task.actual_end = datetime.now().isoformat()
                            task.result = result
                        elif task_status == TaskStatus.FAILED:
                            task.actual_end = datetime.now().isoformat()
                            task.error = error
                        break
                break

        self._check_wave_completion(plan, wave_id)
        self._save_plan(plan)
        return plan

    def _check_wave_completion(self, plan: Plan, completed_wave_id: str):
        for wave in plan.waves:
            if wave.wave_id == completed_wave_id:
                all_completed = all(t.status == TaskStatus.COMPLETED for t in wave.tasks)
                if all_completed:
                    wave.status = WaveStatus.COMPLETED.value

                    next_wave_idx = wave.order
                    if next_wave_idx < len(plan.waves):
                        next_wave = plan.waves[next_wave_idx]
                        if next_wave.status == WaveStatus.PENDING.value:
                            next_wave.status = WaveStatus.RUNNING.value
                            for task in next_wave.tasks:
                                if task.status == TaskStatus.PENDING:
                                    task.status = TaskStatus.PENDING
                    else:
                        plan.status = PlanStatus.COMPLETED.value

        plan.updated_at = datetime.now().isoformat()


def get_plan_manager() -> "PlanManager":
    """Lazily return the singleton PlanManager. Use this instead of module-level plan_manager."""
    return PlanManager.get_instance()


# Backward-compatible alias -- resolved lazily so importing this module does NOT
# trigger filesystem I/O at import time.
class _LazyPlanManager:
    """Proxy that resolves the PlanManager singleton on first attribute access."""
    def __getattr__(self, name):
        return getattr(PlanManager.get_instance(), name)

    def __repr__(self):
        return repr(PlanManager.get_instance())


plan_manager = _LazyPlanManager()
