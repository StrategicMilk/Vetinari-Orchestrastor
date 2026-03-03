import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path


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


class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
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
class Task:
    task_id: str
    agent_type: str
    description: str
    prompt: str
    status: str = TaskStatus.PENDING.value
    dependencies: List[str] = field(default_factory=list)
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
    subtasks: List['Task'] = field(default_factory=list)
    decomposition_seed: str = ""
    dod_level: str = "Standard"
    dor_level: str = "Standard"
    wave_id: str = ""

    def to_dict(self) -> dict:
        return {
            'task_id': self.task_id,
            'agent_type': self.agent_type,
            'description': self.description,
            'prompt': self.prompt,
            'status': self.status,
            'dependencies': self.dependencies,
            'assigned_agent': self.assigned_agent,
            'result': self.result,
            'error': self.error,
            'planned_start': self.planned_start,
            'planned_end': self.planned_end,
            'actual_start': self.actual_start,
            'actual_end': self.actual_end,
            'retry_count': self.retry_count,
            'priority': self.priority,
            'estimated_effort': self.estimated_effort,
            'parent_id': self.parent_id,
            'depth': self.depth,
            'max_depth': self.max_depth,
            'max_depth_override': self.max_depth_override,
            'subtasks': [t.to_dict() for t in self.subtasks],
            'decomposition_seed': self.decomposition_seed,
            'dod_level': self.dod_level,
            'dor_level': self.dor_level,
            'wave_id': self.wave_id
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Task':
        subtasks = [Task.from_dict(t) for t in data.get('subtasks', [])]
        return cls(
            task_id=data.get('task_id', ''),
            agent_type=data.get('agent_type', 'builder'),
            description=data.get('description', ''),
            prompt=data.get('prompt', ''),
            status=data.get('status', TaskStatus.PENDING.value),
            dependencies=data.get('dependencies', []),
            assigned_agent=data.get('assigned_agent', ''),
            result=data.get('result'),
            error=data.get('error', ''),
            planned_start=data.get('planned_start', ''),
            planned_end=data.get('planned_end', ''),
            actual_start=data.get('actual_start', ''),
            actual_end=data.get('actual_end', ''),
            retry_count=data.get('retry_count', 0),
            priority=data.get('priority', 5),
            estimated_effort=data.get('estimated_effort', 1.0),
            parent_id=data.get('parent_id', ''),
            depth=data.get('depth', 0),
            max_depth=data.get('max_depth', 14),
            max_depth_override=data.get('max_depth_override', 0),
            subtasks=subtasks,
            decomposition_seed=data.get('decomposition_seed', ''),
            dod_level=data.get('dod_level', 'Standard'),
            dor_level=data.get('dor_level', 'Standard'),
            wave_id=data.get('wave_id', '')
        )


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
        return sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED.value)

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
                print(f"Error loading plan {file}: {e}")

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
                    task_id = f"task_{i+1}_{j+1}"
                    task = Task(
                        task_id=task_id,
                        agent_type=task_data.get('agent_type', 'builder'),
                        description=task_data.get('description', ''),
                        prompt=task_data.get('prompt', ''),
                        dependencies=task_data.get('dependencies', []),
                        priority=task_data.get('priority', 5)
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
                task.status = TaskStatus.PENDING.value

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
                if task.status in [TaskStatus.PENDING.value, TaskStatus.RUNNING.value]:
                    task.status = TaskStatus.BLOCKED.value

        self._save_plan(plan)
        return plan

    def update_task_status(self, plan_id: str, wave_id: str, task_id: str, status: str, result: Any = None, error: str = "") -> Optional[Plan]:
        plan = self.plans.get(plan_id)
        if not plan:
            return None

        for wave in plan.waves:
            if wave.wave_id == wave_id:
                for task in wave.tasks:
                    if task.task_id == task_id:
                        task.status = status
                        if status == TaskStatus.RUNNING.value:
                            task.actual_start = datetime.now().isoformat()
                        elif status == TaskStatus.COMPLETED.value:
                            task.actual_end = datetime.now().isoformat()
                            task.result = result
                        elif status == TaskStatus.FAILED.value:
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
                all_completed = all(t.status == TaskStatus.COMPLETED.value for t in wave.tasks)
                if all_completed:
                    wave.status = WaveStatus.COMPLETED.value

                    next_wave_idx = wave.order
                    if next_wave_idx < len(plan.waves):
                        next_wave = plan.waves[next_wave_idx]
                        if next_wave.status == WaveStatus.PENDING.value:
                            next_wave.status = WaveStatus.RUNNING.value
                            for task in next_wave.tasks:
                                if task.status == TaskStatus.PENDING.value:
                                    task.status = TaskStatus.PENDING.value
                    else:
                        plan.status = PlanStatus.COMPLETED.value

        plan.updated_at = datetime.now().isoformat()


plan_manager = PlanManager.get_instance()
