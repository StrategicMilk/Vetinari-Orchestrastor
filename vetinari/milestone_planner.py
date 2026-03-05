import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import uuid
from vetinari.config import get_data_dir

logger = logging.getLogger(__name__)

class MilestoneStatus(Enum):
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ON_HOLD = "on_hold"
    CANCELLED = "cancelled"

class PhaseStatus(Enum):
    PENDING = "pending"
    DISCUSSING = "discussing"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"

@dataclass
class Milestone:
    milestone_id: str
    name: str
    description: str
    status: str
    phases: List[str] = field(default_factory=list)
    created_at: str = ""
    completed_at: str = ""
    definition_of_done: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class Phase:
    phase_id: str
    milestone_id: str
    name: str
    description: str
    status: str
    plans: List[Dict] = field(default_factory=list)
    context_file: str = ""
    research_file: str = ""
    created_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class Plan:
    plan_id: str
    phase_id: str
    name: str
    description: str
    files: List[str] = field(default_factory=list)
    action: str = ""
    verify: str = ""
    done: str = ""
    status: str = "pending"
    wave: int = 1
    dependencies: List[str] = field(default_factory=list)
    created_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

class MilestonePlanner:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.storage_path = get_data_dir() / "projects" / project_id / "milestones"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.milestones: List[Milestone] = []
        self.phases: List[Phase] = []
        self.plans: List[Plan] = []
        
        self._load()

    def _load(self):
        milestones_file = self.storage_path / "milestones.json"
        if milestones_file.exists():
            try:
                with open(milestones_file, 'r') as f:
                    data = json.load(f)
                    self.milestones = [Milestone(**m) for m in data.get("milestones", [])]
                    self.phases = [Phase(**p) for p in data.get("phases", [])]
                    self.plans = [Plan(**p) for p in data.get("plans", [])]
            except Exception as e:
                logger.warning(f"Could not load milestones: {e}")

    def _save(self):
        milestones_file = self.storage_path / "milestones.json"
        try:
            with open(milestones_file, 'w') as f:
                json.dump({
                    "milestones": [m.to_dict() for m in self.milestones],
                    "phases": [p.to_dict() for p in self.phases],
                    "plans": [p.to_dict() for p in self.plans]
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save milestones: {e}")

    def create_milestone(self, name: str, description: str, definition_of_done: str = "") -> Milestone:
        milestone_id = f"ms_{len(self.milestones) + 1}"
        
        milestone = Milestone(
            milestone_id=milestone_id,
            name=name,
            description=description,
            status=MilestoneStatus.PLANNING.value,
            definition_of_done=definition_of_done,
            created_at=datetime.now().isoformat()
        )
        
        self.milestones.append(milestone)
        self._save()
        
        return milestone

    def add_phase(self, milestone_id: str, name: str, description: str) -> Phase:
        phase_id = f"phase_{len(self.phases) + 1}"
        
        phase = Phase(
            phase_id=phase_id,
            milestone_id=milestone_id,
            name=name,
            description=description,
            status=PhaseStatus.PENDING.value,
            created_at=datetime.now().isoformat()
        )
        
        self.phases.append(phase)
        
        for m in self.milestones:
            if m.milestone_id == milestone_id:
                m.phases.append(phase_id)
        
        self._save()
        
        return phase

    def add_plan(self, phase_id: str, name: str, description: str, files: List[str] = None, 
                 action: str = "", verify: str = "", done: str = "", wave: int = 1,
                 dependencies: List[str] = None) -> Plan:
        plan_id = f"plan_{len(self.plans) + 1}"
        
        plan = Plan(
            plan_id=plan_id,
            phase_id=phase_id,
            name=name,
            description=description,
            files=files or [],
            action=action,
            verify=verify,
            done=done,
            wave=wave,
            dependencies=dependencies or [],
            created_at=datetime.now().isoformat()
        )
        
        self.plans.append(plan)
        
        for p in self.phases:
            if p.phase_id == phase_id:
                p.plans.append(plan_id)
        
        self._save()
        
        return plan

    def get_milestone(self, milestone_id: str) -> Optional[Milestone]:
        for m in self.milestones:
            if m.milestone_id == milestone_id:
                return m
        return None

    def get_phase(self, phase_id: str) -> Optional[Phase]:
        for p in self.phases:
            if p.phase_id == phase_id:
                return p
        return None

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        for p in self.plans:
            if p.plan_id == plan_id:
                return p
        return None

    def get_plans_by_phase(self, phase_id: str) -> List[Plan]:
        return [p for p in self.plans if p.phase_id == phase_id]

    def get_plans_by_wave(self, phase_id: str, wave: int) -> List[Plan]:
        return [p for p in self.plans if p.phase_id == phase_id and p.wave == wave]

    def get_waves(self, phase_id: str) -> List[int]:
        waves = set(p.wave for p in self.plans if p.phase_id == phase_id)
        return sorted(list(waves))

    def update_milestone_status(self, milestone_id: str, status: str):
        for m in self.milestones:
            if m.milestone_id == milestone_id:
                m.status = status
                if status == MilestoneStatus.COMPLETED.value:
                    m.completed_at = datetime.now().isoformat()
        self._save()

    def update_phase_status(self, phase_id: str, status: str):
        for p in self.phases:
            if p.phase_id == phase_id:
                p.status = status
                if status == PhaseStatus.COMPLETED.value:
                    p.completed_at = datetime.now().isoformat()
        self._save()

    def update_plan_status(self, plan_id: str, status: str):
        for p in self.plans:
            if p.plan_id == plan_id:
                p.status = status
                if status == "completed":
                    p.completed_at = datetime.now().isoformat()
        self._save()

    def get_progress(self, milestone_id: str = None) -> Dict:
        if milestone_id:
            phases = [p for p in self.phases if p.milestone_id == milestone_id]
            all_plans = []
            for phase in phases:
                all_plans.extend([p for p in self.plans if p.phase_id == phase.phase_id])
        else:
            all_plans = self.plans
        
        total = len(all_plans)
        completed = len([p for p in all_plans if p.status == "completed"])
        in_progress = len([p for p in all_plans if p.status == "in_progress"])
        
        return {
            "total_plans": total,
            "completed": completed,
            "in_progress": in_progress,
            "pending": total - completed - in_progress,
            "progress_percent": (completed / total * 100) if total > 0 else 0
        }

    def get_full_status(self) -> Dict:
        return {
            "milestones": [m.to_dict() for m in self.milestones],
            "phases": [p.to_dict() for p in self.phases],
            "plans": [p.to_dict() for p in self.plans],
            "progress": self.get_progress()
        }
