from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SubtaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Subtask:
    subtask_id: str
    plan_id: str
    parent_id: str
    depth: int
    max_depth: int
    description: str
    prompt: str
    agent_type: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    status: str = SubtaskStatus.PENDING.value
    assigned_agent: str = ""
    result: Any = None
    error: str = ""
    planned_start: str = ""
    planned_end: str = ""
    actual_start: str = ""
    actual_end: str = ""
    dod_level: str = "Standard"
    dor_level: str = "Standard"
    decomposition_seed: str = ""
    max_depth_override: int = 0
    estimated_effort: float = 1.0
    ponder_ranking: list[dict] = field(default_factory=list)
    ponder_scores: dict[str, float] = field(default_factory=dict)
    ponder_used: bool = False

    def to_dict(self) -> dict:
        return {
            "subtask_id": self.subtask_id,
            "plan_id": self.plan_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "max_depth": self.max_depth,
            "description": self.description,
            "prompt": self.prompt,
            "agent_type": self.agent_type,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "dependencies": self.dependencies,
            "status": self.status,
            "assigned_agent": self.assigned_agent,
            "result": self.result,
            "error": self.error,
            "planned_start": self.planned_start,
            "planned_end": self.planned_end,
            "actual_start": self.actual_start,
            "actual_end": self.actual_end,
            "dod_level": self.dod_level,
            "dor_level": self.dor_level,
            "decomposition_seed": self.decomposition_seed,
            "max_depth_override": self.max_depth_override,
            "estimated_effort": self.estimated_effort,
            "ponder_ranking": self.ponder_ranking,
            "ponder_scores": self.ponder_scores,
            "ponder_used": self.ponder_used,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Subtask:
        return cls(
            subtask_id=data.get("subtask_id", ""),
            plan_id=data.get("plan_id", ""),
            parent_id=data.get("parent_id", ""),
            depth=data.get("depth", 0),
            max_depth=data.get("max_depth", 14),
            description=data.get("description", ""),
            prompt=data.get("prompt", ""),
            agent_type=data.get("agent_type", "builder"),
            inputs=data.get("inputs", []),
            outputs=data.get("outputs", []),
            dependencies=data.get("dependencies", []),
            status=data.get("status", SubtaskStatus.PENDING.value),
            assigned_agent=data.get("assigned_agent", ""),
            result=data.get("result"),
            error=data.get("error", ""),
            planned_start=data.get("planned_start", ""),
            planned_end=data.get("planned_end", ""),
            actual_start=data.get("actual_start", ""),
            actual_end=data.get("actual_end", ""),
            dod_level=data.get("dod_level", "Standard"),
            dor_level=data.get("dor_level", "Standard"),
            decomposition_seed=data.get("decomposition_seed", ""),
            max_depth_override=data.get("max_depth_override", 0),
            estimated_effort=data.get("estimated_effort", 1.0),
            ponder_ranking=data.get("ponder_ranking", []),
            ponder_scores=data.get("ponder_scores", {}),
            ponder_used=data.get("ponder_used", False),
        )

    def get_effective_max_depth(self) -> int:
        if self.max_depth_override > 0:
            return max(12, min(16, self.max_depth_override))
        return max(12, min(16, self.max_depth))


class SubtaskTree:
    _instance = None

    @classmethod
    def get_instance(cls, storage_path: str | None = None):
        if cls._instance is None:
            cls._instance = cls(storage_path)
        return cls._instance

    def __init__(self, storage_path: str | None = None):
        if storage_path is None:
            storage_path = Path.home() / ".lmstudio" / "projects" / "Vetinari" / "subtasks"

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.trees: dict[str, dict[str, Subtask]] = {}
        self._load_trees()

    def _load_trees(self):
        for file in self.storage_path.glob("*.json"):
            try:
                with open(file) as f:
                    data = json.load(f)
                    plan_id = file.stem
                    self.trees[plan_id] = {}
                    for st_data in data.get("subtasks", []):
                        subtask = Subtask.from_dict(st_data)
                        self.trees[plan_id][subtask.subtask_id] = subtask
            except Exception as e:
                logger.error(f"Error loading subtask tree {file}: {e}")

    def _save_tree(self, plan_id: str):
        file_path = self.storage_path / f"{plan_id}.json"
        subtasks = [st.to_dict() for st in self.trees.get(plan_id, {}).values()]
        with open(file_path, "w") as f:
            json.dump({"plan_id": plan_id, "subtasks": subtasks}, f, indent=2)

    def create_subtask(
        self,
        plan_id: str,
        parent_id: str,
        depth: int,
        description: str,
        prompt: str,
        agent_type: str,
        max_depth: int = 14,
        dod_level: str = "Standard",
        dor_level: str = "Standard",
        estimated_effort: float = 1.0,
        max_depth_override: int = 0,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        decomposition_seed: str = "",
    ) -> Subtask:

        if plan_id not in self.trees:
            self.trees[plan_id] = {}

        subtask_id = f"st_{uuid.uuid4().hex[:8]}"

        subtask = Subtask(
            subtask_id=subtask_id,
            plan_id=plan_id,
            parent_id=parent_id,
            depth=depth,
            max_depth=max_depth,
            description=description,
            prompt=prompt,
            agent_type=agent_type,
            dod_level=dod_level,
            dor_level=dor_level,
            estimated_effort=estimated_effort,
            max_depth_override=max_depth_override,
            inputs=inputs or [],
            outputs=outputs or [],
            decomposition_seed=decomposition_seed,
        )

        self.trees[plan_id][subtask_id] = subtask
        self._save_tree(plan_id)

        return subtask

    def get_subtask(self, plan_id: str, subtask_id: str) -> Subtask | None:
        return self.trees.get(plan_id, {}).get(subtask_id)

    def get_subtasks_by_parent(self, plan_id: str, parent_id: str) -> list[Subtask]:
        return [st for st in self.trees.get(plan_id, {}).values() if st.parent_id == parent_id]

    def get_root_subtasks(self, plan_id: str) -> list[Subtask]:
        return [st for st in self.trees.get(plan_id, {}).values() if st.parent_id == "" or st.parent_id == "root"]

    def get_all_subtasks(self, plan_id: str) -> list[Subtask]:
        return list(self.trees.get(plan_id, {}).values())

    def update_subtask(self, plan_id: str, subtask_id: str, updates: dict) -> Subtask | None:
        subtask = self.trees.get(plan_id, {}).get(subtask_id)
        if not subtask:
            return None

        for key, value in updates.items():
            if hasattr(subtask, key):
                setattr(subtask, key, value)

        self._save_tree(plan_id)
        return subtask

    def delete_tree(self, plan_id: str) -> bool:
        if plan_id in self.trees:
            del self.trees[plan_id]
            file_path = self.storage_path / f"{plan_id}.json"
            if file_path.exists():
                file_path.unlink()
            return True
        return False

    def get_tree_depth(self, plan_id: str) -> int:
        subtasks = self.trees.get(plan_id, {}).values()
        if not subtasks:
            return 0
        return max(st.depth for st in subtasks)


subtask_tree = SubtaskTree.get_instance()
