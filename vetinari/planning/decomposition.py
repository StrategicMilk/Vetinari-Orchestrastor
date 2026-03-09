"""
Decomposition Engine
====================
Provides task decomposition services used by the Decomposition Lab UI.
Wraps the PlannerAgent and planning infrastructure.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Decomposition configuration knobs
DEFAULT_MAX_DEPTH = 14
MIN_MAX_DEPTH = 12
MAX_MAX_DEPTH = 16
SEED_RATE = 0.3   # 30% of tasks seeded with refined subtasks
SEED_MIX = 0.5    # Balance between breadth and depth seeding

# Definition of Done criteria per level
_DOD_CRITERIA = {
    "Light": [
        "Code compiles without errors",
        "Basic functionality works",
        "No blocking security issues",
    ],
    "Standard": [
        "Code compiles and lints cleanly",
        "Unit tests pass (>70% coverage)",
        "Security scan passes",
        "Documentation updated",
        "Code reviewed",
    ],
    "Hard": [
        "Code compiles, lints, and type-checks",
        "Unit + integration tests pass (>85% coverage)",
        "Security scan passes with no high/critical findings",
        "Full API documentation generated",
        "Performance benchmarks met",
        "Accessibility audit passed",
        "Peer reviewed and approved",
    ],
}

# Definition of Ready criteria
_DOR_CRITERIA = {
    "Light": [
        "Task description is clear",
        "Inputs are defined",
    ],
    "Standard": [
        "Task description is unambiguous",
        "Inputs and expected outputs defined",
        "Dependencies identified",
        "Estimated effort provided",
    ],
    "Hard": [
        "Task description is unambiguous and reviewed",
        "All inputs, outputs, and side-effects documented",
        "Dependencies fully resolved",
        "Effort estimate reviewed by at least one peer",
        "Acceptance criteria agreed upon",
        "Risk assessment completed",
    ],
}


@dataclass
class SubtaskSpec:
    """A single decomposed subtask."""
    subtask_id: str
    parent_task_id: str
    description: str
    agent_type: str
    depth: int
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    dod_criteria: List[str] = field(default_factory=list)
    dor_criteria: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DecompositionEvent:
    """A historical decomposition event."""
    event_id: str
    plan_id: str
    task_id: str
    depth: int
    seeds_used: List[str]
    subtasks_created: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class DecompositionEngine:
    """
    Orchestrates task decomposition using the PlannerAgent.
    Used by the Decomposition Lab in the web UI.
    """

    SEED_MIX = SEED_MIX
    SEED_RATE = SEED_RATE
    DEFAULT_MAX_DEPTH = DEFAULT_MAX_DEPTH
    MIN_MAX_DEPTH = MIN_MAX_DEPTH
    MAX_MAX_DEPTH = MAX_MAX_DEPTH

    def __init__(self):
        self._history: List[DecompositionEvent] = []
        self._templates: List[Dict[str, Any]] = self._build_default_templates()

    def _build_default_templates(self) -> List[Dict[str, Any]]:
        """Build built-in decomposition templates."""
        return [
            {
                "template_id": "web_app",
                "name": "Web Application",
                "keywords": ["web", "app", "frontend", "react", "vue", "html"],
                "agent_type": "BUILDER",
                "dod_level": "Standard",
                "subtasks": [
                    "Define requirements and wireframes",
                    "Set up project structure and dependencies",
                    "Implement backend API",
                    "Implement frontend components",
                    "Write tests",
                    "Deploy and configure CI/CD",
                ],
            },
            {
                "template_id": "data_pipeline",
                "name": "Data Pipeline",
                "keywords": ["data", "pipeline", "etl", "database", "sql"],
                "agent_type": "DATA_ENGINEER",
                "dod_level": "Standard",
                "subtasks": [
                    "Define data schema and models",
                    "Implement data ingestion",
                    "Implement transformation logic",
                    "Add validation and error handling",
                    "Write pipeline tests",
                    "Document data flow",
                ],
            },
            {
                "template_id": "research",
                "name": "Research Task",
                "keywords": ["research", "analyze", "investigate", "study"],
                "agent_type": "RESEARCHER",
                "dod_level": "Light",
                "subtasks": [
                    "Define research scope and questions",
                    "Gather sources and references",
                    "Analyze and synthesize findings",
                    "Write research report",
                ],
            },
        ]

    def get_templates(
        self,
        keywords: Optional[List[str]] = None,
        agent_type: Optional[str] = None,
        dod_level: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return matching decomposition templates."""
        results = self._templates[:]
        if keywords:
            kw_lower = [k.lower() for k in keywords]
            results = [
                t for t in results
                if any(kw in t.get("keywords", []) for kw in kw_lower)
            ]
        if agent_type:
            results = [t for t in results if t.get("agent_type") == agent_type.upper()]
        if dod_level:
            results = [t for t in results if t.get("dod_level") == dod_level]
        return results

    def get_dod_criteria(self, level: str = "Standard") -> List[str]:
        return _DOD_CRITERIA.get(level, _DOD_CRITERIA["Standard"])

    def get_dor_criteria(self, level: str = "Standard") -> List[str]:
        return _DOR_CRITERIA.get(level, _DOR_CRITERIA["Standard"])

    def decompose_task(
        self,
        task_prompt: str,
        parent_task_id: str = "root",
        depth: int = 0,
        max_depth: int = DEFAULT_MAX_DEPTH,
        plan_id: str = "default",
    ) -> List[Dict[str, Any]]:
        """
        Decompose a task into subtasks using the PlannerAgent.
        Falls back to keyword-based decomposition.
        """
        max_depth = max(MIN_MAX_DEPTH, min(max_depth, MAX_MAX_DEPTH))

        if depth >= max_depth:
            logger.warning(f"Max decomposition depth {max_depth} reached for task: {task_prompt[:50]}")
            return []

        try:
            from vetinari.agents.planner_agent import get_planner_agent
            from vetinari.agents.contracts import AgentTask, AgentType
            planner = get_planner_agent()
            agent_task = AgentTask(
                task_id=f"decomp_{uuid.uuid4().hex[:8]}",
                agent_type=AgentType.PLANNER,
                description=f"Decompose: {task_prompt}",
                prompt=task_prompt,
                context={"depth": depth, "max_depth": max_depth, "plan_id": plan_id},
            )
            result = planner.execute(agent_task)
            if result.success and isinstance(result.output, dict):
                tasks = result.output.get("tasks", [])
                subtasks = []
                for t in tasks:
                    subtask = {
                        "subtask_id": t.get("id", str(uuid.uuid4())[:8]),
                        "parent_task_id": parent_task_id,
                        "description": t.get("description", ""),
                        "agent_type": t.get("assigned_agent", "BUILDER"),
                        "depth": depth + 1,
                        "inputs": t.get("inputs", []),
                        "outputs": t.get("outputs", []),
                        "dependencies": t.get("dependencies", []),
                        "acceptance_criteria": t.get("acceptance_criteria", ""),
                    }
                    subtasks.append(subtask)

                # Record history
                self._history.append(DecompositionEvent(
                    event_id=str(uuid.uuid4()),
                    plan_id=plan_id,
                    task_id=parent_task_id,
                    depth=depth,
                    seeds_used=[],
                    subtasks_created=len(subtasks),
                ))
                return subtasks
        except Exception as e:
            logger.warning(f"LLM decomposition failed, using keyword fallback: {e}")

        # Keyword fallback
        return self._keyword_decompose(task_prompt, parent_task_id, depth)

    def _keyword_decompose(
        self, task_prompt: str, parent_task_id: str, depth: int
    ) -> List[Dict[str, Any]]:
        """Simple keyword-based decomposition fallback."""
        task_lower = task_prompt.lower()
        subtasks = []

        def make_subtask(desc: str, agent: str, deps: List[str] = None) -> Dict[str, Any]:
            sid = f"st_{uuid.uuid4().hex[:6]}"
            return {
                "subtask_id": sid,
                "parent_task_id": parent_task_id,
                "description": desc,
                "agent_type": agent,
                "depth": depth + 1,
                "inputs": [],
                "outputs": [],
                "dependencies": deps or [],
                "acceptance_criteria": f"{desc} is complete",
            }

        s1 = make_subtask("Analyze requirements and define scope", "EXPLORER")
        subtasks.append(s1)

        if any(kw in task_lower for kw in ["code", "implement", "build", "develop"]):
            s2 = make_subtask("Implement core functionality", "BUILDER", [s1["subtask_id"]])
            subtasks.append(s2)
            s3 = make_subtask("Write tests", "TEST_AUTOMATION", [s2["subtask_id"]])
            subtasks.append(s3)

        if any(kw in task_lower for kw in ["ui", "frontend", "web", "interface"]):
            prev = subtasks[-1]["subtask_id"] if subtasks else s1["subtask_id"]
            subtasks.append(make_subtask("Design and implement UI", "UI_PLANNER", [prev]))

        last = subtasks[-1]["subtask_id"] if subtasks else s1["subtask_id"]
        subtasks.append(make_subtask("Review and document", "EVALUATOR", [last]))

        return subtasks

    def get_decomposition_history(
        self, plan_id: Optional[str] = None
    ) -> List[DecompositionEvent]:
        """Return decomposition history, optionally filtered by plan_id."""
        if plan_id:
            return [e for e in self._history if e.plan_id == plan_id]
        return list(self._history)


# Module-level singleton
_decomposition_engine: Optional[DecompositionEngine] = None


def _get_engine() -> DecompositionEngine:
    global _decomposition_engine
    if _decomposition_engine is None:
        _decomposition_engine = DecompositionEngine()
    return _decomposition_engine


# Exported instance used by web_ui.py
decomposition_engine = _get_engine()
