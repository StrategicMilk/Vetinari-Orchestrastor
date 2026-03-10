"""Rule-based phase routing for task execution. No LLM calls."""
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PhaseRoute:
    phase: str
    primary_agents: List[str]
    fallback_agents: List[str]
    parallel: bool = True


class PhaseCoordinator:
    """Rule-based routing within execution phases. No LLM calls.

    For simple plans (<5 tasks): flat dispatch (unchanged behavior).
    For complex plans (>5 tasks): group tasks by phase, execute phases with appropriate agents.
    """

    PHASES: Dict[str, PhaseRoute] = {
        "analysis": PhaseRoute("analysis", ["RESEARCHER", "ARCHITECT"], ["PLANNER"], parallel=True),
        "planning": PhaseRoute("planning", ["PLANNER"], ["ARCHITECT"], parallel=False),
        "implementation": PhaseRoute("implementation", ["BUILDER"], ["RESILIENCE"], parallel=True),
        "quality": PhaseRoute("quality", ["TESTER"], ["ARCHITECT"], parallel=True),
        "documentation": PhaseRoute("documentation", ["DOCUMENTER"], ["BUILDER"], parallel=True),
        "meta": PhaseRoute("meta", ["META"], ["ARCHITECT"], parallel=False),
    }

    COMPLEXITY_THRESHOLD = 5  # tasks above this use phase routing

    def should_use_phases(self, task_count: int) -> bool:
        return task_count > self.COMPLEXITY_THRESHOLD

    def classify_task(self, description: str) -> str:
        """Classify a task into a phase based on keywords. No LLM call."""
        desc_lower = description.lower()

        phase_keywords = {
            "analysis": ["research", "investigate", "analyze", "explore", "find", "search", "discover"],
            "planning": ["plan", "decompose", "design", "architect", "structure", "organize"],
            "implementation": ["build", "create", "implement", "code", "write", "develop", "generate", "scaffold"],
            "quality": ["test", "verify", "audit", "security", "evaluate", "check", "validate", "review"],
            "documentation": ["document", "readme", "changelog", "comment", "api doc", "git", "commit"],
            "meta": ["improve", "optimize", "experiment", "benchmark", "metric", "performance"],
        }

        best_phase = "implementation"  # default
        best_score = 0

        for phase, keywords in phase_keywords.items():
            score = sum(1 for kw in keywords if kw in desc_lower)
            if score > best_score:
                best_score = score
                best_phase = phase

        return best_phase

    def get_route(self, phase: str) -> PhaseRoute:
        return self.PHASES.get(phase, self.PHASES["implementation"])

    def group_tasks_by_phase(self, tasks: list) -> Dict[str, list]:
        """Group tasks by their classified phase."""
        groups: Dict[str, list] = {}
        for task in tasks:
            desc = getattr(task, "description", str(task))
            phase = self.classify_task(desc)
            groups.setdefault(phase, []).append(task)
        return groups

    def get_execution_order(self) -> List[str]:
        """Return recommended phase execution order."""
        return ["analysis", "planning", "implementation", "quality", "documentation", "meta"]


_coordinator: Optional[PhaseCoordinator] = None


def get_phase_coordinator() -> PhaseCoordinator:
    global _coordinator
    if _coordinator is None:
        _coordinator = PhaseCoordinator()
    return _coordinator
