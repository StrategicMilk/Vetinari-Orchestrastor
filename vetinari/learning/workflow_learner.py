"""
Workflow Learner - Vetinari Self-Improvement Subsystem

Mines execution history to learn which decomposition strategies work
best for different goal types. Uses a simple decision tree approach
(no external ML dependencies).
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from vetinari.config import get_subdirectory

logger = logging.getLogger(__name__)


@dataclass
class WorkflowPattern:
    """A learned workflow pattern mapping goal characteristics to outcomes."""
    pattern_id: str
    domain: str               # "coding" | "research" | "data" | "docs" | "general"
    avg_depth: float          # Optimal decomposition depth
    avg_breadth: float        # Optimal number of parallel tasks
    preferred_agents: List[str] = field(default_factory=list)
    success_rate: float = 0.7
    avg_quality: float = 0.7
    sample_count: int = 0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


class WorkflowLearner:
    """
    Learns effective workflow patterns from execution history.

    Analyzes completed plans to extract:
    - Optimal decomposition depth per domain
    - Best agent sequences per domain
    - Parallelism strategies that work

    Provides recommendations to PlannerAgent and TwoLayerOrchestrator.
    """

    def __init__(self):
        self._patterns: Dict[str, WorkflowPattern] = {}
        self._load_patterns()

    def infer_domain(self, goal: str) -> str:
        """Classify goal into a domain category."""
        g = goal.lower()
        if any(k in g for k in ["code", "implement", "build", "program", "app", "software"]):
            return "coding"
        if any(k in g for k in ["research", "analyze", "investigate", "study"]):
            return "research"
        if any(k in g for k in ["data", "database", "sql", "pipeline", "etl"]):
            return "data"
        if any(k in g for k in ["document", "readme", "write", "report", "explain"]):
            return "docs"
        return "general"

    def get_recommendations(self, goal: str) -> Dict[str, Any]:
        """
        Return workflow recommendations for a goal.

        Returns:
            Dict with keys: domain, recommended_depth, recommended_breadth,
                           preferred_agents, confidence.
        """
        domain = self.infer_domain(goal)
        pattern = self._patterns.get(domain)

        if pattern and pattern.sample_count >= 3:
            return {
                "domain": domain,
                "recommended_depth": int(round(pattern.avg_depth)),
                "recommended_breadth": int(round(pattern.avg_breadth)),
                "preferred_agents": pattern.preferred_agents[:5],
                "confidence": min(1.0, pattern.sample_count / 20),
            }

        # Default recommendations per domain
        defaults = {
            "coding": {"depth": 4, "breadth": 3, "agents": ["EXPLORER", "BUILDER", "TEST_AUTOMATION", "EVALUATOR", "SECURITY_AUDITOR"]},
            "research": {"depth": 3, "breadth": 2, "agents": ["RESEARCHER", "LIBRARIAN", "SYNTHESIZER", "DOCUMENTATION_AGENT"]},
            "data": {"depth": 4, "breadth": 2, "agents": ["DATA_ENGINEER", "EVALUATOR", "DOCUMENTATION_AGENT"]},
            "docs": {"depth": 2, "breadth": 1, "agents": ["RESEARCHER", "SYNTHESIZER", "DOCUMENTATION_AGENT"]},
            "general": {"depth": 3, "breadth": 2, "agents": ["EXPLORER", "BUILDER", "EVALUATOR"]},
        }
        d = defaults.get(domain, defaults["general"])
        return {
            "domain": domain,
            "recommended_depth": d["depth"],
            "recommended_breadth": d["breadth"],
            "preferred_agents": d["agents"],
            "confidence": 0.3,  # Low confidence -- using defaults
        }

    def record_outcome(
        self,
        goal: str,
        plan_depth: int,
        plan_breadth: int,
        agents_used: List[str],
        quality_score: float,
        success: bool,
    ) -> None:
        """
        Record a plan execution outcome to update patterns.

        Args:
            goal: The original goal.
            plan_depth: Maximum depth of the executed plan.
            plan_breadth: Maximum breadth (parallel tasks).
            agents_used: List of agent type strings used.
            quality_score: Overall quality score 0.0-1.0.
            success: Whether the plan succeeded.
        """
        domain = self.infer_domain(goal)
        EMA = 0.3

        if domain in self._patterns:
            p = self._patterns[domain]
            p.avg_depth = (1 - EMA) * p.avg_depth + EMA * plan_depth
            p.avg_breadth = (1 - EMA) * p.avg_breadth + EMA * plan_breadth
            p.success_rate = (1 - EMA) * p.success_rate + EMA * float(success)
            p.avg_quality = (1 - EMA) * p.avg_quality + EMA * quality_score
            p.sample_count += 1
            # Update preferred agents (frequency-based)
            freq: Dict[str, int] = {}
            for a in p.preferred_agents:
                freq[a] = freq.get(a, 0) + 1
            for a in agents_used:
                freq[a] = freq.get(a, 0) + 1
            p.preferred_agents = sorted(freq, key=freq.get, reverse=True)[:5]
            p.last_updated = datetime.now().isoformat()
        else:
            self._patterns[domain] = WorkflowPattern(
                pattern_id=f"pattern_{domain}",
                domain=domain,
                avg_depth=float(plan_depth),
                avg_breadth=float(plan_breadth),
                preferred_agents=agents_used[:5],
                success_rate=float(success),
                avg_quality=quality_score,
                sample_count=1,
            )

        self._save_patterns()
        logger.debug(f"[WorkflowLearner] Updated pattern for domain '{domain}'")

    def get_all_patterns(self) -> List[Dict[str, Any]]:
        """Get all learned patterns."""
        return [asdict(p) for p in self._patterns.values()]

    def _load_patterns(self) -> None:
        try:
            import os
            path = str(get_subdirectory(".vetinari") / "workflow_patterns.json")
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                for item in data:
                    p = WorkflowPattern(**item)
                    self._patterns[p.domain] = p
                logger.debug(f"[WorkflowLearner] Loaded {len(self._patterns)} patterns")
        except Exception as e:
            logger.debug(f"Could not load workflow patterns: {e}")

    def _save_patterns(self) -> None:
        try:
            import os
            state_dir = str(get_subdirectory(".vetinari"))
            path = os.path.join(state_dir, "workflow_patterns.json")
            with open(path, "w") as f:
                json.dump([asdict(p) for p in self._patterns.values()], f, indent=2)
        except Exception as e:
            logger.debug(f"Could not save workflow patterns: {e}")


_workflow_learner: Optional[WorkflowLearner] = None


def get_workflow_learner() -> WorkflowLearner:
    global _workflow_learner
    if _workflow_learner is None:
        _workflow_learner = WorkflowLearner()
    return _workflow_learner
