"""Plan Generator — Layer 1 planning logic for the Two-Layer Orchestration System.

Generates execution graphs from goals using LLM-powered or keyword-based
task decomposition following the assembly-line pattern.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from vetinari.orchestration.execution_graph import ExecutionGraph
from vetinari.types import PlanStatus

logger = logging.getLogger(__name__)


class PlanGenerator:
    """Generates execution plans from goals.

    Features:
    - Multi-candidate plan generation
    - Plan scoring and selection
    - Constraint handling
    """

    def __init__(self, model_router=None):
        self.model_router = model_router

    def generate_plan(
        self,
        goal: str,
        constraints: dict[str, Any] | None = None,
        max_depth: int = 10,
    ) -> ExecutionGraph:
        """Generate an execution graph from a goal.

        Args:
            goal: The goal to achieve
            constraints: Any constraints (budget, time, etc.)
            max_depth: Maximum depth of task decomposition

        Returns:
            ExecutionGraph with decomposed tasks
        """
        constraints = constraints or {}
        plan_id = f"plan-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"

        graph = ExecutionGraph(plan_id=plan_id, goal=goal)

        # Decompose goal into tasks
        tasks = self._decompose_goal(goal, max_depth)

        for task_spec in tasks:
            graph.add_task(
                task_id=task_spec["id"],
                description=task_spec["description"],
                task_type=task_spec.get("type", "general"),
                depends_on=task_spec.get("depends_on", []),
                input_data=task_spec.get("input", {}),
            )

        if self._has_circular_dependency(graph):
            logger.warning("Plan %s has circular dependencies", plan_id)

        graph.status = PlanStatus.DRAFT
        return graph

    def _decompose_goal(self, goal: str, max_depth: int) -> list[dict]:
        """Decompose a goal into tasks using the assembly-line pattern.

        Assembly-line stages:
        1. INPUT ANALYSIS  -- classify request, assess complexity
        2. PLAN GENERATION -- high-level workflow
        3. TASK DECOMP     -- break plan into atomic tasks
        4. MODEL ASSIGNMENT-- assign model to each task
        5. PARALLEL EXEC   -- execute assigned tasks (DAG)
        6. OUTPUT REVIEW   -- verify outputs for consistency
        7. FINAL ASSEMBLY  -- combine outputs

        Uses the PlannerAgent (LLM-powered) when available, falls back to
        keyword-based heuristics.
        """
        # Try to use the PlannerAgent for intelligent decomposition
        try:
            from vetinari.agents.contracts import AgentTask
            from vetinari.agents.planner_agent import get_planner_agent
            from vetinari.types import AgentType

            planner = get_planner_agent()
            task = AgentTask(
                task_id="decomp-0",
                agent_type=AgentType.PLANNER,
                description=goal,
                prompt=goal,
                context={"max_depth": max_depth},
            )
            result = planner.execute(task)
            if result.success and result.output and result.output.get("tasks"):
                return [
                    {
                        "id": t.get("id", f"t{i + 1}"),
                        "description": t.get("description", "Task"),
                        "type": (
                            t.get("assigned_agent", "general").lower()
                            if isinstance(t.get("assigned_agent"), str)
                            else "general"
                        ),
                        "depends_on": t.get("dependencies", []),
                        "input": {"goal": goal, "inputs": t.get("inputs", [])},
                    }
                    for i, t in enumerate(result.output["tasks"])
                ]
        except Exception as e:
            logger.warning("PlannerAgent decomposition failed: %s, using keyword fallback", e)

        # Keyword-based fallback decomposition
        return self._keyword_decomposition(goal)

    def _keyword_decomposition(self, goal: str) -> list[dict]:
        """Fallback keyword-based goal decomposition."""
        tasks: list[dict] = []
        counter = [1]

        def next_id(p: str = "t") -> str:
            """Next id.

            Returns:
                The result string.
            """
            tid = f"{p}{counter[0]}"
            counter[0] += 1
            return tid

        goal_lower = goal.lower()
        is_code = any(
            k in goal_lower
            for k in [
                "code",
                "implement",
                "build",
                "create",
                "program",
                "app",
                "web",
                "software",
            ]
        )
        is_research = any(k in goal_lower for k in ["research", "analyze", "investigate", "study", "review"])
        is_docs = any(k in goal_lower for k in ["document", "readme", "explain", "write", "report"])

        # Stage 1: Analysis
        t1 = next_id()
        tasks.append(
            {
                "id": t1,
                "description": "Analyze requirements and create specification",
                "type": "analysis",
                "depends_on": [],
                "input": {"goal": goal},
            }
        )

        # Stage 2: Implementation
        if is_code:
            t2 = next_id()
            tasks.append(
                {
                    "id": t2,
                    "description": "Set up project structure",
                    "type": "implementation",
                    "depends_on": [t1],
                    "input": {},
                }
            )
            t3 = next_id()
            tasks.append(
                {
                    "id": t3,
                    "description": "Implement core functionality",
                    "type": "implementation",
                    "depends_on": [t2],
                    "input": {},
                }
            )
            t4 = next_id()
            tasks.append(
                {
                    "id": t4,
                    "description": "Write and run tests",
                    "type": "testing",
                    "depends_on": [t3],
                    "input": {},
                }
            )
            t5 = next_id()
            tasks.append(
                {
                    "id": t5,
                    "description": "Verify and validate output",
                    "type": "verification",
                    "depends_on": [t4],
                    "input": {},
                }
            )
        elif is_research:
            t2 = next_id()
            tasks.append(
                {
                    "id": t2,
                    "description": "Gather information and sources",
                    "type": "research",
                    "depends_on": [t1],
                    "input": {},
                }
            )
            t3 = next_id()
            tasks.append(
                {
                    "id": t3,
                    "description": "Analyze and synthesize findings",
                    "type": "analysis",
                    "depends_on": [t2],
                    "input": {},
                }
            )
        else:
            t2 = next_id()
            tasks.append(
                {
                    "id": t2,
                    "description": "Execute primary task",
                    "type": "implementation",
                    "depends_on": [t1],
                    "input": {},
                }
            )

        # Stage 3: Review and Assembly
        prev = tasks[-1]["id"]
        trev = next_id()
        tasks.append(
            {
                "id": trev,
                "description": "Review output quality and consistency",
                "type": "verification",
                "depends_on": [prev],
                "input": {},
            }
        )

        if is_docs or is_code:
            tdoc = next_id()
            tasks.append(
                {
                    "id": tdoc,
                    "description": "Create documentation and final summary",
                    "type": "documentation",
                    "depends_on": [trev],
                    "input": {},
                }
            )

        return tasks

    def _has_circular_dependency(self, graph: ExecutionGraph) -> bool:
        """Check for circular dependencies in the graph."""
        visited: set = set()
        rec_stack: set = set()

        def visit(node_id: str) -> bool:
            """Visit.

            Returns:
                True if successful, False otherwise.
            """
            if node_id in rec_stack:
                return True
            if node_id in visited:
                return False
            visited.add(node_id)
            rec_stack.add(node_id)
            node = graph.nodes.get(node_id)
            if node:
                for dep in node.depends_on:
                    if visit(dep):
                        return True
            rec_stack.remove(node_id)
            return False

        return any(visit(node_id) for node_id in graph.nodes)
