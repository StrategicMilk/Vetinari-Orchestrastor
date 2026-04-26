"""Plan Generator — Layer 1 planning logic for the Two-Layer Orchestration System.

Generates execution graphs from goals using LLM-powered or keyword-based
task decomposition following the assembly-line pattern.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from vetinari.orchestration.execution_graph import ExecutionGraph
from vetinari.types import AgentType, PlanStatus

logger = logging.getLogger(__name__)


def _log_decomposition_async(
    choice: str,
    reasoning: str,
    context: dict[str, Any],
) -> None:
    """Fire-and-forget audit log for plan decomposition decisions.

    Runs in a daemon thread to avoid blocking the plan generation hot path.
    Failures are silently logged at DEBUG level.

    Args:
        choice: Short description of the decomposition method chosen.
        reasoning: Explanation of why this method was used.
        context: Additional context dict for the audit record.
    """

    def _log() -> None:
        try:
            from vetinari.audit import get_audit_logger

            get_audit_logger().log_decision(
                decision_type="plan_decomposition",
                choice=choice,
                reasoning=reasoning,
                context=context,
            )
        except Exception:
            logger.warning("Audit logging failed during decomposition", exc_info=True)

    threading.Thread(target=_log, daemon=True).start()


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
        constraints = constraints or {}  # noqa: VET112 - empty fallback preserves optional request metadata contract
        plan_id = f"plan-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"

        graph = ExecutionGraph(plan_id=plan_id, goal=goal)

        # Decompose goal into tasks
        tasks = self._decompose_goal(goal, max_depth, constraints)

        for task_spec in tasks:
            graph.add_task(
                task_id=task_spec["id"],
                description=task_spec["description"],
                task_type=task_spec.get("type", "general"),
                depends_on=task_spec.get("depends_on", []),
                input_data=task_spec.get("input", {}),
            )

        if self._has_circular_dependency(graph):
            logger.error("Plan %s has circular dependencies — blocking plan", plan_id)
            graph.status = PlanStatus.FAILED
            return graph

        # Analyse DAG shape and record suggested topology as plan metadata (ADR-0080)
        try:
            from vetinari.routing.dag_analyzer import analyze_dag, suggest_topology

            task_dicts = [{"id": t.task_id, "dependencies": list(t.depends_on)} for t in graph.tasks.values()]
            dag_shape = analyze_dag(task_dicts)
            topology = suggest_topology(dag_shape)
            graph.metadata = getattr(graph, "metadata", {}) or {}
            graph.metadata["suggested_topology"] = topology
            graph.metadata["dag_shape"] = dag_shape.to_dict()
            logger.debug("[PlanGenerator] Plan %s: topology=%s", plan_id, topology)
        except Exception as exc:
            logger.warning("[PlanGenerator] DAG analysis skipped for plan %s: %s", plan_id, exc)

        graph.status = PlanStatus.DRAFT
        return graph

    def _decompose_goal(self, goal: str, max_depth: int, constraints: dict[str, Any] | None = None) -> list[dict]:
        """Decompose a goal into tasks using the assembly-line pattern.

        Assembly-line stages:
        1. INPUT ANALYSIS  -- classify request, assess complexity
        2. PLAN GENERATION -- high-level workflow
        3. TASK DECOMP     -- break plan into atomic tasks
        4. MODEL ASSIGNMENT-- assign model to each task
        5. PARALLEL EXEC   -- execute assigned tasks (DAG)
        6. OUTPUT REVIEW   -- verify outputs for consistency
        7. FINAL ASSEMBLY  -- combine outputs

        Uses the ForemanAgent (LLM-powered) when available, falls back to
        keyword-based heuristics.
        """
        # Try to use the ForemanAgent for intelligent decomposition
        try:
            from vetinari.agents import get_foreman_agent
            from vetinari.agents.contracts import AgentTask

            planner = get_foreman_agent()
            # Switch to non-interactive (batch) mode for web/pipeline requests.
            # Clarification questions are auto-answered rather than blocking on user input.
            planner.set_interaction_mode("auto")
            # Bug #15c: inject max_context_tokens from VariantConfig so the planner
            # respects the active variant's token budget (LOW=4096, MEDIUM=16384, HIGH=32768).
            try:
                from vetinari.web.variant_system import get_variant_manager as _get_vm

                _variant_max_ctx = _get_vm().get_config().max_context_tokens
                planner._max_context_tokens = _variant_max_ctx
            except Exception:
                logger.warning("Could not load VariantManager — planner will use its own default context token limit")
            _ctx: dict[str, Any] = {"max_depth": max_depth}
            if constraints:
                # Forward project metadata (category, tech_stack, etc.) to the Foreman
                for _ck in (
                    "category",
                    "tech_stack",
                    "priority",
                    "platforms",
                    "required_features",
                    "things_to_avoid",
                    "expected_outputs",
                ):
                    if _ck in constraints:
                        _ctx[_ck] = constraints[_ck]
            task = AgentTask(
                task_id="decomp-0",
                agent_type=AgentType.FOREMAN,
                description=goal,
                prompt=goal,
                context=_ctx,
            )
            result = planner.execute(task)
            if result.success and isinstance(result.output, dict) and result.output.get("tasks"):
                decomposed = [
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
                # Log plan decomposition decision (US-023)
                _log_decomposition_async(
                    choice=f"llm_decomposition ({len(decomposed)} tasks)",
                    reasoning=f"ForemanAgent decomposed goal into {len(decomposed)} tasks",
                    context={
                        "task_count": len(decomposed),
                        "method": "foreman_agent",
                        "goal_preview": goal[:120],
                    },
                )
                return self._assess_risk(goal, decomposed)
        except Exception as e:
            logger.warning("ForemanAgent decomposition failed: %s, using keyword fallback", e)

        # Keyword-based fallback decomposition
        fallback_tasks = self._keyword_decomposition(goal, constraints=constraints)

        # Log fallback decomposition decision (US-023)
        _log_decomposition_async(
            choice=f"keyword_fallback ({len(fallback_tasks)} tasks)",
            reasoning="ForemanAgent unavailable, used keyword-based heuristic decomposition",
            context={
                "task_count": len(fallback_tasks),
                "method": "keyword_fallback",
                "goal_preview": goal[:120],
            },
        )

        return fallback_tasks

    def _assess_risk(self, goal: str, tasks: list[dict]) -> list[dict]:
        """Tag tasks with risk flags for destructive or irreversible operations.

        Scans goal and task descriptions for destructive keywords (delete,
        drop, remove, overwrite, migrate, deploy, push) and marks matching
        tasks with ``risk_level`` and ``risk_reason`` in their ``input`` dict.

        Args:
            goal: The original goal string.
            tasks: The decomposed task list to annotate.

        Returns:
            The same task list with risk annotations added where applicable.
        """
        _DESTRUCTIVE_KEYWORDS = {
            "high": ["delete", "drop", "destroy", "overwrite", "force push", "rm -rf", "truncate"],
            "medium": ["migrate", "deploy", "push", "upgrade", "rename", "move", "replace"],
        }
        goal_lower = goal.lower()

        for task in tasks:
            desc_lower = task["description"].lower()
            combined = f"{goal_lower} {desc_lower}"
            for level, keywords in _DESTRUCTIVE_KEYWORDS.items():
                matched = [kw for kw in keywords if kw in combined]
                if matched:
                    task.setdefault("input", {})["risk_level"] = level
                    task["input"]["risk_reason"] = f"Destructive operation detected: {', '.join(matched)}"
                    logger.info(
                        "Task %s flagged as %s risk: %s",
                        task["id"],
                        level,
                        ", ".join(matched),
                    )
                    break
        return tasks

    def _keyword_decomposition(self, goal: str, constraints: dict[str, Any] | None = None) -> list[dict]:
        """Fallback keyword-based goal decomposition.

        When ``constraints`` includes ``tech_stack``, a Layer 0 scaffolding
        task is injected before the analysis stage to ensure foundation-first
        execution (architecture and framework setup before implementation).

        Args:
            goal: The user's goal string.
            constraints: Optional project constraints (tech_stack, category, etc.).
        """
        constraints = constraints or {}  # noqa: VET112 - empty fallback preserves optional request metadata contract
        tasks: list[dict] = []
        counter = [1]

        def next_id(p: str = "t") -> str:
            """Generate a unique sequential task ID with the given prefix.

            Returns:
                String of the form ``"{p}{counter}"`` where the counter
                increments globally across all calls within this decomposition.
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

        # Layer 0: Foundation scaffolding when tech_stack is specified (ADR: foundation-first)
        foundation_id: str | None = None
        tech_stack = constraints.get("tech_stack", "")
        if tech_stack and is_code:
            foundation_id = next_id("foundation-")
            tasks.append(
                {
                    "id": foundation_id,
                    "description": f"Set up project architecture and framework scaffolding for: {tech_stack}",
                    "type": "scaffolding",
                    "depends_on": [],
                    "input": {"goal": goal, "tech_stack": tech_stack},
                },
            )
            logger.info("Layer 0 scaffolding task injected for tech_stack=%s", tech_stack)

        # Stage 1: Analysis — include goal in description for worker context
        _goal_summary = goal[:120].rstrip()
        t1 = next_id()
        tasks.append(
            {
                "id": t1,
                "description": f"Analyze requirements and create specification for: {_goal_summary}",
                "type": "analysis",
                "depends_on": [foundation_id] if foundation_id else [],
                "input": {"goal": goal},
            },
        )

        # Stage 2: Implementation
        if is_code:
            t2 = next_id()
            tasks.append(
                {
                    "id": t2,
                    "description": f"Set up project structure and scaffolding for: {_goal_summary}",
                    "type": "implementation",
                    "depends_on": [t1],
                    "input": {"goal": goal},
                },
            )
            t3 = next_id()
            tasks.append(
                {
                    "id": t3,
                    "description": f"Implement core functionality: {_goal_summary}",
                    "type": "implementation",
                    "depends_on": [t2],
                    "input": {"goal": goal},
                },
            )
            t4 = next_id()
            tasks.append(
                {
                    "id": t4,
                    "description": f"Write and run tests for: {_goal_summary}",
                    "type": "testing",
                    "depends_on": [t3],
                    "input": {"goal": goal},
                },
            )
            t5 = next_id()
            tasks.append(
                {
                    "id": t5,
                    "description": f"Verify output quality and completeness for: {_goal_summary}",
                    "type": "verification",
                    "depends_on": [t4],
                    "input": {"goal": goal},
                },
            )
        elif is_research:
            t2 = next_id()
            tasks.append(
                {
                    "id": t2,
                    "description": f"Gather information and sources about: {_goal_summary}",
                    "type": "research",
                    "depends_on": [t1],
                    "input": {"goal": goal},
                },
            )
            t3 = next_id()
            tasks.append(
                {
                    "id": t3,
                    "description": f"Analyze and synthesize findings for: {_goal_summary}",
                    "type": "analysis",
                    "depends_on": [t2],
                    "input": {"goal": goal},
                },
            )
        else:
            t2 = next_id()
            tasks.append(
                {
                    "id": t2,
                    "description": f"Execute primary task: {_goal_summary}",
                    "type": "implementation",
                    "depends_on": [t1],
                    "input": {},
                },
            )

        # Stage 3: Review and Assembly
        prev = tasks[-1]["id"]
        trev = next_id()
        tasks.append(
            {
                "id": trev,
                "description": f"Review output quality and consistency for: {_goal_summary}",
                "type": "verification",
                "depends_on": [prev],
                "input": {"goal": goal},
            },
        )

        if is_docs or is_code:
            tdoc = next_id()
            tasks.append(
                {
                    "id": tdoc,
                    "description": f"Create documentation and final summary for: {_goal_summary}",
                    "type": "documentation",
                    "depends_on": [trev],
                    "input": {"goal": goal},
                },
            )

        return self._assess_risk(goal, tasks)

    def resolve_worker_mode(self, task_description: str) -> str | None:
        """Resolve the best Worker mode for a task using capability-based routing.

        Queries the skill registry's capability index to find which Worker mode
        best matches the task description keywords.

        Args:
            task_description: Description of the task to route.

        Returns:
            The best-matching Worker mode name, or None if no match found.
        """
        try:
            from vetinari.skills.skill_registry import get_skill, get_skills_by_capability

            # Map common task keywords to capabilities
            _keyword_to_capability = {
                "review": "code_review",
                "audit": "security_audit",
                "security": "security_audit",
                "test": "test_writing",
                "document": "documentation_generation",
                "refactor": "refactoring",
                "bug": "bug_diagnosis",
                "fix": "bug_diagnosis",
                "implement": "feature_implementation",
                "build": "feature_implementation",
                "research": "code_discovery",
                "explore": "code_discovery",
                "analyze": "code_discovery",
                "architecture": "architecture_review",
                "design": "architecture_review",
                "risk": "risk_assessment",
                "cost": "cost_analysis",
                "improve": "continuous_improvement",
                "monitor": "monitoring",
                "recover": "error_recovery",
                "experiment": "experiment_runner",
                "deploy": "infrastructure_research",
                "migrate": "infrastructure_research",
            }

            desc_lower = task_description.lower()
            for keyword, capability in _keyword_to_capability.items():
                if keyword in desc_lower:
                    matching_skills = get_skills_by_capability(capability)
                    if matching_skills:
                        worker_skill = get_skill("worker")
                        if worker_skill:
                            # Find the mode that matches this capability
                            _cap_to_mode = {
                                "code_review": "code_review",
                                "security_audit": "security_audit",
                                "test_writing": "build",
                                "documentation_generation": "documentation",
                                "refactoring": "build",
                                "bug_diagnosis": "build",
                                "feature_implementation": "build",
                                "code_discovery": "code_discovery",
                                "architecture_review": "architecture",
                                "risk_assessment": "risk_assessment",
                                "cost_analysis": "cost_analysis",
                                "continuous_improvement": "improvement",
                                "monitoring": "monitor",
                                "error_recovery": "error_recovery",
                                "experiment_runner": "experiment",
                                "infrastructure_research": "devops",
                            }
                            mode = _cap_to_mode.get(capability)
                            if mode and mode in worker_skill.modes:
                                return mode
        except (ImportError, AttributeError, KeyError):
            logger.warning("Capability-based routing unavailable, using default")
        return None

    def _has_circular_dependency(self, graph: ExecutionGraph) -> bool:
        """Check for circular dependencies in the graph."""
        visited: set = set()
        rec_stack: set = set()

        def visit(node_id: str) -> bool:
            """Recursively detect whether ``node_id`` is part of a cycle.

            Returns:
                True if a cycle is detected reachable from this node,
                False if the node and all its transitive dependencies are acyclic.
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
