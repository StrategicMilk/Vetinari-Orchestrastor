"""
Vetinari AgentGraph - Orchestration Engine

Coordinates all 21 specialized agents through a Plan DAG, managing:
- Task decomposition and assignment
- Dependency resolution
- True async parallel execution of independent tasks
- Retry, self-correction, and failure handling
- Result aggregation and synthesis
- Inter-agent delegation via the shared Blackboard
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from enum import Enum

from vetinari.agents.contracts import (
    AgentType,
    AgentResult,
    AgentTask,
    Plan,
    Task,
    TaskStatus,
    get_agent_spec,
    AGENT_REGISTRY,
)

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Strategy for task execution."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"


@dataclass
class TaskNode:
    """A node in the execution DAG."""
    task: Task
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[AgentResult] = None
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    retries: int = 0
    max_retries: int = 3


@dataclass
class ExecutionPlan:
    """An execution plan with task DAG and scheduling."""
    plan_id: str
    original_plan: Plan
    nodes: Dict[str, TaskNode] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class AgentGraph:
    """
    Hierarchical multi-agent orchestration engine for all 21 Vetinari agents.

    New in this revision
    --------------------
    - All 21 agents registered (DevOps, VersionControl, ErrorRecovery,
      ContextManager, Improvement, UserInteraction added)
    - True parallel execution via ``ThreadPoolExecutor`` for independent DAG layers
    - ``execute_plan_async`` now uses ``asyncio.gather`` over thread pool
    - Agent-model affinity routing (VL model for vision tasks)
    - Self-correction loop: failed verification triggers one guided retry
    - Delegates unresolvable failures to ``ErrorRecoveryAgent``
    """

    def __init__(
        self,
        strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE,
        max_workers: int = 5,
    ):
        self._strategy = strategy
        self._max_workers = max_workers
        self._agents: Dict[AgentType, Any] = {}
        self._execution_plans: Dict[str, ExecutionPlan] = {}
        self._initialized = False
        self._goal_tracker: Optional[Any] = None
        self._milestone_manager = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize all 21 agents and mark the graph as ready."""
        if self._initialized:
            return

        from vetinari.agents import (
            # Core
            get_planner_agent,
            get_explorer_agent,
            get_oracle_agent,
            # Core expansion
            get_librarian_agent,
            get_researcher_agent,
            get_evaluator_agent,
            get_synthesizer_agent,
            get_builder_agent,
            get_ui_planner_agent,
            # Extended
            get_security_auditor_agent,
            get_data_engineer_agent,
            get_documentation_agent,
            get_cost_planner_agent,
            get_test_automation_agent,
            get_experimentation_manager_agent,
            # Meta / interaction
            get_improvement_agent,
            get_user_interaction_agent,
            # Operations (may be None if not installed)
            get_devops_agent,
            get_version_control_agent,
            get_error_recovery_agent,
            get_context_manager_agent,
        )

        _agent_map: List[tuple] = [
            (AgentType.PLANNER,                get_planner_agent),
            (AgentType.EXPLORER,               get_explorer_agent),
            (AgentType.ORACLE,                 get_oracle_agent),
            (AgentType.LIBRARIAN,              get_librarian_agent),
            (AgentType.RESEARCHER,             get_researcher_agent),
            (AgentType.EVALUATOR,              get_evaluator_agent),
            (AgentType.SYNTHESIZER,            get_synthesizer_agent),
            (AgentType.BUILDER,                get_builder_agent),
            (AgentType.UI_PLANNER,             get_ui_planner_agent),
            (AgentType.SECURITY_AUDITOR,       get_security_auditor_agent),
            (AgentType.DATA_ENGINEER,          get_data_engineer_agent),
            (AgentType.DOCUMENTATION_AGENT,    get_documentation_agent),
            (AgentType.COST_PLANNER,           get_cost_planner_agent),
            (AgentType.TEST_AUTOMATION,        get_test_automation_agent),
            (AgentType.EXPERIMENTATION_MANAGER, get_experimentation_manager_agent),
            (AgentType.IMPROVEMENT,            get_improvement_agent),
            (AgentType.USER_INTERACTION,       get_user_interaction_agent),
            (AgentType.DEVOPS,                 get_devops_agent),
            (AgentType.VERSION_CONTROL,        get_version_control_agent),
            (AgentType.ERROR_RECOVERY,         get_error_recovery_agent),
            (AgentType.CONTEXT_MANAGER,        get_context_manager_agent),
        ]

        for agent_type, getter in _agent_map:
            if getter is None:
                logger.warning(f"[AgentGraph] Getter for {agent_type} is None — skipping")
                continue
            try:
                agent = getter()
                if agent is not None:
                    agent.initialize({})
                    self._agents[agent_type] = agent
                    logger.debug(f"[AgentGraph] Registered {agent_type.value}")
            except Exception as e:
                logger.warning(f"[AgentGraph] Could not initialize {agent_type.value}: {e}")

        logger.info(
            f"[AgentGraph] Initialized {len(self._agents)}/21 agents "
            f"(strategy={self._strategy.value})"
        )
        self._initialized = True

    # ------------------------------------------------------------------
    # Plan creation
    # ------------------------------------------------------------------

    def create_execution_plan(self, plan: Plan) -> ExecutionPlan:
        """Build an ExecutionPlan with task nodes and topological order."""
        exec_plan = ExecutionPlan(plan_id=plan.plan_id, original_plan=plan)

        for task in plan.tasks:
            node = TaskNode(
                task=task,
                dependencies=set(task.dependencies),
                status=TaskStatus.PENDING,
            )
            exec_plan.nodes[task.id] = node

        # Build reverse edges (dependents)
        for task_id, node in exec_plan.nodes.items():
            for dep_id in node.dependencies:
                if dep_id in exec_plan.nodes:
                    exec_plan.nodes[dep_id].dependents.add(task_id)

        exec_plan.execution_order = self._topological_sort(exec_plan.nodes)
        self._execution_plans[plan.plan_id] = exec_plan
        return exec_plan

    def _topological_sort(self, nodes: Dict[str, TaskNode]) -> List[str]:
        """Kahn's algorithm topological sort."""
        in_degree = {tid: len(n.dependencies) for tid, n in nodes.items()}
        queue = [tid for tid, d in in_degree.items() if d == 0]
        result: List[str] = []

        while queue:
            current = queue.pop(0)
            result.append(current)
            for dependent_id in nodes[current].dependents:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)

        if len(result) != len(nodes):
            raise ValueError("Circular dependency detected in task graph")
        return result

    # ------------------------------------------------------------------
    # Synchronous execution
    # ------------------------------------------------------------------

    def execute_plan(self, plan: Plan) -> Dict[str, AgentResult]:
        """Execute a complete plan, parallelising independent tasks where possible."""
        exec_plan = self.create_execution_plan(plan)
        exec_plan.status = TaskStatus.RUNNING
        exec_plan.started_at = datetime.now().isoformat()

        # Initialize goal tracker for drift detection
        goal_text = getattr(plan, "goal", "") or ""
        if goal_text:
            try:
                from vetinari.drift.goal_tracker import GoalTracker
                self._goal_tracker = GoalTracker(goal_text)
            except Exception:
                self._goal_tracker = None

        # Initialize milestone manager
        try:
            from vetinari.orchestration.milestones import MilestoneManager
            self._milestone_manager = MilestoneManager()
        except Exception:
            self._milestone_manager = None

        results: Dict[str, AgentResult] = {}

        try:
            if self._strategy == ExecutionStrategy.SEQUENTIAL:
                for task_id in exec_plan.execution_order:
                    node = exec_plan.nodes[task_id]
                    result = self._execute_task_node(node, results)
                    results[task_id] = result
                    node.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
            else:
                # ADAPTIVE / PARALLEL: group by layers and run each layer in parallel
                layers = self._build_execution_layers(exec_plan)
                for layer in layers:
                    layer_results = self._execute_layer_parallel(
                        layer, exec_plan, results
                    )
                    results.update(layer_results)
                    # Check milestones after each layer
                    if self._milestone_manager:
                        for tid, res in layer_results.items():
                            node = exec_plan.nodes.get(tid)
                            if node and node.task:
                                approval = self._milestone_manager.check_and_wait(
                                    node.task, res,
                                    [t for t, r in results.items() if r.success],
                                )
                                if hasattr(approval, 'action'):
                                    from vetinari.orchestration.milestones import MilestoneAction
                                    if approval.action == MilestoneAction.ABORT:
                                        raise RuntimeError("Execution aborted at milestone checkpoint")

            # Post-execution suggestions (non-blocking)
            if AgentType.ARCHITECT in self._agents:
                try:
                    from vetinari.agents.contracts import AgentTask as _AT
                    suggest_task = _AT(
                        task_id="suggestion",
                        agent_type=AgentType.ARCHITECT,
                        description="suggest improvements for project",
                        prompt=f"Suggest improvements for: {plan.goal}",
                        context={
                            "insertion_point": "post_execution",
                            "completed_outputs": [
                                str(r.output)[:200] for r in results.values() if r.success
                            ][:5],
                        },
                    )
                    suggestion_result = self._agents[AgentType.ARCHITECT].execute(suggest_task)
                    if suggestion_result.success:
                        results["_suggestions"] = suggestion_result
                except Exception as e:
                    logger.debug(f"[AgentGraph] Suggestion generation failed: {e}")

            exec_plan.status = TaskStatus.COMPLETED

        except Exception as e:
            logger.error(f"[AgentGraph] Plan execution failed: {e}")
            exec_plan.status = TaskStatus.FAILED
            raise
        finally:
            exec_plan.completed_at = datetime.now().isoformat()

        return results

    def _build_execution_layers(
        self, exec_plan: ExecutionPlan
    ) -> List[List[str]]:
        """Group tasks into parallel layers by dependency level."""
        completed: Set[str] = set()
        remaining = set(exec_plan.nodes.keys())
        layers: List[List[str]] = []

        while remaining:
            ready = [
                tid for tid in remaining
                if exec_plan.nodes[tid].dependencies <= completed
            ]
            if not ready:
                # No progress — likely a cycle that slipped through; add remaining
                ready = list(remaining)
            layers.append(ready)
            for tid in ready:
                remaining.discard(tid)
            completed.update(ready)

        return layers

    def _execute_layer_parallel(
        self,
        layer: List[str],
        exec_plan: ExecutionPlan,
        prior_results: Dict[str, AgentResult],
    ) -> Dict[str, AgentResult]:
        """Execute a batch of independent tasks in parallel via thread pool."""
        if len(layer) == 1:
            # Single task — skip thread overhead
            tid = layer[0]
            node = exec_plan.nodes[tid]
            result = self._execute_task_node(node, prior_results)
            node.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
            return {tid: result}

        layer_results: Dict[str, AgentResult] = {}
        workers = min(self._max_workers, len(layer))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {
                pool.submit(
                    self._execute_task_node,
                    exec_plan.nodes[tid],
                    prior_results,
                ): tid
                for tid in layer
            }
            for future in as_completed(future_map):
                tid = future_map[future]
                node = exec_plan.nodes[tid]
                try:
                    result = future.result()
                except Exception as e:
                    result = AgentResult(success=False, output=None, errors=[str(e)])
                node.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
                layer_results[tid] = result

        return layer_results

    # ------------------------------------------------------------------
    # Async execution (wraps thread pool via asyncio)
    # ------------------------------------------------------------------

    async def execute_plan_async(self, plan: Plan) -> Dict[str, AgentResult]:
        """Execute a plan asynchronously, running parallel layers via asyncio."""
        exec_plan = self.create_execution_plan(plan)
        exec_plan.status = TaskStatus.RUNNING
        exec_plan.started_at = datetime.now().isoformat()

        results: Dict[str, AgentResult] = {}
        loop = asyncio.get_event_loop()

        try:
            layers = self._build_execution_layers(exec_plan)
            for layer in layers:
                # Run each layer's tasks concurrently in a thread pool
                futures = [
                    loop.run_in_executor(
                        None,
                        self._execute_task_node,
                        exec_plan.nodes[tid],
                        results,
                    )
                    for tid in layer
                ]
                layer_results_list = await asyncio.gather(*futures, return_exceptions=True)
                for tid, res in zip(layer, layer_results_list):
                    node = exec_plan.nodes[tid]
                    if isinstance(res, Exception):
                        res = AgentResult(success=False, output=None, errors=[str(res)])
                    node.status = TaskStatus.COMPLETED if res.success else TaskStatus.FAILED
                    results[tid] = res

            exec_plan.status = TaskStatus.COMPLETED

        except Exception as e:
            logger.error(f"[AgentGraph] Async plan execution failed: {e}")
            exec_plan.status = TaskStatus.FAILED
            raise
        finally:
            exec_plan.completed_at = datetime.now().isoformat()

        return results

    # ------------------------------------------------------------------
    # Task execution with self-correction
    # ------------------------------------------------------------------

    def _execute_task_node(
        self,
        node: TaskNode,
        prior_results: Optional[Dict[str, AgentResult]] = None,
    ) -> AgentResult:
        """Execute a single task with retries and a self-correction loop.

        Self-correction: if the agent produces output but verification fails,
        the agent is called once more with the verification feedback injected
        into the task description before giving up.
        """
        task = node.task
        agent_type = task.assigned_agent

        if agent_type not in self._agents:
            # Try the blackboard delegation path
            from vetinari.blackboard import get_blackboard
            board = get_blackboard()
            return board.delegate(task, self._agents) or AgentResult(
                success=False,
                output=None,
                errors=[f"No agent registered for type: {agent_type}"],
            )

        agent = self._agents[agent_type]

        # Inject prior results as context if the task depends on them
        context: Dict[str, Any] = dict(task.context if hasattr(task, "context") else {})
        if prior_results and task.dependencies:
            dep_summaries = {
                dep_id: {
                    "success": prior_results[dep_id].success,
                    "output_summary": str(prior_results[dep_id].output)[:500],
                }
                for dep_id in task.dependencies
                if dep_id in prior_results
            }
            context["dependency_results"] = dep_summaries

        agent_task = AgentTask.from_task(task, task.description)
        if hasattr(agent_task, "context"):
            agent_task.context.update(context)

        for attempt in range(node.max_retries + 1):
            try:
                logger.info(
                    f"[AgentGraph] Executing {task.id} with {agent_type.value} "
                    f"(attempt {attempt + 1}/{node.max_retries + 1})"
                )
                result = agent.execute(agent_task)

                # Check goal adherence
                if self._goal_tracker and result.success:
                    try:
                        output_str = str(result.output)[:500] if result.output else ""
                        adherence = self._goal_tracker.check_adherence(
                            output_str, task.description or ""
                        )
                        if adherence.score < 0.4:
                            logger.warning(
                                f"[AgentGraph] Goal drift in {task.id}: "
                                f"score={adherence.score:.2f} — {adherence.deviation_description}"
                            )
                            result.metadata["drift_warning"] = adherence.to_dict()
                    except Exception:
                        pass

                # Handle explicit delegation: agent says "not my domain"
                if result.metadata.get("delegation_requested"):
                    reason = result.metadata.get("delegation_reason", "no reason given")
                    logger.info(
                        "[AgentGraph] %s delegated task '%s': %s — finding substitute",
                        agent_type.value, task.id, reason,
                    )
                    delegate_type = self._find_delegate(task, exclude=agent_type)
                    if delegate_type and delegate_type in self._agents:
                        delegate_agent = self._agents[delegate_type]
                        result = delegate_agent.execute(agent_task)
                    else:
                        return AgentResult(
                            success=False,
                            output=None,
                            errors=[f"Task delegated by {agent_type.value} but no substitute found: {reason}"],
                        )

                verification = agent.verify(result.output)

                if result.success and verification.passed:
                    return result

                if not verification.passed and attempt < node.max_retries:
                    # Self-correction: inject verification feedback and retry
                    issues_text = "; ".join(
                        i.get("message", str(i)) if isinstance(i, dict) else str(i)
                        for i in (verification.issues or [])
                    )
                    logger.warning(
                        f"[AgentGraph] {task.id} verification failed: {issues_text} — "
                        "injecting feedback and retrying"
                    )
                    agent_task.description = (
                        f"{task.description}\n\n"
                        f"[SELF-CORRECTION] Previous attempt failed verification. "
                        f"Issues: {issues_text}. Please fix these issues."
                    )
                    node.retries += 1
                    continue

                # Last attempt failed — try ErrorRecoveryAgent if available
                if AgentType.ERROR_RECOVERY in self._agents and attempt >= node.max_retries:
                    return self._run_error_recovery(task, result, verification)

                return AgentResult(
                    success=False,
                    output=result.output,
                    errors=[f"Verification failed after {attempt + 1} attempts: "
                            + "; ".join(
                                i.get("message", str(i)) if isinstance(i, dict) else str(i)
                                for i in (verification.issues or [])
                            )],
                )

            except Exception as e:
                logger.error(f"[AgentGraph] {task.id} raised exception: {e}")
                if attempt < node.max_retries:
                    continue
                return AgentResult(success=False, output=None, errors=[str(e)])

        return AgentResult(
            success=False,
            output=None,
            errors=["Task failed after all retries"],
        )

    def _run_error_recovery(
        self, task: Task, failed_result: AgentResult, verification: Any
    ) -> AgentResult:
        """Delegate a failed task to the ErrorRecoveryAgent for analysis."""
        try:
            recovery_agent = self._agents[AgentType.ERROR_RECOVERY]
            issues_text = "; ".join(
                i.get("message", str(i)) if isinstance(i, dict) else str(i)
                for i in (verification.issues or [])
            )
            recovery_task = AgentTask.from_task(
                task,
                f"Analyse and recover from failure in task '{task.id}': {issues_text}",
            )
            recovery_task.context["original_output"] = str(failed_result.output)[:1000]
            recovery_task.context["verification_issues"] = issues_text
            return recovery_agent.execute(recovery_task)
        except Exception as e:
            logger.debug(f"[AgentGraph] ErrorRecoveryAgent delegation failed: {e}")
            return AgentResult(
                success=False,
                output=failed_result.output,
                errors=[f"Recovery failed: {e}"],
            )

    def _find_delegate(self, task: Task, exclude: "AgentType" = None) -> Optional["AgentType"]:
        """Find the best available agent to handle a delegated task.

        Strategy:
        1. Query each registered agent's can_handle() method.
        2. Among willing agents, prefer the one whose capabilities best match
           the task description keywords.
        3. Fall back to PLANNER (which can re-route) if nothing else fits.

        Args:
            task: The task needing a substitute handler.
            exclude: The agent type that just delegated (skip it).

        Returns:
            Best-matching AgentType, or None if no substitute found.
        """
        candidates = []
        task_lower = (task.description or "").lower()

        for agent_type, agent in self._agents.items():
            if agent_type == exclude:
                continue
            try:
                agent_task = AgentTask.from_task(task, task.description)
                if agent.can_handle(agent_task):
                    # Score by capability keyword overlap
                    caps = [c.lower() for c in agent.get_capabilities()]
                    score = sum(1 for cap in caps if cap in task_lower)
                    candidates.append((score, agent_type))
            except Exception:
                pass

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_type = candidates[0]

        # If no candidate has any keyword match, fall back to PLANNER for re-routing
        if best_score == 0 and AgentType.PLANNER in self._agents:
            return AgentType.PLANNER

        return best_type

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_execution_plan(self, plan_id: str) -> Optional[ExecutionPlan]:
        return self._execution_plans.get(plan_id)

    def get_agent(self, agent_type: AgentType) -> Optional[Any]:
        return self._agents.get(agent_type)

    def get_registered_agents(self) -> List[AgentType]:
        return list(self._agents.keys())

    def __repr__(self) -> str:
        return (
            f"<AgentGraph(strategy={self._strategy.value}, "
            f"agents={len(self._agents)}/21)>"
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_agent_graph: Optional[AgentGraph] = None


def get_agent_graph(
    strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE,
) -> AgentGraph:
    """Return the singleton AgentGraph, initializing it if needed."""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = AgentGraph(strategy=strategy)
        _agent_graph.initialize()
    return _agent_graph
