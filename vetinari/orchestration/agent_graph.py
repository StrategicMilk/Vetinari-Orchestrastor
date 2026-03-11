"""
Vetinari AgentGraph - Orchestration Engine

Coordinates all agents through a Plan DAG, managing:
- 21 legacy single-purpose agents + 6 consolidated multi-mode agents (Phase 3)
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
    Hierarchical multi-agent orchestration engine for Vetinari agents.

    Registers 21 legacy single-purpose agents + 6 consolidated multi-mode
    agents (Phase 3), for a total of 27 agent slots.

    Key features
    ------------
    - True parallel execution via ``ThreadPoolExecutor`` for independent DAG layers
    - ``execute_plan_async`` uses ``asyncio.gather`` over thread pool
    - Agent-model affinity routing (VL model for vision tasks)
    - Self-correction loop: failed verification triggers one guided retry
    - Delegates unresolvable failures to ``ErrorRecoveryAgent``
    - Phase 3 consolidated agents coexist with legacy agents for backward compat
    - Phase 8 constraint enforcement (delegation rules, resource limits)
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
        """Initialize all legacy + consolidated agents and mark the graph as ready."""
        if self._initialized:
            return

        from vetinari.agents import (
            get_planner_agent,
            get_builder_agent,
            get_consolidated_researcher_agent,
            get_consolidated_oracle_agent,
            get_orchestrator_agent,
            get_architect_agent,
            get_quality_agent,
            get_operations_agent,
        )

        # --- All agent types mapped to consolidated getters ---
        _agent_map: List[tuple] = [
            (AgentType.PLANNER,                get_planner_agent),
            (AgentType.BUILDER,                get_builder_agent),
            (AgentType.EXPLORER,               get_consolidated_researcher_agent),
            (AgentType.LIBRARIAN,              get_consolidated_researcher_agent),
            (AgentType.RESEARCHER,             get_consolidated_researcher_agent),
            (AgentType.UI_PLANNER,             get_consolidated_researcher_agent),
            (AgentType.DATA_ENGINEER,          get_consolidated_researcher_agent),
            (AgentType.ORACLE,                 get_consolidated_oracle_agent),
            (AgentType.EVALUATOR,              get_quality_agent),
            (AgentType.SECURITY_AUDITOR,       get_quality_agent),
            (AgentType.TEST_AUTOMATION,        get_quality_agent),
            (AgentType.SYNTHESIZER,            get_operations_agent),
            (AgentType.DOCUMENTATION_AGENT,    get_operations_agent),
            (AgentType.COST_PLANNER,           get_operations_agent),
            (AgentType.EXPERIMENTATION_MANAGER, get_operations_agent),
            (AgentType.IMPROVEMENT,            get_operations_agent),
            (AgentType.USER_INTERACTION,       get_operations_agent),
            (AgentType.DEVOPS,                 get_operations_agent),
            (AgentType.VERSION_CONTROL,        get_operations_agent),
            (AgentType.ERROR_RECOVERY,         get_operations_agent),
            (AgentType.CONTEXT_MANAGER,        get_planner_agent),
            (AgentType.IMAGE_GENERATOR,        get_builder_agent),
        ]

        # --- Consolidated multi-mode agents (Phase 3: 6 agents) ---
        _consolidated_map: List[tuple] = [
            (AgentType.ORCHESTRATOR,              get_orchestrator_agent),
            (AgentType.CONSOLIDATED_RESEARCHER,   get_consolidated_researcher_agent),
            (AgentType.CONSOLIDATED_ORACLE,       get_consolidated_oracle_agent),
            (AgentType.ARCHITECT,                 get_architect_agent),
            (AgentType.QUALITY,                   get_quality_agent),
            (AgentType.OPERATIONS,                get_operations_agent),
        ]

        legacy_count = 0
        for agent_type, getter in _agent_map:
            if getter is None:
                logger.warning("[AgentGraph] Getter for %s is None — skipping", agent_type)
                continue
            try:
                agent = getter()
                if agent is not None:
                    agent.initialize({})
                    self._agents[agent_type] = agent
                    legacy_count += 1
                    logger.debug("[AgentGraph] Registered legacy %s", agent_type.value)
            except Exception as e:
                logger.warning("[AgentGraph] Could not initialize %s: %s", agent_type.value, e)

        consolidated_count = 0
        for agent_type, getter in _consolidated_map:
            try:
                agent = getter()
                if agent is not None:
                    agent.initialize({})
                    self._agents[agent_type] = agent
                    consolidated_count += 1
                    logger.debug("[AgentGraph] Registered consolidated %s", agent_type.value)
            except Exception as e:
                logger.warning("[AgentGraph] Could not initialize consolidated %s: %s", agent_type.value, e)

        logger.info(
            f"[AgentGraph] Initialized {len(self._agents)} agents "
            f"({legacy_count} legacy + {consolidated_count} consolidated, "
            f"strategy={self._strategy.value})"
        )
        self._initialized = True

    # ------------------------------------------------------------------
    # Plan creation
    # ------------------------------------------------------------------

    def create_execution_plan(self, plan: Plan) -> ExecutionPlan:
        """Build an ExecutionPlan with task nodes and topological order.

        Phase 8.8: Validates delegation constraints between tasks — if task B
        depends on task A, the delegation from A's agent to B's agent must be
        allowed by architecture constraints.
        """
        exec_plan = ExecutionPlan(plan_id=plan.plan_id, original_plan=plan)

        for task in plan.tasks:
            node = TaskNode(
                task=task,
                dependencies=set(task.dependencies),
                status=TaskStatus.PENDING,
            )
            exec_plan.nodes[task.id] = node

        # Phase 8.8: Validate delegation constraints across the DAG
        try:
            from vetinari.constraints.registry import get_constraint_registry
            _reg = get_constraint_registry()
            for task in plan.tasks:
                for dep_id in task.dependencies:
                    dep_task = exec_plan.nodes.get(dep_id)
                    if dep_task is None:
                        continue
                    from_type = dep_task.task.assigned_agent
                    to_type = task.assigned_agent
                    if from_type == to_type:
                        continue  # Same agent, no delegation
                    from_val = from_type.value if hasattr(from_type, 'value') else str(from_type)
                    to_val = to_type.value if hasattr(to_type, 'value') else str(to_type)
                    allowed, reason = _reg.validate_delegation(from_val, to_val)
                    if not allowed:
                        logger.warning(
                            f"[AgentGraph] Delegation constraint violation: "
                            f"{from_val} -> {to_val} for task {task.id}: {reason}"
                        )
        except Exception:
            logger.debug("Constraint system not available, proceeding without validation")

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
            logger.error("[AgentGraph] Plan execution failed: %s", e)
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
            logger.error("[AgentGraph] Async plan execution failed: %s", e)
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

        Phase 8.8: Enforces architecture constraints (delegation rules) and
        resource constraints (max_retries) before and during execution.
        """
        task = node.task
        agent_type = task.assigned_agent

        # ----- Phase 8.8: Resource constraint enforcement -----
        try:
            from vetinari.constraints.registry import get_constraint_registry
            _reg = get_constraint_registry()
            _ac = _reg.get_constraints_for_agent(agent_type.value if hasattr(agent_type, 'value') else str(agent_type))
            if _ac and _ac.resources:
                # Cap retries to the resource-constrained maximum
                constrained_retries = min(node.max_retries, _ac.resources.max_retries)
                if constrained_retries < node.max_retries:
                    logger.debug(
                        f"[AgentGraph] Capping retries for {agent_type} "
                        f"from {node.max_retries} to {constrained_retries} (constraint)"
                    )
                    node.max_retries = constrained_retries
        except Exception:
            logger.debug("Failed to apply agent constraints for %s", agent_type, exc_info=True)

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

        # ----- Phase 7.9H: Permission enforcement before execution -----
        try:
            from vetinari.execution_context import get_context_manager, ToolPermission, ExecutionMode
            ctx_mgr = get_context_manager()
            ctx_mgr.enforce_permission(
                ToolPermission.MODEL_INFERENCE,
                f"agent_execute:{agent_type.value}",
            )
        except PermissionError:
            logger.warning(
                f"[AgentGraph] Permission denied for {agent_type.value} — "
                "MODEL_INFERENCE not allowed in current execution mode"
            )
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Permission denied: MODEL_INFERENCE required for {agent_type.value}"],
            )
        except Exception:
            logger.debug("Context manager not configured, allowing execution")

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

        # ----- Phase 7.9I: Let agents incorporate dependency results -----
        if hasattr(agent, "_incorporate_prior_results"):
            try:
                agent._incorporate_prior_results(agent_task)
            except Exception as e:
                logger.debug("[AgentGraph] _incorporate_prior_results failed: %s", e)

        for attempt in range(node.max_retries + 1):
            try:
                logger.info(
                    f"[AgentGraph] Executing {task.id} with {agent_type.value} "
                    f"(attempt {attempt + 1}/{node.max_retries + 1})"
                )
                # Switch to EXECUTION mode so tool permission checks pass
                try:
                    from vetinari.execution_context import get_context_manager as _get_ctx, ExecutionMode as _ExecMode
                    _ctx_mgr = _get_ctx()
                    _exec_ctx = _ctx_mgr.temporary_mode(_ExecMode.EXECUTION, task_id=task.id)
                    _exec_ctx.__enter__()
                except Exception:
                    _exec_ctx = None  # Context manager unavailable — degrade gracefully

                try:
                    result = agent.execute(agent_task)
                finally:
                    if _exec_ctx is not None:
                        try:
                            _exec_ctx.__exit__(None, None, None)
                        except Exception:
                            pass

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
                    # Phase 8.2: Log output schema deviations (non-blocking)
                    schema_issues = self._validate_output_schema(
                        agent_type, result.output
                    )
                    if schema_issues:
                        logger.info(
                            f"[AgentGraph] {task.id} output schema deviations: "
                            + "; ".join(schema_issues)
                        )

                    # Phase 7.9K: Maker-checker for BUILDER outputs
                    _builder_types = {AgentType.BUILDER}
                    if agent_type in _builder_types and AgentType.QUALITY in self._agents:
                        result = self._apply_maker_checker(task, result)

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
                logger.error("[AgentGraph] %s raised exception: %s", task.id, e)
                if attempt < node.max_retries:
                    continue
                return AgentResult(success=False, output=None, errors=[str(e)])

        return AgentResult(
            success=False,
            output=None,
            errors=["Task failed after all retries"],
        )

    # ------------------------------------------------------------------
    # Phase 7.9J: Dynamic graph modification
    # ------------------------------------------------------------------

    def inject_task(
        self,
        plan_id: str,
        new_task: Task,
        after_task_id: str,
    ) -> bool:
        """Inject a new task into an in-flight execution plan.

        The new task is inserted as a dependent of *after_task_id*.
        Any tasks that previously depended on *after_task_id* are re-wired
        to depend on the new task instead, preserving the DAG structure.

        Use cases:
        - ErrorRecoveryAgent injects corrective tasks
        - PLANNER splits a subtask mid-execution
        - ORCHESTRATOR adds a review task after a risky step

        Args:
            plan_id: ID of the execution plan to modify.
            new_task: The Task to inject.
            after_task_id: The task ID after which to insert.

        Returns:
            True if injection succeeded, False otherwise.
        """
        plan = self._execution_plans.get(plan_id)
        if plan is None:
            logger.warning("[AgentGraph] inject_task: plan %s not found", plan_id)
            return False

        if after_task_id not in plan.nodes:
            logger.warning(
                f"[AgentGraph] inject_task: after_task {after_task_id} not in plan"
            )
            return False

        if new_task.id in plan.nodes:
            logger.warning(
                f"[AgentGraph] inject_task: task {new_task.id} already exists"
            )
            return False

        # Create the new node
        new_node = TaskNode(
            task=new_task,
            dependencies={after_task_id},
        )

        # Re-wire: tasks that depended on after_task_id now depend on new_task
        for node_id, node in plan.nodes.items():
            if after_task_id in node.dependencies:
                node.dependencies.discard(after_task_id)
                node.dependencies.add(new_task.id)

        # Update dependents set on the after_task node
        after_node = plan.nodes[after_task_id]
        after_node.dependents.discard(new_task.id)  # safety
        after_node.dependents = {new_task.id}

        # Set new node's dependents to what after_node previously pointed to
        new_node.dependents = {
            nid for nid, n in plan.nodes.items()
            if new_task.id in n.dependencies
        }

        plan.nodes[new_task.id] = new_node

        # Rebuild execution order
        plan.execution_order = self._topological_sort(plan.nodes)

        logger.info(
            f"[AgentGraph] Injected task {new_task.id} after {after_task_id} "
            f"in plan {plan_id}"
        )
        return True

    # ------------------------------------------------------------------
    # Phase 7.9K: Maker-checker pattern (BUILDER → QUALITY feedback loop)
    # ------------------------------------------------------------------

    _MAKER_CHECKER_MAX_ITERATIONS = 3

    def _apply_maker_checker(
        self,
        task: Task,
        result: AgentResult,
    ) -> AgentResult:
        """Run maker-checker loop: QUALITY reviews BUILDER output.

        If the QUALITY agent's review fails, a new BUILDER task is created
        with the feedback injected, up to ``_MAKER_CHECKER_MAX_ITERATIONS``.

        Only triggers when:
        - The original task's agent is BUILDER (or a legacy builder-like type)
        - QUALITY agent is registered
        - The result was successful

        Returns the final result (either QUALITY-approved or last attempt).
        """
        quality_agent = self._agents.get(AgentType.QUALITY)
        builder_agent = self._agents.get(AgentType.BUILDER)
        if quality_agent is None or builder_agent is None:
            return result

        current_result = result
        for iteration in range(self._MAKER_CHECKER_MAX_ITERATIONS):
            # Ask QUALITY to review
            review_task = AgentTask(
                task_id=f"{task.id}_review_{iteration}",
                agent_type=AgentType.QUALITY,
                description=(
                    f"Review the following output from BUILDER task '{task.id}':\n\n"
                    f"{str(current_result.output)[:2000]}\n\n"
                    f"Original task: {task.description}"
                ),
                prompt=(
                    f"Review the following output from BUILDER task '{task.id}':\n\n"
                    f"{str(current_result.output)[:2000]}\n\n"
                    f"Original task: {task.description}"
                ),
                context={
                    "review_type": "code_review",
                    "original_task_id": task.id,
                    "iteration": iteration,
                },
            )
            try:
                review_result = quality_agent.execute(review_task)
                review_verification = quality_agent.verify(review_result.output)

                if review_result.success and review_verification.passed:
                    logger.info(
                        f"[AgentGraph] Maker-checker: QUALITY approved "
                        f"{task.id} on iteration {iteration + 1}"
                    )
                    # Enrich original result with review metadata
                    if current_result.metadata is None:
                        current_result.metadata = {}
                    current_result.metadata["maker_checker"] = {
                        "approved": True,
                        "iterations": iteration + 1,
                        "review_score": getattr(review_verification, "score", None),
                    }
                    return current_result

                # Review failed — feed back to BUILDER
                issues_text = "; ".join(
                    i.get("message", str(i)) if isinstance(i, dict) else str(i)
                    for i in (review_verification.issues or [])
                )
                logger.warning(
                    f"[AgentGraph] Maker-checker: QUALITY rejected {task.id} "
                    f"(iteration {iteration + 1}): {issues_text}"
                )

                if iteration < self._MAKER_CHECKER_MAX_ITERATIONS - 1:
                    # Re-run BUILDER with feedback
                    fix_task = AgentTask(
                        task_id=f"{task.id}_fix_{iteration}",
                        agent_type=AgentType.BUILDER,
                        description=(
                            f"{task.description}\n\n"
                            f"[MAKER-CHECKER FEEDBACK] Previous output was rejected "
                            f"by QUALITY review. Issues: {issues_text}. "
                            f"Please fix these issues."
                        ),
                        prompt=(
                            f"{task.description}\n\n"
                            f"[MAKER-CHECKER FEEDBACK] Previous output was rejected "
                            f"by QUALITY review. Issues: {issues_text}. "
                            f"Please fix these issues."
                        ),
                        context={
                            "original_output": str(current_result.output)[:1000],
                            "review_issues": issues_text,
                            "iteration": iteration + 1,
                        },
                    )
                    current_result = builder_agent.execute(fix_task)
                    if not current_result.success:
                        break
            except Exception as e:
                logger.debug("[AgentGraph] Maker-checker iteration failed: %s", e)
                break

        # Exhausted iterations
        if current_result.metadata is None:
            current_result.metadata = {}
        current_result.metadata["maker_checker"] = {
            "approved": False,
            "iterations": self._MAKER_CHECKER_MAX_ITERATIONS,
        }
        return current_result

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
            logger.debug("[AgentGraph] ErrorRecoveryAgent delegation failed: %s", e)
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
    # Output schema validation (Phase 8.2 deferred)
    # ------------------------------------------------------------------

    def _validate_output_schema(
        self, agent_type: AgentType, output: Any
    ) -> List[str]:
        """Validate agent output against the SkillSpec output_schema.

        Returns list of validation issues (empty = valid).
        This is a lightweight structural check — verifies required keys
        are present and value types match, without a full JSON Schema
        library dependency.
        """
        spec = self.get_skill_spec(agent_type)
        if spec is None or not spec.output_schema:
            return []

        schema = spec.output_schema
        if not isinstance(output, dict):
            # Non-dict outputs can't be validated against object schemas
            return []

        issues = []
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for key in required:
            if key not in output:
                issues.append(f"Missing required output field: '{key}'")

        _TYPE_MAP = {
            "string": str, "number": (int, float),
            "integer": int, "boolean": bool,
            "array": list, "object": dict,
        }
        for key, prop_schema in properties.items():
            if key not in output:
                continue
            expected_type = prop_schema.get("type")
            if expected_type and expected_type in _TYPE_MAP:
                py_type = _TYPE_MAP[expected_type]
                if not isinstance(output[key], py_type):
                    issues.append(
                        f"Field '{key}' expected type {expected_type}, "
                        f"got {type(output[key]).__name__}"
                    )

        if issues:
            logger.debug(
                f"[AgentGraph] Output schema issues for {agent_type.value}: "
                + "; ".join(issues)
            )
        return issues

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_execution_plan(self, plan_id: str) -> Optional[ExecutionPlan]:
        return self._execution_plans.get(plan_id)

    def get_agent(self, agent_type: AgentType) -> Optional[Any]:
        return self._agents.get(agent_type)

    def get_registered_agents(self) -> List[AgentType]:
        return list(self._agents.keys())

    # ------------------------------------------------------------------
    # Capability-based routing via SkillSpec (Phase 7 deferred)
    # ------------------------------------------------------------------

    def get_agent_by_capability(self, capability: str) -> Optional[Any]:
        """Find the best registered agent that declares a given capability.

        Consults the programmatic SkillSpec registry to map capabilities
        to agent types, then returns the live agent instance if registered.
        Prefers consolidated agents over legacy agents.
        """
        try:
            from vetinari.skills.skill_registry import get_skills_by_capability
            matching_specs = get_skills_by_capability(capability)
            if not matching_specs:
                return None

            # Try consolidated agents first (they have richer mode support)
            _AGENT_TYPE_PRIORITY = {
                "ORCHESTRATOR": 10, "CONSOLIDATED_RESEARCHER": 10,
                "CONSOLIDATED_ORACLE": 10, "ARCHITECT": 10,
                "QUALITY": 10, "OPERATIONS": 10,
                "PLANNER": 8, "BUILDER": 8,
            }
            matching_specs.sort(
                key=lambda s: _AGENT_TYPE_PRIORITY.get(s.agent_type, 0),
                reverse=True,
            )

            for spec in matching_specs:
                agent_type_str = spec.agent_type
                try:
                    at = AgentType(agent_type_str)
                except ValueError:
                    continue
                agent = self._agents.get(at)
                if agent is not None:
                    return agent
        except Exception as e:
            logger.debug("[AgentGraph] Capability lookup for '%s' failed: %s", capability, e)
        return None

    def get_skill_spec(self, agent_type: AgentType) -> Optional[Any]:
        """Return the SkillSpec for a given agent type, if one exists."""
        try:
            from vetinari.skills.skill_registry import get_skill_for_agent_type
            return get_skill_for_agent_type(agent_type.value)
        except Exception:
            return None

    def get_agents_for_task_type(self, task_type: str) -> List[AgentType]:
        """Return agent types whose SkillSpec modes include the given task type."""
        results = []
        try:
            from vetinari.skills.skill_registry import get_all_skills
            for spec in get_all_skills().values():
                if task_type in spec.modes or task_type in spec.capabilities:
                    try:
                        at = AgentType(spec.agent_type)
                        if at in self._agents:
                            results.append(at)
                    except ValueError:
                        logger.debug("Unrecognized agent type %s in skill spec", spec.agent_type, exc_info=True)
        except Exception:
            logger.debug("Failed to resolve agents by skill for task type %s", task_type, exc_info=True)
        return results

    def __repr__(self) -> str:
        return (
            f"<AgentGraph(strategy={self._strategy.value}, "
            f"agents={len(self._agents)}/27)>"
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
