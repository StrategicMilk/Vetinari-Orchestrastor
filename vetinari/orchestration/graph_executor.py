"""Plan and layer execution orchestration for the AgentGraph.

Runs plans synchronously (``execute_plan``), asynchronously
(``execute_plan_async``), or as filtered subgraphs (``execute_subgraph``).
Parallel layers are dispatched via ``ThreadPoolExecutor``; async execution
wraps the thread pool through ``asyncio.gather``.

Per-task logic (retries, safety checks, self-correction) lives in
``graph_task_runner.GraphTaskRunnerMixin``, which this mixin inherits.

This is step 2 of the execution flow: take the ExecutionDAG built by
``GraphPlannerMixin`` and drive each task through the assigned agent in
dependency order.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    Plan,
)
from vetinari.orchestration.graph_task_runner import GraphTaskRunnerMixin
from vetinari.orchestration.graph_types import ExecutionDAG, ExecutionStrategy
from vetinari.types import AgentType, StatusEnum

# Lazy loader for VRAM manager — avoids importing heavy GPU code at module level.
# We store the module reference (not a direct function reference) so that
# unittest.mock.patch("vetinari.models.vram_manager.get_vram_manager", ...) is
# intercepted correctly in tests — patching replaces the attribute on the module
# object, not on a cached function reference.
_vram_manager_mod = None
_vram_manager_checked = False


def _lazy_get_vram_manager():
    """Return the VRAMManager singleton, or None if unavailable."""
    global _vram_manager_mod, _vram_manager_checked
    if not _vram_manager_checked:
        try:
            import vetinari.models.vram_manager as _mod

            _vram_manager_mod = _mod
        except ImportError:
            _vram_manager_mod = None
        _vram_manager_checked = True
    if _vram_manager_mod is None:
        return None
    return _vram_manager_mod.get_vram_manager()


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level lazy getters for optional/potentially-circular imports
# ---------------------------------------------------------------------------

# GoalTracker — optional drift-detection feature
_GoalTracker = None


def _get_goal_tracker_class():
    global _GoalTracker
    if _GoalTracker is None:
        from vetinari.drift.goal_tracker import GoalTracker

        _GoalTracker = GoalTracker
    return _GoalTracker


# MilestoneManager + MilestoneAction — optional milestone approval feature
_MilestoneManager = None
_MilestoneAction = None


def _get_milestone_manager_class():
    global _MilestoneManager
    if _MilestoneManager is None:
        from vetinari.orchestration.milestones import MilestoneManager

        _MilestoneManager = MilestoneManager
    return _MilestoneManager


def _get_milestone_action_class():
    global _MilestoneAction
    if _MilestoneAction is None:
        from vetinari.orchestration.milestones import MilestoneAction

        _MilestoneAction = MilestoneAction
    return _MilestoneAction


class GraphExecutorMixin(GraphTaskRunnerMixin):
    """Plan, layer, and subgraph execution for AgentGraph.

    Provides synchronous and async plan execution, subgraph execution,
    and parallel layer dispatch. Per-task execution logic (retries,
    safety checks, self-correction) is inherited from GraphTaskRunnerMixin.

    Mixed into AgentGraph alongside GraphPlannerMixin, GraphValidatorMixin,
    and GraphRecoveryMixin.

    Attributes expected on ``self``:
        _agents (dict[AgentType, Any]): Registered agent instances.
        _strategy (ExecutionStrategy): Current execution strategy.
        _max_workers (int): Thread pool size cap.
        _wip_tracker: Optional WIP tracking object.
        _goal_tracker: Optional goal drift tracker.
        _milestone_manager: Optional milestone approval manager.
        _quality_reviewed_agents (set): Agent types that trigger maker-checker.
        _execution_plans (dict[str, ExecutionDAG]): Plan cache.
    """

    # ------------------------------------------------------------------
    # Synchronous execution
    # ------------------------------------------------------------------

    def execute_plan(self, plan: Plan) -> dict[str, AgentResult]:
        """Execute a complete plan, parallelising independent tasks where possible.

        Args:
            plan: The Plan to execute.

        Returns:
            Mapping of task ID to AgentResult for every task in the plan,
            reflecting the final status (success or failure) of each.

        Raises:
            RuntimeError: If a milestone checkpoint aborts execution.
            Exception: Re-raises any unhandled exception from task execution.
        """
        exec_plan = self.create_execution_plan(plan)
        exec_plan.status = StatusEnum.RUNNING
        exec_plan.started_at = datetime.now(timezone.utc).isoformat()

        # Initialize goal tracker for drift detection
        goal_text = getattr(plan, "goal", "")
        if goal_text:
            try:
                self._goal_tracker = _get_goal_tracker_class()(goal_text)
            except Exception:  # Broad: optional feature; any failure must not block task execution
                self._goal_tracker = None

        # Pre-execution scope-creep scan: flag tasks that are unlikely to
        # contribute to the stated goal before any work begins.
        if self._goal_tracker is not None and plan.tasks:
            try:
                creep_items = self._goal_tracker.detect_scope_creep(plan.tasks)
                if creep_items:
                    logger.warning(
                        "[AgentGraph] Scope creep detected in plan %s: %d task(s) have low goal "
                        "relevance and may be outside the stated goal — review before accepting results",
                        plan.plan_id,
                        len(creep_items),
                    )
            except Exception:  # Broad: optional feature; any failure must not block task execution
                logger.warning(
                    "[AgentGraph] Scope-creep scan failed for plan %s — "
                    "proceeding without pre-flight goal alignment check",
                    plan.plan_id,
                )

        # Initialize milestone manager and wire the approval callback.
        # The callback is sourced from ``self._milestone_approval_callback`` if
        # the AgentGraph was configured with one (e.g. by the web layer calling
        # set_milestone_approval_callback), otherwise milestones auto-approve.
        try:
            self._milestone_manager = _get_milestone_manager_class()()
            _approval_cb = getattr(self, "_milestone_approval_callback", None)
            if _approval_cb is not None:
                self._milestone_manager.set_approval_callback(_approval_cb)
        except Exception:  # Broad: optional feature; any failure must not block task execution
            self._milestone_manager = None

        results: dict[str, AgentResult] = {}

        try:
            if self._strategy == ExecutionStrategy.SEQUENTIAL:
                _remaining_order = list(exec_plan.execution_order)
                _idx = 0
                while _idx < len(_remaining_order):
                    task_id = _remaining_order[_idx]
                    node = exec_plan.nodes.get(task_id)
                    if node is None or node.status == StatusEnum.COMPLETED:
                        _idx += 1
                        continue
                    result = self._execute_task_node(node, results)
                    results[task_id] = result
                    node.status = StatusEnum.COMPLETED if result.success else StatusEnum.FAILED

                    # Mid-execution replan check
                    if result.success and node.task and self._should_replan(node.task, result):
                        remaining_tasks = [
                            exec_plan.nodes[tid].task
                            for tid in _remaining_order[_idx + 1 :]
                            if tid in exec_plan.nodes
                            and exec_plan.nodes[tid].task is not None
                            and exec_plan.nodes[tid].status == StatusEnum.PENDING
                        ]
                        if remaining_tasks:
                            replan_result = self._trigger_replan(
                                node.task,
                                result,
                                remaining_tasks,
                            )
                            if replan_result.new_tasks != remaining_tasks:
                                self._replace_remaining_tasks(
                                    exec_plan,
                                    replan_result.new_tasks,
                                )
                                # Rebuild remaining order from updated plan
                                _remaining_order = [
                                    tid
                                    for tid in exec_plan.execution_order
                                    if tid in exec_plan.nodes and exec_plan.nodes[tid].status == StatusEnum.PENDING
                                ]
                                _idx = 0
                                continue

                    _idx += 1
            else:
                # ADAPTIVE / PARALLEL: group by layers and run each layer in parallel
                layers = self._build_execution_layers(exec_plan)
                for layer in layers:
                    layer_results = self._execute_layer_parallel(layer, exec_plan, results)
                    results.update(layer_results)
                    # Check milestones after each layer
                    if self._milestone_manager:
                        for tid, res in layer_results.items():
                            node = exec_plan.nodes.get(tid)
                            if node and node.task:
                                approval = self._milestone_manager.check_and_wait(
                                    node.task,
                                    res,
                                    [t for t, r in results.items() if r.success],
                                )
                                if (
                                    hasattr(approval, "action")
                                    and approval.action == _get_milestone_action_class().ABORT
                                ):
                                    raise RuntimeError("Execution aborted at milestone checkpoint")

            # Post-execution suggestions (non-blocking)
            if AgentType.WORKER in self._agents:
                try:
                    suggest_task = AgentTask(
                        task_id="suggestion",
                        agent_type=AgentType.WORKER,
                        description="suggest improvements for project",
                        prompt=f"Suggest improvements for: {plan.goal}",
                        context={
                            "insertion_point": "post_execution",
                            "completed_outputs": [str(r.output)[:200] for r in results.values() if r.success][:5],
                        },
                    )
                    suggestion_result = self._agents[AgentType.WORKER].execute(suggest_task)
                    if suggestion_result.success:
                        results["_suggestions"] = suggestion_result
                except Exception as e:
                    logger.warning("[AgentGraph] Suggestion generation failed: %s", e)

            # Operations auto-synthesis after plan completion
            if AgentType.WORKER in self._agents:
                try:
                    ops_agent = self._agents[AgentType.WORKER]
                    synth_task = AgentTask(
                        task_id="auto_synthesis",
                        agent_type=AgentType.WORKER,
                        description="Synthesise execution results into a summary report",
                        prompt=f"Synthesise results for plan: {plan.goal}",
                        context={
                            "mode": "synthesis",
                            "completed_tasks": [tid for tid, r in results.items() if r.success],
                            "failed_tasks": [tid for tid, r in results.items() if not r.success],
                        },
                    )
                    synth_result = ops_agent.execute(synth_task)
                    if synth_result.success:
                        results["_synthesis"] = synth_result
                except Exception as e:
                    logger.warning("[AgentGraph] Operations auto-synthesis failed: %s", e)

            # Only mark the plan COMPLETED when every node that ran succeeded.
            # If any node ended FAILED (including nodes that raised and were caught
            # per-node), the plan outcome must reflect that failure (fail-closed).
            any_failed = any(node.status == StatusEnum.FAILED for node in exec_plan.nodes.values())
            exec_plan.status = StatusEnum.FAILED if any_failed else StatusEnum.COMPLETED

        except Exception as e:
            logger.error("[AgentGraph] Plan execution failed: %s", e)
            exec_plan.status = StatusEnum.FAILED
            raise
        finally:
            exec_plan.completed_at = datetime.now(timezone.utc).isoformat()

        return results

    def _execute_layer_parallel(
        self,
        layer: list[str],
        exec_plan: ExecutionDAG,
        prior_results: dict[str, AgentResult],
    ) -> dict[str, AgentResult]:
        """Execute a batch of independent tasks in parallel via thread pool.

        For single-task layers, skips the thread pool overhead entirely.
        Worker count is capped by VRAM availability — if there isn't enough
        free + evictable VRAM for all tasks, parallelism is reduced to avoid
        cascading model evictions.  Decision: ADR-0087.

        Args:
            layer: List of task IDs to execute in this layer.
            exec_plan: The execution plan containing the nodes.
            prior_results: Results from all previously completed tasks.

        Returns:
            Mapping of task ID to AgentResult for this layer's tasks.
        """
        if len(layer) == 1:
            # Single task — skip thread overhead
            tid = layer[0]
            node = exec_plan.nodes[tid]
            result = self._execute_task_node(node, prior_results)
            node.status = StatusEnum.COMPLETED if result.success else StatusEnum.FAILED
            return {tid: result}

        layer_results: dict[str, AgentResult] = {}
        workers = min(self._max_workers, len(layer))

        # VRAM-aware worker throttling: estimate how many tasks can run
        # concurrently without forcing eviction of actively-used models.
        workers = self._vram_throttle_workers(workers, layer, exec_plan)

        # Snapshot prior_results so parallel workers don't see concurrent mutations.
        results_snapshot = dict(prior_results) if prior_results else {}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {
                pool.submit(
                    self._execute_task_node,
                    exec_plan.nodes[tid],
                    results_snapshot,
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
                node.status = StatusEnum.COMPLETED if result.success else StatusEnum.FAILED
                layer_results[tid] = result

        return layer_results

    def _vram_throttle_workers(
        self,
        max_workers: int,
        layer: list[str],
        exec_plan: ExecutionDAG,
    ) -> int:
        """Reduce worker count if VRAM can't support full parallelism.

        Estimates per-task VRAM from the average loaded model size and checks
        how many concurrent tasks fit in available + evictable VRAM.  Never
        reduces below 1.

        Args:
            max_workers: Upper bound on workers from config / layer size.
            layer: Task IDs in this layer.
            exec_plan: The execution plan (used to look up agent types).

        Returns:
            Adjusted worker count (1 <= result <= max_workers).
        """
        try:
            mgr = _lazy_get_vram_manager()
            available_gb = mgr.get_max_available_vram_gb()

            # Estimate per-task VRAM: use average of currently-loaded models
            # as a proxy for what each new task will need.
            loaded = list(mgr._estimates.values())
            if not loaded:
                return max_workers
            avg_model_gb = sum(e.total_gpu_gb for e in loaded) / len(loaded)
            if avg_model_gb <= 0:
                return max_workers

            # How many tasks can fit if each needs ~avg_model_gb?
            # Some tasks may share models (warm bonus makes this likely),
            # so we use 0.7x as a sharing discount.
            sharing_discount = 0.7
            fits = max(1, int(available_gb / (avg_model_gb * sharing_discount)))
            throttled = min(max_workers, fits)

            if throttled < max_workers:
                logger.info(
                    "VRAM throttle: layer has %d tasks but only %.1f GB available "
                    "(avg model %.1f GB) — limiting to %d concurrent workers",
                    len(layer),
                    available_gb,
                    avg_model_gb,
                    throttled,
                )

            return throttled

        except Exception:
            logger.warning(
                "VRAMManager unavailable for layer throttling — using default worker count of %d", max_workers
            )
            return max_workers

    def execute_subgraph(
        self,
        plan: Plan,
        task_ids: list[str],
    ) -> dict[str, AgentResult]:
        """Execute a subset of tasks from an existing plan.

        Creates a filtered view of the plan containing only the specified task
        IDs and their intra-set dependencies, then executes them as a subgraph.

        Args:
            plan: The full plan from which tasks are drawn.
            task_ids: List of task IDs to execute from the plan.

        Returns:
            Mapping of task_id to AgentResult for the executed subset.
        """
        logger.info("execute_subgraph: running %d tasks from plan %s", len(task_ids), plan.id)
        task_id_set = set(task_ids)

        # Build a filtered plan with only the requested tasks
        filtered_tasks = [t for t in plan.tasks if t.id in task_id_set]
        if not filtered_tasks:
            logger.warning("execute_subgraph: no matching tasks found in plan")
            return {}

        # Create a mini-plan with the subset
        subgraph_plan = Plan(
            id=f"{plan.id}-sub",
            goal=plan.goal,
            tasks=filtered_tasks,
        )

        return self.execute_plan(subgraph_plan)

    # ------------------------------------------------------------------
    # Async execution (wraps thread pool via asyncio)
    # ------------------------------------------------------------------

    async def execute_plan_async(self, plan: Plan) -> dict[str, AgentResult]:
        """Execute a plan asynchronously, running parallel layers via asyncio.

        Args:
            plan: The Plan to execute.

        Returns:
            Mapping of task ID to AgentResult for every task that completed
            before any exception, mirroring the structure of ``execute_plan``.

        Raises:
            Exception: Re-raises any exception that occurs during layer execution
                after marking the plan as failed.
        """
        exec_plan = self.create_execution_plan(plan)
        exec_plan.status = StatusEnum.RUNNING
        exec_plan.started_at = datetime.now(timezone.utc).isoformat()

        results: dict[str, AgentResult] = {}
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
                    node.status = StatusEnum.COMPLETED if res.success else StatusEnum.FAILED
                    results[tid] = res

            # Mirror the sync path: inspect each node's status so that a plan
            # containing any FAILED node is itself marked FAILED (fail-closed).
            any_failed = any(node.status == StatusEnum.FAILED for node in exec_plan.nodes.values())
            exec_plan.status = StatusEnum.FAILED if any_failed else StatusEnum.COMPLETED

        except Exception as e:
            logger.error("[AgentGraph] Async plan execution failed: %s", e)
            exec_plan.status = StatusEnum.FAILED
            raise
        finally:
            exec_plan.completed_at = datetime.now(timezone.utc).isoformat()

        return results
