"""Error recovery, delegation, and dynamic graph modification for AgentGraph.

Handles all failure-path logic after a task execution:
- maker-checker quality loop (WORKER → INSPECTOR feedback cycle)
- error recovery via the recovery agent and optional FOREMAN re-decomposition
- capability-based delegation when the assigned agent declines a task
- mid-flight task injection into running execution plans

This is the fallback layer that runs when the main execution path cannot
produce a verified result.
"""

from __future__ import annotations

import logging
from typing import Any

from vetinari.agents.contracts import AgentResult, AgentTask, Task
from vetinari.constants import TRUNCATE_OUTPUT_PREVIEW, TRUNCATE_OUTPUT_SUMMARY
from vetinari.orchestration.graph_types import TaskNode
from vetinari.types import AgentType

logger = logging.getLogger(__name__)


class GraphRecoveryMixin:
    """Error handling and recovery methods for AgentGraph.

    Provides maker-checker quality loops, error recovery delegation,
    capability-based agent substitution, and dynamic task injection.
    Mixed into AgentGraph alongside the planner, executor, and validator mixins.

    Attributes expected on ``self``:
        _agents (dict[AgentType, Any]): Registered agent instances.
        _execution_plans (dict[str, ExecutionPlan]): Active execution plans.
        _topological_sort: Method from GraphPlannerMixin.
    """

    # Maximum iterations for the maker-checker approval loop.
    _MAKER_CHECKER_MAX_ITERATIONS = 3

    # ------------------------------------------------------------------
    # Maker-checker quality loop
    # ------------------------------------------------------------------

    def _apply_maker_checker(
        self,
        task: Task,
        result: AgentResult,
    ) -> AgentResult:
        """Run maker-checker loop: INSPECTOR reviews WORKER output.

        If the INSPECTOR agent's review fails, a new WORKER task is created
        with the feedback injected, up to ``_MAKER_CHECKER_MAX_ITERATIONS``.

        Only triggers when:
        - The original task's agent is WORKER
        - INSPECTOR agent is registered
        - The result was successful

        Args:
            task: The original task whose output is being reviewed.
            result: The successful AgentResult to validate.

        Returns:
            The final result — either INSPECTOR-approved or the last attempt
            after all iterations are exhausted.
        """
        quality_agent = self._agents.get(AgentType.INSPECTOR)
        builder_agent = self._agents.get(AgentType.WORKER)
        if quality_agent is None or builder_agent is None:
            return result

        current_result = result
        # ACON-style condensation for maker-checker handoffs
        try:
            from vetinari.context import get_context_condenser

            _mc_condenser = get_context_condenser()
        except Exception:  # Broad: optional feature; any failure must not block task execution
            _mc_condenser = None

        for iteration in range(self._MAKER_CHECKER_MAX_ITERATIONS):
            # Condense Worker output for Inspector review
            if _mc_condenser is not None:
                _condensed_output = _mc_condenser.condense_for_handoff(
                    AgentType.WORKER.value,
                    AgentType.INSPECTOR.value,
                    current_result.output,
                    current_result.metadata,
                )
            else:
                _condensed_output = str(current_result.output)[:TRUNCATE_OUTPUT_PREVIEW]

            # Ask INSPECTOR to review
            review_task = AgentTask(
                task_id=f"{task.id}_review_{iteration}",
                agent_type=AgentType.INSPECTOR,
                description=(
                    f"Review the following output from WORKER task '{task.id}':\n\n"
                    f"{_condensed_output}\n\n"
                    f"Original task: {task.description}"
                ),
                prompt=(
                    f"Review the following output from WORKER task '{task.id}':\n\n"
                    f"{_condensed_output}\n\n"
                    f"Original task: {task.description}"
                ),
                context={
                    "review_type": "code_review",
                    "original_task_id": task.id,
                    "iteration": iteration,
                    "self_check_passed": current_result.metadata.get("self_check_passed"),
                    "self_check_issues": current_result.metadata.get("self_check_issues", []),
                    "schema_valid": current_result.metadata.get("schema_valid"),
                },
            )
            try:
                review_result = quality_agent.execute(review_task)
                review_verification = quality_agent.verify(review_result.output)

                if review_result.success and review_verification.passed:
                    logger.info(
                        "[AgentGraph] Maker-checker: INSPECTOR approved %s on iteration %s",
                        task.id,
                        iteration + 1,
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

                # Review failed — feed back to WORKER
                issues_text = "; ".join(
                    i.get("message", str(i)) if isinstance(i, dict) else str(i) for i in review_verification.issues
                )
                logger.warning(
                    "[AgentGraph] Maker-checker: INSPECTOR rejected %s (iteration %d): %s",
                    task.id,
                    iteration + 1,
                    issues_text,
                )

                if iteration < self._MAKER_CHECKER_MAX_ITERATIONS - 1:
                    # Condense Inspector feedback for Worker rework
                    if _mc_condenser is not None:
                        _rework_context = _mc_condenser.condense_for_handoff(
                            AgentType.INSPECTOR.value,
                            AgentType.WORKER.value,
                            review_result.output,
                            review_result.metadata,
                        )
                    else:
                        _rework_context = issues_text

                    # Re-run WORKER with feedback
                    fix_task = AgentTask(
                        task_id=f"{task.id}_fix_{iteration}",
                        agent_type=AgentType.WORKER,
                        description=(
                            f"{task.description}\n\n"
                            f"[MAKER-CHECKER FEEDBACK] Previous output was rejected "
                            f"by INSPECTOR review.\n{_rework_context}\n"
                            f"Please fix these issues."
                        ),
                        prompt=(
                            f"{task.description}\n\n"
                            f"[MAKER-CHECKER FEEDBACK] Previous output was rejected "
                            f"by INSPECTOR review.\n{_rework_context}\n"
                            f"Please fix these issues."
                        ),
                        context={
                            "original_output": str(current_result.output)[:TRUNCATE_OUTPUT_SUMMARY],
                            "review_issues": issues_text,
                            "iteration": iteration + 1,
                        },
                    )
                    current_result = builder_agent.execute(fix_task)
                    if not current_result.success:
                        break
            except Exception as e:
                logger.warning("[AgentGraph] Maker-checker iteration failed: %s", e)
                break

        # Exhausted iterations
        if current_result.metadata is None:
            current_result.metadata = {}
        current_result.metadata["maker_checker"] = {
            "approved": False,
            "iterations": self._MAKER_CHECKER_MAX_ITERATIONS,
        }
        return current_result

    # ------------------------------------------------------------------
    # Error recovery
    # ------------------------------------------------------------------

    def _run_error_recovery(
        self,
        task: Task,
        failed_result: AgentResult,
        verification: Any,
    ) -> AgentResult:
        """Delegate a failed task to the Worker for error analysis and recovery.

        If the Worker recovery also fails and FOREMAN is available, attempts
        re-decomposition of the task into smaller subtasks.

        Args:
            task: The task that failed verification.
            failed_result: The AgentResult from the failed execution attempt.
            verification: The verification result containing issue details.

        Returns:
            The recovery AgentResult, which may itself indicate failure
            if all recovery strategies are exhausted.
        """
        try:
            recovery_agent = self._agents[AgentType.WORKER]
            issues_text = "; ".join(
                i.get("message", str(i)) if isinstance(i, dict) else str(i) for i in verification.issues
            )
            recovery_task = AgentTask.from_task(
                task,
                f"Analyse and recover from failure in task '{task.id}': {issues_text}",
            )
            recovery_task.context["original_output"] = str(failed_result.output)[:TRUNCATE_OUTPUT_SUMMARY]
            recovery_task.context["verification_issues"] = issues_text
            recovery_result = recovery_agent.execute(recovery_task)

            # Surface verification_issues in the result so callers can see what
            # failed without inspecting recovery_task.context directly.
            if not recovery_result.success and issues_text:
                logger.warning(
                    "[AgentGraph] Error recovery failed for task %s — original verification issues: %s",
                    task.id,
                    issues_text[:200],
                )
                if issues_text not in " ".join(recovery_result.errors):
                    recovery_result.errors.append(f"[verification_issues] {issues_text}")

            # If recovery fails and FOREMAN is available, attempt re-decomposition
            if not recovery_result.success and AgentType.FOREMAN in self._agents:
                try:
                    planner = self._agents[AgentType.FOREMAN]
                    replan_task = AgentTask.from_task(
                        task,
                        f"Re-decompose failed task '{task.id}' into smaller subtasks. Original error: {issues_text}",
                    )
                    replan_task.context["mode"] = "extract"
                    replan_result = planner.execute(replan_task)
                    if replan_result.success:
                        logger.info("[AgentGraph] Planner re-decomposed failed task %s", task.id)
                        return replan_result
                except Exception as re_e:
                    logger.warning("[AgentGraph] Planner re-decomposition failed: %s", re_e)

            return recovery_result
        except Exception as e:
            logger.warning("[AgentGraph] Error recovery delegation failed: %s", e)
            return AgentResult(
                success=False,
                output=failed_result.output,
                errors=[f"Recovery failed: {e}"],
            )

    # ------------------------------------------------------------------
    # Capability-based delegation
    # ------------------------------------------------------------------

    def _find_delegate(self, task: Task, exclude: AgentType | None = None) -> AgentType | None:
        """Find the best available agent to handle a delegated task.

        Strategy:
        1. Query each registered agent's can_handle() method.
        2. Among willing agents, prefer the one whose capabilities best match
           the task description keywords.
        3. Fall back to FOREMAN (which can re-route) if nothing else fits.

        Args:
            task: The task needing a substitute handler.
            exclude: The agent type that just delegated (skip it).

        Returns:
            Best-matching AgentType, or None if no substitute found.
        """
        candidates = []
        task_lower = task.description.lower()

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
            except Exception:  # Broken agent implementation; skip it as a delegation candidate
                logger.warning(
                    "Agent %s raised during capability check for task %s",
                    agent_type,
                    task.id,
                    exc_info=True,
                )

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_type = candidates[0]

        # If no candidate has any keyword match, fall back to FOREMAN for re-routing
        if best_score == 0 and AgentType.FOREMAN in self._agents:
            return AgentType.FOREMAN

        return best_type

    # ------------------------------------------------------------------
    # Dynamic graph modification
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
        - Error recovery agent injects corrective tasks
        - FOREMAN splits a subtask mid-execution
        - FOREMAN adds a review task after a risky step

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
            logger.warning("[AgentGraph] inject_task: after_task %s not in plan", after_task_id)
            return False

        if new_task.id in plan.nodes:
            logger.warning("[AgentGraph] inject_task: task %s already exists", new_task.id)
            return False

        # Create the new node
        new_node = TaskNode(
            task=new_task,
            dependencies={after_task_id},
        )

        # Re-wire: tasks that depended on after_task_id now depend on new_task
        for _node_id, node in plan.nodes.items():
            if after_task_id in node.dependencies:
                node.dependencies.discard(after_task_id)
                node.dependencies.add(new_task.id)

        # Update dependents: after_node's old dependents move to new_node,
        # and after_node now points only to new_task.
        after_node = plan.nodes[after_task_id]
        old_dependents = after_node.dependents.copy()
        after_node.dependents = {new_task.id}

        # new_node inherits the old dependents (tasks that used to follow after_node)
        new_node.dependents = old_dependents

        plan.nodes[new_task.id] = new_node

        # Rebuild execution order
        plan.execution_order = self._topological_sort(plan.nodes)

        logger.info(
            "[AgentGraph] Injected task %s after %s in plan %s",
            new_task.id,
            after_task_id,
            plan_id,
        )
        return True
