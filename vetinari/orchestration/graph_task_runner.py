"""Per-task execution logic for the AgentGraph.

Handles the execution of a single TaskNode including:
- WIP capacity checks
- Structured event emission (task_started / task_completed / task_failed)
- Resource and permission constraint enforcement
- Blackboard delegation for unregistered agent types
- Circuit breaking and cost prediction
- Policy and safety monitoring
- Context injection from dependency results
- Retry loop with self-correction feedback (see task_retry_loop.py)
- Goal adherence checking
- Explicit delegation and needs-info handling
- Output schema validation and maker-checker quality gate

Pre-execution setup lives here; the retry loop is in ``task_retry_loop.py``.
Both are mixed into ``GraphExecutorMixin`` via ``GraphTaskRunnerMixin``.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any

from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
)
from vetinari.constants import TRUNCATE_OUTPUT_PREVIEW
from vetinari.exceptions import SecurityError
from vetinari.orchestration.graph_types import TaskNode
from vetinari.orchestration.task_retry_loop import TaskRetryLoopMixin

logger = logging.getLogger(__name__)

# Suppress unused-import warning: TRUNCATE_OUTPUT_PREVIEW is used in task_retry_loop
# which inherits from this module's mixin — kept here for consistent import locality.
_ = TRUNCATE_OUTPUT_PREVIEW


class GraphTaskRunnerMixin(TaskRetryLoopMixin):
    """Per-task execution mixin for AgentGraph.

    Provides ``_execute_task_node``, which runs a single task through its
    assigned agent with the full suite of safety checks, retries, and
    self-correction feedback. Mixed into AgentGraph via GraphExecutorMixin.

    Attributes expected on ``self``:
        _agents (dict[AgentType, Any]): Registered agent instances.
        _wip_tracker: Optional WIP tracking object.
        _goal_tracker: Optional goal drift tracker.
        _quality_reviewed_agents (set): Agent types that trigger maker-checker.
    """

    def _execute_task_node(
        self,
        node: TaskNode,
        prior_results: dict[str, AgentResult] | None = None,
    ) -> AgentResult:
        """Execute a single task with retries and a self-correction loop.

        Self-correction: if the agent produces output but verification fails,
        the agent is called once more with the verification feedback injected
        into the task description before giving up.

        Enforces architecture constraints (delegation rules) and resource
        constraints (max_retries) before and during execution.

        Args:
            node: The TaskNode to execute, containing the Task and retry state.
            prior_results: Results from dependency tasks, used to inject context.

        Returns:
            AgentResult reflecting the final outcome of the task.
        """
        task = node.task
        agent_type = task.assigned_agent

        # WIP check: ensure agent has capacity (pull-based flow)
        if self._wip_tracker is not None:
            agent_type_str = agent_type.value if hasattr(agent_type, "value") else str(agent_type)
            if not self._wip_tracker.start_task(agent_type_str.upper(), str(task.id)):
                logger.info(
                    "[AgentGraph] WIP limit reached for %s — task %s queued",
                    agent_type_str,
                    task.id,
                )
                self._wip_tracker.enqueue(agent_type_str.upper(), str(task.id))
                # Return a pending result — task will be pulled when capacity frees
                return AgentResult(
                    success=False,
                    output=None,
                    errors=[f"WIP limit reached for {agent_type_str} — task queued"],
                )
            # Wire pool tracking: claim a slot in the per-agent-type pool so
            # get_pool_utilization() reflects live concurrency alongside start_task.
            with contextlib.suppress(Exception):
                self._wip_tracker.start_pool_task(agent_type_str.upper(), str(task.id))

        # Operator pause check: if the agent is paused via the control API,
        # return a non-success result so the task stays pending rather than
        # executing against the operator's intent.
        try:
            from vetinari.web.litestar_agents_api import get_agent_control_state

            _control = get_agent_control_state()
            _agent_type_str = agent_type.value if hasattr(agent_type, "value") else str(agent_type)
            if _agent_type_str in _control["paused"] or str(task.id) in _control["paused"]:
                _pause_reason = (
                    _control["paused"].get(_agent_type_str, {}).get("reason")
                    or _control["paused"].get(str(task.id), {}).get("reason")
                    or "paused by operator"
                )
                logger.info(
                    "[AgentGraph] Agent %s paused — task %s deferred: %s",
                    _agent_type_str,
                    task.id,
                    _pause_reason,
                )
                if self._wip_tracker is not None:
                    self._wip_tracker.release_task(_agent_type_str.upper(), str(task.id))
                    with contextlib.suppress(Exception):
                        self._wip_tracker.complete_pool_task(_agent_type_str.upper(), str(task.id))
                return AgentResult(
                    success=False,
                    output=None,
                    errors=[f"Agent {_agent_type_str} is paused: {_pause_reason}"],
                )
        except ImportError:
            logger.debug("Agent control API not available — skipping pause check")

        # Structured log: task_started event
        _task_node_start_ms = time.time() * 1000
        try:
            from vetinari.structured_logging import log_event as _sl_log_event

            _sl_log_event(
                "info",
                "vetinari.orchestration.agent_graph",
                "task_started",
                task_id=str(task.id),
                agent_type=str(agent_type),
                status="running",
            )
        except Exception:  # Broad: best-effort event emission; never blocks core logic
            logger.warning(
                "Failed to emit task_started structured event for %s",
                task.id,
                exc_info=True,
            )

        def _emit_task_done(final_result: AgentResult) -> AgentResult:
            """Release WIP slot, update bottleneck metrics, and emit structured event."""
            # Release WIP slot on task completion
            if self._wip_tracker is not None:
                _at_str = agent_type.value if hasattr(agent_type, "value") else str(agent_type)
                self._wip_tracker.complete_task(_at_str.upper(), str(task.id))
                # Wire pool tracking: release the per-agent-type pool slot
                with contextlib.suppress(Exception):
                    self._wip_tracker.complete_pool_task(_at_str.upper(), str(task.id))
            # Wire andon scope resume: if this agent's scope was paused due to stagnation,
            # a successful completion means the scope has recovered — resume it.
            if final_result.success:
                try:
                    from vetinari.workflow.andon import get_andon_system as _get_andon

                    _andon = _get_andon()
                    _scope_str = agent_type.value if hasattr(agent_type, "value") else str(agent_type)
                    if _andon.is_scope_paused(_scope_str):
                        _andon.resume_scope(_scope_str)
                        logger.info(
                            "[AgentGraph] Andon scope %r resumed after successful task %s",
                            _scope_str,
                            task.id,
                        )
                except Exception as _exc:
                    logger.warning(
                        "Andon scope resume skipped for scope %r after task %s — andon unavailable: %s",
                        _scope_str,
                        task.id,
                        _exc,
                    )
            _success = final_result.success
            _duration_ms = round(time.time() * 1000 - _task_node_start_ms, 2)

            # Bottleneck metrics: update per-agent throughput/utilization tracking
            try:
                from vetinari.orchestration.bottleneck import get_bottleneck_identifier

                _bottleneck = get_bottleneck_identifier()
                _agent_str = agent_type.value if hasattr(agent_type, "value") else str(agent_type)
                _bottleneck.update_metrics(_agent_str, _duration_ms, _success)
            except Exception:
                logger.warning(
                    "Bottleneck metric update failed for %s — skipping, bottleneck detection may be inaccurate",
                    agent_type,
                )
            _sl_event = "task_completed" if _success else "task_failed"
            try:
                from vetinari.structured_logging import log_event as _sl_done

                _sl_done(
                    "info" if _success else "warning",
                    "vetinari.orchestration.agent_graph",
                    _sl_event,
                    task_id=str(task.id),
                    agent_type=agent_type.value if hasattr(agent_type, "value") else str(agent_type),
                    duration_ms=_duration_ms,
                    status="completed" if _success else "failed",
                )
            except Exception:  # Broad: best-effort event emission; never blocks core logic
                logger.warning(
                    "Failed to emit %s structured event for %s",
                    _sl_event,
                    task.id,
                    exc_info=True,
                )
            return final_result

        # Resource constraint enforcement
        try:
            from vetinari.constraints.registry import get_constraint_registry

            _reg = get_constraint_registry()
            _ac = _reg.get_constraints_for_agent(agent_type.value if hasattr(agent_type, "value") else str(agent_type))
            if _ac and _ac.resources:
                # Cap retries to the resource-constrained maximum
                constrained_retries = min(node.max_retries, _ac.resources.max_retries)
                if constrained_retries < node.max_retries:
                    logger.debug(
                        "[AgentGraph] Capping retries for %s from %d to %d (constraint)",
                        agent_type,
                        node.max_retries,
                        constrained_retries,
                    )
                    node.max_retries = constrained_retries
        except Exception:  # Broad: optional feature; any failure must not block task execution
            logger.warning(
                "Failed to apply agent constraints for %s",
                agent_type,
                exc_info=True,
            )

        # Mode and task-type constraint validation
        try:
            from vetinari.constraints.registry import get_constraint_registry

            _creg = get_constraint_registry()
            agent_str = agent_type.value if hasattr(agent_type, "value") else str(agent_type)
            _task_meta = task.metadata if hasattr(task, "metadata") and task.metadata else {}
            _mode = _task_meta.get("mode") if isinstance(_task_meta, dict) else None
            _task_type_meta = _task_meta.get("task_type") if isinstance(_task_meta, dict) else None
            if _mode:
                _mode_valid, _mode_reason = _creg.validate_mode(agent_str, _mode)
                if not _mode_valid:
                    logger.warning(
                        "[AgentGraph] Mode constraint violation for %s on task %s: mode=%r — %s",
                        agent_type,
                        task.id,
                        _mode,
                        _mode_reason,
                    )
            if _task_type_meta:
                _type_valid, _type_reason = _creg.validate_task_type(agent_str, _task_type_meta)
                if not _type_valid:
                    logger.warning(
                        "[AgentGraph] Task-type constraint violation for %s on task %s: task_type=%r — %s",
                        agent_type,
                        task.id,
                        _task_type_meta,
                        _type_reason,
                    )
        except Exception:  # Broad: optional feature; any failure must not block task execution
            logger.warning(
                "Constraint mode/task-type validation unavailable for %s — task will run without constraint checks",
                agent_type,
            )

        if agent_type not in self._agents:
            # Try the blackboard delegation path
            from vetinari.memory.blackboard import get_blackboard

            board = get_blackboard()
            _delegated = board.delegate(task, self._agents) or AgentResult(
                success=False,
                output=None,
                errors=[f"No agent registered for type: {agent_type}"],
            )
            return _emit_task_done(_delegated)

        agent = self._agents[agent_type]

        # Per-agent permission enforcement
        try:
            from vetinari.execution_context import ToolPermission as _TP
            from vetinari.execution_context import enforce_agent_permissions

            enforce_agent_permissions(agent_type, _TP.MODEL_INFERENCE)
        except (PermissionError, SecurityError) as perm_err:
            logger.warning("[AgentGraph] Agent permission denied: %s", perm_err)
            return _emit_task_done(
                AgentResult(
                    success=False,
                    output=None,
                    errors=[str(perm_err)],
                ),
            )
        except Exception:  # Broad: optional feature; any failure must not block task execution
            logger.warning("Agent permission check not configured, allowing execution")

        # Permission enforcement before execution
        try:
            from vetinari.execution_context import ToolPermission, get_context_manager

            ctx_mgr = get_context_manager()
            ctx_mgr.enforce_permission(
                ToolPermission.MODEL_INFERENCE,
                f"agent_execute:{agent_type.value}",
            )
        except PermissionError:
            logger.warning(
                "[AgentGraph] Permission denied for %s — MODEL_INFERENCE not allowed in current execution mode",
                agent_type.value,
            )
            return _emit_task_done(
                AgentResult(
                    success=False,
                    output=None,
                    errors=[f"Permission denied: MODEL_INFERENCE required for {agent_type.value}"],
                ),
            )
        except Exception:  # Broad: optional feature; any failure must not block task execution
            logger.warning("Context manager not configured, allowing execution")

        # Inject prior results as context if the task depends on them
        # ACON-style: condense prior agent output before passing to next agent
        context: dict[str, Any] = dict(task.context if hasattr(task, "context") else {})
        if prior_results and task.dependencies:
            try:
                from vetinari.context import get_context_condenser

                _condenser = get_context_condenser()
                dep_summaries = {}
                for dep_id in task.dependencies:
                    if dep_id not in prior_results:
                        continue
                    dep_result = prior_results[dep_id]
                    # Determine source agent type from the dependency's node
                    _src_agent = "UNKNOWN"
                    if hasattr(self, "_execution_plan") and self._execution_plan:
                        _dep_node = self._execution_plan.nodes.get(dep_id)
                        if _dep_node:
                            _src_agent = _dep_node.agent_type.value
                    dep_summaries[dep_id] = {
                        "success": dep_result.success,
                        "output_summary": _condenser.condense_for_handoff(
                            _src_agent,
                            agent_type.value,
                            dep_result.output,
                            dep_result.metadata,
                        ),
                    }
            except Exception:  # Broad: optional feature; any failure must not block task execution
                logger.warning("[AgentGraph] Context condenser unavailable, using raw summaries")
                dep_summaries = {
                    dep_id: {
                        "success": prior_results[dep_id].success,
                        "output_summary": str(prior_results[dep_id].output)[:500],
                    }
                    for dep_id in task.dependencies
                    if dep_id in prior_results
                }
            context["dependency_results"] = dep_summaries

        # Build TaskContextManifest and prepend structured context to the prompt
        _manifest_prefix = ""
        try:
            from vetinari.orchestration.task_context import TaskManifestContext

            _dep_results_dict: dict[str, Any] = {}
            if prior_results:
                _dep_results_dict = {tid: str(r.output)[:500] if r.output else "" for tid, r in prior_results.items()}
            _manifest = TaskManifestContext.build_for_task(
                task_id=str(task.id),
                task_description=task.description or "",
                goal=context.get("goal", ""),
                completed_results=_dep_results_dict,
                dependency_ids=list(task.dependencies) if task.dependencies else [],
                constraints=context.get("constraints", {}),
            )
            _manifest_prefix = _manifest.format_for_prompt()
        except Exception:  # Broad: optional feature; never blocks core logic
            logger.warning("TaskContextManifest unavailable for task %s", task.id)

        _task_prompt = f"{_manifest_prefix}\n{task.description}" if _manifest_prefix else task.description
        agent_task = AgentTask.from_task(task, _task_prompt)
        if hasattr(agent_task, "context"):
            agent_task.context.update(context)

        # Let agents incorporate dependency results
        if hasattr(agent, "_incorporate_prior_results"):
            try:
                dep_results = agent._incorporate_prior_results(agent_task)
                if dep_results:
                    agent_task.context["incorporated_results"] = dep_results
                    logger.debug(
                        "[AgentGraph] Agent %s incorporated %d dependency result(s): %s",
                        agent_type.value,
                        len(dep_results),
                        ", ".join(dep_results.keys()),
                    )
            except Exception as e:
                logger.warning("[AgentGraph] _incorporate_prior_results failed: %s", e)

        # Circuit breaker pre-check — uses resilience.circuit_breaker
        _cb = None
        try:
            from vetinari.resilience import CircuitBreaker

            _cb_key = f"agent_{agent_type.value}"
            if not hasattr(self, "_circuit_breakers"):
                self._circuit_breakers: dict[str, CircuitBreaker] = {}
            if _cb_key not in self._circuit_breakers:
                self._circuit_breakers[_cb_key] = CircuitBreaker(_cb_key)
            _cb = self._circuit_breakers[_cb_key]
            if not _cb.allow_request():
                logger.warning(
                    "[AgentGraph] Circuit breaker OPEN for %s, skipping task %s",
                    agent_type.value,
                    task.id,
                )
                return _emit_task_done(
                    AgentResult(
                        success=False,
                        output=None,
                        errors=[f"Circuit breaker open for {agent_type.value}"],
                    ),
                )
        except ImportError:  # noqa: VET022 — vetinari.resilience is optional; skip circuit-breaking when absent
            pass

        # Per-agent circuit breaker — trips on consecutive task failures for
        # a specific agent slot, quarantining stuck agents independently of
        # the backend-level resilience breaker above.
        _agent_cb = None
        try:
            from vetinari.agents.agent_circuit_breaker import AgentCircuitBreaker

            _acb_key = f"agent_task_{agent_type.value}"
            if not hasattr(self, "_agent_circuit_breakers"):
                self._agent_circuit_breakers: dict[str, AgentCircuitBreaker] = {}
            if _acb_key not in self._agent_circuit_breakers:
                self._agent_circuit_breakers[_acb_key] = AgentCircuitBreaker(_acb_key)
            _agent_cb = self._agent_circuit_breakers[_acb_key]
            if not _agent_cb.allow_request():
                logger.warning(
                    "[AgentGraph] Agent circuit breaker OPEN for %s — task %s bypassed",
                    agent_type.value,
                    task.id,
                )
                return _emit_task_done(
                    AgentResult(
                        success=False,
                        output=None,
                        errors=[f"Agent circuit breaker open for {agent_type.value} — too many consecutive failures"],
                    ),
                )
        except ImportError:  # noqa: VET022 — vetinari.agents.agent_circuit_breaker is optional
            pass

        # Cost prediction before execution
        try:
            from vetinari.analytics.cost_predictor import CostPredictor

            _predictor = CostPredictor()
            _agent_str = agent_type.value if hasattr(agent_type, "value") else str(agent_type)
            _cost_est = _predictor.predict(
                task_type=_agent_str,
                complexity=3,  # Default medium complexity
                scope_size=1,
            )
            logger.debug(
                "[AgentGraph] Cost estimate for %s: %d tokens, %.3fs, $%.4f (confidence=%.2f)",
                task.id,
                _cost_est.tokens,
                _cost_est.latency_seconds,
                _cost_est.cost_usd,
                _cost_est.confidence,
            )
        except Exception:  # Broad: optional feature; any failure must not block task execution
            logger.warning("Cost prediction unavailable for %s", task.id)

        # Safety: policy enforcement check before execution
        try:
            from vetinari.safety.policy_enforcer import get_policy_enforcer

            _enforcer = get_policy_enforcer()
            _policy_decision = _enforcer.check_action(
                agent_type=agent_type.value,
                action="execute_task",
                context={"task_id": task.id, "task_type": task.task_type},
            )
            # PolicyDecision is a dataclass — use .allowed attribute, not dict access
            if not _policy_decision.allowed:
                logger.warning(
                    "[AgentGraph] Policy enforcer blocked task %s: %s",
                    task.id,
                    _policy_decision.reason,
                )
                return _emit_task_done(
                    AgentResult(
                        success=False,
                        output=None,
                        errors=[f"Policy blocked: {_policy_decision.reason}"],
                    )
                )
        except Exception:
            logger.warning("PolicyEnforcer unavailable for pre-execution check")

        # Safety monitoring: register agent and send heartbeats
        _monitor = None
        try:
            from vetinari.safety.agent_monitor import get_agent_monitor

            _monitor = get_agent_monitor()
            _monitor.register_agent(f"{agent_type.value}:{task.id}", timeout_seconds=300, max_steps=50)
        except Exception:
            logger.warning("AgentMonitor unavailable — executing without safety monitoring")

        # Hand off to the retry loop (TaskRetryLoopMixin)
        return self._run_task_attempt_loop(
            node=node,
            agent=agent,
            agent_type=agent_type,
            agent_task=agent_task,
            _monitor=_monitor,
            _cb=_cb,
            _agent_cb=_agent_cb,
            _emit_task_done=_emit_task_done,
        )
