"""Retry loop and post-execution logic for per-task graph execution.

Contains the attempt loop that ``GraphTaskRunnerMixin._execute_task_node``
runs after all pre-execution checks pass.  Separated from
``graph_task_runner.py`` to keep both files within the 550-line limit.

Pipeline role: Plan -> Graph -> **TaskRetryLoop** (attempt, verify, delegate) -> Result.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from vetinari.agents.contracts import AgentResult, AgentTask
from vetinari.constants import TRUNCATE_OUTPUT_PREVIEW
from vetinari.types import AgentType

logger = logging.getLogger(__name__)


class TaskRetryLoopMixin:
    """Retry loop and post-verification logic for task execution.

    Mixin that provides ``_run_task_attempt_loop``, called by
    ``GraphTaskRunnerMixin._execute_task_node`` once all pre-execution
    checks have passed.  Assumes the same ``self`` attributes as
    ``GraphTaskRunnerMixin`` (``_agents``, ``_goal_tracker``,
    ``_quality_reviewed_agents``, ``_stagnation_detector``).
    """

    def _run_task_attempt_loop(
        self,
        node: Any,
        agent: Any,
        agent_type: AgentType,
        agent_task: AgentTask,
        _monitor: Any,
        _cb: Any,
        _agent_cb: Any,
        _emit_task_done: Any,
    ) -> AgentResult:
        """Execute the retry loop for a single task attempt.

        Runs the agent up to ``node.max_retries + 1`` times, injecting
        self-correction feedback on verification failure and delegating to
        the Worker for error recovery on final failure.

        Args:
            node: The TaskNode with retry state.
            agent: The agent instance to execute.
            agent_type: The AgentType enum value for this agent.
            agent_task: The prepared AgentTask to pass to the agent.
            _monitor: Optional AgentMonitor instance (may be None).
            _cb: Optional backend-level CircuitBreaker instance.
            _agent_cb: Optional agent-level AgentCircuitBreaker instance.
            _emit_task_done: Callable that releases WIP slots and emits the
                final structured event for the task.

        Returns:
            The final AgentResult for the task.
        """
        task = node.task

        for attempt in range(node.max_retries + 1):
            try:
                if _monitor:
                    with contextlib.suppress(Exception):
                        _monitor.heartbeat(f"{agent_type.value}:{task.id}")

                # Budget check: stop execution if the agent's budget is exhausted.
                # Guard: only use BudgetTracker instances (not MagicMock in tests).
                _agent_budget = getattr(agent, "_budget", None)
                if (
                    _agent_budget is not None
                    and type(_agent_budget).__name__ == "BudgetTracker"
                    and _agent_budget.is_exhausted()
                ):
                    _snap = agent._budget.snapshot()
                    logger.warning(
                        "[AgentGraph] Budget exhausted for %s on task %s — "
                        "tokens=%d/%d, iterations=%d/%d, cost=$%.4f/$%.2f",
                        agent_type.value,
                        task.id,
                        _snap.tokens_used,
                        _snap.token_budget,
                        _snap.iterations_used,
                        _snap.iteration_cap,
                        _snap.cost_used_usd,
                        _snap.cost_budget_usd,
                    )
                    return _emit_task_done(
                        AgentResult(
                            success=False,
                            output=None,
                            errors=[
                                f"Budget exhausted for {agent_type.value}: "
                                f"tokens={_snap.tokens_used}/{_snap.token_budget}, "
                                f"iterations={_snap.iterations_used}/{_snap.iteration_cap}"
                            ],
                        ),
                    )

                logger.info(
                    "[AgentGraph] Executing %s with %s (attempt %d/%d)",
                    task.id,
                    agent_type.value,
                    attempt + 1,
                    node.max_retries + 1,
                )
                # Switch to EXECUTION mode so tool permission checks pass
                try:
                    from vetinari.execution_context import get_context_manager as _get_ctx
                    from vetinari.types import ExecutionMode as _ExecMode

                    _ctx_mgr = _get_ctx()
                    _exec_ctx = _ctx_mgr.temporary_mode(_ExecMode.EXECUTION, task_id=task.id)
                    _exec_ctx.__enter__()
                except Exception:  # Broad: optional feature; any failure must not block task execution
                    _exec_ctx = None  # Context manager unavailable — degrade gracefully

                # Enforcement framework: validate constraints before execution
                try:
                    from vetinari.enforcement import enforce_all

                    enforce_all(
                        agent_type=agent_type,
                        current_depth=getattr(node, "delegation_depth", None),
                    )
                except ImportError:
                    logger.debug("Enforcement module not available, skipping pre-execution check")
                except Exception as enf_exc:
                    logger.warning(
                        "[AgentGraph] Enforcement blocked task %s: %s",
                        task.id,
                        enf_exc,
                    )
                    return _emit_task_done(
                        AgentResult(
                            success=False,
                            output=None,
                            errors=[f"Enforcement blocked: {enf_exc}"],
                        )
                    )

                try:
                    result = agent.execute(agent_task)
                finally:
                    if _exec_ctx is not None:
                        with contextlib.suppress(Exception):
                            _exec_ctx.__exit__(None, None, None)

                # Budget tracking: record iteration after each attempt
                _iter_budget = getattr(agent, "_budget", None)
                if _iter_budget is not None and type(_iter_budget).__name__ == "BudgetTracker":
                    _iter_budget.record_iteration()

                # Stagnation detection: record output to detect repeated/stuck runs
                if hasattr(self, "_stagnation_detector") and self._stagnation_detector is not None:
                    _output_str = str(result.output)[:200] if result.output else ""
                    self._stagnation_detector.record_output(_output_str)
                    if not result.success:
                        self._stagnation_detector.record_error()
                    if self._stagnation_detector.is_stagnant():
                        _reasons = self._stagnation_detector.stagnation_reasons()
                        logger.warning(
                            "[AgentGraph] Stagnation detected for task %s: %s",
                            task.id,
                            "; ".join(_reasons),
                        )
                        return _emit_task_done(
                            AgentResult(
                                success=False,
                                output=result.output,
                                errors=[f"Stagnation detected: {'; '.join(_reasons)}"],
                            ),
                        )

                    # Wire scoped stagnation: track per-agent-type independently so
                    # one stuck agent type doesn't mask healthy agents in other scopes.
                    _scope_str = agent_type.value if hasattr(agent_type, "value") else str(agent_type)
                    _scope_stagnant = self._stagnation_detector.detect_scoped(
                        _scope_str,
                        _output_str,
                        error=not result.success,
                    )
                    if _scope_stagnant:
                        # Pause this agent's scope via the Andon system so the
                        # pipeline knows to stop dispatching to this scope.
                        try:
                            from vetinari.workflow.andon import AndonSignal
                            from vetinari.workflow.andon import get_andon_system as _get_andon

                            _andon = _get_andon()
                            if not _andon.is_scope_paused(_scope_str):
                                _scope_signal = AndonSignal(
                                    source="stagnation_detector",
                                    severity="warning",
                                    message=f"Scoped stagnation in {_scope_str} for task {task.id}",
                                    affected_tasks=[str(task.id)],
                                    scope=_scope_str,
                                )
                                _andon.pause_scope(_scope_str, _scope_signal)
                        except Exception as _exc:
                            logger.warning(
                                "Andon scope pause skipped for scope %r after stagnation on task %s — andon unavailable: %s",
                                _scope_str,
                                task.id,
                                _exc,
                            )

                # Agent monitor: record step after each execution attempt
                if _monitor:
                    with contextlib.suppress(Exception):
                        _monitor.record_step(f"{agent_type.value}:{task.id}")

                # Circuit breaker: record outcome
                if _cb is not None:
                    if result.success:
                        _cb.record_success()
                    else:
                        _cb.record_failure()

                # Agent circuit breaker: record outcome
                if _agent_cb is not None:
                    if result.success:
                        _agent_cb.record_success()
                    else:
                        _agent_cb.record_failure()

                # Inter-agent guardrail check
                if result.success and result.output:
                    try:
                        from vetinari.safety.guardrails import RailContext, get_guardrails

                        _out_text = str(result.output)[:TRUNCATE_OUTPUT_PREVIEW]
                        # Use CODE_EXECUTION for inter-agent traffic to avoid false positives
                        # on code examples while still checking at trust boundaries.
                        _guard_result = get_guardrails().check_output(_out_text, context=RailContext.CODE_EXECUTION)
                        if not _guard_result.allowed:
                            _violations = (
                                "; ".join(str(v) for v in _guard_result.violations)
                                if _guard_result.violations
                                else "policy violation"
                            )
                            logger.warning(
                                "[AgentGraph] Inter-agent guardrail BLOCKED output from %s on task %s: %s",
                                agent_type.value,
                                task.id,
                                _violations,
                            )
                            result = AgentResult(
                                success=False,
                                output=None,
                                errors=[f"Guardrail blocked: {_violations}"],
                            )
                    except ImportError:  # noqa: VET022 — vetinari.safety is optional; skip guardrail check when absent
                        pass

                # Check goal adherence
                if self._goal_tracker and result.success:
                    try:
                        output_str = str(result.output)[:500] if result.output else ""
                        adherence = self._goal_tracker.check_adherence(output_str, task.description or "")
                        if adherence.score < 0.4:
                            logger.warning(
                                "[AgentGraph] Goal drift in %s: score=%.2f — %s",
                                task.id,
                                adherence.score,
                                adherence.deviation_description,
                            )
                            result.metadata["drift_warning"] = adherence.to_dict()
                    except Exception:  # Broad: best-effort event emission; never blocks core logic
                        logger.warning(
                            "Goal adherence check failed for task %s",
                            task.id,
                            exc_info=True,
                        )

                # Handle explicit delegation: agent says "not my domain"
                if result.metadata.get("delegation_requested"):
                    reason = result.metadata.get("delegation_reason", "no reason given")
                    logger.info(
                        "[AgentGraph] %s delegated task '%s': %s — finding substitute",
                        agent_type.value,
                        task.id,
                        reason,
                    )
                    delegate_type = self._find_delegate(task, exclude=agent_type)
                    if delegate_type and delegate_type in self._agents:
                        delegate_agent = self._agents[delegate_type]
                        result = delegate_agent.execute(agent_task)
                    else:
                        return _emit_task_done(
                            AgentResult(
                                success=False,
                                output=None,
                                errors=[f"Task delegated by {agent_type.value} but no substitute found: {reason}"],
                            ),
                        )

                # Handle needs_info: agent needs more information (mid-task andon)
                if result.metadata.get("needs_info"):
                    _delegate_to_str = result.metadata.get("delegate_to")
                    _question = result.metadata.get("question", "")
                    if _delegate_to_str and _delegate_to_str in {at.value for at in self._agents}:
                        # Delegate question to another agent
                        _delegate_type = AgentType(_delegate_to_str)
                        _delegate_agent = self._agents[_delegate_type]
                        logger.info(
                            "[AgentGraph] %s needs info from %s: %s",
                            agent_type.value,
                            _delegate_to_str,
                            _question[:100],
                        )
                        _info_task = AgentTask(
                            task_id=f"{task.id}_info_{attempt}",
                            agent_type=_delegate_type,
                            description=_question,
                            prompt=_question,
                            context={
                                "original_task": task.id,
                                "requesting_agent": agent_type.value,
                            },
                        )
                        _info_result = _delegate_agent.execute(_info_task)
                        if _info_result.success:
                            # Re-execute original agent with answer injected
                            agent_task.context["info_response"] = str(_info_result.output)[:TRUNCATE_OUTPUT_PREVIEW]
                            agent_task.description = (
                                f"{task.description}\n\n"
                                f"[INFO RESPONSE] {str(_info_result.output)[:TRUNCATE_OUTPUT_PREVIEW]}"
                            )
                            result = agent.execute(agent_task)
                        # If delegation failed, fall through to verification
                    elif result.metadata.get("needs_user_input"):
                        # Signal pause — return result as-is for orchestrator to surface
                        logger.info(
                            "[AgentGraph] %s needs user input: %s",
                            agent_type.value,
                            _question[:100],
                        )
                        return _emit_task_done(result)

                verification = agent.verify(result.output)

                if result.success and verification.passed:
                    # Validate output schema and store result in metadata
                    schema_issues = self._validate_output_schema(agent_type, result.output)
                    result.metadata["schema_valid"] = len(schema_issues) == 0
                    if schema_issues:
                        result.metadata["schema_issues"] = schema_issues
                        logger.info(
                            "[AgentGraph] %s output schema deviations: %s",
                            task.id,
                            "; ".join(schema_issues),
                        )

                    # Maker-checker for quality-reviewed agents (configurable)
                    if agent_type in self._quality_reviewed_agents and AgentType.INSPECTOR in self._agents:
                        result = self._apply_maker_checker(task, result)

                    # Config self-tuning: record task completion for threshold tracking
                    try:
                        from vetinari.config.self_tuning import get_config_self_tuner

                        _task_type = getattr(task, "type", None) or node.task_type or "general"
                        get_config_self_tuner().record_task_completion(_task_type)
                    except Exception:
                        logger.warning("Config self-tuner unavailable for task %s", task.id, exc_info=True)

                    return _emit_task_done(result)

                if not verification.passed and attempt < node.max_retries:
                    # Self-correction: inject verification feedback and retry
                    issues_text = "; ".join(
                        i.get("message", str(i)) if isinstance(i, dict) else str(i) for i in verification.issues
                    )

                    # Retry intelligence: consult failure registry before blind retry
                    _fix_hint = ""
                    try:
                        from vetinari.resilience.retry_intelligence import get_retry_analyzer

                        _retry_strategy = get_retry_analyzer().analyze(
                            failure_trace=issues_text,
                            error_msg=issues_text,
                            task_type=getattr(task, "type", ""),
                        )
                        if _retry_strategy.known and _retry_strategy.confidence >= 0.7:
                            _fix_hint = f"\n[KNOWN FIX] {_retry_strategy.fix_action}"
                            logger.info(
                                "[AgentGraph] Retry intelligence: known fix for %s (confidence=%.2f)",
                                task.id,
                                _retry_strategy.confidence,
                            )
                    except Exception:
                        logger.warning("Retry intelligence unavailable for task %s", task.id, exc_info=True)

                    logger.warning(
                        "[AgentGraph] %s verification failed: %s — injecting feedback and retrying",
                        task.id,
                        issues_text,
                    )
                    agent_task.description = (
                        f"{task.description}\n\n"
                        f"[SELF-CORRECTION] Previous attempt failed verification. "
                        f"Issues: {issues_text}. Please fix these issues.{_fix_hint}"
                    )
                    node.retries += 1
                    continue

                # Last attempt failed — try Worker error recovery if available
                if AgentType.WORKER in self._agents and attempt >= node.max_retries:
                    return _emit_task_done(self._run_error_recovery(task, result, verification))

                return _emit_task_done(
                    AgentResult(
                        success=False,
                        output=result.output,
                        errors=[
                            f"Verification failed after {attempt + 1} attempts: "
                            + "; ".join(
                                i.get("message", str(i)) if isinstance(i, dict) else str(i) for i in verification.issues
                            ),
                        ],
                    ),
                )

            except Exception as e:
                logger.error("[AgentGraph] %s raised exception: %s", task.id, e)
                if attempt < node.max_retries:
                    continue
                return _emit_task_done(AgentResult(success=False, output=None, errors=[str(e)]))

        return _emit_task_done(
            AgentResult(
                success=False,
                output=None,
                errors=["Task failed after all retries"],
            ),
        )
