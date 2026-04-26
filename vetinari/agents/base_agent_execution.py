"""Safe execution wrapper for BaseAgent.

Contains ``execute_safely`` extracted from ``base_agent.py`` to keep that
file under the 550-line limit.

This function is the hot-path template method that wraps every agent
execution call:

1. Validate task against agent type
2. Prepare task (token budget, constraints, trace span)
3. Inject prior memories into context
4. Enforce per-agent permission check
5. Enforce input guardrails (fail closed on error)
6. Call the agent's core execution function
7. Run self-check on the output
8. Record MetaAdapter outcome
9. Enforce output guardrails (fail closed on error)
10. Complete task (quality scoring, feedback, training data)
11. Record audit trail entry
12. Close OTel GenAI span

Pipeline role: Validate → Prepare → **Execute** → Guardrail → Complete.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from vetinari.agents.contracts import AgentResult, AgentTask

if TYPE_CHECKING:
    from vetinari.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


def execute_safely(agent: BaseAgent, task: AgentTask, execute_fn: Callable) -> AgentResult:
    """Template wrapper for safe agent execution with validation and error handling.

    Agents call this from their ``execute()`` method, passing their core logic
    as ``execute_fn``. All cross-cutting concerns (validation, guardrails,
    observability, learning) are handled here.

    Args:
        agent: The BaseAgent instance running the task.
        task: The task to execute.
        execute_fn: Callable accepting a prepared AgentTask and returning
            an AgentResult. This is the agent's unique execution logic.

    Returns:
        AgentResult with success/failure status, output, and metadata.
    """
    if not agent.validate_task(task):
        return AgentResult(
            success=False,
            output=None,
            errors=[f"Task validation failed for {agent.agent_type}"],
        )
    task = agent.prepare_task(task)

    # Record task start in the execution context audit trail
    try:
        from vetinari.execution_context import current_context

        current_context().record_operation(
            operation_name=f"agent_task_start:{agent.agent_type.value}",
            params={"task_id": task.task_id},
            result="started",
        )
    except Exception:
        logger.warning("Could not record task start in execution context")

    # Register agent with health monitor for heartbeat tracking
    _monitor_agent_id = None
    try:
        from vetinari.safety.agent_monitor import get_agent_monitor

        _monitor = get_agent_monitor()
        _monitor_agent_id = f"{agent.agent_type.value}:{task.task_id}"
        _monitor.register_agent(_monitor_agent_id, timeout_seconds=300, max_steps=100)
    except Exception:
        logger.warning("AgentMonitor unavailable — heartbeat tracking disabled for %s", agent.agent_type.value)

    # OTel GenAI agent-level span for observability
    _genai_span = None
    try:
        from vetinari.observability.otel_genai import get_genai_tracer

        _genai_tracer = get_genai_tracer()
        _genai_span = _genai_tracer.start_agent_span(
            agent_name=agent.name,
            operation="execute",
            model=getattr(agent, "default_model", ""),
        )
        _genai_span.attributes["agent_type"] = agent.agent_type.value
        _genai_span.attributes["mode"] = getattr(task, "mode", "default")
    except Exception:
        logger.warning("GenAI tracer unavailable for agent span")

    try:
        # Inject relevant memories into task context before execution
        prior_memories = agent._recall_relevant_memories(task.description or "")
        if prior_memories:
            ctx = getattr(task, "context", None) or {}
            ctx["prior_memories"] = prior_memories
            task.context = ctx
            agent._log("debug", "Injected %d prior memories into task context", len(prior_memories))

        # Per-agent permission check
        try:
            from vetinari.exceptions import SecurityError
            from vetinari.execution_context import ToolPermission, enforce_agent_permissions

            enforce_agent_permissions(agent.agent_type, ToolPermission.MODEL_INFERENCE)
        except (PermissionError, SecurityError) as perm_err:  # noqa: VET025 - execution path intentionally preserves captured failure context
            agent._log("warning", "Agent permission denied: %s", perm_err)
            return AgentResult(success=False, output=None, errors=[str(perm_err)])
        except Exception:
            logger.warning("Agent permission check not available in base_agent")

        # Sanitize task description before LLM prompt interpolation.
        # User-provided descriptions may contain prompt injection payloads aimed at
        # overriding agent instructions. Sanitize deterministically here so that every
        # agent's execute_fn receives a description already wrapped in untrusted-content
        # delimiters. Decision: ADR-0097 (prompt injection protection).
        try:
            from vetinari.safety.prompt_sanitizer import is_content_delimited, sanitize_task_description

            if task.description and not is_content_delimited(task.description):
                task.description = sanitize_task_description(task.description)
        except Exception:
            logger.warning(
                "Prompt sanitizer unavailable for task %s — description passed through unsanitized",
                task.task_id,
            )

        # Input guardrail enforcement — block before LLM call
        try:
            from vetinari.safety.guardrails import RailContext, get_guardrails

            _input_text = task.description or ""
            if _input_text:
                _input_check = get_guardrails().check_input(_input_text, context=RailContext.USER_FACING)
                if not _input_check.allowed:
                    _violations_str = "; ".join(v.description for v in _input_check.violations)
                    agent._log("warning", "Input guardrail blocked task: %s", _violations_str)
                    return AgentResult(
                        success=False,
                        output=None,
                        errors=[f"Input guardrail violation: {_violations_str}"],
                    )
        except (ImportError, ModuleNotFoundError):
            logger.warning("Guardrails module not available")
        except Exception as _gr_err:
            # Security checks must fail closed — block execution on unexpected errors
            logger.warning("Input guardrail check failed — blocking execution: %s", _gr_err)
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Input guardrail system error: {_gr_err}"],
            )

        # Expose task context to InferenceMixin
        _task_ctx = getattr(task, "context", None) or {}
        agent._current_task_memories = _task_ctx.get("prior_memories")
        agent._current_task_type = getattr(task, "task_type", None) or _task_ctx.get("type", "general")

        # Heartbeat before core execution — best-effort, must not block execution
        if _monitor_agent_id is not None:
            with contextlib.suppress(Exception):
                _monitor.heartbeat(_monitor_agent_id)

        # Acquire VRAM lease to protect the model from eviction during inference.
        # The model_id comes from the agent's default or task-assigned model.
        # Decision: ADR-0087 (VRAM-aware routing).
        _lease_holder_id: str | None = None
        _lease_model_id = getattr(agent, "default_model", "") or ""
        if _lease_model_id:
            try:
                from vetinari.models.vram_manager import get_vram_manager

                _lease_holder_id = f"{agent.agent_type.value}:{task.task_id}"
                if not get_vram_manager().acquire_lease(_lease_model_id, _lease_holder_id):
                    _lease_holder_id = None  # Lease denied — proceed without protection
            except Exception:
                _lease_holder_id = None
                logger.warning(
                    "VRAM lease unavailable — proceeding without eviction protection for task %s", task.task_id
                )

        result = execute_fn(task)

        # Run self_check on successful output
        if result.success and result.output:
            try:
                from vetinari.agents.skill_contract import SkillOutput, self_check

                _skill_output = SkillOutput(
                    agent_type=agent.agent_type.value,
                    task_summary=task.description or "",
                    verdict=SkillOutput.__dataclass_fields__["verdict"].default
                    if hasattr(SkillOutput, "__dataclass_fields__")
                    else None,
                    confidence=0.8,
                )
                if isinstance(result.output, dict):
                    _skill_output.task_summary = result.output.get("task_summary", task.description or "")
                    _skill_output.confidence = result.output.get("confidence", 0.8)
                _checked = self_check(_skill_output)
                result.metadata["self_check_passed"] = _checked.self_check_passed
                result.metadata["self_check_issues"] = _checked.self_check_issues
                if not _checked.self_check_passed:
                    agent._log(
                        "warning",
                        "Self-check FAILED with %d issue(s) — quality gate will apply stricter threshold: %s",
                        len(_checked.self_check_issues),
                        "; ".join(_checked.self_check_issues[:3]),
                    )
                    result.metadata["self_check_gate_hint"] = "stricter"
            except Exception as _sc_err:
                logger.warning("Self-check failed (non-fatal): %s", _sc_err)

        # Stamp inference metadata for downstream confidence routing
        _inf_meta = getattr(agent, "_last_inference_metadata", None)
        if _inf_meta:
            result.metadata["inference_metadata"] = _inf_meta

        # Record outcome in MetaAdapter for strategy learning
        if result.success:
            try:
                from vetinari.learning.meta_adapter import StrategyBundle, get_meta_adapter

                _mode = getattr(task, "mode", "default")
                get_meta_adapter().record_outcome(
                    task_description=task.description or "",
                    task_type=getattr(task, "task_type", "general"),
                    strategy_used=StrategyBundle(),
                    quality_score=result.metadata.get("quality_score", 0.7),
                    success=True,
                    mode=str(_mode) if _mode else "",
                )
            except Exception:
                logger.warning("MetaAdapter outcome recording unavailable")

        if result.success:
            # Hard-enforce output guardrails — block on violation for USER_FACING context
            try:
                import json as _json

                from vetinari.safety.guardrails import RailContext, get_guardrails

                _output_text = (
                    (result.output if isinstance(result.output, str) else _json.dumps(result.output, default=str))
                    if result.output
                    else ""
                )
                if _output_text:
                    _gr = get_guardrails().check_output(_output_text, context=RailContext.USER_FACING)
                    if not _gr.allowed:
                        _violations_str = "; ".join(v.description for v in _gr.violations)
                        agent._log("warning", "Output guardrail blocked: %s", _violations_str)
                        return AgentResult(
                            success=False,
                            output=None,
                            errors=[f"Output guardrail violation: {_violations_str}"],
                        )
            except (ImportError, ModuleNotFoundError):
                logger.warning("Guardrails module not available for output check")
            except Exception as _gr_err:
                # Security checks must fail closed — block on unexpected errors
                logger.warning("Output guardrail check failed — blocking output: %s", _gr_err)
                return AgentResult(
                    success=False,
                    output=None,
                    errors=[f"Output guardrail system error: {_gr_err}"],
                )
            agent.complete_task(task, result)

        # Record operation outcome in the execution context audit trail
        try:
            from vetinari.execution_context import current_context

            current_context().record_operation(
                operation_name=f"agent_execute:{agent.agent_type.value}",
                params={"task_id": task.task_id, "description": (task.description or "")[:200]},
                result="success" if result.success else "failed",
            )
        except Exception:
            logger.warning("Could not record operation in execution context")

        # Close the GenAI span on success
        if _genai_span is not None:
            try:
                from vetinari.observability.otel_genai import get_genai_tracer

                get_genai_tracer().end_agent_span(
                    _genai_span,
                    status="ok" if result.success else "error",
                )
            except Exception:
                logger.warning("Failed to close GenAI span on success")

        # Release VRAM lease now that inference is complete
        if _lease_holder_id is not None:
            with contextlib.suppress(Exception):
                from vetinari.models.vram_manager import get_vram_manager

                get_vram_manager().release_lease(_lease_holder_id)

        # Clean up per-task state from InferenceMixin wiring
        agent._current_task_memories = None
        agent._current_task_type = None

        return result

    except Exception as e:
        logger.error("[%s] Execute failed: %s", agent.agent_type, e)

        # Release VRAM lease on failure
        if _lease_holder_id is not None:
            with contextlib.suppress(Exception):
                from vetinari.models.vram_manager import get_vram_manager

                get_vram_manager().release_lease(_lease_holder_id)

        # Clean up per-task state from InferenceMixin wiring
        agent._current_task_memories = None
        agent._current_task_type = None

        # Close the GenAI span on error
        if _genai_span is not None:
            try:
                from vetinari.observability.otel_genai import get_genai_tracer

                get_genai_tracer().end_agent_span(_genai_span, status="error")
            except Exception:
                logger.warning("Failed to close GenAI span on error")

        return AgentResult(success=False, output=None, errors=[str(e)])
    finally:
        # Deregister from health monitor — best-effort, must not suppress any exception
        if _monitor_agent_id is not None:
            with contextlib.suppress(Exception):
                _monitor.deregister_agent(_monitor_agent_id)
