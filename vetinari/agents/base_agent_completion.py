"""Task completion logic for BaseAgent.

Contains the ``complete_task`` function extracted from ``base_agent.py``
to keep that file under the 550-line limit.  The function runs six
post-execution subsystems in order:

1. Quality scoring (heuristic + LLM scorer)
2. Feedback loop (Thompson Sampling model selection signal)
3. Quality gate + escalation (block/retry on low scores)
4. Prompt evolver (variant result recording)
5. Training data collection (triggers DataCurator at N-record intervals)
6. Episodic + unified memory (stores execution summaries for future recall)

Pipeline role: Execute → **Completion** → Verify → Learn.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from vetinari.agents.contracts import AgentResult, AgentTask
from vetinari.constants import TRUNCATE_OUTPUT_SUMMARY

if TYPE_CHECKING:
    from vetinari.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


def complete_task(agent: BaseAgent, task: AgentTask, result: AgentResult) -> AgentTask:
    """Run all post-execution subsystems and mark the task complete.

    Called by ``BaseAgent._execute_safely`` after a successful execution.
    Stamps the task with a completion timestamp, runs quality scoring and
    feedback recording, enforces quality gates, and stores the execution in
    training data and episodic memory.

    Args:
        agent: The BaseAgent instance whose subsystems should be used.
        task: The completed task; mutated in-place with timestamps and scores.
        result: The AgentResult produced by the execution function.

    Returns:
        The mutated task with completion metadata populated.
    """
    task.completed_at = datetime.now(timezone.utc).isoformat()
    task.result = result.output
    if not result.success:
        task.error = "; ".join(result.errors)

    # Emit structured trace span for completion
    try:
        from vetinari.structured_logging import log_event

        log_event(
            "info",
            f"agent.{agent.agent_type.value}",
            "task_completed",
            task_id=task.task_id,
            success=result.success,
            agent=agent.agent_type.value,
        )
    except Exception:
        logger.warning("Failed to emit structured trace span for task_completed", exc_info=True)

    # ----- Quality gate enforcement -----
    if result.success and result.output:
        from vetinari.agents.base_agent import _get_agent_constraints

        constraints = _get_agent_constraints(agent.agent_type.value)
        if constraints and constraints.quality_gate:
            qg = constraints.quality_gate
            if not hasattr(task, "_quality_gate"):
                task._quality_gate = qg

    if not (result.success and result.output):
        # Emit a receipt even for failed/empty completions — every Task
        # has a one-to-one receipt regardless of which subsystems run.
        _emit_completion_receipt(agent, task, result, score=0.0, scoring_available=False)
        return task

    # Shared values used across subsystems
    output_str = ""
    model_id = "default"
    task_type = getattr(agent, "_current_task_type", None) or agent.agent_type.value.lower()
    score = None

    # -- Subsystem 1: Quality scoring --
    try:
        import json as _json

        output_str = (
            result.output
            if isinstance(result.output, str)
            else _json.dumps(result.output, default=str)[:TRUNCATE_OUTPUT_SUMMARY]
        )
        model_id = agent._last_inference_model_id or agent.default_model or ""
        if not model_id and agent._adapter_manager:
            try:
                _loaded = getattr(agent._adapter_manager, "_loaded_models", {})
                if _loaded:
                    model_id = next(iter(_loaded))
            except Exception:
                logger.warning("Could not resolve model_id from adapter manager")
        model_id = model_id or "default"

        from vetinari.learning.quality_scorer import get_quality_scorer

        scorer = get_quality_scorer()
        scorer._adapter_manager = agent._adapter_manager
        score = scorer.score(
            task_id=task.task_id,
            model_id=model_id,
            task_type=task_type,
            task_description=task.description or "",
            output=output_str,
            use_llm=False,
        )
    except Exception:
        logger.warning("Quality scoring failed during task completion", exc_info=True)

    if score is None:
        # Quality scoring crashed; emit the receipt with scoring_available=False
        # so the OutcomeSignal basis reflects "scorer down" rather than
        # "score = 0.0 means low quality".
        _emit_completion_receipt(agent, task, result, score=0.0, scoring_available=False)
        return task

    # -- Subsystem 2: Feedback loop --
    try:
        from vetinari.learning.feedback_loop import get_feedback_loop

        get_feedback_loop().record_outcome(
            task_id=task.task_id,
            model_id=model_id,
            task_type=task_type,
            quality_score=score.overall_score,
            success=result.success,
        )
    except Exception:
        logger.warning("Feedback loop recording failed during task completion", exc_info=True)

    # -- Subsystem 2.5: Agent drift detection --
    try:
        from vetinari.agents.drift_detector import get_drift_detector

        _session = getattr(agent, "_session_id", "default")
        _detector = get_drift_detector()
        _detector.record_score(agent.agent_type, _session, score.overall_score)
        _drift_report = _detector.check_drift(agent.agent_type, _session)
        if _drift_report:
            result.metadata["drift_detected"] = True
            result.metadata["drift_magnitude"] = _drift_report.drift_magnitude
    except Exception:
        logger.warning(
            "Drift detection failed for %s (non-fatal)",
            agent.agent_type.value,
            exc_info=True,
        )

    # -- Subsystem 3: Quality gate + escalation --
    _effective_score = score.overall_score
    if result.metadata.get("self_check_gate_hint") == "stricter":
        _effective_score = max(0.0, score.overall_score - 0.1)
        logger.warning(
            "Quality gate using adjusted score %.2f (raw=%.2f) due to self_check failure",
            _effective_score,
            score.overall_score,
        )
    # When self-reflection already verified the output and marked it improved,
    # the agent's own verify() is authoritative — skip the base quality gate
    # override so a passing reflection result is not silently downgraded by
    # the generic text-quality scorer which does not understand domain output.
    _self_reflection = result.metadata.get("self_reflection", {})
    _skip_gate = bool(_self_reflection.get("is_improved"))
    if hasattr(task, "_quality_gate") and task._quality_gate and not _skip_gate:
        try:
            from vetinari.constraints.registry import get_constraint_registry

            passed, reason = get_constraint_registry().check_quality_gate(
                agent.agent_type.value,
                _effective_score,
            )
            if not passed:
                logger.warning(
                    "Quality gate failed for %s — marking execution as failed: %s",
                    agent.agent_type.value,
                    reason,
                )
                result.success = False
                result.errors.append(f"Quality gate failed: {reason}")
        except Exception:
            logger.warning(
                "Failed to check quality gate for %s — gate skipped, result not blocked",
                agent.agent_type.value,
                exc_info=True,
            )

    # Skip critical escalation when self-reflection already verified the output —
    # the agent's own verify() passing is authoritative for the escalation signal.
    if not _skip_gate:
        try:
            from vetinari.constants import CRITICAL_QUALITY_THRESHOLD

            if _effective_score < CRITICAL_QUALITY_THRESHOLD:
                logger.warning(
                    "Quality score %.2f is below critical threshold %.2f — marking result for escalation/retry",
                    _effective_score,
                    CRITICAL_QUALITY_THRESHOLD,
                )
                result.metadata["quality_escalation_required"] = True
                result.metadata["quality_escalation_score"] = _effective_score
                result.errors.append(
                    f"Quality score {_effective_score:.2f} below critical threshold "
                    f"{CRITICAL_QUALITY_THRESHOLD:.2f} — retry or escalation required"
                )
        except Exception:
            logger.warning("Quality escalation check failed (non-fatal)", exc_info=True)

    # -- Subsystem 4: Prompt evolver --
    try:
        from vetinari.learning.prompt_evolver import get_prompt_evolver

        # Read the variant that was ACTUALLY used during inference — do not
        # re-call select_prompt() here because that would pick a potentially
        # different variant and attribute the quality score to the wrong
        # experiment arm.  _last_variant_id is set by inference.py after
        # each successful LLM call (ADR-0091).
        v_id = getattr(agent, "_last_variant_id", None) or "default"
        if v_id and v_id != "default":
            get_prompt_evolver().record_result(agent.agent_type.value, v_id, score.overall_score)
    except Exception:
        logger.warning(
            "Failed to record prompt evolver result for %s",
            agent.agent_type.value,
            exc_info=True,
        )

    # -- Subsystem 5: Training data collection --
    try:
        from vetinari.learning.training_data import get_training_collector

        _collector = get_training_collector()
        _collector.record(
            task=task.description or "",
            prompt=agent.get_system_prompt()[:500] + "\n\n" + (task.prompt or task.description or ""),
            response=output_str,
            score=score.overall_score,
            model_id=model_id,
            task_type=task_type,
            agent_type=agent.agent_type.value,
            success=result.success,
            # Actual telemetry from inference.py — these fields are set by _infer()
            # after each successful LLM call so the collector receives truthful
            # values instead of zeros that would trigger the data quality gate.
            latency_ms=int(getattr(agent, "_last_latency_ms", 0) or 0),
            tokens_used=getattr(agent, "_last_tokens_used", 0) or 0,
            prompt_variant_id=getattr(agent, "_last_variant_id", "") or "",
            trace_id=getattr(agent, "_last_trace_id", "") or "",
        )
        _total = _collector.count_records()
        _CURATE_INTERVAL = 100
        if _total > 0 and _total % _CURATE_INTERVAL == 0:
            import threading as _threading

            def _run_curation() -> None:
                try:
                    from vetinari.training.pipeline import DataCurator

                    DataCurator().curate(min_score=0.8, max_examples=5000)
                    logger.info("[TrainingData] Auto-curation triggered at %d records", _total)
                except Exception as _ce:
                    logger.warning("Auto-curation failed (non-fatal): %s", _ce)

            _threading.Thread(target=_run_curation, daemon=True, name="auto-curator").start()
    except Exception:
        logger.warning(
            "Failed to record execution to training data collector",
            exc_info=True,
        )

    # -- Subsystem 6: Episodic + unified memory --
    ep_id: str | None = None
    try:
        from vetinari.learning.episode_memory import get_episode_memory

        ep_id = get_episode_memory().record(
            task_description=task.description or "",
            agent_type=agent.agent_type.value,
            task_type=task_type,
            output_summary=output_str[:300],
            quality_score=score.overall_score,
            success=result.success,
            model_id=model_id,
        )
    except Exception:
        logger.warning("Failed to record to episodic memory", exc_info=True)

    # -- Subsystem 6b: Difficulty calibration feedback --
    try:
        from vetinari.learning.difficulty_feedback import record_difficulty_feedback
        from vetinari.models.model_router_scoring import assess_difficulty

        predicted = assess_difficulty(task.description or "", task_type)
        record_difficulty_feedback(
            task_type=task_type,
            predicted=predicted,
            signals={
                "quality_score": score.overall_score,
                "success": result.success,
                "duration_ms": getattr(result, "duration_ms", 0.0),
                "retries": getattr(result, "retries", 0),
                "rejections": getattr(result, "rejections", 0),
            },
            episode_id=ep_id,
        )
    except Exception:
        logger.warning("Failed to record difficulty calibration feedback", exc_info=True)

    try:
        from vetinari.memory.shared import get_shared_memory

        get_shared_memory().store.record_episode(
            task_description=task.description or "",
            agent_type=agent.agent_type.value,
            task_type=task_type,
            output_summary=output_str[:300],
            quality_score=score.overall_score,
            success=result.success,
            model_id=model_id,
        )
    except Exception:
        logger.warning("Failed to record episode to unified memory", exc_info=True)

    # -- Subsystem 7: Work receipt emission --
    # Durable per-task record consumed by the Control Center / Attention
    # track. Every completed Task gets exactly one receipt; failure paths
    # emit before their early return so the one-to-one invariant holds.
    _emit_completion_receipt(agent, task, result, score=score.overall_score)

    return task


def _emit_completion_receipt(
    agent: BaseAgent,
    task: AgentTask,
    result: AgentResult,
    *,
    score: float,
    scoring_available: bool = True,
) -> None:
    """Emit a WorkReceipt for the completed task.

    Wraps ``record_agent_completion`` so the import is local to this
    helper and a failure in receipt emission never crashes
    ``complete_task``.

    Args:
        agent: The BaseAgent instance whose work this receipt records.
        task: The completed task.
        result: The AgentResult from execution.
        score: Quality score (already computed); ``0.0`` for early
            returns where scoring did not run.
        scoring_available: True when the quality scorer produced
            ``score``; False when the scorer was bypassed or crashed
            (e.g. failed-execution early return). Drives the
            ``OutcomeSignal.basis`` so consumers can distinguish a
            real low score from a missing one.
    """
    try:
        from vetinari.receipts import record_agent_completion

        record_agent_completion(
            agent=agent,
            task=task,
            result=result,
            score=score,
            scoring_available=scoring_available,
        )
    except Exception:
        logger.warning("Failed to emit WorkReceipt during task completion", exc_info=True)
