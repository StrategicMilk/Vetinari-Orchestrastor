"""Pipeline execution stages 5-8 mixin.

This is the second half of the assembly-line orchestration, covering:
  5. Parallel Execution
  5.5. Self-refinement (Custom tier)
  6. Output Review
  7. Final Assembly
  8. Goal Verification + Correction Loop
  Post-pipeline: telemetry, Thompson Sampling, SPC, ConversationStore

``PipelineStagesMixin`` is composed into ``TwoLayerOrchestrator`` alongside
``PipelineEngineMixin``. Methods are called from ``PipelineEngineMixin._execute_pipeline``
via ``self._run_execution_stages``.

The AgentGraph execution backend (``execute_with_agent_graph``) lives in
``pipeline_agent_graph.py`` and is composed in via ``PipelineAgentGraphMixin``.
"""

from __future__ import annotations

import contextlib
import logging
import random
import time
from collections.abc import Callable
from typing import Any

from vetinari.events import (
    QualityGateResult,
    get_event_bus,
)
from vetinari.orchestration.pipeline_events import PipelineStage
from vetinari.types import StatusEnum

from .pipeline_agent_graph import PipelineAgentGraphMixin
from .pipeline_stages_execution import _PipelineStageExecutionSideEffectsMixin

logger = logging.getLogger(__name__)

# Controls whether the optional collaboration blackboard path is attempted.
# Set to False in tests (via patch) to skip the CollaborationMixin import.
_COLLABORATION_AVAILABLE: bool = True


class PipelineStagesMixin(_PipelineStageExecutionSideEffectsMixin, PipelineAgentGraphMixin):
    """Mixin providing pipeline stages 5-8 and the AgentGraph execution backend.

    Mixed into TwoLayerOrchestrator. All methods access ``self`` attributes
    set by TwoLayerOrchestrator.__init__. Methods are called by
    PipelineEngineMixin._execute_pipeline after stages 0-4 complete.
    """

    def _run_review_gate(
        self,
        *,
        graph: Any,
        stages: dict[str, Any],
        review_result: dict[str, Any],
    ) -> None:
        """Evaluate the Inspector's review result and block assembly on failure.

        Uses a fail-closed default: if ``passed`` is absent, the gate blocks.
        Also blocks when ``quality_score`` is present and below 0.5.
        When the gate blocks, publishes issues to the collaboration blackboard
        if ``_COLLABORATION_AVAILABLE`` is True and the import succeeds.

        Args:
            graph: The active AgentGraph (used for plan_id in log messages).
            stages: Mutable stages dict; sets ``gate_blocked`` and ``gate_issues``
                on failure so downstream assembly can detect and surface the block.
            review_result: Raw dict returned by the Inspector agent, expected to
                contain optional keys ``passed`` (bool), ``issues`` (list), and
                ``quality_score`` (float).
        """
        # Fail-closed: absent "passed" key is treated as passed=False
        inspector_passed = review_result.get("passed", False)
        inspector_issues = review_result.get("issues", [])
        _quality_score = review_result.get("quality_score", 1.0)
        # Block when quality score is present and explicitly below threshold
        _score_failed = isinstance(_quality_score, (int, float)) and _quality_score < 0.5
        if not inspector_passed or _score_failed:
            logger.warning(
                "[Pipeline] Inspector gate FAILED — blocking final assembly "
                "(passed=%s, score=%s, issues=%d)",
                inspector_passed,
                _quality_score,
                len(inspector_issues),
            )
            stages["gate_blocked"] = True
            stages["gate_issues"] = inspector_issues[:10]
            # Publish findings to the blackboard so other agents can query them
            if _COLLABORATION_AVAILABLE:
                try:
                    from vetinari.agents.collaboration import CollaborationMixin

                    _collab = CollaborationMixin()
                    _collab.publish_finding(
                        "inspector_issues",
                        inspector_issues,
                        finding_type="quality",
                    )
                except Exception:
                    logger.warning(
                        "Could not publish inspector findings to blackboard — "
                        "findings not shared with other agents",
                        exc_info=True,
                    )

    def _run_execution_stages(
        self,
        goal: str,
        graph: Any,
        context: dict[str, Any],
        stages: dict[str, Any],
        start_time: float,
        _corr_ctx: Any,
        _pipeline_span: Any,
        task_handler: Callable | None,
        project_id: str | None,
        _intake_tier: Any,
        _intake_features: Any,
    ) -> dict[str, Any]:
        """Execute stages 5-8: execution, review, assembly, verification, telemetry.

        Called by ``PipelineEngineMixin._execute_pipeline`` after model assignment
        is complete and the Andon halt pre-check has passed.

        Args:
            goal: The user goal.
            graph: The ExecutionGraph with model assignments already applied.
            context: Pipeline context dict.
            stages: Stages accumulator dict (mutated in place).
            start_time: Pipeline start timestamp for elapsed time calculation.
            _corr_ctx: CorrelationContext or None.
            _pipeline_span: OTel pipeline span or None.
            task_handler: Optional task handler callback.
            project_id: Optional project identifier.
            _intake_tier: Intake tier classification (for Thompson Sampling).
            _intake_features: Intake features (for Thompson Sampling).

        Returns:
            Pipeline result dict.
        """
        # STAGE 5: Parallel Execution
        # Route through AgentGraph for plans with independent subtasks (true
        # parallel execution via registered agents), falling back to the
        # DurableExecutionEngine handler path when AgentGraph is unavailable.
        self._emit(PipelineStage.EXECUTION, "stage_start", {"total_tasks": len(graph.nodes)})  # type: ignore[attr-defined]
        logger.info("[Pipeline] Stage 5: Parallel Execution")
        exec_results = self._execute_via_agent_graph_or_fallback(graph, task_handler, context)
        stages["execution"] = exec_results
        self._emit(
            PipelineStage.EXECUTION,
            "stage_complete",
            {  # type: ignore[attr-defined]
                "completed": exec_results.get(StatusEnum.COMPLETED.value, 0),
                "failed": exec_results.get(StatusEnum.FAILED.value, 0),
            },
        )

        # Pipeline state: checkpoint execution stage completion
        _pipeline_task_id = str(context.get("_exec_id") or context.get("_trace_id") or "")
        if _pipeline_task_id:
            try:
                from vetinari.orchestration.pipeline_state import get_pipeline_state_store

                get_pipeline_state_store().mark_stage_complete(
                    _pipeline_task_id,
                    "execution",
                    {
                        "completed": exec_results.get(StatusEnum.COMPLETED.value, 0),
                        "failed": exec_results.get(StatusEnum.FAILED.value, 0),
                    },
                )
            except Exception:
                logger.warning("Pipeline state checkpoint skipped for execution stage", exc_info=True)
        # Observability checkpoint — enables replay and cost analysis (ADR-0091)
        with contextlib.suppress(Exception):
            from vetinari.observability.checkpoints import PipelineCheckpoint, get_checkpoint_store

            get_checkpoint_store().save_checkpoint(
                PipelineCheckpoint(
                    trace_id=str(context.get("_trace_id") or _pipeline_task_id),
                    execution_id=str(context.get("_exec_id") or _pipeline_task_id),
                    step_name="execution",
                    step_index=4,
                    status="completed" if exec_results.get(StatusEnum.COMPLETED.value, 0) > 0 else "failed",
                    output_snapshot={
                        "completed": exec_results.get(StatusEnum.COMPLETED.value, 0),
                        "failed": exec_results.get(StatusEnum.FAILED.value, 0),
                    },
                )
            )

        # Emit EventBus events for each completed/failed task
        # Side effects:
        #   - Publishes TaskStarted, TaskCompleted, and TaskTimingRecord to the EventBus
        #   - Subscribers (FeedbackLoop, SPCMonitor, ValueStreamAnalyzer) receive these
        _exec_id = context.get("_exec_id", graph.plan_id)
        _bus = get_event_bus()
        self._publish_task_execution_events(
            bus=_bus,
            graph=graph,
            start_time=start_time,
            execution_id=_exec_id,
        )

        self._record_post_execution_system_updates(graph=graph)

        # Stage-boundary validation (execution → review)
        exec_valid, exec_issues = self._validate_stage_boundary(  # type: ignore[attr-defined]
            "execution",
            exec_results,
            min_keys=[StatusEnum.COMPLETED.value],
        )
        if not exec_valid:
            logger.warning("[Pipeline] Execution validation failed: %s", exec_issues)

        # Constraint enforcement between execution and review stages
        _agent_type = context.get("agent_type", "WORKER")
        _agent_mode = context.get("mode")
        _exec_quality = None
        if isinstance(exec_results, dict):
            # Extract quality from task results if available
            for _tr in exec_results.get("task_results", {}).values():
                if isinstance(_tr, dict) and "quality_score" in _tr:
                    _exec_quality = _tr["quality_score"]
                    break
        constraints_ok, constraint_violations = self._check_stage_constraints(  # type: ignore[attr-defined]
            _agent_type,
            _agent_mode,
            _exec_quality,
        )
        if not constraints_ok:
            stages["constraint_violations"] = constraint_violations

        # Context window boundary management (execution → review)
        # Compress context if above 50%, page out oldest messages when overloaded,
        # and record window state in stages for observability.
        try:
            from vetinari.context.window_manager import get_window_manager

            _model_id = context.get("model_id", "default")
            _wm = get_window_manager(str(_model_id))
            _saved = _wm.stage_boundary_compress("execution→review")
            _win_state = _wm.get_state()
            if _wm.usage_ratio > 0.85:
                _evicted = _wm.page_out(count=10)
                logger.info(
                    "[Pipeline] Context window %.0f%% full after execution — paged out %d messages",
                    _wm.usage_ratio * 100,
                    len(_evicted),
                )
            stages["context_window_after_execution"] = {
                "used_tokens": _win_state.used_tokens,
                "max_tokens": _win_state.max_tokens,
                "tokens_saved_by_compression": _saved,
            }
        except (ImportError, AttributeError):
            logger.debug("Context window management skipped at execution→review boundary — manager unavailable")

        # StepEvaluator plan adherence scoring after execution
        try:
            from vetinari.observability.step_evaluator import get_step_evaluator

            _step_eval = get_step_evaluator()
            _plan_dict = {"plan_id": graph.plan_id, "tasks": [n.to_dict() for n in graph.nodes.values()]}
            _exec_results_for_eval = {
                tid: {"status": "completed" if n.status == StatusEnum.COMPLETED else "failed"}
                for tid, n in graph.nodes.items()
            }
            _adherence = _step_eval.evaluate_all(_plan_dict, _exec_results_for_eval)
            stages["step_evaluation"] = {
                "overall_score": _adherence.overall_score,
                "passed": _adherence.passed,
            }
            logger.info(
                "[Pipeline] StepEvaluator: overall=%.2f, passed=%s",
                _adherence.overall_score,
                _adherence.passed,
            )
        except Exception:
            logger.warning("StepEvaluator unavailable, skipping plan adherence check", exc_info=True)

        # Context budget check after Stage 5 (execution) before refinement/review
        _ctx_mgr = context.get("_context_manager")
        if _ctx_mgr is not None:
            try:
                from vetinari.context.window_manager import WindowConversationMessage, get_window_manager

                _wm = get_window_manager(context.get("_budget_model_id", "default"))
                _msgs = [WindowConversationMessage(role=m["role"], content=m["content"]) for m in _wm.get_messages()]
                if _msgs:
                    _msgs, _budget_check = _ctx_mgr.check_budget("execution", _msgs)
                    logger.debug("[Pipeline] Post-execution context budget: %s", _budget_check.status.value)
            except Exception:
                logger.warning("Context budget check failed after execution stage — continuing", exc_info=True)

        # Confidence-gated routing — classify Worker output confidence from
        # token logprobs and apply the appropriate action (self-reflection,
        # best-of-n resampling, or human deferral) before Inspector review.
        try:
            from vetinari.orchestration.pipeline_confidence import apply_confidence_routing

            exec_results = apply_confidence_routing(exec_results, context)
        except Exception:
            logger.warning(
                "Confidence routing unavailable — proceeding without confidence gating",
                exc_info=True,
            )

        # STAGE 5.5: Self-refinement for Custom tier (Dept 4.3 #36)
        # Skipped for Express tier (speed priority) — only Custom/high-importance
        self._emit(PipelineStage.REFINEMENT, "stage_start", {"tier": context.get("intake_tier", "standard")})  # type: ignore[attr-defined]
        if context.get("intake_tier") == "custom":
            try:
                from vetinari.learning.self_refinement import get_self_refiner

                refiner = get_self_refiner()
                for task_id, task_result in exec_results.get("task_results", {}).items():
                    if isinstance(task_result, dict) and task_result.get("status") == StatusEnum.COMPLETED.value:
                        output = task_result.get("output", "")
                        if output:
                            refined = refiner.refine(
                                task_description=goal,
                                initial_output=str(output),
                                task_type=str(context.get("mode", "general") or "general"),
                                model_id=str(context.get("model_id", "default") or "default"),
                                importance=0.8,
                            )
                            if refined.improved:
                                task_result["output"] = refined.output
                                logger.info(
                                    "[Pipeline] Self-refinement improved task %s (rounds=%d)",
                                    task_id,
                                    refined.rounds_used,
                                )
                stages["self_refinement"] = {"applied": True, "tier": "custom"}
            except Exception:
                logger.warning("Self-refinement unavailable, skipping", exc_info=True)
                stages["self_refinement"] = {"applied": False, "reason": "unavailable"}

        # Halt flag check: abort if Andon triggered during execution
        if self.is_paused():  # type: ignore[attr-defined]
            logger.warning("[Pipeline] Andon halt detected after execution — aborting before review")
            return {
                "plan_id": graph.plan_id,
                "goal": goal,
                "completed": exec_results.get(StatusEnum.COMPLETED.value, 0),
                "failed": exec_results.get(StatusEnum.FAILED.value, 0),
                "error": "Pipeline halted by Andon signal after execution stage",
                "stages": stages,
                "total_time_ms": int((time.time() - start_time) * 1000),
            }

        self._emit(PipelineStage.REFINEMENT, "stage_complete")  # type: ignore[attr-defined]

        # Sandbox validation for code outputs (pre-review gate)
        if isinstance(exec_results, dict) and context.get("mode") == "build":
            for _task_id, _task_result in exec_results.get("task_results", {}).items():
                if isinstance(_task_result, dict) and _task_result.get("code"):
                    _sb_ok, _sb_detail = self._sandbox_validate_code_output(  # type: ignore[attr-defined]
                        _task_result["code"],
                    )
                    _task_result["sandbox_passed"] = _sb_ok
                    _task_result["sandbox_detail"] = _sb_detail
                    if not _sb_ok:
                        logger.warning(
                            "[Pipeline] Sandbox validation failed for task %s: %s",
                            _task_id,
                            _sb_detail,
                        )

        # STAGE 6: Output Review
        self._emit(PipelineStage.REVIEW, "stage_start")  # type: ignore[attr-defined]
        logger.info("[Pipeline] Stage 6: Output Review")
        review_result = self._review_outputs(exec_results, goal, context)  # type: ignore[attr-defined]
        stages["review"] = review_result
        self._emit(
            PipelineStage.REVIEW,
            "stage_complete",
            {  # type: ignore[attr-defined]
                "passed": review_result.get("passed", True),
                "quality_score": review_result.get("quality_score"),
            },
        )
        # Observability checkpoint — enables replay and cost analysis (ADR-0091)
        with contextlib.suppress(Exception):
            from vetinari.observability.checkpoints import PipelineCheckpoint, get_checkpoint_store

            _review_ckpt_tid = str(context.get("_trace_id") or context.get("_exec_id") or "")
            get_checkpoint_store().save_checkpoint(
                PipelineCheckpoint(
                    trace_id=_review_ckpt_tid,
                    execution_id=str(context.get("_exec_id") or _review_ckpt_tid),
                    step_name="review",
                    step_index=5,
                    status="completed" if review_result.get("passed", True) else "failed",
                    output_snapshot={
                        "passed": review_result.get("passed", True),
                        "quality_score": review_result.get("quality_score"),
                        "issues_count": len(review_result.get("issues", [])),
                    },
                    quality_score=float(review_result.get("quality_score") or 0.0) or None,
                )
            )

        # Random Inspector post-hoc audit (10% of passed tasks).
        # Re-runs the review independently to catch quality drift between the
        # primary and a second Inspector invocation. A large delta (>0.2) means
        # the review is non-deterministic or the output is borderline — worth
        # surfacing for calibration even when the original gate passed.
        if review_result.get("passed", False) and random.random() < 0.10:  # noqa: S311 — sampling, not cryptography
            try:
                logger.info("[PostHocAudit] Running random post-hoc audit on completed task")
                posthoc_review = self._review_outputs(exec_results, goal, context)  # type: ignore[attr-defined]
                posthoc_quality = posthoc_review.get("quality_score", 0.5)
                original_quality = review_result.get("quality_score", 0.5)
                if abs(posthoc_quality - original_quality) > 0.2:
                    logger.warning(
                        "[PostHocAudit] Quality drift detected: original=%.2f, posthoc=%.2f (delta=%.2f)",
                        original_quality,
                        posthoc_quality,
                        posthoc_quality - original_quality,
                    )
            except Exception:
                logger.warning("Post-hoc audit failed — non-fatal, primary review result stands")

        # Emit QualityGateResult event — consumed by SPCMonitor
        _review_score = review_result.get("quality_score", 0.0)
        _review_passed = review_result.get("passed", False)
        _review_issues = review_result.get("issues", [])
        _bus.publish(
            QualityGateResult(
                event_type="QualityGateResult",
                timestamp=time.time(),
                task_id=graph.plan_id,
                passed=bool(_review_passed),
                score=float(_review_score) if isinstance(_review_score, (int, float)) else 0.0,
                issues=[str(i) for i in _review_issues[:20]],
            )
        )

        # Check Inspector gate — block assembly if review explicitly failed
        self._run_review_gate(graph=graph, stages=stages, review_result=review_result)

        # Context budget check after Stage 6 (review) before assembly
        _ctx_mgr = context.get("_context_manager")
        if _ctx_mgr is not None:
            try:
                from vetinari.context.window_manager import WindowConversationMessage, get_window_manager

                _wm = get_window_manager(context.get("_budget_model_id", "default"))
                _msgs = [WindowConversationMessage(role=m["role"], content=m["content"]) for m in _wm.get_messages()]
                if _msgs:
                    _msgs, _budget_check = _ctx_mgr.check_budget("review", _msgs)
                    logger.debug("[Pipeline] Post-review context budget: %s", _budget_check.status.value)
            except Exception:
                logger.warning("Context budget check failed after review stage — continuing", exc_info=True)

        # Inspector Extract: surface implicit decisions from review output
        if review_result and context.get("diff"):
            try:
                from vetinari.agents.inspector_extract import (
                    extract_implicit_decisions,
                    log_extracted_decisions,
                )

                _candidates = extract_implicit_decisions(context["diff"])
                if _candidates:
                    _logged_ids = log_extracted_decisions(_candidates)
                    stages["inspector_extract"] = {
                        "candidates_found": len(_candidates),
                        "decisions_logged": len(_logged_ids),
                    }
            except Exception:
                logger.warning(
                    "Inspector Extract skipped — extraction unavailable, proceeding without implicit decision surfacing",
                )

        # STAGE 7: Final Assembly
        self._emit(PipelineStage.ASSEMBLY, "stage_start")  # type: ignore[attr-defined]
        logger.info("[Pipeline] Stage 7: Final Assembly")
        _gate_blocked = stages.get("gate_blocked", False)
        _gate_issues = stages.get("gate_issues", [])
        if _gate_blocked and _gate_issues:
            # Assembly proceeds but output is annotated with gate failure details
            final_output = self._assemble_final_output(exec_results, review_result, goal)  # type: ignore[attr-defined]
            final_output = f"[INSPECTOR GATE FAILED — {len(_gate_issues)} issue(s) found]\n\n{final_output}"
            logger.warning("[Pipeline] Final output annotated with Inspector gate failure")
        else:
            final_output = self._assemble_final_output(exec_results, review_result, goal)  # type: ignore[attr-defined]
        stages["final_assembly"] = {"output_length": len(str(final_output))}

        self._emit(
            PipelineStage.ASSEMBLY,
            "stage_complete",
            {  # type: ignore[attr-defined]
                "output_length": len(str(final_output)),
            },
        )
        # Observability checkpoint — enables replay and cost analysis (ADR-0091)
        with contextlib.suppress(Exception):
            from vetinari.observability.checkpoints import PipelineCheckpoint, get_checkpoint_store

            _assembly_ckpt_tid = str(context.get("_trace_id") or context.get("_exec_id") or "")
            get_checkpoint_store().save_checkpoint(
                PipelineCheckpoint(
                    trace_id=_assembly_ckpt_tid,
                    execution_id=str(context.get("_exec_id") or _assembly_ckpt_tid),
                    step_name="assembly",
                    step_index=6,
                    status="completed",
                    output_snapshot={"output_length": len(str(final_output))},
                )
            )

        # STAGE 8: Goal Verification + Correction Loop
        goal_verification_report = None
        _quality_score = None
        if self.enable_correction_loop:  # type: ignore[attr-defined]
            self._emit(PipelineStage.VERIFICATION, "stage_start")  # type: ignore[attr-defined]
            logger.info("[Pipeline] Stage 8: Goal Verification + Correction Loop")
            try:
                from vetinari.validation import get_goal_verifier

                verifier = get_goal_verifier()
                task_outputs_for_verify = [
                    {"output": str(v)} for v in exec_results.get("task_results", {}).values() if v
                ]
                initial_report = verifier.verify(
                    project_id=project_id or graph.plan_id,
                    goal=goal,
                    final_output=str(final_output),
                    required_features=context.get("required_features"),
                    things_to_avoid=context.get("things_to_avoid"),
                    task_outputs=task_outputs_for_verify,
                )
                corrective_tasks = initial_report.get_corrective_tasks()
                if corrective_tasks and not initial_report.fully_compliant:
                    logger.info(
                        "[Pipeline] Goal verification incomplete (score=%.2f), running %d corrective task(s)",
                        initial_report.compliance_score,
                        len(corrective_tasks),
                    )
                    plan_dict = {
                        "project_id": project_id or graph.plan_id,
                        "required_features": context.get("required_features", []),
                        "things_to_avoid": context.get("things_to_avoid", []),
                        "final_output": str(final_output),
                        "task_outputs": task_outputs_for_verify,
                    }
                    goal_verification_report = self._execute_corrections(  # type: ignore[attr-defined]
                        corrective_tasks=corrective_tasks,
                        plan=plan_dict,
                        goal=goal,
                        context=context,
                    )
                else:
                    goal_verification_report = initial_report
                stages["goal_verification"] = {
                    "compliance_score": goal_verification_report.compliance_score,
                    "fully_compliant": goal_verification_report.fully_compliant,
                    "missing_features": goal_verification_report.missing_features,
                }
            except Exception as _gv_err:
                logger.warning("[Pipeline] Goal verification stage failed (non-fatal): %s", _gv_err)
            self._emit(
                PipelineStage.VERIFICATION,
                "stage_complete",
                {  # type: ignore[attr-defined]
                    "compliance_score": goal_verification_report.compliance_score if goal_verification_report else None,
                    "fully_compliant": goal_verification_report.fully_compliant if goal_verification_report else None,
                },
            )

        total_time = int((time.time() - start_time) * 1000)
        result_dict: dict[str, Any] = {
            "plan_id": graph.plan_id,
            "goal": goal,
            "completed": exec_results.get(StatusEnum.COMPLETED.value, 0),
            "failed": exec_results.get(StatusEnum.FAILED.value, 0),
            "outputs": exec_results.get("task_results", {}),
            "review_result": review_result,
            "final_output": final_output,
            "stages": stages,
            "total_time_ms": total_time,
            "inspector_gate_passed": not stages.get("gate_blocked"),
        }
        if goal_verification_report is not None:
            result_dict["goal_verification"] = goal_verification_report.to_dict()

        # Generate decision trailers for commit linking
        try:
            from vetinari.git.trailers import generate_trailers

            _pipeline_trace = context.get("_trace_id") or context.get("_exec_id")
            if _pipeline_trace:
                _trailers = generate_trailers(trace_id=str(_pipeline_trace))
                if _trailers:
                    result_dict["decision_trailers"] = _trailers
        except Exception:
            logger.warning(
                "Decision trailer generation skipped — trailer system unavailable",
            )

        # Close the pipeline span
        if _pipeline_span is not None:
            try:
                from vetinari.observability.otel_genai import get_genai_tracer

                _status = "ok" if exec_results.get(StatusEnum.FAILED.value, 0) == 0 else "error"
                get_genai_tracer().end_agent_span(_pipeline_span, status=_status)
            except (ImportError, AttributeError):
                logger.warning("Failed to close GenAI pipeline span")

        # Exit the correlation context after the full pipeline completes
        if _corr_ctx is not None:
            try:
                _corr_ctx.__exit__(None, None, None)
            except (AttributeError, TypeError):
                logger.warning("Failed to exit CorrelationContext after pipeline", exc_info=True)

        # Record tier outcome for Thompson Sampling adaptive routing
        if _intake_tier is not None and _intake_features is not None:
            try:
                from vetinari.learning.model_selector import get_thompson_selector

                _quality = 1.0
                if goal_verification_report is not None:
                    _quality = goal_verification_report.compliance_score
                elif exec_results.get(StatusEnum.FAILED.value, 0) > 0:
                    _completed = exec_results.get(StatusEnum.COMPLETED.value, 0)
                    _failed = exec_results.get(StatusEnum.FAILED.value, 0)
                    _quality = _completed / max(_completed + _failed, 1)

                _rework = 0
                if goal_verification_report is not None and hasattr(goal_verification_report, "corrective_rounds"):
                    _rework = goal_verification_report.corrective_rounds

                get_thompson_selector().update_tier(
                    pattern_key=_intake_features.pattern_key,
                    tier_used=_intake_tier.value,
                    quality_score=_quality,
                    rework_count=_rework,
                )
                logger.info(
                    "[Pipeline] Thompson tier outcome recorded: tier=%s, quality=%.2f, rework=%d",
                    _intake_tier.value,
                    _quality,
                    _rework,
                )
            except Exception:
                logger.warning("Thompson tier outcome recording failed (non-fatal)", exc_info=True)

        # Feed SPC from pipeline execution metrics
        try:
            from vetinari.workflow import get_spc_monitor

            _spc = get_spc_monitor()
            _quality_score = result_dict.get("review_result", {}).get("quality_score")
            if isinstance(_quality_score, (int, float)):
                _spc.update("quality_score", float(_quality_score))
            _spc.update("latency_ms", float(total_time))
            # Token count: sum from execution task results if available
            _total_tokens = 0
            for _tr in exec_results.get("task_results", {}).values():
                if isinstance(_tr, dict):
                    _total_tokens += int(_tr.get("tokens_used", 0))
            if _total_tokens > 0:
                _spc.update("token_count", float(_total_tokens))
            logger.debug(
                "[Pipeline] SPC metrics fed: quality=%.2f, latency=%dms, tokens=%d",
                _quality_score if _quality_score is not None else 0.0,
                total_time,
                _total_tokens,
            )
        except Exception:
            logger.warning("SPC metric feed failed (non-fatal)", exc_info=True)

        # Track goal/output in ConversationStore
        try:
            from vetinari.async_support.conversation import get_conversation_store

            _conv = get_conversation_store()
            _session_id = context.get("session_id") or graph.plan_id
            with contextlib.suppress(ValueError, KeyError):
                _conv.create_session(_session_id)
            _conv.add_message(_session_id, "user", goal)
            _conv.add_message(
                _session_id,
                "assistant",
                str(final_output)[:2000] if final_output else "(no output)",
                metadata={"plan_id": graph.plan_id, "quality_score": _quality_score},
            )
        except Exception:
            logger.warning("ConversationStore tracking failed (non-fatal)", exc_info=True)

        return result_dict
