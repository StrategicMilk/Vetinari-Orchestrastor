"""Core pipeline execution loop — the 8-stage assembly-line orchestration.

When to use this module:
    ``PipelineEngineMixin`` is the primary execution path for all standard
    Vetinari requests.  It implements the full 8-stage pipeline from intake
    classification through goal verification and learning, and is composed
    into ``TwoLayerOrchestrator``.  This is the module to read when tracing
    how a request flows from user input to final output.

Pipeline role: **all 8 stages** — this IS the pipeline.
Compare with ``durable_execution.py`` (adds checkpoint persistence on top of
these stages) and ``pipeline_agent_graph.py`` (handles stage 5 DAG dispatch).

Implements the core pipeline stages:
  0. Intake classification (Configure-to-Order)
  0.5. Production leveling (RequestQueue admission)
  0.9. Pre-execution prevention gate
  1. Input Analysis
  2-3. Plan Generation + Task Decomposition
  4. Model Assignment
  5. Parallel Execution
  5.5. Self-refinement (Custom tier)
  6. Output Review
  7. Final Assembly
  8. Goal Verification + Correction Loop

``PipelineEngineMixin`` is composed into ``TwoLayerOrchestrator`` and delegates
quality and helper calls to the other mixins via ``self``.
"""

from __future__ import annotations

import contextlib
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from vetinari.orchestration.pipeline_engine_checkpoints import _save_pipeline_checkpoint
from vetinari.orchestration.pipeline_engine_entry import _PipelineEngineEntryMixin
from vetinari.orchestration.pipeline_events import PipelineStage
from vetinari.types import StatusEnum

logger = logging.getLogger(__name__)


class PipelineEngineMixin(_PipelineEngineEntryMixin):
    """Mixin providing the full 8-stage assembly-line pipeline execution.

    Mixed into TwoLayerOrchestrator. Delegates to PipelineHelpersMixin and
    PipelineQualityMixin methods via ``self``, and accesses attributes set by
    TwoLayerOrchestrator.__init__: ``plan_generator``, ``execution_engine``,
    ``_variant_manager``, ``_event_handler``, ``_request_queue``,
    ``enable_correction_loop``, ``correction_loop_max_rounds``.
    """

    def _execute_pipeline(
        self,
        goal: str,
        constraints: dict[str, Any] | None,
        context: dict[str, Any],
        stages: dict[str, Any],
        start_time: float,
        _corr_ctx: Any,
        _pipeline_span: Any,
        _intake_tier: Any,
        _intake_features: Any,
        task_handler: Callable | None,
        project_id: str | None,
        model_id: str | None,
    ) -> dict[str, Any]:
        """Execute the full pipeline after queue admission.

        This is the inner method called by generate_and_execute() after the
        request has been admitted through the RequestQueue.

        Args:
            goal: The user goal.
            constraints: Optional constraints dict.
            context: Execution context dict.
            stages: Pipeline stages accumulator.
            start_time: Pipeline start timestamp.
            _corr_ctx: CorrelationContext or None.
            _pipeline_span: OTel pipeline span or None.
            _intake_tier: Intake tier classification or None.
            _intake_features: Intake feature extraction or None.
            task_handler: Optional task handler callback.
            project_id: Optional project identifier.
            model_id: Optional model identifier.

        Returns:
            Pipeline result dict.
        """
        # Guard: abort immediately if Andon has halted the pipeline
        if self.is_paused():  # type: ignore[attr-defined]
            logger.warning("Pipeline is paused (Andon halt) — skipping execution")
            return {"status": "paused", "reason": "Andon halt active"}

        # Pipeline state: check for resume point from a previous crash
        _pipeline_task_id = str(context.get("_exec_id") or context.get("_trace_id") or "")

        # Generate a stable trace_id for this pipeline run so all checkpoints,
        # logs, and SSE events can be correlated end-to-end.  Prefer an
        # externally supplied trace_id (e.g. from an upstream SSE client) so
        # replay requests carry the original trace through the re-execution.
        if not context.get("_trace_id"):
            context["_trace_id"] = str(uuid.uuid4())
        self._current_trace_id: str = str(context["_trace_id"])
        _state_store = None
        _completed_stages: list[str] = []
        if _pipeline_task_id:
            try:
                from vetinari.orchestration.pipeline_state import get_pipeline_state_store

                _state_store = get_pipeline_state_store()
                _resume_point = _state_store.get_resume_point(_pipeline_task_id)
                if _resume_point is not None:
                    _completed_stages = _state_store.get_completed_stages(_pipeline_task_id)
                    # NOTE: mid-pipeline resume is NOT implemented — the pipeline always
                    # re-executes from stage 0. This hint is recorded for observability
                    # only (visible in audit logs and the convergence log).
                    logger.info(
                        "[Pipeline] Task %s has prior checkpoint (%d stages recorded). "
                        "Mid-pipeline resume is not implemented — re-executing from start.",
                        _pipeline_task_id,
                        len(_completed_stages),
                    )
                    stages["last_run_stage_hint"] = _completed_stages[-1] if _completed_stages else None
            except Exception:
                logger.warning("Pipeline state store unavailable — no resume capability", exc_info=True)

        # Enrich goal with intake form context
        enriched_goal = self._enrich_goal(goal, context)  # type: ignore[attr-defined]

        # Inject rules into context for agent prompts
        try:
            from vetinari.rules_manager import get_rules_manager

            rm = get_rules_manager()
            context["_rules_prefix"] = rm.build_system_prompt_prefix(project_id=project_id, model_id=model_id)
        except Exception:  # Broad: import + runtime call; any failure must not block pipeline
            logger.warning("Failed to inject rules prefix into orchestration context", exc_info=True)

        # STAGE 0.1: System 1/System 2 routing (ADR-0095)
        _system_decision = None
        try:
            from vetinari.routing.system_router import route_system as _route_system

            _system_decision = _route_system(
                description=enriched_goal,
                intake_tier=_intake_tier.value if _intake_tier is not None else None,
                complexity=context.get("intake_features", {}).get("complexity"),
                confidence=context.get("intake_confidence", 0.5),
                involves_code_generation=True,  # conservative default
            )
            context["system_decision"] = _system_decision.to_dict()
            stages["system_routing"] = {
                "system_type": _system_decision.system_type.value,
                "model_tier": _system_decision.model_tier.value,
                "skip_foreman": _system_decision.skip_foreman,
                "skip_inspector": _system_decision.skip_inspector,
            }
        except Exception:
            logger.warning("System routing unavailable, proceeding with full pipeline", exc_info=True)

        # Express tier: bypass planning, go directly to Builder → Quality
        if _intake_tier is not None:
            try:
                from vetinari.orchestration.intake import CONFIDENCE_THRESHOLD, PipelinePaused
                from vetinari.orchestration.intake import Tier as _Tier

                if _intake_tier == _Tier.EXPRESS:
                    logger.info("[Pipeline] Express lane: skipping planning, direct to Builder")
                    return self._execute_express(  # type: ignore[attr-defined]
                        enriched_goal, context, stages, start_time, _corr_ctx, _pipeline_span, task_handler=task_handler
                    )

                # Custom tier or low confidence: run clarification first
                _needs_clarify = _intake_tier == _Tier.CUSTOM or (
                    _intake_features is not None and _intake_features.confidence < CONFIDENCE_THRESHOLD
                )
                if _needs_clarify:
                    clarify_result = self._run_clarification(enriched_goal, context)  # type: ignore[attr-defined]
                    if clarify_result is not None:
                        if isinstance(clarify_result, dict) and clarify_result.get("needs_user_input"):
                            # Pipeline pauses — return questions to caller
                            paused = PipelinePaused(
                                questions=clarify_result.get("pending_questions", []),
                                pipeline_state={"goal": goal, "tier": _intake_tier.value, "context": context},
                                tier=_intake_tier.value,
                                goal=goal,
                                confidence=_intake_features.confidence if _intake_features else 0.0,
                            )
                            stages["clarification"] = {"paused": True, "questions": len(paused.questions)}
                            logger.info("[Pipeline] Paused for clarification: %d questions", len(paused.questions))
                            return paused.to_dict()
                        # Clarification answered inline — enrich context
                        if isinstance(clarify_result, dict):
                            context.update({k: v for k, v in clarify_result.items() if k.startswith("clarification_")})
                            stages["clarification"] = {"paused": False, "enriched": True}
            except Exception:  # Broad: import + runtime call; any failure must not block pipeline
                logger.warning("Tier routing/clarification failed, proceeding with full pipeline", exc_info=True)

        # Build RequestSpec (engineering drawing) before planning
        _request_spec = None
        try:
            from vetinari.orchestration.intake import Tier as _SpecTier
            from vetinari.orchestration.request_spec import get_spec_builder

            _spec_tier = (
                _SpecTier(context.get("intake_tier", "standard")) if "intake_tier" in context else _SpecTier.STANDARD
            )
            # Prefer user-provided category from intake form over keyword re-classification
            _category = (context.get("category") or "").strip()
            if not _category:
                _category = self._analyze_input(enriched_goal, constraints or {}).get("goal_type", "general")  # type: ignore[attr-defined]  # noqa: VET112 - empty fallback preserves optional request metadata contract
            _request_spec = get_spec_builder().build(
                goal=enriched_goal,
                tier=_spec_tier,
                category=_category,
            )
            context["request_spec"] = _request_spec.to_dict()
            stages["request_spec"] = {
                "confidence": _request_spec.confidence,
                "complexity": _request_spec.estimated_complexity,
                "scope_files": len(_request_spec.scope),
                "criteria_count": len(_request_spec.acceptance_criteria),
            }
            logger.info(
                "[Pipeline] RequestSpec built: complexity=%d, confidence=%.2f, scope=%d files",
                _request_spec.estimated_complexity,
                _request_spec.confidence,
                len(_request_spec.scope),
            )
        except Exception:  # Broad: import + runtime call; any failure must not block pipeline
            logger.warning("RequestSpec builder unavailable, proceeding without spec", exc_info=True)

        # STAGE 0.9: Pre-execution prevention gate (Poka-Yoke — soft gate)
        self._emit(PipelineStage.PREVENTION, "stage_start")
        _prevention_passed = self._run_prevention_gate(enriched_goal, context)  # type: ignore[attr-defined]
        stages["prevention_gate"] = {"passed": _prevention_passed}
        self._emit(PipelineStage.PREVENTION, "stage_complete", {"passed": _prevention_passed})
        if _state_store and _pipeline_task_id:
            with contextlib.suppress(Exception):
                _state_store.mark_stage_complete(_pipeline_task_id, "prevention_gate", {"passed": _prevention_passed})
        # Observability checkpoint — enables replay and cost analysis (ADR-0091)
        _save_pipeline_checkpoint(
            trace_id=str(context.get("_trace_id") or _pipeline_task_id),
            execution_id=str(context.get("_exec_id") or _pipeline_task_id),
            step_name="prevention_gate",
            step_index=0,
            status="completed" if _prevention_passed else "failed",
            output_snapshot={"passed": _prevention_passed},
        )
        if not _prevention_passed:
            logger.warning("[Pipeline] Prevention gate failed — blocking execution")
            stages["prevention_gate"][StatusEnum.BLOCKED.value] = True
            return {
                "success": False,
                "error": "Prevention gate check failed",
                "stages": stages,
            }

        # STAGE 1: Input Analysis
        self._emit(PipelineStage.INTAKE, "stage_start", {"goal": goal[:200]})
        logger.info("[Pipeline] Stage 1: Input Analysis for goal: %s", goal[:80])
        analysis = self._analyze_input(enriched_goal, constraints or {})  # type: ignore[attr-defined]  # noqa: VET112 - empty fallback preserves optional request metadata contract
        stages["input_analysis"] = analysis
        if _state_store and _pipeline_task_id:
            with contextlib.suppress(Exception):
                _state_store.mark_stage_complete(_pipeline_task_id, "input_analysis")
        # Observability checkpoint — enables replay and cost analysis (ADR-0091)
        _save_pipeline_checkpoint(
            trace_id=str(context.get("_trace_id") or _pipeline_task_id),
            execution_id=str(context.get("_exec_id") or _pipeline_task_id),
            step_name="input_analysis",
            step_index=1,
            status="completed",
            output_snapshot={
                k: v for k, v in analysis.items() if k in ("goal_type", "complexity", "requires_planning")
            },
        )

        # STAGE 2 & 3: Plan Generation + Task Decomposition
        self._emit(PipelineStage.PLAN_GEN, "stage_start")
        logger.info("[Pipeline] Stage 2-3: Plan Generation & Decomposition")

        # Query long-term memory for past decisions and patterns relevant to this goal
        _memory_context = self._retrieve_memory_for_planning(enriched_goal)  # type: ignore[attr-defined]
        if _memory_context:
            logger.info("[Pipeline] Enriched planning with %d memory entries", len(_memory_context))

        # Merge project metadata from context into constraints for plan generation
        _plan_constraints = dict(constraints or {})  # noqa: VET112 - empty fallback preserves optional request metadata contract
        for _pck in (
            "category",
            "tech_stack",
            "priority",
            "platforms",
            "required_features",
            "things_to_avoid",
            "expected_outputs",
        ):
            if _pck in context and _pck not in _plan_constraints:
                _plan_constraints[_pck] = context[_pck]
        if _memory_context:
            _plan_constraints["memory_context"] = _memory_context

        # Load accepted ADRs so the plan generator can reference past architectural
        # decisions and avoid re-litigating settled choices.
        try:
            from vetinari.adr import ADRSystem

            _adr_system = ADRSystem()
            _accepted_adrs = _adr_system.list_adrs(status="accepted", limit=10)
            if _accepted_adrs:
                _adr_summaries = [
                    {
                        "id": a.adr_id,
                        "title": a.title,
                        "category": a.category,
                        "decision": a.decision[:300],
                    }
                    for a in _accepted_adrs
                ]
                _plan_constraints["prior_adr_decisions"] = _adr_summaries
                logger.info(
                    "[Pipeline] Injected %d ADR decision(s) into planning context",
                    len(_accepted_adrs),
                )
        except Exception:
            logger.warning("ADR loading for planning context failed (non-fatal)", exc_info=True)

        # Wire MetaAdapter: select strategy for this task before generating the plan
        try:
            from vetinari.learning.meta_adapter import get_meta_adapter

            strategy = get_meta_adapter().select_strategy(
                task_description=enriched_goal[:300],
                task_type=context.get("task_type", "general"),
            )
            context["strategy"] = strategy.to_dict()
            logger.info("MetaAdapter selected strategy: %s", strategy.source)
        except Exception:
            logger.warning("MetaAdapter strategy selection failed — using defaults")

        # Wire WorkflowLearner: get recommendations for plan structure
        try:
            from vetinari.learning.workflow_learner import get_workflow_learner

            recommendations = get_workflow_learner().get_recommendations(enriched_goal)
            if recommendations.get("confidence", 0) > 0.5:
                context["workflow_recommendations"] = recommendations
                logger.info(
                    "WorkflowLearner: domain=%s, confidence=%.2f",
                    recommendations.get("domain"),
                    recommendations.get("confidence"),
                )
        except Exception:
            logger.warning("WorkflowLearner recommendations failed — using defaults")

        graph = self.plan_generator.generate_plan(  # type: ignore[attr-defined]
            enriched_goal,
            _plan_constraints,
            max_depth=self._variant_manager.get_config().max_planning_depth,  # type: ignore[attr-defined]
        )
        stages["plan"] = {"plan_id": graph.plan_id, "tasks": len(graph.nodes)}

        # Scope-creep guard: flag any nodes whose objective has drifted
        # far from the original goal before committing to execution.
        try:
            from vetinari.drift.goal_tracker import create_goal_tracker

            _tracker = create_goal_tracker(enriched_goal)
            _creep_items = _tracker.detect_scope_creep(list(graph.nodes.values()))
            if _creep_items:
                logger.warning(
                    "[Pipeline] Scope creep detected — %d/%d tasks have low goal relevance (plan_id=%s). Tasks: %s",
                    len(_creep_items),
                    len(graph.nodes),
                    graph.plan_id,
                    [getattr(item, "task_id", str(item)) for item in _creep_items[:5]],
                )
                stages["plan"]["scope_creep_count"] = len(_creep_items)
        except ImportError:
            logger.debug("GoalTracker not available — scope-creep detection skipped")
        except Exception:
            logger.warning(
                "Scope-creep detection failed for plan %s — proceeding without drift guard",
                graph.plan_id,
                exc_info=True,
            )

        self._emit(
            PipelineStage.PLAN_GEN,
            "stage_complete",
            {
                "plan_id": graph.plan_id,
                "task_count": len(graph.nodes),
            },
        )
        if _state_store and _pipeline_task_id:
            with contextlib.suppress(Exception):
                _state_store.mark_stage_complete(
                    _pipeline_task_id, "plan_gen", {"plan_id": graph.plan_id, "tasks": len(graph.nodes)}
                )
        # Observability checkpoint — enables replay and cost analysis (ADR-0091)
        _save_pipeline_checkpoint(
            trace_id=str(context.get("_trace_id") or _pipeline_task_id),
            execution_id=str(context.get("_exec_id") or _pipeline_task_id),
            step_name="plan_gen",
            step_index=2,
            status="completed",
            output_snapshot={"plan_id": graph.plan_id, "task_count": len(graph.nodes)},
        )

        # Propagate plan_id into CorrelationContext so all downstream logs include it
        if _corr_ctx is not None:
            try:
                _corr_ctx.set_plan_id(graph.plan_id)
            except (AttributeError, TypeError):
                logger.warning("Failed to set plan_id on CorrelationContext", exc_info=True)

        # Stage-boundary validation (plan → model assignment)
        plan_valid, plan_issues = self._validate_stage_boundary(  # type: ignore[attr-defined]
            "plan",
            stages["plan"],
            min_keys=["plan_id", "tasks"],
        )
        if not plan_valid:
            logger.warning("[Pipeline] Plan validation failed: %s", plan_issues)
            if _pipeline_span is not None:
                try:
                    from vetinari.observability.otel_genai import get_genai_tracer

                    get_genai_tracer().end_agent_span(_pipeline_span, status="error")
                except (ImportError, AttributeError):
                    logger.warning("Failed to close GenAI span", exc_info=True)
            if _corr_ctx is not None:
                with contextlib.suppress(Exception):
                    _corr_ctx.__exit__(None, None, None)
            return {
                "plan_id": graph.plan_id,
                "goal": goal,
                "completed": 0,
                "failed": 1,
                "error": f"Plan validation failed: {plan_issues}",
                "stages": stages,
                "total_time_ms": int((time.time() - start_time) * 1000),
            }

        # Inject project metadata into each task node so Worker and Inspector
        # agents know the tech stack, category, and constraints.
        _project_meta = {
            k: context[k]
            for k in (
                "category",
                "tech_stack",
                "priority",
                "platforms",
                "required_features",
                "things_to_avoid",
                "expected_outputs",
            )
            if k in context
        }
        if _project_meta:
            for node in graph.nodes.values():
                node.input_data.setdefault("project_context", _project_meta)

        # Propagate request_spec into every task node so Worker and Inspector agents
        # can reference the acceptance criteria and scope when executing.
        _request_spec_dict = context.get("request_spec")
        if _request_spec_dict:
            for node in graph.nodes.values():
                node.input_data.setdefault("request_spec", _request_spec_dict)
            logger.debug(
                "[Pipeline] Injected request_spec into %d task node(s) (complexity=%s)",
                len(graph.nodes),
                _request_spec_dict.get("estimated_complexity", "?"),
            )

        # Halt flag check: abort if Andon triggered between plan and model assignment
        if self.is_paused():  # type: ignore[attr-defined]
            logger.warning("[Pipeline] Andon halt detected after planning — aborting")
            return {
                "plan_id": graph.plan_id,
                "goal": goal,
                "completed": 0,
                "failed": 0,
                "error": "Pipeline halted by Andon signal after planning stage",
                "stages": stages,
                "total_time_ms": int((time.time() - start_time) * 1000),
            }

        # STAGE 4: Model Assignment
        # When intake_confidence is low the goal is ambiguous, so request the
        # highest-capability available model to reduce error risk.
        self._emit(PipelineStage.MODEL_ASSIGN, "stage_start")
        logger.info("[Pipeline] Stage 4: Model Assignment")

        # Query context graph for SELF + USER + ENVIRONMENT awareness
        _ctx_snapshot = None
        try:
            from vetinari.awareness.context_graph import get_context_graph
            from vetinari.types import ContextQuadrant

            _ctx_graph = get_context_graph()
            _ctx_snapshot = _ctx_graph.get_context([
                ContextQuadrant.SELF,
                ContextQuadrant.USER,
                ContextQuadrant.ENVIRONMENT,
            ])
            # Inject snapshot into pipeline context for downstream stages
            context["_context_graph_snapshot"] = {
                "vram_utilization": _ctx_snapshot.get(ContextQuadrant.SELF, "vram_utilization"),
                "quality_trend": _ctx_snapshot.get(ContextQuadrant.SELF, "quality_trend"),
                "project_tech_stack": _ctx_snapshot.get(ContextQuadrant.ENVIRONMENT, "project_tech_stack"),
            }
            logger.info("[Pipeline] Context graph queried: %s", _ctx_snapshot)
        except Exception:
            logger.warning(
                "Context graph unavailable for model assignment — proceeding without awareness",
                exc_info=True,
            )

        _intake_confidence: float = float(context.get("intake_confidence", 1.0))
        _low_confidence_mode = _intake_confidence < 0.5
        if _low_confidence_mode:
            logger.info(
                "[Pipeline] Low intake confidence (%.2f) — preferring high-capability model for all tasks",
                _intake_confidence,
            )
        for node in graph.nodes.values():
            assigned = self._route_model_for_task(node)  # type: ignore[attr-defined]
            # If confidence is low, signal the model router to use a higher tier
            # by injecting the hint into the task's input_data before assignment.
            if _low_confidence_mode:
                node.input_data["require_high_capability"] = True
            # Wire CostOptimizer: pick cheapest adequate from ALL loaded models,
            # not just the Thompson-assigned one.  Passing a single-element list
            # makes CostOptimizer a no-op since there is nothing cheaper to choose.
            try:
                from vetinari.learning.cost_optimizer import get_cost_optimizer

                task_type = node.task_type or "general"
                # Source the full model list from the router registered on self.
                # Accessing self.model_router is safe — PipelineHelpersMixin sets
                # it during _route_model_for_task; fall back to [assigned] when
                # the router has not been initialised yet.
                _router = getattr(self, "model_router", None)  # type: ignore[attr-defined]
                if _router is not None and hasattr(_router, "models"):
                    candidate_models = list(_router.models.keys()) or [assigned]
                else:
                    candidate_models = [assigned]
                optimized_model = get_cost_optimizer().select_cheapest_adequate(
                    task_type=task_type,
                    candidate_models=candidate_models,
                )
                if optimized_model:
                    assigned = optimized_model
            except Exception:
                logger.warning("CostOptimizer selection failed — using Thompson selection")
            node.input_data["assigned_model"] = assigned
            logger.debug("  Task %s (%s) -> %s", node.id, node.task_type, assigned)
        stages["model_assignment"] = {nid: n.input_data.get("assigned_model") for nid, n in graph.nodes.items()}
        # Aggregate model-selection confidence levels for observability
        _conf_levels: dict[str, int] = {}
        for _n in graph.nodes.values():
            _cl = _n.input_data.get("_selection_confidence_level", "unknown")
            _conf_levels[_cl] = _conf_levels.get(_cl, 0) + 1
        stages["model_assignment_confidence"] = _conf_levels

        # Log model assignment decisions to the decision journal
        try:
            from vetinari.observability.decision_journal import get_decision_journal
            from vetinari.types import ConfidenceLevel, DecisionType

            _journal = get_decision_journal()
            _trace_id = context.get("_trace_id") or context.get("_exec_id")
            for _nid, _node in graph.nodes.items():
                _assigned_model = _node.input_data.get("assigned_model", "unknown")
                _sel_confidence = _node.input_data.get("_selection_confidence_level", "medium")
                _conf_map = {
                    "high": ConfidenceLevel.HIGH,
                    "medium": ConfidenceLevel.MEDIUM,
                    "low": ConfidenceLevel.LOW,
                    "very_low": ConfidenceLevel.VERY_LOW,
                }
                _journal.log_decision(
                    decision_type=DecisionType.MODEL_SELECTION,
                    chosen=_assigned_model,
                    confidence=_conf_map.get(_sel_confidence, ConfidenceLevel.MEDIUM),
                    reasoning=_node.input_data.get("_selection_confidence_explanation", ""),
                    trace_id=str(_trace_id) if _trace_id else None,
                    metadata={"task_id": _nid, "task_type": _node.task_type or "general"},
                )
        except Exception:
            logger.warning(
                "Decision journal unavailable for model assignment logging",
                exc_info=True,
            )

        self._emit(
            PipelineStage.MODEL_ASSIGN,
            "stage_complete",
            {
                "assignments": len(graph.nodes),
                "confidence_levels": _conf_levels,
            },
        )
        if _state_store and _pipeline_task_id:
            with contextlib.suppress(Exception):
                _state_store.mark_stage_complete(_pipeline_task_id, "model_assignment")
        # Observability checkpoint — enables replay and cost analysis (ADR-0091)
        _save_pipeline_checkpoint(
            trace_id=str(context.get("_trace_id") or _pipeline_task_id),
            execution_id=str(context.get("_exec_id") or _pipeline_task_id),
            step_name="model_assignment",
            step_index=3,
            status="completed",
            output_snapshot={
                "assignments": {nid: n.input_data.get("assigned_model") for nid, n in graph.nodes.items()},
                "confidence_levels": _conf_levels,
            },
        )

        # Confidence check: flag low-confidence model assignments and optionally
        # defer to autonomy governor for intervention.  Decision: Session 11.
        _low_conf_tasks: list[str] = []
        for _nid, _node in graph.nodes.items():
            _sel_level = _node.input_data.get("_selection_confidence_level")
            if _sel_level in ("low", "very_low"):
                _low_conf_tasks.append(_nid)
                logger.warning(
                    "[Pipeline] Low model-selection confidence for task %s: %s",
                    _nid,
                    _node.input_data.get("_selection_confidence_explanation", "no explanation"),
                )
        if _low_conf_tasks:
            try:
                from vetinari.autonomy.governor import get_governor
                from vetinari.types import PermissionDecision

                _gov = get_governor()
                _perm = _gov.request_permission(
                    "low_confidence_execution",
                    details={
                        "low_confidence_tasks": _low_conf_tasks,
                        "task_count": len(_low_conf_tasks),
                        "total_tasks": len(graph.nodes),
                    },
                )
                # DEFER is blocking — must treat it the same as DENY.
                # Only APPROVE may proceed. A DEFER means the action has been
                # queued for human review and must not execute automatically.
                if _perm != PermissionDecision.APPROVE:
                    logger.warning(
                        "[Pipeline] Autonomy governor blocked execution (%s) — %d/%d tasks have low confidence",
                        _perm.value,
                        len(_low_conf_tasks),
                        len(graph.nodes),
                    )
                    return {
                        "plan_id": graph.plan_id,
                        "goal": goal,
                        "completed": 0,
                        "failed": 0,
                        "error": f"Execution blocked: {len(_low_conf_tasks)} task(s) have low model-selection confidence",
                        "stages": stages,
                        "total_time_ms": int((time.time() - start_time) * 1000),
                    }
            except Exception:
                logger.warning(
                    "Autonomy governor unavailable for confidence check — proceeding",
                    exc_info=True,
                )

        # Halt flag check: abort if Andon triggered between model assignment and execution
        if self.is_paused():  # type: ignore[attr-defined]
            logger.warning("[Pipeline] Andon halt detected before execution — aborting")
            return {
                "plan_id": graph.plan_id,
                "goal": goal,
                "completed": 0,
                "failed": 0,
                "error": "Pipeline halted by Andon signal before execution stage",
                "stages": stages,
                "total_time_ms": int((time.time() - start_time) * 1000),
            }

        # Context budget check between planning and execution stages
        try:
            from vetinari.context.pipeline_integration import create_pipeline_context_manager

            # Use the first assigned model's ID for budget sizing
            _assigned_models = list(stages.get("model_assignment", {}).values())
            _budget_model_id = _assigned_models[0] if _assigned_models else "default"
            context["_context_manager"] = create_pipeline_context_manager(_budget_model_id)
            logger.info("[Pipeline] Context budget tracker initialized for model %s", _budget_model_id)
        except Exception:
            logger.warning("Context budget tracker setup failed — proceeding without budget management", exc_info=True)

        return self._run_execution_stages(
            goal,
            graph,
            context,
            stages,
            start_time,
            _corr_ctx,
            _pipeline_span,
            task_handler,
            project_id,
            _intake_tier,
            _intake_features,
        )
