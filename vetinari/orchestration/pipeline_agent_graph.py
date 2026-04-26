"""AgentGraph execution backend — DAG-based parallel execution for multi-task plans.

When to use this module:
    ``PipelineAgentGraphMixin`` handles the graph-topology portion of
    TwoLayerOrchestrator execution: it builds a directed-acyclic-graph of
    tasks, resolves dependency order, and dispatches independent tasks in
    parallel while serialising dependent ones.  Use this path when a plan
    has tasks with inter-task dependencies that allow safe parallelism.

Pipeline role: Plan → **AgentGraph** (dependency-aware dispatch) → collect results.
Compare with ``async_executor.py`` (wave-based async, simpler dependency model) and
``durable_execution.py`` (checkpoint-based, crash-resumable).

Extracted from ``pipeline_stages.py`` to keep that file under the 550-line
ceiling.  ``PipelineAgentGraphMixin`` is composed into
``PipelineStagesMixin`` (and therefore into ``TwoLayerOrchestrator``).

All methods access ``self`` attributes set by
``TwoLayerOrchestrator.__init__``.
"""

from __future__ import annotations

import logging
from typing import Any

from vetinari.types import AgentType, StatusEnum

logger = logging.getLogger(__name__)


class PipelineAgentGraphMixin:
    """Mixin providing the AgentGraph execution backend for the pipeline.

    Mixed into ``PipelineStagesMixin``. Accesses ``self`` attributes such as
    ``plan_generator``, ``_variant_manager``, ``_handle_quality_rejection``,
    and ``_execute_rework_decision`` that are defined on
    ``TwoLayerOrchestrator`` (via other mixins).
    """

    def execute_with_agent_graph(
        self,
        goal: str,
        constraints: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a goal using AgentGraph as the execution backend.

        Delegates task execution to AgentGraph's registered agents instead of
        the generic DurableExecutionEngine handler, enabling agent-specific
        execution with proper verification, self-correction loops, and
        maker-checker pattern for BUILDER outputs.

        Falls back to ``generate_and_execute()`` if AgentGraph is unavailable.

        Args:
            goal: The goal.
            constraints: The constraints.
            context: The context.

        Returns:
            Dict with ``plan_id``, ``goal``, ``backend`` (``"agent_graph"``),
            ``completed``, ``failed``, ``outputs``, and ``errors`` for each
            task. Includes ``rework_decisions`` when RCA-driven re-routing
            was triggered. Falls back to the ``generate_and_execute`` result
            shape when AgentGraph is unavailable.
        """
        try:
            from vetinari.agents.contracts import Plan as ExecutionPlan
            from vetinari.agents.contracts import Task as ContractsTask
            from vetinari.orchestration.agent_graph import get_agent_graph

            agent_graph = get_agent_graph()
            context = context or {}  # noqa: VET112 - empty fallback preserves optional request metadata contract

            # Intake classification for tier routing
            _ag_tier = None
            _ag_features = None
            try:
                from vetinari.orchestration.intake import get_request_intake

                _intake = get_request_intake()
                _ag_tier, _ag_features = _intake.classify_with_features(goal, context)
                context["intake_tier"] = _ag_tier.value
                context["intake_confidence"] = _ag_features.confidence
                context["intake_pattern_key"] = _ag_features.pattern_key
                logger.info(
                    "[TwoLayer] AgentGraph intake: tier=%s, confidence=%.2f", _ag_tier.value, _ag_features.confidence
                )
            except Exception:
                logger.warning("Intake classification unavailable for AgentGraph path")

            # Build RequestSpec for the AgentGraph path
            try:
                from vetinari.orchestration.intake import Tier as _SpecTier2
                from vetinari.orchestration.request_spec import get_spec_builder

                _spec_tier2 = _SpecTier2(context.get("intake_tier", "standard"))
                _ag_spec = get_spec_builder().build(goal=goal, tier=_spec_tier2, category=context.get("category", "code"))
                context["request_spec"] = _ag_spec.to_dict()
                logger.info("[TwoLayer] AgentGraph spec: complexity=%d", _ag_spec.estimated_complexity)
            except Exception:
                logger.warning("RequestSpec unavailable for AgentGraph path")

            # Complexity-based routing — skip/add pipeline stages
            try:
                from vetinari.routing import route_by_complexity

                _routing = route_by_complexity(goal)
                context["routing_decision"] = _routing.to_dict()
                context["complexity"] = _routing.complexity.value
                context["skip_stages"] = _routing.skip_stages
                context["add_stages"] = _routing.add_stages
                logger.info(
                    "[TwoLayer] ComplexityRouter: %s — skip=%s, add=%s",
                    _routing.complexity.value,
                    _routing.skip_stages,
                    _routing.add_stages,
                )
            except Exception:
                logger.warning("ComplexityRouter unavailable for AgentGraph path")

            # Generate plan using PlanGenerator
            graph = self.plan_generator.generate_plan(  # type: ignore[attr-defined]
                goal,
                constraints,
                max_depth=self._variant_manager.get_config().max_planning_depth,  # type: ignore[attr-defined]
            )

            # Check if Plan has follow_up_question — pause if so
            plan = ExecutionPlan.create_new(goal)
            if hasattr(graph, "follow_up_question") and graph.follow_up_question:
                try:
                    from vetinari.orchestration.intake import PipelinePaused

                    paused = PipelinePaused(
                        questions=[graph.follow_up_question],
                        pipeline_state={"goal": goal, "plan_id": graph.plan_id},
                        tier=context.get("intake_tier", "standard"),
                        goal=goal,
                    )
                    logger.info("[TwoLayer] Plan has follow_up_question — pausing pipeline")
                    return paused.to_dict()
                except Exception:
                    logger.warning("follow_up_question pause failed, proceeding")

            # Convert ExecutionGraph nodes -> contracts.Plan for AgentGraph
            for _node_id, node in graph.nodes.items():
                agent_type_str = node.input_data.get("assigned_agent", AgentType.WORKER.value).upper()
                try:
                    agent_type = AgentType[agent_type_str]
                except KeyError:
                    agent_type = AgentType.WORKER

                task = ContractsTask(
                    id=node.id,
                    description=node.description,
                    assigned_agent=agent_type,
                    dependencies=list(node.depends_on),
                    inputs=list(node.input_data.keys()) if node.input_data else [],
                    outputs=[],
                )

                # Attach context manifest (work instructions) to task
                try:
                    from vetinari.orchestration.task_manifest import get_manifest_builder

                    _mode = node.input_data.get("mode", "build")
                    _manifest = get_manifest_builder().build(
                        task_description=node.description,
                        agent_type=agent_type_str,
                        mode=_mode,
                    )
                    task.metadata["manifest"] = _manifest.to_dict()
                except Exception as _mf_err:
                    logger.warning("Failed to build manifest for task %s: %s", node.id, _mf_err)

                plan.tasks.append(task)

            # Apply ComplexityRouter stage filtering — skip/add stages
            _skip = set(context.get("skip_stages", []))
            if _skip:
                _before = len(plan.tasks)
                plan.tasks = [
                    t
                    for t in plan.tasks
                    if not any(
                        s in (t.metadata.get("manifest", {}).get("mode", "") or t.description.lower()) for s in _skip
                    )
                ]
                _after = len(plan.tasks)
                if _before != _after:
                    logger.info(
                        "[TwoLayer] ComplexityRouter skipped %d stages: %s",
                        _before - _after,
                        _skip,
                    )

            # Execute via AgentGraph
            results = agent_graph.execute_plan(plan)

            # RCA-driven corrective routing for failed tasks (Dept 7.8)
            _rework_decisions: dict[str, str] = {}
            _rework_outcomes: dict[str, dict[str, Any]] = {}
            for tid, r in results.items():
                if r.success or tid.startswith("_"):
                    continue
                rc_meta = r.metadata.get("root_cause") if r.metadata else None
                if rc_meta:
                    _rework_count = r.metadata.get("rework_count", 0) if r.metadata else 0
                    decision = self._handle_quality_rejection(  # type: ignore[attr-defined]
                        tid,
                        r.metadata,
                        _rework_count,
                    )
                    _rework_decisions[tid] = decision.value
                    logger.info(
                        "[TwoLayer] AgentGraph task %s failed — RCA routing: %s",
                        tid,
                        decision.value,
                    )
                    rework_result = self._execute_rework_decision(  # type: ignore[attr-defined]
                        decision=decision,
                        task_id=tid,
                        task_result=r.metadata,
                        graph=graph,
                    )
                    if rework_result:
                        _rework_outcomes[tid] = rework_result
                        if rework_result.get("outcome") == StatusEnum.COMPLETED.value:
                            results[tid] = (
                                results[tid]._replace(success=True)
                                if hasattr(results[tid], "_replace")
                                else results[tid]
                            )

            _ag_result = {
                "plan_id": graph.plan_id,
                "goal": goal,
                "backend": "agent_graph",
                "completed": sum(1 for r in results.values() if r.success),
                "failed": sum(1 for r in results.values() if not r.success),
                "outputs": {tid: r.output for tid, r in results.items()},
                "errors": {tid: r.errors for tid, r in results.items() if r.errors},
            }
            if _rework_decisions:
                _ag_result["rework_decisions"] = _rework_decisions
            if _rework_outcomes:
                _ag_result["rework_outcomes"] = _rework_outcomes

            # Record tier outcome for Thompson Sampling adaptive routing
            if _ag_tier is not None and _ag_features is not None:
                try:
                    from vetinari.learning.model_selector import get_thompson_selector

                    _ag_completed = _ag_result[StatusEnum.COMPLETED.value]
                    _ag_failed = _ag_result[StatusEnum.FAILED.value]
                    _ag_quality = _ag_completed / max(_ag_completed + _ag_failed, 1)

                    get_thompson_selector().update_tier(
                        pattern_key=_ag_features.pattern_key,
                        tier_used=_ag_tier.value,
                        quality_score=_ag_quality,
                        rework_count=0,
                    )
                    logger.info(
                        "[TwoLayer] AgentGraph Thompson outcome: tier=%s, quality=%.2f",
                        _ag_tier.value,
                        _ag_quality,
                    )
                except Exception:
                    logger.warning("Thompson tier outcome recording failed in AgentGraph path", exc_info=True)

            return _ag_result

        except Exception as e:
            logger.warning("[TwoLayer] AgentGraph execution failed, falling back: %s", e)
            return self.generate_and_execute(  # type: ignore[attr-defined]
                goal,
                constraints,
                context=context,
            )

    def _execute_via_agent_graph_or_fallback(
        self,
        graph: Any,
        task_handler: Any,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Route Stage 5 execution through AgentGraph or DurableExecutionEngine.

        Tries AgentGraph for parallel subtask execution when 2+ independent
        tasks exist and the AgentGraph is available.  Falls back to the
        DurableExecutionEngine handler path otherwise.

        Args:
            graph: The ExecutionGraph with model assignments applied.
            task_handler: Optional task handler callback.
            context: Pipeline context dict.

        Returns:
            Execution results dict with ``completed``, ``failed``, and
            ``task_results`` keys.
        """
        # Check if we have independent subtasks worth routing through AgentGraph.
        # When the caller supplies an explicit task_handler they want DurableExecutionEngine
        # semantics (durable checkpoints, lifecycle callbacks, handler-per-task).
        # AgentGraph bypasses all of that, so we only use it when no handler is provided.
        _has_parallel = len(graph.nodes) >= 2 and task_handler is None
        if _has_parallel:
            try:
                from vetinari.agents.contracts import Plan as ExecutionPlan
                from vetinari.agents.contracts import Task as ContractsTask
                from vetinari.orchestration.agent_graph import get_agent_graph

                agent_graph = get_agent_graph()

                # Convert ExecutionGraph nodes to a contracts.Plan
                plan = ExecutionPlan.create_new(context.get("goal", ""))
                for _node_id, node in graph.nodes.items():
                    _agent_type_str = node.input_data.get("assigned_agent", AgentType.WORKER.value).upper()
                    try:
                        _agent_type = AgentType[_agent_type_str]
                    except KeyError:
                        _agent_type = AgentType.WORKER

                    task = ContractsTask(
                        id=node.id,
                        description=node.description,
                        assigned_agent=_agent_type,
                        dependencies=list(node.depends_on),
                        inputs=list(node.input_data.keys()) if node.input_data else [],
                        outputs=[],
                    )
                    plan.tasks.append(task)

                ag_results = agent_graph.execute_plan(plan)

                # Convert AgentGraph results to DurableExecutionEngine result shape
                _completed = sum(1 for r in ag_results.values() if r.success)
                _failed = sum(1 for r in ag_results.values() if not r.success)
                return {
                    "plan_id": graph.plan_id,
                    "total_tasks": len(graph.nodes),
                    StatusEnum.COMPLETED.value: _completed,
                    StatusEnum.FAILED.value: _failed,
                    "task_results": {
                        tid: {
                            "output": r.output,
                            "status": StatusEnum.COMPLETED.value if r.success else StatusEnum.FAILED.value,
                            "errors": r.errors,
                            "metadata": r.metadata,
                        }
                        for tid, r in ag_results.items()
                    },
                    "backend": "agent_graph",
                }
            except Exception as e:
                logger.warning(
                    "[Pipeline] AgentGraph unavailable for parallel execution, "
                    "falling back to DurableExecutionEngine: %s",
                    e,
                )

        # Fallback: use DurableExecutionEngine with task handler
        effective_handler = task_handler or self._make_default_handler()  # type: ignore[attr-defined]
        return self.execution_engine.execute_plan(graph, effective_handler)  # type: ignore[attr-defined]
