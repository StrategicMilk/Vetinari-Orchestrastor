"""
TwoLayerOrchestrator and factory functions.

Implements the complete assembly-line orchestration pattern combining
Layer 1 (graph-based planning) with Layer 2 (durable execution).
"""

import logging
import time
import importlib
import os
from typing import List, Dict, Any, Optional, Callable

from vetinari.orchestration.types import TaskStatus, PlanStatus, TaskNode
from vetinari.orchestration.execution_graph import ExecutionGraph
from vetinari.orchestration.plan_generator import PlanGenerator
from vetinari.orchestration.durable_engine import DurableExecutionEngine
from vetinari.orchestration.architect_executor import (
    ArchitectExecutorPipeline, ArchitectPlan, PipelineConfig,
)

logger = logging.getLogger(__name__)


class TwoLayerOrchestrator:
    """
    Complete two-layer orchestration system implementing the assembly-line pattern.

    Assembly-line workflow:
      1. INPUT ANALYSIS   -- classify and assess complexity
      2. PLAN GENERATION  -- LLM-powered task decomposition
      3. TASK DECOMP      -- recursive breakdown to atomic tasks
      4. MODEL ASSIGNMENT -- intelligent model routing per task
      5. PARALLEL EXEC    -- DAG-scheduled execution
      6. OUTPUT REVIEW    -- consistency and quality check
      7. FINAL ASSEMBLY   -- synthesize final output

    Combines:
    - Layer 1: Graph-Based Planning (PlanGenerator)
    - Layer 2: Durable Execution (DurableExecutionEngine)
    """

    def __init__(self,
                 checkpoint_dir: str = None,
                 max_concurrent: int = 4,
                 model_router=None,
                 agent_context: Dict[str, Any] = None):
        self.plan_generator = PlanGenerator(model_router)
        self.execution_engine = DurableExecutionEngine(
            checkpoint_dir=checkpoint_dir,
            max_concurrent=max_concurrent
        )
        self.model_router = model_router
        self.agent_context: Dict[str, Any] = agent_context or {}
        self._agents: Dict[str, Any] = {}

        logger.info("TwoLayerOrchestrator initialized (assembly-line mode)")

    def set_task_handlers(self, handlers: Dict[str, Callable]):
        """Set task handlers for execution."""
        for task_type, handler in handlers.items():
            self.execution_engine.register_handler(task_type, handler)

    def set_agent_context(self, context: Dict[str, Any]) -> None:
        """Set the shared agent context (adapter_manager, web_search, etc.)."""
        self.agent_context = context
        # Re-initialize any cached agents
        self._agents.clear()

    # ------------------------------------------------------------------
    # Architect-Executor 2-stage pipeline support
    # ------------------------------------------------------------------

    def _get_architect_pipeline(self) -> Optional[ArchitectExecutorPipeline]:
        """Get the architect-executor pipeline if configured and enabled."""
        ae_config = self.agent_context.get("architect_executor", {})
        if not ae_config.get("enabled", False):
            return None
        return ArchitectExecutorPipeline(
            architect_model=ae_config.get("architect_model"),
            executor_model=ae_config.get("executor_model"),
            config=ae_config,
        )

    def generate_and_execute_architect(
        self,
        goal: str,
        context: Dict[str, Any] = None,
        executor_fn: Callable = None,
    ) -> Dict[str, Any]:
        """
        Run the 2-stage architect-executor pipeline.

        Falls back to the standard generate_and_execute if the architect
        pipeline is not enabled or fails (when fallback_to_single is True).

        Args:
            goal: The goal to achieve.
            context: Additional context (files, tech_stack, etc.).
            executor_fn: Optional custom executor function.

        Returns:
            Dict with plan, results, and success flag.
        """
        context = context or {}
        pipeline = self._get_architect_pipeline()

        if pipeline is None:
            logger.info("[Architect] Pipeline not enabled, using standard pipeline")
            return self.generate_and_execute(goal=goal, context=context, task_handler=executor_fn)

        try:
            result = pipeline.run(goal=goal, context=context, executor_fn=executor_fn)
            if result.get("skipped"):
                logger.info("[Architect] Pipeline disabled at runtime, falling back")
                return self.generate_and_execute(goal=goal, context=context, task_handler=executor_fn)
            return result
        except Exception as e:
            logger.warning(f"[Architect] Pipeline failed: {e}")
            ae_config = self.agent_context.get("architect_executor", {})
            if ae_config.get("fallback_to_single", True):
                logger.info("[Architect] Falling back to standard pipeline")
                return self.generate_and_execute(goal=goal, context=context, task_handler=executor_fn)
            raise

    # Complete mapping from agent type string to (module, getter_function)
    _AGENT_MODULE_MAP = {
        "PLANNER":                  ("vetinari.agents.planner_agent",               "get_planner_agent"),
        "EXPLORER":                 ("vetinari.agents.explorer_agent",              "get_explorer_agent"),
        "ORACLE":                   ("vetinari.agents.oracle_agent",                "get_oracle_agent"),
        "LIBRARIAN":                ("vetinari.agents.librarian_agent",             "get_librarian_agent"),
        "RESEARCHER":               ("vetinari.agents.researcher_agent",            "get_researcher_agent"),
        "EVALUATOR":                ("vetinari.agents.evaluator_agent",             "get_evaluator_agent"),
        "SYNTHESIZER":              ("vetinari.agents.synthesizer_agent",           "get_synthesizer_agent"),
        "BUILDER":                  ("vetinari.agents.builder_agent",               "get_builder_agent"),
        "UI_PLANNER":               ("vetinari.agents.ui_planner_agent",            "get_ui_planner_agent"),
        "SECURITY_AUDITOR":         ("vetinari.agents.security_auditor_agent",      "get_security_auditor_agent"),
        "DATA_ENGINEER":            ("vetinari.agents.data_engineer_agent",         "get_data_engineer_agent"),
        "DOCUMENTATION_AGENT":      ("vetinari.agents.documentation_agent",         "get_documentation_agent"),
        "COST_PLANNER":             ("vetinari.agents.cost_planner_agent",          "get_cost_planner_agent"),
        "TEST_AUTOMATION":          ("vetinari.agents.test_automation_agent",       "get_test_automation_agent"),
        "EXPERIMENTATION_MANAGER":  ("vetinari.agents.experimentation_manager_agent", "get_experimentation_manager_agent"),
        "IMPROVEMENT":              ("vetinari.agents.improvement_agent",           "get_improvement_agent"),
        "USER_INTERACTION":         ("vetinari.agents.user_interaction_agent",      "get_user_interaction_agent"),
        "DEVOPS":                   ("vetinari.agents.devops_agent",                "get_devops_agent"),
        "VERSION_CONTROL":          ("vetinari.agents.version_control_agent",       "get_version_control_agent"),
        "ERROR_RECOVERY":           ("vetinari.agents.error_recovery_agent",        "get_error_recovery_agent"),
        "CONTEXT_MANAGER":          ("vetinari.agents.context_manager_agent",       "get_context_manager_agent"),
        "IMAGE_GENERATOR":          ("vetinari.agents.image_generator_agent",       "get_image_generator_agent"),
    }

    def _get_agent(self, agent_type_str: str):
        """Get or create an agent by type string, initialized with context."""
        key = agent_type_str.upper()
        if key in self._agents:
            return self._agents[key]
        if key not in self._AGENT_MODULE_MAP:
            logger.debug(f"No agent module registered for type: {key}")
            return None
        try:
            mod_path, fn_name = self._AGENT_MODULE_MAP[key]
            mod = importlib.import_module(mod_path)
            getter = getattr(mod, fn_name, None)
            if getter is None:
                return None
            agent = getter()
            if self.agent_context:
                agent.initialize(self.agent_context)
            self._agents[key] = agent
            return agent
        except Exception as e:
            logger.warning(f"Could not get agent '{key}': {e}")
            return None

    def _route_model_for_task(self, task: "TaskNode") -> str:
        """Select the best model for a task using dynamic model routing."""
        if self.model_router is None:
            try:
                from vetinari.dynamic_model_router import get_model_router
                self.model_router = get_model_router()
            except Exception:
                return "default"
        try:
            from vetinari.dynamic_model_router import TaskType
            task_type_map = {
                "analysis": TaskType.ANALYSIS,
                "implementation": TaskType.CODING,
                "testing": TaskType.TESTING,
                "research": TaskType.ANALYSIS,
                "documentation": TaskType.DOCUMENTATION,
                "verification": TaskType.CODE_REVIEW,
            }
            t_type = task_type_map.get(task.task_type.lower(), TaskType.GENERAL)
            selection = self.model_router.select_model(t_type)
            if selection and selection.model:
                return selection.model.id
        except Exception as e:
            logger.debug(f"Model routing failed for task {task.id}: {e}")
        return "default"

    def generate_and_execute(self,
                             goal: str,
                             constraints: Dict[str, Any] = None,
                             task_handler: Callable = None,
                             context: Dict[str, Any] = None,
                             project_id: Optional[str] = None,
                             model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Full assembly-line pipeline: analyze -> plan -> decompose ->
        assign models -> execute -> review -> assemble.

        Args:
            goal: The goal to achieve
            constraints: Optional constraints (budget, time, etc.)
            task_handler: Optional custom task handler; defaults to
                          the registered agent-based handler
            context: Additional context from the project intake form
                     (required_features, things_to_avoid, tech_stack, etc.)
            project_id: Optional project ID for rules injection
            model_id: Optional model ID for rules injection

        Returns:
            Dict with keys: plan_id, completed, failed, outputs,
                            review_result, final_output, stages
        """
        stages: Dict[str, Any] = {}
        start_time = time.time()
        context = context or {}

        # Enrich goal with intake form context
        enriched_goal = goal
        if context.get("required_features"):
            enriched_goal += "\n\nRequired features:\n" + "\n".join(
                f"- {f}" for f in context["required_features"]
            )
        if context.get("things_to_avoid"):
            enriched_goal += "\n\nDo NOT include:\n" + "\n".join(
                f"- {a}" for a in context["things_to_avoid"]
            )
        if context.get("tech_stack"):
            enriched_goal += f"\n\nTech stack: {context['tech_stack']}"
        if context.get("priority"):
            enriched_goal += f"\n\nPriority: {context['priority']}"

        # Inject rules into context for agent prompts
        try:
            from vetinari.rules_manager import get_rules_manager
            rm = get_rules_manager()
            context["_rules_prefix"] = rm.build_system_prompt_prefix(
                project_id=project_id, model_id=model_id
            )
        except Exception:
            pass

        # ----------------------------------------------------------------
        # STAGE 1: Input Analysis
        # ----------------------------------------------------------------
        logger.info(f"[Pipeline] Stage 1: Input Analysis for goal: {goal[:80]}")
        analysis = self._analyze_input(enriched_goal, constraints or {})
        stages["input_analysis"] = analysis

        # ----------------------------------------------------------------
        # STAGE 2 & 3: Plan Generation + Task Decomposition
        # ----------------------------------------------------------------
        logger.info("[Pipeline] Stage 2-3: Plan Generation & Decomposition")
        graph = self.plan_generator.generate_plan(enriched_goal, constraints)
        stages["plan"] = {"plan_id": graph.plan_id, "tasks": len(graph.nodes)}

        # ----------------------------------------------------------------
        # STAGE 4: Model Assignment
        # ----------------------------------------------------------------
        logger.info("[Pipeline] Stage 4: Model Assignment")
        for node in graph.nodes.values():
            assigned_model = self._route_model_for_task(node)
            node.input_data["assigned_model"] = assigned_model
            logger.debug(f"  Task {node.id} ({node.task_type}) -> {assigned_model}")
        stages["model_assignment"] = {nid: n.input_data.get("assigned_model") for nid, n in graph.nodes.items()}

        # ----------------------------------------------------------------
        # STAGE 5: Parallel Execution
        # ----------------------------------------------------------------
        logger.info("[Pipeline] Stage 5: Parallel Execution")
        effective_handler = task_handler or self._make_default_handler()
        exec_results = self.execution_engine.execute_plan(graph, effective_handler)
        stages["execution"] = exec_results

        # ----------------------------------------------------------------
        # STAGE 6: Output Review
        # ----------------------------------------------------------------
        logger.info("[Pipeline] Stage 6: Output Review")
        review_result = self._review_outputs(exec_results, goal)
        stages["review"] = review_result

        # ----------------------------------------------------------------
        # STAGE 7: Final Assembly
        # ----------------------------------------------------------------
        logger.info("[Pipeline] Stage 7: Final Assembly")
        final_output = self._assemble_final_output(exec_results, review_result, goal)
        stages["final_assembly"] = {"output_length": len(str(final_output))}

        total_time = int((time.time() - start_time) * 1000)

        # I4: Wire execution results to feedback loop for learning
        try:
            from vetinari.learning.feedback_loop import get_feedback_loop
            feedback = get_feedback_loop()
            completed = exec_results.get("completed", 0)
            failed = exec_results.get("failed", 0)
            total_tasks = completed + failed
            quality = completed / max(total_tasks, 1)
            feedback.record_outcome(
                task_id=graph.plan_id,
                model_id="orchestrator",
                task_type="plan_execution",
                quality_score=quality,
                latency_ms=total_time,
                success=failed == 0,
            )
        except Exception as e:
            logger.warning(f"[Pipeline] Feedback loop recording failed: {e}")

        # I6: Periodically check anomaly detector for execution anomalies
        try:
            from vetinari.analytics.anomaly import get_anomaly_detector
            detector = get_anomaly_detector()
            detector.detect("plan.execution_time_ms", float(total_time))
            detector.detect("plan.failure_rate", float(failed) / max(total_tasks, 1) if total_tasks > 0 else 0.0)
        except Exception as e:
            logger.warning(f"[Pipeline] Anomaly detection check failed: {e}")

        return {
            "plan_id": graph.plan_id,
            "goal": goal,
            "completed": exec_results.get("completed", 0),
            "failed": exec_results.get("failed", 0),
            "outputs": exec_results.get("task_results", {}),
            "review_result": review_result,
            "final_output": final_output,
            "stages": stages,
            "total_time_ms": total_time,
        }

    def _analyze_input(self, goal: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Classify the input goal and estimate complexity."""
        result = {
            "goal": goal,
            "estimated_complexity": "medium",
            "domain": "general",
            "needs_research": False,
            "needs_code": False,
            "needs_ui": False,
            "intent": "unknown",
            "suggested_depth": 3,
            "scope": "single",
        }
        g = goal.lower()
        result["needs_code"] = any(k in g for k in ["code", "implement", "build", "create", "program", "software"])
        result["needs_research"] = any(k in g for k in ["research", "analyze", "investigate", "study"])
        result["needs_ui"] = any(k in g for k in ["ui", "frontend", "interface", "web app", "dashboard"])
        result["domain"] = ("coding" if result["needs_code"] else
                            "research" if result["needs_research"] else "general")
        word_count = len(goal.split())
        result["estimated_complexity"] = "simple" if word_count < 10 else "complex" if word_count > 30 else "medium"

        # Classify intent
        if any(k in g for k in ["fix", "bug", "error", "broken", "crash"]):
            result["intent"] = "bugfix"
        elif any(k in g for k in ["refactor", "clean", "simplify", "reorganize"]):
            result["intent"] = "refactor"
        elif any(k in g for k in ["add", "new", "create", "implement", "build"]):
            result["intent"] = "feature"
        elif any(k in g for k in ["document", "explain", "describe", "write docs"]):
            result["intent"] = "documentation"
        elif any(k in g for k in ["research", "analyze", "investigate", "compare"]):
            result["intent"] = "research"
        elif any(k in g for k in ["test", "validate", "verify", "check"]):
            result["intent"] = "testing"
        else:
            result["intent"] = "general"

        # Suggest decomposition depth based on complexity
        complexity = result["estimated_complexity"]
        result["suggested_depth"] = {"simple": 2, "medium": 3, "complex": 5}.get(complexity, 3)

        # Determine scope
        if any(k in g for k in ["full", "entire", "complete", "all", "whole system", "end-to-end"]):
            result["scope"] = "system"
        elif any(k in g for k in ["module", "component", "service", "package"]):
            result["scope"] = "module"
        else:
            result["scope"] = "single"

        return result

    def _make_default_handler(self) -> Callable:
        """Create a default task handler that uses agent inference with token optimisation."""
        def handle_task(task: "TaskNode") -> Dict[str, Any]:
            try:
                # Determine if target model is a cloud model
                assigned_model = task.input_data.get("assigned_model", "default")
                is_cloud = not any(
                    x in assigned_model.lower()
                    for x in ["qwen", "llama", "mistral", "gemma", "phi", "local", "lm_studio", "default"]
                )

                # Apply token optimisation
                task_context = " ".join(
                    str(v)[:500] for v in task.input_data.values() if v
                ) if task.input_data else ""

                try:
                    from vetinari.token_optimizer import get_token_optimizer
                    optimizer = get_token_optimizer()
                    opt_result = optimizer.prepare_prompt(
                        prompt=task.description,
                        context=task_context,
                        task_type=task.task_type or "general",
                        task_description=task.description,
                        is_cloud_model=is_cloud,
                        task_id=task.id,
                    )
                    optimised_prompt = opt_result["prompt"]
                    max_tokens = opt_result["max_tokens"]
                    temperature = opt_result["temperature"]
                except Exception:
                    optimised_prompt = task.description
                    max_tokens = 2048
                    temperature = 0.3

                # Resolve sampling profile for this task + model
                sampling_kwargs = {}
                try:
                    from vetinari.sampling_profile import SamplingProfileManager
                    spm = SamplingProfileManager()
                    profile = spm.resolve(
                        task_type=task.task_type or "general",
                        model_id=assigned_model,
                    )
                    sampling_kwargs = {
                        "temperature": profile.temperature,
                        "top_p": profile.top_p,
                        "top_k": profile.top_k,
                        "min_p": profile.min_p,
                        "repeat_penalty": profile.repeat_penalty,
                        "presence_penalty": profile.presence_penalty,
                        "frequency_penalty": profile.frequency_penalty,
                    }
                    # Token optimizer temperature takes precedence if set
                    if temperature != 0.3:  # non-default from optimizer
                        sampling_kwargs["temperature"] = temperature
                except Exception:
                    pass

                adapter_manager = self.agent_context.get("adapter_manager")
                if adapter_manager:
                    try:
                        from vetinari.adapters.base import InferenceRequest
                        req = InferenceRequest(
                            model_id=assigned_model,
                            prompt=optimised_prompt,
                            system_prompt=f"Execute this {task.task_type or 'general'} task precisely and completely.",
                            max_tokens=max_tokens,
                            temperature=sampling_kwargs.get("temperature", temperature),
                            top_p=sampling_kwargs.get("top_p", 0.9),
                            top_k=sampling_kwargs.get("top_k", 40),
                            min_p=sampling_kwargs.get("min_p"),
                            repeat_penalty=sampling_kwargs.get("repeat_penalty"),
                            presence_penalty=sampling_kwargs.get("presence_penalty"),
                            frequency_penalty=sampling_kwargs.get("frequency_penalty"),
                        )
                        resp = adapter_manager.infer(req)
                        if resp.status == "ok":
                            return {"result": resp.output, "status": "ok", "task_id": task.id}
                    except Exception as e:
                        logger.warning(f"Adapter inference failed for task {task.id}: {e}")

                # Fallback: use LM Studio adapter directly
                from vetinari.lmstudio_adapter import LMStudioAdapter
                host = os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")
                adapter = LMStudioAdapter(host=host)
                result = adapter.chat(
                    model_id=assigned_model,
                    system_prompt=f"You are executing a {task.task_type or 'general'} task.",
                    input_text=optimised_prompt,
                )
                return {"result": result.get("output", ""), "status": "ok", "task_id": task.id}
            except Exception as e:
                logger.error(f"Task handler failed for {task.id}: {e}")
                return {"result": "", "status": "error", "error": str(e), "task_id": task.id}

        return handle_task

    def _review_outputs(self, exec_results: Dict[str, Any], goal: str) -> Dict[str, Any]:
        """Use EvaluatorAgent to review execution outputs for quality, with loop detection."""
        # Loop detection: check for repeated failure patterns
        loop_detected = False
        task_results = exec_results.get("task_results", {})
        if task_results:
            error_signatures = []
            for tid, result in task_results.items():
                if isinstance(result, dict) and result.get("status") == "error":
                    sig = result.get("error", "")[:80]
                    error_signatures.append(sig)
            # Detect loops: same error appearing 3+ times
            if error_signatures:
                from collections import Counter
                counts = Counter(error_signatures)
                repeated = {sig: cnt for sig, cnt in counts.items() if cnt >= 3}
                if repeated:
                    loop_detected = True
                    logger.warning(
                        "[Pipeline] Loop detected in review: %d repeated error patterns",
                        len(repeated),
                    )

        try:
            evaluator = self._get_agent("EVALUATOR")
            if evaluator:
                from vetinari.agents.contracts import AgentTask, AgentType
                artifacts = [str(v) for v in task_results.values() if v]
                eval_task = AgentTask(
                    task_id="review-0",
                    agent_type=AgentType.EVALUATOR,
                    description=f"Review outputs for goal: {goal}",
                    context={"artifacts": artifacts[:5], "focus": "all"},
                )
                result = evaluator.execute(eval_task)
                if result.success:
                    review = result.output if isinstance(result.output, dict) else {"verdict": "pass", "details": result.output}
                    review["loop_detected"] = loop_detected
                    return review
        except Exception as e:
            logger.warning(f"Output review failed: {e}")
        return {
            "verdict": "inconclusive",
            "quality_score": 0.5,
            "summary": "Review skipped (evaluator unavailable)",
            "loop_detected": loop_detected,
        }

    def _assemble_final_output(self, exec_results: Dict[str, Any],
                               review_result: Dict[str, Any], goal: str) -> str:
        """Use SynthesizerAgent to assemble a final coherent output."""
        try:
            synthesizer = self._get_agent("SYNTHESIZER")
            if synthesizer:
                from vetinari.agents.contracts import AgentTask, AgentType
                task_results = exec_results.get("task_results", {})
                sources = [
                    {"agent": k, "artifact": str(v)[:500]}
                    for k, v in task_results.items() if v
                ]
                sources.append({"agent": "review", "artifact": str(review_result)[:200]})
                synth_task = AgentTask(
                    task_id="assemble-0",
                    agent_type=AgentType.SYNTHESIZER,
                    description=f"Final assembly for goal: {goal}",
                    context={"sources": sources, "type": "final_report"},
                )
                result = synthesizer.execute(synth_task)
                if result.success and result.output:
                    return result.output.get("synthesized_artifact", str(result.output))
        except Exception as e:
            logger.warning(f"Final assembly failed: {e}")

        # Fallback: join task_results
        task_results = exec_results.get("task_results", {})
        parts = [f"# Task {k}\n{v}" for k, v in task_results.items() if v]
        return "\n\n".join(parts) if parts else f"Completed: {goal}"

    def generate_plan_only(self,
                           goal: str,
                           constraints: Dict[str, Any] = None) -> "ExecutionGraph":
        """Generate a plan without executing."""
        return self.plan_generator.generate_plan(goal, constraints)

    def execute_plan(self,
                     graph: "ExecutionGraph",
                     task_handler: Callable = None) -> Dict[str, Any]:
        """Execute an existing plan."""
        return self.execution_engine.execute_plan(graph, task_handler)

    def recover_plan(self, plan_id: str) -> Dict[str, Any]:
        """Recover and continue a plan from checkpoint."""
        return self.execution_engine.recover_execution(plan_id)

    def get_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a plan."""
        return self.execution_engine.get_execution_status(plan_id)

    def list_checkpoints(self) -> List[str]:
        """List all available plan checkpoints."""
        return self.execution_engine.list_checkpoints()


# Global orchestrator
_two_layer_orchestrator: Optional[TwoLayerOrchestrator] = None


def get_two_layer_orchestrator() -> TwoLayerOrchestrator:
    """Get or create the global two-layer orchestrator."""
    global _two_layer_orchestrator
    if _two_layer_orchestrator is None:
        _two_layer_orchestrator = TwoLayerOrchestrator()
    return _two_layer_orchestrator


def init_two_layer_orchestrator(checkpoint_dir: str = None, **kwargs) -> TwoLayerOrchestrator:
    """Initialize a new two-layer orchestrator."""
    global _two_layer_orchestrator
    _two_layer_orchestrator = TwoLayerOrchestrator(checkpoint_dir=checkpoint_dir, **kwargs)
    return _two_layer_orchestrator
