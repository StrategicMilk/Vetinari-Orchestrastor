"""
Two-Layer Orchestrator — the main entry point combining planning and execution.

Implements the assembly-line pattern:
  1. INPUT ANALYSIS   — classify and assess complexity
  2. PLAN GENERATION  — LLM-powered task decomposition
  3. TASK DECOMP      — recursive breakdown to atomic tasks
  4. MODEL ASSIGNMENT — intelligent model routing per task
  5. PARALLEL EXEC    — DAG-scheduled execution
  6. OUTPUT REVIEW    — consistency and quality check
  7. FINAL ASSEMBLY   — synthesize final output
"""

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from vetinari.orchestration.durable_execution import DurableExecutionEngine
from vetinari.orchestration.execution_graph import ExecutionGraph, TaskNode
from vetinari.orchestration.plan_generator import PlanGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Goal categorisation (Phase 7.6 — Oh-My-OpenCode task category pattern)
# ---------------------------------------------------------------------------

# Keyword lists for the 9 goal categories.  Order matters: more specific
# categories should be checked first (e.g. "security audit" before "audit").

_GOAL_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "security": ["security", "audit", "vulnerability", "pentest", "cve", "owasp", "exploit"],
    "devops": ["deploy", "ci/cd", "docker", "kubernetes", "pipeline", "devops", "terraform", "helm"],
    "image": ["logo", "icon", "mockup", "diagram", "image", "illustration", "screenshot"],
    "creative": ["story", "poem", "fiction", "narrative", "campaign", "creative writ", "novel"],
    "data": ["database", "schema", "migration", "etl", "sql", "nosql", "data model"],
    "ui": ["ui", "ux", "frontend", "design", "wireframe", "layout", "responsive", "css"],
    "docs": ["document", "readme", "api docs", "manual", "changelog", "guide", "tutorial"],
    "research": ["research", "analyze", "investigate", "study", "explore", "survey", "compare"],
    "code": ["code", "implement", "build", "develop", "fix", "refactor", "function", "class", "module"],
}

# Maps GoalCategory value -> (primary agent type, default mode, model tier hint)
# v0.4.0: Updated to use 6 consolidated agent types
_GOAL_ROUTING_TABLE: Dict[str, tuple] = {
    "code":     ("BUILDER",                "build",            "coder"),
    "research": ("CONSOLIDATED_RESEARCHER", "domain_research", "general"),
    "docs":     ("OPERATIONS",             "documentation",    "general"),
    "creative": ("OPERATIONS",             "creative_writing", "general"),
    "security": ("QUALITY",                "security_audit",   "coder"),
    "data":     ("CONSOLIDATED_RESEARCHER", "database",        "general"),
    "devops":   ("CONSOLIDATED_RESEARCHER", "devops",          "coder"),
    "ui":       ("CONSOLIDATED_RESEARCHER", "ui_design",       "vision"),
    "image":    ("BUILDER",                "image_generation", "general"),
    "general":  ("PLANNER",                "plan",             "general"),
}


def classify_goal(goal: str) -> str:
    """Classify a goal string into one of the 9 categories.

    Returns the GoalCategory *value* string (e.g. ``"code"``, ``"security"``).
    Falls back to ``"general"`` when no keywords match.
    """
    goal_lower = goal.lower()
    for category, keywords in _GOAL_CATEGORY_KEYWORDS.items():
        if any(kw in goal_lower for kw in keywords):
            return category
    return "general"


def get_goal_routing(goal: str) -> tuple:
    """Return ``(agent_type, mode, model_tier)`` for a goal string."""
    category = classify_goal(goal)
    return _GOAL_ROUTING_TABLE.get(category, _GOAL_ROUTING_TABLE["general"])


class TwoLayerOrchestrator:
    """
    Complete two-layer orchestration system implementing the assembly-line pattern.

    Combines:
    - Layer 1: Graph-Based Planning (PlanGenerator)
    - Layer 2: Durable Execution (DurableExecutionEngine)
    """

    def __init__(
        self,
        checkpoint_dir: str = None,
        max_concurrent: int = 4,
        model_router=None,
        agent_context: Dict[str, Any] = None,
    ):
        self.plan_generator = PlanGenerator(model_router)
        self.execution_engine = DurableExecutionEngine(
            checkpoint_dir=checkpoint_dir,
            max_concurrent=max_concurrent,
        )
        self.model_router = model_router
        self.agent_context: Dict[str, Any] = agent_context or {}
        self._agents: Dict[str, Any] = {}

        logger.info("TwoLayerOrchestrator initialized (assembly-line mode)")

    def set_task_handlers(self, handlers: Dict[str, Callable]) -> None:
        """Set task handlers for execution."""
        for task_type, handler in handlers.items():
            self.execution_engine.register_handler(task_type, handler)

    def set_agent_context(self, context: Dict[str, Any]) -> None:
        """Set the shared agent context (adapter_manager, web_search, etc.)."""
        self.agent_context = context
        self._agents.clear()

    # v0.4.0: 6 consolidated agents + legacy aliases via compat shim
    _AGENT_MODULE_MAP = {
        # ── 6 active agents ──
        "PLANNER": ("vetinari.agents.planner_agent", "get_planner_agent"),
        "BUILDER": ("vetinari.agents.builder_agent", "get_builder_agent"),
        "CONSOLIDATED_RESEARCHER": (
            "vetinari.agents.consolidated.researcher_agent",
            "get_consolidated_researcher_agent",
        ),
        "CONSOLIDATED_ORACLE": (
            "vetinari.agents.consolidated.oracle_agent",
            "get_consolidated_oracle_agent",
        ),
        "QUALITY": (
            "vetinari.agents.consolidated.quality_agent",
            "get_quality_agent",
        ),
        "OPERATIONS": (
            "vetinari.agents.consolidated.operations_agent",
            "get_operations_agent",
        ),
        # ── Legacy aliases (routed via compat shim) ──
        "EXPLORER": ("vetinari.agents.compat", "get_explorer_agent"),
        "LIBRARIAN": ("vetinari.agents.compat", "get_librarian_agent"),
        "RESEARCHER": ("vetinari.agents.compat", "get_researcher_agent"),
        "ORACLE": ("vetinari.agents.compat", "get_oracle_agent"),
        "EVALUATOR": ("vetinari.agents.compat", "get_evaluator_agent"),
        "SYNTHESIZER": ("vetinari.agents.compat", "get_synthesizer_agent"),
        "UI_PLANNER": ("vetinari.agents.compat", "get_ui_planner_agent"),
        "SECURITY_AUDITOR": ("vetinari.agents.compat", "get_security_auditor_agent"),
        "DATA_ENGINEER": ("vetinari.agents.compat", "get_data_engineer_agent"),
        "DOCUMENTATION_AGENT": ("vetinari.agents.compat", "get_documentation_agent"),
        "COST_PLANNER": ("vetinari.agents.compat", "get_cost_planner_agent"),
        "TEST_AUTOMATION": ("vetinari.agents.compat", "get_test_automation_agent"),
        "EXPERIMENTATION_MANAGER": ("vetinari.agents.compat", "get_experimentation_manager_agent"),
        "IMPROVEMENT": ("vetinari.agents.compat", "get_improvement_agent"),
        "USER_INTERACTION": ("vetinari.agents.compat", "get_user_interaction_agent"),
        "DEVOPS": ("vetinari.agents.compat", "get_devops_agent"),
        "VERSION_CONTROL": ("vetinari.agents.compat", "get_version_control_agent"),
        "ERROR_RECOVERY": ("vetinari.agents.compat", "get_error_recovery_agent"),
        "CONTEXT_MANAGER": ("vetinari.agents.compat", "get_context_manager_agent"),
        "IMAGE_GENERATOR": ("vetinari.agents.compat", "get_image_generator_agent"),
    }

    def _get_agent(self, agent_type_str: str):
        """Get or create an agent by type string, initialized with context."""
        key = agent_type_str.upper()
        if key in self._agents:
            return self._agents[key]
        if key not in self._AGENT_MODULE_MAP:
            logger.debug("No agent module registered for type: %s", key)
            return None
        try:
            import importlib

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
            logger.warning("Could not get agent '%s': %s", key, e)
            return None

    def _route_model_for_task(self, task: TaskNode) -> str:
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
                # Phase 7 additions
                "creative_writing": TaskType.CREATIVE_WRITING,
                "security_audit": TaskType.SECURITY_AUDIT,
                "devops": TaskType.DEVOPS,
                "image_generation": TaskType.IMAGE_GENERATION,
                "cost_analysis": TaskType.COST_ANALYSIS,
                "specification": TaskType.SPECIFICATION,
                "creative": TaskType.CREATIVE,
                "security": TaskType.SECURITY_AUDIT,
            }
            t_type = task_type_map.get(task.task_type.lower(), TaskType.GENERAL)
            selection = self.model_router.select_model(t_type)
            if selection and selection.model:
                return selection.model.id
        except Exception as e:
            logger.debug("Model routing failed for task %s: %s", task.id, e)
        return "default"

    def generate_and_execute(
        self,
        goal: str,
        constraints: Dict[str, Any] = None,
        task_handler: Callable = None,
        context: Dict[str, Any] = None,
        project_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full assembly-line pipeline: analyze → plan → decompose →
        assign models → execute → review → assemble.
        """
        stages: Dict[str, Any] = {}
        start_time = time.time()
        context = context or {}

        # Enrich goal with intake form context
        enriched_goal = self._enrich_goal(goal, context)

        # Inject rules into context for agent prompts
        try:
            from vetinari.rules_manager import get_rules_manager

            rm = get_rules_manager()
            context["_rules_prefix"] = rm.build_system_prompt_prefix(
                project_id=project_id, model_id=model_id
            )
        except Exception:
            logger.debug("Failed to inject rules prefix into orchestration context", exc_info=True)

        # STAGE 1: Input Analysis
        logger.info("[Pipeline] Stage 1: Input Analysis for goal: %s", goal[:80])
        analysis = self._analyze_input(enriched_goal, constraints or {})
        stages["input_analysis"] = analysis

        # STAGE 2 & 3: Plan Generation + Task Decomposition
        logger.info("[Pipeline] Stage 2-3: Plan Generation & Decomposition")
        graph = self.plan_generator.generate_plan(enriched_goal, constraints)
        stages["plan"] = {"plan_id": graph.plan_id, "tasks": len(graph.nodes)}

        # ── C2: Stage-boundary validation (plan → model assignment) ───
        plan_valid, plan_issues = self._validate_stage_boundary(
            "plan", stages["plan"], min_keys=["plan_id", "tasks"],
        )
        if not plan_valid:
            logger.warning("[Pipeline] Plan validation failed: %s", plan_issues)
            return {
                "plan_id": graph.plan_id, "goal": goal, "completed": 0,
                "failed": 1, "error": f"Plan validation failed: {plan_issues}",
                "stages": stages, "total_time_ms": int((time.time() - start_time) * 1000),
            }

        # STAGE 4: Model Assignment
        logger.info("[Pipeline] Stage 4: Model Assignment")
        for node in graph.nodes.values():
            assigned = self._route_model_for_task(node)
            node.input_data["assigned_model"] = assigned
            logger.debug("  Task %s (%s) -> %s", node.id, node.task_type, assigned)
        stages["model_assignment"] = {
            nid: n.input_data.get("assigned_model") for nid, n in graph.nodes.items()
        }

        # STAGE 5: Parallel Execution
        logger.info("[Pipeline] Stage 5: Parallel Execution")
        effective_handler = task_handler or self._make_default_handler()
        exec_results = self.execution_engine.execute_plan(graph, effective_handler)
        stages["execution"] = exec_results

        # ── C2: Stage-boundary validation (execution → review) ────────
        exec_valid, exec_issues = self._validate_stage_boundary(
            "execution", exec_results, min_keys=["completed"],
        )
        if not exec_valid:
            logger.warning("[Pipeline] Execution validation failed: %s", exec_issues)

        # STAGE 6: Output Review
        logger.info("[Pipeline] Stage 6: Output Review")
        review_result = self._review_outputs(exec_results, goal)
        stages["review"] = review_result

        # STAGE 7: Final Assembly
        logger.info("[Pipeline] Stage 7: Final Assembly")
        final_output = self._assemble_final_output(exec_results, review_result, goal)
        stages["final_assembly"] = {"output_length": len(str(final_output))}

        total_time = int((time.time() - start_time) * 1000)
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

    # ── C2: Stage-boundary validation helper ─────────────────────────

    @staticmethod
    def _validate_stage_boundary(
        stage_name: str,
        stage_output: Any,
        min_keys: List[str] = None,
    ) -> tuple:
        """Validate the output of a pipeline stage before passing to the next.

        Returns ``(is_valid, issues_list)``.
        """
        issues: List[str] = []

        if stage_output is None:
            issues.append(f"Stage '{stage_name}' produced None output")
            return False, issues

        if isinstance(stage_output, dict):
            if min_keys:
                missing = [k for k in min_keys if k not in stage_output]
                if missing:
                    issues.append(
                        f"Stage '{stage_name}' missing required keys: {missing}"
                    )
            # Check for error indicators
            if stage_output.get("error"):
                issues.append(
                    f"Stage '{stage_name}' has error: {stage_output['error']}"
                )
            if stage_output.get("failed", 0) > 0 and stage_output.get("completed", 0) == 0:
                issues.append(
                    f"Stage '{stage_name}': all tasks failed "
                    f"({stage_output['failed']} failures, 0 completed)"
                )

        return (len(issues) == 0, issues)

    @staticmethod
    def _enrich_goal(goal: str, context: Dict[str, Any]) -> str:
        """Enrich goal text with intake form context."""
        enriched = goal
        if context.get("required_features"):
            enriched += "\n\nRequired features:\n" + "\n".join(
                f"- {f}" for f in context["required_features"]
            )
        if context.get("things_to_avoid"):
            enriched += "\n\nDo NOT include:\n" + "\n".join(
                f"- {a}" for a in context["things_to_avoid"]
            )
        if context.get("tech_stack"):
            enriched += f"\n\nTech stack: {context['tech_stack']}"
        if context.get("priority"):
            enriched += f"\n\nPriority: {context['priority']}"
        return enriched

    def _analyze_input(
        self, goal: str, constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Classify the input goal and estimate complexity."""
        result: Dict[str, Any] = {
            "goal": goal,
            "estimated_complexity": "medium",
            "domain": "general",
            "needs_research": False,
            "needs_code": False,
            "needs_ui": False,
        }
        g = goal.lower()
        result["needs_code"] = any(
            k in g
            for k in [
                "code", "implement", "build", "create", "program", "software",
            ]
        )
        result["needs_research"] = any(
            k in g for k in ["research", "analyze", "investigate", "study"]
        )
        result["needs_ui"] = any(
            k in g
            for k in ["ui", "frontend", "interface", "web app", "dashboard"]
        )
        result["domain"] = (
            "coding"
            if result["needs_code"]
            else "research"
            if result["needs_research"]
            else "general"
        )
        word_count = len(goal.split())
        result["estimated_complexity"] = (
            "simple" if word_count < 10 else "complex" if word_count > 30 else "medium"
        )
        return result

    def _make_default_handler(self) -> Callable:
        """Create a default task handler using agent inference with token optimisation."""

        def handle_task(task: TaskNode) -> Dict[str, Any]:
            try:
                assigned_model = task.input_data.get("assigned_model", "default")
                is_cloud = not any(
                    x in assigned_model.lower()
                    for x in [
                        "qwen", "llama", "mistral", "gemma", "phi",
                        "local", "lm_studio", "default",
                    ]
                )

                task_context = (
                    " ".join(str(v)[:500] for v in task.input_data.values() if v)
                    if task.input_data
                    else ""
                )

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

                task_type_label = task.task_type or "general"

                # Post task to blackboard for inter-agent visibility
                try:
                    from vetinari.blackboard import get_blackboard
                    board = get_blackboard()
                    board.post(
                        content=task.description[:500],
                        request_type=task_type_label,
                        requested_by="orchestrator",
                        priority=5,
                        metadata={"task_id": task.id},
                    )
                except Exception:
                    pass  # Blackboard unavailable, continue without

                # Augment with web search for research/exploration tasks
                if task_type_label in ("research", "exploration", "documentation", "fact_finding"):
                    try:
                        from vetinari.tools.web_search import web_search
                        search_query = task.description[:200]
                        results = web_search(search_query, max_results=3)
                        if results:
                            web_context = "\nRelevant web search results:\n"
                            for r in results[:3]:
                                web_context += f"- [{r.get('title','')}]({r.get('url','')}): {r.get('snippet','')}\n"
                            optimised_prompt = web_context + "\n" + optimised_prompt
                    except Exception:
                        pass  # Web search unavailable

                system_prompt = (
                    f"You are Vetinari, an AI orchestration system executing "
                    f"a {task_type_label} task. "
                    "Produce structured, production-quality output. "
                    "Return valid JSON when structured output is requested. "
                    "Include reasoning and confidence scores with decisions. "
                    "Report errors with actionable context."
                )

                adapter_manager = self.agent_context.get("adapter_manager")
                if adapter_manager:
                    try:
                        from vetinari.adapters.base import InferenceRequest

                        req = InferenceRequest(
                            model_id=assigned_model,
                            prompt=optimised_prompt,
                            system_prompt=system_prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                        resp = adapter_manager.infer(req)
                        if resp.status == "ok":
                            return {
                                "result": resp.output,
                                "status": "ok",
                                "task_id": task.id,
                            }
                    except Exception as e:
                        logger.warning(
                            f"Adapter inference failed for task {task.id}: {e}"
                        )

                # Fallback: use LM Studio adapter directly
                from vetinari.lmstudio_adapter import LMStudioAdapter

                host = os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")
                adapter = LMStudioAdapter(host=host)
                result = adapter.chat(
                    model_id=assigned_model,
                    system_prompt=system_prompt,
                    input_text=optimised_prompt,
                )
                return {
                    "result": result.get("output", ""),
                    "status": "ok",
                    "task_id": task.id,
                }
            except Exception as e:
                logger.error("Task handler failed for %s: %s", task.id, e)
                return {
                    "result": "",
                    "status": "error",
                    "error": str(e),
                    "task_id": task.id,
                }

        return handle_task

    def _review_outputs(
        self, exec_results: Dict[str, Any], goal: str
    ) -> Dict[str, Any]:
        """Use QualityAgent to review execution outputs for quality."""
        try:
            quality = self._get_agent("QUALITY")
            if quality:
                from vetinari.agents.contracts import AgentTask, AgentType

                task_results = exec_results.get("task_results", {})
                artifacts = [str(v) for v in task_results.values() if v]
                eval_task = AgentTask(
                    task_id="review-0",
                    agent_type=AgentType.QUALITY,
                    description=f"Review outputs for goal: {goal}",
                    prompt=f"Review outputs for goal: {goal}",
                    context={"artifacts": artifacts[:5], "focus": "all", "mode": "code_review"},
                )
                result = quality.execute(eval_task)
                if result.success:
                    return result.output
        except Exception as e:
            logger.warning("Output review failed: %s", e)
        return {
            "verdict": "inconclusive",
            "quality_score": 0.5,
            "summary": "Review skipped (quality agent unavailable)",
        }

    def _assemble_final_output(
        self,
        exec_results: Dict[str, Any],
        review_result: Dict[str, Any],
        goal: str,
    ) -> str:
        """Use OperationsAgent (synthesis mode) to assemble a final coherent output."""
        try:
            operations = self._get_agent("OPERATIONS")
            if operations:
                from vetinari.agents.contracts import AgentTask, AgentType

                task_results = exec_results.get("task_results", {})
                sources = [
                    {"agent": k, "artifact": str(v)[:500]}
                    for k, v in task_results.items()
                    if v
                ]
                sources.append(
                    {"agent": "review", "artifact": str(review_result)[:200]}
                )
                synth_task = AgentTask(
                    task_id="assemble-0",
                    agent_type=AgentType.OPERATIONS,
                    description=f"Final assembly for goal: {goal}",
                    prompt=f"Final assembly for goal: {goal}",
                    context={"sources": sources, "type": "final_report", "mode": "synthesis"},
                )
                result = operations.execute(synth_task)
                if result.success and result.output:
                    return result.output.get(
                        "synthesized_artifact", str(result.output)
                    )
        except Exception as e:
            logger.warning("Final assembly failed: %s", e)

        # Fallback: join task_results
        task_results = exec_results.get("task_results", {})
        parts = [f"# Task {k}\n{v}" for k, v in task_results.items() if v]
        return "\n\n".join(parts) if parts else f"Completed: {goal}"

    # ------------------------------------------------------------------
    # Phase 7.9A: AgentGraph as execution backend
    # ------------------------------------------------------------------

    def execute_with_agent_graph(
        self,
        goal: str,
        constraints: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Execute a goal using AgentGraph as the execution backend.

        This delegates task execution to AgentGraph's registered agents
        instead of the generic DurableExecutionEngine handler, enabling:
        - Agent-specific execution with proper verification
        - Self-correction loops and error recovery
        - Maker-checker pattern for BUILDER outputs
        - Permission enforcement and constraint checking

        Falls back to ``generate_and_execute()`` if AgentGraph is unavailable.
        """
        try:
            from vetinari.orchestration.agent_graph import get_agent_graph
            from vetinari.agents.contracts import (
                AgentType, Plan, Task as ContractsTask,
            )

            agent_graph = get_agent_graph()
            context = context or {}

            # Generate plan using PlanGenerator
            graph = self.plan_generator.generate_plan(goal, constraints)

            # Convert ExecutionGraph nodes -> contracts.Plan for AgentGraph
            plan = Plan.create_new(goal)
            for node_id, node in graph.nodes.items():
                agent_type_str = node.input_data.get(
                    "assigned_agent", "BUILDER"
                ).upper()
                try:
                    agent_type = AgentType[agent_type_str]
                except KeyError:
                    agent_type = AgentType.BUILDER

                task = ContractsTask(
                    id=node.id,
                    description=node.description,
                    assigned_agent=agent_type,
                    dependencies=list(node.dependencies),
                    inputs=list(node.input_data.keys()) if node.input_data else [],
                    outputs=[],
                )
                plan.tasks.append(task)

            # Execute via AgentGraph
            results = agent_graph.execute_plan(plan)

            return {
                "plan_id": graph.plan_id,
                "goal": goal,
                "backend": "agent_graph",
                "completed": sum(
                    1 for r in results.values() if r.success
                ),
                "failed": sum(
                    1 for r in results.values() if not r.success
                ),
                "outputs": {
                    tid: r.output for tid, r in results.items()
                },
                "errors": {
                    tid: r.errors for tid, r in results.items()
                    if r.errors
                },
            }

        except Exception as e:
            logger.warning(
                f"[TwoLayer] AgentGraph execution failed, falling back: {e}"
            )
            return self.generate_and_execute(
                goal, constraints, context=context,
            )

    def generate_plan_only(
        self,
        goal: str,
        constraints: Dict[str, Any] = None,
    ) -> ExecutionGraph:
        """Generate a plan without executing."""
        return self.plan_generator.generate_plan(goal, constraints)

    def execute_plan(
        self,
        graph: ExecutionGraph,
        task_handler: Callable = None,
    ) -> Dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_two_layer_orchestrator: Optional[TwoLayerOrchestrator] = None


def get_two_layer_orchestrator() -> TwoLayerOrchestrator:
    """Get or create the global two-layer orchestrator."""
    global _two_layer_orchestrator
    if _two_layer_orchestrator is None:
        _two_layer_orchestrator = TwoLayerOrchestrator()
    return _two_layer_orchestrator


def init_two_layer_orchestrator(
    checkpoint_dir: str = None, **kwargs
) -> TwoLayerOrchestrator:
    """Initialize a new two-layer orchestrator."""
    global _two_layer_orchestrator
    _two_layer_orchestrator = TwoLayerOrchestrator(
        checkpoint_dir=checkpoint_dir, **kwargs
    )
    return _two_layer_orchestrator
