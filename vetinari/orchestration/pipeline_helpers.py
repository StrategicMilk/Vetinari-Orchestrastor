"""Pipeline helper methods mixin — agent access, context utilities, and the default task handler.

This is step 3 of the split from two_layer.py: helper methods that support
the pipeline stages but do not implement them directly. Covers agent lookup,
model routing, goal enrichment, memory retrieval, variant config, and the
default inference-backed task handler.

Designed as a mixin: ``PipelineHelpersMixin`` is composed into
``TwoLayerOrchestrator`` and accesses ``self`` attributes set by
``TwoLayerOrchestrator.__init__``.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import Any

from vetinari.constants import INFERENCE_STATUS_OK
from vetinari.orchestration.execution_graph import ExecutionTaskNode
from vetinari.types import AgentType
from vetinari.web.variant_system import VariantConfig

logger = logging.getLogger(__name__)


class PipelineHelpersMixin:
    """Mixin providing agent access, model routing, and the default task handler.

    Mixed into TwoLayerOrchestrator. All methods access ``self`` attributes
    defined in TwoLayerOrchestrator.__init__: ``agent_context``, ``_agents``,
    ``model_router``, ``execution_engine``, ``_variant_manager``.
    """

    # v0.5.0: 3 factory-pipeline agents + string aliases redirected
    _AGENT_MODULE_MAP = {
        AgentType.FOREMAN.value: ("vetinari.agents", "get_foreman_agent"),
        AgentType.WORKER.value: ("vetinari.agents", "get_worker_agent"),
        AgentType.INSPECTOR.value: ("vetinari.agents", "get_inspector_agent"),
    }

    # -- Variant system integration -----------------------------------------

    def get_variant_config(self) -> VariantConfig:
        """Return the active VariantConfig controlling token and depth limits.

        Returns:
            The VariantConfig for the current processing depth level.
        """
        return self._variant_manager.get_config()  # type: ignore[attr-defined]

    def set_variant_level(self, level: str) -> VariantConfig:
        """Switch the processing depth level and return the new config.

        Args:
            level: One of ``"low"``, ``"medium"``, or ``"high"``.

        Returns:
            The VariantConfig for the newly selected level.

        Raises:
            ValueError: If *level* is not a recognised VariantLevel value.
        """
        config = self._variant_manager.set_level(level)  # type: ignore[attr-defined]
        logger.info(
            "Variant level changed to %s (max_context_tokens=%s, max_planning_depth=%s)",
            level,
            config.max_context_tokens,
            config.max_planning_depth,
        )
        return config

    # -- Task handler registration ------------------------------------------

    def set_task_handlers(self, handlers: dict[str, Callable]) -> None:
        """Register task type handlers with the execution engine.

        Args:
            handlers: Mapping of task type string to handler callable.
        """
        for task_type, handler in handlers.items():
            self.execution_engine.register_handler(task_type, handler)  # type: ignore[attr-defined]

    def set_agent_context(self, context: dict[str, Any]) -> None:
        """Replace the shared agent context (adapter_manager, web_search, etc.).

        Clears the cached agent instances so they are re-initialized with the
        new context on their next use.

        Args:
            context: New agent context dict.
        """
        self.agent_context = context  # type: ignore[attr-defined]
        self._agents.clear()  # type: ignore[attr-defined]

    # -- Agent lookup -------------------------------------------------------

    def _get_agent(self, agent_type_str: str) -> Any:
        """Get or create an agent by type string, initialized with shared context.

        Caches agent instances in ``self._agents`` keyed by uppercased type
        string. Uses dynamic import via ``_AGENT_MODULE_MAP`` to avoid circular
        imports at module load time.

        Args:
            agent_type_str: Agent type string (e.g. ``"FOREMAN"``).

        Returns:
            The agent instance, or None if the type is unknown or the import
            fails.
        """
        key = agent_type_str.upper()
        if key in self._agents:  # type: ignore[attr-defined]
            return self._agents[key]  # type: ignore[attr-defined]
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
            if self.agent_context:  # type: ignore[attr-defined]
                agent.initialize(self.agent_context)  # type: ignore[attr-defined]
            self._agents[key] = agent  # type: ignore[attr-defined]
            return agent
        except Exception as e:
            logger.warning("Could not get agent '%s': %s", key, e)
            return None

    # -- Model routing -------------------------------------------------------

    def _route_model_for_task(self, task: ExecutionTaskNode) -> str:
        """Select the best model for a task using dynamic model routing.

        Falls back to ``"auto"`` (resolved by the adapter to the best
        available model) when the model router is unavailable or raises.

        Args:
            task: The task node to route.

        Returns:
            Model ID string, or ``"auto"`` as fallback.
        """
        if self.model_router is None:  # type: ignore[attr-defined]
            try:
                from vetinari.models.dynamic_model_router import get_model_router

                self.model_router = get_model_router()  # type: ignore[attr-defined]
            except Exception:
                logger.warning(
                    "Model router unavailable for task %s — falling back to 'auto'",
                    task.id,
                )
                return "auto"  # Adapter resolves "auto" to best available model
        try:
            from vetinari.models.dynamic_model_router import TaskType

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
            selection = self.model_router.select_model(t_type)  # type: ignore[attr-defined]
            if selection and selection.model:
                # Store confidence on the task node so pipeline stages can check it
                if selection.confidence_result is not None:
                    task.input_data["_selection_confidence"] = selection.confidence_result.score
                    task.input_data["_selection_confidence_level"] = selection.confidence_result.level.value
                    task.input_data["_selection_confidence_explanation"] = selection.confidence_result.explanation
                # Store "I don't know" protocol messages for downstream visibility
                if selection.unknown_situations:
                    task.input_data["_unknown_situations"] = [
                        {"situation": p.situation.value, "message": p.message, "action": p.action}
                        for p in selection.unknown_situations
                    ]
                return selection.model.id
        except Exception as e:
            logger.warning("Model routing failed for task %s: %s", task.id, e)
        return "auto"  # Adapter resolves "auto" to best available model

    # -- Goal enrichment and memory -----------------------------------------

    @staticmethod
    def _enrich_goal(goal: str, context: dict[str, Any]) -> str:
        """Append intake-form context fields to the goal text before planning.

        Args:
            goal: Raw user goal string.
            context: Pipeline context dict possibly containing ``required_features``,
                ``things_to_avoid``, ``tech_stack``, and ``priority``.

        Returns:
            Enriched goal string with structured context appended.
        """
        enriched = goal
        if context.get("required_features"):
            enriched += "\n\nRequired features:\n" + "\n".join(f"- {f}" for f in context["required_features"])
        if context.get("things_to_avoid"):
            enriched += "\n\nDo NOT include:\n" + "\n".join(f"- {a}" for a in context["things_to_avoid"])
        if context.get("tech_stack"):
            enriched += f"\n\nTech stack: {context['tech_stack']}"
        if context.get("priority"):
            enriched += f"\n\nPriority: {context['priority']}"
        return enriched

    def _retrieve_memory_for_planning(self, goal: str) -> list[dict[str, Any]]:
        """Query long-term memory for entries relevant to the current goal.

        Searches for DECISION, PATTERN, WARNING, and SOLUTION entries that
        might inform plan generation — avoiding past mistakes and reusing
        proven approaches.

        Args:
            goal: The enriched goal text to search against (first 200 chars used).

        Returns:
            List of memory entry summaries with type, content, and timestamp.
            Empty list if memory store is unavailable or no matches found.
        """
        try:
            from vetinari.memory.unified import get_unified_memory_store

            store = get_unified_memory_store()
            query = goal[:200]  # Use first 200 chars as search context
            results = store.search(
                query,
                entry_types=["decision", "pattern", "warning", "solution"],
                limit=5,
            )
            return [
                {
                    "type": entry.entry_type,
                    "content": entry.summary or entry.content[:300],
                    "timestamp": str(entry.timestamp),
                }
                for entry in results
            ]
        except Exception:
            logger.warning("Memory store unavailable for planning enrichment", exc_info=True)
            return []

    def _analyze_input(self, goal: str, constraints: dict[str, Any]) -> dict[str, Any]:
        """Classify the input goal and estimate complexity.

        Delegates to classify_goal_detailed() for LLM-backed classification
        with confidence scoring.  Falls back to keyword matching as an explicit
        degraded state when the classifier is unavailable.

        Args:
            goal: Enriched goal text.
            constraints: Pipeline constraints dict (currently unused but
                kept for future rule-based analysis).

        Returns:
            Dict with keys ``goal``, ``estimated_complexity``, ``domain``,
            ``needs_research``, ``needs_code``, ``needs_ui``,
            ``classification_confidence``, ``classification_source``,
            ``cross_cutting``.
        """
        result: dict[str, Any] = {
            "goal": goal,
            "estimated_complexity": "medium",
            "domain": "general",
            "goal_type": "general",
            "needs_research": False,
            "needs_code": False,
            "needs_ui": False,
        }
        # Use confidence-gated classification (LLM when available, keyword fallback)
        try:
            from vetinari.orchestration.request_routing import classify_goal_detailed

            classification = classify_goal_detailed(goal)
            category = classification.get("category", "general")
            result["estimated_complexity"] = classification.get("complexity", "medium")
            result["classification_confidence"] = classification.get("confidence", 0.3)
            result["classification_source"] = classification.get("source", "keyword")
            result["cross_cutting"] = classification.get("cross_cutting", [])
            result["goal_type"] = category

            # Map category to domain flags
            result["needs_code"] = category in ("code", "devops", "git")
            result["needs_research"] = category in ("research", "data")
            result["needs_ui"] = category in ("ui", "image")
            result["domain"] = (
                "coding" if result["needs_code"] else "research" if result["needs_research"] else "general"
            )
        except Exception:
            logger.warning("Goal classification unavailable — using keyword fallback for pipeline analysis")
            # Minimal keyword fallback (degraded state)
            g = goal.lower()
            result["needs_code"] = any(k in g for k in ["code", "implement", "build", "create", "program"])
            result["needs_research"] = any(k in g for k in ["research", "analyze", "investigate", "study"])
            result["needs_ui"] = any(k in g for k in ["ui", "frontend", "interface", "dashboard"])
            result["domain"] = (
                "coding" if result["needs_code"] else "research" if result["needs_research"] else "general"
            )
            # Set goal_type based on keyword fallback
            result["goal_type"] = (
                "code" if result["needs_code"] else "research" if result["needs_research"] else "general"
            )
            word_count = len(goal.split())
            result["estimated_complexity"] = "simple" if word_count < 10 else "complex" if word_count > 30 else "medium"
            result["classification_confidence"] = 0.2  # Explicit degraded-state confidence
            result["classification_source"] = "keyword_fallback"
        return result

    def _run_clarification(self, goal: str, context: dict[str, Any]) -> dict[str, Any] | None:
        """Run ForemanAgent clarification mode to detect ambiguity in the goal.

        Looks up the FOREMAN agent from the agent cache or AgentGraph registry,
        runs it in ``"clarify"`` mode, and returns the result dict if the agent
        is available and the execution succeeds.

        Args:
            goal: The user's goal string.
            context: The pipeline context.

        Returns:
            Clarify result dict, or None if clarification is unavailable.
        """
        try:
            planner = None
            if hasattr(self, "_agents"):
                # _agents dict is keyed by string ("FOREMAN"), not AgentType enum
                planner = self._agents.get(AgentType.FOREMAN.value)  # type: ignore[attr-defined]
            if planner is None:
                # Try AgentGraph registry
                try:
                    from vetinari.orchestration.agent_graph import get_agent_graph

                    ag = get_agent_graph()
                    planner = ag._agents.get(AgentType.FOREMAN.value)
                except Exception:
                    logger.warning("Planner lookup failed", exc_info=True)
            if planner is None:
                logger.debug("[Pipeline] No planner available for clarification")
                return None

            from vetinari.agents.contracts import AgentTask

            clarify_task = AgentTask(
                task_id="clarify-intake",
                agent_type=AgentType.FOREMAN,
                description=f"Check if this goal needs clarification: {goal}",
                prompt=goal,
                context={"goal": goal, "existing_context": context, "mode": "clarify"},
            )
            result = planner.execute(clarify_task)
            if result.success and isinstance(result.output, dict):
                return result.output
            return None
        except Exception as e:
            logger.warning("[Pipeline] Clarification failed: %s", e)
            return None

    # -- Default task handler -----------------------------------------------

    def _make_default_handler(self) -> Callable:
        """Create the default task handler using agent inference with token optimisation.

        The returned callable accepts an ExecutionTaskNode and returns a result
        dict with at least ``status`` and ``output``. Uses the shared
        ``agent_context`` (adapter_manager, etc.) set on this orchestrator
        instance.

        Returns:
            A callable suitable for passing to DurableExecutionEngine.execute_plan.
        """

        def handle_task(task: ExecutionTaskNode) -> dict[str, Any]:
            """Execute a single task node via agent inference.

            Returns:
                Dict with at least ``status`` (``"completed"`` or ``"failed"``)
                and ``output`` containing the inference result text.
            """
            # Switch to EXECUTION mode so tool permission checks pass
            try:
                from vetinari.execution_context import get_context_manager as _get_ctx
                from vetinari.types import ExecutionMode as _ExecMode

                _ctx_mgr = _get_ctx()
                _exec_ctx = _ctx_mgr.temporary_mode(_ExecMode.EXECUTION, task_id=task.id)
                _exec_ctx.__enter__()
            except Exception:
                _exec_ctx = None

            try:
                return _handle_task_inner(task)
            finally:
                if _exec_ctx is not None:
                    with contextlib.suppress(Exception):
                        _exec_ctx.__exit__(None, None, None)

        def _handle_task_inner(task: ExecutionTaskNode) -> dict[str, Any]:
            try:
                assigned_model = task.input_data.get("assigned_model", "default")
                is_cloud = not any(
                    x in assigned_model.lower()
                    for x in [
                        "qwen",
                        "llama",
                        "mistral",
                        "gemma",
                        "phi",
                        "local",
                        "default",
                    ]
                )

                task_context = " ".join(str(v)[:500] for v in task.input_data.values() if v) if task.input_data else ""

                # Inject rework context into prompt when retrying a failed task
                rework_feedback = task.input_data.get("rework_feedback", "")
                rework_hint = task.input_data.get("rework_hint", "")
                if rework_hint == "research_context_before_retry":
                    task_context += " [REWORK: Research additional context before answering]"
                elif rework_hint == "widen_scope":
                    task_context += " [REWORK: Consider a broader solution scope]"
                if rework_feedback:
                    task_context += f" [PRIOR FAILURE FEEDBACK: {rework_feedback[:500]}]"

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
                    # Fall back to InferenceConfigManager defaults
                    try:
                        from vetinari.config.inference_config import get_inference_config

                        _fallback = get_inference_config().get_effective_params(
                            task.task_type or "general",
                        )
                        max_tokens = _fallback.get("max_tokens", 2048)
                        temperature = _fallback.get("temperature", 0.3)
                    except Exception:
                        max_tokens = 2048
                        temperature = 0.3

                task_type_label = task.task_type or "general"

                # Post task to blackboard for inter-agent visibility
                try:
                    from vetinari.memory.blackboard import get_blackboard

                    board = get_blackboard()
                    board.post(
                        content=task.description[:500],
                        request_type=task_type_label,
                        requested_by="orchestrator",
                        priority=5,
                        metadata={"task_id": task.id},
                    )
                except Exception:
                    logger.warning("Blackboard write failed; continuing without it", exc_info=True)

                # Augment with web search for research/exploration tasks
                if task_type_label in ("research", "exploration", "documentation", "fact_finding"):
                    try:
                        from vetinari.tools.web_search import web_search

                        search_query = task.description[:200]
                        results = web_search(search_query, max_results=3)
                        if results:
                            web_context = "\nRelevant web search results:\n"
                            for r in results[:3]:
                                web_context += f"- [{r.get('title', '')}]({r.get('url', '')}): {r.get('snippet', '')}\n"
                            optimised_prompt = web_context + "\n" + optimised_prompt
                    except Exception:
                        logger.warning("Web search unavailable; continuing without augmentation", exc_info=True)

                system_prompt = (
                    f"You are Vetinari, an AI orchestration system executing "
                    f"a {task_type_label} task. "
                    "Produce structured, production-quality output. "
                    "Return valid JSON when structured output is requested. "
                    "Include reasoning and confidence scores with decisions. "
                    "Report errors with actionable context."
                )
                # Append project context from intake form to system prompt
                _proj_ctx = task.input_data.get("project_context")
                if _proj_ctx and isinstance(_proj_ctx, dict):
                    _ctx_parts = []
                    if _proj_ctx.get("tech_stack"):
                        _ctx_parts.append(f"Tech stack: {_proj_ctx['tech_stack']}")
                    if _proj_ctx.get("category"):
                        _ctx_parts.append(f"Category: {_proj_ctx['category']}")
                    if _proj_ctx.get("priority"):
                        _ctx_parts.append(f"Priority: {_proj_ctx['priority']}")
                    if _proj_ctx.get("required_features"):
                        _feats = _proj_ctx["required_features"]
                        if isinstance(_feats, list):
                            _feats = ", ".join(_feats)
                        _ctx_parts.append(f"Required features: {_feats}")
                    if _proj_ctx.get("things_to_avoid"):
                        _avoids = _proj_ctx["things_to_avoid"]
                        if isinstance(_avoids, list):
                            _avoids = ", ".join(_avoids)
                        _ctx_parts.append(f"Constraints: {_avoids}")
                    if _ctx_parts:
                        system_prompt += "\n\nProject context:\n" + "\n".join(_ctx_parts)

                adapter_manager = self.agent_context.get("adapter_manager")  # type: ignore[attr-defined]
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

                        # Use Best-of-N generation for custom/standard tier tasks
                        # to improve output quality via test-time compute scaling.
                        # Express tier (N=1) is a passthrough with no overhead.
                        _output_text: str | None = None
                        _tokens_used: int = 0
                        _resp_metadata: dict = {}
                        try:
                            from vetinari.models.best_of_n import get_n_for_tier
                            from vetinari.models.dynamic_model_router import get_model_router

                            _n = get_n_for_tier(task_type_label)
                            if _n > 1:
                                _router = get_model_router()

                                def _generate_fn(prompt: str) -> str:
                                    _r = InferenceRequest(
                                        model_id=assigned_model,
                                        prompt=prompt,
                                        system_prompt=system_prompt,
                                        max_tokens=max_tokens,
                                        temperature=temperature,
                                        metadata={"task_type": task_type_label},
                                    )
                                    _resp = adapter_manager.infer(_r)
                                    return _resp.output if _resp.status == INFERENCE_STATUS_OK else ""

                                def _scorer(candidate: str) -> float:
                                    # Prefer longer, non-empty candidates as a lightweight
                                    # quality proxy.  A richer scorer (PRM, verifier) can be
                                    # injected later without changing the call site.
                                    return float(len(candidate.strip())) / max(max_tokens, 1)

                                _selector = _router.get_best_of_n_selector(_generate_fn)
                                _output_text = _selector.generate_and_select(
                                    optimised_prompt,
                                    n=_n,
                                    scorer=_scorer,
                                )
                                logger.debug(
                                    "Best-of-%d selection applied for task %s (tier=%s)",
                                    _n,
                                    task.id,
                                    task_type_label,
                                )
                        except Exception:
                            logger.warning(
                                "Best-of-N generation unavailable for task %s — falling back to single inference",
                                task.id,
                            )

                        if _output_text is None:
                            resp = adapter_manager.infer(req)
                            if resp.status == INFERENCE_STATUS_OK:
                                _output_text = resp.output
                                _tokens_used = resp.tokens_used
                                _resp_metadata = resp.metadata

                        if _output_text is not None:
                            return {
                                "result": _output_text,
                                "status": "ok",
                                "task_id": task.id,
                                "tokens_used": _tokens_used,
                                "metadata": _resp_metadata,
                            }
                    except Exception as e:
                        logger.warning("Adapter inference failed for task %s: %s", task.id, e)

                # Fallback: use local inference adapter directly
                from vetinari.adapters.llama_cpp_local_adapter import LocalInferenceAdapter

                adapter = LocalInferenceAdapter()
                result = adapter.chat(
                    model_id=assigned_model,
                    system_prompt=system_prompt,
                    input_text=optimised_prompt,
                )
                return {
                    "result": result.get("output", ""),
                    "status": "ok",
                    "task_id": task.id,
                    "tokens_used": result.get("tokens_used", 0),
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
