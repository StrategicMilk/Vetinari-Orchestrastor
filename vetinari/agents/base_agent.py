"""
Vetinari Base Agent

This module defines the base agent class that all Vetinari agents inherit from.
All agents must implement the execute and verify methods.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from vetinari.agents.contracts import (
    AgentResult,
    AgentSpec,
    AgentTask,
    AgentType,
    VerificationResult,
    get_agent_spec
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constraint-aware helpers (Phase 8.10)
# ---------------------------------------------------------------------------

def _get_agent_constraints(agent_type_value: str, mode: Optional[str] = None):
    """Lazily load constraints for an agent. Returns None on import failure."""
    try:
        from vetinari.constraints.registry import get_constraint_registry
        return get_constraint_registry().get_constraints_for_agent(
            agent_type_value, mode=mode,
        )
    except Exception:
        return None


class BaseAgent(ABC):
    """Base class for all Vetinari agents.
    
    All agents must inherit from this class and implement:
    - execute(): Process a task and return results
    - verify(): Verify output meets quality standards
    - get_system_prompt(): Return the agent's system prompt
    """
    
    def __init__(self, agent_type: AgentType, config: Optional[Dict[str, Any]] = None):
        """Initialize the agent.
        
        Args:
            agent_type: The type of agent
            config: Optional configuration dictionary
        """
        self._agent_type = agent_type
        self._config = config or {}
        self._spec = get_agent_spec(agent_type)
        self._initialized = False
        self._context: Dict[str, Any] = {}
        
    @property
    def agent_type(self) -> AgentType:
        """Return the agent type."""
        return self._agent_type
    
    @property
    def name(self) -> str:
        """Return the human-readable agent name."""
        return self._spec.name if self._spec else self._agent_type.value
    
    @property
    def description(self) -> str:
        """Return the agent description."""
        return self._spec.description if self._spec else ""
    
    @property
    def default_model(self) -> str:
        """Return the default model for this agent."""
        return self._spec.default_model if self._spec else ""
    
    @property
    def thinking_variant(self) -> str:
        """Return the thinking variant for this agent."""
        return self._spec.thinking_variant if self._spec else "medium"
    
    @property
    def is_initialized(self) -> bool:
        """Return whether the agent is initialized."""
        return self._initialized
    
    def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize the agent with context.
        
        Args:
            context: Context information including:
                - adapter_manager: AdapterManager instance for LLM inference
                - web_search: WebSearchTool instance for online research
                - tool_registry: ToolRegistry for registered tools
                - Any agent-specific configuration
        """
        self._context = context
        # Extract key shared services from context
        self._adapter_manager = context.get("adapter_manager")
        self._web_search = context.get("web_search")
        self._tool_registry = context.get("tool_registry")
        self._initialized = True
        self._log("info", f"Agent {self.name} initialized")

    # ------------------------------------------------------------------
    # Base prompt framework (modern best practices)
    # ------------------------------------------------------------------

    @classmethod
    def _get_base_prompt_framework(cls) -> str:
        """Universal prompt framework applied to all agents."""
        return (
            "## Core Operating Principles\n\n"
            "REASONING: Think step-by-step before producing output. For complex decisions:\n"
            "1. Identify the key question or requirement\n"
            "2. Consider 2-3 approaches with trade-offs\n"
            "3. Choose the best approach and explain why\n"
            "4. Execute with verification\n\n"
            "CONFIDENCE: Rate your confidence in each major output:\n"
            "- HIGH (>80%): Well-understood domain, clear requirements, verified data\n"
            "- MEDIUM (50-80%): Some ambiguity, partial information, reasonable inference\n"
            "- LOW (<50%): Significant uncertainty — flag for human review\n\n"
            "VERIFICATION: Before finalizing output:\n"
            "- Does this directly address the task requirements?\n"
            "- Are there logical contradictions or unsupported claims?\n"
            "- Would a domain expert find obvious errors?\n"
            "- Is the output format correct and complete?\n\n"
            "ERROR HANDLING:\n"
            "- If requirements are ambiguous, state your assumptions explicitly\n"
            "- If you lack information, say so rather than fabricating data\n"
            "- If a subtask fails, provide partial results with clear error context\n"
            "- Never silently drop errors — always surface them\n\n"
            "QUALITY:\n"
            "- Cite sources or reasoning for factual claims\n"
            "- Prefer specific, actionable output over vague generalities\n"
            "- If output exceeds expected scope, summarize and offer details on request\n"
            "- Maintain consistent terminology throughout"
        )

    # ------------------------------------------------------------------
    # LLM Inference helper
    # ------------------------------------------------------------------

    def _infer(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_id: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        expect_json: bool = False,
    ) -> str:
        """Call an LLM via the AdapterManager and return the text output.

        Falls back gracefully if the adapter manager is unavailable.

        Args:
            prompt: The user/task prompt.
            system_prompt: Optional system prompt override. Uses agent's
                           get_system_prompt() when not provided.
            model_id: Optional model override. Uses agent's default_model
                      when not provided.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            expect_json: If True, appends a JSON-output instruction and
                         attempts to strip markdown fences from the result.

        Returns:
            The generated text string, or an empty string on error.
        """
        # ── C1: Circuit breaker pre-check ─────────────────────────────
        try:
            from vetinari.resilience.circuit_breaker import get_circuit_breaker_registry
            _cb = get_circuit_breaker_registry().get(self._agent_type.value)
            if not _cb.allow_request():
                self._log("warning", f"Circuit breaker OPEN for {self._agent_type.value}")
                return ""
        except ImportError:
            pass  # resilience module not available
        except Exception as _cb_err:
            logger.debug("Circuit breaker check failed: %s", _cb_err)

        # ── C5: Token budget pre-check ────────────────────────────────
        _budget_remaining = getattr(self, '_token_budget_remaining', None)
        if _budget_remaining is not None and _budget_remaining <= 0:
            self._log("warning", f"Token budget exhausted for {self._agent_type.value}")
            return ""

        if self._adapter_manager is None:
            # No adapter: try to use the singleton if available
            try:
                from vetinari.adapter_manager import get_adapter_manager
                self._adapter_manager = get_adapter_manager()
            except Exception:
                logger.debug("Failed to import singleton adapter_manager", exc_info=True)

        if self._adapter_manager is None:
            # Last resort: call LM Studio directly
            try:
                from vetinari.lmstudio_adapter import LMStudioAdapter
                _host = os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")
                _adapter = LMStudioAdapter(host=_host)
                _sys = system_prompt or self.get_system_prompt()
                _model = model_id or self.default_model or "default"
                resp = _adapter.chat(_model, _sys, prompt)
                return resp.get("output", "")
            except Exception as e:
                self._log("error", f"LLM inference failed (no adapter_manager): {e}")
                return ""

        # Prepend base prompt framework to agent-specific system prompt
        _agent_system = system_prompt or self.get_system_prompt()
        _active_system_prompt = self._get_base_prompt_framework() + "\n\n" + _agent_system
        _variant_id = "default"
        try:
            from vetinari.learning.prompt_evolver import get_prompt_evolver
            evolved_prompt, _variant_id = get_prompt_evolver().select_prompt(self._agent_type.value)
            if evolved_prompt and evolved_prompt != _active_system_prompt:
                _active_system_prompt = evolved_prompt
        except Exception:
            logger.debug("Failed to select evolved prompt for agent %s", self._agent_type.value, exc_info=True)

        # Apply task-specific inference params from external config (Step 16)
        _model_for_config = model_id or self.default_model or ""
        try:
            from vetinari.config.inference_config import get_inference_config
            _task_key = self._agent_type.value.lower()
            _effective = get_inference_config().get_effective_params(_task_key, _model_for_config)
            # Only override if caller used default values
            if max_tokens == 4096:
                max_tokens = _effective.get("max_tokens", max_tokens)
            if temperature == 0.3:
                temperature = _effective.get("temperature", temperature)
        except Exception:
            # Fallback to legacy token_optimizer
            try:
                from vetinari.token_optimizer import get_token_optimizer
                _optimizer = get_token_optimizer()
                _profile = _optimizer.get_task_profile(self._agent_type.value.lower())
                _profile_max_tokens, _profile_temp, _ = _profile
                if max_tokens == 4096:
                    max_tokens = _profile_max_tokens
                if temperature == 0.3:
                    temperature = _profile_temp
            except Exception:
                logger.debug("Failed to load token_optimizer task profile for %s", self._agent_type.value, exc_info=True)

        # Use AdapterManager.infer() path
        try:
            from vetinari.adapters.base import InferenceRequest
        except ImportError:
            # Fallback dataclass if adapters not available
            from dataclasses import dataclass, field as dc_field

            @dataclass
            class InferenceRequest:  # type: ignore[no-redef]
                model_id: str
                prompt: str
                system_prompt: Optional[str] = None
                max_tokens: int = 4096
                temperature: float = 0.3
                top_p: float = 0.9
                top_k: int = 40
                stop_sequences: List[str] = dc_field(default_factory=list)
                metadata: Dict[str, Any] = dc_field(default_factory=dict)

        if expect_json:
            prompt = prompt + "\n\nRespond ONLY with valid JSON. Do not include markdown code fences or explanation."

        request = InferenceRequest(
            model_id=model_id or self.default_model or "default",
            prompt=prompt,
            system_prompt=_active_system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        try:
            response = self._adapter_manager.infer(request)
            if response.status == "ok":
                result = response.output
                if expect_json:
                    # Strip any accidental markdown fences
                    result = result.strip()
                    if result.startswith("```"):
                        lines = result.split("\n")
                        result = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

                # ── C1: Record success ────────────────────────────────
                try:
                    from vetinari.resilience.circuit_breaker import get_circuit_breaker_registry
                    get_circuit_breaker_registry().get(self._agent_type.value).record_success()
                except Exception:
                    pass

                # ── C5: Track token usage ─────────────────────────────
                _estimated_tokens = len(result.split()) if result else 0
                if hasattr(self, '_token_budget_remaining') and self._token_budget_remaining is not None:
                    self._token_budget_remaining -= _estimated_tokens

                return result
            else:
                self._log("warning", f"Inference failed: {response.error}")
                try:
                    from vetinari.resilience.circuit_breaker import get_circuit_breaker_registry
                    get_circuit_breaker_registry().get(self._agent_type.value).record_failure()
                except Exception:
                    pass
                return ""
        except Exception as e:
            self._log("error", f"Inference exception: {e}")
            try:
                from vetinari.resilience.circuit_breaker import get_circuit_breaker_registry
                get_circuit_breaker_registry().get(self._agent_type.value).record_failure()
            except Exception:
                pass
            return ""

    def _infer_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_id: Optional[str] = None,
        fallback: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """Call _infer() and parse the result as JSON.

        Args:
            prompt: The user/task prompt.
            system_prompt: Optional system prompt override.
            model_id: Optional model override.
            fallback: Value to return if LLM output cannot be parsed as JSON.
                      If None, returns None on parse failure.
            **kwargs: Additional arguments passed to _infer().

        Returns:
            Parsed JSON (dict or list), or `fallback` on failure.
        """
        # Remove expect_json from kwargs to avoid duplicate keyword argument
        kwargs.pop("expect_json", None)
        raw = self._infer(
            prompt,
            system_prompt=system_prompt,
            model_id=model_id,
            expect_json=True,
            **kwargs,
        )
        if not raw:
            return fallback
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON object/array from surrounding text
            import re
            match = re.search(r'(\{.*\}|\[.*\])', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            self._log("warning", "Could not parse LLM output as JSON — using fallback")
            return fallback

    # ------------------------------------------------------------------
    # Web search helper
    # ------------------------------------------------------------------

    def _search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Perform a web search and return a list of result dicts.

        Each result dict has keys: title, url, snippet, source_reliability.
        Returns an empty list if no search tool is available.
        """
        if self._web_search is None:
            try:
                from vetinari.tools.web_search_tool import get_search_tool
                self._web_search = get_search_tool()
            except Exception:
                return []
        try:
            response = self._web_search.search(query, max_results=max_results)
            return [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "source_reliability": r.source_reliability,
                }
                for r in response.results
            ]
        except Exception as e:
            self._log("warning", f"Web search failed for '{query}': {e}")
            return []
        
    def _log(self, level: str, message: str, **kwargs) -> None:
        """Emit structured log with agent context."""
        log_data = {
            "agent_type": self._agent_type.value,
            "agent_name": self.name,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        getattr(logger, level)(f"{message} | {log_data}")
    
    # ------------------------------------------------------------------
    # Phase 2.0b: Template method helpers for concrete agents
    # ------------------------------------------------------------------

    def _execute_safely(self, task: AgentTask, execute_fn) -> AgentResult:
        """Template method for safe agent execution with validation and error handling.

        Handles validation, preparation, completion, and error handling.
        Agents provide only their unique core logic via execute_fn.

        Args:
            task: The task to execute.
            execute_fn: Callable(task) -> AgentResult with the agent's core logic.

        Returns:
            AgentResult
        """
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Task validation failed for {self.agent_type}"],
            )
        task = self.prepare_task(task)
        try:
            result = execute_fn(task)
            if result.success:
                self.complete_task(task, result)
            return result
        except Exception as e:
            logger.error("[%s] Execute failed: %s", self.agent_type, e)
            return AgentResult(success=False, output=None, errors=[str(e)])

    def _infer_with_fallback(self, prompt: str, fallback_fn=None, required_keys=None):
        """Infer from LLM with optional fallback and key validation.

        Args:
            prompt: The prompt to send to the LLM.
            fallback_fn: Optional callable to run if LLM fails.
            required_keys: Optional list of keys the JSON response must contain.

        Returns:
            Parsed response dict, or fallback result, or None.
        """
        try:
            response = self._infer_json(prompt)
            if response and required_keys:
                if all(k in response for k in required_keys):
                    return response
            elif response:
                return response
        except Exception as e:
            logger.debug("[%s] LLM inference failed: %s", self.agent_type, e)

        if fallback_fn:
            return fallback_fn()
        return None

    @abstractmethod
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the given task and return results.
        
        Args:
            task: The task to execute
            
        Returns:
            AgentResult containing success status, output, and metadata
        """
        pass
    
    @abstractmethod
    def verify(self, output: Any) -> VerificationResult:
        """Verify the output meets quality standards.
        
        Args:
            output: The output to verify
            
        Returns:
            VerificationResult with pass/fail status and issues
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent.
        
        Returns:
            The system prompt that defines the agent's role and behavior
        """
        pass
    
    def get_capabilities(self) -> List[str]:
        """Return the capabilities of this agent.
        
        Returns:
            List of capability identifiers
        """
        return []
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about this agent.
        
        Returns:
            Dictionary containing agent metadata
        """
        return {
            "agent_type": self._agent_type.value,
            "name": self.name,
            "description": self.description,
            "default_model": self.default_model,
            "thinking_variant": self.thinking_variant,
            "capabilities": self.get_capabilities(),
            "initialized": self._initialized
        }
    
    def validate_task(self, task: AgentTask) -> bool:
        """Validate that the task is appropriate for this agent.
        
        Args:
            task: The task to validate
            
        Returns:
            True if the task is valid for this agent
        """
        if task.agent_type != self._agent_type:
            self._log("warning", f"Task agent type {task.agent_type} does not match {self._agent_type}")
            return False
        return True
    
    def prepare_task(self, task: AgentTask) -> AgentTask:
        """Prepare a task for execution.
        
        This method can be overridden to add preprocessing.
        
        Args:
            task: The task to prepare
            
        Returns:
            The prepared task
        """
        if not self._initialized:
            self._log("warning", "Agent not initialized, initializing with default context")
            self.initialize({})

        # ----- Phase 11.9: Enforce MODEL_INFERENCE permission -----
        try:
            from vetinari.execution_context import get_context_manager, ToolPermission
            get_context_manager().enforce_permission(
                ToolPermission.MODEL_INFERENCE, "agent_execute"
            )
        except (ImportError, AttributeError):
            pass  # Permission system not available — degrade gracefully
        except PermissionError:
            raise  # Permission denied — propagate to caller

        task.started_at = datetime.now().isoformat()

        # ── C5: Initialize per-agent token budget ─────────────────────
        _TOKEN_BUDGETS = {
            "PLANNER": 16384,
            "CONSOLIDATED_RESEARCHER": 24576,
            "CONSOLIDATED_ORACLE": 16384,
            "BUILDER": 32768,
            "QUALITY": 16384,
            "OPERATIONS": 24576,
        }
        _budget = _TOKEN_BUDGETS.get(self._agent_type.value, 16384)
        if not hasattr(self, '_token_budget_remaining') or self._token_budget_remaining is None:
            self._token_budget_total = _budget
            self._token_budget_remaining = _budget

        # ----- Phase 8.10: Enforce resource constraints -----
        constraints = _get_agent_constraints(self._agent_type.value)
        if constraints and constraints.resources:
            rc = constraints.resources
            # Apply max_tokens cap to the task metadata so _infer() can respect it
            if not hasattr(task, '_constraint_max_tokens'):
                task._constraint_max_tokens = rc.max_tokens
            # Apply max_retries cap (accessible by AgentGraph)
            if not hasattr(task, '_constraint_max_retries'):
                task._constraint_max_retries = rc.max_retries
            # Store timeout for monitoring
            if not hasattr(task, '_constraint_timeout'):
                task._constraint_timeout = rc.timeout_seconds
            self._log("debug", f"Constraints applied: max_tokens={rc.max_tokens}, "
                      f"timeout={rc.timeout_seconds}s, max_retries={rc.max_retries}")

        # Emit structured trace span for this task
        try:
            from vetinari.structured_logging import log_event
            log_event("info", f"agent.{self._agent_type.value}", "task_started",
                      task_id=task.task_id, agent=self._agent_type.value)
        except Exception:
            logger.debug("Failed to emit structured trace span for task_started", exc_info=True)

        # Register prompt variant if evolver is available
        try:
            from vetinari.learning.prompt_evolver import get_prompt_evolver
            evolver = get_prompt_evolver()
            evolver.register_baseline(self._agent_type.value, self.get_system_prompt())
        except Exception:
            logger.debug("Failed to register prompt baseline for %s", self._agent_type.value, exc_info=True)

        return task

    # ------------------------------------------------------------------
    # Phase 7.9I: Dependency results incorporation
    # ------------------------------------------------------------------

    def _incorporate_prior_results(
        self, task: AgentTask
    ) -> Dict[str, Any]:
        """Extract and return dependency results from the task context.

        AgentGraph injects ``dependency_results`` into ``task.context`` before
        calling ``execute()``.  Subclasses can override this method to
        customise how prior results influence their execution.

        Returns:
            Dictionary mapping dependency task IDs to their result summaries.
            Empty dict if no dependency results are available.
        """
        ctx = getattr(task, "context", None) or {}
        dep_results = ctx.get("dependency_results", {})
        if dep_results:
            self._log(
                "debug",
                f"Incorporating {len(dep_results)} dependency results: "
                + ", ".join(dep_results.keys()),
            )
        return dep_results

    def complete_task(self, task: AgentTask, result: AgentResult) -> AgentTask:
        """Mark a task as complete.
        
        This method can be overridden to add postprocessing.
        
        Args:
            task: The completed task
            result: The result from execution
            
        Returns:
            The completed task
        """
        task.completed_at = datetime.now().isoformat()
        task.result = result.output
        if not result.success:
            task.error = "; ".join(result.errors)

        # Emit structured trace span for completion
        try:
            from vetinari.structured_logging import log_event
            log_event("info", f"agent.{self._agent_type.value}", "task_completed",
                      task_id=task.task_id, success=result.success,
                      agent=self._agent_type.value)
        except Exception:
            logger.debug("Failed to emit structured trace span for task_completed", exc_info=True)

        # ----- Phase 8.10: Quality gate enforcement -----
        if result.success and result.output:
            constraints = _get_agent_constraints(self._agent_type.value)
            if constraints and constraints.quality_gate:
                qg = constraints.quality_gate
                # Quality gate will be checked after scoring below;
                # store the gate for downstream use
                if not hasattr(task, '_quality_gate'):
                    task._quality_gate = qg

        # Feed results into quality scoring and feedback loop
        if result.success and result.output:
            try:
                import json as _json
                output_str = (result.output if isinstance(result.output, str)
                              else _json.dumps(result.output, default=str)[:1000])
                task_type = self._agent_type.value.lower()
                model_id = self.default_model or "default"

                from vetinari.learning.quality_scorer import get_quality_scorer
                scorer = get_quality_scorer()
                scorer._adapter_manager = self._adapter_manager
                score = scorer.score(
                    task_id=task.task_id,
                    model_id=model_id,
                    task_type=task_type,
                    task_description=task.description or "",
                    output=output_str,
                    use_llm=False,  # Avoid recursive inference calls
                )

                from vetinari.learning.feedback_loop import get_feedback_loop
                get_feedback_loop().record_outcome(
                    task_id=task.task_id,
                    model_id=model_id,
                    task_type=task_type,
                    quality_score=score.overall_score,
                    success=result.success,
                )

                from vetinari.learning.model_selector import get_thompson_selector
                get_thompson_selector().update(model_id, task_type, score.overall_score, result.success)

                # Phase 8.10: Check quality gate threshold
                if hasattr(task, '_quality_gate') and task._quality_gate:
                    try:
                        from vetinari.constraints.registry import get_constraint_registry
                        passed, reason = get_constraint_registry().check_quality_gate(
                            self._agent_type.value, score.overall_score,
                        )
                        if not passed:
                            self._log("warning", f"Quality gate failed: {reason}")
                    except Exception:
                        logger.debug("Failed to check quality gate for %s", self._agent_type.value, exc_info=True)

                # Feed PromptEvolver with the quality result for the active variant
                try:
                    from vetinari.learning.prompt_evolver import get_prompt_evolver
                    _, v_id = get_prompt_evolver().select_prompt(self._agent_type.value)
                    if v_id and v_id != "default":
                        get_prompt_evolver().record_result(self._agent_type.value, v_id, score.overall_score)
                except Exception:
                    logger.debug("Failed to record prompt evolver result for %s", self._agent_type.value, exc_info=True)

                # Record execution to training data collector
                try:
                    from vetinari.learning.training_data import get_training_collector
                    get_training_collector().record(
                        task=task.description or "",
                        prompt=self.get_system_prompt()[:500] + "\n\n" + (task.prompt or task.description or ""),
                        response=output_str,
                        score=score.overall_score,
                        model_id=model_id,
                        task_type=task_type,
                        agent_type=self._agent_type.value,
                        success=result.success,
                    )
                except Exception:
                    logger.debug("Failed to record execution to training data collector", exc_info=True)

                # Record to episodic memory
                try:
                    from vetinari.learning.episode_memory import get_episode_memory
                    get_episode_memory().record(
                        task_description=task.description or "",
                        agent_type=self._agent_type.value,
                        task_type=task_type,
                        output_summary=output_str[:300],
                        quality_score=score.overall_score,
                        success=result.success,
                        model_id=model_id,
                    )
                except Exception:
                    logger.debug("Failed to record to episodic memory", exc_info=True)

            except Exception:
                logger.debug("Learning subsystem error during task completion", exc_info=True)

        return task
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(type={self._agent_type.value}, name={self.name})>"



