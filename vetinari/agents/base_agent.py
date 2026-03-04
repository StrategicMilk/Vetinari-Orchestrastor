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
        if self._adapter_manager is None:
            # No adapter: try to use the singleton if available
            try:
                from vetinari.adapter_manager import get_adapter_manager
                self._adapter_manager = get_adapter_manager()
            except Exception:
                pass

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

        # Apply evolved system prompt if available (A/B test routing)
        _active_system_prompt = system_prompt or self.get_system_prompt()
        _variant_id = "default"
        try:
            from vetinari.learning.prompt_evolver import get_prompt_evolver
            evolved_prompt, _variant_id = get_prompt_evolver().select_prompt(self._agent_type.value)
            if evolved_prompt and evolved_prompt != _active_system_prompt:
                _active_system_prompt = evolved_prompt
        except Exception:
            pass

        # Apply token optimisation: task-specific max_tokens/temperature defaults
        try:
            from vetinari.token_optimizer import get_token_optimizer
            _optimizer = get_token_optimizer()
            _profile = _optimizer.get_task_profile(self._agent_type.value.lower())
            _profile_max_tokens, _profile_temp, _ = _profile
            # Only override if caller didn't explicitly set non-default values
            if max_tokens == 4096:
                max_tokens = _profile_max_tokens
            if temperature == 0.3:
                temperature = _profile_temp
        except Exception:
            pass

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
                return result
            else:
                self._log("warning", f"Inference failed: {response.error}")
                return ""
        except Exception as e:
            self._log("error", f"Inference exception: {e}")
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
        
        task.started_at = datetime.now().isoformat()

        # Emit structured trace span for this task
        try:
            from vetinari.structured_logging import log_event
            log_event("info", f"agent.{self._agent_type.value}", "task_started",
                      task_id=task.task_id, agent=self._agent_type.value)
        except Exception:
            pass

        # Register prompt variant if evolver is available
        try:
            from vetinari.learning.prompt_evolver import get_prompt_evolver
            evolver = get_prompt_evolver()
            evolver.register_baseline(self._agent_type.value, self.get_system_prompt())
        except Exception:
            pass

        return task
    
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
            pass

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

                # Feed PromptEvolver with the quality result for the active variant
                try:
                    from vetinari.learning.prompt_evolver import get_prompt_evolver
                    _, v_id = get_prompt_evolver().select_prompt(self._agent_type.value)
                    if v_id and v_id != "default":
                        get_prompt_evolver().record_result(self._agent_type.value, v_id, score.overall_score)
                except Exception:
                    pass

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
                    pass

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
                    pass

            except Exception:
                pass  # Learning subsystem errors must never crash agents

        return task
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(type={self._agent_type.value}, name={self.name})>"



