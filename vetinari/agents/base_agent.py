"""
Vetinari Base Agent

This module defines the base agent class that all Vetinari agents inherit from.
All agents must implement the execute and verify methods.
"""

from __future__ import annotations

import logging
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
            context: Context information including available models, 
                    permissions, and other agent-specific data
        """
        self._context = context
        self._initialized = True
        self._log("info", f"Agent {self.name} initialized")
        
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
        return task
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(type={self._agent_type.value}, name={self.name})>"


class AsyncBaseAgent(BaseAgent):
    """Base class for async agents.
    
    Agents that need to perform async operations should inherit from this class.
    """
    
    async def execute_async(self, task: AgentTask) -> AgentResult:
        """Execute the given task asynchronously.
        
        Args:
            task: The task to execute
            
        Returns:
            AgentResult containing success status, output, and metadata
        """
        return self.execute(task)
    
    async def verify_async(self, output: Any) -> VerificationResult:
        """Verify the output asynchronously.
        
        Args:
            output: The output to verify
            
        Returns:
            VerificationResult with pass/fail status and issues
        """
        return self.verify(output)
