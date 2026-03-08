"""
Multi-Mode Agent Base Class (Phase 3)
======================================
Base class for consolidated agents that support multiple operational modes.

Each consolidated agent replaces several legacy agents by routing tasks
to the appropriate mode based on task context. This preserves the specialized
logic of each legacy agent while reducing the agent count from 22 to 8.

Usage::

    class QualityAgent(MultiModeAgent):
        MODES = {
            "code_review": "_execute_code_review",
            "security_audit": "_execute_security_audit",
            "test_generation": "_execute_test_generation",
        }

        def _execute_code_review(self, task): ...
        def _execute_security_audit(self, task): ...
        def _execute_test_generation(self, task): ...
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult,
)

logger = logging.getLogger(__name__)


class MultiModeAgent(BaseAgent):
    """Base class for consolidated multi-mode agents.

    Subclasses define ``MODES`` (a dict mapping mode names to handler method
    names) and ``DEFAULT_MODE``.  The mode is determined from
    ``task.context["mode"]`` or inferred from the task description.
    """

    # Subclasses override these
    MODES: Dict[str, str] = {}           # mode_name -> method_name
    DEFAULT_MODE: str = ""               # fallback mode
    MODE_KEYWORDS: Dict[str, List[str]] = {}  # mode_name -> keyword list

    # Legacy agent type -> mode mapping for backward compatibility
    LEGACY_TYPE_TO_MODE: Dict[str, str] = {}  # AgentType.value -> mode_name

    def __init__(self, agent_type: AgentType, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_type, config)
        self._current_mode: Optional[str] = None

    @property
    def current_mode(self) -> Optional[str]:
        """Return the currently active mode."""
        return self._current_mode

    @property
    def available_modes(self) -> List[str]:
        """Return all available modes."""
        return list(self.MODES.keys())

    def _resolve_mode(self, task: AgentTask) -> str:
        """Determine which mode to use for a task.

        Resolution order:
        1. Explicit ``task.context["mode"]``
        2. Legacy agent type mapping (``task.context["legacy_agent_type"]``)
        3. Keyword matching against task description
        4. DEFAULT_MODE fallback
        """
        # 1. Explicit mode
        mode = task.context.get("mode", "")
        if mode and mode in self.MODES:
            return mode

        # 2. Legacy agent type mapping
        legacy_type = task.context.get("legacy_agent_type", "")
        if legacy_type and legacy_type in self.LEGACY_TYPE_TO_MODE:
            return self.LEGACY_TYPE_TO_MODE[legacy_type]

        # Also check the task's agent_type if it's a legacy type
        if hasattr(task, 'agent_type') and hasattr(task.agent_type, 'value'):
            agent_val = task.agent_type.value
            if agent_val in self.LEGACY_TYPE_TO_MODE:
                return self.LEGACY_TYPE_TO_MODE[agent_val]

        # 3. Keyword matching
        desc = (task.description or "").lower()
        best_mode = ""
        best_score = 0
        for mode_name, keywords in self.MODE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in desc)
            if score > best_score:
                best_score = score
                best_mode = mode_name

        if best_mode:
            return best_mode

        # 4. Default
        return self.DEFAULT_MODE or (list(self.MODES.keys())[0] if self.MODES else "")

    def execute(self, task: AgentTask) -> AgentResult:
        """Route task to the appropriate mode handler."""
        task = self.prepare_task(task)

        mode = self._resolve_mode(task)
        self._current_mode = mode

        handler_name = self.MODES.get(mode)
        if not handler_name:
            self._log("error", f"Unknown mode '{mode}' for {self.agent_type.value}")
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Unknown mode '{mode}' for agent {self.agent_type.value}"],
            )

        handler = getattr(self, handler_name, None)
        if handler is None:
            self._log("error", f"Handler '{handler_name}' not found on {self.__class__.__name__}")
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Handler '{handler_name}' not implemented"],
            )

        self._log("info", f"Executing in mode '{mode}' via {handler_name}")
        try:
            result = handler(task)
        except Exception as e:
            self._log("error", f"Mode '{mode}' execution failed: {e}")
            result = AgentResult(
                success=False,
                output=None,
                errors=[f"Mode '{mode}' failed: {str(e)}"],
            )

        self.complete_task(task, result)
        return result

    def verify(self, output: Any) -> VerificationResult:
        """Default verification — subclasses can override per-mode."""
        if output is None:
            return VerificationResult(
                passed=False,
                issues=[{"message": "Output is None"}],
                score=0.0,
            )
        return VerificationResult(passed=True, score=0.7)

    def get_system_prompt(self) -> str:
        """Return mode-aware system prompt."""
        mode = self._current_mode or self.DEFAULT_MODE
        mode_prompt = self._get_mode_system_prompt(mode)
        if mode_prompt:
            return mode_prompt
        return self._get_base_system_prompt()

    def _get_base_system_prompt(self) -> str:
        """Base system prompt — subclasses override."""
        return f"You are the {self.name} agent operating in {self._current_mode or 'default'} mode."

    def _get_mode_system_prompt(self, mode: str) -> str:
        """Mode-specific system prompt — subclasses override for custom prompts."""
        return ""

    def get_capabilities(self) -> List[str]:
        """Return capabilities across all modes."""
        return list(self.MODES.keys())
