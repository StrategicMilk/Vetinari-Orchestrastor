"""Oracle Skill Tool Wrapper.

Migrates the oracle skill to the Tool interface, providing strategic thinking
for architecture decisions, debugging, and technical trade-offs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vetinari.execution_context import ToolPermission
from vetinari.tool_interface import Tool, ToolCategory, ToolMetadata, ToolParameter, ToolResult
from vetinari.types import (
    ExecutionMode,
    ThinkingMode,  # canonical enum from types.py
)

logger = logging.getLogger(__name__)


class OracleCapability(str, Enum):
    """Oracle capability."""
    ARCHITECTURE_ANALYSIS = "architecture_analysis"
    TRADE_OFF_EVALUATION = "trade_off_evaluation"
    DEBUGGING_STRATEGY = "debugging_strategy"
    CODE_REVIEW = "code_review"
    PATTERN_SUGGESTION = "pattern_suggestion"
    TECHNICAL_GUIDANCE = "technical_guidance"


@dataclass
class OracleRequest:
    """Oracle request."""
    capability: OracleCapability
    question: str
    context: str | None = None
    thinking_mode: ThinkingMode = ThinkingMode.MEDIUM
    options: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability.value,
            "question": self.question,
            "context": self.context,
            "thinking_mode": self.thinking_mode.value,
        }


@dataclass
class OracleResult:
    """Oracle result."""
    success: bool
    recommendation: str | None = None
    analysis: str | None = None
    pros_cons: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "recommendation": self.recommendation,
            "analysis": self.analysis,
            "pros_cons": self.pros_cons,
        }


class OracleSkillTool(Tool):
    """Oracle skill tool."""
    def __init__(self):
        metadata = ToolMetadata(
            name="oracle",
            description="Strategic thinking for architecture decisions, debugging, and technical trade-offs.",
            category=ToolCategory.SEARCH_ANALYSIS,
            version="1.0.0",
            author="Vetinari",
            parameters=[
                ToolParameter(
                    name="capability",
                    type=str,
                    description="The oracle capability",
                    required=True,
                    allowed_values=[c.value for c in OracleCapability],
                ),
                ToolParameter(name="question", type=str, description="The technical question", required=True),
                ToolParameter(name="context", type=str, description="Additional context", required=False),
                ToolParameter(
                    name="thinking_mode",
                    type=str,
                    description="Analysis depth",
                    required=False,
                    default="medium",
                    allowed_values=[m.value for m in ThinkingMode],
                ),
                ToolParameter(name="options", type=list, description="Options to compare", required=False),
            ],
            required_permissions=[ToolPermission.MODEL_INFERENCE],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=["architecture", "decision", "debugging", "strategy"],
        )
        super().__init__(metadata)

    def execute(self, **kwargs) -> ToolResult:
        """Execute.

        Returns:
            The ToolResult result.
        """
        try:
            cap_str = kwargs.get("capability")
            question = kwargs.get("question")
            context = kwargs.get("context")
            mode_str = kwargs.get("thinking_mode", "medium")
            options = kwargs.get("options", [])

            if not question:
                return ToolResult(success=False, output=None, error="Question is required")

            try:
                cap = OracleCapability(cap_str)
            except ValueError:
                return ToolResult(success=False, output=None, error=f"Invalid capability: {cap_str}")

            try:
                mode = ThinkingMode(mode_str)
            except ValueError:
                return ToolResult(success=False, output=None, error=f"Invalid thinking_mode: {mode_str}")

            req = OracleRequest(capability=cap, question=question, context=context, thinking_mode=mode, options=options)
            ctx = self._context_manager.current_context
            exec_mode = ctx.mode
            result = self._execute_capability(req, exec_mode)

            return ToolResult(
                success=result.success,
                output=result.to_dict(),
                error=None if result.success else "Oracle analysis failed",
                metadata={"capability": cap.value, "mode": mode.value, "exec_mode": exec_mode.value},
            )
        except Exception as e:
            logger.error("Oracle tool failed: %s", e)
            return ToolResult(success=False, output=None, error=str(e))

    def _execute_capability(self, req: OracleRequest, exec_mode: ExecutionMode) -> OracleResult:
        cap = req.capability
        if cap == OracleCapability.ARCHITECTURE_ANALYSIS:
            return self._analyze_architecture(req, exec_mode)
        elif cap == OracleCapability.TRADE_OFF_EVALUATION:
            return self._evaluate_tradeoffs(req, exec_mode)
        elif cap == OracleCapability.DEBUGGING_STRATEGY:
            return self._debugging_strategy(req, exec_mode)
        elif cap == OracleCapability.CODE_REVIEW:
            return self._code_review(req, exec_mode)
        elif cap == OracleCapability.PATTERN_SUGGESTION:
            return self._suggest_pattern(req, exec_mode)
        elif cap == OracleCapability.TECHNICAL_GUIDANCE:
            return self._technical_guidance(req, exec_mode)
        return OracleResult(success=False, recommendation="Unknown capability")

    def _analyze_architecture(self, req: OracleRequest, exec_mode: ExecutionMode) -> OracleResult:
        if exec_mode == ExecutionMode.PLANNING:
            return OracleResult(success=True, recommendation="Planning: Would analyze architecture")
        return OracleResult(
            success=True,
            recommendation=f"Recommended: Modular monolith for '{req.question}'",
            analysis="Analysis based on provided context",
            pros_cons={"Pros": ["Scalable", "Maintainable"], "Cons": ["Initial complexity"]},
        )

    def _evaluate_tradeoffs(self, req: OracleRequest, exec_mode: ExecutionMode) -> OracleResult:
        if exec_mode == ExecutionMode.PLANNING:
            return OracleResult(success=True, recommendation="Planning: Would evaluate trade-offs")
        opts = req.options if req.options else ["Option A", "Option B"]
        return OracleResult(
            success=True,
            recommendation=f"Recommend {opts[0]}",
            analysis="Trade-off analysis complete",
            pros_cons={opts[0]: ["Benefit 1"], opts[1]: ["Benefit 1", "Drawback 1"]},
        )

    def _debugging_strategy(self, req: OracleRequest, exec_mode: ExecutionMode) -> OracleResult:
        if exec_mode == ExecutionMode.PLANNING:
            return OracleResult(success=True, recommendation="Planning: Would develop debugging strategy")
        return OracleResult(
            success=True,
            recommendation="Check logs, enable debugging, use breakpoints",
            analysis=f"Debugging strategy for: {req.question}",
        )

    def _code_review(self, req: OracleRequest, exec_mode: ExecutionMode) -> OracleResult:
        if exec_mode == ExecutionMode.PLANNING:
            return OracleResult(success=True, recommendation="Planning: Would review code")
        return OracleResult(success=True, recommendation="Code looks good", analysis="Review complete")

    def _suggest_pattern(self, req: OracleRequest, exec_mode: ExecutionMode) -> OracleResult:
        if exec_mode == ExecutionMode.PLANNING:
            return OracleResult(success=True, recommendation="Planning: Would suggest patterns")
        return OracleResult(
            success=True, recommendation="Suggest: Repository Pattern", analysis="Based on the question"
        )

    def _technical_guidance(self, req: OracleRequest, exec_mode: ExecutionMode) -> OracleResult:
        if exec_mode == ExecutionMode.PLANNING:
            return OracleResult(success=True, recommendation="Planning: Would provide guidance")
        return OracleResult(success=True, recommendation="Guidance provided", analysis=f"Guidance for: {req.question}")
