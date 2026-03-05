"""
Oracle Skill Tool Wrapper

Migrates the oracle skill to the Tool interface, providing strategic thinking
for architecture decisions, debugging, and technical trade-offs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging
from enum import Enum

from vetinari.tool_interface import Tool, ToolMetadata, ToolResult, ToolParameter, ToolCategory
from vetinari.execution_context import ToolPermission, ExecutionMode

logger = logging.getLogger(__name__)


class OracleCapability(str, Enum):
    ARCHITECTURE_ANALYSIS = "architecture_analysis"
    TRADE_OFF_EVALUATION = "trade_off_evaluation"
    DEBUGGING_STRATEGY = "debugging_strategy"
    CODE_REVIEW = "code_review"
    PATTERN_SUGGESTION = "pattern_suggestion"
    TECHNICAL_GUIDANCE = "technical_guidance"


class ThinkingMode(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


@dataclass
class OracleRequest:
    capability: OracleCapability
    question: str
    context: Optional[str] = None
    thinking_mode: ThinkingMode = ThinkingMode.MEDIUM
    options: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"capability": self.capability.value, "question": self.question, "context": self.context, "thinking_mode": self.thinking_mode.value}


@dataclass
class OracleResult:
    success: bool
    recommendation: Optional[str] = None
    analysis: Optional[str] = None
    pros_cons: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"success": self.success, "recommendation": self.recommendation, "analysis": self.analysis, "pros_cons": self.pros_cons}


class OracleSkillTool(Tool):
    def __init__(self):
        metadata = ToolMetadata(
            name="oracle",
            description="Strategic thinking for architecture decisions, debugging, and technical trade-offs.",
            category=ToolCategory.SEARCH_ANALYSIS,
            version="1.0.0",
            author="Vetinari",
            parameters=[
                ToolParameter(name="capability", type=str, description="The oracle capability", required=True, allowed_values=[c.value for c in OracleCapability]),
                ToolParameter(name="question", type=str, description="The technical question", required=True),
                ToolParameter(name="context", type=str, description="Additional context", required=False),
                ToolParameter(name="thinking_mode", type=str, description="Analysis depth", required=False, default="medium", allowed_values=[m.value for m in ThinkingMode]),
                ToolParameter(name="options", type=list, description="Options to compare", required=False),
            ],
            required_permissions=[ToolPermission.MODEL_INFERENCE],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=["architecture", "decision", "debugging", "strategy"],
        )
        super().__init__(metadata)

    def execute(self, **kwargs) -> ToolResult:
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

            return ToolResult(success=result.success, output=result.to_dict(), error=None if result.success else "Oracle analysis failed", metadata={"capability": cap.value, "mode": mode.value, "exec_mode": exec_mode.value})
        except Exception as e:
            logger.error(f"Oracle tool failed: {e}")
            return ToolResult(success=False, output=None, error=str(e))

    _SYSTEM_PROMPT = (
        "You are an expert technical oracle. Provide precise, actionable recommendations "
        "with clear reasoning. Structure your response as JSON: "
        '{"recommendation": "...", "analysis": "...", "pros_cons": {"Pro": [...], "Con": [...]}}'
    )

    def _execute_capability(self, req: OracleRequest, exec_mode: ExecutionMode) -> OracleResult:
        if exec_mode == ExecutionMode.PLANNING:
            return OracleResult(
                success=True,
                recommendation=f"Planning mode: Would run {req.capability.value} analysis for '{req.question}'",
                analysis="",
            )
        cap_prompts = {
            OracleCapability.ARCHITECTURE_ANALYSIS: f"Analyze the architecture for: {req.question}",
            OracleCapability.TRADE_OFF_EVALUATION: f"Evaluate trade-offs between {req.options or ['options']} for: {req.question}",
            OracleCapability.DEBUGGING_STRATEGY: f"Provide a debugging strategy for: {req.question}",
            OracleCapability.CODE_REVIEW: f"Review this code/design and identify issues: {req.question}",
            OracleCapability.PATTERN_SUGGESTION: f"Suggest the best design pattern for: {req.question}",
            OracleCapability.TECHNICAL_GUIDANCE: f"Provide technical guidance for: {req.question}",
        }
        user_msg = cap_prompts.get(req.capability, f"Answer: {req.question}")
        if req.context:
            user_msg += f"\n\nContext:\n{req.context}"

        try:
            import json
            raw = self._infer(self._SYSTEM_PROMPT, user_msg, max_tokens=1024)
            # Try to parse JSON from response
            import re
            m = re.search(r'\{.*?\}', raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                return OracleResult(
                    success=True,
                    recommendation=data.get("recommendation", raw[:300]),
                    analysis=data.get("analysis", ""),
                    pros_cons=data.get("pros_cons", {}),
                )
            if raw:
                return OracleResult(success=True, recommendation=raw[:500], analysis="")
        except Exception as e:
            logger.warning(f"Oracle LLM call failed: {e}")
        # Graceful fallback when LLM is unavailable
        return OracleResult(
            success=True,
            recommendation=f"{req.capability.value} analysis (offline fallback — LLM unavailable)",
            analysis="",
        )
