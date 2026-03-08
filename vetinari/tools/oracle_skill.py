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

    def _try_llm_generate(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Attempt LLM-based analysis via BaseAgent._infer_json().

        Returns parsed JSON on success, None on failure.
        """
        try:
            from vetinari.agents.base_agent import BaseAgent
            agent = BaseAgent.__new__(BaseAgent)
            if hasattr(agent, '_infer_json'):
                result = agent._infer_json(prompt, fallback=None)
                if result and isinstance(result, dict):
                    return result
        except Exception as e:
            logger.debug(f"LLM inference attempt failed: {e}")
        return None

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

    def _llm_oracle(self, req: OracleRequest, capability_label: str) -> OracleResult:
        """Common LLM-based oracle analysis for all capabilities."""
        prompt = (
            f"You are a technical oracle. Analyze the following and return JSON with keys: "
            f"recommendation (string), analysis (string), pros_cons (dict of lists).\n\n"
            f"Capability: {capability_label}\n"
            f"Question: {req.question}\n"
        )
        if req.context:
            prompt += f"Context: {req.context}\n"
        if req.options:
            prompt += f"Options to evaluate: {', '.join(req.options)}\n"
        prompt += f"Thinking depth: {req.thinking_mode.value}\n"

        llm_result = self._try_llm_generate(prompt)
        if llm_result:
            return OracleResult(
                success=True,
                recommendation=llm_result.get("recommendation"),
                analysis=llm_result.get("analysis"),
                pros_cons=llm_result.get("pros_cons", {}),
            )

        return OracleResult(
            success=False,
            recommendation="LLM inference unavailable",
            analysis=None,
        )

    def _analyze_architecture(self, req: OracleRequest, exec_mode: ExecutionMode) -> OracleResult:
        if exec_mode == ExecutionMode.PLANNING:
            return OracleResult(success=True, recommendation="Planning: Would analyze architecture")
        return self._llm_oracle(req, "architecture_analysis")

    def _evaluate_tradeoffs(self, req: OracleRequest, exec_mode: ExecutionMode) -> OracleResult:
        if exec_mode == ExecutionMode.PLANNING:
            return OracleResult(success=True, recommendation="Planning: Would evaluate trade-offs")
        return self._llm_oracle(req, "trade_off_evaluation")

    def _debugging_strategy(self, req: OracleRequest, exec_mode: ExecutionMode) -> OracleResult:
        if exec_mode == ExecutionMode.PLANNING:
            return OracleResult(success=True, recommendation="Planning: Would develop debugging strategy")
        return self._llm_oracle(req, "debugging_strategy")

    def _code_review(self, req: OracleRequest, exec_mode: ExecutionMode) -> OracleResult:
        if exec_mode == ExecutionMode.PLANNING:
            return OracleResult(success=True, recommendation="Planning: Would review code")
        return self._llm_oracle(req, "code_review")

    def _suggest_pattern(self, req: OracleRequest, exec_mode: ExecutionMode) -> OracleResult:
        if exec_mode == ExecutionMode.PLANNING:
            return OracleResult(success=True, recommendation="Planning: Would suggest patterns")
        return self._llm_oracle(req, "pattern_suggestion")

    def _technical_guidance(self, req: OracleRequest, exec_mode: ExecutionMode) -> OracleResult:
        if exec_mode == ExecutionMode.PLANNING:
            return OracleResult(success=True, recommendation="Planning: Would provide guidance")
        return self._llm_oracle(req, "technical_guidance")
