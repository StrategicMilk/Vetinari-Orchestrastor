"""
Synthesizer Skill Tool Wrapper

Migrates the synthesizer skill to the Tool interface, providing result combination,
summarization, and report generation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging
from enum import Enum

from vetinari.tool_interface import Tool, ToolMetadata, ToolResult, ToolParameter, ToolCategory
from vetinari.execution_context import ToolPermission, ExecutionMode

logger = logging.getLogger(__name__)


class SynthesizerCapability(str, Enum):
    RESULT_COMBINATION = "result_combination"
    SUMMARIZATION = "summarization"
    REPORT_GENERATION = "report_generation"
    INSIGHT_EXTRACTION = "insight_extraction"
    CONSOLIDATION = "consolidation"
    PRESENTATION = "presentation"


class ThinkingMode(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


@dataclass
class SynthesisRequest:
    capability: SynthesizerCapability
    content: str
    context: Optional[str] = None
    thinking_mode: ThinkingMode = ThinkingMode.MEDIUM

    def to_dict(self) -> Dict[str, Any]:
        return {"capability": self.capability.value, "content": self.content, "context": self.context, "thinking_mode": self.thinking_mode.value}


@dataclass
class SynthesisResult:
    success: bool
    summary: Optional[str] = None
    insights: List[str] = field(default_factory=list)
    report: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"success": self.success, "summary": self.summary, "insights": self.insights, "report": self.report}


class SynthesizerSkillTool(Tool):
    def __init__(self):
        metadata = ToolMetadata(
            name="synthesizer",
            description="Combine results, summarize findings, and consolidate information.",
            category=ToolCategory.SEARCH_ANALYSIS,
            version="1.0.0",
            author="Vetinari",
            parameters=[
                ToolParameter(name="capability", type=str, description="Synthesis capability", required=True, allowed_values=[c.value for c in SynthesizerCapability]),
                ToolParameter(name="content", type=str, description="Content to synthesize", required=True),
                ToolParameter(name="context", type=str, description="Additional context", required=False),
                ToolParameter(name="thinking_mode", type=str, description="Synthesis depth", required=False, default="medium", allowed_values=[m.value for m in ThinkingMode]),
            ],
            required_permissions=[ToolPermission.MODEL_INFERENCE],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=["synthesis", "summary", "consolidation", "report"],
        )
        super().__init__(metadata)

    def _try_llm_generate(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Attempt LLM-based synthesis via BaseAgent._infer_json().

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
            content = kwargs.get("content")
            context = kwargs.get("context")
            mode_str = kwargs.get("thinking_mode", "medium")

            if not content:
                return ToolResult(success=False, output=None, error="Content is required")

            try:
                cap = SynthesizerCapability(cap_str)
            except ValueError:
                return ToolResult(success=False, output=None, error=f"Invalid capability: {cap_str}")

            try:
                mode = ThinkingMode(mode_str)
            except ValueError:
                return ToolResult(success=False, output=None, error=f"Invalid thinking_mode: {mode_str}")

            req = SynthesisRequest(capability=cap, content=content, context=context, thinking_mode=mode)
            ctx = self._context_manager.current_context
            exec_mode = ctx.mode
            result = self._execute_capability(req, exec_mode)

            return ToolResult(success=result.success, output=result.to_dict(), error=None if result.success else "Synthesis failed", metadata={"capability": cap.value, "mode": mode.value, "exec_mode": exec_mode.value})
        except Exception as e:
            logger.error(f"Synthesizer tool failed: {e}")
            return ToolResult(success=False, output=None, error=str(e))

    def _execute_capability(self, req: SynthesisRequest, exec_mode: ExecutionMode) -> SynthesisResult:
        cap = req.capability
        if cap == SynthesizerCapability.RESULT_COMBINATION:
            return self._combine_results(req, exec_mode)
        elif cap == SynthesizerCapability.SUMMARIZATION:
            return self._summarize(req, exec_mode)
        elif cap == SynthesizerCapability.REPORT_GENERATION:
            return self._generate_report(req, exec_mode)
        elif cap == SynthesizerCapability.INSIGHT_EXTRACTION:
            return self._extract_insights(req, exec_mode)
        elif cap == SynthesizerCapability.CONSOLIDATION:
            return self._consolidate(req, exec_mode)
        elif cap == SynthesizerCapability.PRESENTATION:
            return self._present(req, exec_mode)
        return SynthesisResult(success=False, summary="Unknown capability")

    def _llm_synthesize(self, req: SynthesisRequest, capability_label: str) -> SynthesisResult:
        """Common LLM-based synthesis for all capabilities."""
        prompt = (
            f"You are a synthesis engine. Perform the following and return JSON with keys: "
            f"summary (string), insights (list of strings), report (string or null).\n\n"
            f"Capability: {capability_label}\n"
            f"Content to synthesize:\n{req.content}\n"
        )
        if req.context:
            prompt += f"Context: {req.context}\n"
        prompt += f"Thinking depth: {req.thinking_mode.value}\n"

        llm_result = self._try_llm_generate(prompt)
        if llm_result:
            return SynthesisResult(
                success=True,
                summary=llm_result.get("summary"),
                insights=llm_result.get("insights", []),
                report=llm_result.get("report"),
            )

        return SynthesisResult(
            success=False,
            summary="LLM inference unavailable",
            insights=[],
            report=None,
        )

    def _combine_results(self, req: SynthesisRequest, exec_mode: ExecutionMode) -> SynthesisResult:
        if exec_mode == ExecutionMode.PLANNING:
            return SynthesisResult(success=True, summary="Planning: Would combine results")
        return self._llm_synthesize(req, "result_combination")

    def _summarize(self, req: SynthesisRequest, exec_mode: ExecutionMode) -> SynthesisResult:
        if exec_mode == ExecutionMode.PLANNING:
            return SynthesisResult(success=True, summary="Planning: Would summarize content")
        return self._llm_synthesize(req, "summarization")

    def _generate_report(self, req: SynthesisRequest, exec_mode: ExecutionMode) -> SynthesisResult:
        if exec_mode == ExecutionMode.PLANNING:
            return SynthesisResult(success=True, summary="Planning: Would generate report")
        return self._llm_synthesize(req, "report_generation")

    def _extract_insights(self, req: SynthesisRequest, exec_mode: ExecutionMode) -> SynthesisResult:
        if exec_mode == ExecutionMode.PLANNING:
            return SynthesisResult(success=True, summary="Planning: Would extract insights")
        return self._llm_synthesize(req, "insight_extraction")

    def _consolidate(self, req: SynthesisRequest, exec_mode: ExecutionMode) -> SynthesisResult:
        if exec_mode == ExecutionMode.PLANNING:
            return SynthesisResult(success=True, summary="Planning: Would consolidate information")
        return self._llm_synthesize(req, "consolidation")

    def _present(self, req: SynthesisRequest, exec_mode: ExecutionMode) -> SynthesisResult:
        if exec_mode == ExecutionMode.PLANNING:
            return SynthesisResult(success=True, summary="Planning: Would prepare presentation")
        return self._llm_synthesize(req, "presentation")
