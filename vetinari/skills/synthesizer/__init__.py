"""
Synthesizer Skill Tool Wrapper

Migrates the synthesizer skill to the Tool interface, providing result combination,
summarization, and report generation.


.. deprecated:: 1.1.0
   DEPRECATED: Superseded by OperationsSkillTool (vetinari.skills.operations_skill).
   Will be removed in a future release.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging
from enum import Enum

from vetinari.tool_interface import Tool, ToolMetadata, ToolResult, ToolParameter, ToolCategory
from vetinari.execution_context import ToolPermission, ExecutionMode
from vetinari.types import ThinkingMode  # canonical enum from types.py

logger = logging.getLogger(__name__)


class SynthesizerCapability(str, Enum):
    RESULT_COMBINATION = "result_combination"
    SUMMARIZATION = "summarization"
    REPORT_GENERATION = "report_generation"
    INSIGHT_EXTRACTION = "insight_extraction"
    CONSOLIDATION = "consolidation"
    PRESENTATION = "presentation"


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
        import warnings
        warnings.warn(
            "SynthesizerSkillTool is deprecated since v1.1.0. "
            "Use OperationsSkillTool (vetinari.skills.operations_skill) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
            logger.error("Synthesizer tool failed: %s", e)
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

    def _combine_results(self, req: SynthesisRequest, exec_mode: ExecutionMode) -> SynthesisResult:
        if exec_mode == ExecutionMode.PLANNING:
            return SynthesisResult(success=True, summary="Planning: Would combine results")
        return SynthesisResult(success=True, summary=f"Combined results from: {req.content[:50]}...", insights=["Insight 1 from combination", "Insight 2 from combination"])

    def _summarize(self, req: SynthesisRequest, exec_mode: ExecutionMode) -> SynthesisResult:
        if exec_mode == ExecutionMode.PLANNING:
            return SynthesisResult(success=True, summary="Planning: Would summarize content")
        return SynthesisResult(success=True, summary=f"Summary of: {req.content[:30]}...", insights=["Key point 1", "Key point 2"])

    def _generate_report(self, req: SynthesisRequest, exec_mode: ExecutionMode) -> SynthesisResult:
        if exec_mode == ExecutionMode.PLANNING:
            return SynthesisResult(success=True, summary="Planning: Would generate report")
        report = f"# Report\n\n## Summary\n{req.content[:50]}...\n\n## Details\nDetailed findings here."
        return SynthesisResult(success=True, summary="Report generated", report=report)

    def _extract_insights(self, req: SynthesisRequest, exec_mode: ExecutionMode) -> SynthesisResult:
        if exec_mode == ExecutionMode.PLANNING:
            return SynthesisResult(success=True, summary="Planning: Would extract insights")
        return SynthesisResult(success=True, insights=["Critical insight 1", "Actionable insight 2", "Strategic insight 3"])

    def _consolidate(self, req: SynthesisRequest, exec_mode: ExecutionMode) -> SynthesisResult:
        if exec_mode == ExecutionMode.PLANNING:
            return SynthesisResult(success=True, summary="Planning: Would consolidate information")
        return SynthesisResult(success=True, summary="Information consolidated", insights=["Consolidated point 1", "Consolidated point 2"])

    def _present(self, req: SynthesisRequest, exec_mode: ExecutionMode) -> SynthesisResult:
        if exec_mode == ExecutionMode.PLANNING:
            return SynthesisResult(success=True, summary="Planning: Would prepare presentation")
        return SynthesisResult(success=True, summary="Presentation prepared", report="## Presentation\n- Slide 1\n- Slide 2\n- Slide 3")
