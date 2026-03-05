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

    _SYSTEM = (
        "You are an expert synthesizer. Respond ONLY with valid JSON matching: "
        '{"summary": "...", "insights": ["...", "..."], "report": "optional markdown report or null"}'
    )

    _CAP_PROMPTS = {
        SynthesizerCapability.RESULT_COMBINATION: "Combine and reconcile these results into a coherent whole:\n{content}",
        SynthesizerCapability.SUMMARIZATION: "Summarize the following concisely, keeping key points:\n{content}",
        SynthesizerCapability.REPORT_GENERATION: "Generate a structured markdown report from this content:\n{content}",
        SynthesizerCapability.INSIGHT_EXTRACTION: "Extract the most important actionable insights from:\n{content}",
        SynthesizerCapability.CONSOLIDATION: "Consolidate duplicate and overlapping information from:\n{content}",
        SynthesizerCapability.PRESENTATION: "Create a presentation outline (markdown) from:\n{content}",
    }

    def _execute_capability(self, req: SynthesisRequest, exec_mode: ExecutionMode) -> SynthesisResult:
        import json, re
        if exec_mode == ExecutionMode.PLANNING:
            return SynthesisResult(success=True, summary=f"Planning: Would apply {req.capability.value} to content")
        template = self._CAP_PROMPTS.get(req.capability, "Process:\n{content}")
        user_msg = template.format(content=req.content)
        if req.context:
            user_msg += f"\n\nAdditional context: {req.context}"
        try:
            raw = self._infer(self._SYSTEM, user_msg, max_tokens=1024)
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                return SynthesisResult(
                    success=True,
                    summary=data.get("summary", raw[:300]),
                    insights=data.get("insights", []),
                    report=data.get("report"),
                )
            if raw:
                return SynthesisResult(success=True, summary=raw[:300], insights=[])
        except Exception:
            pass  # Fall through to graceful fallback
        # Graceful fallback when LLM is unavailable
        fallback_summary = f"{req.capability.value} synthesis (offline fallback — LLM unavailable)"
        fallback_insights = [f"Insight from: {req.content[:100]}"] if req.content else ["(no content)"]
        fallback_report = f"## Report\n{fallback_summary}\n\nContent: {req.content[:200]}"
        return SynthesisResult(
            success=True,
            summary=fallback_summary,
            insights=fallback_insights,
            report=fallback_report,
        )
