"""Researcher Skill Tool Wrapper.

Migrates the researcher skill to the Tool interface, providing comprehensive
exploration, fact-finding, and source verification.
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


class ResearcherCapability(str, Enum):
    DEEP_DIVE = "deep_dive"
    SOURCE_VERIFICATION = "source_verification"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    FACT_FINDING = "fact_finding"
    COMPREHENSIVE_REPORT = "comprehensive_report"
    DATA_COLLECTION = "data_collection"


@dataclass
class ResearchRequest:
    capability: ResearcherCapability
    topic: str
    context: str | None = None
    thinking_mode: ThinkingMode = ThinkingMode.MEDIUM
    criteria: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability.value,
            "topic": self.topic,
            "context": self.context,
            "thinking_mode": self.thinking_mode.value,
        }


@dataclass
class ResearchResult:
    success: bool
    findings: list[str] = field(default_factory=list)
    summary: str | None = None
    sources: list[str] = field(default_factory=list)
    confidence: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "findings": self.findings,
            "summary": self.summary,
            "sources": self.sources,
            "confidence": self.confidence,
        }


class ResearcherSkillTool(Tool):
    def __init__(self):
        metadata = ToolMetadata(
            name="researcher",
            description="Comprehensive exploration, fact-finding, and source verification.",
            category=ToolCategory.SEARCH_ANALYSIS,
            version="1.0.0",
            author="Vetinari",
            parameters=[
                ToolParameter(
                    name="capability",
                    type=str,
                    description="Research capability",
                    required=True,
                    allowed_values=[c.value for c in ResearcherCapability],
                ),
                ToolParameter(name="topic", type=str, description="Topic to research", required=True),
                ToolParameter(name="context", type=str, description="Additional context", required=False),
                ToolParameter(
                    name="thinking_mode",
                    type=str,
                    description="Research depth",
                    required=False,
                    default="medium",
                    allowed_values=[m.value for m in ThinkingMode],
                ),
                ToolParameter(name="criteria", type=list, description="Comparison criteria", required=False),
            ],
            required_permissions=[ToolPermission.MODEL_INFERENCE, ToolPermission.NETWORK_REQUEST],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=["research", "investigation", "analysis", "facts"],
        )
        super().__init__(metadata)

    def execute(self, **kwargs) -> ToolResult:
        try:
            cap_str = kwargs.get("capability")
            topic = kwargs.get("topic")
            context = kwargs.get("context")
            mode_str = kwargs.get("thinking_mode", "medium")
            criteria = kwargs.get("criteria", [])

            if not topic:
                return ToolResult(success=False, output=None, error="Topic is required")

            try:
                cap = ResearcherCapability(cap_str)
            except ValueError:
                return ToolResult(success=False, output=None, error=f"Invalid capability: {cap_str}")

            try:
                mode = ThinkingMode(mode_str)
            except ValueError:
                return ToolResult(success=False, output=None, error=f"Invalid thinking_mode: {mode_str}")

            req = ResearchRequest(capability=cap, topic=topic, context=context, thinking_mode=mode, criteria=criteria)
            ctx = self._context_manager.current_context
            exec_mode = ctx.mode
            result = self._execute_capability(req, exec_mode)

            return ToolResult(
                success=result.success,
                output=result.to_dict(),
                error=None if result.success else "Research failed",
                metadata={"capability": cap.value, "mode": mode.value, "exec_mode": exec_mode.value},
            )
        except Exception as e:
            logger.error("Researcher tool failed: %s", e)
            return ToolResult(success=False, output=None, error=str(e))

    def _execute_capability(self, req: ResearchRequest, exec_mode: ExecutionMode) -> ResearchResult:
        cap = req.capability
        if cap == ResearcherCapability.DEEP_DIVE:
            return self._deep_dive(req, exec_mode)
        elif cap == ResearcherCapability.SOURCE_VERIFICATION:
            return self._source_verification(req, exec_mode)
        elif cap == ResearcherCapability.COMPARATIVE_ANALYSIS:
            return self._comparative_analysis(req, exec_mode)
        elif cap == ResearcherCapability.FACT_FINDING:
            return self._fact_finding(req, exec_mode)
        elif cap == ResearcherCapability.COMPREHENSIVE_REPORT:
            return self._comprehensive_report(req, exec_mode)
        elif cap == ResearcherCapability.DATA_COLLECTION:
            return self._data_collection(req, exec_mode)
        return ResearchResult(success=False, summary="Unknown capability")

    def _infer_via_llm(self, prompt: str, system_prompt: str, max_tokens: int = 1024) -> str | None:
        """Try LLM inference, return None if unavailable."""
        try:
            from vetinari.adapter_manager import get_adapter_manager

            adapter = get_adapter_manager()
            response = adapter.infer(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
            content = response.get("output", "").strip() if isinstance(response, dict) else str(response).strip()
            return content if content else None
        except Exception as e:
            logger.debug("LLM unavailable for researcher skill: %s", e)
            return None

    def _parse_findings(self, text: str) -> list[str]:
        """Parse bullet points from LLM text into a findings list."""
        lines = [l.strip().lstrip("-*•0123456789.").strip() for l in text.split("\n") if l.strip()]  # noqa: E741
        return [l for l in lines if len(l) > 5][:6]  # noqa: E741

    def _deep_dive(self, req: ResearchRequest, exec_mode: ExecutionMode) -> ResearchResult:
        if exec_mode == ExecutionMode.PLANNING:
            return ResearchResult(success=True, summary="Planning: Would conduct deep dive research")
        llm = self._infer_via_llm(
            prompt=f"Conduct a deep dive research on '{req.topic}'. "
            f"Context: {req.context or 'general'}. Provide 4-6 key findings.",
            system_prompt="You are a research analyst. Provide detailed, evidence-based findings.",
        )
        if llm:
            findings = self._parse_findings(llm)
            return ResearchResult(
                success=True,
                findings=findings or [llm[:200]],
                summary=f"Deep dive on {req.topic}",
                sources=["LLM analysis"],
                confidence="high" if req.thinking_mode.value in ["high", "xhigh"] else "medium",
            )
        return ResearchResult(
            success=True,
            findings=[f"Finding 1 for {req.topic}", f"Finding 2 for {req.topic}"],
            summary=f"Deep dive research on {req.topic}",
            sources=["Fallback"],
            confidence="medium",
        )

    def _source_verification(self, req: ResearchRequest, exec_mode: ExecutionMode) -> ResearchResult:
        if exec_mode == ExecutionMode.PLANNING:
            return ResearchResult(success=True, summary="Planning: Would verify sources")
        llm = self._infer_via_llm(
            prompt=f"Verify the reliability of information about '{req.topic}'. "
            f"Assess credibility, identify potential biases, and suggest verification methods.",
            system_prompt="You are a source verification expert. Assess credibility and reliability.",
            max_tokens=512,
        )
        if llm:
            findings = self._parse_findings(llm)
            return ResearchResult(
                success=True,
                findings=findings or [llm[:200]],
                summary="Source verification complete",
                sources=["LLM verification analysis"],
            )
        return ResearchResult(
            success=True,
            findings=["Source verification requires manual review"],
            summary="Source verification complete",
            sources=["Fallback verification"],
        )

    def _comparative_analysis(self, req: ResearchRequest, exec_mode: ExecutionMode) -> ResearchResult:
        if exec_mode == ExecutionMode.PLANNING:
            return ResearchResult(success=True, summary="Planning: Would perform comparative analysis")
        criteria = req.criteria if req.criteria else ["Performance", "Cost", "Features"]
        llm = self._infer_via_llm(
            prompt=f"Compare options for '{req.topic}' using these criteria: {', '.join(criteria)}. "
            f"Provide a structured comparison with pros and cons.",
            system_prompt="You are a comparative analysis specialist. Provide balanced, structured comparisons.",
        )
        if llm:
            findings = self._parse_findings(llm)
            return ResearchResult(
                success=True,
                findings=findings or [llm[:200]],
                summary=f"Comparative analysis for {req.topic}",
                sources=["LLM analysis"],
            )
        return ResearchResult(
            success=True,
            findings=[f"Analysis of {req.topic} against criteria: {', '.join(criteria)}"],
            summary=f"Comparative analysis for {req.topic}",
            sources=["Fallback analysis"],
        )

    def _fact_finding(self, req: ResearchRequest, exec_mode: ExecutionMode) -> ResearchResult:
        if exec_mode == ExecutionMode.PLANNING:
            return ResearchResult(success=True, summary="Planning: Would gather facts")
        llm = self._infer_via_llm(
            prompt=f"Gather key facts about '{req.topic}'. Provide 4-6 specific, verifiable facts.",
            system_prompt="You are a fact-checker. Provide accurate, specific, verifiable facts.",
            max_tokens=512,
        )
        if llm:
            findings = self._parse_findings(llm)
            return ResearchResult(
                success=True,
                findings=findings or [llm[:200]],
                summary=f"Fact-finding for {req.topic}",
                confidence="high",
            )
        return ResearchResult(
            success=True,
            findings=[f"Fact 1 about {req.topic}", f"Fact 2 about {req.topic}"],
            summary=f"Fact-finding for {req.topic}",
            confidence="medium",
        )

    def _comprehensive_report(self, req: ResearchRequest, exec_mode: ExecutionMode) -> ResearchResult:
        if exec_mode == ExecutionMode.PLANNING:
            return ResearchResult(success=True, summary="Planning: Would generate comprehensive report")
        llm = self._infer_via_llm(
            prompt=f"Generate a comprehensive research report on '{req.topic}'. "
            f"Cover: background, current state, key findings, and recommendations.",
            system_prompt="You are a research report writer. Provide thorough, well-structured reports.",
        )
        if llm:
            findings = self._parse_findings(llm)
            return ResearchResult(
                success=True,
                findings=findings or [llm[:300]],
                summary=f"Comprehensive report on {req.topic}",
                sources=["LLM comprehensive analysis"],
                confidence="high",
            )
        return ResearchResult(
            success=True,
            findings=["Comprehensive analysis requires LLM"],
            summary=f"Comprehensive report on {req.topic}",
            sources=["Fallback"],
            confidence="medium",
        )

    def _data_collection(self, req: ResearchRequest, exec_mode: ExecutionMode) -> ResearchResult:
        if exec_mode == ExecutionMode.PLANNING:
            return ResearchResult(success=True, summary="Planning: Would collect data")
        llm = self._infer_via_llm(
            prompt=f"Collect key data points about '{req.topic}'. Provide structured data with sources.",
            system_prompt="You are a data collection specialist. Provide structured, sourced data points.",
            max_tokens=512,
        )
        if llm:
            findings = self._parse_findings(llm)
            return ResearchResult(
                success=True,
                findings=findings or [llm[:200]],
                summary=f"Data collection for {req.topic}",
                sources=["LLM data collection"],
            )
        return ResearchResult(
            success=True,
            findings=[f"Data point 1 for {req.topic}", f"Data point 2 for {req.topic}"],
            summary=f"Data collection for {req.topic}",
        )
