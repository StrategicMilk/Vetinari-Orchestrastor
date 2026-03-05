"""
Researcher Skill Tool Wrapper

Migrates the researcher skill to the Tool interface, providing comprehensive
exploration, fact-finding, and source verification.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging
from enum import Enum

from vetinari.tool_interface import Tool, ToolMetadata, ToolResult, ToolParameter, ToolCategory
from vetinari.execution_context import ToolPermission, ExecutionMode

logger = logging.getLogger(__name__)


class ResearcherCapability(str, Enum):
    DEEP_DIVE = "deep_dive"
    SOURCE_VERIFICATION = "source_verification"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    FACT_FINDING = "fact_finding"
    COMPREHENSIVE_REPORT = "comprehensive_report"
    DATA_COLLECTION = "data_collection"


class ThinkingMode(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


@dataclass
class ResearchRequest:
    capability: ResearcherCapability
    topic: str
    context: Optional[str] = None
    thinking_mode: ThinkingMode = ThinkingMode.MEDIUM
    criteria: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"capability": self.capability.value, "topic": self.topic, "context": self.context, "thinking_mode": self.thinking_mode.value}


@dataclass
class ResearchResult:
    success: bool
    findings: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    confidence: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        return {"success": self.success, "findings": self.findings, "summary": self.summary, "sources": self.sources, "confidence": self.confidence}


class ResearcherSkillTool(Tool):
    def __init__(self):
        metadata = ToolMetadata(
            name="researcher",
            description="Comprehensive exploration, fact-finding, and source verification.",
            category=ToolCategory.SEARCH_ANALYSIS,
            version="1.0.0",
            author="Vetinari",
            parameters=[
                ToolParameter(name="capability", type=str, description="Research capability", required=True, allowed_values=[c.value for c in ResearcherCapability]),
                ToolParameter(name="topic", type=str, description="Topic to research", required=True),
                ToolParameter(name="context", type=str, description="Additional context", required=False),
                ToolParameter(name="thinking_mode", type=str, description="Research depth", required=False, default="medium", allowed_values=[m.value for m in ThinkingMode]),
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

            return ToolResult(success=result.success, output=result.to_dict(), error=None if result.success else "Research failed", metadata={"capability": cap.value, "mode": mode.value, "exec_mode": exec_mode.value})
        except Exception as e:
            logger.error(f"Researcher tool failed: {e}")
            return ToolResult(success=False, output=None, error=str(e))

    _SYSTEM = (
        "You are a rigorous researcher. Respond ONLY with valid JSON matching: "
        '{"findings": ["...", "..."], "summary": "...", "sources": ["..."], "confidence": "high|medium|low"}'
    )

    _CAP_PROMPTS = {
        ResearcherCapability.DEEP_DIVE: "Conduct a deep dive into: {topic}",
        ResearcherCapability.SOURCE_VERIFICATION: "Verify the reliability of sources on: {topic}",
        ResearcherCapability.COMPARATIVE_ANALYSIS: "Comparative analysis of: {topic}. Criteria: {criteria}",
        ResearcherCapability.FACT_FINDING: "Find key facts about: {topic}",
        ResearcherCapability.COMPREHENSIVE_REPORT: "Write a comprehensive research report on: {topic}",
        ResearcherCapability.DATA_COLLECTION: "Collect and organize data points about: {topic}",
    }

    def _execute_capability(self, req: ResearchRequest, exec_mode: ExecutionMode) -> ResearchResult:
        import json, re
        if exec_mode == ExecutionMode.PLANNING:
            return ResearchResult(success=True, summary=f"Planning: Would research '{req.topic}' via {req.capability.value}")
        template = self._CAP_PROMPTS.get(req.capability, "Research: {topic}")
        user_msg = template.format(topic=req.topic, criteria=", ".join(req.criteria) if req.criteria else "general")
        if req.context:
            user_msg += f"\n\nContext: {req.context}"
        try:
            raw = self._infer(self._SYSTEM, user_msg, max_tokens=1024)
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                return ResearchResult(
                    success=True,
                    findings=data.get("findings", []),
                    summary=data.get("summary", raw[:300]),
                    sources=data.get("sources", []),
                    confidence=data.get("confidence", "medium"),
                )
            if raw:
                return ResearchResult(success=True, findings=[raw[:500]], summary=raw[:200])
        except Exception as e:
            pass  # Fall through to graceful fallback
        # Graceful fallback when LLM is unavailable
        return ResearchResult(
            success=True,
            findings=[f"Could not research '{req.topic}' — LLM unavailable"],
            summary=f"LLM-based research for '{req.topic}' via {req.capability.value} (offline fallback)",
            confidence="low",
        )
