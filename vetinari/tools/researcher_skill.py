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
from vetinari.tools.output_validation import validate_output

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

    def _try_llm_generate(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Attempt LLM-based research via BaseAgent._infer_json().

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

            # Validate output before returning
            validation = validate_output(result, required_fields=["success"])
            if not validation["valid"]:
                logger.warning("Researcher output validation failed: %s", validation["errors"])

            return ToolResult(success=result.success, output=result.to_dict(), error=None if result.success else "Research failed", metadata={"capability": cap.value, "mode": mode.value, "exec_mode": exec_mode.value})
        except Exception as e:
            logger.error(f"Researcher tool failed: {e}")
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

    def _llm_research(self, req: ResearchRequest, capability_label: str) -> ResearchResult:
        """Common LLM-based research for all capabilities."""
        prompt = (
            f"You are a research analyst. Perform the following and return JSON with keys: "
            f"findings (list of strings), summary (string), sources (list of strings), "
            f"confidence (string: low/medium/high).\n\n"
            f"Capability: {capability_label}\n"
            f"Topic: {req.topic}\n"
        )
        if req.context:
            prompt += f"Context: {req.context}\n"
        if req.criteria:
            prompt += f"Criteria: {', '.join(req.criteria)}\n"
        prompt += f"Thinking depth: {req.thinking_mode.value}\n"

        llm_result = self._try_llm_generate(prompt)
        if llm_result:
            return ResearchResult(
                success=True,
                findings=llm_result.get("findings", []),
                summary=llm_result.get("summary"),
                sources=llm_result.get("sources", []),
                confidence=llm_result.get("confidence", "low"),
            )

        return ResearchResult(
            success=False,
            findings=[],
            summary="LLM inference unavailable",
            sources=[],
            confidence="low",
        )

    def _deep_dive(self, req: ResearchRequest, exec_mode: ExecutionMode) -> ResearchResult:
        if exec_mode == ExecutionMode.PLANNING:
            return ResearchResult(success=True, summary="Planning: Would conduct deep dive research")
        return self._llm_research(req, "deep_dive")

    def _source_verification(self, req: ResearchRequest, exec_mode: ExecutionMode) -> ResearchResult:
        if exec_mode == ExecutionMode.PLANNING:
            return ResearchResult(success=True, summary="Planning: Would verify sources")
        return self._llm_research(req, "source_verification")

    def _comparative_analysis(self, req: ResearchRequest, exec_mode: ExecutionMode) -> ResearchResult:
        if exec_mode == ExecutionMode.PLANNING:
            return ResearchResult(success=True, summary="Planning: Would perform comparative analysis")
        return self._llm_research(req, "comparative_analysis")

    def _fact_finding(self, req: ResearchRequest, exec_mode: ExecutionMode) -> ResearchResult:
        if exec_mode == ExecutionMode.PLANNING:
            return ResearchResult(success=True, summary="Planning: Would gather facts")
        return self._llm_research(req, "fact_finding")

    def _comprehensive_report(self, req: ResearchRequest, exec_mode: ExecutionMode) -> ResearchResult:
        if exec_mode == ExecutionMode.PLANNING:
            return ResearchResult(success=True, summary="Planning: Would generate comprehensive report")
        return self._llm_research(req, "comprehensive_report")

    def _data_collection(self, req: ResearchRequest, exec_mode: ExecutionMode) -> ResearchResult:
        if exec_mode == ExecutionMode.PLANNING:
            return ResearchResult(success=True, summary="Planning: Would collect data")
        return self._llm_research(req, "data_collection")
