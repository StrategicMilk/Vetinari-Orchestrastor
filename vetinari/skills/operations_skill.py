"""
Unified Operations Skill Tool
================================
Consolidated skill tool for the OPERATIONS agent role.

Unifies capabilities from 7 legacy agents:
  - SYNTHESIZER -> synthesis mode
  - DOCUMENTATION_AGENT -> documentation mode
  - COST_PLANNER -> cost_analysis mode
  - EXPERIMENTATION_MANAGER -> experiment mode
  - IMPROVEMENT -> improvement mode
  - ERROR_RECOVERY -> error_recovery mode
  - IMAGE_GENERATOR -> image_generation mode

Plus creative_writing mode.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging
from enum import Enum

from vetinari.tool_interface import (
    Tool,
    ToolMetadata,
    ToolResult,
    ToolParameter,
    ToolCategory,
)
from vetinari.execution_context import ToolPermission, ExecutionMode
from vetinari.types import ThinkingMode  # canonical enum from types.py

logger = logging.getLogger(__name__)


class OperationsMode(str, Enum):
    """Modes of the unified operations skill."""
    DOCUMENTATION = "documentation"
    CREATIVE_WRITING = "creative_writing"
    COST_ANALYSIS = "cost_analysis"
    EXPERIMENT = "experiment"
    ERROR_RECOVERY = "error_recovery"
    SYNTHESIS = "synthesis"
    IMAGE_GENERATION = "image_generation"
    IMPROVEMENT = "improvement"


class OutputFormat(str, Enum):
    """Output format options."""
    MARKDOWN = "markdown"
    HTML = "html"
    PLAIN = "plain"
    JSON = "json"


@dataclass
class OperationsRequest:
    """Request structure for operations."""
    mode: OperationsMode
    content: str
    context: Optional[str] = None
    output_format: OutputFormat = OutputFormat.MARKDOWN
    thinking_mode: ThinkingMode = ThinkingMode.MEDIUM

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "content": self.content,
            "context": self.context,
            "output_format": self.output_format.value,
            "thinking_mode": self.thinking_mode.value,
        }


@dataclass
class Section:
    """A section of generated content."""
    title: str
    content: str
    order: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"title": self.title, "content": self.content, "order": self.order}


@dataclass
class OperationsResult:
    """Result of an operations task."""
    success: bool
    content: Optional[str] = None
    content_type: Optional[str] = None
    output_format: OutputFormat = OutputFormat.MARKDOWN
    sections: List[Section] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "content": self.content,
            "type": self.content_type,
            "format": self.output_format.value,
            "sections": [s.to_dict() for s in self.sections],
            "metadata": self.metadata,
            "warnings": self.warnings,
        }


class OperationsSkillTool(Tool):
    """
    Unified tool for the OPERATIONS consolidated agent.

    Replaces: SynthesizerSkillTool, DocumentationSkill, CostPlannerSkill,
              ExperimentationManagerSkill, ImprovementSkill,
              ErrorRecoverySkill, ImageGeneratorSkill.

    Provides documentation, synthesis, cost analysis, experimentation,
    error recovery, and improvement capabilities.
    """

    def __init__(self) -> None:
        metadata = ToolMetadata(
            name="operations",
            description=(
                "Documentation, synthesis, cost analysis, experiments, error recovery, "
                "improvement, and creative writing. Use for operational and support tasks."
            ),
            category=ToolCategory.SEARCH_ANALYSIS,
            version="1.1.0",
            author="Vetinari",
            parameters=[
                ToolParameter(
                    name="mode", type=str,
                    description="Operations mode to use",
                    required=True,
                    allowed_values=[m.value for m in OperationsMode],
                ),
                ToolParameter(
                    name="content", type=str,
                    description="Input content to process",
                    required=True,
                ),
                ToolParameter(
                    name="context", type=str,
                    description="Additional context",
                    required=False,
                ),
                ToolParameter(
                    name="output_format", type=str,
                    description="Output format (markdown/html/plain/json)",
                    required=False, default="markdown",
                    allowed_values=[f.value for f in OutputFormat],
                ),
                ToolParameter(
                    name="thinking_mode", type=str,
                    description="Operations depth (low/medium/high/xhigh)",
                    required=False, default="medium",
                    allowed_values=[m.value for m in ThinkingMode],
                ),
            ],
            required_permissions=[
                ToolPermission.FILE_READ,
                ToolPermission.FILE_WRITE,
                ToolPermission.MODEL_INFERENCE,
            ],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=[
                "documentation", "creative", "operations",
                "cost", "recovery", "synthesis", "improvement",
            ],
        )
        super().__init__(metadata)

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute an operations task."""
        try:
            mode_str = kwargs.get("mode")
            content = kwargs.get("content")
            context = kwargs.get("context")
            format_str = kwargs.get("output_format", "markdown")
            thinking_str = kwargs.get("thinking_mode", "medium")

            if not content:
                return ToolResult(success=False, output=None, error="content parameter is required")

            try:
                mode = OperationsMode(mode_str)
            except ValueError:
                return ToolResult(success=False, output=None, error=f"Invalid mode: {mode_str}")

            try:
                output_format = OutputFormat(format_str)
            except ValueError:
                output_format = OutputFormat.MARKDOWN

            try:
                thinking_mode = ThinkingMode(thinking_str)
            except ValueError:
                thinking_mode = ThinkingMode.MEDIUM

            request = OperationsRequest(
                mode=mode, content=content, context=context,
                output_format=output_format, thinking_mode=thinking_mode,
            )

            result = self._run_mode(request)

            return ToolResult(
                success=result.success,
                output=result.to_dict(),
                error=None if result.success else "Operations task failed",
                metadata={
                    "mode": mode.value,
                    "output_format": output_format.value,
                    "sections_count": len(result.sections),
                },
            )
        except Exception as e:
            logger.error("Operations tool failed: %s", e, exc_info=True)
            return ToolResult(success=False, output=None, error=str(e))

    def _run_mode(self, request: OperationsRequest) -> OperationsResult:
        """Route to the appropriate operations mode."""
        dispatch = {
            OperationsMode.DOCUMENTATION: self._documentation,
            OperationsMode.CREATIVE_WRITING: self._creative_writing,
            OperationsMode.COST_ANALYSIS: self._cost_analysis,
            OperationsMode.EXPERIMENT: self._experiment,
            OperationsMode.ERROR_RECOVERY: self._error_recovery,
            OperationsMode.SYNTHESIS: self._synthesis,
            OperationsMode.IMAGE_GENERATION: self._image_generation,
            OperationsMode.IMPROVEMENT: self._improvement,
        }
        handler = dispatch.get(request.mode)
        if handler is None:
            return OperationsResult(success=False, content=f"Unknown mode: {request.mode.value}")
        return handler(request)

    def _documentation(self, request: OperationsRequest) -> OperationsResult:
        """Generate documentation."""
        logger.info("Generating documentation: %s", request.content[:80])
        sections = [
            Section(title="Overview", content=f"Documentation for: {request.content}", order=1),
            Section(title="Usage", content="Usage examples and getting started guide.", order=2),
            Section(title="API Reference", content="Detailed API documentation.", order=3),
            Section(title="Configuration", content="Configuration options and defaults.", order=4),
        ]
        if request.thinking_mode in (ThinkingMode.HIGH, ThinkingMode.XHIGH):
            sections.append(Section(title="Troubleshooting", content="Common issues and solutions.", order=5))
            sections.append(Section(title="Migration Guide", content="Steps for upgrading.", order=6))

        return OperationsResult(
            success=True,
            content="\n\n".join(f"## {s.title}\n\n{s.content}" for s in sections),
            content_type="documentation",
            output_format=request.output_format,
            sections=sections,
            metadata={"word_count": sum(len(s.content.split()) for s in sections)},
        )

    def _creative_writing(self, request: OperationsRequest) -> OperationsResult:
        """Generate creative writing content."""
        logger.info("Creative writing: %s", request.content[:80])
        return OperationsResult(
            success=True,
            content=f"Creative content for: {request.content}",
            content_type="creative_writing",
            output_format=request.output_format,
            sections=[Section(title="Draft", content=request.content, order=1)],
            metadata={"word_count": len(request.content.split())},
        )

    def _cost_analysis(self, request: OperationsRequest) -> OperationsResult:
        """Perform cost analysis."""
        logger.info("Cost analysis: %s", request.content[:80])
        sections = [
            Section(title="Current Cost", content="Analysis of current resource costs.", order=1),
            Section(title="Projected Cost", content="Projected costs based on growth trends.", order=2),
            Section(title="Savings Opportunities", content="Identified cost reduction opportunities.", order=3),
            Section(title="Recommendations", content="Prioritized cost optimization actions.", order=4),
        ]
        return OperationsResult(
            success=True,
            content="\n\n".join(f"## {s.title}\n\n{s.content}" for s in sections),
            content_type="cost_analysis",
            output_format=request.output_format,
            sections=sections,
            metadata={"estimated_cost": 0.0},
        )

    def _experiment(self, request: OperationsRequest) -> OperationsResult:
        """Design and run experiments."""
        logger.info("Experiment: %s", request.content[:80])
        sections = [
            Section(title="Hypothesis", content=f"Hypothesis for: {request.content}", order=1),
            Section(title="Methodology", content="Experimental methodology and controls.", order=2),
            Section(title="Success Criteria", content="Measurable success criteria.", order=3),
            Section(title="Rollback Plan", content="Steps to revert if experiment fails.", order=4),
        ]
        return OperationsResult(
            success=True,
            content="\n\n".join(f"## {s.title}\n\n{s.content}" for s in sections),
            content_type="experiment",
            output_format=request.output_format,
            sections=sections,
            metadata={"experiment_results": {"status": "designed", "phase": "pre-execution"}},
        )

    def _error_recovery(self, request: OperationsRequest) -> OperationsResult:
        """Create error recovery plan."""
        logger.info("Error recovery: %s", request.content[:80])
        sections = [
            Section(title="Root Cause Analysis", content=f"Analysis of: {request.content}", order=1),
            Section(title="Fix Steps", content="Step-by-step recovery procedure.", order=2),
            Section(title="Verification", content="How to verify the fix is complete.", order=3),
            Section(title="Prevention", content="Measures to prevent recurrence.", order=4),
        ]
        return OperationsResult(
            success=True,
            content="\n\n".join(f"## {s.title}\n\n{s.content}" for s in sections),
            content_type="error_recovery",
            output_format=request.output_format,
            sections=sections,
            warnings=["Review fix steps before applying to production"],
        )

    def _synthesis(self, request: OperationsRequest) -> OperationsResult:
        """Synthesize information from multiple sources."""
        logger.info("Synthesis: %s", request.content[:80])
        sections = [
            Section(title="Summary", content=f"Synthesis of: {request.content}", order=1),
            Section(title="Key Insights", content="Primary insights from source material.", order=2),
            Section(title="Conflicts", content="Any conflicting information across sources.", order=3),
            Section(title="Conclusion", content="Consolidated conclusion.", order=4),
        ]
        return OperationsResult(
            success=True,
            content="\n\n".join(f"## {s.title}\n\n{s.content}" for s in sections),
            content_type="synthesis",
            output_format=request.output_format,
            sections=sections,
            metadata={"source_count": 0},
        )

    def _image_generation(self, request: OperationsRequest) -> OperationsResult:
        """Handle image generation requests."""
        logger.info("Image generation: %s", request.content[:80])
        return OperationsResult(
            success=True,
            content=f"Image generation request: {request.content}",
            content_type="image_generation",
            output_format=request.output_format,
            metadata={"status": "prompt_prepared"},
            warnings=["Actual image generation requires an external image model API"],
        )

    def _improvement(self, request: OperationsRequest) -> OperationsResult:
        """Identify improvement opportunities."""
        logger.info("Improvement analysis: %s", request.content[:80])
        sections = [
            Section(title="Current State", content=f"Analysis of: {request.content}", order=1),
            Section(title="Tech Debt", content="Identified technical debt items.", order=2),
            Section(title="Quick Wins", content="Low-effort, high-impact improvements.", order=3),
            Section(title="Strategic Improvements", content="Long-term improvement roadmap.", order=4),
        ]
        return OperationsResult(
            success=True,
            content="\n\n".join(f"## {s.title}\n\n{s.content}" for s in sections),
            content_type="improvement",
            output_format=request.output_format,
            sections=sections,
        )
