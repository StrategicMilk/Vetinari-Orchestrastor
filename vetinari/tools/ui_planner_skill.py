"""
UI Planner Skill Tool Wrapper

Migrates the ui-planner skill to the Tool interface, providing frontend design,
CSS, animations, and visual polish.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging
from enum import Enum

from vetinari.tool_interface import Tool, ToolMetadata, ToolResult, ToolParameter, ToolCategory
from vetinari.execution_context import ToolPermission, ExecutionMode

logger = logging.getLogger(__name__)


class UIPlannerCapability(str, Enum):
    CSS_DESIGN = "css_design"
    RESPONSIVE_LAYOUT = "responsive_layout"
    ANIMATION = "animation"
    ACCESSIBILITY = "accessibility"
    DESIGN_SYSTEMS = "design_systems"
    VISUAL_POLISH = "visual_polish"


class ThinkingMode(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


@dataclass
class UIRequest:
    capability: UIPlannerCapability
    element: str
    context: Optional[str] = None
    thinking_mode: ThinkingMode = ThinkingMode.MEDIUM

    def to_dict(self) -> Dict[str, Any]:
        return {"capability": self.capability.value, "element": self.element, "context": self.context, "thinking_mode": self.thinking_mode.value}


@dataclass
class UIResult:
    success: bool
    css_code: Optional[str] = None
    summary: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"success": self.success, "css_code": self.css_code, "summary": self.summary, "notes": self.notes}


class UIPlannerSkillTool(Tool):
    def __init__(self):
        metadata = ToolMetadata(
            name="ui-planner",
            description="Frontend design, CSS, animations, and visual polish.",
            category=ToolCategory.CODE_EXECUTION,
            version="1.0.0",
            author="Vetinari",
            parameters=[
                ToolParameter(name="capability", type=str, description="UI capability", required=True, allowed_values=[c.value for c in UIPlannerCapability]),
                ToolParameter(name="element", type=str, description="Element to style", required=True),
                ToolParameter(name="context", type=str, description="Additional context", required=False),
                ToolParameter(name="thinking_mode", type=str, description="Design depth", required=False, default="medium", allowed_values=[m.value for m in ThinkingMode]),
            ],
            required_permissions=[ToolPermission.MODEL_INFERENCE],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=["ui", "css", "design", "frontend", "animation"],
        )
        super().__init__(metadata)

    def _try_llm_generate(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Attempt LLM-based UI design via BaseAgent._infer_json().

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
            element = kwargs.get("element")
            context = kwargs.get("context")
            mode_str = kwargs.get("thinking_mode", "medium")

            if not element:
                return ToolResult(success=False, output=None, error="Element is required")

            try:
                cap = UIPlannerCapability(cap_str)
            except ValueError:
                return ToolResult(success=False, output=None, error=f"Invalid capability: {cap_str}")

            try:
                mode = ThinkingMode(mode_str)
            except ValueError:
                return ToolResult(success=False, output=None, error=f"Invalid thinking_mode: {mode_str}")

            req = UIRequest(capability=cap, element=element, context=context, thinking_mode=mode)
            ctx = self._context_manager.current_context
            exec_mode = ctx.mode
            result = self._execute_capability(req, exec_mode)

            return ToolResult(success=result.success, output=result.to_dict(), error=None if result.success else "UI design failed", metadata={"capability": cap.value, "mode": mode.value, "exec_mode": exec_mode.value})
        except Exception as e:
            logger.error(f"UI Planner tool failed: {e}")
            return ToolResult(success=False, output=None, error=str(e))

    def _execute_capability(self, req: UIRequest, exec_mode: ExecutionMode) -> UIResult:
        cap = req.capability
        if cap == UIPlannerCapability.CSS_DESIGN:
            return self._css_design(req, exec_mode)
        elif cap == UIPlannerCapability.RESPONSIVE_LAYOUT:
            return self._responsive_layout(req, exec_mode)
        elif cap == UIPlannerCapability.ANIMATION:
            return self._animation(req, exec_mode)
        elif cap == UIPlannerCapability.ACCESSIBILITY:
            return self._accessibility(req, exec_mode)
        elif cap == UIPlannerCapability.DESIGN_SYSTEMS:
            return self._design_systems(req, exec_mode)
        elif cap == UIPlannerCapability.VISUAL_POLISH:
            return self._visual_polish(req, exec_mode)
        return UIResult(success=False, summary="Unknown capability")

    def _llm_ui_design(self, req: UIRequest, capability_label: str) -> UIResult:
        """Common LLM-based UI design for all capabilities."""
        prompt = (
            f"You are a frontend CSS designer. Generate CSS and return JSON with keys: "
            f"css_code (string), summary (string), notes (list of strings).\n\n"
            f"Capability: {capability_label}\n"
            f"Element: {req.element}\n"
        )
        if req.context:
            prompt += f"Context: {req.context}\n"
        prompt += f"Design depth: {req.thinking_mode.value}\n"

        llm_result = self._try_llm_generate(prompt)
        if llm_result:
            return UIResult(
                success=True,
                css_code=llm_result.get("css_code"),
                summary=llm_result.get("summary"),
                notes=llm_result.get("notes", []),
            )

        return UIResult(
            success=False,
            css_code=None,
            summary="LLM inference unavailable",
            notes=[],
        )

    def _css_design(self, req: UIRequest, exec_mode: ExecutionMode) -> UIResult:
        if exec_mode == ExecutionMode.PLANNING:
            return UIResult(success=True, summary="Planning: Would design CSS")
        return self._llm_ui_design(req, "css_design")

    def _responsive_layout(self, req: UIRequest, exec_mode: ExecutionMode) -> UIResult:
        if exec_mode == ExecutionMode.PLANNING:
            return UIResult(success=True, summary="Planning: Would create responsive layout")
        return self._llm_ui_design(req, "responsive_layout")

    def _animation(self, req: UIRequest, exec_mode: ExecutionMode) -> UIResult:
        if exec_mode == ExecutionMode.PLANNING:
            return UIResult(success=True, summary="Planning: Would add animations")
        return self._llm_ui_design(req, "animation")

    def _accessibility(self, req: UIRequest, exec_mode: ExecutionMode) -> UIResult:
        if exec_mode == ExecutionMode.PLANNING:
            return UIResult(success=True, summary="Planning: Would ensure accessibility")
        return self._llm_ui_design(req, "accessibility")

    def _design_systems(self, req: UIRequest, exec_mode: ExecutionMode) -> UIResult:
        if exec_mode == ExecutionMode.PLANNING:
            return UIResult(success=True, summary="Planning: Would create design system")
        return self._llm_ui_design(req, "design_systems")

    def _visual_polish(self, req: UIRequest, exec_mode: ExecutionMode) -> UIResult:
        if exec_mode == ExecutionMode.PLANNING:
            return UIResult(success=True, summary="Planning: Would add visual polish")
        return self._llm_ui_design(req, "visual_polish")
