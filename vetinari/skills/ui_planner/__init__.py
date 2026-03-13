"""UI Planner Skill Tool Wrapper.

Migrates the ui-planner skill to the Tool interface, providing frontend design,
CSS, animations, and visual polish.


.. deprecated:: 1.1.0
   DEPRECATED: Superseded by ArchitectSkillTool (vetinari.skills.architect_skill).
   Will be removed in a future release.
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


class UIPlannerCapability(str, Enum):
    CSS_DESIGN = "css_design"
    RESPONSIVE_LAYOUT = "responsive_layout"
    ANIMATION = "animation"
    ACCESSIBILITY = "accessibility"
    DESIGN_SYSTEMS = "design_systems"
    VISUAL_POLISH = "visual_polish"


@dataclass
class UIRequest:
    capability: UIPlannerCapability
    element: str
    context: str | None = None
    thinking_mode: ThinkingMode = ThinkingMode.MEDIUM

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability.value,
            "element": self.element,
            "context": self.context,
            "thinking_mode": self.thinking_mode.value,
        }


@dataclass
class UIResult:
    success: bool
    css_code: str | None = None
    summary: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"success": self.success, "css_code": self.css_code, "summary": self.summary, "notes": self.notes}


class UIPlannerSkillTool(Tool):
    def __init__(self):
        import warnings

        warnings.warn(
            "UIPlannerSkillTool is deprecated since v1.1.0. "
            "Use ArchitectSkillTool (vetinari.skills.architect_skill) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        metadata = ToolMetadata(
            name="ui-planner",
            description="Frontend design, CSS, animations, and visual polish.",
            category=ToolCategory.CODE_EXECUTION,
            version="1.0.0",
            author="Vetinari",
            parameters=[
                ToolParameter(
                    name="capability",
                    type=str,
                    description="UI capability",
                    required=True,
                    allowed_values=[c.value for c in UIPlannerCapability],
                ),
                ToolParameter(name="element", type=str, description="Element to style", required=True),
                ToolParameter(name="context", type=str, description="Additional context", required=False),
                ToolParameter(
                    name="thinking_mode",
                    type=str,
                    description="Design depth",
                    required=False,
                    default="medium",
                    allowed_values=[m.value for m in ThinkingMode],
                ),
            ],
            required_permissions=[ToolPermission.MODEL_INFERENCE],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=["ui", "css", "design", "frontend", "animation"],
        )
        super().__init__(metadata)

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

            return ToolResult(
                success=result.success,
                output=result.to_dict(),
                error=None if result.success else "UI design failed",
                metadata={"capability": cap.value, "mode": mode.value, "exec_mode": exec_mode.value},
            )
        except Exception as e:
            logger.error("UI Planner tool failed: %s", e)
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

    def _css_design(self, req: UIRequest, exec_mode: ExecutionMode) -> UIResult:
        if exec_mode == ExecutionMode.PLANNING:
            return UIResult(success=True, summary="Planning: Would design CSS")
        css = f".{req.element} {{\n  display: block;\n  padding: 16px;\n  margin: 8px;\n}}"
        return UIResult(
            success=True,
            css_code=css,
            summary=f"CSS designed for {req.element}",
            notes=["Use semantic class names", "Consider responsive breakpoints"],
        )

    def _responsive_layout(self, req: UIRequest, exec_mode: ExecutionMode) -> UIResult:
        if exec_mode == ExecutionMode.PLANNING:
            return UIResult(success=True, summary="Planning: Would create responsive layout")
        css = f".{req.element} {{\n  display: flex;\n  flex-direction: column;\n}}\n@media (min-width: 768px) {{\n  .{req.element} {{\n    flex-direction: row;\n  }}\n}}"
        return UIResult(success=True, css_code=css, summary=f"Responsive layout for {req.element}")

    def _animation(self, req: UIRequest, exec_mode: ExecutionMode) -> UIResult:
        if exec_mode == ExecutionMode.PLANNING:
            return UIResult(success=True, summary="Planning: Would add animations")
        css = f"@keyframes fadeIn {{\n  from {{ opacity: 0; }}\n  to {{ opacity: 1; }}\n}}\n.{req.element} {{\n  animation: fadeIn 0.3s ease-in-out;\n}}"
        return UIResult(
            success=True,
            css_code=css,
            summary=f"Animation added to {req.element}",
            notes=["Consider prefers-reduced-motion", "Test on various browsers"],
        )

    def _accessibility(self, req: UIRequest, exec_mode: ExecutionMode) -> UIResult:
        if exec_mode == ExecutionMode.PLANNING:
            return UIResult(success=True, summary="Planning: Would ensure accessibility")
        css = f".{req.element}:focus-visible {{\n  outline: 2px solid #0066cc;\n  outline-offset: 2px;\n}}"
        return UIResult(
            success=True,
            css_code=css,
            summary=f"Accessibility improvements for {req.element}",
            notes=["WCAG 2.1 AA compliant", "Test with screen reader"],
        )

    def _design_systems(self, req: UIRequest, exec_mode: ExecutionMode) -> UIResult:
        if exec_mode == ExecutionMode.PLANNING:
            return UIResult(success=True, summary="Planning: Would create design system")
        css = ":root {\n  --primary-color: #0066cc;\n  --spacing-unit: 8px;\n  --border-radius: 4px;\n}"
        return UIResult(
            success=True,
            css_code=css,
            summary="Design tokens defined",
            notes=["Use CSS custom properties", "Maintain consistency"],
        )

    def _visual_polish(self, req: UIRequest, exec_mode: ExecutionMode) -> UIResult:
        if exec_mode == ExecutionMode.PLANNING:
            return UIResult(success=True, summary="Planning: Would add visual polish")
        css = f".{req.element} {{\n  box-shadow: 0 4px 6px rgba(0,0,0,0.1);\n  border-radius: 8px;\n  transition: transform 0.2s ease;\n}}"
        return UIResult(
            success=True,
            css_code=css,
            summary=f"Visual polish applied to {req.element}",
            notes=["Add subtle shadows", "Consider hover effects"],
        )
