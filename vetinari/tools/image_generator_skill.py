"""Image Generator Skill Tool Wrapper

Returns mermaid/ascii diagram text or SVG descriptions. No actual image
generation — produces structured diagram text output only.
"""

import logging

from vetinari.tool_interface import Tool, ToolMetadata, ToolResult, ToolParameter, ToolCategory
from vetinari.execution_context import ToolPermission, ExecutionMode

logger = logging.getLogger(__name__)

_MERMAID_TEMPLATE = """graph TD
    A[{description}] --> B[Component 1]
    A --> C[Component 2]
    B --> D[Output]
    C --> D
"""

_ASCII_TEMPLATE = """+------------------+
|  {description}   |
+------------------+
        |
   +---------+
   | Comp 1  |
   +---------+
        |
   +---------+
   | Comp 2  |
   +---------+
"""


class ImageGeneratorSkillTool(Tool):
    """Tool wrapper for diagram and visualization generation."""

    def __init__(self):
        metadata = ToolMetadata(
            name="image_generator",
            description="Generate mermaid diagrams, ASCII art, or SVG descriptions for visualization",
            category=ToolCategory.SEARCH_ANALYSIS,
            parameters=[
                ToolParameter("description", str, "Description of diagram or image to generate", required=True),
                ToolParameter("format", str, "Output format: mermaid|ascii|svg", required=False),
            ],
            required_permissions=[],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=["image", "diagram", "visualization"],
        )
        super().__init__(metadata)

    def execute(self, **kwargs) -> ToolResult:
        description = kwargs.get("description", "")
        fmt = kwargs.get("format", "mermaid")

        try:
            if fmt == "mermaid":
                output = _MERMAID_TEMPLATE.format(description=description[:30])
            elif fmt == "ascii":
                output = _ASCII_TEMPLATE.format(description=description[:14])
            elif fmt == "svg":
                output = (
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">'
                    f'<rect width="200" height="100" fill="#eee"/>'
                    f'<text x="10" y="50" font-size="12">{description[:40]}</text>'
                    f'</svg>'
                )
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown format: {fmt}. Use mermaid|ascii|svg",
                )

            return ToolResult(
                success=True,
                output={"format": fmt, "content": output, "description": description},
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
