"""
Migration Template: New Skill / Tool

Copy this file, replace all occurrences of:
    MySkill       → YourSkillName   (PascalCase)
    my_skill      → your_skill_name (snake_case)
    my-skill      → your-skill-name (kebab-case)

Then implement each TODO section.

Location convention:
    vetinari/skills/your_skill_name.py
    vetinari/tools/your_skill_name.py   (Tool interface wrapper)
    tests/test_your_skill_name.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from vetinari.tool_interface import Tool, ToolResult
from vetinari.execution_context import ExecutionContext, ExecutionMode

logger = logging.getLogger(__name__)


# ─── Capability enum ──────────────────────────────────────────────────────────

from enum import Enum


class MySkillCapability(Enum):
    """Declare the distinct operations this skill supports."""
    # TODO: replace with your capabilities
    OPERATION_A = "operation_a"
    OPERATION_B = "operation_b"


# ─── Input schema ─────────────────────────────────────────────────────────────

@dataclass
class MySkillInput:
    """Validated input for MySkill."""
    # TODO: add your input fields
    capability: MySkillCapability = MySkillCapability.OPERATION_A
    target: str = ""
    options: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Return a list of validation error messages (empty = valid)."""
        errors: List[str] = []
        # TODO: add your validation logic
        if not self.target:
            errors.append("'target' is required")
        return errors


# ─── Output schema ────────────────────────────────────────────────────────────

@dataclass
class MySkillOutput:
    """Structured result from MySkill."""
    # TODO: add your output fields
    success: bool = False
    result: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success":  self.success,
            "result":   self.result,
            "metadata": self.metadata,
        }


# ─── Core skill implementation ────────────────────────────────────────────────

class MySkill:
    """
    MySkill — one-line description.

    Longer description of what this skill does, when to use it,
    and any important constraints.
    """

    def __init__(self, context: Optional[ExecutionContext] = None) -> None:
        self.context = context
        # TODO: initialise any resources (clients, caches, etc.)

    def run(self, skill_input: MySkillInput) -> MySkillOutput:
        """Execute the skill."""
        errors = skill_input.validate()
        if errors:
            logger.error("Invalid input: %s", errors)
            return MySkillOutput(success=False,
                                  metadata={"errors": errors})

        try:
            if skill_input.capability == MySkillCapability.OPERATION_A:
                return self._operation_a(skill_input)
            elif skill_input.capability == MySkillCapability.OPERATION_B:
                return self._operation_b(skill_input)
            else:
                return MySkillOutput(
                    success=False,
                    metadata={"error": f"Unknown capability: {skill_input.capability}"},
                )
        except Exception as exc:
            logger.exception("MySkill failed: %s", exc)
            return MySkillOutput(success=False, metadata={"error": str(exc)})

    # ------------------------------------------------------------------
    # Private implementation methods
    # ------------------------------------------------------------------

    def _operation_a(self, inp: MySkillInput) -> MySkillOutput:
        # TODO: implement operation A
        logger.debug("MySkill._operation_a target=%s", inp.target)
        return MySkillOutput(success=True, result=f"operation_a on {inp.target}")

    def _operation_b(self, inp: MySkillInput) -> MySkillOutput:
        # TODO: implement operation B
        logger.debug("MySkill._operation_b target=%s", inp.target)
        return MySkillOutput(success=True, result=f"operation_b on {inp.target}")


# ─── Tool interface wrapper ────────────────────────────────────────────────────

class MySkillTool(Tool):
    """
    Tool interface wrapper for MySkill.

    Adapts the skill to the Vetinari Tool protocol so it can be
    used by the agent orchestration system.
    """

    NAME        = "my-skill"
    DESCRIPTION = "TODO: describe what this tool does"
    VERSION     = "1.0.0"

    # Required by ExecutionMode permission system
    REQUIRED_PERMISSIONS: List[str] = []       # TODO: e.g. ["read", "write"]

    def __init__(self, context: Optional[ExecutionContext] = None) -> None:
        super().__init__(context=context)
        self._skill = MySkill(context=context)

    # ------------------------------------------------------------------
    # Tool.execute() contract
    # ------------------------------------------------------------------

    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the skill via the Tool interface.

        Keyword args:
            capability (str): One of MySkillCapability values.
            target (str):     Target to operate on.
            options (dict):   Additional options.
        """
        skill_input = MySkillInput(
            capability=MySkillCapability(kwargs.get("capability",
                                          MySkillCapability.OPERATION_A.value)),
            target=str(kwargs.get("target", "")),
            options=kwargs.get("options", {}),
        )
        output = self._skill.run(skill_input)
        return ToolResult(
            success=output.success,
            data=output.to_dict(),
            error=output.metadata.get("error") if not output.success else None,
        )

    # ------------------------------------------------------------------
    # Tool metadata
    # ------------------------------------------------------------------

    @classmethod
    def get_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for this tool's inputs."""
        return {
            "type": "object",
            "properties": {
                "capability": {
                    "type": "string",
                    "enum": [c.value for c in MySkillCapability],
                    "description": "Which capability to invoke",
                },
                "target": {
                    "type": "string",
                    "description": "Target to operate on",
                },
                "options": {
                    "type": "object",
                    "description": "Additional options",
                },
            },
            "required": ["target"],
        }
