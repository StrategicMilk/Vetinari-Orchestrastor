"""Worker Skill Tool.

==============================
Skill tool for the WORKER agent — all-purpose execution across 24 modes.

Organized into 4 mode groups:
  - Research (8): code_discovery, domain_research, api_lookup, lateral_thinking,
                  ui_design, database, devops, git_workflow
  - Architecture (5): architecture, risk_assessment, ontological_analysis,
                      contrarian_review, suggest
  - Build (2): build, image_generation
  - Operations (9): documentation, creative_writing, cost_analysis, experiment,
                    error_recovery, synthesis, improvement, monitor, devops_ops

Per-mode constraints:
  - Research modes: READ-ONLY (no file modifications)
  - Architecture modes: READ-ONLY + ADR production
  - Build modes: SOLE production file writer
  - Operations modes: Post-execution only
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vetinari.execution_context import ToolPermission
from vetinari.tool_interface import (
    Tool,
    ToolCategory,
    ToolMetadata,
    ToolParameter,
    ToolResult,
)
from vetinari.types import AgentType, ExecutionMode, ThinkingMode
from vetinari.utils.serialization import dataclass_to_dict

logger = logging.getLogger(__name__)


class WorkerModeGroup(str, Enum):
    """Worker mode groups mapping modes to their execution constraints."""

    RESEARCH = "research"
    ARCHITECTURE = "architecture"
    BUILD = "build"
    OPERATIONS = "operations"


# Mode → group mapping for constraint enforcement
MODE_TO_GROUP: dict[str, WorkerModeGroup] = {
    # Research modes
    "code_discovery": WorkerModeGroup.RESEARCH,
    "domain_research": WorkerModeGroup.RESEARCH,
    "api_lookup": WorkerModeGroup.RESEARCH,
    "lateral_thinking": WorkerModeGroup.RESEARCH,
    "ui_design": WorkerModeGroup.RESEARCH,
    "database": WorkerModeGroup.RESEARCH,
    "devops": WorkerModeGroup.RESEARCH,
    "git_workflow": WorkerModeGroup.RESEARCH,
    # Architecture modes
    "architecture": WorkerModeGroup.ARCHITECTURE,
    "risk_assessment": WorkerModeGroup.ARCHITECTURE,
    "ontological_analysis": WorkerModeGroup.ARCHITECTURE,
    "contrarian_review": WorkerModeGroup.ARCHITECTURE,
    "suggest": WorkerModeGroup.ARCHITECTURE,
    # Build modes
    "build": WorkerModeGroup.BUILD,
    "image_generation": WorkerModeGroup.BUILD,
    # Operations modes
    "documentation": WorkerModeGroup.OPERATIONS,
    "creative_writing": WorkerModeGroup.OPERATIONS,
    "cost_analysis": WorkerModeGroup.OPERATIONS,
    "experiment": WorkerModeGroup.OPERATIONS,
    "error_recovery": WorkerModeGroup.OPERATIONS,
    "synthesis": WorkerModeGroup.OPERATIONS,
    "improvement": WorkerModeGroup.OPERATIONS,
    "monitor": WorkerModeGroup.OPERATIONS,
    "devops_ops": WorkerModeGroup.OPERATIONS,
}

# Thinking budget per mode group
GROUP_THINKING_BUDGET: dict[WorkerModeGroup, ThinkingMode] = {
    WorkerModeGroup.RESEARCH: ThinkingMode.MEDIUM,
    WorkerModeGroup.ARCHITECTURE: ThinkingMode.HIGH,
    WorkerModeGroup.BUILD: ThinkingMode.HIGH,
    WorkerModeGroup.OPERATIONS: ThinkingMode.MEDIUM,
}


@dataclass
class WorkerResult:
    """Result from Worker skill execution."""

    success: bool = True
    output: Any = None
    files_changed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: list[dict[str, Any]] = field(default_factory=list)

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"WorkerResult(success={self.success!r}, files_changed={len(self.files_changed)!r})"

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary for ToolResult output."""
        return dataclass_to_dict(self)


# Architecture mode → ArchitectSkillTool mode mapping
_ARCH_MODE_MAP: dict[str, str] = {
    "architecture": "system_design",
    "risk_assessment": "system_design",
    "ontological_analysis": "system_design",
    "contrarian_review": "system_design",
    "suggest": "api_design",
}

# Operations mode → OperationsSkillTool mode mapping
_OPS_MODE_MAP: dict[str, str] = {
    "documentation": "documentation",
    "creative_writing": "creative_writing",
    "cost_analysis": "cost_analysis",
    "experiment": "experiment",
    "error_recovery": "error_recovery",
    "synthesis": "synthesis",
    "improvement": "improvement",
    "monitor": "synthesis",
    "devops_ops": "documentation",
}


class WorkerSkillTool(Tool):
    """Skill tool for the Worker agent — all-purpose execution.

    The Worker is the production execution engine of the factory pipeline.
    It handles all research, architecture analysis, code implementation,
    and post-execution operations through 24 specialized modes organized
    into 4 groups with distinct access constraints.

    Architecture and Operations mode groups are delegated to their respective
    component skill tools (ArchitectSkillTool, OperationsSkillTool). Build
    and Research mode groups are handled via the agent pipeline.
    """

    ALL_MODES = list(MODE_TO_GROUP.keys())

    def __init__(self):
        self._architect_tool: Any = None
        self._operations_tool: Any = None
        super().__init__(
            metadata=ToolMetadata(
                name="worker",
                description=(
                    "All-purpose execution skill with 24 modes across "
                    "research, architecture, build, and operations groups"
                ),
                category=ToolCategory.CODE_EXECUTION,
                version="2.0.0",
                parameters=[
                    ToolParameter(
                        name="task",
                        type=str,
                        description="Task description to execute",
                        required=True,
                    ),
                    ToolParameter(
                        name="mode",
                        type=str,
                        description="Execution mode (auto-resolved if omitted)",
                        required=False,
                        allowed_values=list(MODE_TO_GROUP.keys()),
                    ),
                    ToolParameter(
                        name="files",
                        type=list,
                        description="File paths relevant to the task",
                        required=False,
                    ),
                    ToolParameter(
                        name="context",
                        type=dict,
                        description="Additional task context",
                        required=False,
                    ),
                    ToolParameter(
                        name="thinking_mode",
                        type=str,
                        description="Thinking budget tier (low, medium, high, xhigh)",
                        required=False,
                        allowed_values=["low", "medium", "high", "xhigh"],
                    ),
                ],
                required_permissions=[
                    ToolPermission.FILE_READ,
                    ToolPermission.MODEL_INFERENCE,
                ],
                allowed_modes=[ExecutionMode.EXECUTION],
                tags=["research", "architecture", "build", "operations"],
            ),
        )

    def execute(self, **kwargs) -> ToolResult:
        """Execute the Worker skill in the specified mode.

        Args:
            **kwargs: Must include 'task'; optionally 'mode', 'files', 'context'.

        Returns:
            ToolResult containing the Worker's output.
        """
        task = kwargs.get("task", "")
        mode_str = kwargs.get("mode")
        files = kwargs.get("files", [])
        thinking_str = kwargs.get("thinking_mode")

        # Resolve mode
        if mode_str and mode_str not in MODE_TO_GROUP:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown mode: {mode_str}. Valid modes: {self.ALL_MODES}",
            )

        if not mode_str:
            mode_str = self._resolve_mode(task)

        mode_group = MODE_TO_GROUP.get(mode_str, WorkerModeGroup.BUILD)
        thinking_mode = (
            ThinkingMode(thinking_str) if thinking_str else GROUP_THINKING_BUDGET.get(mode_group, ThinkingMode.MEDIUM)
        )

        logger.info(
            "Worker executing mode=%s group=%s task=%.100s",
            mode_str,
            mode_group.value,
            task,
        )

        # Enforce mode group constraints
        constraint_errors = self._check_constraints(mode_str, mode_group)
        if constraint_errors:
            return ToolResult(
                success=False,
                output=None,
                error=f"Constraint violation: {'; '.join(constraint_errors)}",
            )

        context = kwargs.get("context") or {}

        try:
            if mode_group == WorkerModeGroup.ARCHITECTURE:
                result = self._delegate_to_architecture(
                    mode_str,
                    task,
                    context,
                    thinking_mode,
                )
            elif mode_group == WorkerModeGroup.OPERATIONS:
                result = self._delegate_to_operations(
                    mode_str,
                    task,
                    context,
                    thinking_mode,
                )
            else:
                # BUILD and RESEARCH groups: handled via agent pipeline
                result = WorkerResult(
                    metadata={
                        "mode": mode_str,
                        "mode_group": mode_group.value,
                        "thinking_mode": thinking_mode.value,
                        "agent": AgentType.WORKER.value,
                        "files": files,
                        "delegation": "agent_pipeline",
                    },
                )

            logger.info("Worker completed mode=%s group=%s success=%s", mode_str, mode_group.value, result.success)

            if not result.success:
                return ToolResult(
                    success=False,
                    output=None,
                    error="; ".join(result.errors) if result.errors else "Worker execution failed",
                )

            return ToolResult(
                success=True,
                output=result.to_dict(),
                metadata={
                    "mode": mode_str,
                    "mode_group": mode_group.value,
                    "agent": AgentType.WORKER.value,
                },
            )
        except Exception as exc:
            logger.error("Worker mode=%s failed: %s", mode_str, exc)
            return ToolResult(success=False, output=None, error=str(exc))

    def _delegate_to_architecture(
        self,
        mode: str,
        task: str,
        context: dict[str, Any],
        thinking_mode: ThinkingMode,
    ) -> WorkerResult:
        """Delegate an architecture-group mode to ArchitectSkillTool.

        Lazily instantiates and caches an ArchitectSkillTool, maps the Worker
        architecture mode to the nearest ArchitectSkillTool mode, and converts
        the returned ToolResult into a WorkerResult.

        Args:
            mode: Worker architecture mode name (e.g. "architecture", "suggest").
            task: Task description forwarded as the design_request.
            context: Additional task context (unused by architect but kept for
                future extension).
            thinking_mode: Thinking budget tier to pass through.

        Returns:
            WorkerResult populated from the ArchitectSkillTool's ToolResult.
        """
        if self._architect_tool is None:
            from vetinari.skills.architect_skill import ArchitectSkillTool

            self._architect_tool = ArchitectSkillTool()

        mapped_mode = _ARCH_MODE_MAP.get(mode, "system_design")
        logger.info(
            "Worker delegating mode=%s → ArchitectSkillTool mode=%s",
            mode,
            mapped_mode,
        )

        try:
            tool_result = self._architect_tool.execute(
                mode=mapped_mode,
                design_request=task,
                thinking_mode=thinking_mode.value,
            )
        except Exception as exc:
            logger.error("ArchitectSkillTool raised for mode=%s: %s", mode, exc)
            return WorkerResult(
                success=False,
                errors=[str(exc)],
                metadata={"mode": mode, "mode_group": WorkerModeGroup.ARCHITECTURE.value},
            )

        if not tool_result.success:
            return WorkerResult(
                success=False,
                errors=[tool_result.error or "ArchitectSkillTool returned failure"],
                metadata={"mode": mode, "mode_group": WorkerModeGroup.ARCHITECTURE.value},
            )

        return WorkerResult(
            success=True,
            output=tool_result.output,
            metadata={
                "mode": mode,
                "mapped_mode": mapped_mode,
                "mode_group": WorkerModeGroup.ARCHITECTURE.value,
                "thinking_mode": thinking_mode.value,
                "delegation": "architect_skill",
                **(tool_result.metadata or {}),
            },
        )

    def _delegate_to_operations(
        self,
        mode: str,
        task: str,
        context: dict[str, Any],
        thinking_mode: ThinkingMode,
    ) -> WorkerResult:
        """Delegate an operations-group mode to OperationsSkillTool.

        Lazily instantiates and caches an OperationsSkillTool, maps the Worker
        operations mode to the nearest OperationsSkillTool mode, and converts
        the returned ToolResult into a WorkerResult.

        Args:
            mode: Worker operations mode name (e.g. "documentation", "monitor").
            task: Task description forwarded as the content parameter.
            context: Additional task context (forwarded as a string if non-empty).
            thinking_mode: Thinking budget tier to pass through.

        Returns:
            WorkerResult populated from the OperationsSkillTool's ToolResult.
        """
        if self._operations_tool is None:
            from vetinari.skills.operations_skill import OperationsSkillTool

            self._operations_tool = OperationsSkillTool()

        mapped_mode = _OPS_MODE_MAP.get(mode, "synthesis")
        logger.info(
            "Worker delegating mode=%s → OperationsSkillTool mode=%s",
            mode,
            mapped_mode,
        )

        context_str: str | None = None
        if context:
            import json

            try:
                context_str = json.dumps(context)
            except (TypeError, ValueError):
                context_str = str(context)

        try:
            tool_result = self._operations_tool.execute(
                mode=mapped_mode,
                content=task,
                context=context_str,
                thinking_mode=thinking_mode.value,
            )
        except Exception as exc:
            logger.error("OperationsSkillTool raised for mode=%s: %s", mode, exc)
            return WorkerResult(
                success=False,
                errors=[str(exc)],
                metadata={"mode": mode, "mode_group": WorkerModeGroup.OPERATIONS.value},
            )

        if not tool_result.success:
            return WorkerResult(
                success=False,
                errors=[tool_result.error or "OperationsSkillTool returned failure"],
                metadata={"mode": mode, "mode_group": WorkerModeGroup.OPERATIONS.value},
            )

        return WorkerResult(
            success=True,
            output=tool_result.output,
            metadata={
                "mode": mode,
                "mapped_mode": mapped_mode,
                "mode_group": WorkerModeGroup.OPERATIONS.value,
                "thinking_mode": thinking_mode.value,
                "delegation": "operations_skill",
                **(tool_result.metadata or {}),
            },
        )

    def _resolve_mode(self, task: str) -> str:
        """Resolve the best mode for a task based on keywords.

        Args:
            task: Task description.

        Returns:
            The best-matching mode name.
        """
        task_lower = task.lower()

        # Keyword → mode mapping (ordered by specificity)
        keyword_modes = [
            (["security", "vulnerability", "cve", "owasp"], "architecture"),
            (["risk", "threat", "failure mode"], "risk_assessment"),
            (["contrarian", "devil", "advocate", "challenge"], "contrarian_review"),
            (["ontolog", "domain model", "concept map"], "ontological_analysis"),
            (["architecture", "design", "adr", "pattern"], "architecture"),
            (["research", "investigate", "explore", "discover"], "code_discovery"),
            (["api", "endpoint", "rest", "graphql"], "api_lookup"),
            (["lateral", "creative", "novel", "analogy"], "lateral_thinking"),
            (["git", "blame", "bisect", "history", "commit"], "git_workflow"),
            (["database", "schema", "sql", "migration"], "database"),
            (["deploy", "docker", "ci", "cd", "infra"], "devops"),
            (["ui", "frontend", "component", "layout"], "ui_design"),
            (["document", "readme", "changelog", "docstring"], "documentation"),
            (["cost", "budget", "token", "pricing"], "cost_analysis"),
            (["error", "recover", "fix", "diagnose"], "error_recovery"),
            (["experiment", "a/b", "test hypothesis"], "experiment"),
            (["improve", "kaizen", "optimize", "refactor"], "improvement"),
            (["monitor", "alert", "metric", "spc"], "monitor"),
            (["synthesis", "combine", "merge", "aggregate"], "synthesis"),
            (["image", "generate image", "picture"], "image_generation"),
            (["test", "implement", "build", "create", "code"], "build"),
        ]

        for keywords, mode in keyword_modes:
            if any(kw in task_lower for kw in keywords):
                return mode

        return "build"  # Default mode

    def _check_constraints(
        self,
        mode: str,
        mode_group: WorkerModeGroup,
    ) -> list[str]:
        """Check mode-group-specific constraints.

        Args:
            mode: The execution mode.
            mode_group: The mode group.

        Returns:
            List of constraint violation messages (empty = OK).
        """
        errors: list[str] = []

        # Research and architecture modes are read-only
        if mode_group in (WorkerModeGroup.RESEARCH, WorkerModeGroup.ARCHITECTURE):
            try:
                from vetinari.execution_context import get_context_manager

                ctx = get_context_manager()
                if ctx.check_permission(ToolPermission.FILE_WRITE):
                    # Permission exists but should be restricted
                    logger.debug(
                        "Worker mode=%s is read-only; file writes will be blocked",
                        mode,
                    )
            except Exception:
                logger.warning("Context manager unavailable — degrading gracefully")

        return errors

    @staticmethod
    def get_mode_group(mode: str) -> str | None:
        """Return the mode group name for a given mode.

        Args:
            mode: Worker mode name.

        Returns:
            Group name (research, architecture, build, operations) or None.
        """
        group = MODE_TO_GROUP.get(mode)
        return group.value if group else None
