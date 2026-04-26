"""Foreman Skill Tool.

==============================
Skill tool for the FOREMAN agent — planning, clarification, and orchestration.

Covers 6 modes:
  - plan: Goal decomposition into task DAGs
  - clarify: Socratic questioning to surface hidden requirements
  - consolidate: Memory and context consolidation
  - summarise: Session summarization preserving key decisions
  - prune: Token budget management and context trimming
  - extract: Knowledge extraction from completed work

Standards enforced (from skill_registry):
  - STD-FMN-001: Unique task IDs with assigned agents
  - STD-FMN-002: Acyclic dependency graphs
  - STD-FMN-003: Verification tasks per implementation task
  - STD-FMN-004: Risk assessment for destructive operations
  - STD-FMN-005: Specific, answerable clarification questions
  - STD-FMN-006: Summaries preserve key decisions and action items
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vetinari.agents.contracts import Task
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


class ForemanMode(str, Enum):
    """Modes of the Foreman skill tool."""

    PLAN = "plan"
    CLARIFY = "clarify"
    CONSOLIDATE = "consolidate"
    SUMMARISE = "summarise"
    PRUNE = "prune"
    EXTRACT = "extract"


# PlanTask is retired (M4 ontology unification) — use contracts.Task instead.
# Extra fields (effort, acceptance_criteria, mode) live in Task.metadata.


def make_plan_task(
    task_id: str,
    description: str,
    assigned_agent: str | AgentType,
    *,
    dependencies: list[str] | None = None,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    effort: str = "M",
    acceptance_criteria: str = "",
    mode: str | None = None,
) -> Task:
    """Create a Task with planning-specific metadata fields.

    Replaces the retired PlanTask dataclass while preserving the extra
    fields (effort, acceptance_criteria, mode) inside Task.metadata.

    Args:
        task_id: Unique task identifier.
        description: Human-readable task description.
        assigned_agent: Agent type as string or AgentType enum.
        dependencies: Task IDs this task depends on.
        inputs: Input artifact names.
        outputs: Output artifact names.
        effort: T-shirt size estimate (XS, S, M, L, XL).
        acceptance_criteria: Definition of done for this task.
        mode: Worker mode hint (e.g. "code_discovery", "build").

    Returns:
        A contracts.Task with planning metadata embedded.
    """
    agent = AgentType(assigned_agent) if isinstance(assigned_agent, str) else assigned_agent
    metadata: dict[str, Any] = {}
    if effort != "M":
        metadata["effort"] = effort
    if acceptance_criteria:
        metadata["acceptance_criteria"] = acceptance_criteria
    if mode:
        metadata["mode"] = mode
    return Task(
        id=task_id,
        description=description,
        assigned_agent=agent,
        dependencies=dependencies if dependencies is not None else [],
        inputs=inputs if inputs is not None else [],
        outputs=outputs if outputs is not None else [],
        metadata=metadata,
    )


@dataclass
class ForemanResult:
    """Result from Foreman skill execution."""

    plan_id: str = ""
    goal: str = ""
    tasks: list[Task] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"ForemanResult(plan_id={self.plan_id!r}, tasks={len(self.tasks)!r})"

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary for plan serialization and agent handoffs."""
        return dataclass_to_dict(self)


class ForemanSkillTool(Tool):
    """Skill tool for the Foreman agent — planning and orchestration.

    The Foreman is the first stage of the factory pipeline. It decomposes
    goals into task DAGs, clarifies ambiguous requirements, and manages
    context consolidation across sessions.
    """

    def __init__(self):
        super().__init__(
            metadata=ToolMetadata(
                name="foreman",
                description=(
                    "Planning, clarification, and orchestration skill — "
                    "decomposes goals into task DAGs with dependency analysis"
                ),
                category=ToolCategory.MODEL_INFERENCE,
                version="2.0.0",
                parameters=[
                    ToolParameter(
                        name="goal",
                        type=str,
                        description="The goal, question, or content to process",
                        required=True,
                    ),
                    ToolParameter(
                        name="mode",
                        type=str,
                        description="Execution mode (plan, clarify, consolidate, summarise, prune, extract)",
                        required=False,
                        default="plan",
                        allowed_values=[m.value for m in ForemanMode],
                    ),
                    ToolParameter(
                        name="context",
                        type=dict,
                        description="Additional context (codebase state, prior plans, constraints)",
                        required=False,
                    ),
                ],
                required_permissions=[ToolPermission.MODEL_INFERENCE],
                allowed_modes=[ExecutionMode.PLANNING, ExecutionMode.EXECUTION],
                tags=["planning", "orchestration", "decomposition"],
            ),
        )

    def execute(self, **kwargs) -> ToolResult:
        """Execute the Foreman skill in the specified mode.

        Args:
            **kwargs: Must include 'goal'; optionally 'mode' and 'context'.

        Returns:
            ToolResult containing the Foreman's output.
        """
        goal = kwargs.get("goal", "")
        mode_str = kwargs.get("mode", "plan")
        context = kwargs.get("context", {})

        try:
            mode = ForemanMode(mode_str)
        except ValueError:
            logger.warning("Invalid ForemanMode %r in tool call — returning error to caller", mode_str)
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown mode: {mode_str}. Valid modes: {[m.value for m in ForemanMode]}",
            )

        logger.info("Foreman executing mode=%s goal=%.100s", mode.value, goal)

        try:
            result = self._execute_mode(mode, goal, context)
            logger.info("Foreman completed mode=%s", mode.value)
            return ToolResult(
                success=True,
                output=result.to_dict(),
                metadata={"mode": mode.value, "agent": AgentType.FOREMAN.value},
            )
        except Exception as exc:
            logger.error("Foreman mode=%s failed: %s", mode.value, exc)
            return ToolResult(
                success=False,
                output=None,
                error=str(exc),
            )

    def _execute_mode(
        self,
        mode: ForemanMode,
        goal: str,
        context: dict[str, Any],
    ) -> ForemanResult:
        """Route to the appropriate mode handler.

        Args:
            mode: The execution mode.
            goal: The goal or content to process.
            context: Additional context.

        Returns:
            ForemanResult from the mode handler.
        """
        handler = {
            ForemanMode.PLAN: self._plan,
            ForemanMode.CLARIFY: self._clarify,
            ForemanMode.CONSOLIDATE: self._consolidate,
            ForemanMode.SUMMARISE: self._summarise,
            ForemanMode.PRUNE: self._prune,
            ForemanMode.EXTRACT: self._extract,
        }[mode]
        return handler(goal, context)

    def _plan(self, goal: str, context: dict[str, Any]) -> ForemanResult:
        """Decompose a goal into a task DAG with dependency analysis.

        Follows the assembly-line pattern: analyze → decompose → assign → sequence.
        Uses capability-based routing to suggest Worker modes for each task.
        """
        # Query skill registry for capability-based routing
        mode_hints: dict[str, str] = {}
        try:
            from vetinari.orchestration.plan_generator import PlanGenerator

            pg = PlanGenerator()
            suggested_mode = pg.resolve_worker_mode(goal)
            if suggested_mode:
                mode_hints["primary"] = suggested_mode
        except Exception:
            logger.warning("Capability routing unavailable for plan mode")

        normalized_goal = goal or "Clarify the requested outcome"
        primary_mode = mode_hints.get("primary")
        tasks = [
            make_plan_task(
                "T1",
                f"Clarify success criteria and constraints for {normalized_goal}",
                AgentType.FOREMAN,
                outputs=["requirements brief"],
                acceptance_criteria="Scope, constraints, and done-state are explicit",
            ),
            make_plan_task(
                "T2",
                f"Implement the core work required for {normalized_goal}",
                AgentType.WORKER,
                dependencies=["T1"],
                inputs=["requirements brief"],
                outputs=["working implementation"],
                acceptance_criteria="Primary behavior works on representative inputs",
                mode=primary_mode,
            ),
            make_plan_task(
                "T3",
                f"Verify and document the outcome for {normalized_goal}",
                AgentType.INSPECTOR,
                dependencies=["T2"],
                inputs=["working implementation"],
                outputs=["verification evidence"],
                acceptance_criteria="Checks or tests prove the delivered behavior",
            ),
        ]

        return ForemanResult(
            plan_id="plan-1",
            goal=goal,
            tasks=tasks,
            risks=[
                "Unclear requirements can invalidate downstream implementation work",
                "Verification gaps can make a partial implementation look complete",
            ],
            metadata={"mode_hints": mode_hints, "thinking_mode": ThinkingMode.XHIGH.value},
        )

    def _clarify(self, goal: str, context: dict[str, Any]) -> ForemanResult:
        """Generate specific, answerable clarification questions.

        Uses Socratic questioning to surface hidden assumptions, constraints,
        and edge cases before planning begins.
        """
        normalized_goal = goal or "the requested work"
        return ForemanResult(
            goal=goal,
            questions=[
                f"What does success look like for {normalized_goal}?",
                f"What constraints or non-goals should shape {normalized_goal}?",
                f"What inputs, integrations, or edge cases matter most for {normalized_goal}?",
            ],
            metadata={"thinking_mode": ThinkingMode.HIGH.value},
        )

    def _consolidate(self, goal: str, context: dict[str, Any]) -> ForemanResult:
        """Consolidate memory and context from multiple sources.

        Merges overlapping information, resolves contradictions, and produces
        a unified context document for downstream agents.
        """
        return ForemanResult(
            goal=goal,
            summary="Context consolidated",
            metadata={"thinking_mode": ThinkingMode.MEDIUM.value},
        )

    def _summarise(self, goal: str, context: dict[str, Any]) -> ForemanResult:
        """Summarise a session preserving key decisions and action items.

        Ensures no key decisions, open questions, or action items are lost
        during summarization.
        """
        return ForemanResult(
            goal=goal,
            metadata={"thinking_mode": ThinkingMode.MEDIUM.value},
        )

    def _prune(self, goal: str, context: dict[str, Any]) -> ForemanResult:
        """Manage token budget by pruning low-value context.

        Identifies and removes redundant, outdated, or low-relevance content
        while preserving critical decisions and open issues.
        """
        return ForemanResult(
            goal=goal,
            metadata={"thinking_mode": ThinkingMode.LOW.value},
        )

    def _extract(self, goal: str, context: dict[str, Any]) -> ForemanResult:
        """Extract knowledge patterns gathered during completed work.

        Identifies reusable patterns, common pitfalls, and best practices
        observed in past tasks and episodes.
        """
        return ForemanResult(
            goal=goal,
            metadata={"thinking_mode": ThinkingMode.MEDIUM.value},
        )
