"""Pure formatting and content-building helpers for project routes.

Contains stateless helper functions for goal enrichment, task plan building,
and final report generation. These are extracted from the monolithic factory
in litestar_projects_api.py per VET127 (file size limit).

All functions are pure (no side effects, no closure dependencies) and can
be called independently from the route handlers.
"""

from __future__ import annotations

import re

# Default system prompt used when no project-specific prompt has been set.
DEFAULT_SYSTEM_PROMPT = (
    "You are a skilled software development assistant. "
    "Produce high-quality, well-structured code with clear documentation."
)

MAX_EXPORTED_TASK_OUTPUT_CHARS = 5000


def markdown_code_block(content: str, language: str = "text", max_chars: int | None = None) -> str:
    """Return content in a fence that cannot be closed by embedded backticks.

    Args:
        content: Raw content to redact and wrap.
        language: Optional markdown info-string.
        max_chars: Optional maximum output length before truncation.

    Returns:
        Safe fenced markdown block.
    """
    from vetinari.security.redaction import redact_text

    safe_content = redact_text("" if content is None else content)
    if max_chars is not None and len(safe_content) > max_chars:
        safe_content = safe_content[:max_chars] + "\n\n[output truncated]"
    longest_tick_run = max((len(match.group(0)) for match in re.finditer(r"`+", safe_content)), default=0)
    fence = "`" * max(3, longest_tick_run + 1)
    info = language.strip() if language else ""
    opening = f"{fence}{info}" if info else fence
    return f"{opening}\n{safe_content}\n{fence}"


def enrich_goal_with_metadata(
    goal: str,
    category: str = "",
    tech_stack: str = "",
    priority: str = "quality",
    platforms: list | None = None,
    required_features: list | None = None,
    things_to_avoid: list | None = None,
    expected_outputs: list | None = None,
) -> str:
    """Return goal enriched with project metadata for the planning engine.

    Args:
        goal: The raw high-level project goal.
        category: Optional project category label.
        tech_stack: Comma-separated technology stack description.
        priority: One of ``quality``, ``speed``, or ``balanced``.
        platforms: Target deployment platforms (e.g. ``["web", "mobile"]``).
        required_features: Features the project must include.
        things_to_avoid: Patterns or approaches to exclude.
        expected_outputs: Deliverables the plan must produce.

    Returns:
        Enriched goal string with all non-empty metadata appended.
    """
    parts = [goal]
    if category:
        parts.append(f"Category: {category}")
    if tech_stack:
        parts.append(f"Tech stack: {tech_stack}")
    if priority and priority != "quality":
        parts.append(f"Priority: {priority}")
    if platforms:
        parts.append(f"Platforms: {', '.join(platforms)}")
    if required_features:
        parts.append(f"Required features: {', '.join(required_features)}")
    if things_to_avoid:
        parts.append(f"Avoid: {', '.join(things_to_avoid)}")
    if expected_outputs:
        parts.append(f"Expected outputs: {', '.join(expected_outputs)}")
    return "\n".join(parts)


def normalize_task(t: dict, i: int, goal: str) -> dict:
    """Normalize a raw subtask dict to the required task schema.

    Ensures every task dict has the mandatory keys (``id``, ``description``,
    ``inputs``, ``outputs``, ``dependencies``, ``type``, ``agent_type``)
    so downstream code can safely access them without KeyError.

    Args:
        t: Raw task dict from the planning engine.
        i: Zero-based index within the task list (used to generate fallback id).
        goal: The project goal (unused; kept for call-site compatibility).

    Returns:
        Normalized task dict with all required keys present.
    """
    from vetinari.types import AgentType

    return {
        "id": t.get("id") or t.get("subtask_id") or f"t{i + 1}",
        "description": t.get("description") or t.get("name") or f"Task {i + 1}",
        "inputs": t.get("inputs") or [],
        "outputs": t.get("outputs") or [],
        "dependencies": t.get("dependencies") or [],
        "type": t.get("type") or "implementation",
        "agent_type": t.get("agent_type") or AgentType.WORKER.value,
    }


def build_agent_system_prompt(
    system_prompt: str = "",
    category: str = "",
    tech_stack: str = "",
    priority: str = "quality",
    platforms: list | None = None,
    required_features: list | None = None,
    things_to_avoid: list | None = None,
    expected_outputs: list | None = None,
) -> str:
    """Build the full agent system prompt from project metadata.

    Appends structured constraint sections to the base system prompt so the
    executing agent understands the project context without reading the YAML.

    Args:
        system_prompt: Base system prompt (defaults to ``DEFAULT_SYSTEM_PROMPT``).
        category: Project category label.
        tech_stack: Target technology stack.
        priority: Execution priority hint (``quality``, ``speed``, ``balanced``).
        platforms: Deployment targets.
        required_features: Mandatory features list.
        things_to_avoid: Excluded patterns/approaches.
        expected_outputs: Deliverables list.

    Returns:
        Complete system prompt string with all non-empty constraints appended.
    """
    base = system_prompt or DEFAULT_SYSTEM_PROMPT
    sections: list[str] = [base]
    if category:
        sections.append(f"\nProject category: {category}")
    if tech_stack:
        sections.append(f"Tech stack: {tech_stack}")
    if priority and priority != "quality":
        sections.append(f"Priority: {priority}")
    if platforms:
        sections.append(f"Platforms: {', '.join(platforms)}")
    if required_features:
        sections.append("Required features:\n" + "\n".join(f"- {f}" for f in required_features))
    if things_to_avoid:
        sections.append("Avoid:\n" + "\n".join(f"- {x}" for x in things_to_avoid))
    if expected_outputs:
        sections.append("Expected outputs:\n" + "\n".join(f"- {o}" for o in expected_outputs))
    return "\n".join(sections)


def build_fallback_task_plan(tasks: list, warnings: list | None = None) -> str:
    """Build a Markdown task list for when the model response is empty.

    Used as a fallback when the model fails to return a substantive response
    during new-project creation, ensuring the conversation has a usable
    initial assistant message.

    Args:
        tasks: List of normalized task dicts (must have ``id`` and ``description``).
        warnings: Optional list of warning strings from the planner.

    Returns:
        Markdown-formatted string describing the planned tasks.
    """
    lines = ["## Planned Tasks", ""]
    lines.extend(f"- **{t.get('id', '?')}**: {t.get('description', '')}" for t in tasks)
    if warnings:
        lines += ["", "### Warnings", ""] + [f"- {w}" for w in warnings]
    return "\n".join(lines)


def format_foreman_response(output: str, tasks: list) -> str:
    """Return the Foreman agent output as-is for use in the conversation.

    The Foreman's output is already formatted for the UI — pass it through
    unchanged.  The ``tasks`` parameter is kept for call-site compatibility.

    Args:
        output: Raw output string from the Foreman agent.
        tasks: Normalized task list (unused; kept for call-site compatibility).

    Returns:
        The output string unchanged.
    """
    return output


def build_final_report(project_id: str, project_config: dict, task_entries: list) -> str:
    """Build a Markdown final-delivery report from assembled task outputs.

    Args:
        project_id: The project directory name.
        project_config: Parsed project YAML dict.
        task_entries: List of dicts with keys ``id``, ``description``,
            ``assigned_model``, ``output``, and ``generated`` (file name list).

    Returns:
        Markdown string ready to write to ``final_delivery/final_report.md``.
    """
    goal = project_config.get("high_level_goal") or project_config.get("goal", "")
    lines = [
        f"# Final Report: {project_id}",
        "",
        f"**Goal:** {goal}",
        "",
        "## Task Outputs",
        "",
    ]
    for entry in task_entries:
        lines += [
            f"### {entry.get('id', '?')}: {entry.get('description', '')}",
            "",
        ]
        if entry.get("assigned_model"):
            lines.append(f"*Model: {entry['assigned_model']}*")
            lines.append("")
        output = entry.get("output", "")
        if output:
            lines += [markdown_code_block(output, max_chars=MAX_EXPORTED_TASK_OUTPUT_CHARS), ""]
        if entry.get("generated"):
            lines.append("**Generated files:** " + ", ".join(entry["generated"]))
            lines.append("")
    return "\n".join(lines)
