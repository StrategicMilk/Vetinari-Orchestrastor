"""Goal decomposition helpers for ForemanAgent.

Contains LLM-based and keyword-based goal-to-task decomposition logic,
extracted from planner_agent.py to keep that file under the 550-line limit.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from vetinari.agents.contracts import Task
from vetinari.types import AgentType

logger = logging.getLogger(__name__)

# Keywords that indicate a vague, under-specified goal.
_VAGUE_INDICATORS = frozenset([
    "something",
    "stuff",
    "things",
    "create something",
    "make it work",
    "fix it",
    "do something",
    "help me",
    "build something",
])


def is_vague_goal(goal: str) -> bool:
    """Return True when the goal string is too vague to decompose safely.

    A goal is considered vague when it is fewer than three words, contains
    common placeholder phrases, or has no alphanumeric characters at all.

    Args:
        goal: The user goal string.

    Returns:
        True if the goal needs clarification before decomposition.
    """
    goal_lower = goal.lower().strip()
    goal_words = goal_lower.split()
    return (
        len(goal_words) < 3
        or (len(goal_words) < 5 and any(v in goal_lower for v in _VAGUE_INDICATORS))
        or not any(c.isalnum() for c in goal)
    )


def decompose_goal_llm(
    agent: Any,
    goal: str,
    context: dict[str, Any],
    max_tasks: int = 15,
) -> list[Task]:
    """Use LLM to intelligently decompose a goal into ordered tasks.

    Injects past successful decompositions as few-shot examples and
    available Worker skill modes to guide agent assignment.

    Args:
        agent: ForemanAgent instance — used to call ``_infer_json``.
        goal: The user goal string to decompose.
        context: Optional context dict; may contain ``request_spec``.
        max_tasks: Maximum number of tasks to request from the LLM.

    Returns:
        List of Task objects with dependencies and depth pre-computed.
        Returns an empty list if the LLM returns nothing useful.
    """
    available_agents = [
        AgentType.FOREMAN.value,
        AgentType.WORKER.value,
        AgentType.INSPECTOR.value,
    ]
    context_str = ""
    if context:
        context_str = f"\nContext: {json.dumps(context, default=str)[:500]}"

    # Wire RequestSpec fields into the decomposition prompt when available
    spec_str = ""
    request_spec = context.get("request_spec") if context else None
    if request_spec and hasattr(request_spec, "acceptance_criteria"):
        spec_parts = []
        if request_spec.acceptance_criteria:
            spec_parts.append(f"Acceptance criteria: {'; '.join(request_spec.acceptance_criteria)}")
        if request_spec.scope:
            spec_parts.append(f"Scope (files/modules): {', '.join(request_spec.scope)}")
        if request_spec.out_of_scope:
            spec_parts.append(f"Out of scope: {', '.join(request_spec.out_of_scope)}")
        if request_spec.constraints:
            spec_parts.append(f"Constraints: {'; '.join(request_spec.constraints)}")
        if spec_parts:
            spec_str = "\n" + "\n".join(spec_parts)

    # Recall past successful decompositions as examples
    past_examples = ""
    try:
        from vetinari.learning.episode_memory import get_episode_memory

        mem = get_episode_memory()
        episodes = mem.recall(goal, task_type="planning", k=3)
        good_episodes = [ep for ep in episodes if getattr(ep, "quality_score", 0.0) > 0.8]
        if good_episodes:
            examples = []
            for ep in good_episodes[:3]:
                summary = getattr(ep, "output_summary", "") or getattr(ep, "task_summary", "")
                if summary:
                    examples.append(f"- {summary}")
            if examples:
                past_examples = "\n\nPast successful plans for similar goals:\n" + "\n".join(examples)
                logger.info(
                    "Injected %d past decomposition examples into planning prompt",
                    len(examples),
                )
    except Exception:
        logger.warning("Episode memory unavailable for planning examples", exc_info=True)

    # Inject available capabilities from skill registry
    capabilities_str = ""
    try:
        from vetinari.skills.skill_registry import get_skill

        worker_skill = get_skill("worker")
        if worker_skill:
            capabilities_str = (
                "\n\nWORKER capabilities (use these to select the right mode):\n"
                f"  Modes: {', '.join(worker_skill.modes)}\n"
                "  For WORKER tasks, add a 'mode' field to specify the execution mode.\n"
                "  Research tasks: code_discovery, domain_research, api_lookup, lateral_thinking\n"
                "  Architecture tasks: architecture, risk_assessment, contrarian_review\n"
                "  Build tasks: build, image_generation\n"
                "  Operations tasks: documentation, cost_analysis, error_recovery, improvement"
            )
    except Exception:
        logger.warning("Could not load skill capabilities for planning prompt")

    decomp_prompt = f"""Goal: {goal}{context_str}{spec_str}{past_examples}

Available agents: {", ".join(available_agents)}{capabilities_str}

Break this goal into 3-{max_tasks} discrete, ordered tasks.
For each task specify: id (t1,t2,...), description, inputs (list), outputs (list),
dependencies (list of task ids), assigned_agent (from available agents list),
acceptance_criteria (string describing done condition).
For WORKER tasks, include a 'mode' field specifying the Worker execution mode.

Output valid JSON array of task objects only — no prose, no markdown:
[
  {{"id": "t1", "description": "...", "inputs": ["goal"], "outputs": ["spec"], "dependencies": [], "assigned_agent": "WORKER", "mode": "code_discovery", "acceptance_criteria": "..."}},
  ...
]"""

    result = agent._infer_json(decomp_prompt)
    if not result or not isinstance(result, list):
        return []

    tasks = []
    for item in result:
        if not isinstance(item, dict):
            continue
        try:
            agent_str = item.get("assigned_agent", AgentType.WORKER.value).upper()
            try:
                agent_type = AgentType[agent_str]
            except KeyError:
                agent_type = AgentType.WORKER
            t = Task(
                id=item.get("id", f"t{len(tasks) + 1}"),
                description=item.get("description", "Task"),
                inputs=item.get("inputs", []),
                outputs=item.get("outputs", []),
                dependencies=item.get("dependencies", []),
                assigned_agent=agent_type,
                depth=0,
            )
            tasks.append(t)
        except (KeyError, TypeError, ValueError):
            logger.warning("Skipping malformed task entry during plan parsing — task omitted from execution plan")
            continue

    # Recalculate actual DAG depths
    if tasks:
        id_to_task = {t.id: t for t in tasks}

        def _get_depth(task_id: str, visited: set[str]) -> int:
            """Recursively compute DAG depth for a task, cycle-safe.

            Args:
                task_id: The task identifier to evaluate.
                visited: Set of already-visited task IDs for cycle detection.

            Returns:
                Integer depth (0 for root tasks with no dependencies).
            """
            if task_id in visited:
                return 0
            visited.add(task_id)
            t = id_to_task.get(task_id)
            if not t or not t.dependencies:
                return 0
            return 1 + max(_get_depth(dep, visited) for dep in t.dependencies)

        for t in tasks:
            t.depth = _get_depth(t.id, set())

    return tasks


def decompose_goal_keyword(goal: str) -> list[Task]:
    """Keyword-based fallback decomposition when LLM is unavailable.

    Produces a minimal DAG covering analysis, setup, implementation,
    testing, documentation, and security review based on keywords
    detected in the goal string.

    Args:
        goal: The user goal string.

    Returns:
        List of Task objects with dependencies and depths assigned.
    """
    goal_lower = goal.lower()
    tasks: list[Task] = []
    task_counter = [1]

    def next_id(prefix: str = "t") -> str:
        """Generate an auto-incrementing task identifier.

        Args:
            prefix: Single-character prefix for the identifier (default ``t``).

        Returns:
            A unique task ID like ``t1``, ``t2``, etc.
        """
        tid = f"{prefix}{task_counter[0]}"
        task_counter[0] += 1
        return tid

    # Analysis task always first
    t1 = Task(
        id=next_id(),
        description="Analyze requirements and create detailed specification",
        inputs=["goal"],
        outputs=["requirements_spec", "architecture_doc"],
        dependencies=[],
        assigned_agent=AgentType.WORKER,
        depth=0,
    )
    tasks.append(t1)

    is_code_heavy = any(
        kw in goal_lower
        for kw in ["code", "implement", "build", "create", "program", "agent", "script", "app", "web", "software"]
    )
    is_ui_needed = any(kw in goal_lower for kw in ["ui", "frontend", "interface", "web", "app", "dashboard", "website"])
    is_research = any(kw in goal_lower for kw in ["research", "analyze", "investigate", "study", "review"])
    is_data = any(kw in goal_lower for kw in ["data", "database", "sql", "query", "schema"])

    t2 = Task(
        id=next_id(),
        description="Set up project structure and dependencies",
        inputs=["requirements_spec"],
        outputs=["project_structure", "package_files"],
        dependencies=[t1.id],
        assigned_agent=AgentType.WORKER,
        depth=1,
    )
    tasks.append(t2)

    if is_research:
        tasks.append(
            Task(
                id=next_id(),
                description="Conduct domain research and competitor analysis",
                inputs=["goal"],
                outputs=["research_report"],
                dependencies=[t1.id],
                assigned_agent=AgentType.WORKER,
                depth=1,
            ),
        )

    if is_code_heavy:
        t_impl = Task(
            id=next_id(),
            description="Implement core business logic and data models",
            inputs=["requirements_spec", "project_structure"],
            outputs=["core_modules"],
            dependencies=[t2.id],
            assigned_agent=AgentType.WORKER,
            depth=1,
        )
        tasks.append(t_impl)
        if is_ui_needed:
            tasks.append(
                Task(
                    id=next_id(),
                    description="Implement user interface and interactions",
                    inputs=["core_modules"],
                    outputs=["ui_components"],
                    dependencies=[t_impl.id],
                    assigned_agent=AgentType.WORKER,
                    depth=2,
                ),
            )
        tasks.append(
            Task(
                id=next_id(),
                description="Write unit tests and integration tests",
                inputs=["core_modules"],
                outputs=["test_files"],
                dependencies=[t_impl.id],
                assigned_agent=AgentType.INSPECTOR,
                depth=2,
            ),
        )

    if is_data:
        tasks.append(
            Task(
                id=next_id(),
                description="Set up database schema and data layer",
                inputs=["requirements_spec"],
                outputs=["schema_files"],
                dependencies=[t1.id],
                assigned_agent=AgentType.WORKER,
                depth=1,
            ),
        )

    last = tasks[-1]
    tasks.extend((
        Task(
            id=next_id(),
            description="Code quality review and refinement",
            inputs=[last.outputs[0] if last.outputs else "result"],
            outputs=["code_review"],
            dependencies=[last.id],
            assigned_agent=AgentType.INSPECTOR,
            depth=2,
        ),
        Task(
            id=next_id(),
            description="Generate documentation and final summary",
            inputs=["code_review"],
            outputs=["documentation"],
            dependencies=[tasks[-1].id],
            assigned_agent=AgentType.WORKER,
            depth=3,
        ),
        Task(
            id=next_id(),
            description="Security review and compliance check",
            inputs=["documentation"],
            outputs=["security_report"],
            dependencies=[tasks[-1].id],
            assigned_agent=AgentType.INSPECTOR,
            depth=4,
        ),
    ))
    return tasks
