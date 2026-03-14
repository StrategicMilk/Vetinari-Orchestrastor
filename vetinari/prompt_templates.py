"""Prompt Templates module."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

PLANNING_PROMPT = """You are a planning agent for Vetinari, an AI orchestration system.

Your role is to:
1. Understand the user's goal
2. Break it down into granular, actionable tasks
3. Assign appropriate models to each task
4. Consider dependencies between tasks
5. Plan for verification and testing

When planning:
- Think step by step about what needs to happen
- Consider edge cases and potential issues
- Ensure each task has clear inputs and outputs
- Plan for validation at each step
- Consider the full lifecycle from idea to delivery

Output format:
- List of tasks with clear descriptions
- Dependencies between tasks
- Suggested models for each task
- Expected outputs from each task
"""

DECOMPOSITION_PROMPT = """Break down the following goal into granular, actionable tasks:

Goal: {goal}

Requirements:
- Each task should be specific and actionable
- Consider the full scope: requirements, implementation, testing, deployment, documentation
- Identify dependencies between tasks
- Assign appropriate model types to different task types
- Tasks should be small enough to complete in a single iteration

Output a detailed task breakdown with:
- Task ID and description
- Inputs and outputs
- Dependencies
- Suggested model type
- Estimated complexity
"""

TOOL_USE_PROMPT = """You have access to the following tools:

{tools}

Current context:
- Project: {project_name}
- Goal: {goal}
- Current task: {current_task}
- Workspace files: {files}

Based on the current task, determine:
1. Which tool to use
2. The exact parameters
3. What to do with the result

Think about:
- Is this the right tool for the job?
- What parameters are needed?
- How will you verify the result?
- What comes next after this tool use?

Output your reasoning, then execute the tool.
"""

EVALUATION_PROMPT = """Evaluate the results of a task execution:

Task: {task_description}
Expected: {expected_output}
Actual: {actual_output}

Consider:
- Did the execution meet the requirements?
- Are there any issues or bugs?
- Is the output complete and correct?
- What could be improved?

Output:
- Status: PASS / FAIL / PARTIAL
- Issues found (if any)
- Suggestions for improvement
"""

DECISION_PROMPT = """You need to make a decision for the project:

Decision: {decision_description}

Options:
{options}

Consider:
- Pros and cons of each option
- Impact on project timeline
- Risk assessment
- User preferences (if known)

For each option, provide:
- Brief description
- Pros
- Cons
- Risk level (low/medium/high)

Recommend one option with reasoning.
"""

REFLECTION_PROMPT = """Reflect on the current state of the project:

Project: {project_name}
Goal: {goal}
Completed tasks: {completed}
Pending tasks: {pending}
Current blockers: {blockers}

Consider:
- What's working well?
- What challenges remain?
- Are there any missed requirements?
- Should the plan be adjusted?
- Any learnings to document for future sessions?

Output your reflection and any recommendations.
"""

ORCHESTRATION_PROMPT = """You are orchestrating multiple agents to complete a complex task:

Main goal: {goal}
Available agents:
- Explorer: Codebase search and pattern matching
- Librarian: Documentation and library research
- Oracle: Strategic thinking and architecture
- UI Planner: Visual design and frontend
- Researcher: Comprehensive domain research
- Evaluator: Quality assurance and testing
- Synthesizer: Combining results from multiple agents

Current task breakdown:
{tasks}

Determine:
1. Which agents to use for which tasks
2. Order of execution (parallel vs sequential)
3. How to combine results
4. Verification strategy

Output an orchestration plan with agent assignments.
"""

MEMORY_SEARCH_PROMPT = """Search the shared memory for relevant information:

Query: {query}
Agent filter: {agent_filter}
Type filter: {type_filter}

Search the memory store for:
- Past decisions related to this query
- Previous solutions to similar problems
- Patterns discovered in previous work
- Warnings or lessons learned

Output relevant memories with:
- What was decided/found
- Who recorded it
- When
- Relevance to current task
"""

MEMORY_STORE_PROMPT = """Store important information in shared memory:

Information type: {memory_type}
Summary: {summary}
Detailed content:
{content}

Tags: {tags}

This will be shared across all agents in future sessions.
Store with appropriate categorization for easy retrieval.
"""

CONTEXT_SUMMARY_PROMPT = """Summarize the current project context for a new agent:

Project: {project_name}
Current phase: {phase}
Recent changes: {recent_changes}

Provide a concise summary covering:
- What the project is
- Current state
- Key decisions made
- Important patterns discovered
- What's pending

This will be injected into the agent's context.
"""


def get_prompt(prompt_type: str, **kwargs) -> str:
    """Get a prompt template with variables filled in.

    Returns:
        The result string.
    """
    prompts = {
        "planning": PLANNING_PROMPT,
        "decomposition": DECOMPOSITION_PROMPT,
        "tool_use": TOOL_USE_PROMPT,
        "evaluation": EVALUATION_PROMPT,
        "decision": DECISION_PROMPT,
        "reflection": REFLECTION_PROMPT,
        "orchestration": ORCHESTRATION_PROMPT,
        "memory_search": MEMORY_SEARCH_PROMPT,
        "memory_store": MEMORY_STORE_PROMPT,
        "context_summary": CONTEXT_SUMMARY_PROMPT,
    }

    template = prompts.get(prompt_type, "")

    if kwargs:
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning("Missing prompt variable: %s", e)
            return template

    return template
