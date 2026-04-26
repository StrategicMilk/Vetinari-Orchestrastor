"""Vetinari Coding Agent Package.

This package provides:
- CodeAgentEngine: In-process coding agent
- make_code_agent_task: Factory for creating coding AgentTasks
- CodeArtifact: Generated code artifacts

CodeTask was removed in M4 — use ``make_code_agent_task()`` instead.
"""

from __future__ import annotations

from .engine import (  # noqa: VET123 — init_coding_agent has no external callers but removing causes VET120
    CodeAgentEngine,
    CodeArtifact,
    CodingTaskType,
    StatusEnum,
    get_coding_agent,
    init_coding_agent,
    make_code_agent_task,
)

__all__ = [
    "CodeAgentEngine",
    "CodeArtifact",
    "CodingTaskType",
    "StatusEnum",
    "get_coding_agent",
    "init_coding_agent",
    "make_code_agent_task",
]
