"""
Vetinari Coding Agent Package.

This package provides:
- CodeAgentEngine: In-process coding agent
- CodeTask, CodeArtifact: Data models
- Integration with memory and plan mode
"""

from .engine import (
    CodeAgentEngine,
    CodeTask,
    CodeArtifact,
    CodingTaskType,
    CodingTaskStatus,
    ArtifactType,
    get_coding_agent,
    init_coding_agent
)

__all__ = [
    "CodeAgentEngine",
    "CodeTask",
    "CodeArtifact",
    "CodingTaskType",
    "CodingTaskStatus",
    "ArtifactType",
    "get_coding_agent",
    "init_coding_agent"
]
