"""Vetinari Coding Agent Package.

This package provides:
- CodeAgentEngine: In-process coding agent
- CodeTask, CodeArtifact: Data models
- Integration with memory and plan mode
"""

from __future__ import annotations

from .engine import (
    ArtifactType,
    CodeAgentEngine,
    CodeArtifact,
    CodeTask,
    CodingTaskStatus,
    CodingTaskType,
    get_coding_agent,
    init_coding_agent,
)

__all__ = [
    "ArtifactType",
    "CodeAgentEngine",
    "CodeArtifact",
    "CodeTask",
    "CodingTaskStatus",
    "CodingTaskType",
    "get_coding_agent",
    "init_coding_agent",
]
