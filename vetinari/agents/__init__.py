"""
Vetinari Agents Module.

This module provides specialized agents for Vetinari's orchestration system.
"""

from .coding_bridge import (
    CodingBridge,
    CodingTask,
    CodingResult,
    CodingTaskType,
    CodingTaskStatus,
    get_coding_bridge,
    init_coding_bridge
)

__all__ = [
    "CodingBridge",
    "CodingTask", 
    "CodingResult",
    "CodingTaskType",
    "CodingTaskStatus",
    "get_coding_bridge",
    "init_coding_bridge"
]
