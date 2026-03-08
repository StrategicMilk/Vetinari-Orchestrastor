"""
CodeBridge compatibility shim.

All functionality has been consolidated into ``vetinari.agents.coding_bridge``.
This module re-exports the public API so existing imports continue to work.
"""

from vetinari.agents.coding_bridge import (  # noqa: F401
    # Primary class (aliased)
    CodingBridge as CodeBridge,
    CodingBridge,
    # Legacy dataclasses
    BridgeTaskType,
    BridgeTaskSpec,
    BridgeTaskStatus,
    BridgeTaskResult,
    # Unified dataclasses
    CodingTask,
    CodingTaskType,
    CodingTaskStatus,
    CodingResult,
    # Singleton helpers
    get_coding_bridge as get_code_bridge,
    init_coding_bridge as init_code_bridge,
    get_coding_bridge,
    init_coding_bridge,
)

__all__ = [
    "CodeBridge",
    "CodingBridge",
    "BridgeTaskType",
    "BridgeTaskSpec",
    "BridgeTaskStatus",
    "BridgeTaskResult",
    "CodingTask",
    "CodingTaskType",
    "CodingTaskStatus",
    "CodingResult",
    "get_code_bridge",
    "init_code_bridge",
    "get_coding_bridge",
    "init_coding_bridge",
]
