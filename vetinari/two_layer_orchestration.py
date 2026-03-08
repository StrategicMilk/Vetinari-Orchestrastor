"""
Two-Layer Orchestration System for Vetinari — backward-compatible shim.

The implementation has been split into focused modules under
``vetinari.orchestration``:

- :mod:`~vetinari.orchestration.execution_graph` — TaskNode, ExecutionGraph
- :mod:`~vetinari.orchestration.plan_generator`  — PlanGenerator
- :mod:`~vetinari.orchestration.durable_execution` — DurableExecutionEngine,
  ExecutionEvent, Checkpoint
- :mod:`~vetinari.orchestration.two_layer` — TwoLayerOrchestrator,
  get_two_layer_orchestrator, init_two_layer_orchestrator

All public names are re-exported here so that existing imports continue to
work unchanged::

    from vetinari.two_layer_orchestration import TwoLayerOrchestrator  # still works

.. deprecated::
    Import directly from ``vetinari.orchestration`` submodules instead.
    This shim will be removed in a future release.
"""

import warnings

warnings.warn(
    "vetinari.two_layer_orchestration is a backward-compatible shim. "
    "Import directly from vetinari.orchestration submodules instead "
    "(e.g. vetinari.orchestration.two_layer.TwoLayerOrchestrator).",
    DeprecationWarning,
    stacklevel=2,
)

# Layer 1: Graph data structures
from vetinari.orchestration.execution_graph import (  # noqa: F401
    ExecutionGraph,
    TaskNode,
)

# Layer 1: Planning
from vetinari.orchestration.plan_generator import PlanGenerator  # noqa: F401

# Layer 2: Durable execution
from vetinari.orchestration.durable_execution import (  # noqa: F401
    Checkpoint,
    DurableExecutionEngine,
    ExecutionEvent,
)

# Combined orchestrator + global singletons
from vetinari.orchestration.two_layer import (  # noqa: F401
    TwoLayerOrchestrator,
    get_two_layer_orchestrator,
    init_two_layer_orchestrator,
)

# Re-export canonical enums for callers that imported them from here
from vetinari.types import PlanStatus, TaskStatus  # noqa: F401

__all__ = [
    "Checkpoint",
    "DurableExecutionEngine",
    "ExecutionEvent",
    "ExecutionGraph",
    "PlanGenerator",
    "PlanStatus",
    "TaskNode",
    "TaskStatus",
    "TwoLayerOrchestrator",
    "get_two_layer_orchestrator",
    "init_two_layer_orchestrator",
]
