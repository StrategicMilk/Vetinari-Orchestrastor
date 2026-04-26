"""Vetinari Orchestration Module.

This module provides the orchestration engine that coordinates all agents
in the hierarchical multi-agent system.

Sub-modules:
- agent_graph       — AgentGraph DAG orchestration protocol
- graph_types       — Shared dataclasses and enums (ExecutionDAG, TaskNode, etc.)
- replan_engine     — Mid-execution replanning mixin for AgentGraph
- execution_graph   — TaskNode & ExecutionGraph for plan execution
- plan_generator    — Goal decomposition into execution graphs
- durable_execution — Checkpoint-based durable execution engine
- request_routing   — RequestQueue, classify_goal, get_goal_routing
- express_path      — ExpressPathMixin for express-tier execution
- two_layer         — Combined TwoLayerOrchestrator
- types             — Re-export shim for shared orchestration data types
- architect_executor — 2-stage architect→executor LLM pipeline
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .agent_graph import (
    AgentGraph,
    get_agent_graph,
)
from .architect_executor import (
    ArchitectExecutorPipeline,
    ArchitectPlan,
    PipelineConfig,
)
from .bottleneck import (
    BottleneckAgentMetrics,
    BottleneckIdentifier,
    get_bottleneck_identifier,
    reset_bottleneck_identifier,
)

# durable_db is safe to import eagerly (no orchestration package imports).
# durable_execution and two_layer are lazy-loaded to break the import cycle:
#   durable_execution imports from vetinari.orchestration.durable_db, which
#   triggers this __init__ before durable_execution finishes loading.
from .checkpoint_store import Checkpoint
from .execution_graph import ExecutionGraph, ExecutionTaskNode
from .express_path import ExpressPathMixin
from .graph_types import (
    ConditionalEdge,
    CycleDetector,
    ExecutionDAG,
    ExecutionStrategy,
    HumanCheckpoint,
    ReplanResult,
    TaskNode,
)
from .graph_types import (
    ExecutionDAG as ExecutionPlan,
)
from .pipeline_confidence import apply_confidence_routing
from .plan_generator import PlanGenerator
from .request_routing import (
    RequestQueue,
    classify_goal,
    get_goal_routing,
)
from .task_context import (
    MAX_OUTPUT_PREVIEW_CHARS,
    TaskContextManifest,
    TaskManifestContext,
)

if TYPE_CHECKING:
    from .durable_execution import DurableExecutionEngine
    from .two_layer import (
        ReworkDecision,
        TwoLayerOrchestrator,
        get_two_layer_orchestrator,
        init_two_layer_orchestrator,
    )

# Lazy-loaded symbols — resolved on first access to avoid circular imports.
# durable_execution.py and two_layer.py both cause a cycle when this __init__
# is triggered mid-load by an import inside those modules.
_LAZY: dict[str, Any] = {}
_LAZY_MAP: dict[str, str] = {
    "DurableExecutionEngine": "durable_execution",
    "ReworkDecision": "two_layer",
    "TwoLayerOrchestrator": "two_layer",
    "get_two_layer_orchestrator": "two_layer",
    "init_two_layer_orchestrator": "two_layer",
}


def __getattr__(name: str) -> Any:
    """Lazy-load durable_execution and two_layer symbols on first access.

    These modules import from vetinari.orchestration, so eagerly importing
    them here would create a circular dependency during package initialisation.

    Args:
        name: Attribute name being looked up.

    Returns:
        The resolved attribute.

    Raises:
        AttributeError: If name is not a known lazy symbol.
    """
    if name in _LAZY:
        return _LAZY[name]
    module_name = _LAZY_MAP.get(name)
    if module_name == "durable_execution":
        from .durable_execution import DurableExecutionEngine

        _LAZY["DurableExecutionEngine"] = DurableExecutionEngine
        return _LAZY[name]
    if module_name == "two_layer":
        from .two_layer import (
            ReworkDecision,
            TwoLayerOrchestrator,
            get_two_layer_orchestrator,
            init_two_layer_orchestrator,
        )

        _LAZY["ReworkDecision"] = ReworkDecision
        _LAZY["TwoLayerOrchestrator"] = TwoLayerOrchestrator
        _LAZY["get_two_layer_orchestrator"] = get_two_layer_orchestrator
        _LAZY["init_two_layer_orchestrator"] = init_two_layer_orchestrator
        return _LAZY[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MAX_OUTPUT_PREVIEW_CHARS",
    "AgentGraph",
    "ArchitectExecutorPipeline",
    "ArchitectPlan",
    "BottleneckAgentMetrics",
    "BottleneckIdentifier",
    "Checkpoint",
    "CheckpointSnapshot",
    "ConditionalEdge",
    "CycleDetector",
    "DurableExecutionEngine",
    "ExecutionDAG",
    "ExecutionEventRecord",
    "ExecutionGraph",
    "ExecutionPlan",
    "ExecutionStrategy",
    "ExecutionTaskNode",
    "ExpressPathMixin",
    "HumanCheckpoint",
    "PipelineConfig",
    "PlanGenerator",
    "ReplanResult",
    "RequestQueue",
    "ReworkDecision",
    "TaskContextManifest",
    "TaskManifestContext",
    "TaskNode",
    "TwoLayerOrchestrator",
    "apply_confidence_routing",
    "classify_goal",
    "get_agent_graph",
    "get_bottleneck_identifier",
    "get_goal_routing",
    "get_two_layer_orchestrator",
    "init_two_layer_orchestrator",
    "reset_bottleneck_identifier",
]
