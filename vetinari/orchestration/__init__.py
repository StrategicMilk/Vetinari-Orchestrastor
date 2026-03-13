"""Vetinari Orchestration Module.

This module provides the orchestration engine that coordinates all agents
in the hierarchical multi-agent system.

Sub-modules:
- agent_graph       — AgentGraph DAG orchestration protocol
- execution_graph   — TaskNode & ExecutionGraph for plan execution
- plan_generator    — Goal decomposition into execution graphs
- durable_execution — Checkpoint-based durable execution engine
- two_layer         — Combined TwoLayerOrchestrator
"""

from __future__ import annotations

from .agent_graph import (
    AgentGraph,
    ExecutionPlan,
    ExecutionStrategy,
    TaskNode,
    get_agent_graph,
)
from .durable_execution import (
    Checkpoint,
    DurableExecutionEngine,
    ExecutionEvent,
)
from .execution_graph import ExecutionGraph
from .execution_graph import TaskNode as ExecutionTaskNode  # disambiguate
from .plan_generator import PlanGenerator
from .two_layer import (
    TwoLayerOrchestrator,
    get_two_layer_orchestrator,
    init_two_layer_orchestrator,
)

__all__ = [
    # agent_graph (original)
    "AgentGraph",
    # durable_execution
    "Checkpoint",
    "DurableExecutionEngine",
    "ExecutionEvent",
    # execution_graph
    "ExecutionGraph",
    "ExecutionPlan",
    "ExecutionStrategy",
    "ExecutionTaskNode",
    # plan_generator
    "PlanGenerator",
    "TaskNode",
    # two_layer
    "TwoLayerOrchestrator",
    "get_agent_graph",
    "get_two_layer_orchestrator",
    "init_two_layer_orchestrator",
]
