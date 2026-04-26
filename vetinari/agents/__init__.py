"""Vetinari Agents Module.

3-agent factory pipeline: Foreman, Worker, Inspector (ADR-0061).
"""

from __future__ import annotations

from vetinari.types import AgentType, ExecutionMode, StatusEnum

from .base_agent import BaseAgent
from .consolidated.quality_agent import InspectorAgent, get_inspector_agent
from .consolidated.worker_agent import WorkerAgent, get_worker_agent
from .contracts import (  # noqa: VET123 - barrel export preserves public import compatibility
    AgentResult,
    AgentSpec,
    AgentTask,
    ExecutionPlan,
    Task,
    VerificationResult,
    get_agent_spec,
)
from .multi_mode_agent import MultiModeAgent
from .planner_agent import ForemanAgent, get_foreman_agent

__all__ = [
    "AgentResult",
    "AgentSpec",
    "AgentTask",
    "AgentType",
    "BaseAgent",
    "ExecutionMode",
    "ExecutionPlan",
    "ForemanAgent",
    "InspectorAgent",
    "MultiModeAgent",
    "StatusEnum",
    "Task",
    "VerificationResult",
    "WorkerAgent",
    "get_agent_spec",
    "get_foreman_agent",
    "get_inspector_agent",
    "get_worker_agent",
]
