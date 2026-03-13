"""Legacy redirect — use PlannerAgent directly."""

from __future__ import annotations

from vetinari.agents.planner_agent import PlannerAgent as OrchestratorAgent
from vetinari.agents.planner_agent import get_planner_agent as get_orchestrator_agent

__all__ = ["OrchestratorAgent", "get_orchestrator_agent"]
