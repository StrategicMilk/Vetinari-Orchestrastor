"""Legacy redirect — use PlannerAgent directly."""

from __future__ import annotations

from vetinari.agents.planner_agent import PlannerAgent as ContextManagerAgent
from vetinari.agents.planner_agent import get_planner_agent as get_context_manager_agent

__all__ = ["ContextManagerAgent", "get_context_manager_agent"]
