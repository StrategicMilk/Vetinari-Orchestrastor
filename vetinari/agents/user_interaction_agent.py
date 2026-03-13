"""Legacy redirect — use PlannerAgent directly."""

from __future__ import annotations

from vetinari.agents.planner_agent import PlannerAgent as UserInteractionAgent
from vetinari.agents.planner_agent import get_planner_agent as get_user_interaction_agent

__all__ = ["UserInteractionAgent", "get_user_interaction_agent"]
