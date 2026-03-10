"""Legacy redirect — use PlannerAgent directly."""
from vetinari.agents.planner_agent import PlannerAgent as ContextManagerAgent, get_planner_agent as get_context_manager_agent  # noqa: F401

__all__ = ["ContextManagerAgent", "get_context_manager_agent"]
