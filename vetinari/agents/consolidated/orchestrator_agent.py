"""Legacy redirect — use PlannerAgent directly."""
from vetinari.agents.planner_agent import PlannerAgent as OrchestratorAgent, get_planner_agent as get_orchestrator_agent  # noqa: F401

__all__ = ["OrchestratorAgent", "get_orchestrator_agent"]
