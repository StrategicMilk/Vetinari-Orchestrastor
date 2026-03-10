"""Legacy redirect — use ConsolidatedResearcherAgent directly."""
from vetinari.agents.consolidated.researcher_agent import ConsolidatedResearcherAgent as UIPlannerAgent, get_consolidated_researcher_agent as get_ui_planner_agent  # noqa: F401

__all__ = ["UIPlannerAgent", "get_ui_planner_agent"]
