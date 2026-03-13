"""Legacy redirect — use ConsolidatedResearcherAgent directly."""

from __future__ import annotations

from vetinari.agents.consolidated.researcher_agent import ConsolidatedResearcherAgent as UIPlannerAgent
from vetinari.agents.consolidated.researcher_agent import get_consolidated_researcher_agent as get_ui_planner_agent

__all__ = ["UIPlannerAgent", "get_ui_planner_agent"]
