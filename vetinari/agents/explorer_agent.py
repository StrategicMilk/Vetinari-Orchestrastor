"""Legacy redirect — use ConsolidatedResearcherAgent directly."""

from __future__ import annotations

from vetinari.agents.consolidated.researcher_agent import ConsolidatedResearcherAgent as ExplorerAgent
from vetinari.agents.consolidated.researcher_agent import get_consolidated_researcher_agent as get_explorer_agent

__all__ = ["ExplorerAgent", "get_explorer_agent"]
