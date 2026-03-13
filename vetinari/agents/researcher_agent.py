"""Legacy redirect — use ConsolidatedResearcherAgent directly."""

from __future__ import annotations

from vetinari.agents.consolidated.researcher_agent import ConsolidatedResearcherAgent as ResearcherAgent
from vetinari.agents.consolidated.researcher_agent import get_consolidated_researcher_agent as get_researcher_agent

__all__ = ["ResearcherAgent", "get_researcher_agent"]
