"""Legacy redirect — use ConsolidatedResearcherAgent directly."""

from __future__ import annotations

from vetinari.agents.consolidated.researcher_agent import ConsolidatedResearcherAgent as ArchitectAgent
from vetinari.agents.consolidated.researcher_agent import get_consolidated_researcher_agent as get_architect_agent

__all__ = ["ArchitectAgent", "get_architect_agent"]
