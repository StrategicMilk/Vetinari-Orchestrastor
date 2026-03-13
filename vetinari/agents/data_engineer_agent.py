"""Legacy redirect — use ConsolidatedResearcherAgent directly."""

from __future__ import annotations

from vetinari.agents.consolidated.researcher_agent import ConsolidatedResearcherAgent as DataEngineerAgent
from vetinari.agents.consolidated.researcher_agent import get_consolidated_researcher_agent as get_data_engineer_agent

__all__ = ["DataEngineerAgent", "get_data_engineer_agent"]
