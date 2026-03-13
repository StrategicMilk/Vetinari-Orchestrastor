"""Legacy redirect — use ConsolidatedResearcherAgent directly."""

from __future__ import annotations

from vetinari.agents.consolidated.researcher_agent import ConsolidatedResearcherAgent as DevOpsAgent
from vetinari.agents.consolidated.researcher_agent import get_consolidated_researcher_agent as get_devops_agent

__all__ = ["DevOpsAgent", "get_devops_agent"]
