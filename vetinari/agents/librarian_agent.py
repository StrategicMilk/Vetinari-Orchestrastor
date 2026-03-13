"""Legacy redirect — use ConsolidatedResearcherAgent directly."""

from __future__ import annotations

from vetinari.agents.consolidated.researcher_agent import ConsolidatedResearcherAgent as LibrarianAgent
from vetinari.agents.consolidated.researcher_agent import get_consolidated_researcher_agent as get_librarian_agent

__all__ = ["LibrarianAgent", "get_librarian_agent"]
