"""Legacy redirect — use ConsolidatedResearcherAgent directly."""
from vetinari.agents.consolidated.researcher_agent import ConsolidatedResearcherAgent as LibrarianAgent, get_consolidated_researcher_agent as get_librarian_agent  # noqa: F401

__all__ = ["LibrarianAgent", "get_librarian_agent"]
