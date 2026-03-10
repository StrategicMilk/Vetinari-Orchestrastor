"""Legacy redirect — use ConsolidatedResearcherAgent directly."""
from vetinari.agents.consolidated.researcher_agent import ConsolidatedResearcherAgent as ExplorerAgent, get_consolidated_researcher_agent as get_explorer_agent  # noqa: F401

__all__ = ["ExplorerAgent", "get_explorer_agent"]
