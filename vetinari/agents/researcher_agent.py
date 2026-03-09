"""Legacy redirect — use ConsolidatedResearcherAgent directly."""
from vetinari.agents.consolidated.researcher_agent import ConsolidatedResearcherAgent as ResearcherAgent, get_consolidated_researcher_agent as get_researcher_agent  # noqa: F401

__all__ = ["ResearcherAgent", "get_researcher_agent"]
