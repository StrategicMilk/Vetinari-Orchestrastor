"""Legacy redirect — use ConsolidatedResearcherAgent directly."""
from vetinari.agents.consolidated.researcher_agent import ConsolidatedResearcherAgent as ArchitectAgent, get_consolidated_researcher_agent as get_architect_agent  # noqa: F401

__all__ = ["ArchitectAgent", "get_architect_agent"]
