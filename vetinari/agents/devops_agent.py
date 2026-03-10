"""Legacy redirect — use ConsolidatedResearcherAgent directly."""
from vetinari.agents.consolidated.researcher_agent import ConsolidatedResearcherAgent as DevOpsAgent, get_consolidated_researcher_agent as get_devops_agent  # noqa: F401

__all__ = ["DevOpsAgent", "get_devops_agent"]
