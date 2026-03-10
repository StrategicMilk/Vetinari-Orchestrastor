"""Legacy redirect — use ConsolidatedResearcherAgent directly."""
from vetinari.agents.consolidated.researcher_agent import ConsolidatedResearcherAgent as DataEngineerAgent, get_consolidated_researcher_agent as get_data_engineer_agent  # noqa: F401

__all__ = ["DataEngineerAgent", "get_data_engineer_agent"]
