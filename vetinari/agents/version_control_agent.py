"""Legacy redirect — use ConsolidatedResearcherAgent directly."""
from vetinari.agents.consolidated.researcher_agent import ConsolidatedResearcherAgent as VersionControlAgent, get_consolidated_researcher_agent as get_version_control_agent  # noqa: F401

__all__ = ["VersionControlAgent", "get_version_control_agent"]
