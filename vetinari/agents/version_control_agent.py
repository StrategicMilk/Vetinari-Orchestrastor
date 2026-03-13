"""Legacy redirect — use ConsolidatedResearcherAgent directly."""

from __future__ import annotations

from vetinari.agents.consolidated.researcher_agent import (
    ConsolidatedResearcherAgent as VersionControlAgent,
)
from vetinari.agents.consolidated.researcher_agent import get_consolidated_researcher_agent as get_version_control_agent

__all__ = ["VersionControlAgent", "get_version_control_agent"]
