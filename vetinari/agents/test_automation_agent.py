"""Legacy redirect — use QualityAgent directly."""

from __future__ import annotations

from vetinari.agents.consolidated.quality_agent import QualityAgent as TestAutomationAgent
from vetinari.agents.consolidated.quality_agent import get_quality_agent as get_test_automation_agent

__all__ = ["TestAutomationAgent", "get_test_automation_agent"]
