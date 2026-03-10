"""Legacy redirect — use QualityAgent directly."""
from vetinari.agents.consolidated.quality_agent import QualityAgent as TestAutomationAgent, get_quality_agent as get_test_automation_agent  # noqa: F401

__all__ = ["TestAutomationAgent", "get_test_automation_agent"]
