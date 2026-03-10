"""Legacy redirect — use QualityAgent directly."""
from vetinari.agents.consolidated.quality_agent import QualityAgent as SecurityAuditorAgent, get_quality_agent as get_security_auditor_agent  # noqa: F401

__all__ = ["SecurityAuditorAgent", "get_security_auditor_agent"]
